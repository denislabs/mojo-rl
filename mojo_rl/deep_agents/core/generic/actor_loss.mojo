"""Actor loss computation strategies for off-policy agents.

Each strategy handles the full actor update cycle:
  1. Forward actor (with cache for backward)
  2. Produce actions for critic (deterministic or reparameterized)
  3. Forward critic to get Q value
  4. Backward critic to get dQ/da (action gradient)
  5. Convert dQ/da into actor gradient (direct for DPG, through rsample for SAC)
  6. Backward actor to accumulate parameter gradients

The agent handles optimizer_step() and alpha update after calling the strategy.

All LayoutTensor dimensions are derived from ActorModel/CriticModel
(IN_DIM, OUT_DIM, PARAM_SIZE, CACHE_SIZE) to match Network expectations.

Implementations:
  - DPGLoss: Deterministic policy gradient -dQ/da (DDPG, TD3)
  - MaxEntLoss: Max-entropy loss with reparameterized sampling (SAC)
"""

from layout import Layout, LayoutTensor
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network
from mojo_rl.nn.gpu.random import gaussian_noise
from mojo_rl.nn.model.stochastic_actor import (
    rsample_with_cache,
    rsample_backward,
)
from mojo_rl.deep_agents.core.kernels import (
    concat_obs_action_kernel,
    actor_grad_from_critic_kernel,
)
from mojo_rl.deep_agents.sac.kernels import (
    sac_rsample_with_cache_kernel,
    sac_rsample_bwd_kernel,
    min_q_dq_kernel,
    add_ci_grads_kernel,
)


trait ActorLoss:
    """Trait for actor loss strategies."""

    comptime HAS_ALPHA: Bool

    @staticmethod
    fn gpu_lp_offset[
        BATCH: Int,
        ACTIONS: Int,
        ACTOR_OUT: Int,
        ACTOR_CS: Int,
    ]() -> Int:
        """Offset of log_probs in GPU strat_ws (for alpha auto-tuning).

        Returns 0 for strategies without alpha (DPGLoss).
        The agent reads BATCH floats from strat_ws at this offset
        after synchronizing, to compute mean_lp for alpha update.
        """
        ...

    @staticmethod
    fn ws_size[
        BATCH: Int,
        OBS: Int,
        ACTIONS: Int,
        ACTOR_OUT: Int,
        ACTOR_CS: Int,
        CRITIC_IN: Int,
        CRITIC_OUT: Int,
        CRITIC_CS: Int,
    ]() -> Int:
        ...

    @staticmethod
    fn update_actor_cpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut actor_grads: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic2_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic2_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        ws: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        alpha: Float64,
    ) -> Float64:
        ...

    @staticmethod
    fn update_actor_gpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        ctx: DeviceContext,
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut actor_grads: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic2_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic2_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        actor_ws: DeviceBuffer[dtype],
        critic_ws: DeviceBuffer[dtype],
        critic2_ws: DeviceBuffer[dtype],
        strat_ws: DeviceBuffer[dtype],
        dq_buf: DeviceBuffer[dtype],
        alpha: Float64,
        rng_seed: UInt32,
    ) raises -> Float64:
        ...


# =============================================================================
# Shared helpers
# =============================================================================


fn _concat_obs_act_inline[
    BATCH: Int, OBS: Int, ACTIONS: Int, CRITIC_IN: Int
](
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    obs_p: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    act_p: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    """Concat [obs, act] into dst with CRITIC_IN stride."""
    for row in range(BATCH):
        for c in range(OBS):
            dst[row * CRITIC_IN + c] = obs_p[row * OBS + c]
        for c in range(ACTIONS):
            dst[row * CRITIC_IN + OBS + c] = act_p[row * ACTIONS + c]


fn _extract_action_grad[
    BATCH: Int, OBS: Int, ACTIONS: Int, CRITIC_IN: Int
](
    d_actions: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_critic_input: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    """Extract dQ/da from d_critic_input (skip obs portion)."""
    for b in range(BATCH):
        for a in range(ACTIONS):
            d_actions[b * ACTIONS + a] = d_critic_input[b * CRITIC_IN + OBS + a]


# =============================================================================
# DPGLoss — deterministic policy gradient (DDPG, TD3)
# =============================================================================


struct DPGLoss(ActorLoss):
    """Deterministic policy gradient: maximize Q by backpropagating
    through the critic to get dQ/da, then through the actor.

    Actor loss = -E[Q(s, actor(s))]

    For DDPG/TD3: ActorModel.OUT_DIM == ACTIONS (deterministic actor).
    """

    comptime HAS_ALPHA: Bool = False

    @staticmethod
    fn gpu_lp_offset[
        BATCH: Int,
        ACTIONS: Int,
        ACTOR_OUT: Int,
        ACTOR_CS: Int,
    ]() -> Int:
        """No log_probs for DPG."""
        return 0

    @staticmethod
    fn ws_size[
        BATCH: Int,
        OBS: Int,
        ACTIONS: Int,
        ACTOR_OUT: Int,
        ACTOR_CS: Int,
        CRITIC_IN: Int,
        CRITIC_OUT: Int,
        CRITIC_CS: Int,
    ]() -> Int:
        return (
            BATCH * ACTIONS
            + BATCH * ACTOR_CS
            + BATCH * CRITIC_IN
            + BATCH * CRITIC_OUT
            + BATCH * CRITIC_CS
            + BATCH * CRITIC_OUT
            + BATCH * CRITIC_IN
            + BATCH * ACTIONS
            + BATCH * OBS
        )

    @staticmethod
    fn update_actor_cpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut actor_grads: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic2_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic2_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        ws: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        alpha: Float64,
    ) -> Float64:
        """Compute deterministic policy gradient and fill actor_grads.

        Returns 0.0 (no log_probs for DPG).
        DPG uses only critic1 for the actor gradient (standard for DDPG/TD3).
        """
        # Dimensions derived from model types
        comptime OBS = ActorModel.IN_DIM
        comptime ACTOR_CS = ActorModel.CACHE_SIZE
        comptime CRITIC_IN = CriticModel.IN_DIM
        comptime CRITIC_OUT = CriticModel.OUT_DIM
        comptime CRITIC_CS = CriticModel.CACHE_SIZE

        # Workspace offsets
        comptime W_ACT = 0
        comptime W_ACACHE = W_ACT + BATCH * ACTIONS
        comptime W_CI = W_ACACHE + BATCH * ACTOR_CS
        comptime W_Q = W_CI + BATCH * CRITIC_IN
        comptime W_CCACHE = W_Q + BATCH * CRITIC_OUT
        comptime W_DQ = W_CCACHE + BATCH * CRITIC_CS
        comptime W_DCI = W_DQ + BATCH * CRITIC_OUT
        comptime W_DACT = W_DCI + BATCH * CRITIC_IN
        comptime W_DOBS = W_DACT + BATCH * ACTIONS

        # 1. Forward actor with cache
        var act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws + W_ACT)
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](ws + W_ACACHE)
        Network[ActorModel, ActorOpt].forward_with_cache[BATCH](
            obs, act_t, actor_params, actor_cache_t
        )

        # 2. Concat obs + actions -> critic_input
        _concat_obs_act_inline[BATCH, OBS, ACTIONS, CRITIC_IN](
            ws + W_CI, obs.ptr, ws + W_ACT
        )
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.IN_DIM), MutAnyOrigin
        ](ws + W_CI)

        # 3. Forward critic with cache
        var q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.OUT_DIM), MutAnyOrigin
        ](ws + W_Q)
        var critic_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](ws + W_CCACHE)
        Network[CriticModel, CriticOpt].forward_with_cache[BATCH](
            ci_t, q_t, critic_params, critic_cache_t
        )

        # 4. Gradient seed: dQ = -1/batch (maximize Q)
        var dq_ptr = ws + W_DQ
        for b in range(BATCH):
            dq_ptr[b] = Scalar[dtype](-1.0 / Float64(BATCH))
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.OUT_DIM), MutAnyOrigin
        ](dq_ptr)

        # 5. Backward critic -> d_critic_input
        var dci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.IN_DIM), MutAnyOrigin
        ](ws + W_DCI)
        # Zero critic grads (we discard them -- just need d_critic_input)
        for i in range(CriticModel.PARAM_SIZE):
            critic_grads.ptr[i] = Scalar[dtype](0)
        Network[CriticModel, CriticOpt].backward[BATCH](
            dq_t, dci_t, critic_params, critic_cache_t, critic_grads
        )

        # 6. Extract d_actions from d_critic_input
        _extract_action_grad[BATCH, OBS, ACTIONS, CRITIC_IN](
            ws + W_DACT, ws + W_DCI
        )
        var da_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws + W_DACT)

        # 7. Backward actor
        var dobs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ](ws + W_DOBS)
        # Zero actor grads before accumulation
        for i in range(ActorModel.PARAM_SIZE):
            actor_grads.ptr[i] = Scalar[dtype](0)
        Network[ActorModel, ActorOpt].backward[BATCH](
            da_t, dobs_t, actor_params, actor_cache_t, actor_grads
        )

        return 0.0  # No log_probs for DPG

    @staticmethod
    fn update_actor_gpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        ctx: DeviceContext,
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut actor_grads: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic2_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic2_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        actor_ws: DeviceBuffer[dtype],
        critic_ws: DeviceBuffer[dtype],
        critic2_ws: DeviceBuffer[dtype],
        strat_ws: DeviceBuffer[dtype],
        dq_buf: DeviceBuffer[dtype],
        alpha: Float64,
        rng_seed: UInt32,
    ) raises -> Float64:
        """GPU deterministic policy gradient: actor -> critic -> dQ/da -> actor bwd.
        DPG uses only critic1 for the actor gradient (standard for DDPG/TD3).

        strat_ws layout:
          [0]                              actor_act      [BATCH * ACTIONS]
          [BATCH * ACTIONS]                actor_cache    [BATCH * ACTOR_CS]
          [+ BATCH * ACTOR_CS]             new_ci         [BATCH * CRITIC_IN]
          [+ BATCH * CRITIC_IN]            new_q          [BATCH * CRITIC_OUT]
          [+ BATCH * CRITIC_OUT]           new_q_cache    [BATCH * CRITIC_CS]
          [+ BATCH * CRITIC_CS]            dq             [BATCH * CRITIC_OUT]
          [+ BATCH * CRITIC_OUT]           d_ci           [BATCH * CRITIC_IN]
          [+ BATCH * CRITIC_IN]            d_act          [BATCH * ACTIONS]
          [+ BATCH * ACTIONS]              d_obs          [BATCH * OBS]
        """
        comptime OBS = ActorModel.IN_DIM
        comptime ACTOR_CS = ActorModel.CACHE_SIZE
        comptime CRITIC_IN = CriticModel.IN_DIM
        comptime CRITIC_OUT = CriticModel.OUT_DIM
        comptime CRITIC_CS = CriticModel.CACHE_SIZE
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime ELEM_BLOCKS = (BATCH * CRITIC_IN + TPB - 1) // TPB
        comptime ACT_BLOCKS = (BATCH * ACTIONS + TPB - 1) // TPB

        # Workspace offsets
        comptime W_ACT = 0
        comptime W_ACACHE = W_ACT + BATCH * ACTIONS
        comptime W_CI = W_ACACHE + BATCH * ACTOR_CS
        comptime W_Q = W_CI + BATCH * CRITIC_IN
        comptime W_CCACHE = W_Q + BATCH * CRITIC_OUT
        comptime W_DQ = W_CCACHE + BATCH * CRITIC_CS
        comptime W_DCI = W_DQ + BATCH * CRITIC_OUT
        comptime W_DACT = W_DCI + BATCH * CRITIC_IN
        comptime W_DOBS = W_DACT + BATCH * ACTIONS

        var ws_ptr = strat_ws.unsafe_ptr()

        # 1. Actor forward with cache
        var actor_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws_ptr + W_ACT)
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](ws_ptr + W_ACACHE)
        Network[ActorModel, ActorOpt].forward_gpu_with_cache[BATCH](
            ctx, obs, actor_act_t, actor_params, actor_cache_t, actor_ws
        )

        # 2. Concat(obs, actor_actions) -> new_ci
        var new_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.IN_DIM), MutAnyOrigin
        ](ws_ptr + W_CI)
        var act_for_concat_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws_ptr + W_ACT)

        @always_inline
        fn concat_new_ci(
            d: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
            o: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
        ):
            concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

        ctx.enqueue_function[concat_new_ci, concat_new_ci](
            new_ci_t,
            obs,
            act_for_concat_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        # 3. Critic forward with cache
        var new_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.OUT_DIM), MutAnyOrigin
        ](ws_ptr + W_Q)
        var critic_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](ws_ptr + W_CCACHE)
        Network[CriticModel, CriticOpt].forward_gpu_with_cache[BATCH](
            ctx, new_ci_t, new_q_t, critic_params, critic_cache_t, critic_ws
        )

        # 4. dq seed = -1/batch from pre-filled GPU buffer (maximize Q)
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.OUT_DIM), MutAnyOrigin
        ](dq_buf.unsafe_ptr())
        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.IN_DIM), MutAnyOrigin
        ](ws_ptr + W_DCI)

        # Critic backward (grads pre-zeroed by agent, discarded — we need d_ci)
        Network[CriticModel, CriticOpt].backward_gpu[BATCH](
            ctx,
            dq_t,
            d_ci_t,
            critic_params,
            critic_cache_t,
            critic_grads,
            critic_ws,
        )

        # 5. Extract action gradients from d_ci
        var d_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws_ptr + W_DACT)

        @always_inline
        fn extract_act_grad(
            da: LayoutTensor[
                dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
            ],
            dnc: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
        ):
            actor_grad_from_critic_kernel[
                dtype, BATCH, OBS, ActorModel.OUT_DIM
            ](da, dnc)

        ctx.enqueue_function[extract_act_grad, extract_act_grad](
            d_act_t,
            d_ci_t,
            grid_dim=(ACT_BLOCKS,),
            block_dim=(TPB,),
        )

        # 6. Actor backward (grads pre-zeroed by agent)
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ](ws_ptr + W_DOBS)
        Network[ActorModel, ActorOpt].backward_gpu[BATCH](
            ctx,
            d_act_t,
            d_obs_t,
            actor_params,
            actor_cache_t,
            actor_grads,
            actor_ws,
        )

        return 0.0  # No log_probs for DPG


# =============================================================================
# MaxEntLoss — max-entropy with reparameterized sampling (SAC)
# =============================================================================


struct MaxEntLoss[
    log_std_scale: Float64 = 3.5,
](ActorLoss):
    """Max-entropy actor loss with reparameterized sampling.

    Actor loss = E[alpha * log_pi(a|s) - Q(s, a)]

    The gradient flows: dQ/da -> rsample_backward -> grad_mean + grad_log_std
    -> concat into actor_grad -> actor backward.

    ACTIONS must be passed separately because ActorModel.OUT_DIM = 2*ACTIONS
    for SAC's Parallel[mean, log_std] architecture.

    log_std_scale: scaling factor for grad_log_std when building actor_grad.
    Applied as: grad_log_std * (0.5 * 2 * log_std_scale). Default 3.5 gives
    the factor 0.5 * 7.0 = 3.5 used in the reference SAC implementation.
    """

    comptime HAS_ALPHA: Bool = True

    @staticmethod
    fn gpu_lp_offset[
        BATCH: Int,
        ACTIONS: Int,
        ACTOR_OUT: Int,
        ACTOR_CS: Int,
    ]() -> Int:
        """Offset of log_probs [BATCH] in GPU strat_ws."""
        return BATCH * ACTOR_OUT + BATCH * ACTOR_CS + BATCH * ACTIONS

    @staticmethod
    fn ws_size[
        BATCH: Int,
        OBS: Int,
        ACTIONS: Int,
        ACTOR_OUT: Int,
        ACTOR_CS: Int,
        CRITIC_IN: Int,
        CRITIC_OUT: Int,
        CRITIC_CS: Int,
    ]() -> Int:
        return (
            BATCH * ACTOR_OUT       # raw_out
            + BATCH * ACTOR_CS      # actor_cache
            + BATCH * ACTIONS       # mean
            + BATCH * ACTIONS       # log_std
            + BATCH * ACTIONS       # noise
            + BATCH * ACTIONS       # act
            + BATCH                 # log_probs
            + BATCH * ACTIONS       # z_cache
            + BATCH * CRITIC_IN     # critic_input
            + BATCH * CRITIC_OUT    # Q1
            + BATCH * CRITIC_CS     # Q1 cache
            + BATCH * CRITIC_OUT    # Q2
            + BATCH * CRITIC_CS     # Q2 cache
            + BATCH * CRITIC_OUT    # dq1
            + BATCH * CRITIC_OUT    # dq2
            + BATCH * CRITIC_IN     # d_ci1
            + BATCH * CRITIC_IN     # d_ci2
            + BATCH * ACTIONS       # d_act
            + BATCH * ACTIONS       # grad_mean
            + BATCH * ACTIONS       # grad_log_std
            + BATCH * ACTOR_OUT     # actor_grad
            + BATCH * OBS           # d_obs
        )

    @staticmethod
    fn update_actor_cpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut actor_grads: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic2_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic2_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        ws: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        alpha: Float64,
    ) -> Float64:
        """Compute max-entropy actor gradient using min(Q1, Q2).

        Returns mean log_prob for alpha auto-tuning by the agent.
        """
        # Dimensions derived from model types
        comptime OBS = ActorModel.IN_DIM
        comptime ACTOR_OUT = ActorModel.OUT_DIM
        comptime ACTOR_CS = ActorModel.CACHE_SIZE
        comptime CRITIC_IN = CriticModel.IN_DIM
        comptime CRITIC_OUT = CriticModel.OUT_DIM
        comptime CRITIC_CS = CriticModel.CACHE_SIZE

        # Workspace offsets
        comptime W_RAW = 0
        comptime W_ACACHE = W_RAW + BATCH * ACTOR_OUT
        comptime W_MEAN = W_ACACHE + BATCH * ACTOR_CS
        comptime W_LSTD = W_MEAN + BATCH * ACTIONS
        comptime W_NOISE = W_LSTD + BATCH * ACTIONS
        comptime W_ACT = W_NOISE + BATCH * ACTIONS
        comptime W_LP = W_ACT + BATCH * ACTIONS
        comptime W_ZCACHE = W_LP + BATCH
        comptime W_CI = W_ZCACHE + BATCH * ACTIONS
        comptime W_Q = W_CI + BATCH * CRITIC_IN
        comptime W_CCACHE = W_Q + BATCH * CRITIC_OUT
        comptime W_Q2 = W_CCACHE + BATCH * CRITIC_CS
        comptime W_C2CACHE = W_Q2 + BATCH * CRITIC_OUT
        comptime W_DQ = W_C2CACHE + BATCH * CRITIC_CS
        comptime W_DQ2 = W_DQ + BATCH * CRITIC_OUT
        comptime W_DCI = W_DQ2 + BATCH * CRITIC_OUT
        comptime W_DCI2 = W_DCI + BATCH * CRITIC_IN
        comptime W_DACT = W_DCI2 + BATCH * CRITIC_IN
        comptime W_GMEAN = W_DACT + BATCH * ACTIONS
        comptime W_GLSTD = W_GMEAN + BATCH * ACTIONS
        comptime W_AGRAD = W_GLSTD + BATCH * ACTIONS
        comptime W_DOBS = W_AGRAD + BATCH * ACTOR_OUT

        # 1. Forward actor with cache -> raw output [mean || log_std]
        var raw_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws + W_RAW)
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](ws + W_ACACHE)
        Network[ActorModel, ActorOpt].forward_with_cache[BATCH](
            obs, raw_out_t, actor_params, actor_cache_t
        )

        # 2. Extract mean + log_std from Parallel output
        for b in range(BATCH):
            for a in range(ACTIONS):
                (ws + W_MEAN)[b * ACTIONS + a] = (ws + W_RAW)[b * ACTOR_OUT + a]
                (ws + W_LSTD)[b * ACTIONS + a] = (ws + W_RAW)[
                    b * ACTOR_OUT + ACTIONS + a
                ]

        # 3. Generate noise for reparameterization
        for i in range(BATCH * ACTIONS):
            (ws + W_NOISE)[i] = Scalar[dtype](gaussian_noise())

        # 4. Reparameterized sample with cache (for backward pass)
        var act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws + W_ACT)
        var lp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](ws + W_LP)
        var z_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws + W_ZCACHE)
        rsample_with_cache[BATCH, ACTIONS](
            LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin](
                ws + W_MEAN
            ),
            LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin](
                ws + W_LSTD
            ),
            LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin](
                ws + W_NOISE
            ),
            act_t,
            lp_t,
            z_cache_t,
        )

        # Guard NaN in log_probs
        for b in range(BATCH):
            var lp = Float64((ws + W_LP)[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                (ws + W_LP)[b] = Scalar[dtype](-1.0)

        # 5. Concat obs + sampled_actions -> critic_input
        _concat_obs_act_inline[BATCH, OBS, ACTIONS, CRITIC_IN](
            ws + W_CI, obs.ptr, ws + W_ACT
        )
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.IN_DIM), MutAnyOrigin
        ](ws + W_CI)

        # 6. Forward both critics with cache -> Q1, Q2
        var q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_OUT), MutAnyOrigin
        ](ws + W_Q)
        var critic_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin,
        ](ws + W_CCACHE)
        Network[CriticModel, CriticOpt].forward_with_cache[BATCH](
            ci_t, q_t, critic_params, critic_cache_t
        )

        var q2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_OUT), MutAnyOrigin
        ](ws + W_Q2)
        var critic2_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin,
        ](ws + W_C2CACHE)
        Network[CriticModel, CriticOpt].forward_with_cache[BATCH](
            ci_t, q2_t, critic2_params, critic2_cache_t
        )

        # 7. min(Q1, Q2) masked gradient seeds
        var neg_inv_batch = Scalar[dtype](-1.0 / Float64(BATCH))
        var zero = Scalar[dtype](0.0)
        for b in range(BATCH):
            if (ws + W_Q)[b] <= (ws + W_Q2)[b]:
                (ws + W_DQ)[b] = neg_inv_batch
                (ws + W_DQ2)[b] = zero
            else:
                (ws + W_DQ)[b] = zero
                (ws + W_DQ2)[b] = neg_inv_batch
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_OUT), MutAnyOrigin
        ](ws + W_DQ)
        var dq2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_OUT), MutAnyOrigin
        ](ws + W_DQ2)

        # 8. Backward both critics -> d_critic_input
        var dci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](ws + W_DCI)
        for i in range(CriticModel.PARAM_SIZE):
            critic_grads.ptr[i] = Scalar[dtype](0)
        Network[CriticModel, CriticOpt].backward[BATCH](
            dq_t, dci_t, critic_params, critic_cache_t, critic_grads
        )

        var dci2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](ws + W_DCI2)
        for i in range(CriticModel.PARAM_SIZE):
            critic2_grads.ptr[i] = Scalar[dtype](0)
        Network[CriticModel, CriticOpt].backward[BATCH](
            dq2_t, dci2_t, critic2_params, critic2_cache_t, critic2_grads
        )

        # 8b. Combine d_ci from both critics
        for i in range(BATCH * CRITIC_IN):
            (ws + W_DCI)[i] = (ws + W_DCI)[i] + (ws + W_DCI2)[i]

        # 9. Extract d_actions from combined d_critic_input
        _extract_action_grad[BATCH, OBS, ACTIONS, CRITIC_IN](
            ws + W_DACT, ws + W_DCI
        )

        # 10. Entropy gradient: grad_log_prob = alpha / batch
        var grad_lp_arr = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
        for b in range(BATCH):
            grad_lp_arr[b] = Scalar[dtype](alpha / Float64(BATCH))

        # 11. rsample_backward: (d_actions, grad_log_prob) -> grad_mean, grad_log_std
        var ga_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws + W_DACT)
        var glp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](grad_lp_arr.unsafe_ptr())
        var gmean_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws + W_GMEAN)
        var glstd_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws + W_GLSTD)
        rsample_backward[BATCH, ACTIONS](
            ga_t,
            glp_t,
            act_t,
            LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin](
                ws + W_LSTD
            ),
            LayoutTensor[dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin](
                ws + W_NOISE
            ),
            gmean_t,
            glstd_t,
        )

        # 12. Build actor_grad = concat(grad_mean, scaled_grad_log_std)
        comptime AFFINE_SCALE = Scalar[dtype](0.5 * 2.0 * Self.log_std_scale)
        for b in range(BATCH):
            for a in range(ACTIONS):
                (ws + W_AGRAD)[b * ACTOR_OUT + a] = (ws + W_GMEAN)[
                    b * ACTIONS + a
                ]
                (ws + W_AGRAD)[b * ACTOR_OUT + ACTIONS + a] = (ws + W_GLSTD)[
                    b * ACTIONS + a
                ] * AFFINE_SCALE

        # 13. Backward actor
        var actor_grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws + W_AGRAD)
        var dobs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ](ws + W_DOBS)
        for i in range(ActorModel.PARAM_SIZE):
            actor_grads.ptr[i] = Scalar[dtype](0)
        Network[ActorModel, ActorOpt].backward[BATCH](
            actor_grad_t, dobs_t, actor_params, actor_cache_t, actor_grads
        )

        # Return mean log_prob for alpha update by agent
        var mean_lp: Float64 = 0.0
        for b in range(BATCH):
            mean_lp += Float64((ws + W_LP)[b])
        mean_lp /= Float64(BATCH)
        return mean_lp

    @staticmethod
    fn update_actor_gpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        ctx: DeviceContext,
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut actor_grads: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic2_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic2_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        actor_ws: DeviceBuffer[dtype],
        critic_ws: DeviceBuffer[dtype],
        critic2_ws: DeviceBuffer[dtype],
        strat_ws: DeviceBuffer[dtype],
        dq_buf: DeviceBuffer[dtype],
        alpha: Float64,
        rng_seed: UInt32,
    ) raises -> Float64:
        """GPU max-entropy actor gradient via reparameterized sampling.

        Uses min(Q1, Q2) for the actor gradient (matching old SAC).
        Returns 0.0 — mean log_prob must be computed by the agent via
        a separate GPU->CPU sync on the log_probs buffer in strat_ws.

        strat_ws layout:
          [0]                              actor_out      [BATCH * ACTOR_OUT]
          [+ BATCH * ACTOR_OUT]            actor_cache    [BATCH * ACTOR_CS]
          [+ BATCH * ACTOR_CS]             curr_act       [BATCH * ACTIONS]
          [+ BATCH * ACTIONS]              curr_lp        [BATCH]
          [+ BATCH]                        eps_cache      [BATCH * ACTIONS]
          [+ BATCH * ACTIONS]              new_ci         [BATCH * CRITIC_IN]
          [+ BATCH * CRITIC_IN]            new_q          [BATCH * CRITIC_OUT]
          [+ BATCH * CRITIC_OUT]           new_q_cache    [BATCH * CRITIC_CS]
          [+ BATCH * CRITIC_CS]            new_q2         [BATCH * CRITIC_OUT]
          [+ BATCH * CRITIC_OUT]           new_q2_cache   [BATCH * CRITIC_CS]
          [+ BATCH * CRITIC_CS]            dq1            [BATCH * CRITIC_OUT]
          [+ BATCH * CRITIC_OUT]           dq2            [BATCH * CRITIC_OUT]
          [+ BATCH * CRITIC_OUT]           d_ci           [BATCH * CRITIC_IN]
          [+ BATCH * CRITIC_IN]            d_ci2          [BATCH * CRITIC_IN]
          [+ BATCH * CRITIC_IN]            grad_act       [BATCH * ACTIONS]
          [+ BATCH * ACTIONS]              actor_grad     [BATCH * ACTOR_OUT]
          [+ BATCH * ACTOR_OUT]            d_obs          [BATCH * OBS]
        """
        comptime OBS = ActorModel.IN_DIM
        comptime ACTOR_OUT = ActorModel.OUT_DIM
        comptime ACTOR_CS = ActorModel.CACHE_SIZE
        comptime CRITIC_IN = CriticModel.IN_DIM
        comptime CRITIC_OUT = CriticModel.OUT_DIM
        comptime CRITIC_CS = CriticModel.CACHE_SIZE
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime ELEM_BLOCKS = (BATCH * CRITIC_IN + TPB - 1) // TPB
        comptime ACT_BLOCKS = (BATCH * ACTIONS + TPB - 1) // TPB

        # Workspace offsets
        comptime W_RAW = 0
        comptime W_ACACHE = W_RAW + BATCH * ACTOR_OUT
        comptime W_ACT = W_ACACHE + BATCH * ACTOR_CS
        comptime W_LP = W_ACT + BATCH * ACTIONS
        comptime W_EPS = W_LP + BATCH
        comptime W_CI = W_EPS + BATCH * ACTIONS
        comptime W_Q = W_CI + BATCH * CRITIC_IN
        comptime W_CCACHE = W_Q + BATCH * CRITIC_OUT
        comptime W_Q2 = W_CCACHE + BATCH * CRITIC_CS
        comptime W_C2CACHE = W_Q2 + BATCH * CRITIC_OUT
        comptime W_DQ = W_C2CACHE + BATCH * CRITIC_CS
        comptime W_DQ2 = W_DQ + BATCH * CRITIC_OUT
        comptime W_DCI = W_DQ2 + BATCH * CRITIC_OUT
        comptime W_DCI2 = W_DCI + BATCH * CRITIC_IN
        comptime W_DACT = W_DCI2 + BATCH * CRITIC_IN
        comptime W_AGRAD = W_DACT + BATCH * ACTIONS
        comptime W_DOBS = W_AGRAD + BATCH * ACTOR_OUT

        var ws_ptr = strat_ws.unsafe_ptr()

        # 1. Actor forward with cache -> actor_out [BATCH, ACTOR_OUT]
        var actor_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws_ptr + W_RAW)
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](ws_ptr + W_ACACHE)
        Network[ActorModel, ActorOpt].forward_gpu_with_cache[BATCH](
            ctx, obs, actor_out_t, actor_params, actor_cache_t, actor_ws
        )

        # 2. sac_rsample with cache -> curr_act, curr_lp, eps_cache
        var curr_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws_ptr + W_ACT)
        var curr_lp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](ws_ptr + W_LP)
        var eps_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws_ptr + W_EPS)

        var log_std_min_s = Scalar[dtype](-5.0)
        var log_std_max_s = Scalar[dtype](2.0)
        var rng_seed_s = Scalar[DType.uint32](rng_seed)

        @always_inline
        fn curr_rsample(
            acts: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            eps: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, ACTIONS + ACTIONS),
                MutAnyOrigin,
            ],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            seed: Scalar[DType.uint32],
        ):
            sac_rsample_with_cache_kernel[dtype, BATCH, ACTIONS](
                acts, lp, eps, ao, lsmin, lsmax, seed
            )

        ctx.enqueue_function[curr_rsample, curr_rsample](
            curr_act_t,
            curr_lp_t,
            eps_cache_t,
            actor_out_t,
            log_std_min_s,
            log_std_max_s,
            rng_seed_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # 3. Concat(obs, curr_act) -> new_ci
        var new_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CriticModel.IN_DIM), MutAnyOrigin
        ](ws_ptr + W_CI)

        @always_inline
        fn concat_new_ci(
            d: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
            o: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
        ):
            concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

        ctx.enqueue_function[concat_new_ci, concat_new_ci](
            new_ci_t,
            obs,
            curr_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        # 4. Forward both critics with cache -> Q1, Q2
        var new_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_OUT), MutAnyOrigin
        ](ws_ptr + W_Q)
        var critic_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin,
        ](ws_ptr + W_CCACHE)
        Network[CriticModel, CriticOpt].forward_gpu_with_cache[BATCH](
            ctx, new_ci_t, new_q_t, critic_params, critic_cache_t, critic_ws
        )

        var new_q2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_OUT), MutAnyOrigin
        ](ws_ptr + W_Q2)
        var critic2_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin,
        ](ws_ptr + W_C2CACHE)
        Network[CriticModel, CriticOpt].forward_gpu_with_cache[BATCH](
            ctx, new_ci_t, new_q2_t, critic2_params, critic2_cache_t, critic2_ws
        )

        # 5. min(Q1, Q2) masked gradient seeds
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_OUT), MutAnyOrigin
        ](ws_ptr + W_DQ)
        var dq2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_OUT), MutAnyOrigin
        ](ws_ptr + W_DQ2)

        @always_inline
        fn min_q_mask(
            dq1: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            dq2: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            q1: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            q2: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
        ):
            min_q_dq_kernel[dtype, BATCH](dq1, dq2, q1, q2)

        ctx.enqueue_function[min_q_mask, min_q_mask](
            dq_t,
            dq2_t,
            new_q_t,
            new_q2_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # 6. Backward both critics -> d_ci, d_ci2
        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](ws_ptr + W_DCI)
        Network[CriticModel, CriticOpt].backward_gpu[BATCH](
            ctx, dq_t, d_ci_t, critic_params, critic_cache_t,
            critic_grads, critic_ws,
        )

        var d_ci2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](ws_ptr + W_DCI2)
        Network[CriticModel, CriticOpt].backward_gpu[BATCH](
            ctx, dq2_t, d_ci2_t, critic2_params, critic2_cache_t,
            critic2_grads, critic2_ws,
        )

        # 6b. Combine d_ci from both critics: d_ci += d_ci2
        @always_inline
        fn add_grads(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
        ):
            add_ci_grads_kernel[dtype, BATCH, CRITIC_IN](dst, src)

        ctx.enqueue_function[add_grads, add_grads](
            d_ci_t,
            d_ci2_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        # 7. Extract action gradients from combined d_ci
        var grad_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](ws_ptr + W_DACT)

        @always_inline
        fn extract_act_grad(
            da: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            dnc: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
        ):
            actor_grad_from_critic_kernel[dtype, BATCH, OBS, ACTIONS](da, dnc)

        ctx.enqueue_function[extract_act_grad, extract_act_grad](
            grad_act_t,
            d_ci_t,
            grid_dim=(ACT_BLOCKS,),
            block_dim=(TPB,),
        )

        # 8. Backward through reparameterization -> actor_grad [BATCH, ACTOR_OUT]
        var actor_grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.OUT_DIM), MutAnyOrigin
        ](ws_ptr + W_AGRAD)
        var alpha_per_sample = Scalar[dtype](alpha / Float64(BATCH))

        @always_inline
        fn rsample_bwd(
            agrad: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, ACTIONS + ACTIONS),
                MutAnyOrigin,
            ],
            ga: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            aps: Scalar[dtype],
            ca: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            eps: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, ACTIONS + ACTIONS),
                MutAnyOrigin,
            ],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
        ):
            sac_rsample_bwd_kernel[dtype, BATCH, ACTIONS](
                agrad, ga, aps, ca, eps, ao, lsmin, lsmax
            )

        ctx.enqueue_function[rsample_bwd, rsample_bwd](
            actor_grad_t,
            grad_act_t,
            alpha_per_sample,
            curr_act_t,
            eps_cache_t,
            actor_out_t,
            log_std_min_s,
            log_std_max_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # 9. Actor backward
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ](ws_ptr + W_DOBS)
        Network[ActorModel, ActorOpt].backward_gpu[BATCH](
            ctx,
            actor_grad_t,
            d_obs_t,
            actor_params,
            actor_cache_t,
            actor_grads,
            actor_ws,
        )

        # Return 0.0 — mean log_prob requires GPU->CPU sync which the agent
        # handles separately by reading curr_lp from strat_ws offset W_LP
        return 0.0


# =============================================================================
# AutodiffMaxEntLoss — composed autodiff replacement for MaxEntLoss
# =============================================================================


from mojo_rl.nn.model import (
    Sequential,
    RSample,
    Min,
    Slice,
)
from mojo_rl.nn.autodiff.combinators import SkipConcat, DualPath, SplitApply


struct AutodiffMaxEntLoss[
    log_std_scale: Float64 = 3.5,
](ActorLoss):
    """Max-entropy actor loss using composed autodiff graph.

    Replaces the manual 260-line forward/backward in MaxEntLoss with
    automatic differentiation via composed Model primitives:

        obs → SkipConcat[Actor → RSample] → [obs, action, log_prob]
            → SplitApply[DualPath[Critic, Critic] → Min, Identity]
            → [min_Q, log_prob]

    The backward pass is fully automatic — no manual gradient stitching.
    Produces identical gradients to MaxEntLoss but with ~30 lines of code.
    """

    comptime HAS_ALPHA: Bool = True

    @staticmethod
    fn gpu_lp_offset[
        BATCH: Int,
        ACTIONS: Int,
        ACTOR_OUT: Int,
        ACTOR_CS: Int,
    ]() -> Int:
        # For now, same offset as MaxEntLoss (agent reads log_probs from here)
        return BATCH * ACTOR_OUT + BATCH * ACTOR_CS + BATCH * ACTIONS

    @staticmethod
    fn ws_size[
        BATCH: Int,
        OBS: Int,
        ACTIONS: Int,
        ACTOR_OUT: Int,
        ACTOR_CS: Int,
        CRITIC_IN: Int,
        CRITIC_OUT: Int,
        CRITIC_CS: Int,
    ]() -> Int:
        # Workspace for the composed graph: we need space for the combined
        # params buffer, cache, output, grad_output, grad_input, and grad_params.
        # The graph bundles actor + critic1 + critic2 params.

        # Build the graph type to get its sizes
        comptime ActorGraph = Sequential[
            SkipConcat[
                Sequential[
                    # Placeholder: the actual actor model will be passed at call time.
                    # For workspace sizing, we use the maximum cache/workspace sizes.
                    Slice[ACTOR_OUT, 0, ACTOR_OUT],  # identity placeholder
                    RSample[ACTIONS],
                ]
            ],
        ]
        # Conservative workspace estimate — same as MaxEntLoss
        return (
            BATCH * ACTOR_OUT  # raw_out
            + BATCH * ACTOR_CS  # actor_cache
            + BATCH * ACTIONS  # mean
            + BATCH * ACTIONS  # log_std
            + BATCH * ACTIONS  # noise
            + BATCH * ACTIONS  # act
            + BATCH  # log_probs
            + BATCH * ACTIONS  # z_cache
            + BATCH * CRITIC_IN  # critic_input
            + BATCH * CRITIC_OUT  # Q1
            + BATCH * CRITIC_CS  # Q1 cache
            + BATCH * CRITIC_OUT  # Q2
            + BATCH * CRITIC_CS  # Q2 cache
            + BATCH * CRITIC_OUT  # dq1
            + BATCH * CRITIC_OUT  # dq2
            + BATCH * CRITIC_IN  # d_ci1
            + BATCH * CRITIC_IN  # d_ci2
            + BATCH * ACTIONS  # d_act
            + BATCH * ACTIONS  # grad_mean
            + BATCH * ACTIONS  # grad_log_std
            + BATCH * ACTOR_OUT  # actor_grad
            + BATCH * OBS  # d_obs
        )

    @staticmethod
    fn update_actor_cpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut actor_grads: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic2_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic2_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        ws: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        alpha: Float64,
    ) -> Float64:
        """Compute max-entropy actor gradient using composed autodiff graph.

        Returns mean log_prob for alpha auto-tuning.
        """
        comptime OBS = ActorModel.IN_DIM
        comptime ACTOR_PS = ActorModel.PARAM_SIZE
        comptime CRITIC_PS = CriticModel.PARAM_SIZE

        # =====================================================================
        # Build the composed SAC graph type
        # =====================================================================
        comptime ActorRSample = Sequential[ActorModel, RSample[ACTIONS]]
        comptime ActorSkip = SkipConcat[ActorRSample]
        # ActorSkip: IN=OBS, OUT=OBS+ACTIONS+1

        comptime TwinCriticMin = Sequential[
            DualPath[CriticModel, CriticModel], Min[1]
        ]
        comptime LogProbPass = Slice[1, 0, 1]
        comptime SACOutput = SplitApply[
            TwinCriticMin, LogProbPass, OBS + ACTIONS
        ]
        comptime SACGraph = Sequential[ActorSkip, SACOutput]
        # SACGraph: IN=OBS, OUT=2 (min_Q, log_prob)

        # =====================================================================
        # Assemble combined params: [actor | critic1 | critic2]
        # =====================================================================
        comptime TOTAL_PS = ACTOR_PS + 2 * CRITIC_PS
        var combined_params = InlineArray[Scalar[dtype], TOTAL_PS](
            uninitialized=True
        )
        for i in range(ACTOR_PS):
            combined_params[i] = actor_params.ptr[i]
        for i in range(CRITIC_PS):
            combined_params[ACTOR_PS + i] = critic_params.ptr[i]
        for i in range(CRITIC_PS):
            combined_params[ACTOR_PS + CRITIC_PS + i] = critic2_params.ptr[i]

        var params_t = LayoutTensor[
            dtype, Layout.row_major(SACGraph.PARAM_SIZE), MutAnyOrigin
        ](combined_params.unsafe_ptr())

        # =====================================================================
        # Forward: obs → [min_Q, log_prob]
        # =====================================================================
        var output = InlineArray[Scalar[dtype], BATCH * 2](uninitialized=True)
        var output_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ](output.unsafe_ptr())

        var cache = InlineArray[Scalar[dtype], BATCH * SACGraph.CACHE_SIZE](
            uninitialized=True
        )
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, SACGraph.CACHE_SIZE), MutAnyOrigin
        ](cache.unsafe_ptr())

        SACGraph.forward[BATCH](obs, output_t, params_t, cache_t)

        # =====================================================================
        # Backward: gradient seed = [-1/BS, alpha/BS] per sample
        # =====================================================================
        # output[:, 0] = min_Q → maximize → gradient = -1/BS
        # output[:, 1] = log_prob → entropy regularization → gradient = alpha/BS
        var grad_out = InlineArray[Scalar[dtype], BATCH * 2](
            uninitialized=True
        )
        var neg_inv_batch = Scalar[dtype](-1.0 / Float64(BATCH))
        var alpha_inv_batch = Scalar[dtype](alpha / Float64(BATCH))
        for b in range(BATCH):
            grad_out[b * 2] = neg_inv_batch
            grad_out[b * 2 + 1] = alpha_inv_batch

        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ](grad_out.unsafe_ptr())

        var grad_obs = InlineArray[Scalar[dtype], BATCH * OBS](
            uninitialized=True
        )
        var grad_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](grad_obs.unsafe_ptr())

        var combined_grads = InlineArray[Scalar[dtype], TOTAL_PS](
            uninitialized=True
        )
        for i in range(TOTAL_PS):
            combined_grads[i] = Scalar[dtype](0.0)
        var grads_t = LayoutTensor[
            dtype, Layout.row_major(SACGraph.PARAM_SIZE), MutAnyOrigin
        ](combined_grads.unsafe_ptr())

        SACGraph.backward[BATCH](
            grad_out_t, grad_obs_t, params_t, cache_t, grads_t
        )

        # =====================================================================
        # Scatter gradients back to separate actor/critic grad buffers
        # =====================================================================
        for i in range(ACTOR_PS):
            actor_grads.ptr[i] = combined_grads[i]
        for i in range(CRITIC_PS):
            critic_grads.ptr[i] = combined_grads[ACTOR_PS + i]
        for i in range(CRITIC_PS):
            critic2_grads.ptr[i] = combined_grads[ACTOR_PS + CRITIC_PS + i]

        # =====================================================================
        # Return mean log_prob for alpha update
        # =====================================================================
        var mean_lp: Float64 = 0.0
        for b in range(BATCH):
            var lp = Float64(output[b * 2 + 1])
            if lp != lp or lp > 100.0 or lp < -100.0:
                lp = -1.0
            mean_lp += lp
        mean_lp /= Float64(BATCH)
        return mean_lp

    @staticmethod
    fn update_actor_gpu[
        BATCH: Int,
        ACTIONS: Int,
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        ctx: DeviceContext,
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, ActorModel.IN_DIM), MutAnyOrigin
        ],
        actor_params: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut actor_grads: LayoutTensor[
            dtype, Layout.row_major(ActorModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        critic2_params: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        mut critic2_grads: LayoutTensor[
            dtype, Layout.row_major(CriticModel.PARAM_SIZE), MutAnyOrigin
        ],
        actor_ws: DeviceBuffer[dtype],
        critic_ws: DeviceBuffer[dtype],
        critic2_ws: DeviceBuffer[dtype],
        strat_ws: DeviceBuffer[dtype],
        dq_buf: DeviceBuffer[dtype],
        alpha: Float64,
        rng_seed: UInt32,
    ) raises -> Float64:
        """GPU max-entropy actor gradient via composed autodiff graph.

        Same graph as CPU but on GPU DeviceBuffers. Steps:
        1. Concat params [actor | critic1 | critic2] on GPU
        2. Forward graph on GPU
        3. Backward graph on GPU
        4. Scatter grads back to separate buffers
        5. Extract log_probs for alpha auto-tuning
        """
        comptime OBS = ActorModel.IN_DIM
        comptime ACTOR_PS = ActorModel.PARAM_SIZE
        comptime CRITIC_PS = CriticModel.PARAM_SIZE
        comptime TOTAL_PS = ACTOR_PS + 2 * CRITIC_PS

        # =================================================================
        # Build the composed SAC graph type (same as CPU)
        # =================================================================
        comptime ActorRSample = Sequential[ActorModel, RSample[ACTIONS]]
        comptime ActorSkip = SkipConcat[ActorRSample]
        comptime TwinCriticMin = Sequential[
            DualPath[CriticModel, CriticModel], Min[1]
        ]
        comptime LogProbPass = Slice[1, 0, 1]
        comptime SACOutput = SplitApply[
            TwinCriticMin, LogProbPass, OBS + ACTIONS
        ]
        comptime SACGraph = Sequential[ActorSkip, SACOutput]
        comptime GRAPH_CS = SACGraph.CACHE_SIZE
        comptime GRAPH_WS = SACGraph.WORKSPACE_SIZE_PER_SAMPLE

        # =================================================================
        # 1. Concat params on GPU: [actor | critic1 | critic2]
        # =================================================================
        var combined_params_buf = ctx.enqueue_create_buffer[dtype](TOTAL_PS)
        var params_t = LayoutTensor[
            dtype, Layout.row_major(TOTAL_PS), MutAnyOrigin
        ](combined_params_buf.unsafe_ptr())

        # Copy via GPU kernel (params are already in device memory)
        comptime PARAM_BLOCKS = (TOTAL_PS + TPB - 1) // TPB

        @always_inline
        fn concat_params_kernel(
            dst: LayoutTensor[
                dtype, Layout.row_major(TOTAL_PS), MutAnyOrigin
            ],
            ap: LayoutTensor[
                dtype, Layout.row_major(ACTOR_PS), MutAnyOrigin
            ],
            cp: LayoutTensor[
                dtype, Layout.row_major(CRITIC_PS), MutAnyOrigin
            ],
            c2p: LayoutTensor[
                dtype, Layout.row_major(CRITIC_PS), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= TOTAL_PS:
                return
            if i < ACTOR_PS:
                dst.ptr[i] = ap.ptr[i]
            elif i < ACTOR_PS + CRITIC_PS:
                dst.ptr[i] = cp.ptr[i - ACTOR_PS]
            else:
                dst.ptr[i] = c2p.ptr[i - ACTOR_PS - CRITIC_PS]

        # Rebind critic2_params to CriticModel.PARAM_SIZE layout
        var c2p_rb = LayoutTensor[
            dtype, Layout.row_major(CRITIC_PS), MutAnyOrigin
        ](critic2_params.ptr)

        ctx.enqueue_function[concat_params_kernel, concat_params_kernel](
            params_t,
            actor_params,
            critic_params,
            c2p_rb,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

        # =================================================================
        # 2. Allocate output, cache, workspace on GPU
        # =================================================================
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * 2)
        var output_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ](output_buf.unsafe_ptr())

        var cache_buf = ctx.enqueue_create_buffer[dtype](
            max(1, BATCH * GRAPH_CS)
        )

        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            max(1, BATCH * GRAPH_WS)
        )

        # =================================================================
        # 3. Forward: obs → [min_Q, log_prob]
        # =================================================================
        # Rebind params/cache to SACGraph's expected layout dimensions
        var graph_params = LayoutTensor[
            dtype, Layout.row_major(SACGraph.PARAM_SIZE), MutAnyOrigin
        ](combined_params_buf.unsafe_ptr())
        var graph_cache = LayoutTensor[
            dtype, Layout.row_major(BATCH, GRAPH_CS), MutAnyOrigin
        ](cache_buf.unsafe_ptr())
        var graph_obs = LayoutTensor[
            dtype, Layout.row_major(BATCH, SACGraph.IN_DIM), MutAnyOrigin
        ](obs.ptr)

        SACGraph.forward_gpu[BATCH](
            ctx,
            output_t,
            graph_obs,
            graph_params,
            graph_cache,
            workspace_buf,
        )

        # =================================================================
        # 4. Backward with gradient seed [-1/BS, alpha/BS]
        # =================================================================
        var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * 2)
        var neg_inv_batch = Scalar[dtype](-1.0 / Float64(BATCH))
        var alpha_inv_batch = Scalar[dtype](alpha / Float64(BATCH))
        for b in range(BATCH):
            grad_out_host[b * 2] = neg_inv_batch
            grad_out_host[b * 2 + 1] = alpha_inv_batch

        var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * 2)
        ctx.enqueue_copy(grad_out_buf, grad_out_host)

        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ](grad_out_buf.unsafe_ptr())

        var grad_obs_buf = ctx.enqueue_create_buffer[dtype](BATCH * OBS)

        # Combined grads buffer (zeroed)
        var combined_grads_buf = ctx.enqueue_create_buffer[dtype](TOTAL_PS)
        var zero_host = ctx.enqueue_create_host_buffer[dtype](TOTAL_PS)
        for i in range(TOTAL_PS):
            zero_host[i] = Scalar[dtype](0.0)
        ctx.enqueue_copy(combined_grads_buf, zero_host)

        # Rebind to graph's expected layout types
        var graph_grads = LayoutTensor[
            dtype, Layout.row_major(SACGraph.PARAM_SIZE), MutAnyOrigin
        ](combined_grads_buf.unsafe_ptr())
        var graph_grad_obs = LayoutTensor[
            dtype, Layout.row_major(BATCH, SACGraph.IN_DIM), MutAnyOrigin
        ](grad_obs_buf.unsafe_ptr())

        SACGraph.backward_gpu[BATCH](
            ctx,
            graph_grad_obs,
            grad_out_t,
            graph_params,
            graph_cache,
            graph_grads,
            workspace_buf,
        )

        # =================================================================
        # 5. Scatter grads back to separate actor/critic grad buffers
        # =================================================================
        var grads_t = LayoutTensor[
            dtype, Layout.row_major(TOTAL_PS), MutAnyOrigin
        ](combined_grads_buf.unsafe_ptr())
        var c2g_rb = LayoutTensor[
            dtype, Layout.row_major(CRITIC_PS), MutAnyOrigin
        ](critic2_grads.ptr)

        @always_inline
        fn scatter_grads_kernel(
            src: LayoutTensor[
                dtype, Layout.row_major(TOTAL_PS), MutAnyOrigin
            ],
            ag: LayoutTensor[
                dtype, Layout.row_major(ACTOR_PS), MutAnyOrigin
            ],
            cg: LayoutTensor[
                dtype, Layout.row_major(CRITIC_PS), MutAnyOrigin
            ],
            c2g: LayoutTensor[
                dtype, Layout.row_major(CRITIC_PS), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= TOTAL_PS:
                return
            if i < ACTOR_PS:
                ag.ptr[i] = src.ptr[i]
            elif i < ACTOR_PS + CRITIC_PS:
                cg.ptr[i - ACTOR_PS] = src.ptr[i]
            else:
                c2g.ptr[i - ACTOR_PS - CRITIC_PS] = src.ptr[i]

        ctx.enqueue_function[scatter_grads_kernel, scatter_grads_kernel](
            grads_t,
            actor_grads,
            critic_grads,
            c2g_rb,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

        # =================================================================
        # 6. Extract log_probs for alpha auto-tuning
        # =================================================================
        comptime LP_OFF = Self.gpu_lp_offset[
            BATCH, ACTIONS, ActorModel.OUT_DIM, ActorModel.CACHE_SIZE
        ]()

        # Extract log_probs from output_buf (every 2nd element starting at 1)
        var lp_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
        var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * 2)
        ctx.enqueue_copy(out_host, output_buf)
        ctx.synchronize()

        for b in range(BATCH):
            lp_host[b] = out_host[b * 2 + 1]

        # Copy log_probs to strat_ws at LP_OFF for the agent to read
        var lp_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        ctx.enqueue_copy(lp_buf, lp_host)

        # Copy to strat_ws at the right offset
        var strat_lp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](strat_ws.unsafe_ptr() + LP_OFF)
        var src_lp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](lp_buf.unsafe_ptr())

        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn copy_lp_kernel(
            dst: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            src: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < BATCH:
                dst.ptr[i] = src.ptr[i]

        ctx.enqueue_function[copy_lp_kernel, copy_lp_kernel](
            strat_lp_t,
            src_lp_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        return 0.0
