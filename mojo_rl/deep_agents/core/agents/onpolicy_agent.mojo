"""Generic on-policy agent parameterized by OnPolicyConfig.

Supports PPO (clipped surrogate, multi-epoch) and A2C (vanilla PG, single pass)
via strategy trait dispatch on Config.PolicyGrad and Config.EpochSched.

GPU support: GenericOnPolicyAgent also conforms to
GPUOnPolicyDiscreteAgent, enabling GPU-accelerated parallel-env training via
run_onpolicy_discrete_train_gpu.
"""

from std.math import exp, log, sqrt
from std.memory import UnsafePointer
from std.random import random_float64
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import (
    Model,
    Sequential,
    CategoricalLogProb,
    Ratio,
    ClipSurrogate,
    Slice,
)
from mojo_rl.nn.autodiff.combinators import SplitApply
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.core import (
    OnPolicyDiscreteState,
    OnPolicyDiscreteAgent,
    Checkpointable,
    GPUOnPolicyState,
    GPUOnPolicyDiscreteAgent,
)
from mojo_rl.deep_agents.core.training.gpu_onpolicy_train import (
    run_onpolicy_discrete_train_gpu,
)
from mojo_rl.deep_agents.core.training.onpolicy_helpers import (
    compute_gae_list,
    normalize_advantages_list,
    fisher_yates_shuffle,
)
from mojo_rl.deep_agents.core.perf_timer import PerfTimer
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.core.utils.softmax import (
    softmax_inline,
    sample_from_probs_inline,
    argmax_probs_inline,
)
from mojo_rl.deep_agents.core.kernels import (
    ppo_gather_minibatch_kernel,
    ppo_gather_minibatch_obs_parallel_kernel,
    ppo_critic_grad_kernel,
    ppo_critic_grad_clipped_kernel,
    normalize_advantages_kernel,
    gradient_norm_kernel,
    gradient_reduce_and_compute_scale_kernel,
    gradient_apply_scale_kernel,
    _store_pre_step_kernel,
    _store_pre_step_obs_parallel_kernel,
    _store_post_step_kernel,
)
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from ..configs.onpolicy_config import OnPolicyConfig
from ..workspace import RolloutWS, MinibatchWS, ActorTrainWS, CriticTrainWS


# =============================================================================
# Generic On-Policy CPU State
# =============================================================================


struct GenericOnPolicyCPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    num_actions: Int,
    rollout_len: Int,
](Movable, OnPolicyDiscreteState):
    """CPU state for discrete on-policy agents (PPO, A2C)."""

    comptime OBS = Self.ActorModel.IN_DIM
    comptime ACTIONS = Self.ActorModel.OUT_DIM
    comptime ROLLOUT = Self.rollout_len
    comptime CRITIC_IN = Self.CriticModel.IN_DIM
    comptime CRITIC_OUT = Self.CriticModel.OUT_DIM

    # Networks
    var actor: NetworkState[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkState[Self.CriticModel, Self.CriticOpt]

    # Rollout buffers
    var buffer_obs: List[Scalar[dtype]]
    var buffer_actions: List[Int]
    var buffer_rewards: List[Scalar[dtype]]
    var buffer_values: List[Scalar[dtype]]
    var buffer_log_probs: List[Scalar[dtype]]
    var buffer_dones: List[Bool]
    var buffer_idx: Int

    # Computed by compute_advantages
    var _advantages: List[Scalar[dtype]]
    var _returns: List[Scalar[dtype]]

    # Bootstrapping state
    var _current_obs: List[Scalar[dtype]]
    var _env_initialized: Bool

    # Minibatch indices
    var _indices: List[Int]

    def __init__(out self):
        self.actor = NetworkState[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier[]]()
        self.critic = NetworkState[Self.CriticModel, Self.CriticOpt]()
        self.critic.initialize[Xavier[]]()

        # Rollout buffers
        self.buffer_obs = List[Scalar[dtype]](capacity=Self.ROLLOUT * Self.OBS)
        self.buffer_actions = List[Int](capacity=Self.ROLLOUT)
        self.buffer_rewards = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        self.buffer_values = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        self.buffer_log_probs = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        self.buffer_dones = List[Bool](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT * Self.OBS):
            self.buffer_obs.append(Scalar[dtype](0))
        for _ in range(Self.ROLLOUT):
            self.buffer_actions.append(0)
            self.buffer_rewards.append(Scalar[dtype](0))
            self.buffer_values.append(Scalar[dtype](0))
            self.buffer_log_probs.append(Scalar[dtype](0))
            self.buffer_dones.append(False)
        self.buffer_idx = 0

        self._advantages = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        self._returns = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._advantages.append(Scalar[dtype](0))
            self._returns.append(Scalar[dtype](0))

        self._current_obs = List[Scalar[dtype]](capacity=Self.OBS)
        for _ in range(Self.OBS):
            self._current_obs.append(Scalar[dtype](0))
        self._env_initialized = False

        self._indices = List[Int](capacity=Self.ROLLOUT)
        for i in range(Self.ROLLOUT):
            self._indices.append(i)

    # OnPolicyDiscreteState trait
    def store_step(
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        value: Scalar[dtype],
        log_prob: Scalar[dtype],
        done: Bool,
    ) -> None:
        var idx = self.buffer_idx
        for i in range(Self.OBS):
            self.buffer_obs[idx * Self.OBS + i] = obs[i]
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = Scalar[dtype](reward)
        self.buffer_values[idx] = value
        self.buffer_log_probs[idx] = log_prob
        self.buffer_dones[idx] = done
        self.buffer_idx += 1

    def is_full(self) -> Bool:
        return self.buffer_idx >= Self.ROLLOUT

    def clear(mut self) -> None:
        self.buffer_idx = 0


# =============================================================================
# PPOGPUStateGeneric — GPU state container for generic on-policy PPO
# =============================================================================


struct PPOGPUStateGeneric[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    num_actions: Int,
    rollout_len: Int,
    n_envs: Int,
    gpu_minibatch: Int,
](GPUOnPolicyState, Movable):
    """GPU-resident state for generic discrete-action PPO training.

    Mirrors PPODiscreteGPUState but parameterized for the generic on-policy agent.

    Parameters:
        ActorModel: Actor network model type.
        ActorOpt: Actor optimizer type.
        CriticModel: Critic network model type.
        CriticOpt: Critic optimizer type.
        obs_dim: Observation space dimension.
        num_actions: Number of discrete actions.
        rollout_len: Steps per rollout per environment.
        n_envs: Number of parallel environments.
        gpu_minibatch: Minibatch size for update epochs.
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime ROLLOUT = Self.rollout_len
    comptime N = Self.n_envs
    comptime MB = Self.gpu_minibatch
    comptime ROLLOUT_TOTAL = Self.ROLLOUT * Self.N

    comptime ACTOR_PARAMS = Self.ActorModel.PARAM_SIZE
    comptime CRITIC_PARAMS = Self.CriticModel.PARAM_SIZE
    comptime ACTOR_GRAD_BLOCKS = (Self.ACTOR_PARAMS + TPB - 1) // TPB
    comptime CRITIC_GRAD_BLOCKS = (Self.CRITIC_PARAMS + TPB - 1) // TPB

    comptime ACTOR_WS_ENV = Self.N * Self.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime ACTOR_WS_MB = Self.MB * Self.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS_ENV = Self.N * Self.CriticModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS_MB = Self.MB * Self.CriticModel.WORKSPACE_SIZE_PER_SAMPLE

    # Autodiff loss graph compile-time sizes (for pre-allocated workspace)
    comptime _A = Self.ACTIONS
    comptime _LOSS_IN = Self._A + 3
    comptime _LossGraph = Sequential[
        SplitApply[
            CategoricalLogProb[Self._A],
            Slice[2, 0, 2],
            Self._A + 1,
        ],
        SplitApply[Ratio[1], Slice[1, 0, 1], 2],
        ClipSurrogate[0.2],
    ]
    comptime _LOSS_OUT = Self._LossGraph.OUT_DIM
    comptime _LOSS_CS = Self._LossGraph.CACHE_SIZE
    comptime _LOSS_WS = max(
        1, Self.MB * Self._LossGraph.WORKSPACE_SIZE_PER_SAMPLE
    )
    comptime _LOSS_PS = max(1, Self._LossGraph.PARAM_SIZE)
    comptime LOSS_WS_TOTAL = (
        Self.MB * Self._LOSS_IN  # loss_input
        + Self.MB * Self._LOSS_OUT  # loss_output
        + max(1, Self.MB * Self._LOSS_CS)  # loss_cache
        + Self._LOSS_PS  # loss_params
        + Self._LOSS_PS  # loss_grads
        + Self.MB * Self._LOSS_IN  # loss_grad_input
        + Self.MB * Self._LOSS_OUT  # loss_grad_output
        + Self._LOSS_WS  # loss_workspace
    )

    # Workspace type aliases
    comptime RolloutWSType = RolloutWS[Self.ROLLOUT_TOTAL, Self.OBS]
    comptime MinibatchWSType = MinibatchWS[Self.MB, Self.OBS]
    comptime ActorWSType = ActorTrainWS[
        Self.MB, Self.ACTIONS, Self.OBS, Self.ActorModel.CACHE_SIZE
    ]
    comptime CriticWSType = CriticTrainWS[
        Self.MB, 1, Self.OBS, Self.CriticModel.CACHE_SIZE
    ]

    # GPU networks (params + grads + optimizer state)
    var gpu_actor: GPUNetworkState[Self.ActorModel, Self.ActorOpt]
    var gpu_critic: GPUNetworkState[Self.CriticModel, Self.CriticOpt]

    # Consolidated workspace buffers
    var rollout_buf: DeviceBuffer[
        dtype
    ]  # RolloutWS: obs, actions, lp, values, rewards, dones, adv, ret
    var minibatch_buf: DeviceBuffer[
        dtype
    ]  # MinibatchWS: obs, actions, adv, ret, old_lp, old_values
    var actor_train_buf: DeviceBuffer[
        dtype
    ]  # ActorTrainWS: logits, cache, grad_out, grad_in
    var critic_train_buf: DeviceBuffer[
        dtype
    ]  # CriticTrainWS: values, cache, grad_out, grad_in
    var rollout_step: Int

    # Pinned host buffers for GAE computation
    var rollout_rewards_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_values_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var rollout_dones_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var advantages_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var returns_host: HostBuffer[dtype]  # [ROLLOUT_TOTAL]
    var bootstrap_values_host: HostBuffer[dtype]  # [N]

    # Minibatch indices (int32 — separate from dtype workspace)
    var mb_indices_buf: DeviceBuffer[DType.int32]  # [MB]
    var mb_indices_host: HostBuffer[DType.int32]  # [MB]

    # Inference logits (N_ENVS sized, not MB)
    var logits_buf: DeviceBuffer[dtype]  # [N * ACTIONS]

    # Diagnostic / scratch buffers
    var kl_divergences_buf: DeviceBuffer[dtype]  # [MB]
    var kl_divergences_host: HostBuffer[dtype]  # [MB]
    var mb_advantages_host: HostBuffer[dtype]  # [MB]
    var diag_entropy_buf: DeviceBuffer[dtype]  # [MB]
    var diag_entropy_host: HostBuffer[dtype]  # [MB]
    var diag_clip_buf: DeviceBuffer[dtype]  # [MB]
    var diag_clip_host: HostBuffer[dtype]  # [MB]
    var diag_values_host: HostBuffer[dtype]  # [MB]
    var diag_returns_host: HostBuffer[dtype]  # [MB]
    var actor_grad_partial_sums_buf: DeviceBuffer[dtype]  # [ACTOR_GRAD_BLOCKS]
    var critic_grad_partial_sums_buf: DeviceBuffer[
        dtype
    ]  # [CRITIC_GRAD_BLOCKS]
    var actor_scale_buf: DeviceBuffer[dtype]  # [1]
    var critic_scale_buf: DeviceBuffer[dtype]  # [1]
    var actor_env_workspace_buf: DeviceBuffer[dtype]
    var actor_mb_workspace_buf: DeviceBuffer[dtype]
    var critic_env_workspace_buf: DeviceBuffer[dtype]
    var critic_mb_workspace_buf: DeviceBuffer[dtype]

    # Pre-allocated autodiff loss workspace (avoids per-minibatch allocations)
    var loss_ws_buf: DeviceBuffer[dtype]  # [LOSS_WS_TOTAL]

    # Env-step scratch buffers
    var values_env_buf: DeviceBuffer[dtype]  # [N]
    var log_probs_env_buf: DeviceBuffer[dtype]  # [N]

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU device and pinned host buffers."""
        self.gpu_actor = GPUNetworkState[Self.ActorModel, Self.ActorOpt](ctx)
        self.gpu_critic = GPUNetworkState[Self.CriticModel, Self.CriticOpt](ctx)

        # Consolidated workspace allocations (4 buffers replace 20+)
        self.rollout_buf = Self.RolloutWSType.alloc_gpu(ctx)
        self.minibatch_buf = Self.MinibatchWSType.alloc_gpu(ctx)
        self.actor_train_buf = Self.ActorWSType.alloc_gpu(ctx)
        self.critic_train_buf = Self.CriticWSType.alloc_gpu(ctx)
        self.rollout_step = 0

        # Pinned host buffers for GAE computation
        self.rollout_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.rollout_dones_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.advantages_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.returns_host = ctx.enqueue_create_host_buffer[dtype](
            Self.ROLLOUT_TOTAL
        )
        self.bootstrap_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.N
        )

        # Minibatch indices (int32)
        self.mb_indices_buf = ctx.enqueue_create_buffer[DType.int32](Self.MB)
        self.mb_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            Self.MB
        )

        # Inference logits
        self.logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.N * Self.ACTIONS
        )

        # Diagnostic / scratch buffers
        self.kl_divergences_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.kl_divergences_host = ctx.enqueue_create_host_buffer[dtype](
            Self.MB
        )
        self.mb_advantages_host = ctx.enqueue_create_host_buffer[dtype](Self.MB)
        self.diag_entropy_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.diag_entropy_host = ctx.enqueue_create_host_buffer[dtype](Self.MB)
        self.diag_clip_buf = ctx.enqueue_create_buffer[dtype](Self.MB)
        self.diag_clip_host = ctx.enqueue_create_host_buffer[dtype](Self.MB)
        self.diag_values_host = ctx.enqueue_create_host_buffer[dtype](Self.MB)
        self.diag_returns_host = ctx.enqueue_create_host_buffer[dtype](Self.MB)

        self.actor_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            Self.ACTOR_GRAD_BLOCKS
        )
        self.critic_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            Self.CRITIC_GRAD_BLOCKS
        )
        self.actor_scale_buf = ctx.enqueue_create_buffer[dtype](1)
        self.critic_scale_buf = ctx.enqueue_create_buffer[dtype](1)

        comptime actor_ws_size = Self.ACTOR_WS_ENV if Self.ACTOR_WS_ENV > 0 else 1
        comptime actor_mb_ws_size = Self.ACTOR_WS_MB if Self.ACTOR_WS_MB > 0 else 1
        comptime critic_ws_size = Self.CRITIC_WS_ENV if Self.CRITIC_WS_ENV > 0 else 1
        comptime critic_mb_ws_size = Self.CRITIC_WS_MB if Self.CRITIC_WS_MB > 0 else 1

        self.actor_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            actor_ws_size
        )
        self.actor_mb_workspace_buf = ctx.enqueue_create_buffer[dtype](
            actor_mb_ws_size
        )
        self.critic_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            critic_ws_size
        )
        self.critic_mb_workspace_buf = ctx.enqueue_create_buffer[dtype](
            critic_mb_ws_size
        )

        self.loss_ws_buf = ctx.enqueue_create_buffer[dtype](Self.LOSS_WS_TOTAL)

        self.values_env_buf = ctx.enqueue_create_buffer[dtype](Self.N)
        self.log_probs_env_buf = ctx.enqueue_create_buffer[dtype](Self.N)

    # -------------------------------------------------------------------------
    # Workspace accessors
    # -------------------------------------------------------------------------

    def rollout_ws(self) -> Self.RolloutWSType:
        """Get typed rollout workspace view."""
        return Self.RolloutWSType(self.rollout_buf.unsafe_ptr())

    def minibatch_ws(self) -> Self.MinibatchWSType:
        """Get typed minibatch workspace view."""
        return Self.MinibatchWSType(self.minibatch_buf.unsafe_ptr())

    def actor_ws(self) -> Self.ActorWSType:
        """Get typed actor training workspace view."""
        return Self.ActorWSType(self.actor_train_buf.unsafe_ptr())

    def critic_ws(self) -> Self.CriticWSType:
        """Get typed critic training workspace view."""
        return Self.CriticWSType(self.critic_train_buf.unsafe_ptr())

    # -------------------------------------------------------------------------
    # GPUOnPolicyState trait methods
    # -------------------------------------------------------------------------

    def gpu_rollout_reset(mut self) -> None:
        """Reset rollout write pointer to 0 for the next update cycle."""
        self.rollout_step = 0

    def gpu_store_pre_step[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        log_probs_buf: DeviceBuffer[dtype],
        values_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Store pre-step data (obs, actions, log_probs, values) into rollout buffers.
        """
        var t_offset = self.rollout_step * N_ENVS
        var rws = self.rollout_ws()
        var r_obs = rws.obs_at[N_ENVS](t_offset)
        var r_actions = rws.actions_at[N_ENVS](t_offset)
        var r_log_probs = rws.log_probs_at[N_ENVS](t_offset)
        var r_values = rws.values_at[N_ENVS](t_offset)

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var log_probs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())
        var values_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](values_buf.unsafe_ptr())

        # Parallel obs store: 2D grid (OBS_BLOCKS, N_ENVS)
        comptime obs_store_wrapper = _store_pre_step_obs_parallel_kernel[
            dtype, N_ENVS, Self.OBS
        ]
        comptime OBS_BLOCKS = (Self.OBS + TPB - 1) // TPB
        ctx.enqueue_function[obs_store_wrapper, obs_store_wrapper](
            r_obs,
            obs_t,
            grid_dim=(OBS_BLOCKS, N_ENVS),
            block_dim=(TPB,),
        )

        # Scalar store: actions, log_probs, values (tiny kernel)
        comptime blocks = (N_ENVS + TPB - 1) // TPB

        @parameter
        @always_inline
        def store_scalars_wrapper(
            r_a: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            r_lp: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            r_v: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            a: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            lp: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            v: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= N_ENVS:
                return
            r_a[i] = a[i]
            r_lp[i] = lp[i]
            r_v[i] = v[i]

        ctx.enqueue_function[store_scalars_wrapper, store_scalars_wrapper](
            r_actions,
            r_log_probs,
            r_values,
            actions_t,
            log_probs_t,
            values_t,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

    def gpu_store_post_step[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        rewards_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Store post-step data (rewards, dones) into rollout buffers, advance pointer.
        """
        var t_offset = self.rollout_step * N_ENVS
        var rws = self.rollout_ws()
        var r_rewards = rws.rewards_at[N_ENVS](t_offset)
        var r_dones = rws.dones_at[N_ENVS](t_offset)

        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime store_wrapper = _store_post_step_kernel[dtype, N_ENVS]
        comptime blocks = (N_ENVS + TPB - 1) // TPB
        ctx.enqueue_function[store_wrapper, store_wrapper](
            r_rewards,
            r_dones,
            rewards_t,
            dones_t,
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )
        self.rollout_step += 1

    def gpu_rollout_is_full(self) -> Bool:
        """Return True when rollout_len steps have been stored."""
        return self.rollout_step >= Self.ROLLOUT


# =============================================================================
# GPU sample actions kernel (discrete categorical sampling)
# =============================================================================


def _generic_sample_actions_kernel[
    dtype: DType where dtype.is_floating_point(),
    N_ENVS: Int,
    NUM_ACTIONS: Int,
](
    logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, NUM_ACTIONS), MutAnyOrigin
    ],
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    seed: Scalar[DType.uint32],
):
    """Sample actions from categorical distribution and compute log probs."""
    from std.random.philox import Random as PhiloxRandom
    from std.gpu import block_dim, block_idx, thread_idx

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    # Per-thread RNG using PhiloxRandom
    var rng = PhiloxRandom(
        seed=UInt64(seed) * UInt64(N_ENVS) + UInt64(i), offset=0
    )
    var rand_vals = rng.step_uniform()
    var rand_val = Scalar[dtype](rand_vals[0])

    # Compute softmax probabilities
    var max_logit = logits[i, 0]
    for a in range(1, NUM_ACTIONS):
        var l = logits[i, a]
        if l > max_logit:
            max_logit = l

    var sum_exp = logits[i, 0] - logits[i, 0]  # zero with correct type
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        sum_exp = sum_exp + exp(logit_val)

    # Sample action using cumulative sum
    var cumsum_val = Scalar[dtype](0.0)
    var selected_action: actions.element_type = 0
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        var prob = exp(logit_val) / sum_exp
        var prob_scalar = Scalar[dtype](prob[0])
        cumsum_val = cumsum_val + prob_scalar
        if rand_val < cumsum_val:
            selected_action = Scalar[dtype](a)
            break

    actions[i] = selected_action

    # Compute log probability
    var logit_sel = logits[i, Int(selected_action)] - max_logit
    var selected_prob_simd = exp(logit_sel) / sum_exp
    var selected_prob = Float32(selected_prob_simd[0])
    var eps = Float32(1e-8)
    var log_prob_val = log(selected_prob + eps)
    log_probs[i] = Scalar[dtype](log_prob_val)


# =============================================================================
# GenericOnPolicyAgent[Config: OnPolicyConfig]
# =============================================================================


struct GenericOnPolicyAgent[
    Config: OnPolicyConfig,
    n_envs: Int = 1024,
    gpu_minibatch_size: Int = 256,
](OnPolicyDiscreteAgent & GPUOnPolicyDiscreteAgent & Checkpointable):
    """Generic on-policy agent. PPO vs A2C via Config.PolicyGrad + Config.EpochSched.

    Also conforms to GPUOnPolicyDiscreteAgent for GPU-accelerated training.

    Parameters:
        Config: On-policy configuration trait (PPOConfig or A2CConfig).
        n_envs: Number of parallel environments for GPU training (default: 1024).
        gpu_minibatch_size: Minibatch size for GPU update epochs (default: 256).
    """

    # Derive ALL dimensions from Model types for unification consistency
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.ActorModel.OUT_DIM  # Must match ActorModel.OUT_DIM
    comptime ROLLOUT: Int = Self.Config.rollout_len
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM  # Should == OBS
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]
    comptime ActorModel = Self.Config.ActorModel
    comptime CriticModel = Self.Config.CriticModel
    comptime ActorOpt = Self.Config.ActorOpt
    comptime CriticOpt = Self.Config.CriticOpt

    comptime CPUStateType = GenericOnPolicyCPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.rollout_len,
    ]

    # GPU-specific comptime constants
    comptime TOTAL_ROLLOUT_SIZE: Int = Self.n_envs * Self.Config.rollout_len
    comptime GPU_MINIBATCH: Int = Self.gpu_minibatch_size

    # GPUOnPolicyDiscreteAgent trait constants
    comptime OBS_DIM: Int = Self.OBS
    comptime NUM_ACTIONS: Int = Self.ACTIONS
    comptime ROLLOUT_LEN: Int = Self.ROLLOUT
    comptime MAX_N_ENVS: Int = Self.n_envs

    comptime GPUStateType = PPOGPUStateGeneric[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.OBS,
        Self.ACTIONS,
        Self.ROLLOUT,
        Self.n_envs,
        Self.GPU_MINIBATCH,
    ]

    # Internal CPU state for GPU upload/download
    var cpu_state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var gae_lambda: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var max_grad_norm: Float64
    var normalize_advantages: Bool

    # PPO-specific
    var clip_epsilon: Float64
    var num_epochs: Int
    var minibatch_size: Int
    var target_kl: Float64
    var clip_value: Bool
    var norm_adv_per_minibatch: Bool

    # Training state
    var train_step_count: Int
    var target_total_steps: Int

    # Checkpoint
    var checkpoint_every: Int
    var checkpoint_path: String

    def __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        max_grad_norm: Float64 = 0.5,
        normalize_advantages: Bool = True,
        clip_epsilon: Float64 = 0.2,
        num_epochs: Int = 4,
        minibatch_size: Int = 64,
        target_kl: Float64 = 0.015,
        clip_value: Bool = True,
        norm_adv_per_minibatch: Bool = True,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        target_total_steps: Int = 0,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.clip_epsilon = clip_epsilon
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.clip_value = clip_value
        self.norm_adv_per_minibatch = norm_adv_per_minibatch
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.cpu_state = Self.CPUStateType()

    # =========================================================================
    # OnPolicyDiscreteAgent trait
    # =========================================================================

    def make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    def collect_rollout[
        E: BoxDiscreteActionEnv
    ](mut self, mut cpu_state: Self.CPUStateType, mut env: E) -> None:
        if not cpu_state._env_initialized:
            var obs_list = env.reset_obs_list()
            for i in range(Self.OBS):
                cpu_state._current_obs[i] = Scalar[dtype](obs_list[i])
            cpu_state._env_initialized = True

        cpu_state.buffer_idx = 0

        for _ in range(Self.ROLLOUT):
            # Build obs tensor
            var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                uninitialized=True
            )
            for i in range(Self.OBS):
                obs_arr[i] = cpu_state._current_obs[i]
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](obs_arr.unsafe_ptr())

            # Actor forward → logits
            var logits_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            var logits_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](logits_arr.unsafe_ptr())
            var p_a = cpu_state.actor.params_view()
            var s_a = cpu_state.actor.model_state_view()
            Self.ActorNet.forward[1](obs_t, logits_t, p_a, s_a)

            # Softmax → probs → sample
            var probs = softmax_inline[dtype, Self.ACTIONS](logits_arr)
            var action = sample_from_probs_inline[dtype, Self.ACTIONS](probs)
            var log_prob = log(probs[action] + Scalar[dtype](1e-8))

            # Critic forward → value (use CRITIC_IN for input dim)
            var c_obs_arr = InlineArray[Scalar[dtype], Self.CRITIC_IN](
                uninitialized=True
            )
            for ci in range(Self.CRITIC_IN):
                c_obs_arr[ci] = obs_arr[ci]
            var c_obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
            ](c_obs_arr.unsafe_ptr())
            var val_arr = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                uninitialized=True
            )
            var val_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.CRITIC_OUT), MutAnyOrigin
            ](val_arr.unsafe_ptr())
            var p_c = cpu_state.critic.params_view()
            var s_c = cpu_state.critic.model_state_view()
            Self.CriticNet.forward[1](c_obs_t, val_t, p_c, s_c)
            var value = val_arr[0]

            # Step env
            var result = env.step_obs(action)
            var reward = Float64(result[1])
            var done = result[2]

            # Store in buffer
            var idx = cpu_state.buffer_idx
            for i in range(Self.OBS):
                cpu_state.buffer_obs[idx * Self.OBS + i] = obs_arr[i]
            cpu_state.buffer_actions[idx] = action
            cpu_state.buffer_rewards[idx] = Scalar[dtype](reward)
            cpu_state.buffer_values[idx] = value
            cpu_state.buffer_log_probs[idx] = log_prob
            cpu_state.buffer_dones[idx] = done
            cpu_state.buffer_idx += 1

            # Update current obs
            if done:
                var next_obs = env.reset_obs_list()
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](next_obs[i])
            else:
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](result[0][i])

    def compute_advantages(mut self, mut cpu_state: Self.CPUStateType) -> None:
        var buf_len = cpu_state.buffer_idx
        if buf_len == 0:
            return

        # Bootstrap value from current obs (use CRITIC_IN for input)
        var c_obs_arr = InlineArray[Scalar[dtype], Self.CRITIC_IN](
            uninitialized=True
        )
        for i in range(Self.CRITIC_IN):
            c_obs_arr[i] = cpu_state._current_obs[i]
        var c_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
        ](c_obs_arr.unsafe_ptr())
        var val_arr = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
            uninitialized=True
        )
        var val_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.CRITIC_OUT), MutAnyOrigin
        ](val_arr.unsafe_ptr())
        var p_c = cpu_state.critic.params_view()
        var s_c = cpu_state.critic.model_state_view()
        Self.CriticNet.forward[1](c_obs_t, val_t, p_c, s_c)
        var next_value = val_arr[0]

        compute_gae_list[dtype](
            cpu_state.buffer_rewards,
            cpu_state.buffer_values,
            cpu_state.buffer_dones,
            next_value,
            buf_len,
            self.gamma,
            self.gae_lambda,
            cpu_state._advantages,
            cpu_state._returns,
        )

        if self.normalize_advantages and buf_len > 1:
            normalize_advantages_list[dtype](cpu_state._advantages, buf_len)

    def update_epochs(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        var buf_len = cpu_state.buffer_idx
        if buf_len == 0:
            return 0.0

        # Determine number of epochs and minibatch size
        var n_epochs = Self.Config.EpochSched.get_num_epochs(self.num_epochs)
        var mb_size = Self.Config.EpochSched.get_minibatch_size(
            self.minibatch_size, buf_len
        )

        var total_loss: Float64 = 0.0
        var sample_count = 0

        for epoch in range(n_epochs):
            # Shuffle indices each epoch (PPO)
            comptime if Self.Config.EpochSched.USES_SHUFFLE:
                fisher_yates_shuffle(cpu_state._indices, buf_len)

            var batch_start = 0
            while batch_start < buf_len:
                var batch_end = batch_start + mb_size
                if batch_end > buf_len:
                    batch_end = buf_len
                var this_mb = batch_end - batch_start

                # Per-minibatch advantage normalization (PPO)
                comptime if Self.Config.EpochSched.SUPPORTS_MINIBATCH_NORM:
                    if self.norm_adv_per_minibatch and this_mb > 1:
                        var mb_adv = List[Scalar[dtype]](capacity=this_mb)
                        for b in range(batch_start, batch_end):
                            var t = cpu_state._indices[b]
                            mb_adv.append(cpu_state._advantages[t])
                        normalize_advantages_list[dtype](mb_adv, this_mb)
                        for b in range(this_mb):
                            var t = cpu_state._indices[batch_start + b]
                            cpu_state._advantages[t] = mb_adv[b]

                # Process each sample in the minibatch
                for b_idx in range(batch_start, batch_end):
                    var t = b_idx
                    comptime if Self.Config.EpochSched.USES_SHUFFLE:
                        t = cpu_state._indices[b_idx]

                    var old_log_prob = cpu_state.buffer_log_probs[t]
                    var advantage = cpu_state._advantages[t]
                    var return_t = cpu_state._returns[t]
                    var action = cpu_state.buffer_actions[t]
                    var old_value = cpu_state.buffer_values[t]

                    # Actor forward
                    var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                        uninitialized=True
                    )
                    for i in range(Self.OBS):
                        obs_arr[i] = cpu_state.buffer_obs[t * Self.OBS + i]
                    var obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](obs_arr.unsafe_ptr())

                    var logits_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    var logits_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](logits_arr.unsafe_ptr())
                    var actor_cache = InlineArray[Scalar[dtype], Self.ACTOR_CS](
                        uninitialized=True
                    )
                    var actor_cache_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTOR_CS), MutAnyOrigin
                    ](actor_cache.unsafe_ptr())
                    var p_a = cpu_state.actor.params_view()
                    var s_a = cpu_state.actor.model_state_view()
                    Self.ActorNet.forward_with_cache[1](
                        obs_t, logits_t, p_a, s_a, actor_cache_t
                    )

                    var probs = softmax_inline[dtype, Self.ACTIONS](logits_arr)
                    var new_log_prob = log(probs[action] + Scalar[dtype](1e-8))

                    # Entropy
                    var entropy = Scalar[dtype](0.0)
                    for a in range(Self.ACTIONS):
                        if probs[a] > Scalar[dtype](1e-8):
                            entropy -= probs[a] * log(probs[a])

                    # Policy gradient
                    var d_logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )

                    comptime if Self.Config.USE_AUTODIFF_GRAD:
                        # ---- True autodiff: LossGraph forward + backward ----
                        comptime A = Self.ACTIONS
                        comptime LOSS_IN = A + 3
                        comptime LossGraph = Sequential[
                            SplitApply[
                                CategoricalLogProb[A],
                                Slice[2, 0, 2],
                                A + 1,
                            ],
                            SplitApply[Ratio[1], Slice[1, 0, 1], 2],
                            ClipSurrogate[0.2],
                        ]
                        comptime LOSS_OUT = LossGraph.OUT_DIM  # 1
                        comptime LOSS_CS = LossGraph.CACHE_SIZE

                        # Pack input: [logits(A) || action_idx(1) || old_lp(1) || adv(1)]
                        var loss_in_arr = InlineArray[Scalar[dtype], LOSS_IN](
                            uninitialized=True
                        )
                        for j in range(A):
                            loss_in_arr[j] = logits_arr[j]
                        loss_in_arr[A] = Scalar[dtype](Float64(action))
                        loss_in_arr[A + 1] = old_log_prob
                        loss_in_arr[A + 2] = advantage

                        var loss_in_t = LayoutTensor[
                            dtype,
                            Layout.row_major(1, LOSS_IN),
                            MutAnyOrigin,
                        ](loss_in_arr.unsafe_ptr())

                        # Forward
                        var loss_out_arr = InlineArray[Scalar[dtype], LOSS_OUT](
                            uninitialized=True
                        )
                        var loss_out_t = LayoutTensor[
                            dtype,
                            Layout.row_major(1, LOSS_OUT),
                            MutAnyOrigin,
                        ](loss_out_arr.unsafe_ptr())
                        var loss_cache_arr = InlineArray[
                            Scalar[dtype], LOSS_CS
                        ](uninitialized=True)
                        var loss_cache_t = LayoutTensor[
                            dtype,
                            Layout.row_major(1, LOSS_CS),
                            MutAnyOrigin,
                        ](loss_cache_arr.unsafe_ptr())
                        var loss_params_arr = InlineArray[
                            Scalar[dtype], max(1, LossGraph.PARAM_SIZE)
                        ](fill=Scalar[dtype](0.0))
                        var loss_params_t = LayoutTensor[
                            dtype,
                            Layout.row_major(LossGraph.PARAM_SIZE),
                            MutAnyOrigin,
                        ](loss_params_arr.unsafe_ptr())

                        # Zero-length model state slice (LossGraph is stateless)
                        var loss_state_t = LayoutTensor[
                            dtype, Layout.row_major(LossGraph.STATE_SIZE), MutAnyOrigin
                        ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))

                        LossGraph.forward[1](
                            loss_in_t,
                            loss_out_t,
                            loss_params_t,
                            loss_state_t,
                            loss_cache_t,
                        )

                        # Backward
                        var loss_go_arr = InlineArray[Scalar[dtype], LOSS_OUT](
                            uninitialized=True
                        )
                        loss_go_arr[0] = Scalar[dtype](1.0)
                        var loss_go_t = LayoutTensor[
                            dtype,
                            Layout.row_major(1, LOSS_OUT),
                            MutAnyOrigin,
                        ](loss_go_arr.unsafe_ptr())
                        var loss_gi_arr = InlineArray[Scalar[dtype], LOSS_IN](
                            uninitialized=True
                        )
                        var loss_gi_t = LayoutTensor[
                            dtype,
                            Layout.row_major(1, LOSS_IN),
                            MutAnyOrigin,
                        ](loss_gi_arr.unsafe_ptr())
                        var loss_grads_arr = InlineArray[
                            Scalar[dtype], max(1, LossGraph.PARAM_SIZE)
                        ](fill=Scalar[dtype](0.0))
                        var loss_grads_t = LayoutTensor[
                            dtype,
                            Layout.row_major(LossGraph.PARAM_SIZE),
                            MutAnyOrigin,
                        ](loss_grads_arr.unsafe_ptr())

                        LossGraph.backward[1](
                            loss_go_t,
                            loss_gi_t,
                            loss_params_t,
                            loss_state_t,
                            loss_cache_t,
                            loss_grads_t,
                        )

                        # Extract d_logits[:A] and add entropy bonus
                        for j in range(A):
                            # d_logit from graph
                            var d_lp = loss_gi_arr[j]
                            # Entropy gradient: d(-sum(p*log(p)))/d(logit[j]) = -p[j]*(1+log(p[j]))
                            var d_ent = -probs[j] * (
                                Scalar[dtype](1.0)
                                + log(probs[j] + Scalar[dtype](1e-8))
                            )
                            d_logits[j] = (
                                d_lp - Scalar[dtype](self.entropy_coef) * d_ent
                            )

                    else:
                        Self.Config.PolicyGrad.compute_d_logits[Self.ACTIONS](
                            probs,
                            action,
                            new_log_prob,
                            old_log_prob,
                            advantage,
                            self.clip_epsilon,
                            self.entropy_coef,
                            d_logits,
                        )

                    # Actor backward
                    var d_logits_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](d_logits.unsafe_ptr())
                    var d_obs = InlineArray[Scalar[dtype], Self.OBS](
                        uninitialized=True
                    )
                    var d_obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](d_obs.unsafe_ptr())
                    var g_a = cpu_state.actor.grads_view()
                    Self.ActorNet.backward[1](
                        d_logits_t, d_obs_t, p_a, s_a, actor_cache_t, g_a
                    )
                    cpu_state.actor.optimizer_step()

                    # Critic forward + backward (use CRITIC_IN for input)
                    var c_obs = InlineArray[Scalar[dtype], Self.CRITIC_IN](
                        uninitialized=True
                    )
                    for ci in range(Self.CRITIC_IN):
                        c_obs[ci] = obs_arr[ci]
                    var c_obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
                    ](c_obs.unsafe_ptr())
                    var val_out = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                        uninitialized=True
                    )
                    var val_out_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_OUT),
                        MutAnyOrigin,
                    ](val_out.unsafe_ptr())
                    var critic_cache = InlineArray[
                        Scalar[dtype], Self.CRITIC_CS
                    ](uninitialized=True)
                    var critic_cache_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.CRITIC_CS), MutAnyOrigin
                    ](critic_cache.unsafe_ptr())
                    var p_c = cpu_state.critic.params_view()
                    var s_c = cpu_state.critic.model_state_view()
                    Self.CriticNet.forward_with_cache[1](
                        c_obs_t, val_out_t, p_c, s_c, critic_cache_t
                    )
                    var value = val_out[0]

                    var d_value = InlineArray[Scalar[dtype], Self.CRITIC_OUT](
                        uninitialized=True
                    )
                    d_value[0] = (
                        Scalar[dtype](2.0)
                        * Scalar[dtype](self.value_loss_coef)
                        * (value - return_t)
                    )
                    var d_value_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CRITIC_OUT),
                        MutAnyOrigin,
                    ](d_value.unsafe_ptr())
                    var d_obs_c = InlineArray[Scalar[dtype], Self.CRITIC_IN](
                        uninitialized=True
                    )
                    var d_obs_c_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.CRITIC_IN), MutAnyOrigin
                    ](d_obs_c.unsafe_ptr())
                    var g_c = cpu_state.critic.grads_view()
                    Self.CriticNet.backward[1](
                        d_value_t, d_obs_c_t, p_c, s_c, critic_cache_t, g_c
                    )
                    cpu_state.critic.optimizer_step()

                    total_loss += Float64(-new_log_prob * advantage)
                    sample_count += 1

                batch_start = batch_end

        self.train_step_count += 1
        if sample_count > 0:
            return total_loss / Float64(sample_count)
        return 0.0

    def select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[dtype](obs[i])
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var logits_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](logits_arr.unsafe_ptr())
        var p = cpu_state.actor.params_view()
        var s = cpu_state.actor.model_state_view()
        Self.ActorNet.forward[1](obs_t, logits_t, p, s)

        var probs = softmax_inline[dtype, Self.ACTIONS](logits_arr)
        var action = argmax_probs_inline[dtype, Self.ACTIONS](probs)

        var result = List[Float64](capacity=1)
        result.append(Float64(action))
        return result^

    def get_explore_rate(self) -> Float64:
        return self.entropy_coef

    # Checkpointable
    def save_checkpoint(self, path: String) raises -> None:
        """Save agent state to a checkpoint file.

        Saves actor and critic network params + optimizer states,
        plus hyperparameters. Rollout buffer is NOT saved.
        """
        from mojo_rl.nn.checkpoint import (
            write_checkpoint_header,
            write_metadata_section,
            save_checkpoint_file,
        )

        var content = write_checkpoint_header(
            "generic_onpolicy_agent",
            Self.Config.ActorModel.PARAM_SIZE
            + Self.Config.CriticModel.PARAM_SIZE,
            0,
        )
        content += self.cpu_state.actor.write_sections("actor_")
        content += self.cpu_state.critic.write_sections("critic_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("gae_lambda=" + String(self.gae_lambda))
        metadata.append("entropy_coef=" + String(self.entropy_coef))
        metadata.append("clip_epsilon=" + String(self.clip_epsilon))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(path, content)

    def load_checkpoint(mut self, path: String) raises -> None:
        """Load agent state from a checkpoint file."""
        from mojo_rl.nn.checkpoint import (
            read_checkpoint_file,
            parse_checkpoint_header,
            read_metadata_section,
            get_metadata_value,
        )

        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)

        self.cpu_state.actor.read_sections(content, "actor_")
        self.cpu_state.critic.read_sections(content, "critic_")

        var metadata = read_metadata_section(content)

        var gamma_str = get_metadata_value(metadata, "gamma")
        if gamma_str.byte_length() > 0:
            self.gamma = atof(gamma_str)

        var gae_str = get_metadata_value(metadata, "gae_lambda")
        if gae_str.byte_length() > 0:
            self.gae_lambda = atof(gae_str)

        var entropy_str = get_metadata_value(metadata, "entropy_coef")
        if entropy_str.byte_length() > 0:
            self.entropy_coef = atof(entropy_str)

        var clip_str = get_metadata_value(metadata, "clip_epsilon")
        if clip_str.byte_length() > 0:
            self.clip_epsilon = atof(clip_str)

        var step_str = get_metadata_value(metadata, "train_step_count")
        if step_str.byte_length() > 0:
            self.train_step_count = Int(atol(step_str))

    # =========================================================================
    # CPU Convenience training
    # =========================================================================

    def train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_updates: Int = 1000,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train the on-policy agent.

        Args:
            env: Environment implementing BoxDiscreteActionEnv.
            num_updates: Number of rollout-update cycles.
            verbose: Print progress (default: False).
            print_every: Print every N updates if verbose (default: 10).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        from mojo_rl.deep_agents.core.training.onpolicy_train import (
            run_onpolicy_discrete_train,
        )

        var cpu_state = self.make_cpu_state()
        var ckpt_path = String(self.checkpoint_path)
        var algo_name = Self.Config.NAME
        var metrics = run_onpolicy_discrete_train(
            self,
            cpu_state,
            env,
            num_updates,
            checkpoint_every=self.checkpoint_every,
            checkpoint_path=ckpt_path,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
        )
        self.cpu_state = cpu_state^
        return metrics

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent on the environment with optional rendering.

        Args:
            env: Environment to evaluate on (must implement RenderableEnv).
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps_per_episode: Maximum steps per episode (default: 500).
            verbose: Print per-episode results (default: False).
            render: If True, render each frame (default: False).
            frame_delay_ms: Delay between frames in ms (default: 16 ~60fps).

        Returns:
            Average reward across episodes.
        """
        # Use a separate cpu_state reference to avoid aliasing self + self.cpu_state
        var eval_state = self.make_cpu_state()
        eval_state.actor.copy_params_from(self.cpu_state.actor)
        eval_state.critic.copy_params_from(self.cpu_state.critic)

        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for episode in range(num_episodes):
            if quit_requested:
                break

            var obs_raw = env.reset_obs_list()
            var obs = List[Float64]()
            for i in range(len(obs_raw)):
                obs.append(Float64(obs_raw[i]))

            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for _ in range(max_steps_per_episode):
                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break
                    if (
                        env.renderer_is_paused()
                        and not env.renderer_step_once()
                    ):
                        continue

                var action_list = self.select_greedy_action(eval_state, obs)
                var action_int = Int(Float64(action_list[0]))
                var result = env.step_obs(action_int)
                var next_obs = List[Float64]()
                for i in range(len(result[0])):
                    next_obs.append(Float64(result[0][i]))
                var reward = Float64(result[1])
                var done = result[2]

                episode_reward += reward
                episode_steps += 1
                obs = next_obs^

                if done:
                    break

            total_reward += episode_reward

            if verbose:
                print(
                    "Eval Episode",
                    episode + 1,
                    "| Reward:",
                    String(episode_reward)[byte=:10],
                    "| Steps:",
                    episode_steps,
                )

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)

    # =========================================================================
    # GPUOnPolicyDiscreteAgent trait conformance
    # =========================================================================

    def make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for this agent."""
        return Self.GPUStateType(ctx)

    def upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights to GPU state."""
        gpu_state.gpu_actor.upload_from(self.cpu_state.actor, ctx)
        gpu_state.gpu_critic.upload_from(self.cpu_state.critic, ctx)

    def download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.gpu_actor.download_to(self.cpu_state.actor, ctx)
        gpu_state.gpu_critic.download_to(self.cpu_state.critic, ctx)
        ctx.synchronize()

    def select_actions_with_meta_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        mut log_probs_buf: DeviceBuffer[dtype],
        mut values_buf: DeviceBuffer[dtype],
        rng_seed: UInt32 = 0,
    ) raises -> None:
        """Forward actor + critic on GPU and sample actions."""
        comptime blocks = (N_ENVS + TPB - 1) // TPB

        var actor_params_t = gpu_state.gpu_actor.params_view()
        var actor_state_t = gpu_state.gpu_actor.model_state_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()
        var critic_state_t = gpu_state.gpu_critic.model_state_view()

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.logits_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var log_probs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())
        var values_t = LayoutTensor[
            dtype,
            Layout.row_major(N_ENVS, Self.CriticModel.OUT_DIM),
            MutAnyOrigin,
        ](values_buf.unsafe_ptr())

        # Actor forward → logits
        Self.ActorModel.forward_gpu_no_cache[N_ENVS](
            ctx,
            logits_t,
            obs_t,
            actor_params_t,
            actor_state_t,
            gpu_state.actor_env_workspace_buf,
        )

        # Critic forward → values (rebind obs for CRITIC_IN dim)
        var c_obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.CRITIC_IN), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        Self.CriticModel.forward_gpu_no_cache[N_ENVS](
            ctx,
            values_t,
            c_obs_t,
            critic_params_t,
            critic_state_t,
            gpu_state.critic_env_workspace_buf,
        )

        # Sample actions from categorical distribution
        comptime sample_wrapper = _generic_sample_actions_kernel[
            dtype, N_ENVS, Self.ACTIONS
        ]
        ctx.enqueue_function[sample_wrapper, sample_wrapper](
            logits_t,
            actions_t,
            log_probs_t,
            Scalar[DType.uint32](rng_seed),
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

    def compute_advantages_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        final_obs_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Compute GAE advantages from the collected rollout (CPU-side)."""
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE

        var critic_params_t = gpu_state.gpu_critic.params_view()
        var critic_state_t = gpu_state.gpu_critic.model_state_view()
        var final_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.CRITIC_IN), MutAnyOrigin
        ](final_obs_buf.unsafe_ptr())
        var bootstrap_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.CriticModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu_state.values_env_buf.unsafe_ptr())

        # Forward critic on final obs to get bootstrap values
        Self.CriticModel.forward_gpu_no_cache[Self.n_envs](
            ctx,
            bootstrap_t,
            final_obs_t,
            critic_params_t,
            critic_state_t,
            gpu_state.critic_env_workspace_buf,
        )

        # Copy rollout data to host for GAE computation
        var rws = gpu_state.rollout_ws()
        ctx.enqueue_copy(
            gpu_state.bootstrap_values_host, gpu_state.values_env_buf
        )
        ctx.enqueue_copy(
            gpu_state.rollout_rewards_host, rws.rewards_subbuf(ctx)
        )
        ctx.enqueue_copy(gpu_state.rollout_values_host, rws.values_subbuf(ctx))
        ctx.enqueue_copy(gpu_state.rollout_dones_host, rws.dones_subbuf(ctx))
        ctx.synchronize()

        # GAE computation per environment
        for env_idx in range(Self.n_envs):
            var gae = Scalar[dtype](0.0)
            var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
            var bootstrap_val = Scalar[dtype](
                gpu_state.bootstrap_values_host[env_idx]
            )

            for t in range(Self.ROLLOUT - 1, -1, -1):
                var idx = t * Self.n_envs + env_idx
                var reward = gpu_state.rollout_rewards_host[idx]
                var value = gpu_state.rollout_values_host[idx]
                var done = gpu_state.rollout_dones_host[idx]

                var next_val: Scalar[dtype]
                if t == Self.ROLLOUT - 1:
                    next_val = bootstrap_val
                else:
                    var next_idx = (t + 1) * Self.n_envs + env_idx
                    next_val = gpu_state.rollout_values_host[next_idx]

                if done > Scalar[dtype](0.5):
                    next_val = Scalar[dtype](0.0)
                    gae = Scalar[dtype](0.0)

                var delta = (
                    reward + Scalar[dtype](self.gamma) * next_val - value
                )
                gae = delta + gae_decay * gae
                gpu_state.advantages_host[idx] = gae
                gpu_state.returns_host[idx] = gae + value

        # Normalize advantages globally if requested
        if self.normalize_advantages:
            var mean = Scalar[dtype](0.0)
            var var_sum = Scalar[dtype](0.0)
            for i in range(ROLLOUT_TOTAL):
                mean += gpu_state.advantages_host[i]
            mean /= Scalar[dtype](ROLLOUT_TOTAL)
            for i in range(ROLLOUT_TOTAL):
                var diff = gpu_state.advantages_host[i] - mean
                var_sum += diff * diff
            var std = sqrt(
                var_sum / Scalar[dtype](ROLLOUT_TOTAL) + Scalar[dtype](1e-8)
            )
            for i in range(ROLLOUT_TOTAL):
                gpu_state.advantages_host[i] = (
                    gpu_state.advantages_host[i] - mean
                ) / (std + Scalar[dtype](1e-8))

        # Copy advantages and returns to GPU
        ctx.enqueue_copy(rws.advantages_subbuf(ctx), gpu_state.advantages_host)
        ctx.enqueue_copy(rws.returns_subbuf(ctx), gpu_state.returns_host)
        ctx.synchronize()

    def update_epochs_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        update_idx: Int,
    ) raises -> None:
        """Run on-policy update epochs on GPU (PPO or A2C via strategy dispatch).
        """
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ACTOR_PARAMS = Self.ActorModel.PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CriticModel.PARAM_SIZE
        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB

        # Typed workspace views (replace 20+ ad-hoc LayoutTensor constructions)
        var actor_params_t = gpu_state.gpu_actor.params_view()
        var actor_state_t = gpu_state.gpu_actor.model_state_view()
        var actor_grads_t = gpu_state.gpu_actor.grads_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()
        var critic_state_t = gpu_state.gpu_critic.model_state_view()
        var critic_grads_t = gpu_state.gpu_critic.grads_view()

        var rws = gpu_state.rollout_ws()
        var mws = gpu_state.minibatch_ws()
        var aws = gpu_state.actor_ws()
        var cws = gpu_state.critic_ws()

        # Minibatch views (use local MINIBATCH comptime for type consistency)
        var mb_obs_t = mws.obs()
        var mb_actions_t = mws.actions()
        var mb_advantages_t = mws.advantages()
        var mb_returns_t = mws.returns()
        var mb_old_log_probs_t = mws.old_log_probs()
        var mb_old_values_t = mws.old_values()
        var mb_indices_t = LayoutTensor[
            DType.int32, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_indices_buf.unsafe_ptr())

        # Rollout views
        var rollout_obs_t = rws.obs()
        var rollout_actions_t = rws.actions()
        var advantages_t = rws.advantages()
        var returns_t = rws.returns()
        var rollout_log_probs_t = rws.log_probs()
        var rollout_values_t = rws.values()

        # Actor forward/backward views
        var actor_logits_t = aws.logits()
        var actor_grad_output_t = aws.grad_output()
        var actor_cache_t = aws.cache()
        var actor_grad_input_t = aws.grad_input()

        # Critic forward/backward views (rebind with local MINIBATCH comptime
        # for type consistency with Self.CriticModel.forward_gpu/backward_gpu)
        comptime C_OUT = Self.CriticModel.OUT_DIM
        comptime C_CS = Self.CriticModel.CACHE_SIZE
        comptime C_IN = Self.CriticModel.IN_DIM
        comptime _C_O_CACHE = MINIBATCH * C_OUT
        comptime _C_O_GO = _C_O_CACHE + MINIBATCH * C_CS
        comptime _C_O_GI = _C_O_GO + MINIBATCH * C_OUT
        var critic_values_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, C_OUT), MutAnyOrigin
        ](cws.ptr)
        var critic_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, C_CS),
            MutAnyOrigin,
        ](cws.ptr + _C_O_CACHE)
        var critic_grad_output_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, C_OUT), MutAnyOrigin
        ](cws.ptr + _C_O_GO)
        var critic_grad_input_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, C_IN),
            MutAnyOrigin,
        ](cws.ptr + _C_O_GI)

        var kl_divergences_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.kl_divergences_buf.unsafe_ptr())
        var diag_entropy_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.diag_entropy_buf.unsafe_ptr())
        var diag_clip_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.diag_clip_buf.unsafe_ptr())
        var actor_grad_partial_sums_t = LayoutTensor[
            dtype, Layout.row_major(ACTOR_GRAD_BLOCKS), MutAnyOrigin
        ](gpu_state.actor_grad_partial_sums_buf.unsafe_ptr())
        var critic_grad_partial_sums_t = LayoutTensor[
            dtype, Layout.row_major(CRITIC_GRAD_BLOCKS), MutAnyOrigin
        ](gpu_state.critic_grad_partial_sums_buf.unsafe_ptr())
        var actor_scale_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.actor_scale_buf.unsafe_ptr())
        var critic_scale_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.critic_scale_buf.unsafe_ptr())

        # Kernel wrappers
        comptime gather_wrapper = ppo_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, ROLLOUT_TOTAL
        ]
        comptime critic_grad_wrapper = ppo_critic_grad_kernel[dtype, MINIBATCH]
        comptime critic_grad_clipped_wrapper = ppo_critic_grad_clipped_kernel[
            dtype, MINIBATCH
        ]
        comptime normalize_advantages_wrapper = normalize_advantages_kernel[
            dtype, MINIBATCH
        ]
        comptime actor_grad_norm_wrapper = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_wrapper = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime actor_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime actor_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, ACTOR_PARAMS
        ]
        comptime critic_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime critic_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, CRITIC_PARAMS
        ]

        var kl_early_stop = False
        var n_epochs = Self.Config.EpochSched.get_num_epochs(self.num_epochs)
        var num_minibatches = ROLLOUT_TOTAL // MINIBATCH

        for epoch in range(n_epochs):
            if kl_early_stop:
                break

            # Fisher-Yates shuffle on CPU (PPO)
            var indices_list = List[Int]()
            for i in range(ROLLOUT_TOTAL):
                indices_list.append(i)
            comptime if Self.Config.EpochSched.USES_SHUFFLE:
                for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                    var j = Int(random_float64() * Float64(i + 1))
                    var temp = indices_list[i]
                    indices_list[i] = indices_list[j]
                    indices_list[j] = temp

            for mb_idx in range(num_minibatches):
                if kl_early_stop:
                    break
                var start_idx = mb_idx * MINIBATCH

                for i in range(MINIBATCH):
                    gpu_state.mb_indices_host[i] = Int32(
                        indices_list[start_idx + i]
                    )
                ctx.enqueue_copy(
                    gpu_state.mb_indices_buf, gpu_state.mb_indices_host
                )

                # Parallel obs gather: 2D grid (OBS_BLOCKS, MINIBATCH)
                comptime gather_obs_wrapper = ppo_gather_minibatch_obs_parallel_kernel[
                    dtype, MINIBATCH, Self.OBS, ROLLOUT_TOTAL
                ]
                comptime GATHER_OBS_BLOCKS = (Self.OBS + TPB - 1) // TPB
                ctx.enqueue_function[gather_obs_wrapper, gather_obs_wrapper](
                    mb_obs_t,
                    rollout_obs_t,
                    mb_indices_t,
                    MINIBATCH,
                    grid_dim=(GATHER_OBS_BLOCKS, MINIBATCH),
                    block_dim=(TPB,),
                )

                # Scalar gather: actions, advantages, returns, log_probs, values
                @parameter
                @always_inline
                def gather_scalars_mb_wrapper(
                    mb_a: LayoutTensor[
                        dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
                    ],
                    mb_adv: LayoutTensor[
                        dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
                    ],
                    mb_ret: LayoutTensor[
                        dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
                    ],
                    mb_olp: LayoutTensor[
                        dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
                    ],
                    mb_ov: LayoutTensor[
                        dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
                    ],
                    r_a: LayoutTensor[
                        dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
                    ],
                    adv: LayoutTensor[
                        dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
                    ],
                    ret: LayoutTensor[
                        dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
                    ],
                    r_lp: LayoutTensor[
                        dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
                    ],
                    r_v: LayoutTensor[
                        dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
                    ],
                    idx: LayoutTensor[
                        DType.int32,
                        Layout.row_major(MINIBATCH),
                        MutAnyOrigin,
                    ],
                    bs: Int,
                ):
                    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                    if i >= bs:
                        return
                    var src = Int(idx[i])
                    mb_a[i] = r_a[src]
                    mb_adv[i] = adv[src]
                    mb_ret[i] = ret[src]
                    mb_olp[i] = r_lp[src]
                    mb_ov[i] = r_v[src]

                ctx.enqueue_function[
                    gather_scalars_mb_wrapper,
                    gather_scalars_mb_wrapper,
                ](
                    mb_actions_t,
                    mb_advantages_t,
                    mb_returns_t,
                    mb_old_log_probs_t,
                    mb_old_values_t,
                    rollout_actions_t,
                    advantages_t,
                    returns_t,
                    rollout_log_probs_t,
                    rollout_values_t,
                    mb_indices_t,
                    MINIBATCH,
                    grid_dim=(MINIBATCH_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                # Per-minibatch advantage normalization (PPO)
                comptime if Self.Config.EpochSched.SUPPORTS_MINIBATCH_NORM:
                    if self.norm_adv_per_minibatch:
                        ctx.enqueue_copy(
                            gpu_state.mb_advantages_host,
                            mws.advantages_subbuf(ctx),
                        )
                        ctx.synchronize()
                        var adv_mean = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            adv_mean += gpu_state.mb_advantages_host[i]
                        adv_mean /= Scalar[dtype](MINIBATCH)
                        var adv_var = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            var diff = (
                                gpu_state.mb_advantages_host[i] - adv_mean
                            )
                            adv_var += diff * diff
                        var adv_std = sqrt(
                            adv_var / Scalar[dtype](MINIBATCH)
                            + Scalar[dtype](1e-8)
                        )
                        ctx.enqueue_function[
                            normalize_advantages_wrapper,
                            normalize_advantages_wrapper,
                        ](
                            mb_advantages_t,
                            adv_mean,
                            adv_std,
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.synchronize()

                # ---- Train actor ----
                gpu_state.gpu_actor.zero_grads(ctx)
                Self.ActorModel.forward_gpu[MINIBATCH](
                    ctx,
                    actor_logits_t,
                    mb_obs_t,
                    actor_params_t,
                    actor_state_t,
                    actor_cache_t,
                    gpu_state.actor_mb_workspace_buf,
                )
                ctx.synchronize()

                # Policy gradient via strategy dispatch
                comptime if Self.Config.USE_AUTODIFF_GRAD:
                    # ---- True autodiff: LossGraph forward + backward ----
                    comptime A = Self.ACTIONS
                    comptime LOSS_IN = A + 3
                    comptime LossGraph = Sequential[
                        SplitApply[
                            CategoricalLogProb[A],
                            Slice[2, 0, 2],
                            A + 1,
                        ],
                        SplitApply[Ratio[1], Slice[1, 0, 1], 2],
                        ClipSurrogate[0.2],
                    ]
                    comptime LOSS_OUT = LossGraph.OUT_DIM  # 1
                    comptime LOSS_CS = LossGraph.CACHE_SIZE
                    comptime LOSS_WS = max(
                        1, MINIBATCH * LossGraph.WORKSPACE_SIZE_PER_SAMPLE
                    )

                    # Slice pre-allocated workspace into sub-buffers
                    comptime LOSS_PS = max(1, LossGraph.PARAM_SIZE)
                    comptime LOSS_CACHE_SZ = max(1, MINIBATCH * LOSS_CS)
                    var ws_ptr = gpu_state.loss_ws_buf.unsafe_ptr()
                    var off = 0

                    var loss_input_ptr = ws_ptr + off
                    off += MINIBATCH * LOSS_IN

                    var loss_output_ptr = ws_ptr + off
                    off += MINIBATCH * LOSS_OUT

                    var loss_cache_ptr = ws_ptr + off
                    off += LOSS_CACHE_SZ

                    var loss_params_ptr = ws_ptr + off
                    off += LOSS_PS

                    var loss_grads_ptr = ws_ptr + off
                    off += LOSS_PS

                    var loss_grad_input_ptr = ws_ptr + off
                    off += MINIBATCH * LOSS_IN

                    var loss_grad_output_ptr = ws_ptr + off
                    off += MINIBATCH * LOSS_OUT

                    var loss_workspace_buf = DeviceBuffer[dtype](
                        ctx, ws_ptr + off, LOSS_WS, owning=False
                    )

                    # Pack input: [logits(A) || action_idx(1) || old_lp(1) || adv(1)]
                    var loss_input_t = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH, LOSS_IN),
                        MutAnyOrigin,
                    ](loss_input_ptr)

                    @parameter
                    @always_inline
                    def pack_discrete_loss_input_k(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH, LOSS_IN),
                            MutAnyOrigin,
                        ],
                        logits_in: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH, A),
                            ImmutAnyOrigin,
                        ],
                        actions_in: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH),
                            ImmutAnyOrigin,
                        ],
                        old_lp_in: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH),
                            ImmutAnyOrigin,
                        ],
                        adv_in: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= MINIBATCH:
                            return
                        # Copy logits
                        for j in range(A):
                            dst.ptr[idx * LOSS_IN + j] = logits_in.ptr[
                                idx * A + j
                            ]
                        # Copy action index, old_log_prob, advantage
                        dst.ptr[idx * LOSS_IN + A] = actions_in.ptr[idx]
                        dst.ptr[idx * LOSS_IN + A + 1] = old_lp_in.ptr[idx]
                        dst.ptr[idx * LOSS_IN + A + 2] = adv_in.ptr[idx]

                    var actor_logits_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH, A),
                        ImmutAnyOrigin,
                    ](actor_logits_t.ptr)
                    var mb_actions_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH),
                        ImmutAnyOrigin,
                    ](mb_actions_t.ptr)
                    var mb_old_lp_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH),
                        ImmutAnyOrigin,
                    ](mb_old_log_probs_t.ptr)
                    var mb_adv_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH),
                        ImmutAnyOrigin,
                    ](mb_advantages_t.ptr)

                    ctx.enqueue_function[
                        pack_discrete_loss_input_k,
                        pack_discrete_loss_input_k,
                    ](
                        loss_input_t,
                        actor_logits_immut,
                        mb_actions_immut,
                        mb_old_lp_immut,
                        mb_adv_immut,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                    # Forward the loss graph
                    var loss_output_t = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH, LOSS_OUT),
                        MutAnyOrigin,
                    ](loss_output_ptr)
                    var loss_cache_t = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH, LOSS_CS),
                        MutAnyOrigin,
                    ](loss_cache_ptr)
                    var loss_params_t = LayoutTensor[
                        dtype,
                        Layout.row_major(LossGraph.PARAM_SIZE),
                        MutAnyOrigin,
                    ](loss_params_ptr)

                    # Zero-length model state slice (LossGraph is stateless)
                    var loss_state_t = LayoutTensor[
                        dtype, Layout.row_major(LossGraph.STATE_SIZE), MutAnyOrigin
                    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))

                    LossGraph.forward_gpu[MINIBATCH](
                        ctx,
                        loss_output_t,
                        loss_input_t,
                        loss_params_t,
                        loss_state_t,
                        loss_cache_t,
                        loss_workspace_buf,
                    )
                    ctx.synchronize()

                    # Seed grad_output = 1 / mb_size per sample
                    var loss_grad_output_t = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH, LOSS_OUT),
                        MutAnyOrigin,
                    ](loss_grad_output_ptr)

                    @parameter
                    @always_inline
                    def seed_discrete_grad_k(
                        go: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH, LOSS_OUT),
                            MutAnyOrigin,
                        ],
                    ):
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= MINIBATCH:
                            return
                        go[idx, 0] = Scalar[dtype](1.0) / Scalar[dtype](
                            MINIBATCH
                        )

                    ctx.enqueue_function[
                        seed_discrete_grad_k, seed_discrete_grad_k
                    ](
                        loss_grad_output_t,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Backward the loss graph
                    var loss_grad_input_t = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH, LOSS_IN),
                        MutAnyOrigin,
                    ](loss_grad_input_ptr)
                    var loss_grads_t = LayoutTensor[
                        dtype,
                        Layout.row_major(LossGraph.PARAM_SIZE),
                        MutAnyOrigin,
                    ](loss_grads_ptr)

                    LossGraph.backward_gpu[MINIBATCH](
                        ctx,
                        loss_grad_input_t,
                        loss_grad_output_t,
                        loss_params_t,
                        loss_state_t,
                        loss_cache_t,
                        loss_grads_t,
                        loss_workspace_buf,
                    )
                    ctx.synchronize()

                    # Extract d_logits[:, :A] and add entropy bonus + compute diagnostics
                    var loss_gi_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(MINIBATCH, LOSS_IN),
                        ImmutAnyOrigin,
                    ](loss_grad_input_ptr)

                    @parameter
                    @always_inline
                    def extract_entropy_diag_k(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH, A),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH, LOSS_IN),
                            ImmutAnyOrigin,
                        ],
                        logits_in: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH, A),
                            ImmutAnyOrigin,
                        ],
                        old_lp: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH),
                            ImmutAnyOrigin,
                        ],
                        actions_in: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH),
                            ImmutAnyOrigin,
                        ],
                        kl_out: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH),
                            MutAnyOrigin,
                        ],
                        ent_out: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH),
                            MutAnyOrigin,
                        ],
                        clip_out: LayoutTensor[
                            dtype,
                            Layout.row_major(MINIBATCH),
                            MutAnyOrigin,
                        ],
                        ent_coef: Scalar[dtype],
                        clip_eps: Scalar[dtype],
                    ):
                        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if b >= MINIBATCH:
                            return

                        # Compute softmax probs for entropy gradient + diagnostics
                        var max_logit = logits_in.ptr[b * A]
                        for j in range(1, A):
                            var lj = logits_in.ptr[b * A + j]
                            if lj > max_logit:
                                max_logit = lj

                        var sum_exp = Scalar[dtype](0.0)
                        for j in range(A):
                            sum_exp += exp(logits_in.ptr[b * A + j] - max_logit)

                        var probs_local = InlineArray[Scalar[dtype], A](
                            fill=Scalar[dtype](0.0)
                        )
                        for j in range(A):
                            probs_local[j] = (
                                exp(logits_in.ptr[b * A + j] - max_logit)
                                / sum_exp
                            )

                        # Extract graph grad + entropy bonus
                        for j in range(A):
                            var d_lp = src.ptr[b * LOSS_IN + j]
                            var prob_for_log = Float32(
                                probs_local[j]
                            ) + Float32(1e-8)
                            var d_ent = -probs_local[j] * (
                                Scalar[dtype](1.0)
                                + Scalar[dtype](log(prob_for_log))
                            )
                            dst.ptr[b * A + j] = (
                                d_lp
                                - ent_coef / Scalar[dtype](MINIBATCH) * d_ent
                            )

                        # Compute entropy
                        var ent = Scalar[dtype](0.0)
                        for j in range(A):
                            if probs_local[j] > Scalar[dtype](1e-10):
                                var p_log = Float32(probs_local[j]) + Float32(
                                    1e-8
                                )
                                ent -= probs_local[j] * Scalar[dtype](
                                    log(p_log)
                                )
                        ent_out[b] = ent

                        # Compute KL divergence and clip flags
                        var action_idx = Int(
                            rebind[Scalar[dtype]](actions_in.ptr[b])
                        )
                        var new_log_prob = Scalar[dtype](
                            log(
                                Float32(probs_local[action_idx]) + Float32(1e-8)
                            )
                        )
                        var old_lp_val = old_lp.ptr[b]
                        var ratio = exp(new_log_prob - old_lp_val)
                        var log_ratio = new_log_prob - old_lp_val
                        var kl = (ratio - Scalar[dtype](1.0)) - log_ratio
                        if kl < Scalar[dtype](0.0):
                            kl = Scalar[dtype](0.0)
                        kl_out[b] = kl

                        var lo = Scalar[dtype](1.0) - clip_eps
                        var hi = Scalar[dtype](1.0) + clip_eps
                        if ratio < lo or ratio > hi:
                            clip_out[b] = Scalar[dtype](1.0)
                        else:
                            clip_out[b] = Scalar[dtype](0.0)

                    ctx.enqueue_function[
                        extract_entropy_diag_k,
                        extract_entropy_diag_k,
                    ](
                        actor_grad_output_t,
                        loss_gi_immut,
                        actor_logits_immut,
                        mb_old_lp_immut,
                        mb_actions_immut,
                        kl_divergences_t,
                        diag_entropy_t,
                        diag_clip_t,
                        Scalar[dtype](self.entropy_coef),
                        Scalar[dtype](self.clip_epsilon),
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                else:
                    Self.Config.PolicyGrad.compute_d_logits_gpu[
                        MINIBATCH, Self.ACTIONS
                    ](
                        ctx,
                        actor_grad_output_t,
                        kl_divergences_t,
                        diag_entropy_t,
                        diag_clip_t,
                        actor_logits_t,
                        mb_old_log_probs_t,
                        mb_advantages_t,
                        mb_actions_t,
                        self.clip_epsilon,
                        self.entropy_coef,
                    )
                    ctx.synchronize()

                # KL early stopping (PPO)
                comptime if Self.Config.EpochSched.USES_KL_EARLY_STOP:
                    if self.target_kl > 0.0:
                        ctx.enqueue_copy(
                            gpu_state.kl_divergences_host,
                            gpu_state.kl_divergences_buf,
                        )
                        ctx.synchronize()
                        var kl_sum = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            kl_sum += gpu_state.kl_divergences_host[i]
                        if (
                            Float64(kl_sum) / Float64(MINIBATCH)
                            > self.target_kl
                        ):
                            kl_early_stop = True
                            break

                Self.ActorModel.backward_gpu[MINIBATCH](
                    ctx,
                    actor_grad_input_t,
                    actor_grad_output_t,
                    actor_params_t,
                    actor_state_t,
                    actor_cache_t,
                    actor_grads_t,
                    gpu_state.actor_mb_workspace_buf,
                )

                if self.max_grad_norm > 0.0:
                    ctx.enqueue_function[
                        actor_grad_norm_wrapper, actor_grad_norm_wrapper
                    ](
                        actor_grad_partial_sums_t,
                        actor_grads_t,
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        actor_reduce_scale_wrapper, actor_reduce_scale_wrapper
                    ](
                        actor_scale_t,
                        actor_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        actor_apply_scale_wrapper, actor_apply_scale_wrapper
                    ](
                        actor_grads_t,
                        actor_scale_t,
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                gpu_state.gpu_actor.optimizer_step(ctx)
                ctx.synchronize()

                # ---- Train critic ---- (rebind obs for CRITIC_IN)
                var mb_c_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(MINIBATCH, Self.CRITIC_IN),
                    MutAnyOrigin,
                ](mws.obs().ptr)
                gpu_state.gpu_critic.zero_grads(ctx)
                Self.CriticModel.forward_gpu[MINIBATCH](
                    ctx,
                    critic_values_t,
                    mb_c_obs_t,
                    critic_params_t,
                    critic_state_t,
                    critic_cache_t,
                    gpu_state.critic_mb_workspace_buf,
                )
                ctx.synchronize()

                if self.clip_value:
                    ctx.enqueue_function[
                        critic_grad_clipped_wrapper,
                        critic_grad_clipped_wrapper,
                    ](
                        critic_grad_output_t,
                        critic_values_t,
                        mb_returns_t,
                        mb_old_values_t,
                        Scalar[dtype](self.clip_epsilon),
                        Scalar[dtype](self.value_loss_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                else:
                    ctx.enqueue_function[
                        critic_grad_wrapper, critic_grad_wrapper
                    ](
                        critic_grad_output_t,
                        critic_values_t,
                        mb_returns_t,
                        Scalar[dtype](self.value_loss_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                ctx.synchronize()

                Self.CriticModel.backward_gpu[MINIBATCH](
                    ctx,
                    critic_grad_input_t,
                    critic_grad_output_t,
                    critic_params_t,
                    critic_state_t,
                    critic_cache_t,
                    critic_grads_t,
                    gpu_state.critic_mb_workspace_buf,
                )

                if self.max_grad_norm > 0.0:
                    ctx.enqueue_function[
                        critic_grad_norm_wrapper, critic_grad_norm_wrapper
                    ](
                        critic_grad_partial_sums_t,
                        critic_grads_t,
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        critic_reduce_scale_wrapper,
                        critic_reduce_scale_wrapper,
                    ](
                        critic_scale_t,
                        critic_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        critic_apply_scale_wrapper, critic_apply_scale_wrapper
                    ](
                        critic_grads_t,
                        critic_scale_t,
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                gpu_state.gpu_critic.optimizer_step(ctx)
                ctx.synchronize()

        # Reset rollout step so next rollout collection starts from position 0
        gpu_state.rollout_step = 0
        self.train_step_count += 1

    # =========================================================================
    # GPU Training convenience method
    # =========================================================================

    def train_gpu[
        EnvType: GPUDiscreteEnv,
        CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
    ](
        mut self,
        ctx: DeviceContext,
        num_updates: Int,
        verbose: Bool = False,
        print_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train on GPU with parallel environments (PPO or A2C via strategy)."""
        var timer = PerfTimer[False]()
        var ckpt_every = self.checkpoint_every
        var ckpt_path = String(self.checkpoint_path)
        var tgt_steps = self.target_total_steps
        return run_onpolicy_discrete_train_gpu[
            EnvType, Self, 0, NoOpLogger, CurriculumType
        ](
            self,
            ctx,
            num_updates,
            timer,
            target_total_steps=tgt_steps,
            checkpoint_every=ckpt_every,
            checkpoint_path=ckpt_path,
            verbose=verbose,
            print_every=print_every,
        )
