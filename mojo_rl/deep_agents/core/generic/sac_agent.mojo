"""Generic SAC agent parameterized by OffPolicyConfig.

Separate from GenericOffPolicyAgent because SAC has fundamentally different:
  - Actor architecture (Parallel[mean, log_std] → 2*ACTIONS output)
  - Action selection (reparameterized sampling vs deterministic + noise)
  - Actor backward pass (through rsample, not direct Q gradient)
  - Alpha auto-tuning (scalar Adam optimizer for entropy coefficient)
  - No target actor network
"""

from std.random import random_float64
from std.math import exp, log, sqrt
from layout import Layout, LayoutTensor
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import (
    Network,
    NetworkState,
    NetworkPair,
    GPUNetworkState,
    GPUNetworkPair,
)
from mojo_rl.nn.initializer import Kaiming, Xavier
from mojo_rl.nn.gpu.random import gaussian_noise
from mojo_rl.nn.model.stochastic_actor import (
    rsample,
    rsample_with_cache,
    rsample_backward,
    get_deterministic_action,
)

from mojo_rl.deep_agents.core import (
    OffPolicyState,
    OffPolicyContinuousAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    run_offpolicy_continuous_train,
    run_offpolicy_continuous_eval,
    run_offpolicy_continuous_train_gpu,
    Checkpointable,
)
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.deep_agents.core.kernels import (
    concat_obs_action_kernel,
    td_mse_grad_kernel,
    actor_grad_from_critic_kernel,
    td_target_min_twin_kernel,
)
from mojo_rl.deep_agents.sac.kernels import (
    sac_rsample_with_cache_kernel,
    sac_rsample_bwd_kernel,
    sac_sample_actions_kernel,
    min_q_dq_kernel,
    add_ci_grads_kernel,
)
from mojo_rl.core import TrainingMetrics, BoxContinuousActionEnv, GPUContinuousEnv

from .offpolicy_config import OffPolicyConfig
from .offpolicy_agent import _concat_obs_act


# =============================================================================
# SAC CPU State
# =============================================================================


struct SACCPUStateGeneric[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
](Movable, OffPolicyState):
    """CPU state for SAC: actor + twin critics (no target actor) + workspace."""

    comptime BUFFER_DTYPE = dtype

    # Networks — no target actor for SAC
    var actor: NetworkState[Self.ActorModel, Self.ActorOpt]
    var critic1: NetworkPair[Self.CriticModel, Self.CriticOpt]
    var critic2: NetworkPair[Self.CriticModel, Self.CriticOpt]

    # Replay buffer
    var buffer: HeapReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

    # Workspace
    var ws: List[Scalar[dtype]]

    fn __init__(out self):
        self.actor = NetworkState[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier[]]()
        self.critic1 = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        self.critic1.initialize[Kaiming[]]()
        self.critic2 = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        self.critic2.initialize[Kaiming[]]()
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        # Workspace — SAC needs extra regions for actor output, log_probs, etc.
        comptime ACT = Self.action_dim
        comptime BS = Self.batch_size
        comptime CI = Self.CriticModel.IN_DIM
        comptime CO = Self.CriticModel.OUT_DIM
        comptime ACS = Self.ActorModel.CACHE_SIZE
        comptime CCS = Self.CriticModel.CACHE_SIZE
        comptime OBS = Self.obs_dim
        comptime AOUT = Self.ActorModel.OUT_DIM  # 2*ACTIONS for SAC

        comptime WS_SIZE = (
            BS * AOUT      # next_out (actor output on next_obs)
            + BS * ACT     # next_act (sampled next actions)
            + BS * CO      # next_log_pi
            + BS * CI      # next_ci
            + BS * CO      # nq1
            + BS * CO      # nq2
            + BS * CO      # targets
            + BS * CI      # ci
            + BS * CO      # q1_out
            + BS * CCS     # q1_cache
            + BS * CO      # q2_out
            + BS * CCS     # q2_cache
            + BS * CO      # q_grad (reused for both critics)
            + BS * CI      # d_ci
            + BS * AOUT    # curr_out (actor output on obs, for actor update)
            + BS * ACS     # actor_cache
            + BS * ACT     # curr_act (sampled actions for actor update)
            + BS * CO      # curr_log_pi
            + BS * CI      # new_ci (obs + curr_act for critic eval)
            + BS * CO      # new_q1
            + BS * CCS     # new_c1_cache
            + BS * ACT     # grad_act (dQ/da from critic backward)
            + BS * AOUT    # actor_grad (grad_mean + scaled grad_log_std)
            + BS * OBS     # d_obs
        )
        self.ws = List[Scalar[dtype]](capacity=WS_SIZE)
        for _ in range(WS_SIZE):
            self.ws.append(Scalar[dtype](0))

    # OffPolicyState trait
    fn store[
        d: DType
    ](
        mut self,
        obs: List[Scalar[d]],
        action: List[Scalar[d]],
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.action_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        for i in range(Self.action_dim):
            act_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(action[i]))
        self.buffer.add(
            obs_arr, act_arr, Scalar[Self.BUFFER_DTYPE](reward), next_arr, done
        )

    fn is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# SACGPUStateGeneric — GPU buffer container for SAC
# =============================================================================


struct SACGPUStateGeneric[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
    max_n_envs: Int,
](GPUOffPolicyState):
    """GPU-resident state for generic SAC training.

    Key SAC differences from TD3:
      - actor is GPUNetworkState (online only, SAC has no target actor)
      - eps_cache buffer to save noise for backward through reparameterization
      - Actor output shape is [BATCH, ACTOR_OUT=2*ACTION_DIM] (mean || log_std)
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime BATCH = Self.batch_size
    comptime MAX_N = Self.max_n_envs
    comptime ACTOR_OUT = Self.ActorModel.OUT_DIM  # SAC: mean || log_std
    comptime CRITIC_IN = Self.CriticModel.IN_DIM
    comptime CRITIC_OUT = Self.CriticModel.OUT_DIM
    comptime ACTOR_CS = Self.ActorModel.CACHE_SIZE
    comptime CRITIC_CS = Self.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.ActorModel, Self.ActorOpt]
    comptime CriticNet = Network[Self.CriticModel, Self.CriticOpt]
    comptime ACTOR_WS = Self.ActorNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS = Self.CriticNet.WORKSPACE_SIZE_PER_SAMPLE

    # GPU networks: actor (online only), twin critics (online + target each)
    var actor: GPUNetworkState[Self.ActorModel, Self.ActorOpt]
    var critic1: GPUNetworkPair[Self.CriticModel, Self.CriticOpt]
    var critic2: GPUNetworkPair[Self.CriticModel, Self.CriticOpt]

    # GPU replay buffer
    var buffer: GPUReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim
    ]

    # Exploration buffers (inference, sized by max_n_envs)
    var raw_act: DeviceBuffer[dtype]  # [max_n_envs * ACTOR_OUT]
    var inf_ws: DeviceBuffer[dtype]  # [max_n_envs * ACTOR_WS]

    # Training scratch — replay sample output
    var s_obs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_act: DeviceBuffer[dtype]  # [batch_size * action_dim]
    var s_rew: DeviceBuffer[dtype]  # [batch_size]
    var s_nobs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_done: DeviceBuffer[dtype]  # [batch_size]
    var s_idx: DeviceBuffer[DType.int32]  # [batch_size]

    # Training scratch — TD target computation (next-state actor sampling)
    var next_actor_out: DeviceBuffer[dtype]  # [batch_size * ACTOR_OUT]
    var next_act: DeviceBuffer[dtype]  # [batch_size * action_dim]
    var next_lp: DeviceBuffer[dtype]  # [batch_size]
    var eps_cache: DeviceBuffer[dtype]  # [batch_size * action_dim]
    var next_ci: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var nq1: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var nq2: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var targets: DeviceBuffer[dtype]  # [batch_size]

    # Training scratch — critic update
    var ci: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var q1_out: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var q1_cache: DeviceBuffer[dtype]  # [batch_size * CRITIC_CS]
    var critic1_ws: DeviceBuffer[dtype]  # [batch_size * CRITIC_WS]
    var q1_grad: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var d_ci1: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var q2_out: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var q2_cache: DeviceBuffer[dtype]  # [batch_size * CRITIC_CS]
    var critic2_ws: DeviceBuffer[dtype]  # [batch_size * CRITIC_WS]
    var q2_grad: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var d_ci2: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]

    # Training scratch — actor update
    var curr_actor_out: DeviceBuffer[dtype]  # [batch_size * ACTOR_OUT]
    var actor_cache: DeviceBuffer[dtype]  # [batch_size * ACTOR_CS]
    var actor_ws: DeviceBuffer[dtype]  # [batch_size * ACTOR_WS]
    var curr_act: DeviceBuffer[dtype]  # [batch_size * action_dim]
    var curr_lp: DeviceBuffer[dtype]  # [batch_size]
    var new_ci: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var new_q1: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var new_c1_cache: DeviceBuffer[dtype]  # [batch_size * CRITIC_CS]
    var dq1: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var dq2: DeviceBuffer[dtype]  # [batch_size * CRITIC_OUT]
    var d_ci: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var d_ci2_pg: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var grad_act: DeviceBuffer[dtype]  # [batch_size * action_dim]
    var actor_grad: DeviceBuffer[dtype]  # [batch_size * ACTOR_OUT]
    var d_obs: DeviceBuffer[dtype]  # [batch_size * obs_dim]

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers. CPU weights are uploaded separately."""
        self.actor = GPUNetworkState[Self.ActorModel, Self.ActorOpt](ctx)
        self.critic1 = GPUNetworkPair[Self.CriticModel, Self.CriticOpt](ctx)
        self.critic2 = GPUNetworkPair[Self.CriticModel, Self.CriticOpt](ctx)
        self.buffer = GPUReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim
        ](ctx)

        comptime BS = Self.BATCH
        comptime MNE = Self.MAX_N

        # Exploration buffers
        self.raw_act = ctx.enqueue_create_buffer[dtype](MNE * Self.ACTOR_OUT)
        self.inf_ws = ctx.enqueue_create_buffer[dtype](
            max(1, MNE * Self.ACTOR_WS)
        )

        # Replay sample output
        self.s_obs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)
        self.s_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.s_rew = ctx.enqueue_create_buffer[dtype](BS)
        self.s_nobs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)
        self.s_done = ctx.enqueue_create_buffer[dtype](BS)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](BS)

        # TD target computation
        self.next_actor_out = ctx.enqueue_create_buffer[dtype](
            BS * Self.ACTOR_OUT
        )
        self.next_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.next_lp = ctx.enqueue_create_buffer[dtype](BS)
        self.eps_cache = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.next_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.nq1 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.nq2 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.targets = ctx.enqueue_create_buffer[dtype](BS)

        # Critic update
        self.ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.q1_out = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.q1_cache = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_CS)
        self.critic1_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.CRITIC_WS)
        )
        self.q1_grad = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.d_ci1 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.q2_out = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.q2_cache = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_CS)
        self.critic2_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.CRITIC_WS)
        )
        self.q2_grad = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.d_ci2 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)

        # Actor update
        self.curr_actor_out = ctx.enqueue_create_buffer[dtype](
            BS * Self.ACTOR_OUT
        )
        self.actor_cache = ctx.enqueue_create_buffer[dtype](BS * Self.ACTOR_CS)
        self.actor_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.ACTOR_WS)
        )
        self.curr_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.curr_lp = ctx.enqueue_create_buffer[dtype](BS)
        self.new_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.new_q1 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.new_c1_cache = ctx.enqueue_create_buffer[dtype](
            BS * Self.CRITIC_CS
        )
        self.dq1 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.dq2 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.d_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.d_ci2_pg = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.grad_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.actor_grad = ctx.enqueue_create_buffer[dtype](
            BS * Self.ACTOR_OUT
        )
        self.d_obs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)

    # GPUOffPolicyState trait
    fn gpu_store[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        rewards_buf: DeviceBuffer[dtype],
        obs_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        self.buffer.store[N_ENVS](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, dones_buf
        )

    fn gpu_buffer_is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# GenericSACAgent[Config: OffPolicyConfig]
# =============================================================================


struct GenericSACAgent[
    Config: OffPolicyConfig,
](OffPolicyContinuousAgent & GPUOffPolicyAgent & Checkpointable):
    """Generic SAC agent. Config must have HAS_TARGET_ACTOR=False, NUM_CRITICS=2."""

    # Dimensions from Config's Model types
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.action_dim
    comptime ACTOR_OUT: Int = Self.Config.ActorModel.OUT_DIM  # 2*ACTIONS
    comptime BATCH: Int = Self.Config.batch_size
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]

    # Workspace offsets (sequential layout)
    comptime _O_NEXT_OUT: Int = 0
    comptime _O_NEXT_ACT: Int = Self._O_NEXT_OUT + Self.BATCH * Self.ACTOR_OUT
    comptime _O_NEXT_LP: Int = Self._O_NEXT_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_NEXT_CI: Int = Self._O_NEXT_LP + Self.BATCH * Self.CRITIC_OUT
    comptime _O_NQ1: Int = Self._O_NEXT_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_NQ2: Int = Self._O_NQ1 + Self.BATCH * Self.CRITIC_OUT
    comptime _O_TARGETS: Int = Self._O_NQ2 + Self.BATCH * Self.CRITIC_OUT
    comptime _O_CI: Int = Self._O_TARGETS + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q1_OUT: Int = Self._O_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_Q1_CACHE: Int = Self._O_Q1_OUT + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q2_OUT: Int = Self._O_Q1_CACHE + Self.BATCH * Self.CRITIC_CS
    comptime _O_Q2_CACHE: Int = Self._O_Q2_OUT + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q_GRAD: Int = Self._O_Q2_CACHE + Self.BATCH * Self.CRITIC_CS
    comptime _O_D_CI: Int = Self._O_Q_GRAD + Self.BATCH * Self.CRITIC_OUT
    comptime _O_CURR_OUT: Int = Self._O_D_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_ACTOR_CACHE: Int = Self._O_CURR_OUT + Self.BATCH * Self.ACTOR_OUT
    comptime _O_CURR_ACT: Int = Self._O_ACTOR_CACHE + Self.BATCH * Self.ACTOR_CS
    comptime _O_CURR_LP: Int = Self._O_CURR_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_NEW_CI: Int = Self._O_CURR_LP + Self.BATCH * Self.CRITIC_OUT
    comptime _O_NEW_Q1: Int = Self._O_NEW_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_NEW_C1_CACHE: Int = Self._O_NEW_Q1 + Self.BATCH * Self.CRITIC_OUT
    comptime _O_GRAD_ACT: Int = Self._O_NEW_C1_CACHE + Self.BATCH * Self.CRITIC_CS
    comptime _O_ACTOR_GRAD: Int = Self._O_GRAD_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_D_OBS: Int = Self._O_ACTOR_GRAD + Self.BATCH * Self.ACTOR_OUT

    # CPU state type — SAC-specific (no target actor, twin critics)
    comptime CPUStateType = SACCPUStateGeneric[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.buffer_capacity,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.action_dim,
        Self.Config.batch_size,
    ]

    # GPUOffPolicyAgent required comptime constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACTIONS
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = 64
    comptime GPUStateType = SACGPUStateGeneric[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.buffer_capacity,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.action_dim,
        Self.Config.batch_size,
        64,  # max_n_envs
    ]

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64

    # Entropy tuning
    var alpha: Float64
    var log_alpha: Float64
    var target_entropy: Float64
    var alpha_lr: Float64
    var auto_alpha: Bool
    var alpha_adam_m: Float64
    var alpha_adam_v: Float64
    var alpha_adam_t: Int

    # Policy delay
    var policy_delay: Int

    # Training state
    var total_steps: Int
    var train_step_count: Int
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        alpha: Float64 = 0.2,
        auto_alpha: Bool = True,
        alpha_lr: Float64 = 0.0003,
        target_entropy: Float64 = -1.0,
        policy_delay: Int = 1,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.alpha = alpha
        self.log_alpha = log(alpha)
        self.target_entropy = target_entropy
        self.alpha_lr = alpha_lr
        self.auto_alpha = auto_alpha
        self.alpha_adam_m = 0.0
        self.alpha_adam_v = 0.0
        self.alpha_adam_t = 0
        self.policy_delay = policy_delay
        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    # =========================================================================
    # Helper: extract mean + log_std from actor output, apply affine rescale
    # =========================================================================

    @always_inline
    fn _extract_mean_logstd(
        self,
        actor_out_p: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        mut mean_arr: InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS],
        mut ls_arr: InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS],
    ):
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                var m = Float64(actor_out_p[b * Self.ACTOR_OUT + a])
                var raw_ls = Float64(
                    actor_out_p[b * Self.ACTOR_OUT + Self.ACTIONS + a]
                )
                if m != m:
                    m = 0.0
                elif m > 10.0:
                    m = 10.0
                elif m < -10.0:
                    m = -10.0
                if raw_ls != raw_ls:
                    raw_ls = 0.0
                # Affine rescale: tanh already applied by LinearTanh head
                var ls = -5.0 + 0.5 * 7.0 * (raw_ls + 1.0)
                mean_arr[b * Self.ACTIONS + a] = Scalar[dtype](m)
                ls_arr[b * Self.ACTIONS + a] = Scalar[dtype](ls)

    # =========================================================================
    # OffPolicyContinuousAgent trait
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn select_action[
        d: DType
    ](
        mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[d]]
    ) -> List[Scalar[d]]:
        """SAC: sample from stochastic policy (reparameterization trick)."""
        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var out_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
        ](out_arr.unsafe_ptr())

        var p = cpu_state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, out_t, p)

        # Extract mean + log_std
        var mean_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var ls_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for a in range(Self.ACTIONS):
            var m = Float64(out_arr[a])
            var raw_ls = Float64(out_arr[Self.ACTIONS + a])
            if m != m:
                m = 0.0
            if raw_ls != raw_ls:
                raw_ls = 0.0
            var ls = -5.0 + 0.5 * 7.0 * (raw_ls + 1.0)
            mean_arr[a] = Scalar[dtype](m)
            ls_arr[a] = Scalar[dtype](ls)

        # Sample action
        var noise_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            noise_arr[i] = Scalar[dtype](gaussian_noise())

        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var lp_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var mean_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](mean_arr.unsafe_ptr())
        var ls_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](ls_arr.unsafe_ptr())
        var noise_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](noise_arr.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())
        var lp_t = LayoutTensor[
            dtype, Layout.row_major(1, 1), MutAnyOrigin
        ](lp_arr.unsafe_ptr())
        rsample[1, Self.ACTIONS](mean_t, ls_t, noise_t, act_t, lp_t)

        var result = List[Scalar[d]](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            var a = Float64(act_arr[i]) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            result.append(Scalar[d](a))
        return result^

    fn store_transition[
        d: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[d]],
        action: List[Scalar[d]],
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        var normalized = List[Scalar[d]](capacity=len(action))
        for i in range(len(action)):
            normalized.append(
                Scalar[d](Float64(action[i]) / self.action_scale)
            )
        cpu_state.store[d](obs, normalized, reward, next_obs, done)
        self.total_steps += 1

    fn do_cpu_train_step(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        # Phase 1: Sample batch
        var b_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_act = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var b_rew = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var b_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_done = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        cpu_state.buffer.sample[Self.BATCH](
            b_obs, b_act, b_rew, b_next, b_done
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_next.unsafe_ptr())

        var ws = cpu_state.ws.unsafe_ptr()
        var p_actor = cpu_state.actor.params_view()

        # Phase 2: TD targets using current actor (no target actor!)
        # Forward actor on next_obs → mean + log_std
        var next_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](ws + Self._O_NEXT_OUT)
        Self.ActorNet.forward[Self.BATCH](next_obs_t, next_out_t, p_actor)

        # Extract mean + log_std
        var next_mean = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var next_ls = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        self._extract_mean_logstd(ws + Self._O_NEXT_OUT, next_mean, next_ls)

        # Sample next actions + log_probs
        var next_noise = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.BATCH * Self.ACTIONS):
            next_noise[i] = Scalar[dtype](gaussian_noise())

        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](ws + Self._O_NEXT_ACT)
        # log_prob uses literal 1 (rsample hardcodes it) — use local InlineArray
        var next_lp_arr = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var next_lp_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](next_lp_arr.unsafe_ptr())
        rsample[Self.BATCH, Self.ACTIONS](
            LayoutTensor[dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin](next_mean.unsafe_ptr()),
            LayoutTensor[dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin](next_ls.unsafe_ptr()),
            LayoutTensor[dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin](next_noise.unsafe_ptr()),
            next_act_t,
            next_lp_t,
        )

        # Guard NaN in log_probs
        for b in range(Self.BATCH):
            var lp = Float64(next_lp_arr[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                next_lp_arr[b] = Scalar[dtype](-1.0)

        # Concat next_obs + next_act → next_ci
        _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
            ws + Self._O_NEXT_CI, b_next.unsafe_ptr(), ws + Self._O_NEXT_ACT
        )
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_NEXT_CI)

        # Forward both target critics
        var nq1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_NQ1)
        var nq2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_NQ2)
        var p_c1t = cpu_state.critic1.target.params_view()
        var p_c2t = cpu_state.critic2.target.params_view()
        Self.CriticNet.forward[Self.BATCH](next_ci_t, nq1_t, p_c1t)
        Self.CriticNet.forward[Self.BATCH](next_ci_t, nq2_t, p_c2t)

        # TD targets: r + γ * (min(Q1,Q2) - α * log_π) * (1 - done)
        var tgt_p = ws + Self._O_TARGETS
        var nq1_p = ws + Self._O_NQ1
        var nq2_p = ws + Self._O_NQ2
        for b in range(Self.BATCH):
            var q1 = Float64(nq1_p[b])
            var q2 = Float64(nq2_p[b])
            if q1 != q1:
                q1 = 0.0
            if q2 != q2:
                q2 = 0.0
            var min_q = q1 if q1 < q2 else q2
            var lp = Float64(next_lp_arr[b])
            var dm = 1.0 - Float64(b_done[b])
            var tgt = Float64(b_rew[b]) + self.gamma * (min_q - self.alpha * lp) * dm
            if tgt != tgt:
                tgt = 0.0
            elif tgt > 1000.0:
                tgt = 1000.0
            elif tgt < -1000.0:
                tgt = -1000.0
            tgt_p[b] = Scalar[dtype](tgt)

        # Phase 3: Update both critics
        _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
            ws + Self._O_CI, b_obs.unsafe_ptr(), b_act.unsafe_ptr()
        )
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_CI)
        var qg_p = ws + Self._O_Q_GRAD

        # --- Critic 1 ---
        var q1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_Q1_OUT)
        var c1_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
        ](ws + Self._O_Q1_CACHE)
        var p_c1 = cpu_state.critic1.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](ci_t, q1_t, p_c1, c1_cache_t)

        var q1_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](qg_p)
        var q1o_p = ws + Self._O_Q1_OUT
        var c1_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = q1o_p[b] - tgt_p[b]
            c1_loss += Float64(td_err * td_err)
            qg_p[b] = Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
        c1_loss /= Float64(Self.BATCH)

        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_D_CI)
        var g_c1 = cpu_state.critic1.grads_view()
        cpu_state.critic1.zero_grads()
        Self.CriticNet.backward[Self.BATCH](q1_grad_t, d_ci_t, p_c1, c1_cache_t, g_c1)
        cpu_state.critic1.optimizer_step()

        # --- Critic 2 ---
        var q2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_Q2_OUT)
        var c2_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
        ](ws + Self._O_Q2_CACHE)
        var p_c2 = cpu_state.critic2.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](ci_t, q2_t, p_c2, c2_cache_t)

        var q2o_p = ws + Self._O_Q2_OUT
        var c2_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = q2o_p[b] - tgt_p[b]
            c2_loss += Float64(td_err * td_err)
            qg_p[b] = Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
        c2_loss /= Float64(Self.BATCH)

        var q2_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](qg_p)
        var g_c2 = cpu_state.critic2.grads_view()
        cpu_state.critic2.zero_grads()
        Self.CriticNet.backward[Self.BATCH](q2_grad_t, d_ci_t, p_c2, c2_cache_t, g_c2)
        cpu_state.critic2.optimizer_step()

        var avg_critic_loss = (c1_loss + c2_loss) / 2.0

        # Phase 4: Actor update (every policy_delay steps)
        # Forward actor with cache on current obs
        var curr_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](ws + Self._O_CURR_OUT)
        var actor_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_CS), MutAnyOrigin
        ](ws + Self._O_ACTOR_CACHE)
        Self.ActorNet.forward_with_cache[Self.BATCH](
            obs_t, curr_out_t, p_actor, actor_cache_t
        )

        # Extract mean + log_std for current obs
        var curr_mean = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var curr_ls = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        self._extract_mean_logstd(ws + Self._O_CURR_OUT, curr_mean, curr_ls)

        # Sample current actions + log_probs with cache
        var curr_noise = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.BATCH * Self.ACTIONS):
            curr_noise[i] = Scalar[dtype](gaussian_noise())

        var curr_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](ws + Self._O_CURR_ACT)
        # log_prob: local InlineArray (rsample expects Layout(BATCH, 1))
        var curr_lp_arr = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var curr_lp_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](curr_lp_arr.unsafe_ptr())
        var z_cache = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )

        var z_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](z_cache.unsafe_ptr())
        rsample_with_cache[Self.BATCH, Self.ACTIONS](
            LayoutTensor[dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin](curr_mean.unsafe_ptr()),
            LayoutTensor[dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin](curr_ls.unsafe_ptr()),
            LayoutTensor[dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin](curr_noise.unsafe_ptr()),
            curr_act_t,
            curr_lp_t,
            z_cache_t,
        )

        # Guard NaN in log_probs
        for b in range(Self.BATCH):
            var lp = Float64(curr_lp_arr[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                curr_lp_arr[b] = Scalar[dtype](-1.0)

        if self.train_step_count % self.policy_delay == 0:
            # Concat obs + curr_act → new_ci
            _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
                ws + Self._O_NEW_CI, b_obs.unsafe_ptr(), ws + Self._O_CURR_ACT
            )
            var new_ci_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
            ](ws + Self._O_NEW_CI)

            # Forward critic1 with cache for policy gradient
            var new_q1_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
            ](ws + Self._O_NEW_Q1)
            var new_c1_cache_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
            ](ws + Self._O_NEW_C1_CACHE)
            Self.CriticNet.forward_with_cache[Self.BATCH](
                new_ci_t, new_q1_t, p_c1, new_c1_cache_t
            )

            # Backward critic1 → dQ/da
            for b in range(Self.BATCH):
                qg_p[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))
            var dq_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
            ](qg_p)
            cpu_state.critic1.zero_grads()
            Self.CriticNet.backward[Self.BATCH](
                dq_t, d_ci_t, p_c1, new_c1_cache_t, g_c1
            )

            # Extract action gradients
            var ga_p = ws + Self._O_GRAD_ACT
            var dci_p = ws + Self._O_D_CI
            for b in range(Self.BATCH):
                for a in range(Self.ACTIONS):
                    ga_p[b * Self.ACTIONS + a] = dci_p[
                        b * Self.CRITIC_IN + Self.OBS + a
                    ]

            # Entropy gradient
            var grad_lp = InlineArray[Scalar[dtype], Self.BATCH](
                uninitialized=True
            )
            for b in range(Self.BATCH):
                grad_lp[b] = Scalar[dtype](self.alpha / Float64(Self.BATCH))

            # Backward through reparameterization
            var grad_mean = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
                uninitialized=True
            )
            var grad_ls = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
                uninitialized=True
            )

            var ga_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](ga_p)
            var glp_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](grad_lp.unsafe_ptr())
            var cls_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](curr_ls.unsafe_ptr())
            var cn_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](curr_noise.unsafe_ptr())
            var gm_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](grad_mean.unsafe_ptr())
            var gls_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](grad_ls.unsafe_ptr())
            rsample_backward[Self.BATCH, Self.ACTIONS](
                ga_t, glp_t, curr_act_t, cls_t, cn_t, gm_t, gls_t,
            )

            # Build actor_grad = concat(grad_mean, scaled_grad_log_std)
            comptime AFFINE_SCALE = Scalar[dtype](0.5 * 7.0)
            var ag_p = ws + Self._O_ACTOR_GRAD
            for b in range(Self.BATCH):
                for a in range(Self.ACTIONS):
                    ag_p[b * Self.ACTOR_OUT + a] = grad_mean[b * Self.ACTIONS + a]
                    ag_p[b * Self.ACTOR_OUT + Self.ACTIONS + a] = (
                        grad_ls[b * Self.ACTIONS + a] * AFFINE_SCALE
                    )

            # Backward through actor
            var actor_grad_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
            ](ag_p)
            var d_obs_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
            ](ws + Self._O_D_OBS)
            var g_actor = cpu_state.actor.grads_view()
            cpu_state.actor.zero_grads()
            Self.ActorNet.backward[Self.BATCH](
                actor_grad_t, d_obs_t, p_actor, actor_cache_t, g_actor
            )
            cpu_state.actor.optimizer_step()

            # Phase 5: Alpha update
            if self.auto_alpha:
                var mean_lp: Float64 = 0.0
                for b in range(Self.BATCH):
                    mean_lp += Float64(curr_lp_arr[b])
                mean_lp /= Float64(Self.BATCH)

                var grad = -self.alpha * (mean_lp + self.target_entropy)
                self.alpha_adam_t += 1
                var beta1: Float64 = 0.9
                var beta2: Float64 = 0.999
                var eps: Float64 = 1e-8
                self.alpha_adam_m = beta1 * self.alpha_adam_m + (1.0 - beta1) * grad
                self.alpha_adam_v = beta2 * self.alpha_adam_v + (1.0 - beta2) * grad * grad
                var m_hat = self.alpha_adam_m / (1.0 - beta1 ** Float64(self.alpha_adam_t))
                var v_hat = self.alpha_adam_v / (1.0 - beta2 ** Float64(self.alpha_adam_t))
                self.log_alpha -= self.alpha_lr * m_hat / (sqrt(v_hat) + eps)
                self.alpha = exp(self.log_alpha)

        # Phase 6: Soft update critic targets (every step)
        cpu_state.critic1.soft_update(self.tau)
        cpu_state.critic2.soft_update(self.tau)

        self.train_step_count += 1
        return avg_critic_loss

    fn decay_explore(mut self) -> None:
        pass  # SAC uses entropy, not explicit noise

    fn get_explore_rate(self) -> Float64:
        return self.alpha

    fn random_action[d: DType](self) -> List[Scalar[d]]:
        var result = List[Scalar[d]](capacity=Self.ACTIONS)
        for _ in range(Self.ACTIONS):
            result.append(
                Scalar[d]((random_float64() * 2.0 - 1.0) * self.action_scale)
            )
        return result^

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        """Deterministic action: tanh(mean), no sampling."""
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var out_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
        ](out_arr.unsafe_ptr())
        var p = cpu_state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, out_t, p)

        # Extract mean and apply tanh
        var mean_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var mean_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](mean_arr.unsafe_ptr())
        for i in range(Self.ACTIONS):
            mean_arr[i] = out_arr[i]
        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())
        get_deterministic_action[1, Self.ACTIONS](mean_t, act_t)

        var result = List[Float64](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            var a = Float64(act_arr[i]) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            result.append(a)
        return result^

    # Checkpointable
    fn save_checkpoint(self, path: String) raises -> None:
        pass

    fn load_checkpoint(mut self, path: String) raises -> None:
        pass

    # =========================================================================
    # GPUOffPolicyAgent trait
    # =========================================================================

    fn get_action_scale(self) -> Float64:
        return self.action_scale

    fn get_total_steps(self) -> Int:
        return self.total_steps

    fn set_total_steps(mut self, steps: Int):
        self.total_steps = steps

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        # Create fresh CPU state with initialized weights, upload to GPU
        var cpu = self.make_cpu_state()
        gpu_state.actor.upload_from(cpu.actor, ctx)
        gpu_state.critic1.upload_from(cpu.critic1, ctx)
        gpu_state.critic2.upload_from(cpu.critic2, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        pass

    fn select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward SAC actor on GPU + reparameterized sampling."""
        comptime BLOCKS = (N_ENVS + TPB - 1) // TPB

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.raw_act.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        var p = gpu_state.actor.params_view()
        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.inf_ws
        )

        var scale_s = Scalar[dtype](self.action_scale)
        var log_std_min_s = Scalar[dtype](-5.0)
        var log_std_max_s = Scalar[dtype](2.0)
        var rng_seed_s = Scalar[DType.uint32](
            UInt32(self.total_steps) * UInt32(Self.ACTIONS)
        )

        @always_inline
        fn sample_actions(
            acts: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS, Self.ACTOR_OUT),
                MutAnyOrigin,
            ],
            sc: Scalar[dtype],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            rng_seed: Scalar[DType.uint32],
        ):
            sac_sample_actions_kernel[dtype, N_ENVS, Self.ACTIONS](
                acts, ao, sc, lsmin, lsmax, rng_seed
            )

        ctx.enqueue_function[sample_actions, sample_actions](
            act_t,
            raw_t,
            scale_s,
            log_std_min_s,
            log_std_max_s,
            rng_seed_s,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Full SAC GPU training step: sample -> TD targets -> critics -> actor -> alpha."""
        comptime BS = Self.BATCH
        comptime ELEM_BLOCKS = (BS * Self.CRITIC_IN + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (BS + TPB - 1) // TPB
        comptime ACT_BLOCKS = (BS * Self.ACTIONS + TPB - 1) // TPB

        self.train_step_count += 1

        var log_std_min_s = Scalar[dtype](-5.0)
        var log_std_max_s = Scalar[dtype](2.0)
        var seed_stride = UInt32(BS * Self.ACTIONS + 1)
        var next_rng_seed_s = Scalar[DType.uint32](
            UInt32(self.train_step_count) * seed_stride * 2
        )
        var curr_rng_seed_s = Scalar[DType.uint32](
            UInt32(self.train_step_count) * seed_stride * 2 + seed_stride
        )

        # ----- Phase 1: Sample batch -----
        gpu_state.buffer.sample[BS](
            ctx,
            rng_seed=UInt32(self.train_step_count) * UInt32(BS + 1),
            sampled_obs=gpu_state.s_obs,
            sampled_actions=gpu_state.s_act,
            sampled_rewards=gpu_state.s_rew,
            sampled_next_obs=gpu_state.s_nobs,
            sampled_dones=gpu_state.s_done,
            indices=gpu_state.s_idx,
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var nobs_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())
        var rew_t = LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            gpu_state.s_rew.unsafe_ptr()
        )
        var done_t = LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            gpu_state.s_done.unsafe_ptr()
        )

        var p_actor = gpu_state.actor.params_view()
        var p_c1 = gpu_state.critic1.online.params_view()
        var p_c2 = gpu_state.critic2.online.params_view()
        var p_c1t = gpu_state.critic1.target.params_view()
        var p_c2t = gpu_state.critic2.target.params_view()

        # ----- Phase 2: Actor forward on next_obs (current actor, no target) -----
        var next_actor_out_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.next_actor_out.unsafe_ptr())
        Self.ActorNet.forward_gpu[BS](
            ctx, nobs_t, next_actor_out_t, p_actor, gpu_state.actor_ws
        )

        # ----- Phase 3: sac_rsample next actions + log_probs -----
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.next_act.unsafe_ptr())
        var next_lp_t = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.next_lp.unsafe_ptr())
        var eps_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.eps_cache.unsafe_ptr())

        @always_inline
        fn next_rsample(
            acts: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            lp: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            eps: LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype,
                Layout.row_major(BS, Self.ACTOR_OUT),
                MutAnyOrigin,
            ],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            rng_seed: Scalar[DType.uint32],
        ):
            sac_rsample_with_cache_kernel[dtype, BS, Self.ACTIONS, Self.ACTOR_OUT](
                acts, lp, eps, ao, lsmin, lsmax, rng_seed
            )

        ctx.enqueue_function[next_rsample, next_rsample](
            next_act_t,
            next_lp_t,
            eps_cache_t,
            next_actor_out_t,
            log_std_min_s,
            log_std_max_s,
            next_rng_seed_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ----- Phase 4: Concat(next_obs, next_act) → both critic targets → SAC TD targets -----
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.next_ci.unsafe_ptr())

        @always_inline
        fn concat_next_k(
            d: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin],
            o: LayoutTensor[dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin],
            a: LayoutTensor[dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin],
        ):
            concat_obs_action_kernel[dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](d, o, a)

        ctx.enqueue_function[concat_next_k, concat_next_k](
            next_ci_t, nobs_t, next_act_t,
            grid_dim=(ELEM_BLOCKS,), block_dim=(TPB,),
        )

        var nq1_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.nq1.unsafe_ptr())
        var nq2_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.nq2.unsafe_ptr())

        Self.CriticNet.forward_gpu[BS](
            ctx, next_ci_t, nq1_t, p_c1t, gpu_state.critic1_ws
        )
        Self.CriticNet.forward_gpu[BS](
            ctx, next_ci_t, nq2_t, p_c2t, gpu_state.critic2_ws
        )

        # SAC TD targets with entropy: r + γ*(min(Q1,Q2) - α*log_π)*(1-done)
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())
        var gamma_s = Scalar[dtype](self.gamma)
        var alpha_s = Scalar[dtype](self.alpha)
        var nq1_flat = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.nq1.unsafe_ptr())
        var nq2_flat = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.nq2.unsafe_ptr())

        @always_inline
        fn sac_targets(
            tgt: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            r: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            q1: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            q2: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            d: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            lp: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            g: Scalar[dtype],
            a: Scalar[dtype],
        ):
            td_target_min_twin_kernel[dtype, BS, True](tgt, r, q1, q2, d, lp, g, a)

        ctx.enqueue_function[sac_targets, sac_targets](
            targets_t, rew_t, nq1_flat, nq2_flat, done_t, next_lp_t,
            gamma_s, alpha_s,
            grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
        )

        # ----- Phase 5: Critic1 forward + MSE grad + backward + optim -----
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.ci.unsafe_ptr())

        @always_inline
        fn concat_ci_k(
            d: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin],
            o: LayoutTensor[dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin],
            a: LayoutTensor[dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin],
        ):
            concat_obs_action_kernel[dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](d, o, a)

        ctx.enqueue_function[concat_ci_k, concat_ci_k](
            ci_t, obs_t, act_t,
            grid_dim=(ELEM_BLOCKS,), block_dim=(TPB,),
        )

        var q1_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q1_out.unsafe_ptr())
        var q1_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
        ](gpu_state.q1_cache.unsafe_ptr())
        var q1_grad_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q1_grad.unsafe_ptr())
        var d_ci1_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci1.unsafe_ptr())

        Self.CriticNet.forward_gpu_with_cache[BS](
            ctx, ci_t, q1_t, p_c1, q1_cache_t, gpu_state.critic1_ws
        )

        @always_inline
        fn mse_grad1(
            qg: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin],
            q: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
        ):
            td_mse_grad_kernel[dtype, BS, Self.CRITIC_OUT](qg, q, tgt)

        ctx.enqueue_function[mse_grad1, mse_grad1](
            q1_grad_t, q1_t, targets_t,
            grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
        )

        var g_c1 = gpu_state.critic1.online.grads_view()
        gpu_state.critic1.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BS](
            ctx, q1_grad_t, d_ci1_t, p_c1, q1_cache_t,
            g_c1, gpu_state.critic1_ws
        )
        gpu_state.critic1.online.optimizer_step(ctx)

        # ----- Phase 6: Critic2 forward + MSE grad + backward + optim -----
        var q2_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q2_out.unsafe_ptr())
        var q2_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
        ](gpu_state.q2_cache.unsafe_ptr())
        var q2_grad_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q2_grad.unsafe_ptr())
        var d_ci2_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci2.unsafe_ptr())

        Self.CriticNet.forward_gpu_with_cache[BS](
            ctx, ci_t, q2_t, p_c2, q2_cache_t, gpu_state.critic2_ws
        )

        @always_inline
        fn mse_grad2(
            qg: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin],
            q: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
        ):
            td_mse_grad_kernel[dtype, BS, Self.CRITIC_OUT](qg, q, tgt)

        ctx.enqueue_function[mse_grad2, mse_grad2](
            q2_grad_t, q2_t, targets_t,
            grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
        )

        var g_c2 = gpu_state.critic2.online.grads_view()
        gpu_state.critic2.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BS](
            ctx, q2_grad_t, d_ci2_t, p_c2, q2_cache_t,
            g_c2, gpu_state.critic2_ws
        )
        gpu_state.critic2.online.optimizer_step(ctx)

        # ----- Phase 7: Delayed actor update -----
        if self.train_step_count % self.policy_delay == 0:
            # 7a: Actor forward with cache on sampled obs
            var actor_out_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
            ](gpu_state.curr_actor_out.unsafe_ptr())
            var actor_cache_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTOR_CS), MutAnyOrigin
            ](gpu_state.actor_cache.unsafe_ptr())

            Self.ActorNet.forward_gpu_with_cache[BS](
                ctx, obs_t, actor_out_t, p_actor, actor_cache_t,
                gpu_state.actor_ws
            )

            # 7b: sac_rsample with cache → curr_act, curr_lp, eps_cache
            var curr_act_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ](gpu_state.curr_act.unsafe_ptr())
            var curr_lp_t = LayoutTensor[
                dtype, Layout.row_major(BS), MutAnyOrigin
            ](gpu_state.curr_lp.unsafe_ptr())

            @always_inline
            fn curr_rsample(
                acts: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
                ],
                lp: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                eps: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
                ],
                ao: LayoutTensor[
                    dtype,
                    Layout.row_major(BS, Self.ACTOR_OUT),
                    MutAnyOrigin,
                ],
                lsmin: Scalar[dtype],
                lsmax: Scalar[dtype],
                rng_seed: Scalar[DType.uint32],
            ):
                sac_rsample_with_cache_kernel[dtype, BS, Self.ACTIONS, Self.ACTOR_OUT](
                    acts, lp, eps, ao, lsmin, lsmax, rng_seed
                )

            ctx.enqueue_function[curr_rsample, curr_rsample](
                curr_act_t,
                curr_lp_t,
                eps_cache_t,
                actor_out_t,
                log_std_min_s,
                log_std_max_s,
                curr_rng_seed_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # 7c: Concat(obs, curr_act) → new_ci
            var new_ci_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
            ](gpu_state.new_ci.unsafe_ptr())

            @always_inline
            fn concat_new_ci(
                d: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
                ],
                o: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
                ],
                a: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
                ],
            ):
                concat_obs_action_kernel[dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](d, o, a)

            ctx.enqueue_function[concat_new_ci, concat_new_ci](
                new_ci_t, obs_t, curr_act_t,
                grid_dim=(ELEM_BLOCKS,), block_dim=(TPB,),
            )

            # 7d: Both critics forward with cache for min(Q1, Q2) policy gradient
            var new_q1_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
            ](gpu_state.new_q1.unsafe_ptr())
            var new_c1_cache_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
            ](gpu_state.new_c1_cache.unsafe_ptr())

            Self.CriticNet.forward_gpu_with_cache[BS](
                ctx, new_ci_t, new_q1_t, p_c1, new_c1_cache_t,
                gpu_state.critic1_ws
            )

            # Q2 forward on policy actions (reuse q2_out/q2_cache — Phase 6 is done)
            var new_q2_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
            ](gpu_state.q2_out.unsafe_ptr())
            var new_q2_cache_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
            ](gpu_state.q2_cache.unsafe_ptr())

            Self.CriticNet.forward_gpu_with_cache[BS](
                ctx, new_ci_t, new_q2_t, p_c2, new_q2_cache_t,
                gpu_state.critic2_ws
            )

            # 7d2: min(Q1, Q2) mask → dq1 and dq2
            var dq1_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
            ](gpu_state.dq1.unsafe_ptr())
            var dq2_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
            ](gpu_state.dq2.unsafe_ptr())

            @always_inline
            fn min_q_mask(
                dq1: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
                ],
                dq2: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
                ],
                q1: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
                ],
                q2: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
                ],
            ):
                min_q_dq_kernel[dtype, BS, Self.CRITIC_OUT](dq1, dq2, q1, q2)

            ctx.enqueue_function[min_q_mask, min_q_mask](
                dq1_t, dq2_t, new_q1_t, new_q2_t,
                grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
            )

            # 7e: Backward Q1 with masked dq1 → d_ci
            var d_ci_pg_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
            ](gpu_state.d_ci.unsafe_ptr())

            var g_c1_pg = gpu_state.critic1.online.grads_view()
            gpu_state.critic1.online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BS](
                ctx, dq1_t, d_ci_pg_t, p_c1, new_c1_cache_t,
                g_c1_pg, gpu_state.critic1_ws
            )
            # Intentionally NO optimizer_step here

            # 7e2: Backward Q2 with masked dq2 → d_ci2_pg
            var d_ci2_pg_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
            ](gpu_state.d_ci2_pg.unsafe_ptr())

            var g_c2_pg = gpu_state.critic2.online.grads_view()
            gpu_state.critic2.online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BS](
                ctx, dq2_t, d_ci2_pg_t, p_c2, new_q2_cache_t,
                g_c2_pg, gpu_state.critic2_ws
            )
            # Intentionally NO optimizer_step here

            # 7e3: d_ci += d_ci2_pg (combine gradients from both critics)
            @always_inline
            fn add_grads(
                dst: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
                ],
                src: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
                ],
            ):
                add_ci_grads_kernel[dtype, BS, Self.CRITIC_IN](dst, src)

            ctx.enqueue_function[add_grads, add_grads](
                d_ci_pg_t, d_ci2_pg_t,
                grid_dim=(ELEM_BLOCKS,), block_dim=(TPB,),
            )

            # 7f: Extract action gradients from d_ci
            var grad_act_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
            ](gpu_state.grad_act.unsafe_ptr())

            @always_inline
            fn extract_act_grad(
                da: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
                ],
                dnc: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
                ],
            ):
                actor_grad_from_critic_kernel[dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
                    da, dnc
                )

            ctx.enqueue_function[extract_act_grad, extract_act_grad](
                grad_act_t, d_ci_pg_t,
                grid_dim=(ACT_BLOCKS,), block_dim=(TPB,),
            )

            # 7g: Backward through reparameterization → actor_grad [BS, ACTOR_OUT]
            var actor_grad_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.ACTOR_OUT), MutAnyOrigin
            ](gpu_state.actor_grad.unsafe_ptr())
            var alpha_per_sample = Scalar[dtype](self.alpha / Float64(BS))

            @always_inline
            fn rsample_bwd(
                agrad: LayoutTensor[
                    dtype,
                    Layout.row_major(BS, Self.ACTOR_OUT),
                    MutAnyOrigin,
                ],
                ga: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
                ],
                aps: Scalar[dtype],
                ca: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
                ],
                eps: LayoutTensor[
                    dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
                ],
                ao: LayoutTensor[
                    dtype,
                    Layout.row_major(BS, Self.ACTOR_OUT),
                    MutAnyOrigin,
                ],
                lsmin: Scalar[dtype],
                lsmax: Scalar[dtype],
            ):
                sac_rsample_bwd_kernel[dtype, BS, Self.ACTIONS, Self.ACTOR_OUT](
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

            # 7h: Actor backward + optimizer step
            var d_obs_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
            ](gpu_state.d_obs.unsafe_ptr())

            var g_actor = gpu_state.actor.grads_view()
            gpu_state.actor.zero_grads(ctx)
            Self.ActorNet.backward_gpu[BS](
                ctx, actor_grad_t, d_obs_t, p_actor, actor_cache_t,
                g_actor, gpu_state.actor_ws
            )
            gpu_state.actor.optimizer_step(ctx)

            # ----- Phase 8: Alpha update via CPU-GPU sync -----
            if self.auto_alpha:
                ctx.synchronize()
                var lp_host = ctx.enqueue_create_host_buffer[dtype](BS)
                ctx.enqueue_copy(lp_host, gpu_state.curr_lp)
                ctx.synchronize()

                var mean_lp: Float64 = 0.0
                for b in range(BS):
                    mean_lp += Float64(lp_host[b])
                mean_lp /= Float64(BS)

                var grad = -self.alpha * (mean_lp + self.target_entropy)
                self.alpha_adam_t += 1
                var beta1: Float64 = 0.9
                var beta2: Float64 = 0.999
                var eps: Float64 = 1e-8
                self.alpha_adam_m = (
                    beta1 * self.alpha_adam_m + (1.0 - beta1) * grad
                )
                self.alpha_adam_v = (
                    beta2 * self.alpha_adam_v + (1.0 - beta2) * grad * grad
                )
                var m_hat = self.alpha_adam_m / (
                    1.0 - beta1 ** Float64(self.alpha_adam_t)
                )
                var v_hat = self.alpha_adam_v / (
                    1.0 - beta2 ** Float64(self.alpha_adam_t)
                )
                self.log_alpha -= self.alpha_lr * m_hat / (sqrt(v_hat) + eps)
                self.alpha = exp(self.log_alpha)

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Soft-update target critic networks on GPU. SAC has NO target actor."""
        gpu_state.critic1.soft_update(self.tau, ctx)
        gpu_state.critic2.soft_update(self.tau, ctx)

    fn decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        pass  # SAC uses entropy-based exploration, no epsilon

    # Convenience
    fn train[
        E: BoxContinuousActionEnv
    ](mut self, mut env: E, num_episodes: Int = 300) raises -> TrainingMetrics:
        var cpu_state = self.make_cpu_state()
        var ckpt_path = String(self.checkpoint_path)
        return run_offpolicy_continuous_train(
            self, cpu_state, env, num_episodes,
            checkpoint_every=self.checkpoint_every,
            checkpoint_path=ckpt_path,
        )

    fn train_gpu[
        E: GPUContinuousEnv,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
    ) raises -> TrainingMetrics:
        """Train using GPU-accelerated training loop."""
        from mojo_rl.deep_agents.core.perf_timer import PerfTimer
        var timer = PerfTimer[False]()
        return run_offpolicy_continuous_train_gpu[E, Self, 0](
            self, ctx, num_steps, timer,
            warmup_steps=warmup_steps,
        )
