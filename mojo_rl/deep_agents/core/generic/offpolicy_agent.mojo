"""Generic off-policy agent parameterized by OffPolicyConfig.

Supports DDPG (single critic) and TD3 (twin critics) via comptime if
branching on Config.NUM_CRITICS. Follows the ModelDef pattern: components
as trait-bounded type params, all dims derived from Self.Config.*.
"""

from std.random import random_float64
from layout import Layout, LayoutTensor
from std.memory import UnsafePointer

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.constants import TPB
from mojo_rl.nn.training import Network, NetworkState, NetworkPair, GPUNetworkPair
from mojo_rl.nn.initializer import Kaiming, Xavier
from mojo_rl.nn.gpu.random import gaussian_noise
from std.gpu.host import DeviceContext, DeviceBuffer

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
from mojo_rl.deep_agents.core.utils import obs_to_inline, concat_obs_action_batch
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.deep_agents.core.kernels import (
    concat_obs_action_kernel,
    ddpg_exploration_kernel,
    td_mse_grad_kernel,
    actor_grad_from_critic_kernel,
    td_target_continuous_kernel,
    td_target_min_twin_kernel,
)
from mojo_rl.core import TrainingMetrics, BoxContinuousActionEnv, GPUContinuousEnv

from .offpolicy_config import OffPolicyConfig
from .exploration import GaussianNoise
from .update_schedule import EveryStep, DelayedActorAndTargets


# =============================================================================
# GenericCPUState — supports single and twin critics
# =============================================================================


struct GenericCPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
    num_critics: Int = 1,
](Movable, OffPolicyState):
    """CPU state supporting single (DDPG) and twin (TD3) critics."""

    comptime BUFFER_DTYPE = dtype

    # Networks — critic2 always allocated (cheap), only used when num_critics==2
    var actor: NetworkPair[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkPair[Self.CriticModel, Self.CriticOpt]
    var critic2: NetworkPair[Self.CriticModel, Self.CriticOpt]

    # Replay buffer
    var buffer: HeapReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

    # Workspace
    var ws: List[Scalar[dtype]]

    fn __init__(out self):
        self.actor = NetworkPair[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier[]]()
        self.critic = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        self.critic.initialize[Kaiming[]]()
        self.critic2 = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        comptime if Self.num_critics == 2:
            self.critic2.initialize[Kaiming[]]()
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        # Workspace size — includes twin critic regions when num_critics==2
        comptime CO = Self.CriticModel.OUT_DIM
        comptime ACT = Self.ActorModel.OUT_DIM
        comptime BS = Self.batch_size
        comptime CI = Self.CriticModel.IN_DIM
        comptime ACS = Self.ActorModel.CACHE_SIZE
        comptime CCS = Self.CriticModel.CACHE_SIZE
        comptime OBS = Self.ActorModel.IN_DIM
        # Base regions (shared by DDPG and TD3)
        comptime BASE_SIZE = (
            BS * ACT       # next_act
            + BS * CI      # next_ci
            + BS * CO      # next_q (critic 1)
            + BS * CO      # targets
            + BS * CI      # ci
            + BS * CO      # q_out (critic 1)
            + BS * CCS     # q_cache (critic 1)
            + BS * CO      # q_grad
            + BS * CI      # d_ci
            + BS * ACT     # actor_act
            + BS * ACS     # actor_cache
            + BS * CI      # new_ci
            + BS * CO      # new_q
            + BS * CO      # dq
            + BS * CI      # d_new_ci
            + BS * ACT     # d_act
            + BS * OBS     # d_obs
        )
        # Twin critic extra regions (q2_out, q2_cache, nq2)
        comptime TWIN_EXTRA = (
            BS * CO        # nq2 (critic 2 target output)
            + BS * CO      # q2_out (critic 2 online output)
            + BS * CCS     # q2_cache (critic 2 cache)
        ) if Self.num_critics == 2 else 0
        comptime WS_SIZE = BASE_SIZE + TWIN_EXTRA

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
# Inline concat helper — avoids type unification issues
# =============================================================================


@always_inline
fn _concat_obs_act[
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


# =============================================================================
# GenericGPUState — GPU buffer container for off-policy agents
# =============================================================================


struct GenericGPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
    max_n_envs: Int,
    num_critics: Int = 1,
](GPUOffPolicyState):
    """GPU state for off-policy agents (DDPG/TD3)."""

    comptime OBS = Self.ActorModel.IN_DIM
    comptime ACTIONS = Self.ActorModel.OUT_DIM
    comptime CRITIC_IN = Self.CriticModel.IN_DIM
    comptime CRITIC_OUT = Self.CriticModel.OUT_DIM
    comptime ACTOR_CS = Self.ActorModel.CACHE_SIZE
    comptime CRITIC_CS = Self.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.ActorModel, Self.ActorOpt]
    comptime CriticNet = Network[Self.CriticModel, Self.CriticOpt]
    comptime ACTOR_WS = Self.ActorNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS = Self.CriticNet.WORKSPACE_SIZE_PER_SAMPLE

    # GPU networks
    var actor: GPUNetworkPair[Self.ActorModel, Self.ActorOpt]
    var critic: GPUNetworkPair[Self.CriticModel, Self.CriticOpt]
    var critic2: GPUNetworkPair[Self.CriticModel, Self.CriticOpt]

    # GPU replay buffer
    var buffer: GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim, Self.action_dim]

    # Exploration buffers (sized by max_n_envs)
    var raw_act: DeviceBuffer[dtype]
    var inf_ws: DeviceBuffer[dtype]

    # Replay sample output
    var s_obs: DeviceBuffer[dtype]
    var s_act: DeviceBuffer[dtype]
    var s_rew: DeviceBuffer[dtype]
    var s_nobs: DeviceBuffer[dtype]
    var s_done: DeviceBuffer[dtype]
    var s_idx: DeviceBuffer[DType.int32]

    # TD targets
    var next_act: DeviceBuffer[dtype]
    var next_ci: DeviceBuffer[dtype]
    var next_q: DeviceBuffer[dtype]
    var targets: DeviceBuffer[dtype]

    # Critic update
    var ci: DeviceBuffer[dtype]
    var q_out: DeviceBuffer[dtype]
    var q_cache: DeviceBuffer[dtype]
    var critic_ws: DeviceBuffer[dtype]
    var q_grad: DeviceBuffer[dtype]
    var d_ci: DeviceBuffer[dtype]

    # Actor update
    var actor_act: DeviceBuffer[dtype]
    var new_ci: DeviceBuffer[dtype]
    var new_q: DeviceBuffer[dtype]
    var new_q_cache: DeviceBuffer[dtype]
    var actor_cache: DeviceBuffer[dtype]
    var actor_ws: DeviceBuffer[dtype]
    var dq: DeviceBuffer[dtype]
    var d_new_ci: DeviceBuffer[dtype]
    var d_act: DeviceBuffer[dtype]
    var d_obs: DeviceBuffer[dtype]

    # Twin critic extra (only used when num_critics==2)
    var nq2: DeviceBuffer[dtype]
    var q2_out: DeviceBuffer[dtype]
    var q2_cache: DeviceBuffer[dtype]
    var critic2_ws: DeviceBuffer[dtype]

    fn __init__(out self, ctx: DeviceContext) raises:
        self.actor = GPUNetworkPair[Self.ActorModel, Self.ActorOpt](ctx)
        self.critic = GPUNetworkPair[Self.CriticModel, Self.CriticOpt](ctx)
        self.critic2 = GPUNetworkPair[Self.CriticModel, Self.CriticOpt](ctx)
        self.buffer = GPUReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim
        ](ctx)

        comptime BS = Self.batch_size
        comptime MNE = Self.max_n_envs

        # Exploration
        self.raw_act = ctx.enqueue_create_buffer[dtype](MNE * Self.ACTIONS)
        self.inf_ws = ctx.enqueue_create_buffer[dtype](
            max(1, MNE * Self.ACTOR_WS)
        )

        # Sample output
        self.s_obs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)
        self.s_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.s_rew = ctx.enqueue_create_buffer[dtype](BS)
        self.s_nobs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)
        self.s_done = ctx.enqueue_create_buffer[dtype](BS)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](BS)

        # TD targets
        self.next_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.next_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.next_q = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.targets = ctx.enqueue_create_buffer[dtype](BS)

        # Critic
        self.ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.q_out = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.q_cache = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_CS)
        self.critic_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.CRITIC_WS)
        )
        self.q_grad = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.d_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)

        # Actor
        self.actor_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.new_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.new_q = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.new_q_cache = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_CS)
        self.actor_cache = ctx.enqueue_create_buffer[dtype](BS * Self.ACTOR_CS)
        self.actor_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.ACTOR_WS)
        )
        self.dq = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.d_new_ci = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_IN)
        self.d_act = ctx.enqueue_create_buffer[dtype](BS * Self.ACTIONS)
        self.d_obs = ctx.enqueue_create_buffer[dtype](BS * Self.OBS)

        # Twin critic extra
        self.nq2 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.q2_out = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.q2_cache = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_CS)
        self.critic2_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.CRITIC_WS)
        )

        # Pre-fill dq with -1/batch_size
        ctx.synchronize()
        var dq_host = ctx.enqueue_create_host_buffer[dtype](BS * Self.CRITIC_OUT)
        for i in range(BS * Self.CRITIC_OUT):
            dq_host[i] = Scalar[dtype](-1.0 / Float64(BS))
        ctx.enqueue_copy(self.dq, dq_host)

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
# GenericOffPolicyAgent[Config: OffPolicyConfig]
# =============================================================================


struct GenericOffPolicyAgent[
    Config: OffPolicyConfig,
](OffPolicyContinuousAgent & GPUOffPolicyAgent & Checkpointable):
    """Generic off-policy agent. Supports DDPG and TD3 via Config."""

    # Dimensions from Config's Model types
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.ActorModel.OUT_DIM
    comptime BATCH: Int = Self.Config.batch_size
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]

    # Workspace offsets (base regions — always present)
    comptime _O_NEXT_ACT: Int = 0
    comptime _O_NEXT_CI: Int = Self._O_NEXT_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_NEXT_Q: Int = Self._O_NEXT_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_TARGETS: Int = Self._O_NEXT_Q + Self.BATCH * Self.CRITIC_OUT
    comptime _O_CI: Int = Self._O_TARGETS + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q_OUT: Int = Self._O_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_Q_CACHE: Int = Self._O_Q_OUT + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q_GRAD: Int = Self._O_Q_CACHE + Self.BATCH * Self.CRITIC_CS
    comptime _O_D_CI: Int = Self._O_Q_GRAD + Self.BATCH * Self.CRITIC_OUT
    comptime _O_ACTOR_ACT: Int = Self._O_D_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_ACTOR_CACHE: Int = Self._O_ACTOR_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_NEW_CI: Int = Self._O_ACTOR_CACHE + Self.BATCH * Self.ACTOR_CS
    comptime _O_NEW_Q: Int = Self._O_NEW_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_DQ: Int = Self._O_NEW_Q + Self.BATCH * Self.CRITIC_OUT
    comptime _O_D_NEW_CI: Int = Self._O_DQ + Self.BATCH * Self.CRITIC_OUT
    comptime _O_D_ACT: Int = Self._O_D_NEW_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_D_OBS: Int = Self._O_D_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_BASE_END: Int = Self._O_D_OBS + Self.BATCH * Self.OBS

    # Twin critic extra offsets (only valid when NUM_CRITICS==2)
    comptime _O_NQ2: Int = Self._O_BASE_END
    comptime _O_Q2_OUT: Int = Self._O_NQ2 + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q2_CACHE: Int = Self._O_Q2_OUT + Self.BATCH * Self.CRITIC_OUT

    # CPU state type
    comptime CPUStateType = GenericCPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.buffer_capacity,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.ActorModel.OUT_DIM,
        Self.Config.batch_size,
        Self.Config.NUM_CRITICS,
    ]

    # GPUOffPolicyAgent required comptime constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACTIONS
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = 64  # default, can be overridden via Config
    comptime GPUStateType = GenericGPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.buffer_capacity,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.ActorModel.OUT_DIM,
        Self.Config.batch_size,
        64,  # max_n_envs
        Self.Config.NUM_CRITICS,
    ]

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64
    var explore: GaussianNoise

    # TD3-specific
    var policy_delay: Int
    var target_noise_std: Float64
    var target_noise_clip: Float64
    var update_count: Int

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
        noise_std: Float64 = 0.1,
        noise_std_min: Float64 = 0.01,
        noise_decay: Float64 = 0.995,
        policy_delay: Int = 2,
        target_noise_std: Float64 = 0.2,
        target_noise_clip: Float64 = 0.5,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.explore = GaussianNoise(noise_std, noise_std_min, noise_decay)
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.update_count = 0
        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

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
        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())
        var p = cpu_state.actor.online.params_view()
        Self.ActorNet.forward[1](obs_t, act_t, p)
        var raw = List[Scalar[d]](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            raw.append(Scalar[d](Float64(act_arr[i]) * self.action_scale))
        return self.explore.explore[d](raw, self.action_scale)

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

        self.update_count += 1

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
        var act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](b_act.unsafe_ptr())

        var ws = cpu_state.ws.unsafe_ptr()

        # Phase 2: TD targets
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](ws + Self._O_NEXT_ACT)
        var p_at = cpu_state.actor.target.params_view()
        Self.ActorNet.forward[Self.BATCH](next_obs_t, next_act_t, p_at)

        # TD3: target policy smoothing (add clipped noise to target actions)
        comptime if Self.Config.NUM_CRITICS == 2:
            for b in range(Self.BATCH):
                for i in range(Self.ACTIONS):
                    var idx = b * Self.ACTIONS + i
                    var noise = gaussian_noise() * self.target_noise_std
                    if noise > self.target_noise_clip:
                        noise = self.target_noise_clip
                    elif noise < -self.target_noise_clip:
                        noise = -self.target_noise_clip
                    var noisy_a = Float64((ws + Self._O_NEXT_ACT)[idx]) + noise
                    if noisy_a > 1.0:
                        noisy_a = 1.0
                    elif noisy_a < -1.0:
                        noisy_a = -1.0
                    (ws + Self._O_NEXT_ACT)[idx] = Scalar[dtype](noisy_a)

        # Concat next_obs + next_act → next_ci
        _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
            ws + Self._O_NEXT_CI, b_next.unsafe_ptr(), ws + Self._O_NEXT_ACT
        )
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_NEXT_CI)

        # Forward critic target(s)
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_NEXT_Q)
        var p_ct = cpu_state.critic.target.params_view()
        Self.CriticNet.forward[Self.BATCH](next_ci_t, next_q_t, p_ct)

        var tgt_p = ws + Self._O_TARGETS
        var nq_p = ws + Self._O_NEXT_Q

        # DDPG: single Q target
        comptime if Self.Config.NUM_CRITICS == 1:
            for b in range(Self.BATCH):
                var q = Float64(nq_p[b])
                if q != q:
                    q = 0.0
                var dm = 1.0 - Float64(b_done[b])
                var tgt = Float64(b_rew[b]) + self.gamma * q * dm
                if tgt != tgt:
                    tgt = 0.0
                elif tgt > 1000.0:
                    tgt = 1000.0
                elif tgt < -1000.0:
                    tgt = -1000.0
                tgt_p[b] = Scalar[dtype](tgt)

        # TD3: min(Q1, Q2) target
        comptime if Self.Config.NUM_CRITICS == 2:
            var nq2_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
            ](ws + Self._O_NQ2)
            var p_c2t = cpu_state.critic2.target.params_view()
            Self.CriticNet.forward[Self.BATCH](next_ci_t, nq2_t, p_c2t)

            var nq2_p = ws + Self._O_NQ2
            for b in range(Self.BATCH):
                var q1 = Float64(nq_p[b])
                var q2 = Float64(nq2_p[b])
                if q1 != q1:
                    q1 = 0.0
                if q2 != q2:
                    q2 = 0.0
                var min_q = q1 if q1 < q2 else q2
                var dm = 1.0 - Float64(b_done[b])
                var tgt = Float64(b_rew[b]) + self.gamma * min_q * dm
                if tgt != tgt:
                    tgt = 0.0
                elif tgt > 1000.0:
                    tgt = 1000.0
                elif tgt < -1000.0:
                    tgt = -1000.0
                tgt_p[b] = Scalar[dtype](tgt)

        # Phase 3: Critic update
        _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
            ws + Self._O_CI, b_obs.unsafe_ptr(), b_act.unsafe_ptr()
        )
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_CI)

        # --- Critic 1 update ---
        var q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_Q_OUT)
        var q_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
        ](ws + Self._O_Q_CACHE)
        var p_c = cpu_state.critic.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q_t, p_c, q_cache_t
        )

        var q_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_Q_GRAD)
        var qo_p = ws + Self._O_Q_OUT
        var qg_p = ws + Self._O_Q_GRAD
        var critic_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = qo_p[b] - tgt_p[b]
            critic_loss += Float64(td_err * td_err)
            qg_p[b] = Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
        critic_loss /= Float64(Self.BATCH)

        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_D_CI)
        var g_c = cpu_state.critic.grads_view()
        cpu_state.critic.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q_grad_t, d_ci_t, p_c, q_cache_t, g_c
        )
        cpu_state.critic.optimizer_step()

        # --- Critic 2 update (TD3 only) ---
        comptime if Self.Config.NUM_CRITICS == 2:
            var q2_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
            ](ws + Self._O_Q2_OUT)
            var q2_cache_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
            ](ws + Self._O_Q2_CACHE)
            var p_c2 = cpu_state.critic2.params_view()
            Self.CriticNet.forward_with_cache[Self.BATCH](
                ci_t, q2_t, p_c2, q2_cache_t
            )

            var q2o_p = ws + Self._O_Q2_OUT
            var critic2_loss: Float64 = 0.0
            for b in range(Self.BATCH):
                var td_err = q2o_p[b] - tgt_p[b]
                critic2_loss += Float64(td_err * td_err)
                # Reuse q_grad for critic2 (same target)
                qg_p[b] = Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
            critic2_loss /= Float64(Self.BATCH)

            var g_c2 = cpu_state.critic2.grads_view()
            cpu_state.critic2.zero_grads()
            Self.CriticNet.backward[Self.BATCH](
                q_grad_t, d_ci_t, p_c2, q2_cache_t, g_c2
            )
            cpu_state.critic2.optimizer_step()
            critic_loss = (critic_loss + critic2_loss) / 2.0

        # Phase 4: Actor update
        # DDPG: every step; TD3: every policy_delay steps
        var do_actor_update = True
        comptime if Self.Config.NUM_CRITICS == 2:
            do_actor_update = self.update_count % self.policy_delay == 0

        if do_actor_update:
            var aa_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](ws + Self._O_ACTOR_ACT)
            var ac_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTOR_CS), MutAnyOrigin
            ](ws + Self._O_ACTOR_CACHE)
            var p_a = cpu_state.actor.params_view()
            Self.ActorNet.forward_with_cache[Self.BATCH](
                obs_t, aa_t, p_a, ac_t
            )

            # Concat obs + actor_act → new_ci
            _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
                ws + Self._O_NEW_CI, b_obs.unsafe_ptr(), ws + Self._O_ACTOR_ACT
            )
            var nci_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
            ](ws + Self._O_NEW_CI)

            var nq_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
            ](ws + Self._O_NEW_Q)
            var nc_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
            ](ws + Self._O_Q_CACHE)  # reuse q_cache
            # Always use critic1 for policy gradient (both DDPG and TD3)
            Self.CriticNet.forward_with_cache[Self.BATCH](
                nci_t, nq_t, p_c, nc_t
            )

            var dq_p = ws + Self._O_DQ
            for b in range(Self.BATCH):
                dq_p[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))
            var dq_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
            ](dq_p)

            var dnci_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
            ](ws + Self._O_D_NEW_CI)
            cpu_state.critic.zero_grads()
            Self.CriticNet.backward[Self.BATCH](dq_t, dnci_t, p_c, nc_t, g_c)

            var da_p = ws + Self._O_D_ACT
            var dnci_p = ws + Self._O_D_NEW_CI
            for b in range(Self.BATCH):
                for i in range(Self.ACTIONS):
                    da_p[b * Self.ACTIONS + i] = dnci_p[
                        b * Self.CRITIC_IN + Self.OBS + i
                    ]

            var da_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](da_p)
            var do_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
            ](ws + Self._O_D_OBS)
            var g_a = cpu_state.actor.grads_view()
            cpu_state.actor.zero_grads()
            Self.ActorNet.backward[Self.BATCH](da_t, do_t, p_a, ac_t, g_a)
            cpu_state.actor.optimizer_step()

        # Phase 5: Soft update targets
        comptime if Self.Config.NUM_CRITICS == 1:
            # DDPG: update every step
            cpu_state.actor.soft_update(self.tau)
            cpu_state.critic.soft_update(self.tau)
        comptime if Self.Config.NUM_CRITICS == 2:
            # TD3: only on delayed steps
            if self.update_count % self.policy_delay == 0:
                cpu_state.actor.soft_update(self.tau)
                cpu_state.critic.soft_update(self.tau)
                cpu_state.critic2.soft_update(self.tau)

        self.train_step_count += 1
        return critic_loss

    fn decay_explore(mut self) -> None:
        self.explore.decay()

    fn get_explore_rate(self) -> Float64:
        return self.explore.get_rate()

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
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())
        var p = cpu_state.actor.online.params_view()
        Self.ActorNet.forward[1](obs_t, act_t, p)
        var result = List[Float64](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            var a = Float64(act_arr[i]) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            result.append(a)
        return result^

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
        gpu_state.critic.upload_from(cpu.critic, ctx)
        comptime if Self.Config.NUM_CRITICS == 2:
            gpu_state.critic2.upload_from(cpu.critic2, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        # Download into a temporary CPU state — we don't hold persistent state
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
        """Forward actor on GPU + exploration noise."""
        comptime BLOCKS = (N_ENVS * Self.ACTIONS + TPB - 1) // TPB

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.raw_act.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        var p = gpu_state.actor.online.params_view()
        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.inf_ws
        )

        var noise_std_s = Scalar[dtype](self.explore.noise_std)
        var scale_s = Scalar[dtype](self.action_scale)
        var rng_seed_s = Scalar[DType.uint32](
            UInt32(self.total_steps) * UInt32(Self.ACTIONS)
        )

        @always_inline
        fn exploration_wrapper(
            out_t: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            raw_in: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            ns: Scalar[dtype],
            sc: Scalar[dtype],
            rng_seed: Scalar[DType.uint32],
        ):
            ddpg_exploration_kernel[dtype, N_ENVS, Self.ACTIONS](
                out_t, raw_in, ns, sc, rng_seed
            )

        ctx.enqueue_function[exploration_wrapper, exploration_wrapper](
            act_t, raw_t, noise_std_s, scale_s, rng_seed_s,
            grid_dim=(BLOCKS,), block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """DDPG GPU training step: sample → TD targets → critic → actor."""
        comptime BS = Self.BATCH
        comptime ELEM_BLOCKS = (BS * Self.CRITIC_IN + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (BS + TPB - 1) // TPB
        comptime ACT_BLOCKS = (BS * Self.ACTIONS + TPB - 1) // TPB

        # Phase 1: Sample batch
        gpu_state.buffer.sample[BS](
            ctx,
            rng_seed=UInt32(self.total_steps) * UInt32(BS + 1),
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
        var rew_t = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.s_rew.unsafe_ptr())
        var done_t = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.s_done.unsafe_ptr())

        # Phase 2: TD targets
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.next_act.unsafe_ptr())
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.next_ci.unsafe_ptr())
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.next_q.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BS), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())

        var p_actor_t = gpu_state.actor.target.params_view()
        var p_critic_t = gpu_state.critic.target.params_view()
        var p_actor = gpu_state.actor.online.params_view()
        var p_critic = gpu_state.critic.online.params_view()

        Self.ActorNet.forward_gpu[BS](
            ctx, nobs_t, next_act_t, p_actor_t, gpu_state.actor_ws
        )

        # TD3: target policy smoothing (add clipped noise to target actions)
        comptime if Self.Config.NUM_CRITICS == 2:
            from mojo_rl.deep_agents.td3.kernels import add_gaussian_noise_kernel

            var tn_std = Scalar[dtype](self.target_noise_std)
            var tn_clip = Scalar[dtype](self.target_noise_clip)
            var a_min = Scalar[dtype](-1.0)
            var a_max = Scalar[dtype](1.0)
            var noise_seed = Scalar[DType.uint32](
                UInt32(self.total_steps) * UInt32(BS * Self.ACTIONS + 7)
            )

            @always_inline
            fn noise_k(
                out_t: LayoutTensor[dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin],
                in_t: LayoutTensor[dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin],
                ns: Scalar[dtype], nc: Scalar[dtype],
                amin: Scalar[dtype], amax: Scalar[dtype],
                seed_v: Scalar[DType.uint32],
            ):
                add_gaussian_noise_kernel[dtype, BS, Self.ACTIONS](
                    out_t, in_t, ns, nc, amin, amax, seed_v
                )

            # Write noisy actions back into next_act buffer
            ctx.enqueue_function[noise_k, noise_k](
                next_act_t, next_act_t, tn_std, tn_clip, a_min, a_max, noise_seed,
                grid_dim=(ACT_BLOCKS,), block_dim=(TPB,),
            )

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

        var gamma_s = Scalar[dtype](self.gamma)

        # DDPG: single critic target
        comptime if Self.Config.NUM_CRITICS == 1:
            Self.CriticNet.forward_gpu[BS](
                ctx, next_ci_t, next_q_t, p_critic_t, gpu_state.critic_ws
            )
            var nq_flat_t = LayoutTensor[
                dtype, Layout.row_major(BS), MutAnyOrigin
            ](gpu_state.next_q.unsafe_ptr())

            @always_inline
            fn compute_targets_single(
                tgt: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                r: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                nq: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                d: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                g: Scalar[dtype],
            ):
                td_target_continuous_kernel[dtype, BS](tgt, r, nq, d, g)

            ctx.enqueue_function[compute_targets_single, compute_targets_single](
                targets_t, rew_t, nq_flat_t, done_t, gamma_s,
                grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
            )

        # TD3: twin critic targets with min(Q1, Q2)
        comptime if Self.Config.NUM_CRITICS == 2:
            Self.CriticNet.forward_gpu[BS](
                ctx, next_ci_t, next_q_t, p_critic_t, gpu_state.critic_ws
            )
            var nq2_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
            ](gpu_state.nq2.unsafe_ptr())
            var p_c2t = gpu_state.critic2.target.params_view()
            Self.CriticNet.forward_gpu[BS](
                ctx, next_ci_t, nq2_t, p_c2t, gpu_state.critic2_ws
            )

            var nq1_flat = LayoutTensor[
                dtype, Layout.row_major(BS), MutAnyOrigin
            ](gpu_state.next_q.unsafe_ptr())
            var nq2_flat = LayoutTensor[
                dtype, Layout.row_major(BS), MutAnyOrigin
            ](gpu_state.nq2.unsafe_ptr())
            # Dummy log_probs (unused for TD3, use_entropy=False)
            var dummy_lp = nq1_flat  # won't be read
            var zero_alpha = Scalar[dtype](0.0)

            @always_inline
            fn compute_targets_twin(
                tgt: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                r: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                q1: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                q2: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                d: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                lp: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
                g: Scalar[dtype],
                a: Scalar[dtype],
            ):
                td_target_min_twin_kernel[dtype, BS, False](
                    tgt, r, q1, q2, d, lp, g, a
                )

            ctx.enqueue_function[compute_targets_twin, compute_targets_twin](
                targets_t, rew_t, nq1_flat, nq2_flat, done_t,
                dummy_lp, gamma_s, zero_alpha,
                grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
            )

        # Phase 3: Critic update
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.ci.unsafe_ptr())
        var q_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q_out.unsafe_ptr())
        var q_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
        ](gpu_state.q_cache.unsafe_ptr())
        var q_grad_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.q_grad.unsafe_ptr())
        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci.unsafe_ptr())

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

        Self.CriticNet.forward_gpu_with_cache[BS](
            ctx, ci_t, q_t, p_critic, q_cache_t, gpu_state.critic_ws
        )

        @always_inline
        fn mse_grad_k(
            qg: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin],
            q: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
        ):
            td_mse_grad_kernel[dtype, BS, Self.CRITIC_OUT](qg, q, tgt)

        ctx.enqueue_function[mse_grad_k, mse_grad_k](
            q_grad_t, q_t, targets_t,
            grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
        )

        var g_critic = gpu_state.critic.online.grads_view()
        gpu_state.critic.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BS](
            ctx, q_grad_t, d_ci_t, p_critic, q_cache_t,
            g_critic, gpu_state.critic_ws
        )
        gpu_state.critic.online.optimizer_step(ctx)

        # TD3: update critic2
        comptime if Self.Config.NUM_CRITICS == 2:
            var q2_out_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
            ](gpu_state.q2_out.unsafe_ptr())
            var q2_cache_t = LayoutTensor[
                dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
            ](gpu_state.q2_cache.unsafe_ptr())
            var p_c2 = gpu_state.critic2.online.params_view()
            Self.CriticNet.forward_gpu_with_cache[BS](
                ctx, ci_t, q2_out_t, p_c2, q2_cache_t, gpu_state.critic2_ws
            )

            # Reuse q_grad for critic2 MSE
            @always_inline
            fn mse_grad_c2(
                qg: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin],
                q: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin],
                tgt: LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin],
            ):
                td_mse_grad_kernel[dtype, BS, Self.CRITIC_OUT](qg, q, tgt)

            ctx.enqueue_function[mse_grad_c2, mse_grad_c2](
                q_grad_t, q2_out_t, targets_t,
                grid_dim=(BATCH_BLOCKS,), block_dim=(TPB,),
            )

            var g_c2 = gpu_state.critic2.online.grads_view()
            gpu_state.critic2.online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BS](
                ctx, q_grad_t, d_ci_t, p_c2, q2_cache_t,
                g_c2, gpu_state.critic2_ws
            )
            gpu_state.critic2.online.optimizer_step(ctx)

        # Phase 4: Actor update
        # TD3: only update every policy_delay steps
        var do_actor_update = True
        comptime if Self.Config.NUM_CRITICS == 2:
            self.update_count += 1
            do_actor_update = self.update_count % self.policy_delay == 0

        if not do_actor_update:
            self.train_step_count += 1
            return

        # Phase 4 (cont): Actor update
        var actor_act_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.actor_act.unsafe_ptr())
        var new_ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.new_ci.unsafe_ptr())
        var new_q_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.new_q.unsafe_ptr())
        var new_q_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
        ](gpu_state.new_q_cache.unsafe_ptr())
        var actor_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTOR_CS), MutAnyOrigin
        ](gpu_state.actor_cache.unsafe_ptr())
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](gpu_state.dq.unsafe_ptr())
        var d_new_ci_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_new_ci.unsafe_ptr())
        var d_act_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.d_act.unsafe_ptr())
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
        ](gpu_state.d_obs.unsafe_ptr())

        Self.ActorNet.forward_gpu_with_cache[BS](
            ctx, obs_t, actor_act_t, p_actor, actor_cache_t,
            gpu_state.actor_ws
        )

        @always_inline
        fn concat_new_k(
            d: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin],
            o: LayoutTensor[dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin],
            a: LayoutTensor[dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin],
        ):
            concat_obs_action_kernel[dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](d, o, a)

        ctx.enqueue_function[concat_new_k, concat_new_k](
            new_ci_t, obs_t, actor_act_t,
            grid_dim=(ELEM_BLOCKS,), block_dim=(TPB,),
        )

        Self.CriticNet.forward_gpu_with_cache[BS](
            ctx, new_ci_t, new_q_t, p_critic, new_q_cache_t,
            gpu_state.critic_ws
        )

        var g_critic2 = gpu_state.critic.online.grads_view()
        gpu_state.critic.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BS](
            ctx, dq_t, d_new_ci_t, p_critic, new_q_cache_t,
            g_critic2, gpu_state.critic_ws
        )

        @always_inline
        fn extract_grad_k(
            da: LayoutTensor[dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin],
            dnc: LayoutTensor[dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin],
        ):
            actor_grad_from_critic_kernel[dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](da, dnc)

        ctx.enqueue_function[extract_grad_k, extract_grad_k](
            d_act_t, d_new_ci_t,
            grid_dim=(ACT_BLOCKS,), block_dim=(TPB,),
        )

        var g_actor = gpu_state.actor.online.grads_view()
        gpu_state.actor.online.zero_grads(ctx)
        Self.ActorNet.backward_gpu[BS](
            ctx, d_act_t, d_obs_t, p_actor, actor_cache_t,
            g_actor, gpu_state.actor_ws
        )
        gpu_state.actor.online.optimizer_step(ctx)

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        comptime if Self.Config.NUM_CRITICS == 1:
            gpu_state.actor.soft_update(self.tau, ctx)
            gpu_state.critic.soft_update(self.tau, ctx)
        comptime if Self.Config.NUM_CRITICS == 2:
            # TD3: soft update only on delayed steps (already checked in train_step)
            gpu_state.actor.soft_update(self.tau, ctx)
            gpu_state.critic.soft_update(self.tau, ctx)
            gpu_state.critic2.soft_update(self.tau, ctx)

    fn decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        """No-op for DDPG/TD3 (Gaussian noise decay is per-episode, not per-step)."""
        pass

    # Checkpointable
    fn save_checkpoint(self, path: String) raises -> None:
        pass

    fn load_checkpoint(mut self, path: String) raises -> None:
        pass

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
