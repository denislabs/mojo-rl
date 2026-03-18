"""Generic off-policy agent parameterized by OffPolicyConfig.

Supports DDPG, TD3, and SAC via strategy delegation from Config.
Follows the ModelDef pattern: components as trait-bounded type params,
all dims derived from Self.Config.*. Algorithm-specific behavior is
delegated to Config.Explore, Config.Schedule, Config.TargetAction,
Config.TargetValue, and Config.ActorLoss strategy types.
"""

from std.random import random_float64
from std.math import exp, log, sqrt
from layout import Layout, LayoutTensor
from std.memory import UnsafePointer
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    save_checkpoint_file,
    read_checkpoint_file,
)
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.constants import TPB
from mojo_rl.nn.training import (
    Network,
    NetworkState,
    NetworkPair,
    GPUNetworkPair,
)
from mojo_rl.nn.initializer import Kaiming, Xavier
from mojo_rl.nn.gpu.random import gaussian_noise
from mojo_rl.nn.model.stochastic_actor import (
    rsample,
    get_deterministic_action,
)
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.deep_agents.core import (
    OffPolicyState,
    OffPolicyContinuousAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    run_offpolicy_continuous_train,
    run_offpolicy_continuous_train_gpu,
    Checkpointable,
)
from mojo_rl.deep_agents.core.utils import (
    obs_to_inline,
    concat_obs_action_batch,
)
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.deep_agents.core.kernels import (
    concat_obs_action_kernel,
    ddpg_exploration_kernel,
    td_mse_grad_kernel,
    actor_grad_from_critic_kernel,
    td_target_continuous_kernel,
    td_target_min_twin_kernel,
)
from mojo_rl.deep_agents.sac.kernels import sac_sample_actions_kernel
from mojo_rl.deep_agents.dreamer_v3.kernels import (
    gradient_norm_kernel,
    gradient_reduce_apply_fused_kernel,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)

from mojo_rl.deep_agents.core.perf_timer import PerfTimer

# PerfTimerPtr not used (L3 profiling requires concrete Model types)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.eval import run_offpolicy_continuous_eval

from .offpolicy_config import OffPolicyConfig
from .exploration import GaussianNoise
from .update_schedule import EveryStep, DelayedAll


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
    has_target_actor: Bool = True,
    actor_loss_ws: Int = 0,
    target_action_ws: Int = 0,
](Movable, OffPolicyState):
    """CPU state supporting single (DDPG), twin (TD3), and SAC critics."""

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

        # Workspace size: 3 regions
        comptime CO = Self.CriticModel.OUT_DIM
        comptime ACT = Self.action_dim  # ACTIONS (not ActorModel.OUT_DIM for SAC)
        comptime ACTOR_OUT = Self.ActorModel.OUT_DIM
        comptime BS = Self.batch_size
        comptime CI = Self.CriticModel.IN_DIM
        comptime CCS = Self.CriticModel.CACHE_SIZE
        comptime OBS = Self.ActorModel.IN_DIM

        # Region 1: target computation
        comptime R1_SIZE = (
            BS * ACT  # next_act
            + BS  # next_lp (log probs, used by SAC)
            + BS * CI  # next_ci
            + BS * CO  # next_q1
            + BS * CO  # next_q2 (only used when num_critics==2)
            + BS * CO  # targets
        )

        # Region 2: critic update
        comptime R2_SIZE = (
            BS * CI  # ci
            + BS * CO  # q1_out
            + BS * CCS  # q1_cache
            + BS * CO  # q2_out (only used when num_critics==2)
            + BS * CCS  # q2_cache (only used when num_critics==2)
            + BS * CO  # q_grad
            + BS * CI  # d_ci
        )

        # Region 3: strategy workspaces (max of ActorLoss and TargetAction ws)
        comptime R3_SIZE = Self.actor_loss_ws if Self.actor_loss_ws > Self.target_action_ws else Self.target_action_ws

        comptime WS_SIZE = R1_SIZE + R2_SIZE + R3_SIZE

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
    strat_ws_size: Int = 0,
    target_strat_ws_size: Int = 0,
](GPUOffPolicyState):
    """GPU state for off-policy agents (DDPG/TD3/SAC).

    Strategy workspaces replace individual actor-update buffers.
    The strategy types manage their own internal workspace layout.
    """

    comptime OBS = Self.ActorModel.IN_DIM
    comptime ACTIONS = Self.action_dim
    comptime ACTOR_OUT = Self.ActorModel.OUT_DIM
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
    var buffer: GPUReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim
    ]

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
    var next_lp: DeviceBuffer[dtype]
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

    # Network workspaces (passed to strategies)
    var actor_ws: DeviceBuffer[dtype]

    # Strategy workspaces
    var strat_ws: DeviceBuffer[dtype]
    var target_strat_ws: DeviceBuffer[dtype]

    # Alpha auto-tuning (SAC): small buffers for log_prob GPU→CPU transfer
    var curr_lp: DeviceBuffer[dtype]  # [BS] current log_probs from actor loss

    # Pre-filled dq buffer: constant -1/BATCH for actor policy gradient
    var dq: DeviceBuffer[dtype]

    # Gradient clipping scratch (partial sums for norm reduction)
    var grad_clip_ps: DeviceBuffer[dtype]

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
        self.raw_act = ctx.enqueue_create_buffer[dtype](MNE * Self.ACTOR_OUT)
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
        self.next_lp = ctx.enqueue_create_buffer[dtype](BS)
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

        # Network workspaces
        self.actor_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.ACTOR_WS)
        )

        # Strategy workspaces
        self.strat_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.strat_ws_size)
        )
        self.target_strat_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.target_strat_ws_size)
        )

        # Alpha auto-tuning
        self.curr_lp = ctx.enqueue_create_buffer[dtype](BS)

        # Pre-fill dq with -1/BATCH (constant policy-gradient seed)
        self.dq = ctx.enqueue_create_buffer[dtype](BS)
        var dq_host = ctx.enqueue_create_host_buffer[dtype](BS)
        for i in range(BS):
            dq_host[i] = Scalar[dtype](-1.0 / Float64(BS))
        ctx.enqueue_copy(self.dq, dq_host)

        # Gradient clipping partial sums buffer (sized for largest network)
        comptime ACTOR_PS = Self.ActorModel.PARAM_SIZE
        comptime CRITIC_PS = Self.CriticModel.PARAM_SIZE
        comptime MAX_PS = ACTOR_PS if ACTOR_PS > CRITIC_PS else CRITIC_PS
        comptime MAX_BLOCKS = (MAX_PS + TPB - 1) // TPB
        self.grad_clip_ps = ctx.enqueue_create_buffer[dtype](MAX_BLOCKS)

        # Twin critic extra
        self.nq2 = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.q2_out = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_OUT)
        self.q2_cache = ctx.enqueue_create_buffer[dtype](BS * Self.CRITIC_CS)
        self.critic2_ws = ctx.enqueue_create_buffer[dtype](
            max(1, BS * Self.CRITIC_WS)
        )

    # =========================================================================
    # LayoutTensor views (zero-copy pointer casts over DeviceBuffers)
    # =========================================================================

    fn obs_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
        ](self.s_obs.unsafe_ptr())

    fn nobs_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.OBS), MutAnyOrigin
        ](self.s_nobs.unsafe_ptr())

    fn act_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](self.s_act.unsafe_ptr())

    fn rew_view[
        BS: Int
    ](self) -> LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin]:
        return LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            self.s_rew.unsafe_ptr()
        )

    fn done_view[
        BS: Int
    ](self) -> LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin]:
        return LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            self.s_done.unsafe_ptr()
        )

    fn next_act_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.ACTIONS), MutAnyOrigin
        ](self.next_act.unsafe_ptr())

    fn next_lp_view[
        BS: Int
    ](self) -> LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin]:
        return LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            self.next_lp.unsafe_ptr()
        )

    fn next_ci_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](self.next_ci.unsafe_ptr())

    fn next_q_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](self.next_q.unsafe_ptr())

    fn nq2_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](self.nq2.unsafe_ptr())

    fn targets_view[
        BS: Int
    ](self) -> LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin]:
        return LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            self.targets.unsafe_ptr()
        )

    fn ci_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](self.ci.unsafe_ptr())

    fn q_out_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](self.q_out.unsafe_ptr())

    fn q_cache_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
        ](self.q_cache.unsafe_ptr())

    fn q_grad_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](self.q_grad.unsafe_ptr())

    fn d_ci_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_IN), MutAnyOrigin
        ](self.d_ci.unsafe_ptr())

    fn q2_out_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_OUT), MutAnyOrigin
        ](self.q2_out.unsafe_ptr())

    fn q2_cache_view[
        BS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(BS, Self.CRITIC_CS), MutAnyOrigin
        ](self.q2_cache.unsafe_ptr())

    # =========================================================================
    # GPUOffPolicyState trait
    # =========================================================================

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
    profile: Int = 0,
    L: Logger = NoOpLogger,
](OffPolicyContinuousAgent & GPUOffPolicyAgent & Checkpointable):
    """Generic off-policy agent. Supports DDPG, TD3, and SAC via Config strategies.
    """

    # Dimensions from Config's Model types
    comptime OBS: Int = Self.Config.ActorModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.action_dim
    comptime ACTOR_OUT: Int = Self.Config.ActorModel.OUT_DIM
    comptime BATCH: Int = Self.Config.batch_size
    comptime CRITIC_IN: Int = Self.Config.CriticModel.IN_DIM
    comptime CRITIC_OUT: Int = Self.Config.CriticModel.OUT_DIM
    comptime ACTOR_CS: Int = Self.Config.ActorModel.CACHE_SIZE
    comptime CRITIC_CS: Int = Self.Config.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.Config.ActorModel, Self.Config.ActorOpt]
    comptime CriticNet = Network[Self.Config.CriticModel, Self.Config.CriticOpt]

    # Strategy workspace sizes
    comptime _AL_WS: Int = Self.Config.ActorLoss.ws_size[
        Self.BATCH,
        Self.OBS,
        Self.ACTIONS,
        Self.ACTOR_OUT,
        Self.ACTOR_CS,
        Self.CRITIC_IN,
        Self.CRITIC_OUT,
        Self.CRITIC_CS,
    ]()
    comptime _TA_WS: Int = Self.Config.TargetAction.ws_size[
        Self.BATCH,
        Self.ACTIONS,
        Self.ACTOR_OUT,
    ]()

    # Workspace offsets — Region 1: target computation
    comptime _O_NEXT_ACT: Int = 0
    comptime _O_NEXT_LP: Int = Self._O_NEXT_ACT + Self.BATCH * Self.ACTIONS
    comptime _O_NEXT_CI: Int = Self._O_NEXT_LP + Self.BATCH
    comptime _O_NEXT_Q1: Int = Self._O_NEXT_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_NEXT_Q2: Int = Self._O_NEXT_Q1 + Self.BATCH * Self.CRITIC_OUT
    comptime _O_TARGETS: Int = Self._O_NEXT_Q2 + Self.BATCH * Self.CRITIC_OUT

    # Region 2: critic update
    comptime _O_CI: Int = Self._O_TARGETS + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q1_OUT: Int = Self._O_CI + Self.BATCH * Self.CRITIC_IN
    comptime _O_Q1_CACHE: Int = Self._O_Q1_OUT + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q2_OUT: Int = Self._O_Q1_CACHE + Self.BATCH * Self.CRITIC_CS
    comptime _O_Q2_CACHE: Int = Self._O_Q2_OUT + Self.BATCH * Self.CRITIC_OUT
    comptime _O_Q_GRAD: Int = Self._O_Q2_CACHE + Self.BATCH * Self.CRITIC_CS
    comptime _O_D_CI: Int = Self._O_Q_GRAD + Self.BATCH * Self.CRITIC_OUT

    # Region 3: strategy workspace (shared between ActorLoss and TargetAction)
    comptime _O_STRAT_WS: Int = Self._O_D_CI + Self.BATCH * Self.CRITIC_IN

    # CPU state type
    comptime CPUStateType = GenericCPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.buffer_capacity,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.action_dim,
        Self.Config.batch_size,
        Self.Config.NUM_CRITICS,
        Self.Config.HAS_TARGET_ACTOR,
        Self._AL_WS,
        Self._TA_WS,
    ]

    # GPUOffPolicyAgent required comptime constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACTIONS
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = 64  # default, can be overridden via Config
    comptime _AL_WS_SIZE: Int = Self.Config.ActorLoss.ws_size[
        Self.BATCH,
        Self.OBS,
        Self.ACTIONS,
        Self.ACTOR_OUT,
        Self.ACTOR_CS,
        Self.CRITIC_IN,
        Self.CRITIC_OUT,
        Self.CRITIC_CS,
    ]()
    comptime _TA_WS_SIZE: Int = Self.Config.TargetAction.ws_size[
        Self.BATCH,
        Self.ACTIONS,
        Self.ACTOR_OUT,
    ]()
    comptime GPUStateType = GenericGPUState[
        Self.Config.ActorModel,
        Self.Config.ActorOpt,
        Self.Config.CriticModel,
        Self.Config.CriticOpt,
        Self.Config.buffer_capacity,
        Self.Config.ActorModel.IN_DIM,
        Self.Config.action_dim,
        Self.Config.batch_size,
        64,  # max_n_envs
        Self.Config.NUM_CRITICS,
        Self._AL_WS_SIZE,
        Self._TA_WS_SIZE,
    ]

    # Persistent CPU state (for evaluate() after train/train_gpu)
    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64
    var noise_std: Float64

    # Schedule
    var policy_delay: Int
    var update_count: Int

    # TD3-specific (kept for GPU path)
    var target_noise_std: Float64
    var target_noise_clip: Float64

    # SAC-specific (always allocated, only used when Config.ActorLoss.HAS_ALPHA)
    var alpha: Float64
    var log_alpha: Float64
    var target_entropy: Float64
    var auto_alpha: Bool
    var alpha_lr: Float64
    var alpha_adam_m: Float64
    var alpha_adam_v: Float64
    var alpha_adam_t: Int

    # Training state
    var total_steps: Int
    var train_step_count: Int
    var target_total_steps: Int
    var checkpoint_every: Int
    var checkpoint_path: String

    # Profiling (compile-time gated)
    var train_timer: PerfTimer[Self.profile >= 1]

    # Note: L3 per-layer profiling slots not available through trait-bounded Config.
    # Old per-algorithm agents support L3 via concrete Model.register_forward_slots().

    # Gradient clipping
    var max_grad_norm: Float64

    # Diagnostic logging
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        noise_std: Float64 = -1.0,
        noise_std_min: Float64 = 0.01,
        noise_decay: Float64 = 0.995,
        policy_delay: Int = -1,
        target_noise_std: Float64 = 0.2,
        target_noise_clip: Float64 = 0.5,
        auto_alpha: Bool = True,
        alpha: Float64 = 0.2,
        alpha_lr: Float64 = 0.0003,
        max_grad_norm: Float64 = 40.0,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        target_total_steps: Int = 0,
    ):
        self.state = Self.CPUStateType()
        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        # Use Config.Explore.INITIAL_STD if noise_std not explicitly set
        self.noise_std = (
            Self.Config.Explore.INITIAL_STD if noise_std < 0.0 else noise_std
        )
        # Use Config.Schedule.DEFAULT_POLICY_DELAY if policy_delay not explicitly set
        self.policy_delay = (
            Self.Config.Schedule.DEFAULT_POLICY_DELAY if policy_delay
            < 0 else policy_delay
        )
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.update_count = 0

        # SAC alpha fields
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.log_alpha = log(alpha)
        self.target_entropy = -Float64(Self.ACTIONS)
        self.alpha_lr = alpha_lr
        self.alpha_adam_m = 0.0
        self.alpha_adam_v = 0.0
        self.alpha_adam_t = 0

        self.total_steps = 0
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

        # Profiling
        self.train_timer = PerfTimer[Self.profile >= 1]()
        comptime if Self.profile >= 2:
            _ = self.train_timer.add_slot("sample_batch")  # 0
            _ = self.train_timer.add_slot("target_actions")  # 1
            _ = self.train_timer.add_slot("td_targets")  # 2
            _ = self.train_timer.add_slot("critic_update")  # 3
            comptime if Self.Config.NUM_CRITICS == 2:
                _ = self.train_timer.add_slot("critic2_update")  # 4
            _ = self.train_timer.add_slot("actor_update")  # 4 or 5

        # Note: L3 per-layer profiling (profile >= 3) requires concrete Model types
        # with register_forward/backward_slots methods (on Sequential, not on Model trait).
        # This is available when using the old per-algorithm agents.
        # The generic agent supports L1 + L2 profiling.

        # Logging
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    # =========================================================================
    # OffPolicyContinuousAgent trait
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn select_action[
        d: DType
    ](mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[d]]) -> List[
        Scalar[d]
    ]:
        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())

        comptime if Self.Config.Explore.IS_STOCHASTIC:
            # SAC: stochastic sampling via reparameterization
            var out_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
                uninitialized=True
            )
            var out_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
            ](out_arr.unsafe_ptr())
            var p = cpu_state.actor.online.params_view()
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
        else:
            # DDPG/TD3: deterministic actor + Gaussian noise
            var act_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
                uninitialized=True
            )
            var act_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
            ](act_arr.unsafe_ptr())
            var p = cpu_state.actor.online.params_view()
            Self.ActorNet.forward[1](obs_t, act_t, p)
            var raw = List[Scalar[d]](capacity=Self.ACTIONS)
            for i in range(Self.ACTIONS):
                raw.append(Scalar[d](Float64(act_arr[i]) * self.action_scale))
            return Self.Config.Explore.explore[d](
                raw, self.action_scale, self.noise_std
            )

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
            normalized.append(Scalar[d](Float64(action[i]) / self.action_scale))
        cpu_state.store[d](obs, normalized, reward, next_obs, done)
        self.total_steps += 1

    fn do_cpu_train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
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
        var b_rew = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var b_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_done = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        cpu_state.buffer.sample[Self.BATCH](b_obs, b_act, b_rew, b_next, b_done)

        var ws = cpu_state.ws.unsafe_ptr()

        # Phase 2: Target actions — delegate to Config.TargetAction
        # Select correct actor params: target if HAS_TARGET_ACTOR, else online
        # Select target or online actor params for target action computation
        # SAC uses online (no target actor), DDPG/TD3 use target
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_next.unsafe_ptr())
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](ws + Self._O_NEXT_ACT)
        comptime if Self.Config.HAS_TARGET_ACTOR:
            Self.Config.TargetAction.compute_cpu[
                Self.BATCH,
                Self.ACTIONS,
                Self.Config.ActorModel,
                Self.Config.ActorOpt,
            ](
                next_obs_t,
                next_act_t,
                ws + Self._O_NEXT_LP,
                cpu_state.actor.target.params_view(),
                ws + Self._O_STRAT_WS,
            )
        else:
            Self.Config.TargetAction.compute_cpu[
                Self.BATCH,
                Self.ACTIONS,
                Self.Config.ActorModel,
                Self.Config.ActorOpt,
            ](
                next_obs_t,
                next_act_t,
                ws + Self._O_NEXT_LP,
                cpu_state.actor.online.params_view(),
                ws + Self._O_STRAT_WS,
            )

        # Phase 2b: Concat next_obs + next_act → next_ci
        _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
            ws + Self._O_NEXT_CI, b_next.unsafe_ptr(), ws + Self._O_NEXT_ACT
        )
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_NEXT_CI)

        # Forward critic1 target
        var next_q1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_NEXT_Q1)
        var p_ct = cpu_state.critic.target.params_view()
        Self.CriticNet.forward[Self.BATCH](next_ci_t, next_q1_t, p_ct)

        # Forward critic2 target (only when twin critics)
        comptime if Self.Config.NUM_CRITICS == 2:
            var next_q2_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CRITIC_OUT),
                MutAnyOrigin,
            ](ws + Self._O_NEXT_Q2)
            var p_c2t = cpu_state.critic2.target.params_view()
            Self.CriticNet.forward[Self.BATCH](next_ci_t, next_q2_t, p_c2t)

        # Phase 2c: TD targets — delegate to Config.TargetValue
        var q1_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](ws + Self._O_NEXT_Q1)
        var q2_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](ws + Self._O_NEXT_Q2)
        var lp_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](ws + Self._O_NEXT_LP)
        var rew_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](b_rew.unsafe_ptr())
        var done_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](b_done.unsafe_ptr())
        var tgt_tv = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](ws + Self._O_TARGETS)
        Self.Config.TargetValue.compute_cpu[Self.BATCH](
            q1_tv,
            q2_tv,
            lp_tv,
            rew_tv,
            done_tv,
            tgt_tv,
            self.gamma,
            self.alpha,
        )

        # Phase 3: Critic update
        _concat_obs_act[Self.BATCH, Self.OBS, Self.ACTIONS, Self.CRITIC_IN](
            ws + Self._O_CI, b_obs.unsafe_ptr(), b_act.unsafe_ptr()
        )
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_CI)
        var tgt_p = ws + Self._O_TARGETS

        # --- Critic 1 update ---
        var q1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_Q1_OUT)
        var q1_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_CS), MutAnyOrigin
        ](ws + Self._O_Q1_CACHE)
        var p_c = cpu_state.critic.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q1_t, p_c, q1_cache_t
        )

        var q_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_OUT), MutAnyOrigin
        ](ws + Self._O_Q_GRAD)
        var q1o_p = ws + Self._O_Q1_OUT
        var qg_p = ws + Self._O_Q_GRAD
        var critic_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = q1o_p[b] - tgt_p[b]
            critic_loss += Float64(td_err * td_err)
            qg_p[b] = Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
        critic_loss /= Float64(Self.BATCH)

        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](ws + Self._O_D_CI)
        var g_c = cpu_state.critic.grads_view()
        cpu_state.critic.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q_grad_t, d_ci_t, p_c, q1_cache_t, g_c
        )
        cpu_state.critic.optimizer_step()

        # --- Critic 2 update (twin critics only) ---
        comptime if Self.Config.NUM_CRITICS == 2:
            var q2_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CRITIC_OUT),
                MutAnyOrigin,
            ](ws + Self._O_Q2_OUT)
            var q2_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CRITIC_CS),
                MutAnyOrigin,
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
                qg_p[b] = (
                    Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
                )
            critic2_loss /= Float64(Self.BATCH)

            var g_c2 = cpu_state.critic2.grads_view()
            cpu_state.critic2.zero_grads()
            Self.CriticNet.backward[Self.BATCH](
                q_grad_t, d_ci_t, p_c2, q2_cache_t, g_c2
            )
            cpu_state.critic2.optimizer_step()
            critic_loss = (critic_loss + critic2_loss) / 2.0

        # Diagnostic logging
        if self.logger and (
            self.diag_every <= 0
            or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                self.logger[].log_scalar("loss", critic_loss, step)
                self.logger[].log_scalar(
                    "explore_rate", self.get_explore_rate(), step
                )
                comptime if Self.Config.ActorLoss.HAS_ALPHA:
                    self.logger[].log_scalar("alpha", self.alpha, step)
            except:
                pass

        # Phase 4: Actor update — delegate to Config.ActorLoss
        if Self.Config.Schedule.should_update_actor(
            self.update_count, self.policy_delay
        ):
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
            ](b_obs.unsafe_ptr())
            var a_grads = cpu_state.actor.online.grads_view()
            var c_grads = cpu_state.critic.online.grads_view()
            var c2_grads = cpu_state.critic.online.grads_view()
            var c2_params = cpu_state.critic.online.params_view()
            comptime if Self.Config.NUM_CRITICS == 2:
                c2_grads = cpu_state.critic2.online.grads_view()
                c2_params = cpu_state.critic2.online.params_view()
            var mean_lp = Self.Config.ActorLoss.update_actor_cpu[
                Self.BATCH,
                Self.ACTIONS,
                Self.Config.ActorModel,
                Self.Config.ActorOpt,
                Self.Config.CriticModel,
                Self.Config.CriticOpt,
            ](
                obs_t,
                cpu_state.actor.online.params_view(),
                a_grads,
                cpu_state.critic.online.params_view(),
                c_grads,
                c2_params,
                c2_grads,
                ws + Self._O_STRAT_WS,
                self.alpha,
            )
            cpu_state.actor.optimizer_step()

            # Alpha update (SAC only)
            comptime if Self.Config.ActorLoss.HAS_ALPHA:
                if self.auto_alpha:
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
                    self.log_alpha -= (
                        self.alpha_lr * m_hat / (sqrt(v_hat) + eps)
                    )
                    self.alpha = exp(self.log_alpha)

        # Phase 5: Soft update targets — delegate to Config.Schedule
        if Self.Config.Schedule.should_update_targets(
            self.update_count, self.policy_delay
        ):
            comptime if Self.Config.HAS_TARGET_ACTOR:
                cpu_state.actor.soft_update(self.tau)
            cpu_state.critic.soft_update(self.tau)
            comptime if Self.Config.NUM_CRITICS == 2:
                cpu_state.critic2.soft_update(self.tau)

        self.train_step_count += 1
        return critic_loss

    fn decay_explore(mut self) -> None:
        Self.Config.Explore.decay(self.noise_std)

    fn get_explore_rate(self) -> Float64:
        comptime if Self.Config.Explore.IS_STOCHASTIC:
            return self.alpha
        else:
            return Self.Config.Explore.get_rate(self.noise_std)

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

        comptime if Self.Config.Explore.IS_STOCHASTIC:
            # SAC: deterministic action = tanh(mean)
            var out_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
                uninitialized=True
            )
            var out_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
            ](out_arr.unsafe_ptr())
            var p = cpu_state.actor.online.params_view()
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
        else:
            # DDPG/TD3: deterministic actor output
            var act_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
                uninitialized=True
            )
            var act_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
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
        """Upload CPU network weights to GPU."""
        gpu_state.actor.upload_from(self.state.actor, ctx)
        gpu_state.critic.upload_from(self.state.critic, ctx)
        comptime if Self.Config.NUM_CRITICS == 2:
            gpu_state.critic2.upload_from(self.state.critic2, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.actor.download_to(self.state.actor, ctx)
        gpu_state.critic.download_to(self.state.critic, ctx)
        comptime if Self.Config.NUM_CRITICS == 2:
            gpu_state.critic2.download_to(self.state.critic2, ctx)

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
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.raw_act.unsafe_ptr())
        var p = gpu_state.actor.online.params_view()

        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.inf_ws
        )

        var act_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rng_seed_s = Scalar[DType.uint32](
            UInt32(self.total_steps) * UInt32(Self.ACTIONS)
        )

        comptime if Self.Config.Explore.IS_STOCHASTIC:
            # SAC: stochastic sample from actor output
            comptime BLOCKS = (N_ENVS + TPB - 1) // TPB
            comptime sac_explore_k = sac_sample_actions_kernel[
                dtype, N_ENVS, Self.ACTIONS, Self.ACTOR_OUT
            ]
            ctx.enqueue_function[sac_explore_k, sac_explore_k](
                act_t,
                raw_t,
                Scalar[dtype](self.action_scale),
                Scalar[dtype](-5.0),
                Scalar[dtype](2.0),
                rng_seed_s,
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            # DDPG/TD3: deterministic + Gaussian noise
            comptime BLOCKS = (N_ENVS * Self.ACTIONS + TPB - 1) // TPB
            comptime ddpg_explore_k = ddpg_exploration_kernel[
                dtype, N_ENVS, Self.ACTIONS
            ]
            ctx.enqueue_function[ddpg_explore_k, ddpg_explore_k](
                act_t,
                raw_t,
                Scalar[dtype](self.noise_std),
                Scalar[dtype](self.action_scale),
                rng_seed_s,
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """GPU training step delegating to strategies."""
        comptime BS = Self.BATCH
        comptime ELEM_BLOCKS = (BS * Self.CRITIC_IN + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (BS + TPB - 1) // TPB
        comptime concat_k = concat_obs_action_kernel[
            dtype, BS, Self.OBS, Self.ACTIONS, Self.CRITIC_IN
        ]
        comptime mse_grad_k = td_mse_grad_kernel[dtype, BS, Self.CRITIC_OUT]

        # Phase 1: Sample batch
        self.train_step_count += 1
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

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_mark(ctx)

        var obs_t = gpu_state.obs_view[BS]()
        var nobs_t = gpu_state.nobs_view[BS]()
        var act_t = gpu_state.act_view[BS]()
        var rew_t = gpu_state.rew_view[BS]()
        var done_t = gpu_state.done_view[BS]()
        var p_actor_t = gpu_state.actor.target.params_view()
        var p_critic_t = gpu_state.critic.target.params_view()
        var p_actor = gpu_state.actor.online.params_view()
        var p_critic = gpu_state.critic.online.params_view()
        var rng_seed = UInt32(self.train_step_count) * UInt32(BS * Self.ACTIONS + 7)

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(0, ctx)
            self.train_timer.mark()

        # Phase 2: Target actions — delegate to Config.TargetAction
        var next_act_t = gpu_state.next_act_view[BS]()
        var next_lp_t = gpu_state.next_lp_view[BS]()
        comptime if Self.Config.HAS_TARGET_ACTOR:
            Self.Config.TargetAction.compute_gpu[
                BS,
                Self.ACTIONS,
                Self.Config.ActorModel,
                Self.Config.ActorOpt,
            ](
                ctx,
                nobs_t,
                next_act_t,
                next_lp_t,
                p_actor_t,
                gpu_state.actor_ws,
                gpu_state.target_strat_ws,
                rng_seed,
            )
        else:
            Self.Config.TargetAction.compute_gpu[
                BS,
                Self.ACTIONS,
                Self.Config.ActorModel,
                Self.Config.ActorOpt,
            ](
                ctx,
                nobs_t,
                next_act_t,
                next_lp_t,
                p_actor,
                gpu_state.actor_ws,
                gpu_state.target_strat_ws,
                rng_seed,
            )

        # Phase 2b: Concat next_obs + next_act → next_ci, forward target critics
        var next_ci_t = gpu_state.next_ci_view[BS]()
        ctx.enqueue_function[concat_k, concat_k](
            next_ci_t,
            nobs_t,
            next_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        var next_q_t = gpu_state.next_q_view[BS]()
        Self.CriticNet.forward_gpu[BS](
            ctx, next_ci_t, next_q_t, p_critic_t, gpu_state.critic_ws
        )
        comptime if Self.Config.NUM_CRITICS == 2:
            var nq2_t = gpu_state.nq2_view[BS]()
            var p_c2t = gpu_state.critic2.target.params_view()
            Self.CriticNet.forward_gpu[BS](
                ctx, next_ci_t, nq2_t, p_c2t, gpu_state.critic2_ws
            )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(1, ctx)
            self.train_timer.mark()

        # Phase 2c: TD targets — delegate to Config.TargetValue
        var nq1_flat = LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            gpu_state.next_q.unsafe_ptr()
        )
        var nq2_flat = LayoutTensor[dtype, Layout.row_major(BS), MutAnyOrigin](
            gpu_state.nq2.unsafe_ptr()
        )
        var targets_t = gpu_state.targets_view[BS]()
        Self.Config.TargetValue.compute_gpu[BS](
            ctx,
            nq1_flat,
            nq2_flat,
            next_lp_t,
            rew_t,
            done_t,
            targets_t,
            self.gamma,
            self.alpha,
        )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(2, ctx)
            self.train_timer.mark()

        # Phase 3: Critic update
        var ci_t = gpu_state.ci_view[BS]()
        var q_t = gpu_state.q_out_view[BS]()
        var q_cache_t = gpu_state.q_cache_view[BS]()
        var q_grad_t = gpu_state.q_grad_view[BS]()
        var d_ci_t = gpu_state.d_ci_view[BS]()

        ctx.enqueue_function[concat_k, concat_k](
            ci_t,
            obs_t,
            act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )
        Self.CriticNet.forward_gpu_with_cache[BS](
            ctx,
            ci_t,
            q_t,
            p_critic,
            q_cache_t,
            gpu_state.critic_ws,
        )
        ctx.enqueue_function[mse_grad_k, mse_grad_k](
            q_grad_t,
            q_t,
            targets_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        var g_critic = gpu_state.critic.online.grads_view()
        gpu_state.critic.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BS](
            ctx,
            q_grad_t,
            d_ci_t,
            p_critic,
            q_cache_t,
            g_critic,
            gpu_state.critic_ws,
        )
        gpu_state.critic.online.optimizer_step(ctx)

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(3, ctx)
            self.train_timer.mark()

        # Critic2 update (twin critics only)
        comptime if Self.Config.NUM_CRITICS == 2:
            var q2_out_t = gpu_state.q2_out_view[BS]()
            var q2_cache_t = gpu_state.q2_cache_view[BS]()
            var p_c2 = gpu_state.critic2.online.params_view()
            Self.CriticNet.forward_gpu_with_cache[BS](
                ctx,
                ci_t,
                q2_out_t,
                p_c2,
                q2_cache_t,
                gpu_state.critic2_ws,
            )
            ctx.enqueue_function[mse_grad_k, mse_grad_k](
                q_grad_t,
                q2_out_t,
                targets_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            var g_c2 = gpu_state.critic2.online.grads_view()
            gpu_state.critic2.online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BS](
                ctx,
                q_grad_t,
                d_ci_t,
                p_c2,
                q2_cache_t,
                g_c2,
                gpu_state.critic2_ws,
            )
            gpu_state.critic2.online.optimizer_step(ctx)

            comptime if Self.profile >= 2:
                self.train_timer.sync_and_accumulate(4, ctx)
                self.train_timer.mark()

        # GPU Diagnostic logging (every diag_every train steps)
        if self.logger and self.diag_every > 0 and self.train_step_count % self.diag_every == 0:
            try:
                # Sync Q-values and targets from GPU to CPU for diagnostics
                var diag_q_host = ctx.enqueue_create_host_buffer[dtype](BS)
                var diag_tgt_host = ctx.enqueue_create_host_buffer[dtype](BS)
                var diag_rew_host = ctx.enqueue_create_host_buffer[dtype](BS)
                var diag_done_host = ctx.enqueue_create_host_buffer[dtype](BS)
                var diag_act_host = ctx.enqueue_create_host_buffer[dtype](
                    BS * Self.ACTIONS
                )
                var diag_nq_host = ctx.enqueue_create_host_buffer[dtype](BS)
                ctx.enqueue_copy(diag_q_host, gpu_state.q_out)
                ctx.enqueue_copy(diag_tgt_host, gpu_state.targets)
                ctx.enqueue_copy(diag_rew_host, gpu_state.s_rew)
                ctx.enqueue_copy(diag_done_host, gpu_state.s_done)
                ctx.enqueue_copy(diag_act_host, gpu_state.s_act)
                ctx.enqueue_copy(diag_nq_host, gpu_state.next_q)
                ctx.synchronize()

                # Compute mean Q, mean target, critic loss, mean reward
                var mean_q: Float64 = 0.0
                var mean_tgt: Float64 = 0.0
                var mean_rew: Float64 = 0.0
                var mean_done: Float64 = 0.0
                var critic_loss: Float64 = 0.0
                var mean_nq: Float64 = 0.0
                var mean_abs_act: Float64 = 0.0
                for b in range(BS):
                    var q_val = Float64(diag_q_host[b])
                    var tgt_val = Float64(diag_tgt_host[b])
                    mean_q += q_val
                    mean_tgt += tgt_val
                    mean_rew += Float64(diag_rew_host[b])
                    mean_done += Float64(diag_done_host[b])
                    mean_nq += Float64(diag_nq_host[b])
                    critic_loss += (q_val - tgt_val) * (q_val - tgt_val)
                for i in range(BS * Self.ACTIONS):
                    var a = Float64(diag_act_host[i])
                    mean_abs_act += (a if a >= 0.0 else -a)
                mean_q /= Float64(BS)
                mean_tgt /= Float64(BS)
                mean_rew /= Float64(BS)
                mean_done /= Float64(BS)
                mean_nq /= Float64(BS)
                critic_loss /= Float64(BS)
                mean_abs_act /= Float64(BS * Self.ACTIONS)

                var step = self.train_step_count
                self.logger[].log_scalar("critic_loss", critic_loss, step)
                self.logger[].log_scalar("mean_q", mean_q, step)
                self.logger[].log_scalar("mean_target", mean_tgt, step)
                self.logger[].log_scalar("mean_reward", mean_rew, step)
                self.logger[].log_scalar("mean_next_q", mean_nq, step)
                self.logger[].log_scalar("mean_done", mean_done, step)
                self.logger[].log_scalar("mean_abs_action", mean_abs_act, step)
                comptime if Self.Config.ActorLoss.HAS_ALPHA:
                    self.logger[].log_scalar("alpha", self.alpha, step)
            except:
                pass

        # Phase 4: Actor update — delegate to Config.ActorLoss
        self.update_count += 1
        if Self.Config.Schedule.should_update_actor(
            self.update_count, self.policy_delay
        ):
            gpu_state.actor.online.zero_grads(ctx)
            gpu_state.critic.online.zero_grads(ctx)
            var a_grads = gpu_state.actor.online.grads_view()
            var c_grads = gpu_state.critic.online.grads_view()
            # For twin critics, zero and pass critic2; otherwise reuse critic1
            var c2_grads = c_grads
            var p_c2 = p_critic
            var c2_ws = gpu_state.critic_ws
            comptime if Self.Config.NUM_CRITICS == 2:
                gpu_state.critic2.online.zero_grads(ctx)
                c2_grads = gpu_state.critic2.online.grads_view()
                p_c2 = gpu_state.critic2.online.params_view()
                c2_ws = gpu_state.critic2_ws
            _ = Self.Config.ActorLoss.update_actor_gpu[
                BS,
                Self.ACTIONS,
                Self.Config.ActorModel,
                Self.Config.ActorOpt,
                Self.Config.CriticModel,
                Self.Config.CriticOpt,
            ](
                ctx,
                obs_t,
                p_actor,
                a_grads,
                p_critic,
                c_grads,
                p_c2,
                c2_grads,
                gpu_state.actor_ws,
                gpu_state.critic_ws,
                c2_ws,
                gpu_state.strat_ws,
                gpu_state.dq,
                self.alpha,
                UInt32(self.train_step_count),
            )
            # Log actor grad norm before optimizer step overwrites grads
            if self.logger and self.diag_every > 0 and self.train_step_count % self.diag_every == 0:
                try:
                    comptime A_PS = Self.Config.ActorModel.PARAM_SIZE
                    var ag_host = ctx.enqueue_create_host_buffer[dtype](A_PS)
                    ctx.enqueue_copy(
                        ag_host, gpu_state.actor.online.grads_buf
                    )
                    ctx.synchronize()
                    var grad_norm: Float64 = 0.0
                    for i in range(A_PS):
                        var g = Float64(ag_host[i])
                        grad_norm += g * g
                    grad_norm = sqrt(grad_norm)
                    self.logger[].log_scalar(
                        "actor_grad_norm",
                        grad_norm,
                        self.train_step_count,
                    )
                except:
                    pass

            # Clip actor gradients
            if self.max_grad_norm > 0.0:
                comptime A_PS = Self.Config.ActorModel.PARAM_SIZE
                comptime A_BLOCKS = (A_PS + TPB - 1) // TPB
                var ps_t = LayoutTensor[
                    dtype, Layout.row_major(A_BLOCKS), MutAnyOrigin
                ](gpu_state.grad_clip_ps.unsafe_ptr())

                @always_inline
                fn run_norm(
                    p: LayoutTensor[
                        dtype, Layout.row_major(A_BLOCKS), MutAnyOrigin
                    ],
                    g: LayoutTensor[
                        dtype, Layout.row_major(A_PS), MutAnyOrigin
                    ],
                ):
                    gradient_norm_kernel[dtype, A_PS, A_BLOCKS, TPB](p, g)

                ctx.enqueue_function[run_norm, run_norm](
                    ps_t,
                    a_grads,
                    grid_dim=(A_BLOCKS,),
                    block_dim=(TPB,),
                )

                var mgn = Scalar[dtype](self.max_grad_norm)

                @always_inline
                fn run_clip(
                    g: LayoutTensor[
                        dtype, Layout.row_major(A_PS), MutAnyOrigin
                    ],
                    p: LayoutTensor[
                        dtype, Layout.row_major(A_BLOCKS), MutAnyOrigin
                    ],
                    m: Scalar[dtype],
                ):
                    gradient_reduce_apply_fused_kernel[
                        dtype, A_PS, A_BLOCKS, TPB
                    ](g, p, m)

                ctx.enqueue_function[run_clip, run_clip](
                    a_grads,
                    ps_t,
                    mgn,
                    grid_dim=(A_BLOCKS,),
                    block_dim=(TPB,),
                )

            gpu_state.actor.online.optimizer_step(ctx)

            # Alpha auto-tuning (SAC only): copy log_probs GPU→CPU, Adam update
            comptime if Self.Config.ActorLoss.HAS_ALPHA:
                if self.auto_alpha:
                    comptime LP_OFF = Self.Config.ActorLoss.gpu_lp_offset[
                        BS,
                        Self.ACTIONS,
                        Self.ACTOR_OUT,
                        Self.ACTOR_CS,
                    ]()
                    # Copy log_probs from strat_ws[LP_OFF] → curr_lp via kernel
                    var src_lp = LayoutTensor[
                        dtype, Layout.row_major(BS), MutAnyOrigin
                    ](gpu_state.strat_ws.unsafe_ptr() + LP_OFF)
                    var dst_lp = LayoutTensor[
                        dtype, Layout.row_major(BS), MutAnyOrigin
                    ](gpu_state.curr_lp.unsafe_ptr())

                    @always_inline
                    fn copy_lp(
                        d: LayoutTensor[
                            dtype, Layout.row_major(BS), MutAnyOrigin
                        ],
                        s: LayoutTensor[
                            dtype, Layout.row_major(BS), MutAnyOrigin
                        ],
                    ):
                        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if i < BS:
                            d.ptr[i] = s.ptr[i]

                    ctx.enqueue_function[copy_lp, copy_lp](
                        dst_lp,
                        src_lp,
                        grid_dim=(BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # GPU → CPU copy
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
                    self.log_alpha -= (
                        self.alpha_lr * m_hat / (sqrt(v_hat) + eps)
                    )
                    self.alpha = exp(self.log_alpha)

            comptime if Self.profile >= 2:
                comptime ACTOR_SLOT = 4 if Self.Config.NUM_CRITICS == 1 else 5
                self.train_timer.sync_and_accumulate(ACTOR_SLOT, ctx)

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        if Self.Config.Schedule.should_update_targets(
            self.update_count, self.policy_delay
        ):
            comptime if Self.Config.HAS_TARGET_ACTOR:
                gpu_state.actor.soft_update(self.tau, ctx)
            gpu_state.critic.soft_update(self.tau, ctx)
            comptime if Self.Config.NUM_CRITICS == 2:
                gpu_state.critic2.soft_update(self.tau, ctx)

    fn decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        """No-op for DDPG/TD3 (Gaussian noise decay is per-episode, not per-step).
        """
        pass

    # Checkpointable — saves agent hyperparameters and training state.
    # Network weights require save_cpu_state(cpu_state, path) separately
    # because the Checkpointable trait doesn't include state access.
    fn save_checkpoint(self, path: String) raises -> None:
        from mojo_rl.nn.checkpoint import (
            write_checkpoint_header,
            write_metadata_section,
            save_checkpoint_file,
        )

        var content = write_checkpoint_header("generic_offpolicy", 0, 0)
        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("action_scale=" + String(self.action_scale))
        metadata.append("noise_std=" + String(self.noise_std))
        metadata.append("policy_delay=" + String(self.policy_delay))
        metadata.append("update_count=" + String(self.update_count))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("train_step_count=" + String(self.train_step_count))
        metadata.append("alpha=" + String(self.alpha))
        metadata.append("log_alpha=" + String(self.log_alpha))
        metadata.append("alpha_adam_t=" + String(self.alpha_adam_t))
        content += write_metadata_section(metadata)
        save_checkpoint_file(path, content)

    fn load_checkpoint(mut self, path: String) raises -> None:
        pass

    fn save_cpu_state(self, cpu_state: Self.CPUStateType, path: String) raises:
        """Save network weights and optimizer state from cpu_state.

        Saves actor (online+target) and critic(s) (online+target)
        params and optimizer states. The replay buffer is NOT saved.
        """

        var content = write_checkpoint_header(
            "generic_offpolicy_state",
            Self.Config.ActorModel.PARAM_SIZE
            + Self.Config.CriticModel.PARAM_SIZE * Self.Config.NUM_CRITICS,
            0,
        )
        content += cpu_state.actor.write_sections("actor_")
        content += cpu_state.critic.write_sections("critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            content += cpu_state.critic2.write_sections("critic2_")
        save_checkpoint_file(path, content)

    fn load_cpu_state(
        self, mut cpu_state: Self.CPUStateType, path: String
    ) raises:
        """Load network weights and optimizer state into cpu_state."""

        var content = read_checkpoint_file(path)
        cpu_state.actor.read_sections(content, "actor_")
        cpu_state.critic.read_sections(content, "critic_")
        comptime if Self.Config.NUM_CRITICS == 2:
            cpu_state.critic2.read_sections(content, "critic2_")

    # =========================================================================
    # CPU Convenience training
    # =========================================================================

    fn train[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 300,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train the agent on a continuous-action environment.

        Args:
            env: Environment implementing BoxContinuousActionEnv.
            num_episodes: Number of training episodes.
            max_steps_per_episode: Maximum steps per episode (default: 1000).
            warmup_steps: Random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 1).
            verbose: Print progress (default: False).
            print_every: Print every N episodes if verbose (default: 10).
            environment_name: Name for metrics labeling.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N train steps. 0 = every step
                when logger is set (default: 0).

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType()
        var ckpt_path = String(self.checkpoint_path)
        var metrics = run_offpolicy_continuous_train[E, Self, Self.L](
            self,
            cpu_state,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            warmup_steps=warmup_steps,
            train_every=train_every,
            checkpoint_every=self.checkpoint_every,
            checkpoint_path=ckpt_path,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            logger=logger,
        )
        self.state = cpu_state^
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^

    # =========================================================================
    # Evaluation
    # =========================================================================

    fn evaluate[
        E: BoxContinuousActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: Environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps_per_episode: Maximum steps per episode (default: 1000).
            verbose: Print per-episode results (default: False).
            render: Render the environment (default: False).
            frame_delay_ms: Delay between frames in ms (default: 16).

        Returns:
            Average reward across episodes.
        """
        var metrics = run_offpolicy_continuous_eval(
            self,
            self.state,
            env,
            num_episodes=num_episodes,
            max_steps=max_steps_per_episode,
            verbose=verbose,
            render=render,
            frame_delay_ms=frame_delay_ms,
        )
        return metrics.mean_reward()

    # =========================================================================
    # GPU Training
    # =========================================================================

    fn train_gpu[
        E: GPUContinuousEnv,
        CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train using GPU-accelerated training loop.

        After training, CPU state holds the trained weights so evaluate()
        works immediately.

        Args:
            ctx: GPU device context.
            num_steps: Total env transitions across all parallel envs.
            warmup_steps: Transitions before training starts (default: 1000).
            verbose: Print progress (default: False).
            print_every: Print interval in transitions (default: 50000).
            environment_name: Name for metrics labeling.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N train steps (default: 0).

        Returns:
            TrainingMetrics with episode-level statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var timer = PerfTimer[Self.profile >= 1]()
        comptime if Self.profile >= 1:
            _ = timer.add_slot("copy_prev_obs")
            _ = timer.add_slot("select_actions")
            _ = timer.add_slot("env_step")
            _ = timer.add_slot("buffer_store")
            _ = timer.add_slot("episode_tracking")
            _ = timer.add_slot("reset")
            _ = timer.add_slot("train_step")
            _ = timer.add_slot("gpu_cpu_sync")

        var ckpt_every = self.checkpoint_every
        var ckpt_path = String(self.checkpoint_path)
        var tgt_steps = self.target_total_steps
        var metrics = run_offpolicy_continuous_train_gpu[
            E, Self, Self.profile, Self.L, CurriculumType
        ](
            self,
            ctx,
            num_steps,
            timer,
            warmup_steps=warmup_steps,
            checkpoint_every=ckpt_every,
            checkpoint_path=ckpt_path,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            logger=logger,
            target_total_steps=tgt_steps,
        )

        comptime if Self.profile >= 2:
            timer.merge_children(6, self.train_timer)
        comptime if Self.profile >= 1:
            timer.print_report(Self.Config.NAME + " GPU Profile")

        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^
