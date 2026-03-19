"""Generic DQN agent parameterized by DiscreteOffPolicyConfig.

Supports standard DQN, Double DQN, and Dueling DQN via strategy types:
  - QTargetStrat: StandardQTarget (DQN) or DoubleQTarget (Double/Dueling DQN)
  - QOutputStrat: DirectQ (standard/double) or DuelingQ (dueling architecture)

GPU support via GPUOffPolicyAgent trait + run_offpolicy_discrete_train_gpu.
"""

from std.math import exp
from std.random import random_float64
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from mojo_rl.deep_agents.core import (
    run_offpolicy_discrete_train_gpu,
    PerfTimer,
)
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model, Linear, LinearReLU, Sequential, Parallel, Conv2DReLU, FlattenLayer
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import (
    Network,
    NetworkState,
    NetworkPair,
    GPUNetworkState,
)
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)

from mojo_rl.deep_agents.core import (
    OffPolicyDiscreteState,
    OffPolicyDiscreteAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    Checkpointable,
)
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer, PrioritizedReplayBuffer
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.eval import (
    run_offpolicy_discrete_eval,
)

from .q_target import QTarget, StandardQTarget, DoubleQTarget
from .q_output import QOutput, DirectQ, DuelingQ
from .q_gradient import QGradient, ManualQGradient, AutodiffQGradient


# =============================================================================
# DiscreteOffPolicyConfig trait
# =============================================================================


trait DiscreteOffPolicyConfig:
    """Compile-time config for DQN family agents."""

    comptime NAME: String
    comptime obs_dim: Int
    comptime num_actions: Int
    comptime batch_size: Int
    comptime buffer_capacity: Int
    comptime QModel: Model
    comptime QOpt: Optimizer
    comptime QTargetStrat: QTarget
    comptime QOutputStrat: QOutput
    comptime QGradStrat: QGradient


# =============================================================================
# DQNConfig
# =============================================================================


struct DQNConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 120,
    HIDDEN2: Int = 84,
    CAP: Int = 10000,
    BS: Int = 128,
    lr: Float64 = 2.5e-4,
](DiscreteOffPolicyConfig):
    """Standard DQN config."""

    comptime NAME: String = "DQN"
    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN2],
        Linear[Self.HIDDEN2, Self.ACT],
    ]
    comptime QOpt = Adam[Self.lr]
    comptime QTargetStrat = StandardQTarget
    comptime QOutputStrat = DirectQ
    comptime QGradStrat = ManualQGradient


struct DoubleDQNConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 120,
    HIDDEN2: Int = 84,
    CAP: Int = 10000,
    BS: Int = 128,
    lr: Float64 = 2.5e-4,
](DiscreteOffPolicyConfig):
    """Double DQN config."""

    comptime NAME: String = "Double DQN"
    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN2],
        Linear[Self.HIDDEN2, Self.ACT],
    ]
    comptime QOpt = Adam[Self.lr]
    comptime QTargetStrat = DoubleQTarget
    comptime QOutputStrat = DirectQ
    comptime QGradStrat = ManualQGradient


struct DuelingDQNConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 120,
    STREAM_H: Int = 84,
    CAP: Int = 10000,
    BS: Int = 128,
    lr: Float64 = 2.5e-4,
](DiscreteOffPolicyConfig):
    """Dueling DQN config (Double DQN target + dueling output)."""

    comptime NAME: String = "Dueling DQN"
    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Sequential[
                LinearReLU[Self.HIDDEN, Self.STREAM_H], Linear[Self.STREAM_H, 1]
            ],
            Sequential[
                LinearReLU[Self.HIDDEN, Self.STREAM_H],
                Linear[Self.STREAM_H, Self.ACT],
            ],
        ],
    ]
    comptime QOpt = Adam[Self.lr]
    comptime QTargetStrat = DoubleQTarget
    comptime QOutputStrat = DuelingQ
    comptime QGradStrat = ManualQGradient


# =============================================================================
# DQN CNN Config (Nature DQN architecture for pixel observations)
# =============================================================================


struct DQNCNNConfig[
    ACT: Int,
    CAP: Int = 10000,
    BS: Int = 32,
    lr: Float64 = 0.00025,
](DiscreteOffPolicyConfig):
    """DQN with Nature CNN for 4x84x84 pixel observations (Double DQN).

    Architecture: Conv2DReLU(8,4) -> Conv2DReLU(4,2) -> Conv2DReLU(3,1)
                  -> Flatten(3136) -> LinearReLU(512) -> Linear(num_actions)

    Matches the Nature DQN (Mnih et al., 2015) and CleanRL's DQN Atari.
    """

    comptime NAME: String = "DQN CNN"
    comptime obs_dim: Int = 4 * 84 * 84  # 28224 (4 stacked 84x84 frames)
    comptime num_actions: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        Conv2DReLU[4, 32, 8, 4, 0, 84, 84],
        Conv2DReLU[32, 64, 4, 2, 0, 20, 20],
        Conv2DReLU[64, 64, 3, 1, 0, 9, 9],
        FlattenLayer[64 * 7 * 7],
        LinearReLU[64 * 7 * 7, 512],
        Linear[512, Self.ACT],
    ]
    comptime QOpt = Adam[Self.lr]
    comptime QTargetStrat = DoubleQTarget
    comptime QOutputStrat = DirectQ
    comptime QGradStrat = ManualQGradient


# =============================================================================
# AutodiffDQNConfig -- Double DQN with AutodiffQGradient
# =============================================================================


struct AutodiffDQNConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 120,
    HIDDEN2: Int = 84,
    CAP: Int = 10000,
    BS: Int = 128,
    lr: Float64 = 2.5e-4,
](DiscreteOffPolicyConfig):
    """Double DQN config using AutodiffQGradient (GatherOp-based gradient)."""

    comptime NAME: String = "Autodiff DQN"
    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN2],
        Linear[Self.HIDDEN2, Self.ACT],
    ]
    comptime QOpt = Adam[Self.lr]
    comptime QTargetStrat = DoubleQTarget
    comptime QOutputStrat = DirectQ
    comptime QGradStrat = AutodiffQGradient


# =============================================================================
# DQN CPU State
# =============================================================================


struct DQNCPUStateGeneric[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    batch_size: Int,
](Movable, OffPolicyDiscreteState):
    """CPU state for DQN: online + target Q-networks + replay buffer."""

    comptime BUFFER_DTYPE = dtype

    var online: NetworkState[Self.QModel, Self.QOpt]
    var target: NetworkState[Self.QModel, Self.QOpt]
    var buffer: HeapReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]

    fn __init__(out self):
        self.online = NetworkState[Self.QModel, Self.QOpt]()
        self.online.initialize[Xavier[]]()
        self.target = NetworkState[Self.QModel, Self.QOpt]()
        self.target.initialize[Xavier[]]()
        # Copy online -> target
        self.target.copy_params_from(self.online)
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

    fn store[
        d: DType
    ](
        mut self,
        obs: List[Scalar[d]],
        action: Int,
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
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], 1](
            uninitialized=True
        )
        act_arr[0] = Scalar[Self.BUFFER_DTYPE](action)
        self.buffer.add(
            obs_arr,
            act_arr,
            Scalar[Self.BUFFER_DTYPE](reward),
            next_arr,
            done,
        )

    fn is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# DQN GPU State
# =============================================================================


struct DQNGPUStateGeneric[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    num_actions: Int,
    batch_size: Int,
    max_n_envs: Int,
](GPUOffPolicyState):
    """GPU-resident state for generic DQN training.

    Holds all device buffers needed for one DQN GPU training loop:
      - Online and target GPU network states
      - GPU replay buffer (discrete action stored as float scalar index)
      - Inference scratch buffers (sized by max_n_envs)
      - Training scratch buffers (sample output, Q caches, grad buffers)

    For Dueling DQN, QModel.OUT_DIM (RAW_OUT) = 1 + num_actions.
    Raw output buffers are sized by RAW_OUT; combined Q-value buffers by num_actions.

    Parameters:
        QModel: Q-network model type.
        QOpt: Q-network optimizer type.
        buffer_capacity: GPU replay buffer capacity.
        obs_dim: Observation space dimension.
        num_actions: Number of discrete actions.
        batch_size: Training batch size.
        max_n_envs: Max parallel environments (sizes inference buffers).
    """

    comptime Q_Net = Network[Self.QModel, Self.QOpt]
    comptime CACHE_SIZE = Self.QModel.CACHE_SIZE
    comptime WS_PER_SAMPLE = Self.Q_Net.WORKSPACE_SIZE_PER_SAMPLE
    comptime RAW_OUT = Self.QModel.OUT_DIM

    # GPU network states (online + target)
    var online: GPUNetworkState[Self.QModel, Self.QOpt]
    var target: GPUNetworkState[Self.QModel, Self.QOpt]

    # GPU replay buffer (ACTION_DIM=1 default: scalar discrete action)
    var buffer: GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim]

    # Inference buffers (max_n_envs sized, for select_actions_gpu)
    var env_raw_buf: DeviceBuffer[dtype]  # [max_n_envs * RAW_OUT]
    var env_q_buf: DeviceBuffer[dtype]  # [max_n_envs * num_actions]
    var inf_ws: DeviceBuffer[dtype]  # [max(1, max_n_envs * WS_PER_SAMPLE)]

    # Training scratch -- replay sample output
    var s_obs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_act: DeviceBuffer[dtype]  # [batch_size]
    var s_rew: DeviceBuffer[dtype]  # [batch_size]
    var s_nobs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_done: DeviceBuffer[dtype]  # [batch_size]
    var s_idx: DeviceBuffer[DType.int32]  # [batch_size]

    # Training scratch -- raw forward output (sized by RAW_OUT)
    var q_raw: DeviceBuffer[dtype]  # [batch_size * RAW_OUT]
    var next_q_raw: DeviceBuffer[dtype]  # [batch_size * RAW_OUT]
    var online_next_q_raw: DeviceBuffer[dtype]  # [batch_size * RAW_OUT]

    # Training scratch -- combined Q-values (sized by num_actions)
    var q_values: DeviceBuffer[dtype]  # [batch_size * num_actions]
    var next_q_values: DeviceBuffer[dtype]  # [batch_size * num_actions]
    var online_next_q: DeviceBuffer[dtype]  # [batch_size * num_actions]

    # Training scratch -- targets, cache, gradients
    var cache: DeviceBuffer[dtype]  # [batch_size * CACHE_SIZE]
    var targets: DeviceBuffer[dtype]  # [batch_size]
    var grad_q: DeviceBuffer[dtype]  # [batch_size * num_actions]
    var grad_raw: DeviceBuffer[dtype]  # [batch_size * RAW_OUT]
    var grad_input: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var train_ws: DeviceBuffer[dtype]  # [max(1, batch_size * WS_PER_SAMPLE)]

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers. CPU weights are uploaded separately."""
        self.online = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.target = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.buffer = GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim](ctx)

        # Inference buffers
        self.env_raw_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.RAW_OUT
        )
        self.env_q_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.num_actions
        )
        var inf_ws_size = max(1, Self.max_n_envs * Self.WS_PER_SAMPLE)
        self.inf_ws = ctx.enqueue_create_buffer[dtype](inf_ws_size)

        # Replay sample output
        self.s_obs = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        self.s_act = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.s_rew = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.s_nobs = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        self.s_done = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](Self.batch_size)

        # Raw forward output buffers
        var batch_raw_size = Self.batch_size * Self.RAW_OUT
        self.q_raw = ctx.enqueue_create_buffer[dtype](batch_raw_size)
        self.next_q_raw = ctx.enqueue_create_buffer[dtype](batch_raw_size)
        self.online_next_q_raw = ctx.enqueue_create_buffer[dtype](
            batch_raw_size
        )

        # Combined Q-value buffers
        var batch_q_size = Self.batch_size * Self.num_actions
        self.q_values = ctx.enqueue_create_buffer[dtype](batch_q_size)
        self.next_q_values = ctx.enqueue_create_buffer[dtype](batch_q_size)
        self.online_next_q = ctx.enqueue_create_buffer[dtype](batch_q_size)

        # Cache, targets, gradients
        self.cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CACHE_SIZE
        )
        self.targets = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.grad_q = ctx.enqueue_create_buffer[dtype](batch_q_size)
        self.grad_raw = ctx.enqueue_create_buffer[dtype](batch_raw_size)
        self.grad_input = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        var train_ws_size = max(1, Self.batch_size * Self.WS_PER_SAMPLE)
        self.train_ws = ctx.enqueue_create_buffer[dtype](train_ws_size)

    # -------------------------------------------------------------------------
    # GPUOffPolicyState required methods
    # -------------------------------------------------------------------------

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
        """Store N_ENVS transitions into the GPU replay buffer."""
        self.buffer.store[N_ENVS](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, dones_buf
        )

    fn gpu_buffer_is_ready(self) -> Bool:
        """Return True if the GPU replay buffer has enough samples to train."""
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# GenericDQNAgent[Config: DiscreteOffPolicyConfig]
# =============================================================================


struct GenericDQNAgent[
    Config: DiscreteOffPolicyConfig,
    n_envs: Int = 1024,
    L: Logger = NoOpLogger,
](OffPolicyDiscreteAgent & GPUOffPolicyAgent & Checkpointable):
    """Generic DQN agent. Supports standard, double, and dueling DQN via Config.

    CPU + GPU unified. GPU support via GPUOffPolicyAgent trait.

    Parameters:
        Config: Compile-time config (DQNConfig, DoubleDQNConfig, DuelingDQNConfig).
        n_envs: Number of parallel environments for GPU training (default: 1024).
        L: Logger type for diagnostic logging (default: NoOpLogger).
    """

    comptime OBS: Int = Self.Config.QModel.IN_DIM
    comptime RAW_OUT: Int = Self.Config.QModel.OUT_DIM
    comptime ACTIONS: Int = Self.Config.num_actions
    comptime BATCH: Int = Self.Config.batch_size
    comptime Q_CS: Int = Self.Config.QModel.CACHE_SIZE
    comptime QNet = Network[Self.Config.QModel, Self.Config.QOpt]

    comptime CPUStateType = DQNCPUStateGeneric[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.batch_size,
    ]

    # GPUOffPolicyAgent compile-time constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1  # discrete action stored as float scalar index
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = DQNGPUStateGeneric[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.batch_size,
        Self.n_envs,
    ]

    # Persistent CPU state (for evaluate() after train/train_gpu)
    var state: Self.CPUStateType

    var gamma: Float64
    var tau: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64
    var target_update_freq: Int
    var train_step_count: Int
    var target_total_steps: Int
    var _target_update_ctr: Int
    var checkpoint_every: Int
    var checkpoint_path: String

    # Logger
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 1.0,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.05,
        epsilon_decay: Float64 = 0.995,
        target_update_freq: Int = 500,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        target_total_steps: Int = 0,
    ):
        self.state = Self.CPUStateType()
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self._target_update_ctr = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn select_action[
        d: DType
    ](mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[d]]) -> Int:
        # Epsilon-greedy
        if random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.ACTIONS))

        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var raw_arr = InlineArray[Scalar[dtype], Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.QNet.forward[1](obs_t, raw_t, p)

        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        Self.Config.QOutputStrat.combine_cpu[1, Self.ACTIONS, Self.RAW_OUT](
            raw_arr, q_arr
        )

        var best = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best = a
        return best

    fn store_transition[
        d: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[d]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        cpu_state.store[d](obs, action, reward, next_obs, done)

    fn do_cpu_train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        # Sample batch
        var b_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_act1 = InlineArray[Scalar[dtype], Self.BATCH * 1](
            uninitialized=True
        )
        var b_rew = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var b_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_done = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        cpu_state.buffer.sample[Self.BATCH](
            b_obs, b_act1, b_rew, b_next, b_done
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_next.unsafe_ptr())

        # Online forward with cache (produces RAW_OUT per sample)
        var raw_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var cache_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.Q_CS](
            uninitialized=True
        )
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_CS), MutAnyOrigin
        ](cache_arr.unsafe_ptr())
        var p_online = cpu_state.online.params_view()
        Self.QNet.forward_with_cache[Self.BATCH](
            obs_t, raw_t, p_online, cache_t
        )

        # Apply Q-output strategy (identity for DirectQ, V+A-mean(A) for DuelingQ)
        var q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        Self.Config.QOutputStrat.combine_cpu[
            Self.BATCH, Self.ACTIONS, Self.RAW_OUT
        ](raw_arr, q_arr)

        # Target forward (produces RAW_OUT, then combine to Q-values)
        var next_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        var next_raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](next_raw_arr.unsafe_ptr())
        var p_target = cpu_state.target.params_view()
        Self.QNet.forward[Self.BATCH](next_obs_t, next_raw_t, p_target)

        var next_q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        Self.Config.QOutputStrat.combine_cpu[
            Self.BATCH, Self.ACTIONS, Self.RAW_OUT
        ](next_raw_arr, next_q_arr)

        # For Double DQN: also forward online net on next_obs and combine
        var online_next_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        var online_next_raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](online_next_raw_arr.unsafe_ptr())
        Self.QNet.forward[Self.BATCH](next_obs_t, online_next_raw_t, p_online)
        var online_next_q_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        Self.Config.QOutputStrat.combine_cpu[
            Self.BATCH, Self.ACTIONS, Self.RAW_OUT
        ](online_next_raw_arr, online_next_q_arr)

        # TD targets via strategy
        var targets = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        Self.Config.QTargetStrat.compute_targets_cpu[
            Self.BATCH,
            Self.ACTIONS,
        ](
            online_next_q_arr,
            next_q_arr,
            b_rew,
            b_done,
            targets,
            self.gamma,
        )

        # Gradient (MSE, masked to taken action) -- in Q-space via strategy
        var grad_q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var total_loss = Self.Config.QGradStrat.compute_grad_cpu[
            Self.BATCH, Self.ACTIONS
        ](q_arr, targets, b_act1, grad_q_arr)

        # Log DQN diagnostics
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count

                # Q-value stats
                var q_min = Float64(q_arr[0])
                var q_max = Float64(q_arr[0])
                var q_sum: Float64 = 0.0
                for i in range(Self.BATCH * Self.ACTIONS):
                    var v = Float64(q_arr[i])
                    q_sum += v
                    if v < q_min:
                        q_min = v
                    if v > q_max:
                        q_max = v
                self.logger[].log_scalar(
                    "q_mean",
                    q_sum / Float64(Self.BATCH * Self.ACTIONS),
                    step,
                )
                self.logger[].log_scalar("q_min", q_min, step)
                self.logger[].log_scalar("q_max", q_max, step)

                # TD target stats
                var tgt_sum: Float64 = 0.0
                for i in range(Self.BATCH):
                    tgt_sum += Float64(targets[i])
                self.logger[].log_scalar(
                    "td_target_mean",
                    tgt_sum / Float64(Self.BATCH),
                    step,
                )

                # TD error stats
                var td_err_abs_sum: Float64 = 0.0
                var td_err_max_abs: Float64 = 0.0
                for b2 in range(Self.BATCH):
                    var act2 = Int(b_act1[b2])
                    var td_err2 = Float64(
                        q_arr[b2 * Self.ACTIONS + act2]
                    ) - Float64(targets[b2])
                    var abs_err = td_err2 if td_err2 >= 0 else -td_err2
                    td_err_abs_sum += abs_err
                    if abs_err > td_err_max_abs:
                        td_err_max_abs = abs_err
                self.logger[].log_scalar(
                    "td_error_abs_mean",
                    td_err_abs_sum / Float64(Self.BATCH),
                    step,
                )
                self.logger[].log_scalar("td_error_max", td_err_max_abs, step)
                self.logger[].log_scalar("loss", total_loss, step)
            except:
                pass

        # Transform gradient from Q-space to raw output space
        var grad_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        Self.Config.QOutputStrat.grad_transform_cpu[
            Self.BATCH, Self.ACTIONS, Self.RAW_OUT
        ](grad_q_arr, grad_raw_arr)

        # Backward + optimizer step
        var grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](grad_raw_arr.unsafe_ptr())
        var d_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](d_obs.unsafe_ptr())
        var g = cpu_state.online.grads_view()
        cpu_state.online.zero_grads()
        Self.QNet.backward[Self.BATCH](grad_t, d_obs_t, p_online, cache_t, g)
        cpu_state.online.optimizer_step()

        self.train_step_count += 1

        # Target update (hard or soft)
        if (
            self.train_step_count - self._target_update_ctr
            >= self.target_update_freq
        ):
            self._target_update_ctr = self.train_step_count
            if self.tau >= 1.0:
                cpu_state.target.copy_params_from(cpu_state.online)
            else:
                cpu_state.target.soft_update_from(cpu_state.online, self.tau)
        return total_loss

    fn decay_explore(mut self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn get_explore_rate(self) -> Float64:
        return self.epsilon

    fn random_action(self) -> Int:
        return Int(random_float64() * Float64(Self.ACTIONS))

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> Int:
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var raw_arr = InlineArray[Scalar[dtype], Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.QNet.forward[1](obs_t, raw_t, p)

        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        Self.Config.QOutputStrat.combine_cpu[1, Self.ACTIONS, Self.RAW_OUT](
            raw_arr, q_arr
        )

        var best = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best = a
        return best

    # Checkpointable
    fn save_checkpoint(self, path: String) raises -> None:
        """Save DQN agent state to a checkpoint file.

        Saves online and target network params + optimizer states,
        plus epsilon and training counters. Replay buffer is NOT saved.
        """
        comptime PARAM_SIZE = Self.QNet.PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * Self.Config.QOpt.STATE_PER_PARAM

        var content = write_checkpoint_header(
            "generic_dqn_agent", PARAM_SIZE, STATE_SIZE
        )
        content += self.state.online.write_sections("online_")
        content += self.state.target.write_sections("target_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("epsilon=" + String(self.epsilon))
        metadata.append("epsilon_min=" + String(self.epsilon_min))
        metadata.append("epsilon_decay=" + String(self.epsilon_decay))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(path, content)

    fn load_checkpoint(mut self, path: String) raises -> None:
        """Load DQN agent state from a checkpoint file."""
        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)

        self.state.online.read_sections(content, "online_")
        self.state.target.read_sections(content, "target_")

        var metadata = read_metadata_section(content)

        var gamma_str = get_metadata_value(metadata, "gamma")
        if len(gamma_str) > 0:
            self.gamma = atof(gamma_str)

        var tau_str = get_metadata_value(metadata, "tau")
        if len(tau_str) > 0:
            self.tau = atof(tau_str)

        var epsilon_str = get_metadata_value(metadata, "epsilon")
        if len(epsilon_str) > 0:
            self.epsilon = atof(epsilon_str)

        var epsilon_min_str = get_metadata_value(metadata, "epsilon_min")
        if len(epsilon_min_str) > 0:
            self.epsilon_min = atof(epsilon_min_str)

        var epsilon_decay_str = get_metadata_value(metadata, "epsilon_decay")
        if len(epsilon_decay_str) > 0:
            self.epsilon_decay = atof(epsilon_decay_str)

        var train_step_str = get_metadata_value(metadata, "train_step_count")
        if len(train_step_str) > 0:
            self.train_step_count = Int(atol(train_step_str))

    # =========================================================================
    # CPU Convenience training
    # =========================================================================

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 300,
        max_steps_per_episode: Int = 500,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train the DQN agent on a discrete-action environment.

        Args:
            env: Environment implementing BoxDiscreteActionEnv.
            num_episodes: Number of training episodes.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 4).
            verbose: Print progress (default: False).
            print_every: Print every N episodes if verbose (default: 10).
            environment_name: Name for metrics labeling.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N train steps. 0 = every step
                when logger is set (default: 0).

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        from mojo_rl.deep_agents.core.offpolicy_train import (
            run_offpolicy_discrete_train,
        )

        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType()
        var ckpt_path = String(self.checkpoint_path)
        var algo_name = Self.Config.NAME
        var metrics = run_offpolicy_discrete_train[E, Self, Self.L](
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
            algorithm_name=algo_name,
            logger=logger,
        )
        self.state = cpu_state^
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics

    # =========================================================================
    # Evaluation
    # =========================================================================

    fn evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: Environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps_per_episode: Maximum steps per episode (default: 500).
            verbose: Print per-episode results (default: False).
            render: Render the environment (default: False).
            frame_delay_ms: Delay between frames in ms (default: 16).

        Returns:
            Average reward across episodes.
        """
        var metrics = run_offpolicy_discrete_eval(
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
    # GPUOffPolicyAgent trait conformance
    # =========================================================================

    fn get_action_scale(self) -> Float64:
        return 1.0  # Discrete actions don't use action_scale

    fn get_total_steps(self) -> Int:
        return self.train_step_count

    fn set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for DQN training."""
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights to GPU online and target networks."""
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.online.download_to(self.state.online, ctx)
        gpu_state.target.download_to(self.state.target, ctx)

    fn select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward Q-network on GPU for N_ENVS environments + epsilon-greedy selection.

        Pipeline: obs -> forward (raw) -> combine (Q-values) -> argmax + epsilon-greedy.
        """
        # Forward pass: obs -> raw output
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.env_raw_buf.unsafe_ptr())
        var p = gpu_state.online.params_view()
        Self.QNet.forward_gpu[N_ENVS](ctx, obs_t, raw_t, p, gpu_state.inf_ws)

        # Combine raw output to Q-values
        var q_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.env_q_buf.unsafe_ptr())
        Self.Config.QOutputStrat.combine_gpu[
            N_ENVS, Self.ACTIONS, Self.RAW_OUT
        ](ctx, raw_t, q_t)

        # Epsilon-greedy action selection
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var epsilon_s = Scalar[dtype](self.epsilon)
        var seed_val = Scalar[DType.uint64](
            UInt64(self.get_total_steps()) * UInt64(2654435761)
        )

        @always_inline
        fn argmax_wrapper(
            eps: Scalar[dtype],
            q_vals: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            acts: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            base_seed: Scalar[DType.uint64],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= N_ENVS:
                return

            var rng = PhiloxRandom(
                seed=UInt64(base_seed) + UInt64(b),
                offset=0,
            )
            var rand_vals = rng.step_uniform()
            var rand_val = Scalar[dtype](rand_vals[0])

            if rand_val < eps:
                acts[b] = Scalar[dtype](
                    Int(
                        Scalar[dtype](rand_vals[1])
                        * Scalar[dtype](Self.ACTIONS)
                    )
                    % Self.ACTIONS
                )
                return

            var best_q = q_vals[b, 0]
            var best_action = 0
            for a in range(1, Self.ACTIONS):
                var qv = q_vals[b, a]
                if qv > best_q:
                    best_q = qv
                    best_action = a

            acts[b] = Scalar[dtype](best_action)

        ctx.enqueue_function[argmax_wrapper, argmax_wrapper](
            epsilon_s,
            q_t,
            actions_t,
            seed_val,
            grid_dim=((N_ENVS + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """One DQN training step on GPU: sample -> TD targets -> backward -> update.

        Soft-update of target network is handled separately by soft_update_targets_gpu.
        """
        comptime BATCH = Self.BATCH
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # ---- Phase 1: Sample batch ----
        gpu_state.buffer.sample[BATCH](
            ctx,
            UInt32(self.train_step_count * (BATCH + 1)),
            gpu_state.s_obs,
            gpu_state.s_act,
            gpu_state.s_rew,
            gpu_state.s_nobs,
            gpu_state.s_done,
            gpu_state.s_idx,
        )

        # LayoutTensor views for sampled batch
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_rew.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_done.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())

        # Raw output tensors
        var q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.q_raw.unsafe_ptr())
        var next_q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.next_q_raw.unsafe_ptr())
        var online_next_q_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.online_next_q_raw.unsafe_ptr())

        # Combined Q-value tensors
        var q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.q_values.unsafe_ptr())
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.next_q_values.unsafe_ptr())
        var online_next_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.online_next_q.unsafe_ptr())

        # Cache tensor
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Q_CS),
            MutAnyOrigin,
        ](gpu_state.cache.unsafe_ptr())

        # Grad tensors
        var grad_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.grad_q.unsafe_ptr())
        var grad_raw_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.grad_raw.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.grad_input.unsafe_ptr())

        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()

        # ---- Phase 2: Online forward with cache -> raw -> combine ----
        Self.QNet.forward_gpu_with_cache[BATCH](
            ctx,
            obs_t,
            q_raw_t,
            p_online,
            cache_t,
            gpu_state.train_ws,
        )
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, q_raw_t, q_t
        )

        # ---- Phase 3: Target forward -> raw -> combine ----
        Self.QNet.forward_gpu[BATCH](
            ctx,
            next_obs_t,
            next_q_raw_t,
            p_target,
            gpu_state.train_ws,
        )
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, next_q_raw_t, next_q_t
        )

        # ---- Phase 3b: Online forward on next_obs -> raw -> combine (for Double DQN) ----
        Self.QNet.forward_gpu[BATCH](
            ctx, next_obs_t, online_next_q_raw_t, p_online, gpu_state.train_ws
        )
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, online_next_q_raw_t, online_next_q_t
        )

        # ---- Phase 4: Compute TD targets via strategy ----
        Self.Config.QTargetStrat.compute_targets_gpu[BATCH, Self.ACTIONS](
            ctx,
            targets_t,
            online_next_q_t,
            next_q_t,
            rewards_t,
            dones_t,
            self.gamma,
        )

        # ---- Phase 5: Gradient via QGradient strategy ----
        Self.Config.QGradStrat.compute_grad_gpu[BATCH, Self.ACTIONS](
            ctx, q_t, targets_t, actions_t, grad_q_t
        )

        # ---- Phase 5b: Transform grad from Q-space to raw output space ----
        Self.Config.QOutputStrat.grad_transform_gpu[
            BATCH, Self.ACTIONS, Self.RAW_OUT
        ](ctx, grad_q_t, grad_raw_t)

        # ---- Phase 6: Backward + optimizer step ----
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.QNet.backward_gpu[BATCH](
            ctx,
            grad_raw_t,
            grad_in_t,
            p_online,
            cache_t,
            g,
            gpu_state.train_ws,
        )
        gpu_state.online.optimizer_step(ctx)

        self.train_step_count += 1

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Soft-update target Q-network on GPU every target_update_freq gradient steps.

        Called once per collection iteration by the training loop (after grad_steps
        training steps). Uses train_step_count (incremented in do_gpu_train_step)
        to track actual gradient steps, matching CleanRL's target_network_frequency.
        """
        # Check how many gradient steps happened since last target update
        if (
            self.train_step_count - self._target_update_ctr
            >= self.target_update_freq
        ):
            gpu_state.target.soft_update_from_gpu(
                gpu_state.online, self.tau, ctx
            )
            self._target_update_ctr = self.train_step_count

    fn decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        """Linear epsilon schedule matching CleanRL:
        epsilon = max(end_e, start_e + (end_e - start_e) * t / duration).
        Exploration fraction = 0.1 (CleanRL default: decay over first 10%).
        """
        var duration = (
            Float64(num_steps) * 0.1
        )  # exploration_fraction = 0.1 (CleanRL)
        var slope = (self.epsilon_min - 1.0) / duration
        self.epsilon = max(
            self.epsilon_min,
            slope * Float64(total_steps) + 1.0,
        )

    # =========================================================================
    # GPU Training convenience
    # =========================================================================

    fn train_gpu[
        E: GPUDiscreteEnv,
        CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 0,
        sync_every: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 100,
    ) raises -> TrainingMetrics:
        """Train on GPU using the shared off-policy discrete GPU loop.

        GPU state is created locally. After training, CPU state holds the
        trained weights so evaluate() works immediately.

        Parameters:
            E: GPU environment type implementing GPUDiscreteEnv.
            CurriculumType: Curriculum scheduler type (default: NoCurriculumScheduler).

        Args:
            ctx: GPU device context.
            num_steps: Total env transitions across all parallel envs.
            warmup_steps: Transitions before training starts (default: 1000).
            gradient_steps: Training steps per env collection iteration.
                0 (default) = n_envs for 1:1 replay ratio.
            sync_every: GPU->CPU sync interval in transitions (default: 5000).
            verbose: Print progress (default: False).
            print_every: Print interval in transitions (default: 50000).
            environment_name: Name for metrics labeling.
            logger: Optional metrics logger.
            diag_every: Log diagnostics every N train steps (default: 100).

        Returns:
            TrainingMetrics with episode-level statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var algo_name = Self.Config.NAME + " (GPU)"
        var timer = PerfTimer[False]()
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
        var metrics = run_offpolicy_discrete_train_gpu[
            E, Self, 0, Self.L, CurriculumType
        ](
            self,
            ctx,
            num_steps,
            timer,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            checkpoint_every=ckpt_every,
            checkpoint_path=ckpt_path,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
            logger=logger,
            target_total_steps=tgt_steps,
        )

        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^


# =============================================================================
# DQN + PER Config
# =============================================================================


struct DQNPERConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 128,
    HIDDEN2: Int = 128,
    CAP: Int = 20000,
    BS: Int = 64,
    lr: Float64 = 0.0005,
](DiscreteOffPolicyConfig):
    """DQN + Prioritized Experience Replay config (Double DQN).

    PER is CPU-only; GPU path uses uniform replay.
    """

    comptime NAME: String = "DQN PER"
    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN2],
        Linear[Self.HIDDEN2, Self.ACT],
    ]
    comptime QOpt = Adam[Self.lr]
    comptime QTargetStrat = DoubleQTarget
    comptime QOutputStrat = DirectQ
    comptime QGradStrat = ManualQGradient


# =============================================================================
# DQN PER CPU State (uses PrioritizedReplayBuffer)
# =============================================================================


struct DQNPERCPUStateGeneric[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    batch_size: Int,
](Movable, OffPolicyDiscreteState):
    """CPU state for DQN+PER: online + target Q-networks + prioritized replay buffer."""

    comptime BUFFER_DTYPE = dtype

    var online: NetworkState[Self.QModel, Self.QOpt]
    var target: NetworkState[Self.QModel, Self.QOpt]
    var buffer: PrioritizedReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, dtype
    ]

    fn __init__(out self, alpha: Float64 = 0.6, beta: Float64 = 0.4):
        self.online = NetworkState[Self.QModel, Self.QOpt]()
        self.online.initialize[Xavier[]]()
        self.target = NetworkState[Self.QModel, Self.QOpt]()
        self.target.initialize[Xavier[]]()
        self.target.copy_params_from(self.online)
        self.buffer = PrioritizedReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ](
            alpha=Scalar[dtype](alpha),
            beta=Scalar[dtype](beta),
        )

    fn store[
        d: DType
    ](
        mut self,
        obs: List[Scalar[d]],
        action: Int,
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
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], 1](
            uninitialized=True
        )
        act_arr[0] = Scalar[Self.BUFFER_DTYPE](action)
        self.buffer.add(
            obs_arr,
            act_arr,
            Scalar[Self.BUFFER_DTYPE](reward),
            next_arr,
            done,
        )

    fn is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# GenericDQNPERAgent
# =============================================================================


struct GenericDQNPERAgent[
    Config: DiscreteOffPolicyConfig,
    n_envs: Int = 1024,
    L: Logger = NoOpLogger,
](OffPolicyDiscreteAgent & GPUOffPolicyAgent & Checkpointable):
    """Generic DQN + Prioritized Experience Replay agent.

    CPU path uses PrioritizedReplayBuffer with importance sampling weights
    and priority updates. GPU path uses uniform replay (PER sum-tree is
    inherently serial).

    Parameters:
        Config: DQN config (DQNPERConfig or any DiscreteOffPolicyConfig).
        n_envs: Number of parallel environments for GPU training (default: 1024).
        L: Logger type for diagnostic logging (default: NoOpLogger).
    """

    comptime OBS: Int = Self.Config.QModel.IN_DIM
    comptime RAW_OUT: Int = Self.Config.QModel.OUT_DIM
    comptime ACTIONS: Int = Self.Config.num_actions
    comptime BATCH: Int = Self.Config.batch_size
    comptime Q_CS: Int = Self.Config.QModel.CACHE_SIZE
    comptime QNet = Network[Self.Config.QModel, Self.Config.QOpt]

    # CPU state uses PrioritizedReplayBuffer
    comptime CPUStateType = DQNPERCPUStateGeneric[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.batch_size,
    ]

    # GPU state uses uniform replay (same as standard DQN)
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = DQNGPUStateGeneric[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.batch_size,
        Self.n_envs,
    ]

    # Persistent CPU state
    var state: Self.CPUStateType

    var gamma: Float64
    var tau: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64
    var target_update_freq: Int
    var train_step_count: Int
    var target_total_steps: Int
    var _target_update_ctr: Int
    var checkpoint_every: Int
    var checkpoint_path: String

    # PER-specific
    var beta: Float64
    var beta_start: Float64
    var beta_frames: Int

    # Logger
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.995,
        target_update_freq: Int = 500,
        alpha: Float64 = 0.6,
        beta: Float64 = 0.4,
        beta_frames: Int = 100000,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        target_total_steps: Int = 0,
    ):
        self.state = Self.CPUStateType(alpha=alpha, beta=beta)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self._target_update_ctr = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.beta = beta
        self.beta_start = beta
        self.beta_frames = beta_frames
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn select_action[
        d: DType
    ](mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[d]]) -> Int:
        if random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.ACTIONS))

        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var raw_arr = InlineArray[Scalar[dtype], Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.QNet.forward[1](obs_t, raw_t, p)

        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        Self.Config.QOutputStrat.combine_cpu[1, Self.ACTIONS, Self.RAW_OUT](
            raw_arr, q_arr
        )

        var best = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best = a
        return best

    fn store_transition[
        d: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[d]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        cpu_state.store[d](obs, action, reward, next_obs, done)

    fn do_cpu_train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """One DQN+PER training step: sample with IS weights, update priorities."""
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        # Anneal beta
        self.beta = self.beta_start + (1.0 - self.beta_start) * min(
            1.0, Float64(self.train_step_count) / Float64(self.beta_frames)
        )
        cpu_state.buffer.beta = Scalar[dtype](self.beta)

        # Sample with IS weights and indices
        var b_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_act1 = InlineArray[Scalar[dtype], Self.BATCH * 1](
            uninitialized=True
        )
        var b_rew = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var b_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_done = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var b_weights = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var b_indices = InlineArray[Int, Self.BATCH](uninitialized=True)

        cpu_state.buffer.sample[Self.BATCH](
            b_obs, b_act1, b_rew, b_next, b_done, b_weights, b_indices
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_next.unsafe_ptr())

        # Online forward with cache
        var raw_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var cache_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.Q_CS](
            uninitialized=True
        )
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_CS), MutAnyOrigin
        ](cache_arr.unsafe_ptr())
        var p_online = cpu_state.online.params_view()
        Self.QNet.forward_with_cache[Self.BATCH](
            obs_t, raw_t, p_online, cache_t
        )

        # Combine raw → Q-values
        var q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        Self.Config.QOutputStrat.combine_cpu[
            Self.BATCH, Self.ACTIONS, Self.RAW_OUT
        ](raw_arr, q_arr)

        # Target forward → combine
        var next_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        var next_raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](next_raw_arr.unsafe_ptr())
        var p_target = cpu_state.target.params_view()
        Self.QNet.forward[Self.BATCH](next_obs_t, next_raw_t, p_target)
        var next_q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        Self.Config.QOutputStrat.combine_cpu[
            Self.BATCH, Self.ACTIONS, Self.RAW_OUT
        ](next_raw_arr, next_q_arr)

        # Online next (for Double DQN strategy)
        var online_next_raw = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        var online_next_raw_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](online_next_raw.unsafe_ptr())
        Self.QNet.forward[Self.BATCH](
            next_obs_t, online_next_raw_t, p_online
        )
        var online_next_q = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        Self.Config.QOutputStrat.combine_cpu[
            Self.BATCH, Self.ACTIONS, Self.RAW_OUT
        ](online_next_raw, online_next_q)

        # TD targets via strategy
        var targets = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        Self.Config.QTargetStrat.compute_targets_cpu[Self.BATCH, Self.ACTIONS](
            online_next_q, next_q_arr, b_rew, b_done, targets, self.gamma
        )

        # Weighted gradient (PER importance sampling)
        var grad_q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            fill=Scalar[dtype](0.0)
        )
        var td_errors = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var total_loss: Float64 = 0.0

        for b in range(Self.BATCH):
            var action = Int(b_act1[b])
            var q_pred = q_arr[b * Self.ACTIONS + action]
            var td_error = q_pred - targets[b]
            td_errors[b] = td_error

            var weight = b_weights[b]
            var weighted_error = weight * td_error
            total_loss += Float64(weighted_error * weighted_error)

            grad_q_arr[b * Self.ACTIONS + action] = (
                Scalar[dtype](2.0) * weighted_error / Scalar[dtype](Self.BATCH)
            )
        total_loss /= Float64(Self.BATCH)

        # Transform grad Q-space → raw output space
        var grad_raw_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.RAW_OUT
        ](uninitialized=True)
        Self.Config.QOutputStrat.grad_transform_cpu[
            Self.BATCH, Self.ACTIONS, Self.RAW_OUT
        ](grad_q_arr, grad_raw_arr)

        # Backward + optimizer step
        var grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.RAW_OUT), MutAnyOrigin
        ](grad_raw_arr.unsafe_ptr())
        var d_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](d_obs.unsafe_ptr())
        var g = cpu_state.online.grads_view()
        cpu_state.online.zero_grads()
        Self.QNet.backward[Self.BATCH](grad_t, d_obs_t, p_online, cache_t, g)
        cpu_state.online.optimizer_step()

        # Update priorities
        cpu_state.buffer.update_priorities[Self.BATCH](b_indices, td_errors)

        # Target update
        self.train_step_count += 1
        if (
            self.train_step_count - self._target_update_ctr
            >= self.target_update_freq
        ):
            self._target_update_ctr = self.train_step_count
            if self.tau >= 1.0:
                cpu_state.target.copy_params_from(cpu_state.online)
            else:
                cpu_state.target.soft_update_from(cpu_state.online, self.tau)

        return total_loss

    fn decay_explore(mut self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn get_explore_rate(self) -> Float64:
        return self.epsilon

    fn random_action(self) -> Int:
        return Int(random_float64() * Float64(Self.ACTIONS))

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> Int:
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var raw_arr = InlineArray[Scalar[dtype], Self.RAW_OUT](
            uninitialized=True
        )
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.RAW_OUT), MutAnyOrigin
        ](raw_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.QNet.forward[1](obs_t, raw_t, p)

        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        Self.Config.QOutputStrat.combine_cpu[1, Self.ACTIONS, Self.RAW_OUT](
            raw_arr, q_arr
        )

        var best = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best = a
        return best

    # Checkpointable
    fn save_checkpoint(self, path: String) raises -> None:
        comptime PARAM_SIZE = Self.QNet.PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * Self.Config.QOpt.STATE_PER_PARAM

        var content = write_checkpoint_header(
            "generic_dqn_per_agent", PARAM_SIZE, STATE_SIZE
        )
        content += self.state.online.write_sections("online_")
        content += self.state.target.write_sections("target_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("epsilon=" + String(self.epsilon))
        metadata.append("epsilon_min=" + String(self.epsilon_min))
        metadata.append("beta=" + String(self.beta))
        metadata.append("beta_start=" + String(self.beta_start))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(path, content)

    fn load_checkpoint(mut self, path: String) raises -> None:
        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)
        self.state.online.read_sections(content, "online_")
        self.state.target.read_sections(content, "target_")

        var metadata = read_metadata_section(content)
        var gamma_str = get_metadata_value(metadata, "gamma")
        if len(gamma_str) > 0:
            self.gamma = atof(gamma_str)
        var tau_str = get_metadata_value(metadata, "tau")
        if len(tau_str) > 0:
            self.tau = atof(tau_str)
        var eps_str = get_metadata_value(metadata, "epsilon")
        if len(eps_str) > 0:
            self.epsilon = atof(eps_str)
        var beta_str = get_metadata_value(metadata, "beta")
        if len(beta_str) > 0:
            self.beta = atof(beta_str)
        var step_str = get_metadata_value(metadata, "train_step_count")
        if len(step_str) > 0:
            self.train_step_count = Int(atol(step_str))

    # =========================================================================
    # CPU Training
    # =========================================================================

    fn train[
        E: BoxDiscreteActionEnv
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
        """Train with PER on CPU."""
        from mojo_rl.deep_agents.core.offpolicy_train import (
            run_offpolicy_discrete_train,
        )

        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType()
        var ckpt_path = String(self.checkpoint_path)
        var algo_name = Self.Config.NAME + " +PER"
        var metrics = run_offpolicy_discrete_train[E, Self, Self.L](
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
            algorithm_name=algo_name,
            logger=logger,
        )
        self.state = cpu_state^
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics

    fn evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate using greedy policy."""
        var metrics = run_offpolicy_discrete_eval(
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
    # GPUOffPolicyAgent trait (uniform replay — no PER on GPU)
    # =========================================================================

    fn get_action_scale(self) -> Float64:
        return 1.0

    fn get_total_steps(self) -> Int:
        return self.train_step_count

    fn set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.download_to(self.state.online, ctx)
        gpu_state.target.download_to(self.state.target, ctx)

    fn select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward Q-network on GPU + epsilon-greedy (same as standard DQN)."""
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
        ](gpu_state.env_raw_buf.unsafe_ptr())
        var p = gpu_state.online.params_view()
        Self.QNet.forward_gpu[N_ENVS](ctx, obs_t, raw_t, p, gpu_state.inf_ws)

        var q_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.env_q_buf.unsafe_ptr())
        Self.Config.QOutputStrat.combine_gpu[
            N_ENVS, Self.ACTIONS, Self.RAW_OUT
        ](ctx, raw_t, q_t)

        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var epsilon_s = Scalar[dtype](self.epsilon)
        var seed_val = Scalar[DType.uint64](
            UInt64(self.get_total_steps()) * UInt64(2654435761)
        )

        @always_inline
        fn argmax_wrapper(
            eps: Scalar[dtype],
            q_vals: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            acts: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            base_seed: Scalar[DType.uint64],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= N_ENVS:
                return
            var rng = PhiloxRandom(
                seed=UInt64(base_seed) + UInt64(b), offset=0
            )
            var rand_vals = rng.step_uniform()
            if Scalar[dtype](rand_vals[0]) < eps:
                acts[b] = Scalar[dtype](
                    Int(Scalar[dtype](rand_vals[1]) * Scalar[dtype](Self.ACTIONS))
                    % Self.ACTIONS
                )
                return
            var best_q = q_vals[b, 0]
            var best_action = 0
            for a in range(1, Self.ACTIONS):
                var qv = q_vals[b, a]
                if qv > best_q:
                    best_q = qv
                    best_action = a
            acts[b] = Scalar[dtype](best_action)

        ctx.enqueue_function[argmax_wrapper, argmax_wrapper](
            epsilon_s, q_t, actions_t, seed_val,
            grid_dim=((N_ENVS + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """GPU train step with UNIFORM replay (no PER on GPU)."""
        comptime BATCH = Self.BATCH
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        gpu_state.buffer.sample[BATCH](
            ctx,
            UInt32(self.train_step_count * (BATCH + 1)),
            gpu_state.s_obs, gpu_state.s_act, gpu_state.s_rew,
            gpu_state.s_nobs, gpu_state.s_done, gpu_state.s_idx,
        )

        var obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin](gpu_state.s_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin](gpu_state.s_nobs.unsafe_ptr())
        var actions_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](gpu_state.s_act.unsafe_ptr())
        var rewards_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](gpu_state.s_rew.unsafe_ptr())
        var dones_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](gpu_state.s_done.unsafe_ptr())
        var targets_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](gpu_state.targets.unsafe_ptr())
        var q_raw_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin](gpu_state.q_raw.unsafe_ptr())
        var next_q_raw_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin](gpu_state.next_q_raw.unsafe_ptr())
        var online_next_q_raw_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin](gpu_state.online_next_q_raw.unsafe_ptr())
        var q_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin](gpu_state.q_values.unsafe_ptr())
        var next_q_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin](gpu_state.next_q_values.unsafe_ptr())
        var online_next_q_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin](gpu_state.online_next_q.unsafe_ptr())
        var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Q_CS), MutAnyOrigin](gpu_state.cache.unsafe_ptr())
        var grad_q_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin](gpu_state.grad_q.unsafe_ptr())
        var grad_raw_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.RAW_OUT), MutAnyOrigin](gpu_state.grad_raw.unsafe_ptr())
        var grad_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin](gpu_state.grad_input.unsafe_ptr())

        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()

        # Online forward + combine
        Self.QNet.forward_gpu_with_cache[BATCH](ctx, obs_t, q_raw_t, p_online, cache_t, gpu_state.train_ws)
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](ctx, q_raw_t, q_t)

        # Target forward + combine
        Self.QNet.forward_gpu[BATCH](ctx, next_obs_t, next_q_raw_t, p_target, gpu_state.train_ws)
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](ctx, next_q_raw_t, next_q_t)

        # Online next forward + combine (for Double DQN)
        Self.QNet.forward_gpu[BATCH](ctx, next_obs_t, online_next_q_raw_t, p_online, gpu_state.train_ws)
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](ctx, online_next_q_raw_t, online_next_q_t)

        # TD targets
        Self.Config.QTargetStrat.compute_targets_gpu[BATCH, Self.ACTIONS](
            ctx, targets_t, online_next_q_t, next_q_t, rewards_t, dones_t, self.gamma
        )

        # Gradient via QGradient strategy
        Self.Config.QGradStrat.compute_grad_gpu[BATCH, Self.ACTIONS](
            ctx, q_t, targets_t, actions_t, grad_q_t
        )

        # Grad transform + backward
        Self.Config.QOutputStrat.grad_transform_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](ctx, grad_q_t, grad_raw_t)
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.QNet.backward_gpu[BATCH](ctx, grad_raw_t, grad_in_t, p_online, cache_t, g, gpu_state.train_ws)
        gpu_state.online.optimizer_step(ctx)

        self.train_step_count += 1

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        if self.train_step_count - self._target_update_ctr >= self.target_update_freq:
            gpu_state.target.soft_update_from_gpu(gpu_state.online, self.tau, ctx)
            self._target_update_ctr = self.train_step_count

    fn decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        var duration = Float64(num_steps) * 0.5
        var slope = (self.epsilon_min - 1.0) / duration
        self.epsilon = max(self.epsilon_min, slope * Float64(total_steps) + 1.0)

    # =========================================================================
    # GPU Training convenience
    # =========================================================================

    fn train_gpu[
        E: GPUDiscreteEnv,
        CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 0,
        sync_every: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 100,
    ) raises -> TrainingMetrics:
        """Train on GPU (uniform replay, no PER)."""
        self.logger = logger
        self.diag_every = diag_every
        var algo_name = Self.Config.NAME + " +PER (GPU)"
        var timer = PerfTimer[False]()
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
        var metrics = run_offpolicy_discrete_train_gpu[
            E, Self, 0, Self.L, CurriculumType
        ](
            self, ctx, num_steps, timer,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            checkpoint_every=ckpt_every,
            checkpoint_path=ckpt_path,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
            target_total_steps=tgt_steps,
            logger=logger,
        )

        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^
