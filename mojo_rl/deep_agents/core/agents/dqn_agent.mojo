"""Generic DQN agent parameterized by DiscreteOffPolicyConfig.

Supports standard DQN, Double DQN, and Dueling DQN via strategy types:
  - QTargetStrat: StandardQTarget (DQN) or DoubleQTarget (Double/Dueling DQN)
  - QOutputStrat: DirectQ (standard/double) or DuelingQ (dueling architecture)

GPU support via GPUOffPolicyAgent trait + run_offpolicy_discrete_train_gpu.
"""

from std.math import exp
from std.random import random_float64
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from mojo_rl.deep_agents.core import (
    run_offpolicy_discrete_train_gpu,
    PerfTimer,
)
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    Sequential,
    Parallel,
    Conv2DReLU,
    FlattenLayer,
    HuberLoss,
    NoisyLinear,
    NoisyLinearReLU,
)
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
from mojo_rl.deep_agents.core.replay import (
    HeapReplayBuffer,
    GPUReplayBuffer,
    PrioritizedReplayBuffer,
    GPUPrioritizedReplayBuffer,
)
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

from ..strategies.q_target import QTarget, StandardQTarget, DoubleQTarget
from ..strategies.q_output import QOutput, DirectQ, DuelingQ
from ..strategies.q_gradient import (
    QGradient,
    ManualQGradient,
    AutodiffQGradient,
)


# =============================================================================
# DQNTrainWS — Training workspace for DQN GPU training buffers
# =============================================================================


struct DQNTrainWS[
    BS: Int,  # Batch size
    OBS: Int,  # Observation dimension
    ACTIONS: Int,  # Number of discrete actions
    RAW_OUT: Int,  # Raw network output dim (=ACTIONS for DirectQ, 1+ACTIONS for Dueling)
    CACHE_SIZE: Int,  # Q-network cache size per sample
    GRAD_WS: Int = 0,  # QGradient strategy workspace size
](ImplicitlyCopyable, Movable):
    """Typed workspace providing named LayoutTensor views over flat GPU memory.

    All offsets are computed at compile time. The struct is just a pointer
    wrapper — zero overhead, zero allocation, works on CPU and GPU.

    Network workspace (forward_gpu/backward_gpu) is kept as a separate
    DeviceBuffer because the Network API requires DeviceBuffer[dtype].

    Layout:
        Region 1: Raw forward outputs (q_raw, next_q_raw, online_next_q_raw)
        Region 2: Combined Q-values (q_values, next_q_values, online_next_q)
        Region 3: Targets
        Region 4: Cache
        Region 5: Gradients (grad_q, grad_raw, grad_input)
        Region 6: Loss/gradient strategy workspace
    """

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    # --- Region 1: Raw forward outputs ---
    comptime _O_Q_RAW: Int = 0
    comptime _O_NEXT_Q_RAW: Int = Self._O_Q_RAW + Self.BS * Self.RAW_OUT
    comptime _O_ONLINE_NEXT_Q_RAW: Int = Self._O_NEXT_Q_RAW + Self.BS * Self.RAW_OUT

    # --- Region 2: Combined Q-values ---
    comptime _O_Q_VALUES: Int = Self._O_ONLINE_NEXT_Q_RAW + Self.BS * Self.RAW_OUT
    comptime _O_NEXT_Q_VALUES: Int = Self._O_Q_VALUES + Self.BS * Self.ACTIONS
    comptime _O_ONLINE_NEXT_Q: Int = Self._O_NEXT_Q_VALUES + Self.BS * Self.ACTIONS

    # --- Region 3: Targets ---
    comptime _O_TARGETS: Int = Self._O_ONLINE_NEXT_Q + Self.BS * Self.ACTIONS

    # --- Region 4: Cache ---
    comptime _O_CACHE: Int = Self._O_TARGETS + Self.BS

    # --- Region 5: Gradients ---
    comptime _O_GRAD_Q: Int = Self._O_CACHE + Self.BS * Self.CACHE_SIZE
    comptime _O_GRAD_RAW: Int = Self._O_GRAD_Q + Self.BS * Self.ACTIONS
    comptime _O_GRAD_INPUT: Int = Self._O_GRAD_RAW + Self.BS * Self.RAW_OUT

    # --- Region 6: Loss workspace ---
    comptime _O_LOSS_WS: Int = Self._O_GRAD_INPUT + Self.BS * Self.OBS

    # --- Total ---
    comptime TOTAL_SIZE: Int = Self._O_LOSS_WS + max(1, Self.GRAD_WS)

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    @staticmethod
    def alloc_gpu(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
        """Allocate a GPU buffer for this workspace."""
        return ctx.enqueue_create_buffer[dtype](Self.TOTAL_SIZE)

    # --- Region 1: Raw forward output views ---

    def q_raw(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.RAW_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.RAW_OUT), MutAnyOrigin
        ](self.ptr + Self._O_Q_RAW)

    def next_q_raw(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.RAW_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.RAW_OUT), MutAnyOrigin
        ](self.ptr + Self._O_NEXT_Q_RAW)

    def online_next_q_raw(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.RAW_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.RAW_OUT), MutAnyOrigin
        ](self.ptr + Self._O_ONLINE_NEXT_Q_RAW)

    # --- Region 2: Combined Q-value views ---

    def q_values(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.ACTIONS), MutAnyOrigin
        ](self.ptr + Self._O_Q_VALUES)

    def next_q_values(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.ACTIONS), MutAnyOrigin
        ](self.ptr + Self._O_NEXT_Q_VALUES)

    def online_next_q(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.ACTIONS), MutAnyOrigin
        ](self.ptr + Self._O_ONLINE_NEXT_Q)

    # --- Region 3: Targets ---

    def targets(
        self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.BS), MutAnyOrigin]:
        return LayoutTensor[dtype, Layout.row_major(Self.BS), MutAnyOrigin](
            self.ptr + Self._O_TARGETS
        )

    # --- Region 4: Cache ---

    def cache(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.CACHE_SIZE), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.CACHE_SIZE), MutAnyOrigin
        ](self.ptr + Self._O_CACHE)

    # --- Region 5: Gradient views ---

    def grad_q(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.ACTIONS), MutAnyOrigin
        ](self.ptr + Self._O_GRAD_Q)

    def grad_raw(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BS, Self.RAW_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.RAW_OUT), MutAnyOrigin
        ](self.ptr + Self._O_GRAD_RAW)

    def grad_input(
        self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.BS, Self.OBS), MutAnyOrigin]:
        return LayoutTensor[
            dtype, Layout.row_major(Self.BS, Self.OBS), MutAnyOrigin
        ](self.ptr + Self._O_GRAD_INPUT)

    # --- Region 6: Loss workspace pointer ---

    def loss_ws_ptr(self) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
        """Raw pointer to loss/gradient workspace region."""
        return self.ptr + Self._O_LOSS_WS


# =============================================================================
# DQNInferenceWS — Inference-time buffers (sized by max_n_envs)
# =============================================================================


struct DQNInferenceWS[
    MAX_N_ENVS: Int,
    ACTIONS: Int,
    RAW_OUT: Int,
](ImplicitlyCopyable, Movable):
    """Workspace for DQN inference-time raw output and Q-value views.

    Network workspace (forward_gpu) is kept as a separate DeviceBuffer
    because the Network API requires DeviceBuffer[dtype].
    """

    var ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    comptime _O_RAW: Int = 0
    comptime _O_Q: Int = Self._O_RAW + Self.MAX_N_ENVS * Self.RAW_OUT
    comptime TOTAL_SIZE: Int = Self._O_Q + Self.MAX_N_ENVS * Self.ACTIONS

    def __init__(out self, ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
        self.ptr = ptr

    def __init__(out self, *, copy: Self):
        self.ptr = copy.ptr

    def __init__(out self, *, deinit take: Self):
        self.ptr = take.ptr

    @staticmethod
    def alloc_gpu(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
        return ctx.enqueue_create_buffer[dtype](Self.TOTAL_SIZE)

    def raw[
        N_ENVS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.RAW_OUT), MutAnyOrigin
        ](self.ptr + Self._O_RAW)

    def q[
        N_ENVS: Int
    ](self) -> LayoutTensor[
        dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
    ]:
        return LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](self.ptr + Self._O_Q)


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
    comptime QGradStrat = AutodiffQGradient[]


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
    comptime QGradStrat = AutodiffQGradient[]


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
    comptime QGradStrat = AutodiffQGradient[]


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
    comptime QGradStrat = AutodiffQGradient[]


# AutodiffDQNConfig is now just an alias for DoubleDQNConfig (which uses autodiff by default)
comptime AutodiffDQNConfig = DoubleDQNConfig


# =============================================================================
# HuberDQNConfig -- Double DQN with Huber loss (robust to outliers)
# =============================================================================


struct HuberDQNConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 120,
    HIDDEN2: Int = 84,
    CAP: Int = 10000,
    BS: Int = 128,
    lr: Float64 = 2.5e-4,
    huber_delta: Float64 = 1.0,
](DiscreteOffPolicyConfig):
    """Double DQN with Huber loss — robust to large TD errors.

    Swapping MSE for Huber is a one-line change in the loss graph:
        AutodiffQGradient[MSELoss]    → standard DQN
        AutodiffQGradient[HuberLoss]  → robust DQN

    This demonstrates the composability of the autodiff system.
    """

    comptime NAME: String = "Huber DQN"
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
    comptime QGradStrat = AutodiffQGradient[HuberLoss[Self.huber_delta]]


# =============================================================================
# NoisyDQNConfig -- Double DQN with NoisyLinear layers (no epsilon-greedy)
# =============================================================================


struct NoisyDQNConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 128,
    HIDDEN2: Int = 128,
    CAP: Int = 10000,
    BS: Int = 128,
    lr: Float64 = 2.5e-4,
](DiscreteOffPolicyConfig):
    """Noisy DQN: replaces Linear with NoisyLinear for learned exploration.

    No epsilon-greedy needed — noise on weights provides exploration.
    Uses Double DQN target computation.

    Inference (forward without cache) uses mu-only weights for
    deterministic evaluation.
    """

    comptime NAME: String = "Noisy DQN"
    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        NoisyLinearReLU[Self.OBS, Self.HIDDEN],
        NoisyLinearReLU[Self.HIDDEN, Self.HIDDEN2],
        NoisyLinear[Self.HIDDEN2, Self.ACT],
    ]
    comptime QOpt = Adam[Self.lr]
    comptime QTargetStrat = DoubleQTarget
    comptime QOutputStrat = DirectQ
    comptime QGradStrat = AutodiffQGradient[]


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

    def __init__(out self):
        self.online = NetworkState[Self.QModel, Self.QOpt]()
        self.online.initialize[Xavier[]]()
        self.target = NetworkState[Self.QModel, Self.QOpt]()
        self.target.initialize[Xavier[]]()
        # Copy online -> target
        self.target.copy_params_from(self.online)
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

    def store[
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

    def is_ready(self) -> Bool:
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
    grad_ws_size: Int = 0,
](GPUOffPolicyState):
    """GPU-resident state for generic DQN training.

    Uses typed workspaces to consolidate GPU buffers:
      - DQNTrainWS: all training scratch (raw outputs, Q-values, cache, grads)
      - DQNInferenceWS: inference-time action selection buffers
      - SampleBatch: replay sample output (obs, act, rew, nobs, done)

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
        grad_ws_size: QGradient strategy workspace size.
    """

    comptime Q_Net = Network[Self.QModel, Self.QOpt]
    comptime CACHE_SIZE = Self.QModel.CACHE_SIZE
    comptime WS_PER_SAMPLE = Self.Q_Net.WORKSPACE_SIZE_PER_SAMPLE
    comptime RAW_OUT = Self.QModel.OUT_DIM

    # Workspace type aliases
    comptime TrainWS = DQNTrainWS[
        Self.batch_size,
        Self.obs_dim,
        Self.num_actions,
        Self.RAW_OUT,
        Self.CACHE_SIZE,
        Self.grad_ws_size,
    ]
    comptime InfWS = DQNInferenceWS[
        Self.max_n_envs,
        Self.num_actions,
        Self.RAW_OUT,
    ]
    # GPU network states (online + target)
    var online: GPUNetworkState[Self.QModel, Self.QOpt]
    var target: GPUNetworkState[Self.QModel, Self.QOpt]

    # GPU replay buffer (ACTION_DIM=1 default: scalar discrete action)
    var buffer: GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim]

    # Consolidated workspaces (single allocation each)
    var inf_buf: DeviceBuffer[dtype]  # backing for DQNInferenceWS
    var train_buf: DeviceBuffer[dtype]  # backing for DQNTrainWS

    # Separate DeviceBuffers required by Network API (takes DeviceBuffer[dtype])
    var net_ws: DeviceBuffer[dtype]  # [max(1, batch_size * WS_PER_SAMPLE)]
    var inf_net_ws: DeviceBuffer[dtype]  # [max(1, max_n_envs * WS_PER_SAMPLE)]
    var loss_ws: DeviceBuffer[dtype]  # [max(1, grad_ws_size)]

    # Replay sample output (separate DeviceBuffers required by GPUReplayBuffer.sample API)
    var s_obs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_act: DeviceBuffer[dtype]  # [batch_size]
    var s_rew: DeviceBuffer[dtype]  # [batch_size]
    var s_nobs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_done: DeviceBuffer[dtype]  # [batch_size]
    var s_idx: DeviceBuffer[DType.int32]  # [batch_size]

    # Diagnostic host buffers for GPU→CPU readback (pre-allocated)
    var diag_train_host: HostBuffer[
        dtype
    ]  # [TrainWS.TOTAL_SIZE] — full train workspace
    var diag_act_host: HostBuffer[dtype]  # [batch_size]
    var diag_rew_host: HostBuffer[dtype]  # [batch_size]
    var diag_done_host: HostBuffer[dtype]  # [batch_size]
    var rng_counter: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers. CPU weights are uploaded separately."""
        self.online = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.target = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.buffer = GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim](ctx)

        # Consolidated workspace allocations
        self.inf_buf = Self.InfWS.alloc_gpu(ctx)
        self.train_buf = Self.TrainWS.alloc_gpu(ctx)

        # Separate DeviceBuffers required by Network forward_gpu/backward_gpu
        self.net_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.batch_size * Self.WS_PER_SAMPLE)
        )
        self.inf_net_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.max_n_envs * Self.WS_PER_SAMPLE)
        )
        self.loss_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.grad_ws_size)
        )

        # Replay sample output buffers
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

        self.diag_train_host = ctx.enqueue_create_host_buffer[dtype](
            Self.TrainWS.TOTAL_SIZE
        )
        self.diag_act_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.diag_rew_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.diag_done_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.rng_counter = ctx.enqueue_create_buffer[DType.uint32](1)
        self.rng_counter.enqueue_fill(UInt32(0))

    # -------------------------------------------------------------------------
    # Workspace accessors
    # -------------------------------------------------------------------------

    def inference_ws(self) -> Self.InfWS:
        """Typed inference workspace views."""
        return Self.InfWS(self.inf_buf.unsafe_ptr())

    def train(self) -> Self.TrainWS:
        """Typed training workspace views."""
        return Self.TrainWS(self.train_buf.unsafe_ptr())

    # -------------------------------------------------------------------------
    # GPUOffPolicyState required methods
    # -------------------------------------------------------------------------

    def gpu_store[
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

    def gpu_buffer_is_ready(self) -> Bool:
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
    comptime GRAD_WS_SIZE: Int = Self.Config.QGradStrat.gpu_ws_size[
        Self.Config.batch_size, Self.Config.num_actions
    ]()
    comptime GPUStateType = DQNGPUStateGeneric[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.batch_size,
        Self.n_envs,
        Self.GRAD_WS_SIZE,
    ]

    # Persistent CPU state (for evaluate() after train/train_gpu)
    var state: Self.CPUStateType

    var gamma: Float64
    var tau: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64
    var exploration_fraction: Float64
    var target_update_freq: Int
    var train_step_count: Int
    var target_total_steps: Int
    var _target_update_ctr: Int
    var checkpoint_every: Int
    var checkpoint_path: String

    # Logger
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

    def __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 1.0,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.05,
        epsilon_decay: Float64 = 0.995,
        exploration_fraction: Float64 = 0.5,
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
        self.exploration_fraction = exploration_fraction
        self.target_update_freq = target_update_freq
        self.train_step_count = 0
        self.target_total_steps = target_total_steps
        self._target_update_ctr = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    def make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    def select_action[
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

    def store_transition[
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

    def do_cpu_train_step(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
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

    def decay_explore(mut self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def get_explore_rate(self) -> Float64:
        return self.epsilon

    def random_action(self) -> Int:
        return Int(random_float64() * Float64(Self.ACTIONS))

    def select_greedy_action(
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
    def save_checkpoint(self, path: String) raises -> None:
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

    def load_checkpoint(mut self, path: String) raises -> None:
        """Load DQN agent state from a checkpoint file."""
        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)

        self.state.online.read_sections(content, "online_")
        self.state.target.read_sections(content, "target_")

        var metadata = read_metadata_section(content)

        var gamma_str = get_metadata_value(metadata, "gamma")
        if gamma_str.byte_length() > 0:
            self.gamma = atof(gamma_str)

        var tau_str = get_metadata_value(metadata, "tau")
        if tau_str.byte_length() > 0:
            self.tau = atof(tau_str)

        var epsilon_str = get_metadata_value(metadata, "epsilon")
        if epsilon_str.byte_length() > 0:
            self.epsilon = atof(epsilon_str)

        var epsilon_min_str = get_metadata_value(metadata, "epsilon_min")
        if epsilon_min_str.byte_length() > 0:
            self.epsilon_min = atof(epsilon_min_str)

        var epsilon_decay_str = get_metadata_value(metadata, "epsilon_decay")
        if epsilon_decay_str.byte_length() > 0:
            self.epsilon_decay = atof(epsilon_decay_str)

        var train_step_str = get_metadata_value(metadata, "train_step_count")
        if train_step_str.byte_length() > 0:
            self.train_step_count = Int(atol(train_step_str))

    # =========================================================================
    # CPU Convenience training
    # =========================================================================

    def train[
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
        from mojo_rl.deep_agents.core.training.offpolicy_train import (
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

    def evaluate[
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

    def get_action_scale(self) -> Float64:
        return 1.0  # Discrete actions don't use action_scale

    def get_total_steps(self) -> Int:
        return self.train_step_count

    def set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    def make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for DQN training."""
        return Self.GPUStateType(ctx)

    def upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights to GPU online and target networks."""
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)

    def download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.online.download_to(self.state.online, ctx)
        gpu_state.target.download_to(self.state.target, ctx)

    def select_actions_gpu[
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
        var iws = gpu_state.inference_ws()

        # Forward pass: obs -> raw output
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = iws.raw[N_ENVS]()
        var p = gpu_state.online.params_view()
        Self.QNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.inf_net_ws
        )

        # Combine raw output to Q-values
        var q_t = iws.q[N_ENVS]()
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

        @parameter
        @always_inline
        def argmax_wrapper(
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

    def do_gpu_train_step(
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
            gpu_state.rng_counter,
            gpu_state.s_obs,
            gpu_state.s_act,
            gpu_state.s_rew,
            gpu_state.s_nobs,
            gpu_state.s_done,
            gpu_state.s_idx,
        )

        # Typed workspace views (replaces ~40 lines of manual LayoutTensor construction)
        var ws = gpu_state.train()

        # Sample batch views (obs, actions, rewards, dones still from separate DeviceBuffers)
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

        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()

        # ---- Phase 2: Online forward with cache -> raw -> combine ----
        var q_raw_t = ws.q_raw()
        var cache_t = ws.cache()
        Self.QNet.forward_gpu_with_cache[BATCH](
            ctx,
            obs_t,
            q_raw_t,
            p_online,
            cache_t,
            gpu_state.net_ws,
        )
        var q_t = ws.q_values()
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, q_raw_t, q_t
        )

        # ---- Phase 3: Target forward -> raw -> combine ----
        var next_q_raw_t = ws.next_q_raw()
        Self.QNet.forward_gpu[BATCH](
            ctx,
            next_obs_t,
            next_q_raw_t,
            p_target,
            gpu_state.net_ws,
        )
        var next_q_t = ws.next_q_values()
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, next_q_raw_t, next_q_t
        )

        # ---- Phase 3b: Online forward on next_obs -> raw -> combine (for Double DQN) ----
        var online_next_q_raw_t = ws.online_next_q_raw()
        Self.QNet.forward_gpu[BATCH](
            ctx, next_obs_t, online_next_q_raw_t, p_online, gpu_state.net_ws
        )
        var online_next_q_t = ws.online_next_q()
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, online_next_q_raw_t, online_next_q_t
        )

        # ---- Phase 4: Compute TD targets via strategy ----
        var targets_t = ws.targets()
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
        var grad_q_t = ws.grad_q()
        Self.Config.QGradStrat.compute_grad_gpu[BATCH, Self.ACTIONS](
            ctx, q_t, targets_t, actions_t, grad_q_t, gpu_state.loss_ws
        )

        # ---- Phase 5b: Transform grad from Q-space to raw output space ----
        var grad_raw_t = ws.grad_raw()
        Self.Config.QOutputStrat.grad_transform_gpu[
            BATCH, Self.ACTIONS, Self.RAW_OUT
        ](ctx, grad_q_t, grad_raw_t)

        # ---- Phase 6: Backward + optimizer step ----
        var grad_in_t = ws.grad_input()
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.QNet.backward_gpu[BATCH](
            ctx,
            grad_raw_t,
            grad_in_t,
            p_online,
            cache_t,
            g,
            gpu_state.net_ws,
        )
        gpu_state.online.optimizer_step(ctx)

        self.train_step_count += 1

        # ---- GPU Diagnostic logging ----
        if (
            self.logger
            and self.diag_every > 0
            and self.train_step_count % self.diag_every == 0
        ):
            try:
                # Copy training workspace and sample batch fields to host
                ctx.enqueue_copy(gpu_state.diag_train_host, gpu_state.train_buf)
                ctx.enqueue_copy(gpu_state.diag_act_host, gpu_state.s_act)
                ctx.enqueue_copy(gpu_state.diag_rew_host, gpu_state.s_rew)
                ctx.enqueue_copy(gpu_state.diag_done_host, gpu_state.s_done)
                ctx.synchronize()

                # Create workspace view over host buffer for typed access
                var diag_ws = Self.GPUStateType.TrainWS(
                    gpu_state.diag_train_host.unsafe_ptr()
                )
                var diag_q = diag_ws.q_values()
                var diag_tgt = diag_ws.targets()
                var diag_act_host = gpu_state.diag_act_host

                var step = self.train_step_count

                # Q-value stats
                var q_min = Float64(diag_q.ptr[0])
                var q_max = Float64(diag_q.ptr[0])
                var q_sum: Float64 = 0.0
                for i in range(BATCH * Self.ACTIONS):
                    var v = Float64(diag_q.ptr[i])
                    q_sum += v
                    if v < q_min:
                        q_min = v
                    if v > q_max:
                        q_max = v
                self.logger[].log_scalar(
                    "q_mean",
                    q_sum / Float64(BATCH * Self.ACTIONS),
                    step,
                )
                self.logger[].log_scalar("q_min", q_min, step)
                self.logger[].log_scalar("q_max", q_max, step)

                # TD target stats
                var tgt_sum: Float64 = 0.0
                var tgt_min = Float64(diag_tgt.ptr[0])
                var tgt_max = Float64(diag_tgt.ptr[0])
                for i in range(BATCH):
                    var v = Float64(diag_tgt.ptr[i])
                    tgt_sum += v
                    if v < tgt_min:
                        tgt_min = v
                    if v > tgt_max:
                        tgt_max = v
                self.logger[].log_scalar(
                    "td_target_mean",
                    tgt_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("td_target_min", tgt_min, step)
                self.logger[].log_scalar("td_target_max", tgt_max, step)

                # Done fraction and reward stats from sampled batch
                var done_count: Float64 = 0.0
                var rew_sum: Float64 = 0.0
                var rew_min: Float64 = Float64(gpu_state.diag_rew_host[0])
                var rew_max: Float64 = Float64(gpu_state.diag_rew_host[0])
                for b in range(BATCH):
                    var d = Float64(gpu_state.diag_done_host[b])
                    done_count += d
                    var r = Float64(gpu_state.diag_rew_host[b])
                    rew_sum += r
                    if r < rew_min:
                        rew_min = r
                    if r > rew_max:
                        rew_max = r
                self.logger[].log_scalar(
                    "done_fraction",
                    done_count / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar(
                    "reward_mean",
                    rew_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("reward_min", rew_min, step)
                self.logger[].log_scalar("reward_max", rew_max, step)

                # TD error stats
                var td_err_abs_sum: Float64 = 0.0
                var td_err_max_abs: Float64 = 0.0
                var total_loss: Float64 = 0.0
                for b in range(BATCH):
                    var act = Int(Float64(diag_act_host[b]))
                    var q_val = Float64(diag_q.ptr[b * Self.ACTIONS + act])
                    var tgt_val = Float64(diag_tgt.ptr[b])
                    var td_err = q_val - tgt_val
                    var abs_err = td_err if td_err >= 0.0 else -td_err
                    td_err_abs_sum += abs_err
                    total_loss += td_err * td_err
                    if abs_err > td_err_max_abs:
                        td_err_max_abs = abs_err
                self.logger[].log_scalar(
                    "td_error_abs_mean",
                    td_err_abs_sum / Float64(BATCH),
                    step,
                )
                self.logger[].log_scalar("td_error_max", td_err_max_abs, step)
                self.logger[].log_scalar(
                    "loss", total_loss / Float64(BATCH), step
                )
            except:
                pass

    def _gpu_train_kernels(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Pure GPU kernel sequence — calls do_gpu_train_step for now."""
        pass

    def _gpu_train_diagnostics(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        steps: Int,
    ) raises -> None:
        """CPU-side bookkeeping — no-op for DQN (inline in do_gpu_train_step)."""
        pass

    def soft_update_targets_gpu(
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

    def decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        """Linear epsilon schedule matching CleanRL:
        epsilon = max(end_e, start_e + (end_e - start_e) * t / duration).
        """
        var duration = Float64(num_steps) * self.exploration_fraction
        var slope = (self.epsilon_min - 1.0) / duration
        self.epsilon = max(
            self.epsilon_min,
            slope * Float64(total_steps) + 1.0,
        )

    # =========================================================================
    # GPU Training convenience
    # =========================================================================

    def train_gpu[
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

    CPU path uses PrioritizedReplayBuffer with sum-tree sampling.
    GPU path uses GPUPrioritizedReplayBuffer (CPU sum-tree + GPU data).
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
    comptime QGradStrat = AutodiffQGradient[]


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
    """CPU state for DQN+PER: online + target Q-networks + prioritized replay buffer.
    """

    comptime BUFFER_DTYPE = dtype

    var online: NetworkState[Self.QModel, Self.QOpt]
    var target: NetworkState[Self.QModel, Self.QOpt]
    var buffer: PrioritizedReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, dtype
    ]

    def __init__(out self, alpha: Float64 = 0.6, beta: Float64 = 0.4):
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

    def store[
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

    def is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# DQN PER GPU State (uses GPUPrioritizedReplayBuffer)
# =============================================================================


struct DQNPERGPUState[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    num_actions: Int,
    batch_size: Int,
    max_n_envs: Int,
    grad_ws_size: Int = 0,
](GPUOffPolicyState):
    """GPU state for DQN+PER: networks + prioritized replay buffer.

    Uses typed workspaces (DQNTrainWS, DQNInferenceWS) to consolidate
    training scratch buffers. Adds IS weights and TD errors buffers
    compared to standard DQN GPU state.
    """

    comptime Q_Net = Network[Self.QModel, Self.QOpt]
    comptime CACHE_SIZE = Self.QModel.CACHE_SIZE
    comptime WS_PER_SAMPLE = Self.Q_Net.WORKSPACE_SIZE_PER_SAMPLE
    comptime RAW_OUT = Self.QModel.OUT_DIM

    # Workspace type aliases (shared with standard DQN)
    comptime TrainWS = DQNTrainWS[
        Self.batch_size,
        Self.obs_dim,
        Self.num_actions,
        Self.RAW_OUT,
        Self.CACHE_SIZE,
        Self.grad_ws_size,
    ]
    comptime InfWS = DQNInferenceWS[
        Self.max_n_envs,
        Self.num_actions,
        Self.RAW_OUT,
    ]

    # GPU network states
    var online: GPUNetworkState[Self.QModel, Self.QOpt]
    var target: GPUNetworkState[Self.QModel, Self.QOpt]

    # GPU prioritized replay buffer
    var buffer: GPUPrioritizedReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, Self.batch_size
    ]

    # Consolidated workspaces
    var inf_buf: DeviceBuffer[dtype]  # backing for DQNInferenceWS
    var train_buf: DeviceBuffer[dtype]  # backing for DQNTrainWS

    # Separate DeviceBuffers required by Network API
    var net_ws: DeviceBuffer[dtype]  # [max(1, batch_size * WS_PER_SAMPLE)]
    var inf_net_ws: DeviceBuffer[dtype]  # [max(1, max_n_envs * WS_PER_SAMPLE)]
    var loss_ws: DeviceBuffer[dtype]  # [max(1, grad_ws_size)]

    # Replay sample output (separate DeviceBuffers required by GPUPrioritizedReplayBuffer.sample)
    var s_obs: DeviceBuffer[dtype]
    var s_act: DeviceBuffer[dtype]
    var s_rew: DeviceBuffer[dtype]
    var s_nobs: DeviceBuffer[dtype]
    var s_done: DeviceBuffer[dtype]
    var s_idx: DeviceBuffer[DType.int32]

    # PER-specific: IS weights and TD errors
    var s_weights: DeviceBuffer[dtype]
    var td_errors: DeviceBuffer[dtype]

    # Diagnostic
    var diag_q_host: HostBuffer[dtype]
    var diag_tgt_host: HostBuffer[dtype]
    var diag_act_host: HostBuffer[dtype]
    var diag_rew_host: HostBuffer[dtype]
    var diag_done_host: HostBuffer[dtype]

    def __init__(
        out self,
        ctx: DeviceContext,
        alpha: Float64 = 0.6,
        beta: Float64 = 0.4,
    ) raises:
        self.online = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.target = GPUNetworkState[Self.QModel, Self.QOpt](ctx)
        self.buffer = GPUPrioritizedReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, Self.batch_size
        ](ctx, alpha=alpha, beta=beta)

        # Consolidated workspace allocations
        self.inf_buf = Self.InfWS.alloc_gpu(ctx)
        self.train_buf = Self.TrainWS.alloc_gpu(ctx)

        # Separate DeviceBuffers required by Network API
        self.net_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.batch_size * Self.WS_PER_SAMPLE)
        )
        self.inf_net_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.max_n_envs * Self.WS_PER_SAMPLE)
        )
        self.loss_ws = ctx.enqueue_create_buffer[dtype](
            max(1, Self.grad_ws_size)
        )

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

        # PER buffers
        self.s_weights = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.td_errors = ctx.enqueue_create_buffer[dtype](Self.batch_size)

        var batch_q = Self.batch_size * Self.num_actions
        self.diag_q_host = ctx.enqueue_create_host_buffer[dtype](batch_q)
        self.diag_tgt_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.diag_act_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.diag_rew_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )
        self.diag_done_host = ctx.enqueue_create_host_buffer[dtype](
            Self.batch_size
        )

    # -------------------------------------------------------------------------
    # Workspace accessors
    # -------------------------------------------------------------------------

    def inference_ws(self) -> Self.InfWS:
        """Typed inference workspace views."""
        return Self.InfWS(self.inf_buf.unsafe_ptr())

    def train(self) -> Self.TrainWS:
        """Typed training workspace views."""
        return Self.TrainWS(self.train_buf.unsafe_ptr())

    # -------------------------------------------------------------------------
    # GPUOffPolicyState required methods
    # -------------------------------------------------------------------------

    def gpu_store[
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

    def gpu_buffer_is_ready(self) -> Bool:
        return self.buffer.gpu_buffer_is_ready()


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
    and priority updates. GPU path uses GPUPrioritizedReplayBuffer with
    CPU sum-tree sampling and GPU data gather.

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

    # GPU state uses prioritized replay
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1
    comptime BUFFER_CAPACITY: Int = Self.Config.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GRAD_WS_SIZE: Int = Self.Config.QGradStrat.gpu_ws_size[
        Self.Config.batch_size, Self.Config.num_actions
    ]()
    comptime GPUStateType = DQNPERGPUState[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.num_actions,
        Self.Config.batch_size,
        Self.n_envs,
        Self.GRAD_WS_SIZE,
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

    def __init__(
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

    def make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    def select_action[
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

    def store_transition[
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

    def do_cpu_train_step(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
        """One DQN+PER training step: sample with IS weights, update priorities.
        """
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
        Self.QNet.forward[Self.BATCH](next_obs_t, online_next_raw_t, p_online)
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

    def decay_explore(mut self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def get_explore_rate(self) -> Float64:
        return self.epsilon

    def random_action(self) -> Int:
        return Int(random_float64() * Float64(Self.ACTIONS))

    def select_greedy_action(
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
    def save_checkpoint(self, path: String) raises -> None:
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

    def load_checkpoint(mut self, path: String) raises -> None:
        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)
        self.state.online.read_sections(content, "online_")
        self.state.target.read_sections(content, "target_")

        var metadata = read_metadata_section(content)
        var gamma_str = get_metadata_value(metadata, "gamma")
        if gamma_str.byte_length() > 0:
            self.gamma = atof(gamma_str)
        var tau_str = get_metadata_value(metadata, "tau")
        if tau_str.byte_length() > 0:
            self.tau = atof(tau_str)
        var eps_str = get_metadata_value(metadata, "epsilon")
        if eps_str.byte_length() > 0:
            self.epsilon = atof(eps_str)
        var beta_str = get_metadata_value(metadata, "beta")
        if beta_str.byte_length() > 0:
            self.beta = atof(beta_str)
        var step_str = get_metadata_value(metadata, "train_step_count")
        if step_str.byte_length() > 0:
            self.train_step_count = Int(atol(step_str))

    # =========================================================================
    # CPU Training
    # =========================================================================

    def train[
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
        from mojo_rl.deep_agents.core.training.offpolicy_train import (
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

    def evaluate[
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

    def get_action_scale(self) -> Float64:
        return 1.0

    def get_total_steps(self) -> Int:
        return self.train_step_count

    def set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    def make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        return Self.GPUStateType(
            ctx,
            alpha=Float64(self.state.buffer.alpha),
            beta=Float64(self.beta_start),
        )

    def upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)

    def download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        gpu_state.online.download_to(self.state.online, ctx)
        gpu_state.target.download_to(self.state.target, ctx)

    def select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward Q-network on GPU + epsilon-greedy (same as standard DQN)."""
        var iws = gpu_state.inference_ws()

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = iws.raw[N_ENVS]()
        var p = gpu_state.online.params_view()
        Self.QNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.inf_net_ws
        )

        var q_t = iws.q[N_ENVS]()
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

        @parameter
        @always_inline
        def argmax_wrapper(
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
            var rng = PhiloxRandom(seed=UInt64(base_seed) + UInt64(b), offset=0)
            var rand_vals = rng.step_uniform()
            if Scalar[dtype](rand_vals[0]) < eps:
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

    def do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """GPU train step with prioritized experience replay.

        Phases: PER sample → forward → TD targets → IS-weighted gradient
                → backward → priority update.
        """
        comptime BATCH = Self.BATCH
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # ---- Phase 1: Priority-based sampling ----
        var progress = Scalar[dtype](
            Float64(self.train_step_count) / Float64(max(1, self.beta_frames))
        )
        if progress > Scalar[dtype](1.0):
            progress = Scalar[dtype](1.0)
        gpu_state.buffer.anneal_beta(progress, Scalar[dtype](self.beta_start))

        gpu_state.buffer.sample[BATCH](
            ctx,
            gpu_state.s_obs,
            gpu_state.s_act,
            gpu_state.s_rew,
            gpu_state.s_nobs,
            gpu_state.s_done,
            gpu_state.s_idx,
            gpu_state.s_weights,
        )

        # Typed workspace views (replaces ~18 lines of manual LayoutTensor construction)
        var ws = gpu_state.train()

        # Sample batch views (from separate DeviceBuffers)
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
        var weights_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_weights.unsafe_ptr())
        var td_errors_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.td_errors.unsafe_ptr())

        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()

        # ---- Phase 2: Forward passes (using workspace views) ----
        var q_raw_t = ws.q_raw()
        var cache_t = ws.cache()
        Self.QNet.forward_gpu_with_cache[BATCH](
            ctx, obs_t, q_raw_t, p_online, cache_t, gpu_state.net_ws
        )
        var q_t = ws.q_values()
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, q_raw_t, q_t
        )

        var next_q_raw_t = ws.next_q_raw()
        Self.QNet.forward_gpu[BATCH](
            ctx, next_obs_t, next_q_raw_t, p_target, gpu_state.net_ws
        )
        var next_q_t = ws.next_q_values()
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, next_q_raw_t, next_q_t
        )

        var online_next_q_raw_t = ws.online_next_q_raw()
        Self.QNet.forward_gpu[BATCH](
            ctx, next_obs_t, online_next_q_raw_t, p_online, gpu_state.net_ws
        )
        var online_next_q_t = ws.online_next_q()
        Self.Config.QOutputStrat.combine_gpu[BATCH, Self.ACTIONS, Self.RAW_OUT](
            ctx, online_next_q_raw_t, online_next_q_t
        )

        # ---- Phase 3: TD targets ----
        var targets_t = ws.targets()
        Self.Config.QTargetStrat.compute_targets_gpu[BATCH, Self.ACTIONS](
            ctx,
            targets_t,
            online_next_q_t,
            next_q_t,
            rewards_t,
            dones_t,
            self.gamma,
        )

        # ---- Phase 4: IS-weighted gradient + TD errors ----
        var grad_q_t = ws.grad_q()

        @parameter
        @always_inline
        def per_weighted_grad_kernel(
            grd: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            act: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            wgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            tde: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var action = Int(rebind[Scalar[dtype]](act[b]))
            var td_error = rebind[Scalar[dtype]](qv[b, action]) - rebind[
                Scalar[dtype]
            ](tgt[b])
            var weight = rebind[Scalar[dtype]](wgt[b])
            var weighted_error = weight * td_error

            # Store raw TD error for priority update
            tde[b] = td_error

            # IS-weighted MSE gradient: 2 * w * (Q(s,a) - target) / BATCH
            for a in range(Self.ACTIONS):
                if a == action:
                    grd[b, a] = (
                        Scalar[dtype](2.0)
                        * weighted_error
                        / Scalar[dtype](BATCH)
                    )
                else:
                    grd[b, a] = Scalar[dtype](0.0)

        ctx.enqueue_function[
            per_weighted_grad_kernel, per_weighted_grad_kernel
        ](
            grad_q_t,
            q_t,
            targets_t,
            actions_t,
            weights_t,
            td_errors_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 5: Backward + optimizer ----
        var grad_raw_t = ws.grad_raw()
        Self.Config.QOutputStrat.grad_transform_gpu[
            BATCH, Self.ACTIONS, Self.RAW_OUT
        ](ctx, grad_q_t, grad_raw_t)
        var grad_in_t = ws.grad_input()
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.QNet.backward_gpu[BATCH](
            ctx, grad_raw_t, grad_in_t, p_online, cache_t, g, gpu_state.net_ws
        )
        gpu_state.online.optimizer_step(ctx)

        self.train_step_count += 1

        # ---- Phase 6: Priority update (GPU→CPU) ----
        gpu_state.buffer.update_priorities[BATCH](ctx, gpu_state.td_errors)

    def _gpu_train_kernels(
        self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Pure GPU kernel sequence — calls do_gpu_train_step for now."""
        pass

    def _gpu_train_diagnostics(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        steps: Int,
    ) raises -> None:
        """CPU-side bookkeeping — no-op for DQN+PER (inline in do_gpu_train_step)."""
        pass

    def soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        if (
            self.train_step_count - self._target_update_ctr
            >= self.target_update_freq
        ):
            gpu_state.target.soft_update_from_gpu(
                gpu_state.online, self.tau, ctx
            )
            self._target_update_ctr = self.train_step_count

    def decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        var duration = Float64(num_steps) * 0.5
        var slope = (self.epsilon_min - 1.0) / duration
        self.epsilon = max(self.epsilon_min, slope * Float64(total_steps) + 1.0)

    # =========================================================================
    # GPU Training convenience
    # =========================================================================

    def train_gpu[
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
            target_total_steps=tgt_steps,
            logger=logger,
        )

        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^
