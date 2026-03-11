"""DQN CPU and GPU state containers."""

from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.core import GPUOffPolicyState, OffPolicyDiscreteState
from std.gpu.host import DeviceContext, DeviceBuffer


# =============================================================================
# DQNGPUState — GPU buffer container for DQN
# =============================================================================


struct DQNGPUState[
    Q_Model: Model,
    Q_Opt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    num_actions: Int,
    batch_size: Int,
    max_n_envs: Int,
](GPUOffPolicyState):
    """GPU-resident state for DQN training.

    Holds all device buffers needed for one DQN GPU training loop:
      - Online and target GPU network states
      - GPU replay buffer (discrete action stored as float scalar index)
      - Inference scratch buffers (sized by max_n_envs)
      - Training scratch buffers (sample output, Q caches, grad buffers)

    Created once at the start of GPU training via DQNAgent.make_gpu_state.
    CPU weights are uploaded separately via DQNAgent.upload_to_gpu.

    Parameters:
        Q_Model: Q-network model type.
        Q_Opt: Q-network optimizer type.
        buffer_capacity: GPU replay buffer capacity.
        obs_dim: Observation space dimension.
        num_actions: Number of discrete actions.
        batch_size: Training batch size.
        max_n_envs: Max parallel environments (sizes inference buffers).
    """

    comptime Q_Net = Network[Self.Q_Model, Self.Q_Opt]
    comptime CACHE_SIZE = Self.Q_Model.CACHE_SIZE
    comptime WS_PER_SAMPLE = Self.Q_Net.WORKSPACE_SIZE_PER_SAMPLE

    # GPU network states (online + target)
    var online: GPUNetworkState[Self.Q_Model, Self.Q_Opt]
    var target: GPUNetworkState[Self.Q_Model, Self.Q_Opt]

    # GPU replay buffer (ACTION_DIM=1 default: scalar discrete action)
    var buffer: GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim]

    # Inference buffers (max_n_envs sized, for select_actions_gpu)
    var env_q_buf: DeviceBuffer[dtype]  # [max_n_envs * num_actions]
    var inf_ws: DeviceBuffer[dtype]  # [max(1, max_n_envs * WS_PER_SAMPLE)]

    # Training scratch — replay sample output
    var s_obs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_act: DeviceBuffer[dtype]  # [batch_size]
    var s_rew: DeviceBuffer[dtype]  # [batch_size]
    var s_nobs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_done: DeviceBuffer[dtype]  # [batch_size]
    var s_idx: DeviceBuffer[DType.int32]  # [batch_size]

    # Training scratch — Q-value forward/backward
    var q_values: DeviceBuffer[dtype]  # [batch_size * num_actions]
    var next_q_values: DeviceBuffer[dtype]  # [batch_size * num_actions]
    var online_next_q: DeviceBuffer[
        dtype
    ]  # [batch_size * num_actions] (Double DQN)
    var cache: DeviceBuffer[dtype]  # [batch_size * CACHE_SIZE]
    var targets: DeviceBuffer[dtype]  # [batch_size]
    var grad_output: DeviceBuffer[dtype]  # [batch_size * num_actions]
    var grad_input: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var train_ws: DeviceBuffer[dtype]  # [max(1, batch_size * WS_PER_SAMPLE)]

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers. CPU weights are uploaded separately."""
        self.online = GPUNetworkState[Self.Q_Model, Self.Q_Opt](ctx)
        self.target = GPUNetworkState[Self.Q_Model, Self.Q_Opt](ctx)
        self.buffer = GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim](ctx)

        # Inference buffers
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

        # Training scratch
        var batch_q_size = Self.batch_size * Self.num_actions
        self.q_values = ctx.enqueue_create_buffer[dtype](batch_q_size)
        self.next_q_values = ctx.enqueue_create_buffer[dtype](batch_q_size)
        self.online_next_q = ctx.enqueue_create_buffer[dtype](batch_q_size)
        self.cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CACHE_SIZE
        )
        self.targets = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.grad_output = ctx.enqueue_create_buffer[dtype](batch_q_size)
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
# DQNCPUState — CPU buffer container for DQN
# =============================================================================


struct DQNCPUState[
    Q_Model: Model,
    Q_Opt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    batch_size: Int,
](Movable, OffPolicyDiscreteState):
    """CPU-resident state for DQN training.

    Holds all heap-allocated data: online network, target network, replay buffer.
    Created via DQNAgent.make_cpu_state() and held by the caller (training loop
    or user code).

    Parameters:
        Q_Model: Q-network model type.
        Q_Opt: Q-network optimizer type.
        buffer_capacity: Replay buffer capacity.
        obs_dim: Observation space dimension.
        batch_size: Training batch size (used by is_ready()).
    """

    comptime BUFFER_DTYPE = dtype  # module-level float32; avoids shadowing in store()

    var online: NetworkState[Self.Q_Model, Self.Q_Opt]
    var target: NetworkState[Self.Q_Model, Self.Q_Opt]
    var buffer: HeapReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]

    fn __init__(out self):
        """Allocate and initialize online/target networks and replay buffer."""
        self.online = NetworkState[Self.Q_Model, Self.Q_Opt]()
        self.online.initialize[Kaiming[]]()
        self.target = NetworkState[Self.Q_Model, Self.Q_Opt](copy=self.online)
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

    fn __init__(out self, *, deinit take: Self):
        self.online = take.online^
        self.target = take.target^
        self.buffer = take.buffer^

    fn store[
        dtype: DType
    ](
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        """Push one discrete transition into the replay buffer."""
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        var action_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], 1](
            fill=Scalar[Self.BUFFER_DTYPE](action)
        )
        self.buffer.add(
            obs_arr,
            action_arr,
            Scalar[Self.BUFFER_DTYPE](reward),
            next_arr,
            done,
        )

    fn is_ready(self) -> Bool:
        """Return True if the replay buffer has enough samples to train."""
        return self.buffer.is_ready[Self.batch_size]()
