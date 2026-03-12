"""Dueling DQN GPU state container."""

from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.core import GPUOffPolicyState
from std.gpu.host import DeviceContext, DeviceBuffer


# =============================================================================
# DuelingDQNGPUState — GPU buffer container for Dueling DQN
# =============================================================================


struct DuelingDQNGPUState[
    DuelingModel: Model,
    Opt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    num_actions: Int,
    dueling_out: Int,
    batch_size: Int,
    max_n_envs: Int,
](GPUOffPolicyState):
    """GPU-resident state for Dueling DQN training.

    Holds all device buffers needed for one Dueling DQN GPU training loop:
      - Online and target GPU network states
      - GPU replay buffer (discrete action stored as float scalar index)
      - Inference scratch buffers (sized by max_n_envs)
      - Training scratch buffers (sample output, dueling output, Q caches, grad buffers)

    Created once at the start of GPU training via DuelingDQNAgent.make_gpu_state.
    CPU weights are uploaded separately via DuelingDQNAgent.upload_to_gpu.

    Parameters:
        DuelingModel: Dueling Q-network model type (backbone + Parallel[V, A]).
        Opt: Optimizer type.
        buffer_capacity: GPU replay buffer capacity.
        obs_dim: Observation space dimension.
        num_actions: Number of discrete actions.
        dueling_out: Dueling model output dimension (1 + num_actions).
        batch_size: Training batch size.
        max_n_envs: Max parallel environments (sizes inference buffers).
    """

    comptime DuelingNet = Network[Self.DuelingModel, Self.Opt]
    comptime CACHE_SIZE = Self.DuelingModel.CACHE_SIZE
    comptime WS_PER_SAMPLE = Self.DuelingNet.WORKSPACE_SIZE_PER_SAMPLE

    # GPU network states (online + target)
    var online: GPUNetworkState[Self.DuelingModel, Self.Opt]
    var target: GPUNetworkState[Self.DuelingModel, Self.Opt]

    # GPU replay buffer (ACTION_DIM=1 default: scalar discrete action)
    var buffer: GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim]

    # Inference buffers (max_n_envs sized, for select_actions_gpu)
    var env_dueling_buf: DeviceBuffer[dtype]  # [max_n_envs * dueling_out] raw model output
    var env_q_buf: DeviceBuffer[dtype]  # [max_n_envs * num_actions] combined Q values
    var inf_ws: DeviceBuffer[dtype]  # [max(1, max_n_envs * WS_PER_SAMPLE)]

    # Training scratch — replay sample output
    var s_obs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_act: DeviceBuffer[dtype]  # [batch_size]
    var s_rew: DeviceBuffer[dtype]  # [batch_size]
    var s_nobs: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var s_done: DeviceBuffer[dtype]  # [batch_size]
    var s_idx: DeviceBuffer[DType.int32]  # [batch_size]

    # Training scratch — dueling output forward/backward
    var dueling_out_buf: DeviceBuffer[dtype]  # [batch_size * dueling_out] raw online output
    var dueling_next_out: DeviceBuffer[dtype]  # [batch_size * dueling_out] raw target output
    var online_dueling_next_out: DeviceBuffer[dtype]  # [batch_size * dueling_out] Double DQN
    var q_values: DeviceBuffer[dtype]  # [batch_size * num_actions] combined Q
    var next_q_values: DeviceBuffer[dtype]  # [batch_size * num_actions] combined next Q
    var online_next_q: DeviceBuffer[dtype]  # [batch_size * num_actions] Double DQN
    var cache: DeviceBuffer[dtype]  # [batch_size * CACHE_SIZE]
    var targets: DeviceBuffer[dtype]  # [batch_size]
    var grad_output: DeviceBuffer[dtype]  # [batch_size * dueling_out] gradient in dueling space
    var grad_input: DeviceBuffer[dtype]  # [batch_size * obs_dim]
    var train_ws: DeviceBuffer[dtype]  # [max(1, batch_size * WS_PER_SAMPLE)]

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers. CPU weights are uploaded separately."""
        self.online = GPUNetworkState[Self.DuelingModel, Self.Opt](ctx)
        self.target = GPUNetworkState[Self.DuelingModel, Self.Opt](ctx)
        self.buffer = GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim](ctx)

        # Inference buffers
        self.env_dueling_buf = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.dueling_out
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

        # Training scratch — dueling
        var batch_dueling = Self.batch_size * Self.dueling_out
        var batch_q = Self.batch_size * Self.num_actions
        self.dueling_out_buf = ctx.enqueue_create_buffer[dtype](batch_dueling)
        self.dueling_next_out = ctx.enqueue_create_buffer[dtype](batch_dueling)
        self.online_dueling_next_out = ctx.enqueue_create_buffer[dtype](
            batch_dueling
        )
        self.q_values = ctx.enqueue_create_buffer[dtype](batch_q)
        self.next_q_values = ctx.enqueue_create_buffer[dtype](batch_q)
        self.online_next_q = ctx.enqueue_create_buffer[dtype](batch_q)
        self.cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CACHE_SIZE
        )
        self.targets = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.grad_output = ctx.enqueue_create_buffer[dtype](batch_dueling)
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
