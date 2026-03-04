from nn.model import Model
from nn.optimizer import Optimizer
from nn.training import Network, NetworkPair
from nn.replay import ReplayBuffer, GPUReplayBuffer
from nn.constants import dtype
from nn.initializer import Xavier, Kaiming
from core import GPUOffPolicyState
from std.gpu.host import DeviceContext, DeviceBuffer
from nn.training import GPUNetworkPair

# =============================================================================
# TD3CPUState — CPU buffer container for TD3
# =============================================================================


struct TD3CPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
]:
    """CPU-resident state for TD3 training.

    Holds all heap-allocated data needed for one TD3 training loop:
      - Actor and twin-critic NetworkPairs (online + target, params + grads + optimizer)
      - CPU replay buffer
      - Pre-allocated scratch Lists for train_step (avoids per-call allocation)

    Created once in DeepTD3Agent.__init__ via `Self.CPUStateType()`.
    Mirrors the TD3GPUState pattern for symmetry.

    Parameters:
        ActorModel: Actor network model type.
        ActorOpt: Actor optimizer type.
        CriticModel: Critic network model type (shared by both critics).
        CriticOpt: Critic optimizer type.
        buffer_capacity: Replay buffer capacity.
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        batch_size: Training batch size.
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime BATCH = Self.batch_size
    comptime CRITIC_IN = Self.OBS + Self.ACTIONS

    # Network pairs: actor + twin critics (online + target each)
    var actor: NetworkPair[Self.ActorModel, Self.ActorOpt]
    var critic1: NetworkPair[Self.CriticModel, Self.CriticOpt]
    var critic2: NetworkPair[Self.CriticModel, Self.CriticOpt]

    # Replay buffer
    var buffer: ReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

    # Scratch — replay sample output
    var _batch_obs: List[Scalar[dtype]]  # [BATCH * OBS]
    var _batch_act: List[Scalar[dtype]]  # [BATCH * ACTIONS]
    var _batch_rew: List[Scalar[dtype]]  # [BATCH]
    var _batch_next: List[Scalar[dtype]]  # [BATCH * OBS]
    var _batch_done: List[Scalar[dtype]]  # [BATCH]

    # Scratch — TD target computation (TD3: twin critics + target smoothing)
    var _next_act: List[Scalar[dtype]]  # [BATCH * ACTIONS]
    var _next_ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _nq1: List[Scalar[dtype]]  # [BATCH] critic1_target output
    var _nq2: List[Scalar[dtype]]  # [BATCH] critic2_target output
    var _targets: List[Scalar[dtype]]  # [BATCH]

    # Scratch — critic updates (reused buffers for both critics)
    var _ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _q1_out: List[Scalar[dtype]]  # [BATCH]
    var _q2_out: List[Scalar[dtype]]  # [BATCH]
    var _q1_cache: List[Scalar[dtype]]  # [BATCH * CriticModel.CACHE_SIZE]
    var _q2_cache: List[Scalar[dtype]]  # [BATCH * CriticModel.CACHE_SIZE]
    var _q_grad: List[Scalar[dtype]]  # [BATCH] reused for q1_grad and q2_grad
    var _d_ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN] reused for d_ci1/d_ci2

    # Scratch — delayed actor update
    var _actor_act: List[Scalar[dtype]]  # [BATCH * ACTIONS]
    var _actor_cache: List[Scalar[dtype]]  # [BATCH * ActorModel.CACHE_SIZE]
    var _new_ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _new_q: List[Scalar[dtype]]  # [BATCH]
    var _dq: List[Scalar[dtype]]  # [BATCH]  constant -1/BATCH
    var _d_new_ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _d_act: List[Scalar[dtype]]  # [BATCH * ACTIONS]
    var _d_obs: List[Scalar[dtype]]  # [BATCH * OBS]

    fn __init__(out self):
        """Allocate networks, replay buffer, and all scratch buffers."""
        self.actor = NetworkPair[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier]()
        self.critic1 = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        self.critic1.initialize[Kaiming]()
        self.critic2 = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        self.critic2.initialize[Kaiming]()

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        # Allocate scratch with capacity
        self._batch_obs = List[Scalar[dtype]](capacity=Self.BATCH * Self.OBS)
        self._batch_act = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ACTIONS
        )
        self._batch_rew = List[Scalar[dtype]](capacity=Self.BATCH)
        self._batch_next = List[Scalar[dtype]](capacity=Self.BATCH * Self.OBS)
        self._batch_done = List[Scalar[dtype]](capacity=Self.BATCH)
        self._next_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._next_ci = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CRITIC_IN
        )
        self._nq1 = List[Scalar[dtype]](capacity=Self.BATCH)
        self._nq2 = List[Scalar[dtype]](capacity=Self.BATCH)
        self._targets = List[Scalar[dtype]](capacity=Self.BATCH)
        self._ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._q1_out = List[Scalar[dtype]](capacity=Self.BATCH)
        self._q2_out = List[Scalar[dtype]](capacity=Self.BATCH)
        self._q1_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CriticModel.CACHE_SIZE
        )
        self._q2_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CriticModel.CACHE_SIZE
        )
        self._q_grad = List[Scalar[dtype]](capacity=Self.BATCH)
        self._d_ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._actor_act = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ACTIONS
        )
        self._actor_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ActorModel.CACHE_SIZE
        )
        self._new_ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._new_q = List[Scalar[dtype]](capacity=Self.BATCH)
        self._dq = List[Scalar[dtype]](capacity=Self.BATCH)
        self._d_new_ci = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CRITIC_IN
        )
        self._d_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._d_obs = List[Scalar[dtype]](capacity=Self.BATCH * Self.OBS)

        # Fill scratch with zeros so LayoutTensor views are valid from the start
        for _ in range(Self.BATCH * Self.OBS):
            self._batch_obs.append(Scalar[dtype](0))
            self._batch_next.append(Scalar[dtype](0))
            self._d_obs.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.ACTIONS):
            self._batch_act.append(Scalar[dtype](0))
            self._next_act.append(Scalar[dtype](0))
            self._actor_act.append(Scalar[dtype](0))
            self._d_act.append(Scalar[dtype](0))
        for _ in range(Self.BATCH):
            self._batch_rew.append(Scalar[dtype](0))
            self._batch_done.append(Scalar[dtype](0))
            self._nq1.append(Scalar[dtype](0))
            self._nq2.append(Scalar[dtype](0))
            self._targets.append(Scalar[dtype](0))
            self._q1_out.append(Scalar[dtype](0))
            self._q2_out.append(Scalar[dtype](0))
            self._q_grad.append(Scalar[dtype](0))
            self._new_q.append(Scalar[dtype](0))
            self._dq.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.CRITIC_IN):
            self._next_ci.append(Scalar[dtype](0))
            self._ci.append(Scalar[dtype](0))
            self._d_ci.append(Scalar[dtype](0))
            self._new_ci.append(Scalar[dtype](0))
            self._d_new_ci.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.CriticModel.CACHE_SIZE):
            self._q1_cache.append(Scalar[dtype](0))
            self._q2_cache.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.ActorModel.CACHE_SIZE):
            self._actor_cache.append(Scalar[dtype](0))

    fn is_ready(self) -> Bool:
        """Return True if the replay buffer has enough samples to train."""
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# TD3GPUState — GPU buffer container for TD3
# =============================================================================


struct TD3GPUState[
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
    """GPU-resident state for TD3 training.

    Holds all device buffers needed for one TD3 training loop:
      - Six GPU network states (actor + 2 critics, each online + target)
      - GPU replay buffer
      - Exploration RNG states + inference scratch (sized by Self.max_n_envs)
      - Training scratch buffers (sample output, Q caches, grad buffers)
      - TD3-specific: target-smoothing noise buffer

    Created once at the start of GPU training via DeepTD3Agent.make_gpu_state.
    CPU weights are uploaded separately via DeepTD3Agent.upload_to_gpu.

    Parameters:
        ActorModel: Actor network model type.
        ActorOpt: Actor optimizer type.
        CriticModel: Critic network model type (shared between critic1 and critic2).
        CriticOpt: Critic optimizer type.
        buffer_capacity: GPU replay buffer capacity.
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        batch_size: Training batch size.
        max_n_envs: Max parallel environments (sizes exploration buffers).
    """

    comptime CRITIC_IN = Self.obs_dim + Self.action_dim
    comptime ACTOR_CS = Self.ActorModel.CACHE_SIZE
    comptime CRITIC_CS = Self.CriticModel.CACHE_SIZE
    comptime ActorNet = Network[Self.ActorModel, Self.ActorOpt]
    comptime CriticNet = Network[Self.CriticModel, Self.CriticOpt]
    comptime ACTOR_WS = Self.ActorNet.WORKSPACE_SIZE_PER_SAMPLE
    comptime CRITIC_WS = Self.CriticNet.WORKSPACE_SIZE_PER_SAMPLE

    # GPU network states: actor (online+target), critic1 (online+target), critic2 (online+target)
    var actor: GPUNetworkPair[Self.ActorModel, Self.ActorOpt]
    var critic1: GPUNetworkPair[Self.CriticModel, Self.CriticOpt]
    var critic2: GPUNetworkPair[Self.CriticModel, Self.CriticOpt]

    # GPU replay buffer
    var buffer: GPUReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim
    ]

    # Exploration buffers (sized by Self.max_n_envs)
    var rng_states: DeviceBuffer[
        DType.uint32
    ]  # [Self.max_n_envs * Self.action_dim]
    var raw_act: DeviceBuffer[dtype]  # [Self.max_n_envs * Self.action_dim]
    var inf_ws: DeviceBuffer[dtype]  # [Self.max_n_envs * ACTOR_WS]

    # Training scratch — replay sample output
    var s_obs: DeviceBuffer[dtype]  # [Self.batch_size * Self.obs_dim]
    var s_act: DeviceBuffer[dtype]  # [Self.batch_size * Self.action_dim]
    var s_rew: DeviceBuffer[dtype]  # [Self.batch_size]
    var s_nobs: DeviceBuffer[dtype]  # [Self.batch_size * Self.obs_dim]
    var s_done: DeviceBuffer[dtype]  # [Self.batch_size]
    var s_idx: DeviceBuffer[DType.int32]  # [Self.batch_size]

    # Training scratch — target computation (TD3: twin critics)
    var next_act: DeviceBuffer[
        dtype
    ]  # [Self.batch_size * Self.action_dim] clean target actor output
    var noisy_next_act: DeviceBuffer[
        dtype
    ]  # [Self.batch_size * Self.action_dim] smoothed target actions
    var next_ci: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]
    var nq1: DeviceBuffer[dtype]  # [Self.batch_size] critic1_target output
    var nq2: DeviceBuffer[dtype]  # [Self.batch_size] critic2_target output
    var targets: DeviceBuffer[dtype]  # [Self.batch_size] TD targets

    # Training scratch — critic1 update
    var ci: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]
    var q1_out: DeviceBuffer[dtype]  # [Self.batch_size * 1]
    var q1_cache: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_CS]
    var critic1_ws: DeviceBuffer[dtype]  # workspace
    var q1_grad: DeviceBuffer[dtype]  # [Self.batch_size * 1]
    var d_ci1: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]

    # Training scratch — critic2 update
    var q2_out: DeviceBuffer[dtype]  # [Self.batch_size * 1]
    var q2_cache: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_CS]
    var critic2_ws: DeviceBuffer[dtype]  # workspace
    var q2_grad: DeviceBuffer[dtype]  # [Self.batch_size * 1]
    var d_ci2: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]

    # Training scratch — actor update (delayed)
    var actor_act: DeviceBuffer[dtype]  # [Self.batch_size * Self.action_dim]
    var new_ci: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]
    var new_q: DeviceBuffer[dtype]  # [Self.batch_size * 1]
    var new_q_cache: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_CS]
    var actor_cache: DeviceBuffer[dtype]  # [Self.batch_size * ACTOR_CS]
    var actor_ws: DeviceBuffer[dtype]  # [Self.batch_size * ACTOR_WS]
    var dq: DeviceBuffer[
        dtype
    ]  # [Self.batch_size * 1] constant -1/Self.batch_size
    var d_new_ci: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]
    var d_act: DeviceBuffer[dtype]  # [Self.batch_size * Self.action_dim]
    var d_obs: DeviceBuffer[dtype]  # [Self.batch_size * Self.obs_dim]

    # TD3-specific: target policy smoothing RNG (separate from exploration RNG)
    var td3_noise_rng: DeviceBuffer[
        DType.uint32
    ]  # [Self.batch_size * Self.action_dim]

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers. CPU weights are uploaded separately."""
        self.actor = GPUNetworkPair[Self.ActorModel, Self.ActorOpt](ctx)
        self.critic1 = GPUNetworkPair[Self.CriticModel, Self.CriticOpt](ctx)
        self.critic2 = GPUNetworkPair[Self.CriticModel, Self.CriticOpt](ctx)
        self.buffer = GPUReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim
        ](ctx)

        # Exploration buffers
        self.rng_states = ctx.enqueue_create_buffer[DType.uint32](
            Self.max_n_envs * Self.action_dim
        )
        self.raw_act = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.action_dim
        )
        self.inf_ws = ctx.enqueue_create_buffer[dtype](
            Self.max_n_envs * Self.ACTOR_WS
        )

        # Replay sample output
        self.s_obs = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        self.s_act = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.action_dim
        )
        self.s_rew = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.s_nobs = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )
        self.s_done = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](Self.batch_size)

        # Target computation
        self.next_act = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.action_dim
        )
        self.noisy_next_act = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.action_dim
        )
        self.next_ci = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )
        self.nq1 = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.nq2 = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        self.targets = ctx.enqueue_create_buffer[dtype](Self.batch_size)

        # Critic1 update
        self.ci = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )
        self.q1_out = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.q1_cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_CS
        )
        self.critic1_ws = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_WS
        )
        self.q1_grad = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.d_ci1 = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )

        # Critic2 update
        self.q2_out = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.q2_cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_CS
        )
        self.critic2_ws = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_WS
        )
        self.q2_grad = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.d_ci2 = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )

        # Actor update (delayed)
        self.actor_act = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.action_dim
        )
        self.new_ci = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )
        self.new_q = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.new_q_cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_CS
        )
        self.actor_cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.ACTOR_CS
        )
        self.actor_ws = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.ACTOR_WS
        )
        self.dq = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.d_new_ci = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )
        self.d_act = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.action_dim
        )
        self.d_obs = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.obs_dim
        )

        # Target policy smoothing RNG
        self.td3_noise_rng = ctx.enqueue_create_buffer[DType.uint32](
            Self.batch_size * Self.action_dim
        )

        # Pre-fill dq with -1/Self.batch_size (constant policy-gradient weight)
        ctx.synchronize()
        var dq_host = ctx.enqueue_create_host_buffer[dtype](Self.batch_size)
        for i in range(Self.batch_size):
            dq_host[i] = Scalar[dtype](-1.0 / Float64(Self.batch_size))
        ctx.enqueue_copy(self.dq, dq_host)

        # Initialize persistent exploration RNG states (xorshift32)
        ctx.synchronize()
        var rng_host = ctx.enqueue_create_host_buffer[DType.uint32](
            Self.max_n_envs * Self.action_dim
        )
        var rng_s: UInt32 = 12345
        for i in range(Self.max_n_envs * Self.action_dim):
            rng_s = rng_s ^ (rng_s << 13)
            rng_s = rng_s ^ (rng_s >> 17)
            rng_s = rng_s ^ (rng_s << 5)
            rng_host[i] = rng_s
        ctx.enqueue_copy(self.rng_states, rng_host)

        # Initialize target-smoothing RNG states (separate seed)
        ctx.synchronize()
        var noise_rng_host = ctx.enqueue_create_host_buffer[DType.uint32](
            Self.batch_size * Self.action_dim
        )
        var noise_s: UInt32 = 54321
        for i in range(Self.batch_size * Self.action_dim):
            noise_s = noise_s ^ (noise_s << 13)
            noise_s = noise_s ^ (noise_s >> 17)
            noise_s = noise_s ^ (noise_s << 5)
            noise_rng_host[i] = noise_s
        ctx.enqueue_copy(self.td3_noise_rng, noise_rng_host)

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
        """Return True if the GPU replay buffer has at least Self.batch_size samples.
        """
        return self.buffer.is_ready[Self.batch_size]()
