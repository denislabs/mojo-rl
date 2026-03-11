from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network, NetworkPair
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.deep_agents.core import GPUOffPolicyState, OffPolicyState
from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.training import GPUNetworkPair

# =============================================================================
# DDPGCPUState — CPU buffer container for DDPG
# =============================================================================


struct DDPGCPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
](Movable, OffPolicyState):
    """CPU-resident state for DDPG training.

    Holds all heap-allocated data needed for one DDPG training loop:
      - Actor and critic NetworkPairs (online + target weights, grads, optimizer)
      - CPU replay buffer
      - Pre-allocated scratch Lists for train_step (avoids per-call allocation)

    Created once in DeepDDPGAgent.__init__ via `Self.CPUStateType()`.
    Mirrors the DDPGGPUState pattern for symmetry.

    Parameters:
        ActorModel: Actor network model type.
        ActorOpt: Actor optimizer type.
        CriticModel: Critic network model type.
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
    comptime BUFFER_DTYPE = dtype  # module-level float32 constant; avoids shadowing in store()

    # Network pairs (online + target, params + grads + optimizer state)
    var actor: NetworkPair[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkPair[Self.CriticModel, Self.CriticOpt]

    # Replay buffer
    var buffer: HeapReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

    # Scratch — replay sample output (local InlineArrays are used for the actual
    # sample call; these Lists hold intermediate computations)
    var _batch_obs: List[Scalar[dtype]]  # [BATCH * OBS]
    var _batch_act: List[Scalar[dtype]]  # [BATCH * ACTIONS]
    var _batch_rew: List[Scalar[dtype]]  # [BATCH]
    var _batch_next: List[Scalar[dtype]]  # [BATCH * OBS]
    var _batch_done: List[Scalar[dtype]]  # [BATCH]

    # Scratch — TD target computation
    var _next_act: List[Scalar[dtype]]  # [BATCH * ACTIONS]
    var _next_ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _next_q: List[Scalar[dtype]]  # [BATCH]
    var _targets: List[Scalar[dtype]]  # [BATCH]

    # Scratch — critic update
    var _ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _q_out: List[Scalar[dtype]]  # [BATCH]
    var _q_cache: List[Scalar[dtype]]  # [BATCH * CriticModel.CACHE_SIZE]
    var _q_grad: List[Scalar[dtype]]  # [BATCH]
    var _d_ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]

    # Scratch — actor update
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
        self.actor.initialize[Xavier[]]()
        self.critic = NetworkPair[Self.CriticModel, Self.CriticOpt]()
        self.critic.initialize[Kaiming[]]()

        self.buffer = HeapReplayBuffer[
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
        self._next_q = List[Scalar[dtype]](capacity=Self.BATCH)
        self._targets = List[Scalar[dtype]](capacity=Self.BATCH)
        self._ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._q_out = List[Scalar[dtype]](capacity=Self.BATCH)
        self._q_cache = List[Scalar[dtype]](
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
            self._next_q.append(Scalar[dtype](0))
            self._targets.append(Scalar[dtype](0))
            self._q_out.append(Scalar[dtype](0))
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
            self._q_cache.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.ActorModel.CACHE_SIZE):
            self._actor_cache.append(Scalar[dtype](0))

    fn store[
        dtype: DType
    ](
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        """Push one transition into the replay buffer.

        Expects action already normalized to actor output range ([-1, 1]).
        The `dtype` parameter is the observation dtype (may differ from
        BUFFER_DTYPE which is the internal float32 storage type).
        """
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.OBS](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.OBS](
            uninitialized=True
        )
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        for i in range(Self.ACTIONS):
            act_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(action[i]))
        self.buffer.add(
            obs_arr, act_arr, Scalar[Self.BUFFER_DTYPE](reward), next_arr, done
        )

    fn is_ready(self) -> Bool:
        """Return True if the replay buffer has enough samples to train."""
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# DDPGGPUState — GPU buffer container for DDPG
# =============================================================================


struct DDPGGPUState[
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
    """GPU-resident state for DDPG training.

    Holds all device buffers needed for one DDPG training loop:
      - Four GPU network states (actor + critic, online + target)
      - GPU replay buffer
      - Exploration RNG states + inference scratch (sized by Self.max_n_envs)
      - Training scratch buffers (sample output, Q caches, grad buffers)

    Created once at the start of GPU training via DeepDDPGAgent.make_gpu_state.
    CPU weights are uploaded separately via DeepDDPGAgent.upload_to_gpu.

    Parameters:
        ActorModel: Actor network model type.
        ActorOpt: Actor optimizer type.
        CriticModel: Critic network model type.
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

    # GPU network states
    var actor: GPUNetworkPair[Self.ActorModel, Self.ActorOpt]
    var critic: GPUNetworkPair[Self.CriticModel, Self.CriticOpt]

    # GPU replay buffer
    var buffer: GPUReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim
    ]

    # Exploration buffers (sized by Self.max_n_envs)
    var raw_act: DeviceBuffer[dtype]  # [Self.max_n_envs * Self.action_dim]
    var inf_ws: DeviceBuffer[dtype]  # [Self.max_n_envs * ACTOR_WS]

    # Training scratch — replay sample output
    var s_obs: DeviceBuffer[dtype]  # [Self.batch_size * Self.obs_dim]
    var s_act: DeviceBuffer[dtype]  # [Self.batch_size * Self.action_dim]
    var s_rew: DeviceBuffer[dtype]  # [Self.batch_size]
    var s_nobs: DeviceBuffer[dtype]  # [Self.batch_size * Self.obs_dim]
    var s_done: DeviceBuffer[dtype]  # [Self.batch_size]
    var s_idx: DeviceBuffer[DType.int32]  # [Self.batch_size]

    # Training scratch — TD target computation
    var next_act: DeviceBuffer[dtype]  # [Self.batch_size * Self.action_dim]
    var next_ci: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]
    var next_q: DeviceBuffer[dtype]  # [Self.batch_size * 1]
    var targets: DeviceBuffer[dtype]  # [Self.batch_size]

    # Training scratch — critic update
    var ci: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]
    var q_out: DeviceBuffer[dtype]  # [Self.batch_size * 1]
    var q_cache: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_CS]
    var critic_ws: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_WS]
    var q_grad: DeviceBuffer[dtype]  # [Self.batch_size * 1]
    var d_ci: DeviceBuffer[dtype]  # [Self.batch_size * CRITIC_IN]

    # Training scratch — actor update
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

    fn __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers. CPU weights are uploaded separately."""
        self.actor = GPUNetworkPair[Self.ActorModel, Self.ActorOpt](ctx)
        self.critic = GPUNetworkPair[Self.CriticModel, Self.CriticOpt](ctx)
        self.buffer = GPUReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim
        ](ctx)

        # Exploration buffers
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

        # TD target computation
        self.next_act = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.action_dim
        )
        self.next_ci = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )
        self.next_q = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.targets = ctx.enqueue_create_buffer[dtype](Self.batch_size)

        # Critic update
        self.ci = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )
        self.q_out = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.q_cache = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_CS
        )
        self.critic_ws = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_WS
        )
        self.q_grad = ctx.enqueue_create_buffer[dtype](Self.batch_size * 1)
        self.d_ci = ctx.enqueue_create_buffer[dtype](
            Self.batch_size * Self.CRITIC_IN
        )

        # Actor update
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

        # Pre-fill dq with -1/Self.batch_size (constant policy-gradient weight)
        ctx.synchronize()
        var dq_host = ctx.enqueue_create_host_buffer[dtype](Self.batch_size)
        for i in range(Self.batch_size):
            dq_host[i] = Scalar[dtype](-1.0 / Float64(Self.batch_size))
        ctx.enqueue_copy(self.dq, dq_host)

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
