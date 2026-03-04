from nn.model import Model
from nn.optimizer import Optimizer
from nn.training import Network, NetworkState, NetworkPair, GPUNetworkState, GPUNetworkPair
from nn.replay import ReplayBuffer, GPUReplayBuffer
from nn.constants import dtype
from nn.initializer import Kaiming
from deep_agents.core import OffPolicyState, GPUOffPolicyState
from std.gpu.host import DeviceContext, DeviceBuffer

# =============================================================================
# SACCPUState — CPU buffer container for SAC
# =============================================================================


struct SACCPUState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    batch_size: Int,
](Movable, OffPolicyState):
    """CPU-resident state for SAC training.

    Holds all heap-allocated data needed for one SAC training loop:
      - Actor NetworkState (online only — SAC has no target actor)
      - Twin-critic NetworkPairs (online + target each)
      - CPU replay buffer
      - Pre-allocated scratch Lists for train_step (avoids per-call allocation)

    Created once in DeepSACAgent.__init__ via `Self.CPUStateType()`.
    Mirrors the DDPG/TD3 CPUState pattern.

    Parameters:
        ActorModel: Actor network model type (StochasticActor output).
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
    comptime BUFFER_DTYPE = dtype
    # StochasticActor outputs mean + log_std
    comptime ACTOR_OUT = Self.ACTIONS * 2

    # Actor: online only (SAC has no target actor)
    var actor: NetworkState[Self.ActorModel, Self.ActorOpt]

    # Twin critics: online + target each
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

    # Scratch — TD target computation (next-state actor + twin critics)
    var _next_out: List[Scalar[dtype]]  # [BATCH * ACTOR_OUT] actor mean+log_std
    var _next_act: List[Scalar[dtype]]  # [BATCH * ACTIONS] sampled next actions
    var _next_log_pi: List[Scalar[dtype]]  # [BATCH] log_probs for next actions
    var _next_ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _nq1: List[Scalar[dtype]]  # [BATCH] critic1_target output
    var _nq2: List[Scalar[dtype]]  # [BATCH] critic2_target output
    var _targets: List[Scalar[dtype]]  # [BATCH]

    # Scratch — critic updates (reused for both critics)
    var _ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _q1_out: List[Scalar[dtype]]  # [BATCH]
    var _q2_out: List[Scalar[dtype]]  # [BATCH]
    var _q1_cache: List[Scalar[dtype]]  # [BATCH * CriticModel.CACHE_SIZE]
    var _q2_cache: List[Scalar[dtype]]  # [BATCH * CriticModel.CACHE_SIZE]
    var _q_grad: List[Scalar[dtype]]  # [BATCH] reused for q1_grad, q2_grad, dq
    var _d_ci: List[
        Scalar[dtype]
    ]  # [BATCH * CRITIC_IN] reused for d_c1/d_c2/d_new_ci

    # Scratch — actor update
    var _curr_out: List[Scalar[dtype]]  # [BATCH * ACTOR_OUT] actor mean+log_std
    var _curr_act: List[
        Scalar[dtype]
    ]  # [BATCH * ACTIONS] current sampled actions
    var _curr_log_pi: List[
        Scalar[dtype]
    ]  # [BATCH] log_probs for current actions
    var _actor_cache: List[Scalar[dtype]]  # [BATCH * ActorModel.CACHE_SIZE]
    var _new_ci: List[Scalar[dtype]]  # [BATCH * CRITIC_IN]
    var _new_q1: List[Scalar[dtype]]  # [BATCH]
    var _new_c1_cache: List[Scalar[dtype]]  # [BATCH * CriticModel.CACHE_SIZE]
    var _actor_grad_arr: List[Scalar[dtype]]  # [BATCH * ACTOR_OUT]
    var _grad_act: List[Scalar[dtype]]  # [BATCH * ACTIONS]
    var _d_obs: List[Scalar[dtype]]  # [BATCH * OBS]

    fn __init__(out self):
        """Allocate networks, replay buffer, and all scratch buffers."""
        self.actor = NetworkState[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Kaiming]()
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
        self._next_out = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ACTOR_OUT
        )
        self._next_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._next_log_pi = List[Scalar[dtype]](capacity=Self.BATCH)
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
        self._curr_out = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ACTOR_OUT
        )
        self._curr_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._curr_log_pi = List[Scalar[dtype]](capacity=Self.BATCH)
        self._actor_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ActorModel.CACHE_SIZE
        )
        self._new_ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._new_q1 = List[Scalar[dtype]](capacity=Self.BATCH)
        self._new_c1_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CriticModel.CACHE_SIZE
        )
        self._actor_grad_arr = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ACTOR_OUT
        )
        self._grad_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._d_obs = List[Scalar[dtype]](capacity=Self.BATCH * Self.OBS)

        # Fill scratch with zeros so LayoutTensor views are valid from the start
        for _ in range(Self.BATCH * Self.OBS):
            self._batch_obs.append(Scalar[dtype](0))
            self._batch_next.append(Scalar[dtype](0))
            self._d_obs.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.ACTIONS):
            self._batch_act.append(Scalar[dtype](0))
            self._next_act.append(Scalar[dtype](0))
            self._curr_act.append(Scalar[dtype](0))
            self._grad_act.append(Scalar[dtype](0))
        for _ in range(Self.BATCH):
            self._batch_rew.append(Scalar[dtype](0))
            self._batch_done.append(Scalar[dtype](0))
            self._next_log_pi.append(Scalar[dtype](0))
            self._nq1.append(Scalar[dtype](0))
            self._nq2.append(Scalar[dtype](0))
            self._targets.append(Scalar[dtype](0))
            self._q1_out.append(Scalar[dtype](0))
            self._q2_out.append(Scalar[dtype](0))
            self._q_grad.append(Scalar[dtype](0))
            self._curr_log_pi.append(Scalar[dtype](0))
            self._new_q1.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.ACTOR_OUT):
            self._next_out.append(Scalar[dtype](0))
            self._curr_out.append(Scalar[dtype](0))
            self._actor_grad_arr.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.CRITIC_IN):
            self._next_ci.append(Scalar[dtype](0))
            self._ci.append(Scalar[dtype](0))
            self._d_ci.append(Scalar[dtype](0))
            self._new_ci.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.CriticModel.CACHE_SIZE):
            self._q1_cache.append(Scalar[dtype](0))
            self._q2_cache.append(Scalar[dtype](0))
            self._new_c1_cache.append(Scalar[dtype](0))
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
# SACGPUState — GPU buffer container for SAC
# =============================================================================


struct SACGPUState[
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
    """GPU-resident state for SAC training.

    Key SAC differences from TD3:
      - actor is GPUNetworkState (online only, SAC has no target actor)
      - Separate training RNG for policy sampling during train_step
      - eps_cache buffer to save noise for backward through reparameterization
      - Actor output shape is [BATCH, ACTOR_OUT=2*ACTION_DIM] (mean || log_std)

    Created once at the start of GPU training via DeepSACAgent.make_gpu_state.
    CPU weights are uploaded separately via DeepSACAgent.upload_to_gpu.

    Parameters:
        ActorModel: Actor network model type (outputs mean + log_std).
        ActorOpt: Actor optimizer type.
        CriticModel: Critic network model type (shared between critic1 and critic2).
        CriticOpt: Critic optimizer type.
        buffer_capacity: GPU replay buffer capacity.
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        batch_size: Training batch size.
        max_n_envs: Max parallel environments (sizes exploration buffers).
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime BATCH = Self.batch_size
    comptime MAX_N = Self.max_n_envs
    comptime ACTOR_OUT = Self.ACTIONS * 2  # SAC: mean || log_std
    comptime CRITIC_IN = Self.OBS + Self.ACTIONS
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
    var buffer: GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim, Self.action_dim]

    # Exploration buffers (inference, sized by max_n_envs)
    var rng_states: DeviceBuffer[DType.uint32]  # [max_n_envs * action_dim]
    var inf_out: DeviceBuffer[dtype]  # [max_n_envs * ACTOR_OUT]
    var inf_ws: DeviceBuffer[dtype]  # [max_n_envs * ACTOR_WS]

    # Training RNG (separate seed from exploration, for next-state and curr-state sampling)
    var training_rng: DeviceBuffer[DType.uint32]  # [batch_size * action_dim]

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
    # eps_cache is reused: first for next-state (discarded), then for curr-state (used in backward)
    var eps_cache: DeviceBuffer[dtype]  # [batch_size * action_dim]
    var next_ci: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var nq1: DeviceBuffer[dtype]  # [batch_size]
    var nq2: DeviceBuffer[dtype]  # [batch_size]
    var targets: DeviceBuffer[dtype]  # [batch_size]

    # Training scratch — critic update
    var ci: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var q1_out: DeviceBuffer[dtype]  # [batch_size * 1]
    var q1_cache: DeviceBuffer[dtype]  # [batch_size * CRITIC_CS]
    var critic1_ws: DeviceBuffer[dtype]  # [batch_size * CRITIC_WS]
    var q1_grad: DeviceBuffer[dtype]  # [batch_size * 1]
    var d_ci1: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var q2_out: DeviceBuffer[dtype]  # [batch_size * 1]
    var q2_cache: DeviceBuffer[dtype]  # [batch_size * CRITIC_CS]
    var critic2_ws: DeviceBuffer[dtype]  # [batch_size * CRITIC_WS]
    var q2_grad: DeviceBuffer[dtype]  # [batch_size * 1]
    var d_ci2: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]

    # Training scratch — actor update
    var actor_out: DeviceBuffer[dtype]  # [batch_size * ACTOR_OUT]
    var actor_cache: DeviceBuffer[dtype]  # [batch_size * ACTOR_CS]
    var actor_ws: DeviceBuffer[dtype]  # [batch_size * ACTOR_WS]
    var curr_act: DeviceBuffer[dtype]  # [batch_size * action_dim]
    var curr_lp: DeviceBuffer[dtype]  # [batch_size] (downloaded for alpha update)
    var new_ci: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
    var new_q: DeviceBuffer[dtype]  # [batch_size * 1]
    var new_q_cache: DeviceBuffer[dtype]  # [batch_size * CRITIC_CS]
    var dq: DeviceBuffer[dtype]  # [batch_size * 1] constant -1/batch_size
    var d_new_ci: DeviceBuffer[dtype]  # [batch_size * CRITIC_IN]
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

        # Exploration buffers
        self.rng_states = ctx.enqueue_create_buffer[DType.uint32](
            Self.MAX_N * Self.ACTIONS
        )
        self.inf_out = ctx.enqueue_create_buffer[dtype](
            Self.MAX_N * Self.ACTOR_OUT
        )
        self.inf_ws = ctx.enqueue_create_buffer[dtype](
            Self.MAX_N * Self.ACTOR_WS
        )

        # Training RNG
        self.training_rng = ctx.enqueue_create_buffer[DType.uint32](
            Self.BATCH * Self.ACTIONS
        )

        # Replay sample output
        self.s_obs = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.OBS)
        self.s_act = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTIONS
        )
        self.s_rew = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.s_nobs = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.OBS)
        self.s_done = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](Self.BATCH)

        # TD target computation
        self.next_actor_out = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTOR_OUT
        )
        self.next_act = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTIONS
        )
        self.next_lp = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.eps_cache = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTIONS
        )
        self.next_ci = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_IN
        )
        self.nq1 = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.nq2 = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.targets = ctx.enqueue_create_buffer[dtype](Self.BATCH)

        # Critic update
        self.ci = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.CRITIC_IN)
        self.q1_out = ctx.enqueue_create_buffer[dtype](Self.BATCH * 1)
        self.q1_cache = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_CS
        )
        self.critic1_ws = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_WS
        )
        self.q1_grad = ctx.enqueue_create_buffer[dtype](Self.BATCH * 1)
        self.d_ci1 = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_IN
        )
        self.q2_out = ctx.enqueue_create_buffer[dtype](Self.BATCH * 1)
        self.q2_cache = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_CS
        )
        self.critic2_ws = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_WS
        )
        self.q2_grad = ctx.enqueue_create_buffer[dtype](Self.BATCH * 1)
        self.d_ci2 = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_IN
        )

        # Actor update
        self.actor_out = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTOR_OUT
        )
        self.actor_cache = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTOR_CS
        )
        self.actor_ws = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTOR_WS
        )
        self.curr_act = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTIONS
        )
        self.curr_lp = ctx.enqueue_create_buffer[dtype](Self.BATCH)
        self.new_ci = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_IN
        )
        self.new_q = ctx.enqueue_create_buffer[dtype](Self.BATCH * 1)
        self.new_q_cache = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_CS
        )
        self.dq = ctx.enqueue_create_buffer[dtype](Self.BATCH * 1)
        self.d_new_ci = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.CRITIC_IN
        )
        self.grad_act = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTIONS
        )
        self.actor_grad = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACTOR_OUT
        )
        self.d_obs = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.OBS)

        # Pre-fill dq with -1/batch_size (constant policy-gradient weight)
        ctx.synchronize()
        var dq_host = ctx.enqueue_create_host_buffer[dtype](Self.BATCH)
        for i in range(Self.BATCH):
            dq_host[i] = Scalar[dtype](-1.0 / Float64(Self.BATCH))
        ctx.enqueue_copy(self.dq, dq_host)

        # Initialize exploration RNG states
        ctx.synchronize()
        var rng_host = ctx.enqueue_create_host_buffer[DType.uint32](
            Self.MAX_N * Self.ACTIONS
        )
        var rng_s: UInt32 = 12345
        for i in range(Self.MAX_N * Self.ACTIONS):
            rng_s = rng_s ^ (rng_s << 13)
            rng_s = rng_s ^ (rng_s >> 17)
            rng_s = rng_s ^ (rng_s << 5)
            rng_host[i] = rng_s
        ctx.enqueue_copy(self.rng_states, rng_host)

        # Initialize training RNG states (separate seed)
        ctx.synchronize()
        var trng_host = ctx.enqueue_create_host_buffer[DType.uint32](
            Self.BATCH * Self.ACTIONS
        )
        var trng_s: UInt32 = 98765
        for i in range(Self.BATCH * Self.ACTIONS):
            trng_s = trng_s ^ (trng_s << 13)
            trng_s = trng_s ^ (trng_s >> 17)
            trng_s = trng_s ^ (trng_s << 5)
            trng_host[i] = trng_s
        ctx.enqueue_copy(self.training_rng, trng_host)

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
        """Return True if the GPU replay buffer has at least batch_size samples."""
        return self.buffer.is_ready[Self.batch_size]()
