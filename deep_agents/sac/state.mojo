from nn.model import Model
from nn.optimizer import Optimizer
from nn.training import Network, NetworkState, NetworkPair
from nn.replay import ReplayBuffer
from nn.constants import dtype
from nn.initializer import Kaiming

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
]:
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
    var _batch_obs: List[Scalar[dtype]]   # [BATCH * OBS]
    var _batch_act: List[Scalar[dtype]]   # [BATCH * ACTIONS]
    var _batch_rew: List[Scalar[dtype]]   # [BATCH]
    var _batch_next: List[Scalar[dtype]]  # [BATCH * OBS]
    var _batch_done: List[Scalar[dtype]]  # [BATCH]

    # Scratch — TD target computation (next-state actor + twin critics)
    var _next_out: List[Scalar[dtype]]     # [BATCH * ACTOR_OUT] actor mean+log_std
    var _next_act: List[Scalar[dtype]]     # [BATCH * ACTIONS] sampled next actions
    var _next_log_pi: List[Scalar[dtype]]  # [BATCH] log_probs for next actions
    var _next_ci: List[Scalar[dtype]]      # [BATCH * CRITIC_IN]
    var _nq1: List[Scalar[dtype]]          # [BATCH] critic1_target output
    var _nq2: List[Scalar[dtype]]          # [BATCH] critic2_target output
    var _targets: List[Scalar[dtype]]      # [BATCH]

    # Scratch — critic updates (reused for both critics)
    var _ci: List[Scalar[dtype]]       # [BATCH * CRITIC_IN]
    var _q1_out: List[Scalar[dtype]]   # [BATCH]
    var _q2_out: List[Scalar[dtype]]   # [BATCH]
    var _q1_cache: List[Scalar[dtype]] # [BATCH * CriticModel.CACHE_SIZE]
    var _q2_cache: List[Scalar[dtype]] # [BATCH * CriticModel.CACHE_SIZE]
    var _q_grad: List[Scalar[dtype]]   # [BATCH] reused for q1_grad, q2_grad, dq
    var _d_ci: List[Scalar[dtype]]     # [BATCH * CRITIC_IN] reused for d_c1/d_c2/d_new_ci

    # Scratch — actor update
    var _curr_out: List[Scalar[dtype]]      # [BATCH * ACTOR_OUT] actor mean+log_std
    var _curr_act: List[Scalar[dtype]]      # [BATCH * ACTIONS] current sampled actions
    var _curr_log_pi: List[Scalar[dtype]]   # [BATCH] log_probs for current actions
    var _actor_cache: List[Scalar[dtype]]   # [BATCH * ActorModel.CACHE_SIZE]
    var _new_ci: List[Scalar[dtype]]        # [BATCH * CRITIC_IN]
    var _new_q1: List[Scalar[dtype]]        # [BATCH]
    var _new_c1_cache: List[Scalar[dtype]]  # [BATCH * CriticModel.CACHE_SIZE]
    var _actor_grad_arr: List[Scalar[dtype]] # [BATCH * ACTOR_OUT]
    var _grad_act: List[Scalar[dtype]]      # [BATCH * ACTIONS]
    var _d_obs: List[Scalar[dtype]]         # [BATCH * OBS]

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
        self._batch_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._batch_rew = List[Scalar[dtype]](capacity=Self.BATCH)
        self._batch_next = List[Scalar[dtype]](capacity=Self.BATCH * Self.OBS)
        self._batch_done = List[Scalar[dtype]](capacity=Self.BATCH)
        self._next_out = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTOR_OUT)
        self._next_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._next_log_pi = List[Scalar[dtype]](capacity=Self.BATCH)
        self._next_ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
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
        self._curr_out = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTOR_OUT)
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

    fn is_ready(self) -> Bool:
        """Return True if the replay buffer has enough samples to train."""
        return self.buffer.is_ready[Self.batch_size]()
