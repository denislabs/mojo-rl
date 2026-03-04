"""Deep DDPG Agent using the new trait-based deep learning architecture.

This DDPG (Deep Deterministic Policy Gradient) implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- Sequential composition for actor and critic networks
- Tanh output activation for bounded actions
- ReplayBuffer from nn.replay for experience replay
- OffPolicyAgent trait for shared CPU training loop
- GPUOffPolicyAgent trait for shared GPU training loop

Features:
- Works with any BoxContinuousActionEnv (continuous obs, continuous actions)
- Deterministic policy with Gaussian exploration noise
- Target networks for both actor and critic with soft updates
- Single critic network (unlike TD3/SAC which use twin critics)
- lr is a compile-time parameter (Adam LR baked in at compile time)
- Checkpoint via NetworkState.write_sections / read_sections
- Unified CPU+GPU agent — same struct for both training modes

Usage:
    from deep_agents.ddpg import DeepDDPGAgent
    from envs import PendulumEnv

    var env = PendulumEnv()
    var agent = DeepDDPGAgent[3, 1, 256, 100000, 64]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=300)

    # GPU Training
    var ctx = DeviceContext()
    var metrics = agent.train_gpu[PendulumGPUEnv](ctx, num_steps=100000)

Reference: Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)
"""

from std.math import exp, sqrt
from std.random import random_float64, seed

from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Model, Linear, LinearReLU, LinearTanh, Sequential
from nn.optimizer import Optimizer, Adam
from nn.initializer import Kaiming, Xavier
from nn.training import (
    Network,
    NetworkState,
    GPUNetworkState,
    NetworkPair,
    GPUNetworkPair,
)
from nn.utils import fill_inline, obs_to_inline, concat_obs_action_batch
from deep_agents.offpolicy_helpers import (
    deterministic_select_action,
    greedy_continuous_action,
    store_continuous_transition,
    random_continuous_action,
)
from nn.replay import ReplayBuffer, GPUReplayBuffer
from nn.gpu.random import gaussian_noise
from nn.gpu import (
    concat_obs_action_kernel,
    ddpg_exploration_kernel,
    td_mse_grad_kernel,
    actor_grad_from_critic_kernel,
    td_target_continuous_kernel,
)
from std.gpu.host import DeviceContext, DeviceBuffer
from nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)
from core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    RenderableEnv,
    OffPolicyAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    run_offpolicy_continuous_train,
    run_offpolicy_continuous_eval,
    run_offpolicy_continuous_train_gpu,
    GPUContinuousEnv,
)


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


# =============================================================================
# Deep DDPG Agent
# =============================================================================


struct DeepDDPGAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
    max_n_envs: Int = 64,
](OffPolicyAgent & GPUOffPolicyAgent):
    """Deep Deterministic Policy Gradient agent — unified CPU + GPU.

    DDPG is an off-policy actor-critic algorithm that uses a deterministic
    policy with additive exploration noise for continuous action spaces.

    Key features:
    - Deterministic policy (actor outputs action directly, not distribution)
    - Single Q-network critic (unlike TD3/SAC which use twin critics)
    - Target networks for both actor and critic with soft updates
    - Gaussian exploration noise with decay
    - GPU training via GPUOffPolicyAgent trait + DDPGGPUState

    Parameters:
        obs_dim: Dimension of observation space.
        action_dim: Dimension of action space.
        hidden_dim: Hidden layer size (default: 256).
        buffer_capacity: Replay buffer capacity (default: 100000).
        batch_size: Training batch size (default: 64).
        actor_lr: Actor Adam learning rate — compile-time (default: 0.001).
        critic_lr: Critic Adam learning rate — compile-time (default: 0.001).
        max_n_envs: Max parallel environments for GPU training (default: 64).
    """

    # Convenience compile-time aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime HIDDEN = Self.hidden_dim
    comptime BATCH = Self.batch_size

    # Critic input dimension: obs + action concatenated
    comptime CRITIC_IN = Self.OBS + Self.ACTIONS

    # Actor: obs → hidden (ReLU) → hidden (ReLU) → action (Tanh)
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        LinearTanh[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime ActorNet = Network[Self.ActorModel, Adam[Self.actor_lr]]

    # Critic: (obs ‖ action) → hidden (ReLU) → hidden (ReLU) → Q-value
    comptime CriticModel = Sequential[
        LinearReLU[Self.CRITIC_IN, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime CriticNet = Network[Self.CriticModel, Adam[Self.critic_lr]]

    # GPUOffPolicyAgent required compile-time constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACTIONS
    comptime BUFFER_CAPACITY: Int = Self.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.max_n_envs
    comptime GPUStateType = DDPGGPUState[
        Self.ActorModel,
        Adam[Self.actor_lr],
        Self.CriticModel,
        Adam[Self.critic_lr],
        Self.buffer_capacity,
        Self.obs_dim,
        Self.action_dim,
        Self.batch_size,
        Self.max_n_envs,
    ]

    # Network states (heap-allocated params + grads + optimizer state)
    var actor: NetworkPair[Self.ActorModel, Adam[Self.actor_lr]]
    var critic: NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]

    # Replay buffer (action_dim-dimensional continuous actions)
    var buffer: ReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64
    var noise_std: Float64
    var noise_std_min: Float64
    var noise_decay: Float64

    # Training state
    var total_steps: Int
    var train_step_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    # Pre-allocated train_step scratch (heap, avoids per-call stack allocation)
    var _batch_obs: List[Scalar[dtype]]
    var _batch_act: List[Scalar[dtype]]
    var _batch_rew: List[Scalar[dtype]]
    var _batch_next: List[Scalar[dtype]]
    var _batch_done: List[Scalar[dtype]]
    var _next_act: List[Scalar[dtype]]
    var _next_ci: List[Scalar[dtype]]
    var _next_q: List[Scalar[dtype]]
    var _targets: List[Scalar[dtype]]
    var _ci: List[Scalar[dtype]]
    var _q_out: List[Scalar[dtype]]
    var _q_cache: List[Scalar[dtype]]
    var _q_grad: List[Scalar[dtype]]
    var _d_ci: List[Scalar[dtype]]
    var _actor_act: List[Scalar[dtype]]
    var _actor_cache: List[Scalar[dtype]]
    var _new_ci: List[Scalar[dtype]]
    var _new_q: List[Scalar[dtype]]
    var _dq: List[Scalar[dtype]]
    var _d_new_ci: List[Scalar[dtype]]
    var _d_act: List[Scalar[dtype]]
    var _d_obs: List[Scalar[dtype]]

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        noise_std: Float64 = 0.1,
        noise_std_min: Float64 = 0.01,
        noise_decay: Float64 = 0.995,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize Deep DDPG agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update coefficient (default: 0.005).
            action_scale: Action scaling factor (default: 1.0).
            noise_std: Initial exploration noise std (default: 0.1).
            noise_std_min: Minimum exploration noise std (default: 0.01).
            noise_decay: Noise decay per episode (default: 0.995).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path to save checkpoints.
        """
        self.actor = NetworkPair[Self.ActorModel, Adam[Self.actor_lr]]()
        self.actor.initialize[Xavier]()
        self.critic = NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]()
        self.critic.initialize[Kaiming]()

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_decay = noise_decay
        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

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

    # =========================================================================
    # OffPolicyAgent trait — required methods (CPU training)
    # =========================================================================

    fn select_action_list(mut self, obs: List[Float64]) -> List[Float64]:
        """Select action with Gaussian exploration noise (training)."""
        return deterministic_select_action[
            Self.ActorModel, Adam[Self.actor_lr]
        ](self.actor.online, obs, self.action_scale, self.noise_std)

    fn store_list_transition(
        mut self,
        obs: List[Float64],
        action: List[Float64],
        reward: Float64,
        next_obs: List[Float64],
        done: Bool,
    ) -> None:
        """Store transition in the replay buffer."""
        store_continuous_transition[
            Self.OBS, Self.ACTIONS, Self.buffer_capacity
        ](
            self.buffer,
            obs,
            action,
            reward,
            next_obs,
            done,
            self.action_scale,
            self.total_steps,
        )

    fn is_ready(self) -> Bool:
        """Return True if buffer has enough samples to begin training."""
        return self.buffer.is_ready[Self.BATCH]()

    fn do_train_step(mut self) -> Float64:
        """Perform one DDPG gradient update step.

        Returns:
            Critic loss value.
        """
        return self.train_step()

    fn decay_explore(mut self) -> None:
        """Decay exploration noise (call once per episode)."""
        self.noise_std *= self.noise_decay
        if self.noise_std < self.noise_std_min:
            self.noise_std = self.noise_std_min

    fn get_explore_rate(self) -> Float64:
        """Return current exploration noise std."""
        return self.noise_std

    fn random_action_list(self) -> List[Float64]:
        """Return a uniformly random action in [-action_scale, action_scale]."""
        return random_continuous_action(Self.action_dim, self.action_scale)

    fn select_greedy_action_list(self, obs: List[Float64]) -> List[Float64]:
        """Select action using deterministic policy (no exploration noise)."""
        return greedy_continuous_action[Self.ActorModel, Adam[Self.actor_lr]](
            self.actor.online, obs, self.action_scale
        )

    # =========================================================================
    # Core DDPG CPU Training Step
    # =========================================================================

    fn train_step(mut self) -> Float64:
        """Perform one DDPG training step (critic update → actor update → soft target update).

        Returns:
            Critic loss value, or 0.0 if buffer not ready.
        """
        if not self.buffer.is_ready[Self.BATCH]():
            return 0.0

        # Phase 1: Sample batch from replay buffer
        # These 5 must remain local InlineArrays — ReplayBuffer.sample takes mut InlineArray
        var batch_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_act = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var batch_rew = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_done = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        self.buffer.sample[Self.BATCH](
            batch_obs, batch_act, batch_rew, batch_next, batch_done
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_next.unsafe_ptr())

        # Phase 2: Compute TD targets
        # y = r + γ * Q_target(s', µ_target(s')) * (1 − done)
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self._next_act.unsafe_ptr())
        var p_actor_target = self.actor.target.params_view()
        Self.ActorNet.forward[Self.BATCH](
            next_obs_t, next_act_t, p_actor_target
        )

        # Build next critic input: concat(batch_next, _next_act) via manual loop
        # (batch_next is a local InlineArray; _next_act is a pre-allocated List)
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._next_ci.unsafe_ptr())
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                self._next_ci[b * Self.CRITIC_IN + i] = batch_next[
                    b * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                self._next_ci[
                    b * Self.CRITIC_IN + Self.OBS + i
                ] = self._next_act[b * Self.ACTIONS + i]

        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._next_q.unsafe_ptr())
        var p_critic_target = self.critic.target.params_view()
        Self.CriticNet.forward[Self.BATCH](next_ci_t, next_q_t, p_critic_target)

        for b in range(Self.BATCH):
            var q = Float64(self._next_q[b])
            if q != q:
                q = 0.0
            var done_mask = 1.0 - Float64(batch_done[b])
            var tgt = Float64(batch_rew[b]) + self.gamma * q * done_mask
            if tgt != tgt:
                tgt = 0.0
            elif tgt > 1000.0:
                tgt = 1000.0
            elif tgt < -1000.0:
                tgt = -1000.0
            self._targets[b] = Scalar[dtype](tgt)

        # Phase 3: Update Critic
        # Build critic input: concat(batch_obs, batch_act) via manual loop
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._ci.unsafe_ptr())
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                self._ci[b * Self.CRITIC_IN + i] = batch_obs[b * Self.OBS + i]
            for i in range(Self.ACTIONS):
                self._ci[b * Self.CRITIC_IN + Self.OBS + i] = batch_act[
                    b * Self.ACTIONS + i
                ]

        var q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q_out.unsafe_ptr())
        var critic_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._q_cache.unsafe_ptr())

        var p_critic = self.critic.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q_t, p_critic, critic_cache_t
        )

        var q_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q_grad.unsafe_ptr())
        var critic_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = self._q_out[b] - self._targets[b]
            critic_loss += Float64(td_err * td_err)
            self._q_grad[b] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
            )
        critic_loss /= Float64(Self.BATCH)

        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._d_ci.unsafe_ptr())

        var g_critic = self.critic.grads_view()
        self.critic.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q_grad_t, d_ci_t, p_critic, critic_cache_t, g_critic
        )
        self.critic.optimizer_step()

        # Phase 4: Update Actor
        var actor_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self._actor_act.unsafe_ptr())
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._actor_cache.unsafe_ptr())

        var p_actor = self.actor.params_view()
        Self.ActorNet.forward_with_cache[Self.BATCH](
            obs_t, actor_act_t, p_actor, actor_cache_t
        )

        # Build actor critic input: concat(batch_obs, _actor_act) via manual loop
        var new_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._new_ci.unsafe_ptr())
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                self._new_ci[b * Self.CRITIC_IN + i] = batch_obs[
                    b * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                self._new_ci[
                    b * Self.CRITIC_IN + Self.OBS + i
                ] = self._actor_act[b * Self.ACTIONS + i]

        var new_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._new_q.unsafe_ptr())
        var new_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._q_cache.unsafe_ptr())

        Self.CriticNet.forward_with_cache[Self.BATCH](
            new_ci_t, new_q_t, p_critic, new_cache_t
        )

        var dq_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._dq.unsafe_ptr())
        for b in range(Self.BATCH):
            self._dq[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))

        var d_new_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._d_new_ci.unsafe_ptr())

        self.critic.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            dq_t, d_new_ci_t, p_critic, new_cache_t, g_critic
        )

        var d_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self._d_act.unsafe_ptr())
        for b in range(Self.BATCH):
            for i in range(Self.ACTIONS):
                self._d_act[b * Self.ACTIONS + i] = self._d_new_ci[
                    b * Self.CRITIC_IN + Self.OBS + i
                ]

        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](self._d_obs.unsafe_ptr())

        var g_actor = self.actor.grads_view()
        self.actor.zero_grads()
        Self.ActorNet.backward[Self.BATCH](
            d_act_t, d_obs_t, p_actor, actor_cache_t, g_actor
        )
        self.actor.optimizer_step()

        # Phase 5: Soft update target networks
        self.actor.soft_update(self.tau)
        self.critic.soft_update(self.tau)

        self.train_step_count += 1
        return critic_loss

    # =========================================================================
    # GPUOffPolicyAgent trait — required methods
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for DDPG training.

        Does NOT upload CPU weights — call upload_to_gpu after this.
        """
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network states and replay buffer to GPU."""
        gpu_state.actor.upload_from(self.actor, ctx)
        gpu_state.critic.upload_from(self.critic, ctx)
        gpu_state.buffer.upload_from(self.buffer, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.actor.download_to(self.actor, ctx)
        gpu_state.critic.download_to(self.critic, ctx)

    fn select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward actor on GPU for N_ENVS environments + add exploration noise.
        """
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
        var rng_t = LayoutTensor[
            DType.uint32,
            Layout.row_major(N_ENVS, Self.ACTIONS),
            MutAnyOrigin,
        ](gpu_state.rng_states.unsafe_ptr())

        var p = gpu_state.actor.online.params_view()
        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.inf_ws
        )

        var noise_std_s = Scalar[dtype](self.noise_std)
        var scale_s = Scalar[dtype](self.action_scale)

        @always_inline
        fn exploration_wrapper(
            out_t: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS, Self.ACTIONS),
                MutAnyOrigin,
            ],
            raw_in: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS, Self.ACTIONS),
                MutAnyOrigin,
            ],
            rng_in: LayoutTensor[
                DType.uint32,
                Layout.row_major(N_ENVS, Self.ACTIONS),
                MutAnyOrigin,
            ],
            ns: Scalar[dtype],
            sc: Scalar[dtype],
        ):
            ddpg_exploration_kernel[dtype, N_ENVS, Self.ACTIONS](
                out_t, raw_in, rng_in, ns, sc
            )

        ctx.enqueue_function[exploration_wrapper, exploration_wrapper](
            act_t,
            raw_t,
            rng_t,
            noise_std_s,
            scale_s,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """One DDPG training step on GPU.

        Phases: sample → TD targets → critic update → actor update.
        Uses self for hyperparams (gamma, tau) and gpu_state for all buffers.
        """
        comptime BATCH = Self.BATCH
        comptime OBS = Self.OBS
        comptime ACTIONS = Self.ACTIONS
        comptime CRITIC_IN = Self.CRITIC_IN
        comptime CRITIC_CS = Self.CriticModel.CACHE_SIZE
        comptime ACTOR_CS = Self.ActorModel.CACHE_SIZE
        comptime ELEM_BLOCKS = (BATCH * CRITIC_IN + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime ACT_BLOCKS = (BATCH * ACTIONS + TPB - 1) // TPB

        # ---- Phase 1: Sample batch ----
        gpu_state.buffer.sample[BATCH](
            ctx,
            rng_seed=UInt32(self.total_steps),
            sampled_obs=gpu_state.s_obs,
            sampled_actions=gpu_state.s_act,
            sampled_rewards=gpu_state.s_rew,
            sampled_next_obs=gpu_state.s_nobs,
            sampled_dones=gpu_state.s_done,
            indices=gpu_state.s_idx,
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var nobs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())
        var rew_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu_state.s_rew.unsafe_ptr()
        )
        var done_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu_state.s_done.unsafe_ptr()
        )

        # ---- Phase 2: TD targets ----
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.next_act.unsafe_ptr())
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.next_ci.unsafe_ptr())
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.next_q.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())

        var p_actor_t = gpu_state.actor.target.params_view()
        var p_critic_t = gpu_state.critic.target.params_view()
        var p_actor = gpu_state.actor.online.params_view()
        var p_critic = gpu_state.critic.online.params_view()

        Self.ActorNet.forward_gpu[BATCH](
            ctx, nobs_t, next_act_t, p_actor_t, gpu_state.actor_ws
        )

        @always_inline
        fn concat_next(
            d: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
            o: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
        ):
            concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

        ctx.enqueue_function[concat_next, concat_next](
            next_ci_t,
            nobs_t,
            next_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        Self.CriticNet.forward_gpu[BATCH](
            ctx, next_ci_t, next_q_t, p_critic_t, gpu_state.critic_ws
        )

        var gamma_s = Scalar[dtype](self.gamma)
        var nq_flat_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.next_q.unsafe_ptr())

        @always_inline
        fn compute_targets(
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            r: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            nq: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            d: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
        ):
            td_target_continuous_kernel[dtype, BATCH](tgt, r, nq, d, g)

        ctx.enqueue_function[compute_targets, compute_targets](
            targets_t,
            rew_t,
            nq_flat_t,
            done_t,
            gamma_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ---- Phase 3: Critic update ----
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.ci.unsafe_ptr())
        var q_t = LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin](
            gpu_state.q_out.unsafe_ptr()
        )
        var q_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
        ](gpu_state.q_cache.unsafe_ptr())
        var q_grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q_grad.unsafe_ptr())
        var d_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci.unsafe_ptr())

        @always_inline
        fn concat_ci(
            d: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
            o: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
        ):
            concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

        ctx.enqueue_function[concat_ci, concat_ci](
            ci_t,
            obs_t,
            act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        Self.CriticNet.forward_gpu_with_cache[BATCH](
            ctx, ci_t, q_t, p_critic, q_cache_t, gpu_state.critic_ws
        )

        @always_inline
        fn mse_grad(
            qg: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            q: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            td_mse_grad_kernel[dtype, BATCH](qg, q, tgt)

        ctx.enqueue_function[mse_grad, mse_grad](
            q_grad_t,
            q_t,
            targets_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        var g_critic = gpu_state.critic.online.grads_view()
        gpu_state.critic.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BATCH](
            ctx,
            q_grad_t,
            d_ci_t,
            p_critic,
            q_cache_t,
            g_critic,
            gpu_state.critic_ws,
        )
        gpu_state.critic.online.optimizer_step(ctx)

        # ---- Phase 4: Actor update ----
        var actor_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.actor_act.unsafe_ptr())
        var new_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.new_ci.unsafe_ptr())
        var new_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.new_q.unsafe_ptr())
        var new_q_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
        ](gpu_state.new_q_cache.unsafe_ptr())
        var actor_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTOR_CS), MutAnyOrigin
        ](gpu_state.actor_cache.unsafe_ptr())
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.dq.unsafe_ptr())
        var d_new_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_new_ci.unsafe_ptr())
        var d_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.d_act.unsafe_ptr())
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](gpu_state.d_obs.unsafe_ptr())

        Self.ActorNet.forward_gpu_with_cache[BATCH](
            ctx, obs_t, actor_act_t, p_actor, actor_cache_t, gpu_state.actor_ws
        )

        @always_inline
        fn concat_new_ci(
            d: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
            o: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
        ):
            concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

        ctx.enqueue_function[concat_new_ci, concat_new_ci](
            new_ci_t,
            obs_t,
            actor_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB,),
        )

        Self.CriticNet.forward_gpu_with_cache[BATCH](
            ctx, new_ci_t, new_q_t, p_critic, new_q_cache_t, gpu_state.critic_ws
        )

        var g_critic2 = gpu_state.critic.online.grads_view()
        gpu_state.critic.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BATCH](
            ctx,
            dq_t,
            d_new_ci_t,
            p_critic,
            new_q_cache_t,
            g_critic2,
            gpu_state.critic_ws,
        )

        @always_inline
        fn extract_act_grad(
            da: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            dnc: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
        ):
            actor_grad_from_critic_kernel[dtype, BATCH, OBS, ACTIONS](da, dnc)

        ctx.enqueue_function[extract_act_grad, extract_act_grad](
            d_act_t,
            d_new_ci_t,
            grid_dim=(ACT_BLOCKS,),
            block_dim=(TPB,),
        )

        var g_actor = gpu_state.actor.online.grads_view()
        gpu_state.actor.online.zero_grads(ctx)
        Self.ActorNet.backward_gpu[BATCH](
            ctx,
            d_act_t,
            d_obs_t,
            p_actor,
            actor_cache_t,
            g_actor,
            gpu_state.actor_ws,
        )
        gpu_state.actor.online.optimizer_step(ctx)

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Soft-update actor and critic target networks on GPU."""
        gpu_state.actor.soft_update(self.tau, ctx)
        gpu_state.critic.soft_update(self.tau, ctx)

    # =========================================================================
    # High-level CPU training loop (delegates to shared off-policy runner)
    # =========================================================================

    fn train[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 200,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the DDPG agent on a continuous action environment (CPU).

        Args:
            env: Environment implementing BoxContinuousActionEnv.
            num_episodes: Number of training episodes.
            max_steps_per_episode: Maximum steps per episode (default: 200).
            warmup_steps: Random steps to pre-fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 1).
            verbose: Print progress (default: False).
            print_every: Print every N episodes if verbose (default: 10).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        return run_offpolicy_continuous_train(
            self,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            warmup_steps=warmup_steps,
            train_every=train_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name="Deep DDPG",
        )

    # =========================================================================
    # GPU training — delegates to shared run_offpolicy_continuous_train_gpu
    # =========================================================================

    fn train_gpu[
        E: GPUContinuousEnv,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        sync_every: Int = 50,
        verbose: Bool = False,
        print_every: Int = 50,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train on GPU using the shared off-policy GPU loop.

        GPU state (networks, replay buffer, scratch buffers) is created
        locally for the duration of training and freed when the method returns.
        After this call self.actor / critic (online and target) hold the
        trained GPU weights.

        Parameters:
            E: GPU environment type implementing GPUContinuousEnv.

        Args:
            ctx: GPU device context.
            num_steps: Total environment steps.
            warmup_steps: Random steps before training starts (default: 1000).
            train_every: Train step every N env steps (default: 1).
            sync_every: Download GPU→CPU every N steps (default: 50).
            verbose: Print progress (default: False).
            print_every: Print every N steps if verbose (default: 50).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics with step-level statistics.
        """
        return run_offpolicy_continuous_train_gpu[E, Self](
            self,
            ctx,
            num_steps,
            warmup_steps=warmup_steps,
            train_every=train_every,
            sync_every=sync_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name="Deep DDPG GPU",
        )

    # =========================================================================
    # Evaluation (deterministic policy, no noise)
    # =========================================================================

    fn evaluate[
        E: BoxContinuousActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 200,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent using the deterministic policy (no noise).

        Args:
            env: Environment to evaluate on (must also implement RenderableEnv).
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps: Maximum steps per episode (default: 200).
            verbose: Print per-episode results (default: False).
            render: Render the environment (default: False).
            frame_delay_ms: Delay between frames in milliseconds (default: 16).

        Returns:
            Average reward across evaluation episodes.
        """
        return run_offpolicy_continuous_eval(
            self,
            env,
            num_episodes=num_episodes,
            max_steps=max_steps,
            verbose=verbose,
            render=render,
            frame_delay_ms=frame_delay_ms,
            algorithm_name="Deep DDPG",
        ).mean_reward()

    # =========================================================================
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save agent state to a checkpoint file.

        Saves actor (online+target) and critic (online+target) params
        and optimizer states, plus runtime hyperparameters.
        The replay buffer is NOT saved.

        Args:
            filepath: Destination path (e.g. "ddpg_agent.ckpt").
        """
        comptime ACTOR_PARAM_SIZE = Self.ActorNet.PARAM_SIZE
        comptime CRITIC_PARAM_SIZE = Self.CriticNet.PARAM_SIZE
        comptime ACTOR_STATE_SIZE = ACTOR_PARAM_SIZE * Adam[
            Self.actor_lr
        ].STATE_PER_PARAM
        comptime CRITIC_STATE_SIZE = CRITIC_PARAM_SIZE * Adam[
            Self.critic_lr
        ].STATE_PER_PARAM

        var content = write_checkpoint_header(
            "ddpg_agent",
            ACTOR_PARAM_SIZE + CRITIC_PARAM_SIZE,
            ACTOR_STATE_SIZE + CRITIC_STATE_SIZE,
        )
        content += self.actor.write_sections("actor_")
        content += self.critic.write_sections("critic_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("actor_lr=" + String(Self.actor_lr))
        metadata.append("critic_lr=" + String(Self.critic_lr))
        metadata.append("action_scale=" + String(self.action_scale))
        metadata.append("noise_std=" + String(self.noise_std))
        metadata.append("noise_std_min=" + String(self.noise_std_min))
        metadata.append("noise_decay=" + String(self.noise_decay))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        var content = read_checkpoint_file(filepath)

        self.actor.read_sections(content, "actor_")
        self.critic.read_sections(content, "critic_")

        var metadata = read_metadata_section(content)

        var gamma_str = get_metadata_value(metadata, "gamma")
        if len(gamma_str) > 0:
            self.gamma = atof(gamma_str)

        var tau_str = get_metadata_value(metadata, "tau")
        if len(tau_str) > 0:
            self.tau = atof(tau_str)

        var action_scale_str = get_metadata_value(metadata, "action_scale")
        if len(action_scale_str) > 0:
            self.action_scale = atof(action_scale_str)

        var noise_std_str = get_metadata_value(metadata, "noise_std")
        if len(noise_std_str) > 0:
            self.noise_std = atof(noise_std_str)

        var noise_std_min_str = get_metadata_value(metadata, "noise_std_min")
        if len(noise_std_min_str) > 0:
            self.noise_std_min = atof(noise_std_min_str)

        var noise_decay_str = get_metadata_value(metadata, "noise_decay")
        if len(noise_decay_str) > 0:
            self.noise_decay = atof(noise_decay_str)

        var total_steps_str = get_metadata_value(metadata, "total_steps")
        if len(total_steps_str) > 0:
            self.total_steps = Int(atol(total_steps_str))

        var train_step_str = get_metadata_value(metadata, "train_step_count")
        if len(train_step_str) > 0:
            self.train_step_count = Int(atol(train_step_str))
