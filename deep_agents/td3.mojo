"""Deep TD3 Agent using the new trait-based deep learning architecture.

This TD3 (Twin Delayed Deep Deterministic Policy Gradient) implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- Sequential composition for actor and critic networks
- Tanh output activation for bounded actions
- ReplayBuffer from nn.replay for experience replay
- OffPolicyAgent trait for shared CPU training loop
- GPUOffPolicyAgent trait for shared GPU training loop

TD3 improves upon DDPG with three key innovations:
1. Twin Q-networks: Use two critics and take min(Q1, Q2) to reduce overestimation
2. Delayed policy updates: Update actor less frequently than critics
3. Target policy smoothing: Add clipped noise to target actions

Features:
- Works with any BoxContinuousActionEnv (continuous obs, continuous actions)
- Deterministic policy with Gaussian exploration noise
- Twin Q-networks to reduce overestimation bias
- Target networks for both actor and critics with soft updates
- Delayed actor updates (every policy_delay critic updates)
- Target policy smoothing with clipped noise
- lr is a compile-time parameter (Adam LR baked in at compile time)
- Checkpoint via NetworkState.write_sections / read_sections
- Unified CPU+GPU agent — same struct for both training modes

Usage:
    from deep_agents.td3 import DeepTD3Agent
    from envs import PendulumEnv

    var env = PendulumEnv()
    var agent = DeepTD3Agent[3, 1, 256, 100000, 64]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=300)

    # GPU Training
    var ctx = DeviceContext()
    var metrics = agent.train_gpu[PendulumGPUEnv](ctx, num_steps=100000)

Reference: Fujimoto et al., "Addressing Function Approximation Error in
Actor-Critic Methods" (2018)
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
from nn.utils import obs_to_inline, concat_obs_action_batch
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
    td_target_min_twin_kernel,
    add_gaussian_noise_kernel,
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
    OffPolicyAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    run_offpolicy_continuous_train,
    run_offpolicy_continuous_eval,
    run_offpolicy_continuous_train_gpu,
    GPUContinuousEnv,
)


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


# =============================================================================
# Deep TD3 Agent
# =============================================================================


struct DeepTD3Agent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
    max_n_envs: Int = 64,
](OffPolicyAgent & GPUOffPolicyAgent):
    """Deep Twin Delayed DDPG agent — unified CPU + GPU.

    TD3 improves upon DDPG by addressing function approximation error through
    three key techniques: twin Q-networks, delayed policy updates, and target
    policy smoothing.

    Key features:
    - Deterministic policy (actor outputs action directly, not distribution)
    - Twin Q-networks to reduce overestimation bias (min of Q1, Q2)
    - Target networks for both actor and critics with soft updates
    - Delayed actor updates (every policy_delay critic updates)
    - Target policy smoothing (clipped Gaussian noise on target actions)
    - GPU training via GPUOffPolicyAgent trait + TD3GPUState

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
    # Both critics share the same model architecture
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
    comptime GPUStateType = TD3GPUState[
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
    var critic1: NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]
    var critic2: NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]

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

    # TD3-specific hyperparameters
    var policy_delay: Int
    var target_noise_std: Float64
    var target_noise_clip: Float64

    # Training state
    var total_steps: Int
    var train_step_count: Int
    var update_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    # Pre-allocated train_step scratch (heap, avoids per-call stack allocation)
    var _batch_obs: List[Scalar[dtype]]
    var _batch_act: List[Scalar[dtype]]
    var _batch_rew: List[Scalar[dtype]]
    var _batch_next: List[Scalar[dtype]]
    var _batch_done: List[Scalar[dtype]]
    var _next_act: List[Scalar[dtype]]  # BATCH * ACTIONS
    var _next_ci: List[Scalar[dtype]]  # BATCH * CRITIC_IN
    var _nq1: List[Scalar[dtype]]  # BATCH * 1
    var _nq2: List[Scalar[dtype]]  # BATCH * 1
    var _targets: List[Scalar[dtype]]  # BATCH
    var _ci: List[Scalar[dtype]]  # BATCH * CRITIC_IN
    var _q1_out: List[Scalar[dtype]]  # BATCH * 1
    var _q2_out: List[Scalar[dtype]]  # BATCH * 1
    var _q1_cache: List[Scalar[dtype]]  # BATCH * CriticModel.CACHE_SIZE
    var _q2_cache: List[Scalar[dtype]]  # BATCH * CriticModel.CACHE_SIZE
    var _q_grad: List[
        Scalar[dtype]
    ]  # BATCH * 1 (reused for q1_grad and q2_grad)
    var _d_ci: List[
        Scalar[dtype]
    ]  # BATCH * CRITIC_IN (reused for d_c1 and d_c2)
    var _actor_act: List[Scalar[dtype]]  # BATCH * ACTIONS
    var _actor_cache: List[Scalar[dtype]]  # BATCH * ActorModel.CACHE_SIZE
    var _new_ci: List[Scalar[dtype]]  # BATCH * CRITIC_IN
    var _new_q: List[Scalar[dtype]]  # BATCH * 1
    var _dq: List[Scalar[dtype]]  # BATCH * 1
    var _d_new_ci: List[Scalar[dtype]]  # BATCH * CRITIC_IN
    var _d_act: List[Scalar[dtype]]  # BATCH * ACTIONS
    var _d_obs: List[Scalar[dtype]]  # BATCH * OBS

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        noise_std: Float64 = 0.1,
        noise_std_min: Float64 = 0.01,
        noise_decay: Float64 = 0.995,
        policy_delay: Int = 2,
        target_noise_std: Float64 = 0.2,
        target_noise_clip: Float64 = 0.5,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize Deep TD3 agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update coefficient (default: 0.005).
            action_scale: Action scaling factor (default: 1.0).
            noise_std: Initial exploration noise std (default: 0.1).
            noise_std_min: Minimum exploration noise std (default: 0.01).
            noise_decay: Noise decay per episode (default: 0.995).
            policy_delay: Update actor every N critic updates (default: 2).
            target_noise_std: Target policy smoothing noise std (default: 0.2).
            target_noise_clip: Clip range for target noise (default: 0.5).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path to save checkpoints.
        """
        self.actor = NetworkPair[Self.ActorModel, Adam[Self.actor_lr]]()
        self.actor.initialize[Xavier]()
        self.critic1 = NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]()
        self.critic1.initialize[Kaiming]()
        self.critic2 = NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]()
        self.critic2.initialize[Kaiming]()

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_decay = noise_decay
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.total_steps = 0
        self.train_step_count = 0
        self.update_count = 0
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
        """Perform one TD3 gradient update step.

        Returns:
            Average critic loss value.
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
    # Core TD3 CPU Training Step
    # =========================================================================

    fn train_step(mut self) -> Float64:
        """Perform one TD3 training step.

        TD3 update procedure:
        1. Always update both critics using TD error with min(Q1, Q2) target
        2. Every policy_delay updates:
           - Update actor using policy gradient from Q1
           - Soft update all target networks

        Returns:
            Average critic loss, or 0.0 if buffer not ready.
        """
        if not self.buffer.is_ready[Self.BATCH]():
            return 0.0

        self.update_count += 1

        # =================================================================
        # Phase 1: Sample batch
        # These 5 must remain local InlineArrays — ReplayBuffer.sample takes mut InlineArray
        # =================================================================
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

        # =================================================================
        # Phase 2: Compute TD targets with target policy smoothing
        # y = r + γ * min(Q1_t, Q2_t)(s', µ_t(s') + clip_noise) * (1 − done)
        # =================================================================

        # next_actions = actor_target(next_obs) — stored in pre-allocated _next_act
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self._next_act.unsafe_ptr())
        var p_actor_target = self.actor.target.params_view()
        Self.ActorNet.forward[Self.BATCH](
            next_obs_t, next_act_t, p_actor_target
        )

        # TD3 Innovation #3: target policy smoothing with clipped noise
        for b in range(Self.BATCH):
            for i in range(Self.ACTIONS):
                var idx = b * Self.ACTIONS + i
                var noise = gaussian_noise() * self.target_noise_std
                if noise > self.target_noise_clip:
                    noise = self.target_noise_clip
                elif noise < -self.target_noise_clip:
                    noise = -self.target_noise_clip
                var noisy_a = Float64(self._next_act[idx]) + noise
                if noisy_a > 1.0:
                    noisy_a = 1.0
                elif noisy_a < -1.0:
                    noisy_a = -1.0
                self._next_act[idx] = Scalar[dtype](noisy_a)

        # Build next critic input: concat(batch_next, _next_act) via manual loop
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

        # Forward both target critics
        var nq1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._nq1.unsafe_ptr())
        var nq2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._nq2.unsafe_ptr())

        var p_c1t = self.critic1.target.params_view()
        var p_c2t = self.critic2.target.params_view()
        Self.CriticNet.forward[Self.BATCH](next_ci_t, nq1_t, p_c1t)
        Self.CriticNet.forward[Self.BATCH](next_ci_t, nq2_t, p_c2t)

        # TD3 Innovation #1: take min(Q1, Q2) for TD targets
        for b in range(Self.BATCH):
            var q1 = Float64(self._nq1[b])
            var q2 = Float64(self._nq2[b])
            if q1 != q1:
                q1 = 0.0
            if q2 != q2:
                q2 = 0.0
            var min_q = q1 if q1 < q2 else q2
            var done_mask = 1.0 - Float64(batch_done[b])
            var tgt = Float64(batch_rew[b]) + self.gamma * min_q * done_mask
            if tgt != tgt:
                tgt = 0.0
            elif tgt > 1000.0:
                tgt = 1000.0
            elif tgt < -1000.0:
                tgt = -1000.0
            self._targets[b] = Scalar[dtype](tgt)

        # =================================================================
        # Phase 3: Update Both Critics with the same targets
        # =================================================================

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

        # --- Update Critic 1 ---
        var q1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q1_out.unsafe_ptr())
        var c1_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._q1_cache.unsafe_ptr())

        var p_c1 = self.critic1.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q1_t, p_c1, c1_cache_t
        )

        var q1_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q_grad.unsafe_ptr())
        var critic1_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = self._q1_out[b] - self._targets[b]
            critic1_loss += Float64(td_err * td_err)
            self._q_grad[b] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
            )
        critic1_loss /= Float64(Self.BATCH)

        var d_c1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._d_ci.unsafe_ptr())

        var g_c1 = self.critic1.grads_view()
        self.critic1.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q1_grad_t, d_c1_t, p_c1, c1_cache_t, g_c1
        )
        self.critic1.optimizer_step()

        # --- Update Critic 2 ---
        var q2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q2_out.unsafe_ptr())
        var c2_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._q2_cache.unsafe_ptr())

        var p_c2 = self.critic2.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q2_t, p_c2, c2_cache_t
        )

        var q2_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q_grad.unsafe_ptr())
        var critic2_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = self._q2_out[b] - self._targets[b]
            critic2_loss += Float64(td_err * td_err)
            self._q_grad[b] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
            )
        critic2_loss /= Float64(Self.BATCH)

        var d_c2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._d_ci.unsafe_ptr())

        var g_c2 = self.critic2.grads_view()
        self.critic2.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q2_grad_t, d_c2_t, p_c2, c2_cache_t, g_c2
        )
        self.critic2.optimizer_step()

        var avg_critic_loss = (critic1_loss + critic2_loss) / 2.0

        # =================================================================
        # Phase 4: Delayed Actor Update (TD3 Innovation #2)
        # Only update actor and all targets every policy_delay critic steps
        # =================================================================
        if self.update_count % self.policy_delay == 0:
            # actor_actions = actor_online(obs) with cache
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
                dtype,
                Layout.row_major(Self.BATCH, Self.CRITIC_IN),
                MutAnyOrigin,
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
            var new_c1_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
                MutAnyOrigin,
            ](self._q1_cache.unsafe_ptr())

            # TD3 uses Q1 for the actor gradient
            Self.CriticNet.forward_with_cache[Self.BATCH](
                new_ci_t, new_q_t, p_c1, new_c1_cache_t
            )

            # Gradient ascent: -1/BATCH
            var dq_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](self._dq.unsafe_ptr())
            for b in range(Self.BATCH):
                self._dq[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))

            var d_new_ci_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CRITIC_IN),
                MutAnyOrigin,
            ](self._d_new_ci.unsafe_ptr())

            # Backward through critic1 to get dQ/d(actions) — do NOT update critic
            self.critic1.zero_grads()
            Self.CriticNet.backward[Self.BATCH](
                dq_t, d_new_ci_t, p_c1, new_c1_cache_t, g_c1
            )
            # Intentionally NOT calling critic1_online.optimizer_step() here

            # Extract action gradients
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

            # Soft update all target networks
            self.actor.soft_update(self.tau)
            self.critic1.soft_update(self.tau)
            self.critic2.soft_update(self.tau)

        self.train_step_count += 1
        return avg_critic_loss

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
        """Train the TD3 agent on a continuous action environment.

        Delegates to run_offpolicy_continuous_train which handles warmup,
        episode loop, decay, and metric logging.

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
            algorithm_name="Deep TD3",
        )

    # =========================================================================
    # Evaluation (deterministic policy, no noise)
    # =========================================================================

    fn evaluate[
        E: BoxContinuousActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 200,
        verbose: Bool = False,
    ) -> Float64:
        """Evaluate the agent using the deterministic policy (no noise).

        Delegates to run_offpolicy_continuous_eval (uses select_greedy_action_list).

        Args:
            env: Environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps: Maximum steps per episode (default: 200).
            verbose: Print per-episode results (default: False).

        Returns:
            Average reward across evaluation episodes.
        """
        return run_offpolicy_continuous_eval(
            self,
            env,
            num_episodes=num_episodes,
            max_steps=max_steps,
            verbose=verbose,
            algorithm_name="Deep TD3",
        ).mean_reward()

    # =========================================================================
    # GPUOffPolicyAgent trait — required methods
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for TD3 training.

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
        gpu_state.critic1.upload_from(self.critic1, ctx)
        gpu_state.critic2.upload_from(self.critic2, ctx)
        gpu_state.buffer.upload_from(self.buffer, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.actor.download_to(self.actor, ctx)
        gpu_state.critic1.download_to(self.critic1, ctx)
        gpu_state.critic2.download_to(self.critic2, ctx)

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
        """One TD3 training step on GPU.

        Always updates both critics.
        Every policy_delay steps: also updates actor.
        Uses self for hyperparams (gamma, tau, policy_delay, etc.)
        and gpu_state for all device buffers.
        """
        comptime BATCH = Self.BATCH
        comptime OBS = Self.OBS
        comptime ACTIONS = Self.ACTIONS
        comptime CRITIC_IN = Self.CRITIC_IN
        comptime CRITIC_CS = Self.CriticModel.CACHE_SIZE
        comptime ACTOR_CS = Self.ActorModel.CACHE_SIZE
        comptime TPB256 = 256
        comptime ELEM_BLOCKS = (BATCH * CRITIC_IN + TPB256 - 1) // TPB256
        comptime BATCH_BLOCKS = (BATCH + TPB256 - 1) // TPB256
        comptime ACT_BLOCKS = (BATCH * ACTIONS + TPB256 - 1) // TPB256

        self.update_count += 1

        # ----- Phase 1: Sample batch -----
        gpu_state.buffer.sample[BATCH](
            ctx,
            rng_seed=UInt32(self.update_count),
            sampled_obs=gpu_state.s_obs,
            sampled_actions=gpu_state.s_act,
            sampled_rewards=gpu_state.s_rew,
            sampled_next_obs=gpu_state.s_nobs,
            sampled_dones=gpu_state.s_done,
            indices=gpu_state.s_idx,
        )

        # LayoutTensor views on sampled data
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

        # LayoutTensor views on target/twin scratch
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.next_act.unsafe_ptr())
        var noisy_next_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.noisy_next_act.unsafe_ptr())
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.next_ci.unsafe_ptr())
        var nq1_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu_state.nq1.unsafe_ptr()
        )
        var nq2_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu_state.nq2.unsafe_ptr()
        )
        var nq1_2d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.nq1.unsafe_ptr())
        var nq2_2d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.nq2.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())

        var p_actor_t = gpu_state.actor.target.params_view()
        var p_c1t = gpu_state.critic1.target.params_view()
        var p_c2t = gpu_state.critic2.target.params_view()
        var p_actor = gpu_state.actor.online.params_view()
        var p_c1 = gpu_state.critic1.online.params_view()
        var p_c2 = gpu_state.critic2.online.params_view()

        # ----- Phase 2: Actor target → next_act -----
        Self.ActorNet.forward_gpu[BATCH](
            ctx, nobs_t, next_act_t, p_actor_t, gpu_state.actor_ws
        )

        # ----- Phase 3: Target policy smoothing (TD3 Innovation #3) -----
        var tns_s = Scalar[dtype](self.target_noise_std)
        var tnc_s = Scalar[dtype](self.target_noise_clip)
        var act_min_s = Scalar[dtype](-self.action_scale)
        var act_max_s = Scalar[dtype](self.action_scale)
        var noise_rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.td3_noise_rng.unsafe_ptr())

        @always_inline
        fn smooth_noise(
            noisy: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            clean: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            rng: LayoutTensor[
                DType.uint32, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            ns: Scalar[dtype],
            nc: Scalar[dtype],
            amin: Scalar[dtype],
            amax: Scalar[dtype],
        ):
            add_gaussian_noise_kernel[dtype, BATCH, ACTIONS](
                noisy, clean, rng, ns, nc, amin, amax
            )

        ctx.enqueue_function[smooth_noise, smooth_noise](
            noisy_next_act_t,
            next_act_t,
            noise_rng_t,
            tns_s,
            tnc_s,
            act_min_s,
            act_max_s,
            grid_dim=(ACT_BLOCKS,),
            block_dim=(TPB256,),
        )

        # ----- Phase 4: Concat(next_obs, noisy_next_act) → next_ci -----
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
            noisy_next_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB256,),
        )

        # ----- Phase 5: Both critics target forward -----
        Self.CriticNet.forward_gpu[BATCH](
            ctx, next_ci_t, nq1_2d_t, p_c1t, gpu_state.critic1_ws
        )
        Self.CriticNet.forward_gpu[BATCH](
            ctx, next_ci_t, nq2_2d_t, p_c2t, gpu_state.critic2_ws
        )

        # ----- Phase 6: TD targets (TD3 Innovation #1: min(Q1, Q2)) -----
        var gamma_s = Scalar[dtype](self.gamma)
        var zero_s = Scalar[dtype](0.0)
        var log_probs_dummy = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())

        @always_inline
        fn td3_targets(
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            r: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            d: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
            a: Scalar[dtype],
        ):
            td_target_min_twin_kernel[dtype, BATCH, False](
                tgt, r, q1, q2, d, lp, g, a
            )

        ctx.enqueue_function[td3_targets, td3_targets](
            targets_t,
            rew_t,
            nq1_t,
            nq2_t,
            done_t,
            log_probs_dummy,
            gamma_s,
            zero_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB256,),
        )

        # ----- Phase 7: Concat(obs, actions) → ci -----
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.ci.unsafe_ptr())

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
            block_dim=(TPB256,),
        )

        # ----- Phase 8: Critic1 forward + MSE grad + backward + optim -----
        var q1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q1_out.unsafe_ptr())
        var q1_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
        ](gpu_state.q1_cache.unsafe_ptr())
        var q1_grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q1_grad.unsafe_ptr())
        var d_ci1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci1.unsafe_ptr())

        Self.CriticNet.forward_gpu_with_cache[BATCH](
            ctx, ci_t, q1_t, p_c1, q1_cache_t, gpu_state.critic1_ws
        )

        @always_inline
        fn mse_grad1(
            qg: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            q: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            td_mse_grad_kernel[dtype, BATCH](qg, q, tgt)

        ctx.enqueue_function[mse_grad1, mse_grad1](
            q1_grad_t,
            q1_t,
            targets_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB256,),
        )

        var g_c1 = gpu_state.critic1.online.grads_view()
        gpu_state.critic1.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BATCH](
            ctx,
            q1_grad_t,
            d_ci1_t,
            p_c1,
            q1_cache_t,
            g_c1,
            gpu_state.critic1_ws,
        )
        gpu_state.critic1.online.optimizer_step(ctx)

        # ----- Phase 9: Critic2 forward + MSE grad + backward + optim -----
        var q2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q2_out.unsafe_ptr())
        var q2_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
        ](gpu_state.q2_cache.unsafe_ptr())
        var q2_grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q2_grad.unsafe_ptr())
        var d_ci2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci2.unsafe_ptr())

        Self.CriticNet.forward_gpu_with_cache[BATCH](
            ctx, ci_t, q2_t, p_c2, q2_cache_t, gpu_state.critic2_ws
        )

        @always_inline
        fn mse_grad2(
            qg: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            q: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            td_mse_grad_kernel[dtype, BATCH](qg, q, tgt)

        ctx.enqueue_function[mse_grad2, mse_grad2](
            q2_grad_t,
            q2_t,
            targets_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB256,),
        )

        var g_c2 = gpu_state.critic2.online.grads_view()
        gpu_state.critic2.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BATCH](
            ctx,
            q2_grad_t,
            d_ci2_t,
            p_c2,
            q2_cache_t,
            g_c2,
            gpu_state.critic2_ws,
        )
        gpu_state.critic2.online.optimizer_step(ctx)

        # ----- Phase 10+: Delayed actor update (TD3 Innovation #2) -----
        if self.update_count % self.policy_delay == 0:
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

            # Actor forward with cache
            Self.ActorNet.forward_gpu_with_cache[BATCH](
                ctx,
                obs_t,
                actor_act_t,
                p_actor,
                actor_cache_t,
                gpu_state.actor_ws,
            )

            # Concat(obs, actor_actions) → new_ci
            @always_inline
            fn concat_new_ci(
                d: LayoutTensor[
                    dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
                ],
                o: LayoutTensor[
                    dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
                ],
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
                block_dim=(TPB256,),
            )

            # Critic1 forward for policy gradient (use critic1 per TD3 paper)
            Self.CriticNet.forward_gpu_with_cache[BATCH](
                ctx,
                new_ci_t,
                new_q_t,
                p_c1,
                new_q_cache_t,
                gpu_state.critic1_ws,
            )

            # Critic1 backward for action gradients (no optimizer step)
            var g_c1_pg = gpu_state.critic1.online.grads_view()
            gpu_state.critic1.online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BATCH](
                ctx,
                dq_t,
                d_new_ci_t,
                p_c1,
                new_q_cache_t,
                g_c1_pg,
                gpu_state.critic1_ws,
            )

            # Extract action gradients from critic input gradient
            @always_inline
            fn extract_act_grad(
                da: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
                dnc: LayoutTensor[
                    dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
                ],
            ):
                actor_grad_from_critic_kernel[dtype, BATCH, OBS, ACTIONS](
                    da, dnc
                )

            ctx.enqueue_function[extract_act_grad, extract_act_grad](
                d_act_t,
                d_new_ci_t,
                grid_dim=(ACT_BLOCKS,),
                block_dim=(TPB256,),
            )

            # Actor backward + optimizer step
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
        """Soft-update target networks on GPU (only when actor was updated).

        TD3 delays target updates to match actor updates: only every
        policy_delay critic steps. This method checks update_count to decide.
        """
        if self.update_count % self.policy_delay == 0:
            gpu_state.actor.soft_update(self.tau, ctx)
            gpu_state.critic1.soft_update(self.tau, ctx)
            gpu_state.critic2.soft_update(self.tau, ctx)

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
        After this call self.actor / critic1 / critic2 (online and target) hold
        the trained GPU weights (synced by download_from_gpu).

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
            algorithm_name="Deep TD3 GPU",
        )

    # =========================================================================
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save agent state to a checkpoint file.

        Saves all six NetworkState objects plus runtime hyperparameters.
        The replay buffer is NOT saved.

        Args:
            filepath: Destination path (e.g. "td3_agent.ckpt").
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
            "td3_agent",
            ACTOR_PARAM_SIZE + 2 * CRITIC_PARAM_SIZE,
            ACTOR_STATE_SIZE + 2 * CRITIC_STATE_SIZE,
        )
        content += self.actor.write_sections("actor_")
        content += self.critic1.write_sections("critic1_")
        content += self.critic2.write_sections("critic2_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("actor_lr=" + String(Self.actor_lr))
        metadata.append("critic_lr=" + String(Self.critic_lr))
        metadata.append("action_scale=" + String(self.action_scale))
        metadata.append("noise_std=" + String(self.noise_std))
        metadata.append("noise_std_min=" + String(self.noise_std_min))
        metadata.append("noise_decay=" + String(self.noise_decay))
        metadata.append("policy_delay=" + String(self.policy_delay))
        metadata.append("target_noise_std=" + String(self.target_noise_std))
        metadata.append("target_noise_clip=" + String(self.target_noise_clip))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("train_step_count=" + String(self.train_step_count))
        metadata.append("update_count=" + String(self.update_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        var content = read_checkpoint_file(filepath)

        self.actor.read_sections(content, "actor_")
        self.critic1.read_sections(content, "critic1_")
        self.critic2.read_sections(content, "critic2_")

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

        var policy_delay_str = get_metadata_value(metadata, "policy_delay")
        if len(policy_delay_str) > 0:
            self.policy_delay = Int(atol(policy_delay_str))

        var tns_str = get_metadata_value(metadata, "target_noise_std")
        if len(tns_str) > 0:
            self.target_noise_std = atof(tns_str)

        var tnc_str = get_metadata_value(metadata, "target_noise_clip")
        if len(tnc_str) > 0:
            self.target_noise_clip = atof(tnc_str)

        var total_steps_str = get_metadata_value(metadata, "total_steps")
        if len(total_steps_str) > 0:
            self.total_steps = Int(atol(total_steps_str))

        var train_step_str = get_metadata_value(metadata, "train_step_count")
        if len(train_step_str) > 0:
            self.train_step_count = Int(atol(train_step_str))

        var update_count_str = get_metadata_value(metadata, "update_count")
        if len(update_count_str) > 0:
            self.update_count = Int(atol(update_count_str))
