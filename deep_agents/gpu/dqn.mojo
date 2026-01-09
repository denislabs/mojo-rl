"""GPU-Accelerated Deep Q-Network (DQN) Agent for discrete action spaces.

This is the GPU-accelerated version of DQN, using Metal/CUDA for neural network
computations. Key optimizations:
- Persistent GPU buffers allocated once at training start
- Only batch data transferred CPU<->GPU per step (not weights)
- Adam optimizer runs entirely on GPU
- Target network soft updates on GPU

Architecture: obs_dim -> hidden (relu) -> hidden (relu) -> num_actions (linear)

Reference: Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
           Mnih et al. "Human-level control through deep RL" (2015, Nature)

Example usage:
    from deep_agents.gpu import GPUDeepDQNAgent
    from envs import LunarLanderEnv

    with DeviceContext() as ctx:
        var env = LunarLanderEnv()
        var agent = GPUDeepDQNAgent[obs_dim=8, num_actions=4, hidden_dim=128]()
        var metrics = agent.train(ctx, env, num_episodes=500)
"""

from random import random_float64
from math import sqrt

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from deep_rl.cpu import ReplayBuffer
from core import TrainingMetrics, BoxDiscreteActionEnv

# Import GPU kernels from deep_rl.gpu
from deep_rl.gpu import (
    linear_forward_kernel,
    linear_forward_relu_kernel,
    linear_backward_dW_kernel,
    linear_backward_db_kernel,
    linear_backward_dx_kernel,
    adam_update_kernel,
    soft_update_kernel,
    relu_grad_mul_kernel,
)


# =============================================================================
# GPU Q-Network for DQN (CPU weights for init/export, GPU for training)
# =============================================================================


struct GPUQNetwork[
    obs_dim: Int,
    num_actions: Int,
    hidden1_dim: Int = 256,
    hidden2_dim: Int = 256,
    batch_size: Int = 64,
    dtype: DType = DType.float32,
]:
    """GPU-accelerated Q-Network.

    Architecture: obs_dim -> hidden1 (relu) -> hidden2 (relu) -> num_actions (linear).

    Weights stored on CPU for initialization. During training, weights live on GPU.
    """

    # Sizes
    comptime W1_SIZE = Self.obs_dim * Self.hidden1_dim
    comptime B1_SIZE = Self.hidden1_dim
    comptime W2_SIZE = Self.hidden1_dim * Self.hidden2_dim
    comptime B2_SIZE = Self.hidden2_dim
    comptime W3_SIZE = Self.hidden2_dim * Self.num_actions
    comptime B3_SIZE = Self.num_actions
    comptime H1_SIZE = Self.batch_size * Self.hidden1_dim
    comptime H2_SIZE = Self.batch_size * Self.hidden2_dim
    comptime OUT_SIZE = Self.batch_size * Self.num_actions
    comptime OBS_SIZE = Self.batch_size * Self.obs_dim

    # Tile size for GPU matmul
    comptime TILE: Int = 16
    comptime TPB: Int = 64

    # CPU storage for weights (used for init and export)
    var W1: List[Scalar[Self.dtype]]
    var b1: List[Scalar[Self.dtype]]
    var W2: List[Scalar[Self.dtype]]
    var b2: List[Scalar[Self.dtype]]
    var W3: List[Scalar[Self.dtype]]
    var b3: List[Scalar[Self.dtype]]

    fn __init__(out self):
        """Initialize Q-Network with Xavier initialization."""
        # Initialize weights with Xavier
        var std1 = sqrt(2.0 / Float64(Self.obs_dim + Self.hidden1_dim))
        var std2 = sqrt(2.0 / Float64(Self.hidden1_dim + Self.hidden2_dim))
        var std3 = sqrt(2.0 / Float64(Self.hidden2_dim + Self.num_actions))

        self.W1 = List[Scalar[Self.dtype]](capacity=Self.W1_SIZE)
        for i in range(Self.W1_SIZE):
            self.W1.append(
                Scalar[Self.dtype]((random_float64() * 2 - 1) * std1)
            )

        self.W2 = List[Scalar[Self.dtype]](capacity=Self.W2_SIZE)
        for i in range(Self.W2_SIZE):
            self.W2.append(
                Scalar[Self.dtype]((random_float64() * 2 - 1) * std2)
            )

        self.W3 = List[Scalar[Self.dtype]](capacity=Self.W3_SIZE)
        for i in range(Self.W3_SIZE):
            self.W3.append(
                Scalar[Self.dtype]((random_float64() * 2 - 1) * std3)
            )

        # Initialize biases to zero
        self.b1 = List[Scalar[Self.dtype]](capacity=Self.B1_SIZE)
        for i in range(Self.B1_SIZE):
            self.b1.append(Scalar[Self.dtype](0))

        self.b2 = List[Scalar[Self.dtype]](capacity=Self.B2_SIZE)
        for i in range(Self.B2_SIZE):
            self.b2.append(Scalar[Self.dtype](0))

        self.b3 = List[Scalar[Self.dtype]](capacity=Self.B3_SIZE)
        for i in range(Self.B3_SIZE):
            self.b3.append(Scalar[Self.dtype](0))

    fn forward_cpu(
        self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
    ) -> InlineArray[Scalar[Self.dtype], Self.num_actions]:
        """CPU forward pass for single observation (action selection)."""
        # Layer 1: ReLU(obs @ W1 + b1)
        var h1 = InlineArray[Scalar[Self.dtype], Self.hidden1_dim](fill=0)
        for j in range(Self.hidden1_dim):
            var sum_val: Scalar[Self.dtype] = self.b1[j]
            for k in range(Self.obs_dim):
                sum_val += obs[k] * self.W1[k * Self.hidden1_dim + j]
            h1[j] = sum_val if sum_val > 0 else 0

        # Layer 2: ReLU(h1 @ W2 + b2)
        var h2 = InlineArray[Scalar[Self.dtype], Self.hidden2_dim](fill=0)
        for j in range(Self.hidden2_dim):
            var sum_val: Scalar[Self.dtype] = self.b2[j]
            for k in range(Self.hidden1_dim):
                sum_val += h1[k] * self.W2[k * Self.hidden2_dim + j]
            h2[j] = sum_val if sum_val > 0 else 0

        # Layer 3: h2 @ W3 + b3
        var q = InlineArray[Scalar[Self.dtype], Self.num_actions](fill=0)
        for j in range(Self.num_actions):
            var sum_val: Scalar[Self.dtype] = self.b3[j]
            for k in range(Self.hidden2_dim):
                sum_val += h2[k] * self.W3[k * Self.num_actions + j]
            q[j] = sum_val

        return q^

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        return (
            Self.W1_SIZE
            + Self.B1_SIZE
            + Self.W2_SIZE
            + Self.B2_SIZE
            + Self.W3_SIZE
            + Self.B3_SIZE
        )

    fn print_info(self, name: String = "GPUQNetwork"):
        """Print network architecture."""
        print(name + ":")
        print(
            "  Architecture: "
            + String(Self.obs_dim)
            + " -> "
            + String(Self.hidden1_dim)
            + " (relu)"
            + " -> "
            + String(Self.hidden2_dim)
            + " (relu)"
            + " -> "
            + String(Self.num_actions)
        )
        print("  Parameters: " + String(self.num_parameters()))


# =============================================================================
# GPU Deep DQN Agent
# =============================================================================


struct GPUDeepDQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    dtype: DType = DType.float32,
    double_dqn: Bool = True,
]:
    """GPU-Accelerated Deep DQN Agent.

    Parameters:
        obs_dim: Observation dimension (e.g., 8 for LunarLander).
        num_actions: Number of discrete actions (e.g., 4 for LunarLander).
        hidden_dim: Hidden layer size for Q-network.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        dtype: Data type for computations.
        double_dqn: If True, use Double DQN (recommended).
    """

    # Sizes for GPU buffers
    comptime W1_SIZE = Self.obs_dim * Self.hidden_dim
    comptime B1_SIZE = Self.hidden_dim
    comptime W2_SIZE = Self.hidden_dim * Self.hidden_dim
    comptime B2_SIZE = Self.hidden_dim
    comptime W3_SIZE = Self.hidden_dim * Self.num_actions
    comptime B3_SIZE = Self.num_actions
    comptime TILE: Int = 16
    comptime TPB: Int = 64

    var q_network: GPUQNetwork[
        Self.obs_dim,
        Self.num_actions,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.batch_size,
        Self.dtype,
    ]
    var target_network: GPUQNetwork[
        Self.obs_dim,
        Self.num_actions,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.batch_size,
        Self.dtype,
    ]

    var buffer: ReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, DType.float64
    ]

    var gamma: Scalar[Self.dtype]
    var tau: Scalar[Self.dtype]
    var lr: Scalar[Self.dtype]
    var epsilon: Scalar[Self.dtype]
    var epsilon_min: Scalar[Self.dtype]
    var epsilon_decay: Scalar[Self.dtype]

    var total_steps: Int
    var total_episodes: Int
    var adam_t: Int

    fn __init__(
        out self,
        gamma: Scalar[Self.dtype] = 0.99,
        tau: Scalar[Self.dtype] = 0.005,
        lr: Scalar[Self.dtype] = 0.0005,
        epsilon: Scalar[Self.dtype] = 1.0,
        epsilon_min: Scalar[Self.dtype] = 0.01,
        epsilon_decay: Scalar[Self.dtype] = 0.995,
    ):
        """Initialize GPU Deep DQN agent."""
        self.q_network = GPUQNetwork[
            Self.obs_dim,
            Self.num_actions,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.batch_size,
            Self.dtype,
        ]()

        self.target_network = GPUQNetwork[
            Self.obs_dim,
            Self.num_actions,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.batch_size,
            Self.dtype,
        ]()

        # Copy Q-network weights to target network (CPU)
        for i in range(Self.W1_SIZE):
            self.target_network.W1[i] = self.q_network.W1[i]
        for i in range(Self.B1_SIZE):
            self.target_network.b1[i] = self.q_network.b1[i]
        for i in range(Self.W2_SIZE):
            self.target_network.W2[i] = self.q_network.W2[i]
        for i in range(Self.B2_SIZE):
            self.target_network.b2[i] = self.q_network.b2[i]
        for i in range(Self.W3_SIZE):
            self.target_network.W3[i] = self.q_network.W3[i]
        for i in range(Self.B3_SIZE):
            self.target_network.b3[i] = self.q_network.b3[i]

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, DType.float64
        ]()

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.total_episodes = 0
        self.adam_t = 0

    fn select_action(
        self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        training: Bool = True,
    ) -> Int:
        """Select action using epsilon-greedy policy (CPU forward pass)."""
        if training and random_float64() < Float64(self.epsilon):
            return (
                Int(random_float64() * Float64(Self.num_actions))
                % Self.num_actions
            )

        var q_values = self.q_network.forward_cpu(obs)

        var best_action = 0
        var best_q = q_values[0]
        for a in range(1, Self.num_actions):
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[DType.float64], Self.obs_dim],
        action: Int,
        reward: Scalar[DType.float64],
        next_obs: InlineArray[Scalar[DType.float64], Self.obs_dim],
        done: Bool,
    ):
        """Store a transition in the CPU replay buffer."""
        var action_arr = InlineArray[Scalar[DType.float64], 1](fill=0)
        action_arr[0] = Scalar[DType.float64](action)
        self.buffer.add(obs, action_arr, reward, next_obs, done)
        self.total_steps += 1

    fn decay_epsilon(mut self):
        """Decay exploration rate."""
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn print_info(self):
        """Print agent information."""

        @parameter
        if Self.double_dqn:
            print("GPU Deep Double DQN Agent:")
        else:
            print("GPU Deep DQN Agent:")
        print("  Obs dim: " + String(Self.obs_dim))
        print("  Num actions: " + String(Self.num_actions))
        print("  Hidden dim: " + String(Self.hidden_dim))
        print("  Buffer capacity: " + String(Self.buffer_capacity))
        print("  Batch size: " + String(Self.batch_size))
        print("  Double DQN: " + String(Self.double_dqn))
        print("  Gamma: " + String(self.gamma)[:6])
        print("  Tau: " + String(self.tau)[:6])
        print("  LR: " + String(self.lr)[:8])
        print("  Epsilon: " + String(self.epsilon)[:5])
        self.q_network.print_info("  Q-Network")

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        ctx: DeviceContext,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = True,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train the GPU Deep DQN agent on a discrete action environment.

        Key optimization: All GPU buffers are allocated ONCE here and reused
        throughout training. Only batch data is transferred per step.
        """
        var metrics = TrainingMetrics(
            algorithm_name="GPU Deep DQN",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("GPU Deep DQN Training on " + environment_name)
            print("=" * 60)
            self.print_info()
            print("-" * 60)

        # =====================================================================
        # ALLOCATE ALL GPU BUFFERS ONCE (this is the key optimization!)
        # =====================================================================

        comptime BATCH = Self.batch_size
        comptime OBS = Self.obs_dim
        comptime H1 = Self.hidden_dim
        comptime H2 = Self.hidden_dim
        comptime OUT = Self.num_actions

        # Q-Network weights on GPU
        var qnet_W1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W1_SIZE)
        var qnet_b1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B1_SIZE)
        var qnet_W2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W2_SIZE)
        var qnet_b2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B2_SIZE)
        var qnet_W3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W3_SIZE)
        var qnet_b3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B3_SIZE)

        # Target network weights on GPU
        var target_W1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W1_SIZE)
        var target_b1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B1_SIZE)
        var target_W2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W2_SIZE)
        var target_b2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B2_SIZE)
        var target_W3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W3_SIZE)
        var target_b3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B3_SIZE)

        # Adam optimizer state on GPU (momentum and velocity for each weight)
        var m_W1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W1_SIZE)
        var v_W1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W1_SIZE)
        var m_b1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B1_SIZE)
        var v_b1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B1_SIZE)
        var m_W2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W2_SIZE)
        var v_W2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W2_SIZE)
        var m_b2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B2_SIZE)
        var v_b2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B2_SIZE)
        var m_W3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W3_SIZE)
        var v_W3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W3_SIZE)
        var m_b3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B3_SIZE)
        var v_b3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B3_SIZE)

        # Batch data buffers (reused every training step)
        var batch_obs_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OBS)
        var batch_next_obs_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OBS)

        # Activation caches (reused every training step)
        var h1_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H1)
        var h2_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H2)
        var q_values_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OUT)

        # Target network forward pass buffers
        var target_h1_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H1)
        var target_h2_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H2)
        var target_q_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OUT)

        # For Double DQN: online network next state Q-values
        var online_next_h1_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H1)
        var online_next_h2_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H2)
        var online_next_q_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OUT)

        # Gradient buffers (reused every training step)
        var dW1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W1_SIZE)
        var db1_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B1_SIZE)
        var dW2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W2_SIZE)
        var db2_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B2_SIZE)
        var dW3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.W3_SIZE)
        var db3_gpu = ctx.enqueue_create_buffer[Self.dtype](Self.B3_SIZE)
        var dh1_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H1)
        var dh2_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H2)
        var dq_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OUT)

        # =====================================================================
        # COPY INITIAL WEIGHTS TO GPU (once!)
        # =====================================================================

        with qnet_W1_gpu.map_to_host() as host:
            for i in range(Self.W1_SIZE):
                host[i] = self.q_network.W1[i]
        with qnet_b1_gpu.map_to_host() as host:
            for i in range(Self.B1_SIZE):
                host[i] = self.q_network.b1[i]
        with qnet_W2_gpu.map_to_host() as host:
            for i in range(Self.W2_SIZE):
                host[i] = self.q_network.W2[i]
        with qnet_b2_gpu.map_to_host() as host:
            for i in range(Self.B2_SIZE):
                host[i] = self.q_network.b2[i]
        with qnet_W3_gpu.map_to_host() as host:
            for i in range(Self.W3_SIZE):
                host[i] = self.q_network.W3[i]
        with qnet_b3_gpu.map_to_host() as host:
            for i in range(Self.B3_SIZE):
                host[i] = self.q_network.b3[i]

        # Copy to target network (same initial weights)
        with target_W1_gpu.map_to_host() as host:
            for i in range(Self.W1_SIZE):
                host[i] = self.q_network.W1[i]
        with target_b1_gpu.map_to_host() as host:
            for i in range(Self.B1_SIZE):
                host[i] = self.q_network.b1[i]
        with target_W2_gpu.map_to_host() as host:
            for i in range(Self.W2_SIZE):
                host[i] = self.q_network.W2[i]
        with target_b2_gpu.map_to_host() as host:
            for i in range(Self.B2_SIZE):
                host[i] = self.q_network.b2[i]
        with target_W3_gpu.map_to_host() as host:
            for i in range(Self.W3_SIZE):
                host[i] = self.q_network.W3[i]
        with target_b3_gpu.map_to_host() as host:
            for i in range(Self.B3_SIZE):
                host[i] = self.q_network.b3[i]

        # Initialize Adam state to zero
        m_W1_gpu.enqueue_fill(0)
        v_W1_gpu.enqueue_fill(0)
        m_b1_gpu.enqueue_fill(0)
        v_b1_gpu.enqueue_fill(0)
        m_W2_gpu.enqueue_fill(0)
        v_W2_gpu.enqueue_fill(0)
        m_b2_gpu.enqueue_fill(0)
        v_b2_gpu.enqueue_fill(0)
        m_W3_gpu.enqueue_fill(0)
        v_W3_gpu.enqueue_fill(0)
        m_b3_gpu.enqueue_fill(0)
        v_b3_gpu.enqueue_fill(0)

        ctx.synchronize()

        if verbose:
            print("GPU buffers allocated and initialized")

        # =====================================================================
        # WARMUP PHASE
        # =====================================================================

        if verbose:
            print(
                "Warmup: collecting "
                + String(warmup_steps)
                + " random steps..."
            )

        var warmup_done = 0
        while warmup_done < warmup_steps:
            var obs_list = env.reset_obs_list()
            var done = False

            while not done and warmup_done < warmup_steps:
                var action = (
                    Int(random_float64() * Float64(Self.num_actions))
                    % Self.num_actions
                )
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                var obs = _list_to_inline[Self.obs_dim, DType.float64](obs_list)
                var next_obs = _list_to_inline[Self.obs_dim, DType.float64](
                    step_result[0]
                )

                self.store_transition(
                    obs, action, Scalar[DType.float64](reward), next_obs, done
                )
                obs_list = env.get_obs_list()
                warmup_done += 1

        if verbose:
            print("Warmup complete. Buffer size: " + String(self.buffer.len()))
            print("-" * 60)

        # =====================================================================
        # TRAINING LOOP
        # =====================================================================

        # Grid dimensions for kernels (computed once)
        comptime blocks_y1 = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_x1 = (H1 + Self.TILE - 1) // Self.TILE
        comptime blocks_y2 = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_x2 = (H2 + Self.TILE - 1) // Self.TILE
        comptime blocks_y3 = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_x3 = (OUT + Self.TILE - 1) // Self.TILE

        # Backward pass grid dimensions
        comptime blocks_dW3_y = (H2 + Self.TILE - 1) // Self.TILE
        comptime blocks_dW3_x = (OUT + Self.TILE - 1) // Self.TILE
        comptime blocks_dh2_y = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_dh2_x = (H2 + Self.TILE - 1) // Self.TILE
        comptime blocks_dW2_y = (H1 + Self.TILE - 1) // Self.TILE
        comptime blocks_dW2_x = (H2 + Self.TILE - 1) // Self.TILE
        comptime blocks_dh1_y = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_dh1_x = (H1 + Self.TILE - 1) // Self.TILE
        comptime blocks_dW1_y = (OBS + Self.TILE - 1) // Self.TILE
        comptime blocks_dW1_x = (H1 + Self.TILE - 1) // Self.TILE

        # Adam/soft update grid dimensions
        comptime blocks_W1 = (Self.W1_SIZE + Self.TPB - 1) // Self.TPB
        comptime blocks_b1 = (Self.B1_SIZE + Self.TPB - 1) // Self.TPB
        comptime blocks_W2 = (Self.W2_SIZE + Self.TPB - 1) // Self.TPB
        comptime blocks_b2 = (Self.B2_SIZE + Self.TPB - 1) // Self.TPB
        comptime blocks_W3 = (Self.W3_SIZE + Self.TPB - 1) // Self.TPB
        comptime blocks_b3 = (Self.B3_SIZE + Self.TPB - 1) // Self.TPB
        comptime blocks_relu1 = (BATCH * H1 + Self.TPB - 1) // Self.TPB
        comptime blocks_relu2 = (BATCH * H2 + Self.TPB - 1) // Self.TPB

        # CPU buffers for batch sampling
        var batch_obs_cpu = InlineArray[
            Scalar[DType.float64], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_actions_cpu = InlineArray[Scalar[DType.float64], Self.batch_size](
            fill=0
        )
        var batch_rewards_cpu = InlineArray[Scalar[DType.float64], Self.batch_size](
            fill=0
        )
        var batch_next_obs_cpu = InlineArray[
            Scalar[DType.float64], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_dones_cpu = InlineArray[Scalar[DType.float64], Self.batch_size](
            fill=0
        )
        var batch_actions_arr = InlineArray[
            Scalar[DType.float64], Self.batch_size
        ](fill=0)

        # CPU buffers for reading back Q-values
        var current_q_cpu = List[Scalar[Self.dtype]](capacity=BATCH * OUT)
        var target_q_cpu = List[Scalar[Self.dtype]](capacity=BATCH * OUT)
        var online_next_q_cpu = List[Scalar[Self.dtype]](capacity=BATCH * OUT)

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var steps = 0
            var done = False

            while not done and steps < max_steps_per_episode:
                var obs_dtype = _list_to_inline_dtype[Self.obs_dim, Self.dtype](
                    obs_list
                )
                var action = self.select_action(obs_dtype, training=True)
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                var obs = _list_to_inline[Self.obs_dim, DType.float64](obs_list)
                var next_obs = _list_to_inline[Self.obs_dim, DType.float64](
                    step_result[0]
                )
                self.store_transition(
                    obs, action, Scalar[DType.float64](reward), next_obs, done
                )

                # Training step
                if steps % train_every == 0 and self.buffer.is_ready[Self.batch_size]():
                    # =========================================================
                    # SAMPLE BATCH (CPU)
                    # =========================================================
                    self.buffer.sample[Self.batch_size](
                        batch_obs_cpu,
                        batch_actions_arr,
                        batch_rewards_cpu,
                        batch_next_obs_cpu,
                        batch_dones_cpu,
                    )
                    for i in range(Self.batch_size):
                        batch_actions_cpu[i] = batch_actions_arr[i]

                    # =========================================================
                    # COPY BATCH DATA TO GPU (only batch data, not weights!)
                    # =========================================================
                    with batch_obs_gpu.map_to_host() as host:
                        for i in range(BATCH * OBS):
                            host[i] = Scalar[Self.dtype](batch_obs_cpu[i])
                    with batch_next_obs_gpu.map_to_host() as host:
                        for i in range(BATCH * OBS):
                            host[i] = Scalar[Self.dtype](batch_next_obs_cpu[i])

                    # Initialize activation buffers to 0
                    h1_gpu.enqueue_fill(0)
                    h2_gpu.enqueue_fill(0)
                    q_values_gpu.enqueue_fill(0)
                    target_h1_gpu.enqueue_fill(0)
                    target_h2_gpu.enqueue_fill(0)
                    target_q_gpu.enqueue_fill(0)

                    @parameter
                    if Self.double_dqn:
                        online_next_h1_gpu.enqueue_fill(0)
                        online_next_h2_gpu.enqueue_fill(0)
                        online_next_q_gpu.enqueue_fill(0)

                    # =========================================================
                    # GPU FORWARD PASS: Q-NETWORK ON CURRENT OBS
                    # =========================================================
                    var obs_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, OBS), ImmutAnyOrigin
                    ](batch_obs_gpu)
                    var h1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H1), MutAnyOrigin
                    ](h1_gpu)
                    var h2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H2), MutAnyOrigin
                    ](h2_gpu)
                    var q_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
                    ](q_values_gpu)
                    var qnet_W1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(OBS, H1), ImmutAnyOrigin
                    ](qnet_W1_gpu)
                    var qnet_b1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H1), ImmutAnyOrigin
                    ](qnet_b1_gpu)
                    var qnet_W2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H1, H2), ImmutAnyOrigin
                    ](qnet_W2_gpu)
                    var qnet_b2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H2), ImmutAnyOrigin
                    ](qnet_b2_gpu)
                    var qnet_W3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H2, OUT), ImmutAnyOrigin
                    ](qnet_W3_gpu)
                    var qnet_b3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(OUT), ImmutAnyOrigin
                    ](qnet_b3_gpu)

                    # Layer 1
                    comptime fwd_kernel1 = linear_forward_relu_kernel[
                        Self.dtype, BATCH, OBS, H1, Self.TILE
                    ]
                    ctx.enqueue_function_checked[fwd_kernel1, fwd_kernel1](
                        h1_t, obs_t, qnet_W1_t, qnet_b1_t,
                        grid_dim=(blocks_x1, blocks_y1),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # Layer 2
                    var h1_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H1), ImmutAnyOrigin
                    ](h1_gpu)
                    comptime fwd_kernel2 = linear_forward_relu_kernel[
                        Self.dtype, BATCH, H1, H2, Self.TILE
                    ]
                    ctx.enqueue_function_checked[fwd_kernel2, fwd_kernel2](
                        h2_t, h1_immut, qnet_W2_t, qnet_b2_t,
                        grid_dim=(blocks_x2, blocks_y2),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # Layer 3
                    var h2_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H2), ImmutAnyOrigin
                    ](h2_gpu)
                    comptime fwd_kernel3 = linear_forward_kernel[
                        Self.dtype, BATCH, H2, OUT, Self.TILE
                    ]
                    ctx.enqueue_function_checked[fwd_kernel3, fwd_kernel3](
                        q_t, h2_immut, qnet_W3_t, qnet_b3_t,
                        grid_dim=(blocks_x3, blocks_y3),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # =========================================================
                    # GPU FORWARD PASS: TARGET NETWORK ON NEXT OBS
                    # =========================================================
                    var next_obs_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, OBS), ImmutAnyOrigin
                    ](batch_next_obs_gpu)
                    var target_h1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H1), MutAnyOrigin
                    ](target_h1_gpu)
                    var target_h2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H2), MutAnyOrigin
                    ](target_h2_gpu)
                    var target_q_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
                    ](target_q_gpu)
                    var target_W1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(OBS, H1), ImmutAnyOrigin
                    ](target_W1_gpu)
                    var target_b1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H1), ImmutAnyOrigin
                    ](target_b1_gpu)
                    var target_W2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H1, H2), ImmutAnyOrigin
                    ](target_W2_gpu)
                    var target_b2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H2), ImmutAnyOrigin
                    ](target_b2_gpu)
                    var target_W3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H2, OUT), ImmutAnyOrigin
                    ](target_W3_gpu)
                    var target_b3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(OUT), ImmutAnyOrigin
                    ](target_b3_gpu)

                    ctx.enqueue_function_checked[fwd_kernel1, fwd_kernel1](
                        target_h1_t, next_obs_t, target_W1_t, target_b1_t,
                        grid_dim=(blocks_x1, blocks_y1),
                        block_dim=(Self.TILE, Self.TILE),
                    )
                    var target_h1_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H1), ImmutAnyOrigin
                    ](target_h1_gpu)
                    ctx.enqueue_function_checked[fwd_kernel2, fwd_kernel2](
                        target_h2_t, target_h1_immut, target_W2_t, target_b2_t,
                        grid_dim=(blocks_x2, blocks_y2),
                        block_dim=(Self.TILE, Self.TILE),
                    )
                    var target_h2_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H2), ImmutAnyOrigin
                    ](target_h2_gpu)
                    ctx.enqueue_function_checked[fwd_kernel3, fwd_kernel3](
                        target_q_t, target_h2_immut, target_W3_t, target_b3_t,
                        grid_dim=(blocks_x3, blocks_y3),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # =========================================================
                    # GPU FORWARD PASS: ONLINE NETWORK ON NEXT OBS (Double DQN)
                    # =========================================================
                    @parameter
                    if Self.double_dqn:
                        var online_next_h1_t = LayoutTensor[
                            Self.dtype, Layout.row_major(BATCH, H1), MutAnyOrigin
                        ](online_next_h1_gpu)
                        var online_next_h2_t = LayoutTensor[
                            Self.dtype, Layout.row_major(BATCH, H2), MutAnyOrigin
                        ](online_next_h2_gpu)
                        var online_next_q_t = LayoutTensor[
                            Self.dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
                        ](online_next_q_gpu)

                        ctx.enqueue_function_checked[fwd_kernel1, fwd_kernel1](
                            online_next_h1_t, next_obs_t, qnet_W1_t, qnet_b1_t,
                            grid_dim=(blocks_x1, blocks_y1),
                            block_dim=(Self.TILE, Self.TILE),
                        )
                        var online_next_h1_immut = LayoutTensor[
                            Self.dtype, Layout.row_major(BATCH, H1), ImmutAnyOrigin
                        ](online_next_h1_gpu)
                        ctx.enqueue_function_checked[fwd_kernel2, fwd_kernel2](
                            online_next_h2_t, online_next_h1_immut, qnet_W2_t, qnet_b2_t,
                            grid_dim=(blocks_x2, blocks_y2),
                            block_dim=(Self.TILE, Self.TILE),
                        )
                        var online_next_h2_immut = LayoutTensor[
                            Self.dtype, Layout.row_major(BATCH, H2), ImmutAnyOrigin
                        ](online_next_h2_gpu)
                        ctx.enqueue_function_checked[fwd_kernel3, fwd_kernel3](
                            online_next_q_t, online_next_h2_immut, qnet_W3_t, qnet_b3_t,
                            grid_dim=(blocks_x3, blocks_y3),
                            block_dim=(Self.TILE, Self.TILE),
                        )

                    ctx.synchronize()

                    # =========================================================
                    # READ Q-VALUES BACK TO CPU FOR LOSS COMPUTATION
                    # =========================================================
                    current_q_cpu.clear()
                    with q_values_gpu.map_to_host() as host:
                        for i in range(BATCH * OUT):
                            current_q_cpu.append(host[i])

                    target_q_cpu.clear()
                    with target_q_gpu.map_to_host() as host:
                        for i in range(BATCH * OUT):
                            target_q_cpu.append(host[i])

                    @parameter
                    if Self.double_dqn:
                        online_next_q_cpu.clear()
                        with online_next_q_gpu.map_to_host() as host:
                            for i in range(BATCH * OUT):
                                online_next_q_cpu.append(host[i])

                    # =========================================================
                    # COMPUTE TARGETS AND GRADIENTS (CPU)
                    # =========================================================
                    var max_next_q = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)

                    @parameter
                    if Self.double_dqn:
                        for i in range(Self.batch_size):
                            var best_action = 0
                            var best_online_q = online_next_q_cpu[i * Self.num_actions]
                            for a in range(1, Self.num_actions):
                                var q = online_next_q_cpu[i * Self.num_actions + a]
                                if q > best_online_q:
                                    best_online_q = q
                                    best_action = a
                            max_next_q[i] = target_q_cpu[i * Self.num_actions + best_action]
                    else:
                        for i in range(Self.batch_size):
                            var max_q = target_q_cpu[i * Self.num_actions]
                            for a in range(1, Self.num_actions):
                                var q = target_q_cpu[i * Self.num_actions + a]
                                if q > max_q:
                                    max_q = q
                            max_next_q[i] = max_q

                    # Compute target values
                    var target_values = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
                    for i in range(Self.batch_size):
                        target_values[i] = (
                            Scalar[Self.dtype](batch_rewards_cpu[i])
                            + self.gamma
                            * (1.0 - Scalar[Self.dtype](batch_dones_cpu[i]))
                            * max_next_q[i]
                        )

                    # Compute loss gradients
                    var dq_cpu = List[Scalar[Self.dtype]](capacity=BATCH * OUT)
                    for i in range(BATCH * OUT):
                        dq_cpu.append(Scalar[Self.dtype](0))

                    var batch_size_scalar = Scalar[Self.dtype](Self.batch_size)
                    for i in range(Self.batch_size):
                        var action_idx = Int(batch_actions_cpu[i])
                        var q_idx = i * Self.num_actions + action_idx
                        var td_error = current_q_cpu[q_idx] - target_values[i]
                        dq_cpu[q_idx] = 2.0 * td_error / batch_size_scalar

                    # =========================================================
                    # COPY GRADIENTS TO GPU
                    # =========================================================
                    with dq_gpu.map_to_host() as host:
                        for i in range(BATCH * OUT):
                            host[i] = dq_cpu[i]

                    dW1_gpu.enqueue_fill(0)
                    db1_gpu.enqueue_fill(0)
                    dW2_gpu.enqueue_fill(0)
                    db2_gpu.enqueue_fill(0)
                    dW3_gpu.enqueue_fill(0)
                    db3_gpu.enqueue_fill(0)
                    dh1_gpu.enqueue_fill(0)
                    dh2_gpu.enqueue_fill(0)

                    # =========================================================
                    # GPU BACKWARD PASS
                    # =========================================================
                    var dq_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin
                    ](dq_gpu)
                    var dW3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H2, OUT), MutAnyOrigin
                    ](dW3_gpu)
                    var db3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(OUT), MutAnyOrigin
                    ](db3_gpu)
                    var dW2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H1, H2), MutAnyOrigin
                    ](dW2_gpu)
                    var db2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H2), MutAnyOrigin
                    ](db2_gpu)
                    var dW1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(OBS, H1), MutAnyOrigin
                    ](dW1_gpu)
                    var db1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(H1), MutAnyOrigin
                    ](db1_gpu)
                    var dh1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H1), MutAnyOrigin
                    ](dh1_gpu)
                    var dh2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H2), MutAnyOrigin
                    ](dh2_gpu)

                    # Layer 3 backward: dW3 = h2.T @ dq
                    comptime kernel_dW3 = linear_backward_dW_kernel[
                        Self.dtype, BATCH, H2, OUT, Self.TILE
                    ]
                    ctx.enqueue_function_checked[kernel_dW3, kernel_dW3](
                        dW3_t, h2_immut, dq_t,
                        grid_dim=(blocks_dW3_x, blocks_dW3_y),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # db3 = sum(dq, axis=0)
                    comptime kernel_db3 = linear_backward_db_kernel[
                        Self.dtype, BATCH, OUT, Self.TPB
                    ]
                    ctx.enqueue_function_checked[kernel_db3, kernel_db3](
                        db3_t, dq_t,
                        grid_dim=(OUT,),
                        block_dim=(Self.TPB,),
                    )

                    # dh2_pre = dq @ W3.T
                    comptime kernel_dx3 = linear_backward_dx_kernel[
                        Self.dtype, BATCH, H2, OUT, Self.TILE
                    ]
                    ctx.enqueue_function_checked[kernel_dx3, kernel_dx3](
                        dh2_t, dq_t, qnet_W3_t,
                        grid_dim=(blocks_dh2_x, blocks_dh2_y),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # Apply ReLU gradient: dh2 = dh2_pre * (h2 > 0)
                    var dh2_flat = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH * H2), MutAnyOrigin
                    ](dh2_gpu)
                    var dh2_in_flat = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH * H2), ImmutAnyOrigin
                    ](dh2_gpu)
                    var h2_flat = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH * H2), ImmutAnyOrigin
                    ](h2_gpu)
                    comptime kernel_relu2 = relu_grad_mul_kernel[
                        Self.dtype, BATCH * H2, Self.TPB
                    ]
                    ctx.enqueue_function_checked[kernel_relu2, kernel_relu2](
                        dh2_flat, dh2_in_flat, h2_flat,
                        grid_dim=(blocks_relu2,),
                        block_dim=(Self.TPB,),
                    )

                    # Layer 2 backward: dW2 = h1.T @ dh2
                    var dh2_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H2), ImmutAnyOrigin
                    ](dh2_gpu)
                    comptime kernel_dW2 = linear_backward_dW_kernel[
                        Self.dtype, BATCH, H1, H2, Self.TILE
                    ]
                    ctx.enqueue_function_checked[kernel_dW2, kernel_dW2](
                        dW2_t, h1_immut, dh2_immut,
                        grid_dim=(blocks_dW2_x, blocks_dW2_y),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # db2 = sum(dh2, axis=0)
                    comptime kernel_db2 = linear_backward_db_kernel[
                        Self.dtype, BATCH, H2, Self.TPB
                    ]
                    ctx.enqueue_function_checked[kernel_db2, kernel_db2](
                        db2_t, dh2_immut,
                        grid_dim=(H2,),
                        block_dim=(Self.TPB,),
                    )

                    # dh1_pre = dh2 @ W2.T
                    comptime kernel_dx2 = linear_backward_dx_kernel[
                        Self.dtype, BATCH, H1, H2, Self.TILE
                    ]
                    ctx.enqueue_function_checked[kernel_dx2, kernel_dx2](
                        dh1_t, dh2_immut, qnet_W2_t,
                        grid_dim=(blocks_dh1_x, blocks_dh1_y),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # Apply ReLU gradient: dh1 = dh1_pre * (h1 > 0)
                    var dh1_flat = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH * H1), MutAnyOrigin
                    ](dh1_gpu)
                    var dh1_in_flat = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH * H1), ImmutAnyOrigin
                    ](dh1_gpu)
                    var h1_flat = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH * H1), ImmutAnyOrigin
                    ](h1_gpu)
                    comptime kernel_relu1 = relu_grad_mul_kernel[
                        Self.dtype, BATCH * H1, Self.TPB
                    ]
                    ctx.enqueue_function_checked[kernel_relu1, kernel_relu1](
                        dh1_flat, dh1_in_flat, h1_flat,
                        grid_dim=(blocks_relu1,),
                        block_dim=(Self.TPB,),
                    )

                    # Layer 1 backward: dW1 = obs.T @ dh1
                    var dh1_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(BATCH, H1), ImmutAnyOrigin
                    ](dh1_gpu)
                    comptime kernel_dW1 = linear_backward_dW_kernel[
                        Self.dtype, BATCH, OBS, H1, Self.TILE
                    ]
                    ctx.enqueue_function_checked[kernel_dW1, kernel_dW1](
                        dW1_t, obs_t, dh1_immut,
                        grid_dim=(blocks_dW1_x, blocks_dW1_y),
                        block_dim=(Self.TILE, Self.TILE),
                    )

                    # db1 = sum(dh1, axis=0)
                    comptime kernel_db1 = linear_backward_db_kernel[
                        Self.dtype, BATCH, H1, Self.TPB
                    ]
                    ctx.enqueue_function_checked[kernel_db1, kernel_db1](
                        db1_t, dh1_immut,
                        grid_dim=(H1,),
                        block_dim=(Self.TPB,),
                    )

                    # =========================================================
                    # GPU ADAM UPDATE
                    # =========================================================
                    self.adam_t += 1
                    var bias_correction1 = Scalar[Self.dtype](
                        1.0 - (Float64(0.9) ** self.adam_t)
                    )
                    var bias_correction2 = Scalar[Self.dtype](
                        1.0 - (Float64(0.999) ** self.adam_t)
                    )
                    var beta1 = Scalar[Self.dtype](0.9)
                    var beta2 = Scalar[Self.dtype](0.999)
                    var eps = Scalar[Self.dtype](1e-8)

                    # Create mutable weight tensors for Adam update
                    var qnet_W1_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W1_SIZE), MutAnyOrigin
                    ](qnet_W1_gpu)
                    var qnet_b1_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B1_SIZE), MutAnyOrigin
                    ](qnet_b1_gpu)
                    var qnet_W2_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W2_SIZE), MutAnyOrigin
                    ](qnet_W2_gpu)
                    var qnet_b2_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B2_SIZE), MutAnyOrigin
                    ](qnet_b2_gpu)
                    var qnet_W3_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W3_SIZE), MutAnyOrigin
                    ](qnet_W3_gpu)
                    var qnet_b3_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B3_SIZE), MutAnyOrigin
                    ](qnet_b3_gpu)

                    var dW1_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W1_SIZE), ImmutAnyOrigin
                    ](dW1_gpu)
                    var db1_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B1_SIZE), ImmutAnyOrigin
                    ](db1_gpu)
                    var dW2_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W2_SIZE), ImmutAnyOrigin
                    ](dW2_gpu)
                    var db2_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B2_SIZE), ImmutAnyOrigin
                    ](db2_gpu)
                    var dW3_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W3_SIZE), ImmutAnyOrigin
                    ](dW3_gpu)
                    var db3_immut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B3_SIZE), ImmutAnyOrigin
                    ](db3_gpu)

                    var m_W1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W1_SIZE), MutAnyOrigin
                    ](m_W1_gpu)
                    var v_W1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W1_SIZE), MutAnyOrigin
                    ](v_W1_gpu)
                    var m_b1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B1_SIZE), MutAnyOrigin
                    ](m_b1_gpu)
                    var v_b1_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B1_SIZE), MutAnyOrigin
                    ](v_b1_gpu)
                    var m_W2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W2_SIZE), MutAnyOrigin
                    ](m_W2_gpu)
                    var v_W2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W2_SIZE), MutAnyOrigin
                    ](v_W2_gpu)
                    var m_b2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B2_SIZE), MutAnyOrigin
                    ](m_b2_gpu)
                    var v_b2_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B2_SIZE), MutAnyOrigin
                    ](v_b2_gpu)
                    var m_W3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W3_SIZE), MutAnyOrigin
                    ](m_W3_gpu)
                    var v_W3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W3_SIZE), MutAnyOrigin
                    ](v_W3_gpu)
                    var m_b3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B3_SIZE), MutAnyOrigin
                    ](m_b3_gpu)
                    var v_b3_t = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B3_SIZE), MutAnyOrigin
                    ](v_b3_gpu)

                    # Adam update W1
                    comptime adam_W1 = adam_update_kernel[Self.dtype, Self.W1_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[adam_W1, adam_W1](
                        qnet_W1_mut, dW1_immut, m_W1_t, v_W1_t,
                        self.lr, beta1, beta2, eps, bias_correction1, bias_correction2,
                        grid_dim=(blocks_W1,),
                        block_dim=(Self.TPB,),
                    )

                    # Adam update b1
                    comptime adam_b1 = adam_update_kernel[Self.dtype, Self.B1_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[adam_b1, adam_b1](
                        qnet_b1_mut, db1_immut, m_b1_t, v_b1_t,
                        self.lr, beta1, beta2, eps, bias_correction1, bias_correction2,
                        grid_dim=(blocks_b1,),
                        block_dim=(Self.TPB,),
                    )

                    # Adam update W2
                    comptime adam_W2 = adam_update_kernel[Self.dtype, Self.W2_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[adam_W2, adam_W2](
                        qnet_W2_mut, dW2_immut, m_W2_t, v_W2_t,
                        self.lr, beta1, beta2, eps, bias_correction1, bias_correction2,
                        grid_dim=(blocks_W2,),
                        block_dim=(Self.TPB,),
                    )

                    # Adam update b2
                    comptime adam_b2 = adam_update_kernel[Self.dtype, Self.B2_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[adam_b2, adam_b2](
                        qnet_b2_mut, db2_immut, m_b2_t, v_b2_t,
                        self.lr, beta1, beta2, eps, bias_correction1, bias_correction2,
                        grid_dim=(blocks_b2,),
                        block_dim=(Self.TPB,),
                    )

                    # Adam update W3
                    comptime adam_W3 = adam_update_kernel[Self.dtype, Self.W3_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[adam_W3, adam_W3](
                        qnet_W3_mut, dW3_immut, m_W3_t, v_W3_t,
                        self.lr, beta1, beta2, eps, bias_correction1, bias_correction2,
                        grid_dim=(blocks_W3,),
                        block_dim=(Self.TPB,),
                    )

                    # Adam update b3
                    comptime adam_b3 = adam_update_kernel[Self.dtype, Self.B3_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[adam_b3, adam_b3](
                        qnet_b3_mut, db3_immut, m_b3_t, v_b3_t,
                        self.lr, beta1, beta2, eps, bias_correction1, bias_correction2,
                        grid_dim=(blocks_b3,),
                        block_dim=(Self.TPB,),
                    )

                    # =========================================================
                    # GPU SOFT UPDATE TARGET NETWORK
                    # =========================================================
                    var qnet_W1_src = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W1_SIZE), ImmutAnyOrigin
                    ](qnet_W1_gpu)
                    var qnet_b1_src = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B1_SIZE), ImmutAnyOrigin
                    ](qnet_b1_gpu)
                    var qnet_W2_src = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W2_SIZE), ImmutAnyOrigin
                    ](qnet_W2_gpu)
                    var qnet_b2_src = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B2_SIZE), ImmutAnyOrigin
                    ](qnet_b2_gpu)
                    var qnet_W3_src = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W3_SIZE), ImmutAnyOrigin
                    ](qnet_W3_gpu)
                    var qnet_b3_src = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B3_SIZE), ImmutAnyOrigin
                    ](qnet_b3_gpu)

                    var target_W1_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W1_SIZE), MutAnyOrigin
                    ](target_W1_gpu)
                    var target_b1_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B1_SIZE), MutAnyOrigin
                    ](target_b1_gpu)
                    var target_W2_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W2_SIZE), MutAnyOrigin
                    ](target_W2_gpu)
                    var target_b2_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B2_SIZE), MutAnyOrigin
                    ](target_b2_gpu)
                    var target_W3_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.W3_SIZE), MutAnyOrigin
                    ](target_W3_gpu)
                    var target_b3_mut = LayoutTensor[
                        Self.dtype, Layout.row_major(Self.B3_SIZE), MutAnyOrigin
                    ](target_b3_gpu)

                    comptime soft_W1 = soft_update_kernel[Self.dtype, Self.W1_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[soft_W1, soft_W1](
                        target_W1_mut, qnet_W1_src, self.tau,
                        grid_dim=(blocks_W1,),
                        block_dim=(Self.TPB,),
                    )

                    comptime soft_b1 = soft_update_kernel[Self.dtype, Self.B1_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[soft_b1, soft_b1](
                        target_b1_mut, qnet_b1_src, self.tau,
                        grid_dim=(blocks_b1,),
                        block_dim=(Self.TPB,),
                    )

                    comptime soft_W2 = soft_update_kernel[Self.dtype, Self.W2_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[soft_W2, soft_W2](
                        target_W2_mut, qnet_W2_src, self.tau,
                        grid_dim=(blocks_W2,),
                        block_dim=(Self.TPB,),
                    )

                    comptime soft_b2 = soft_update_kernel[Self.dtype, Self.B2_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[soft_b2, soft_b2](
                        target_b2_mut, qnet_b2_src, self.tau,
                        grid_dim=(blocks_b2,),
                        block_dim=(Self.TPB,),
                    )

                    comptime soft_W3 = soft_update_kernel[Self.dtype, Self.W3_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[soft_W3, soft_W3](
                        target_W3_mut, qnet_W3_src, self.tau,
                        grid_dim=(blocks_W3,),
                        block_dim=(Self.TPB,),
                    )

                    comptime soft_b3 = soft_update_kernel[Self.dtype, Self.B3_SIZE, Self.TPB]
                    ctx.enqueue_function_checked[soft_b3, soft_b3](
                        target_b3_mut, qnet_b3_src, self.tau,
                        grid_dim=(blocks_b3,),
                        block_dim=(Self.TPB,),
                    )

                episode_reward += reward
                obs_list = env.get_obs_list()
                steps += 1

                # Periodically sync weights back to CPU for action selection
                if steps % 100 == 0:
                    ctx.synchronize()
                    with qnet_W1_gpu.map_to_host() as host:
                        for i in range(Self.W1_SIZE):
                            self.q_network.W1[i] = host[i]
                    with qnet_b1_gpu.map_to_host() as host:
                        for i in range(Self.B1_SIZE):
                            self.q_network.b1[i] = host[i]
                    with qnet_W2_gpu.map_to_host() as host:
                        for i in range(Self.W2_SIZE):
                            self.q_network.W2[i] = host[i]
                    with qnet_b2_gpu.map_to_host() as host:
                        for i in range(Self.B2_SIZE):
                            self.q_network.b2[i] = host[i]
                    with qnet_W3_gpu.map_to_host() as host:
                        for i in range(Self.W3_SIZE):
                            self.q_network.W3[i] = host[i]
                    with qnet_b3_gpu.map_to_host() as host:
                        for i in range(Self.B3_SIZE):
                            self.q_network.b3[i] = host[i]

            metrics.log_episode(
                episode, episode_reward, steps, Float64(self.epsilon)
            )
            self.total_episodes += 1
            self.decay_epsilon()

            if verbose and (episode + 1) % print_every == 0:
                var start_idx = max(0, len(metrics.episodes) - print_every)
                var sum_reward: Float64 = 0.0
                for j in range(start_idx, len(metrics.episodes)):
                    sum_reward += metrics.episodes[j].total_reward
                var avg_reward = sum_reward / Float64(
                    len(metrics.episodes) - start_idx
                )
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg Reward: "
                    + String(avg_reward)[:8]
                    + " | Steps: "
                    + String(steps)
                    + " | Epsilon: "
                    + String(self.epsilon)[:5]
                )

        # =====================================================================
        # COPY FINAL WEIGHTS BACK TO CPU
        # =====================================================================
        ctx.synchronize()
        with qnet_W1_gpu.map_to_host() as host:
            for i in range(Self.W1_SIZE):
                self.q_network.W1[i] = host[i]
        with qnet_b1_gpu.map_to_host() as host:
            for i in range(Self.B1_SIZE):
                self.q_network.b1[i] = host[i]
        with qnet_W2_gpu.map_to_host() as host:
            for i in range(Self.W2_SIZE):
                self.q_network.W2[i] = host[i]
        with qnet_b2_gpu.map_to_host() as host:
            for i in range(Self.B2_SIZE):
                self.q_network.b2[i] = host[i]
        with qnet_W3_gpu.map_to_host() as host:
            for i in range(Self.W3_SIZE):
                self.q_network.W3[i] = host[i]
        with qnet_b3_gpu.map_to_host() as host:
            for i in range(Self.B3_SIZE):
                self.q_network.b3[i] = host[i]

        if verbose:
            print("-" * 60)
            print("Training complete!")
            var start_idx = max(0, len(metrics.episodes) - 100)
            var sum_reward: Float64 = 0.0
            for j in range(start_idx, len(metrics.episodes)):
                sum_reward += metrics.episodes[j].total_reward
            var final_avg = sum_reward / Float64(
                len(metrics.episodes) - start_idx
            )
            print("Final avg reward (last 100): " + String(final_avg)[:8])

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
    ) -> Float64:
        """Evaluate the trained agent using greedy policy."""
        var total_reward: Float64 = 0.0

        for ep in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var done = False
            var steps = 0

            while not done and steps < max_steps_per_episode:
                var obs = _list_to_inline_dtype[Self.obs_dim, Self.dtype](
                    obs_list
                )
                var action = self.select_action(obs, training=False)
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                episode_reward += reward
                obs_list = step_result[0].copy()
                steps += 1

            total_reward += episode_reward

            if verbose:
                print(
                    "  Eval episode "
                    + String(ep + 1)
                    + ": "
                    + String(episode_reward)[:10]
                    + " (steps: "
                    + String(steps)
                    + ")"
                )

        return total_reward / Float64(num_episodes)


# ============================================================================
# Helper functions
# ============================================================================


fn _list_to_inline[
    size: Int, dtype: DType = DType.float64
](obs_list: List[Float64]) -> InlineArray[Scalar[dtype], size]:
    """Convert List[Float64] to InlineArray."""
    var obs = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        if i < len(obs_list):
            obs[i] = Scalar[dtype](obs_list[i])
    return obs^


fn _list_to_inline_dtype[
    size: Int, dtype: DType
](obs_list: List[Float64]) -> InlineArray[Scalar[dtype], size]:
    """Convert List[Float64] to InlineArray with specific dtype."""
    var obs = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        if i < len(obs_list):
            obs[i] = Scalar[dtype](obs_list[i])
    return obs^


fn max(a: Int, b: Int) -> Int:
    """Return maximum of two integers."""
    return a if a > b else b
