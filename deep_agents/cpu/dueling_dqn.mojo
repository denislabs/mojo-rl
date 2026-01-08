"""Deep Dueling DQN Agent for discrete action spaces.

Dueling DQN separates the Q-network into two streams:
- Value stream V(s): Estimates state value (how good is this state?)
- Advantage stream A(s,a): Estimates relative action advantages

Final Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A))

This decomposition helps the network learn which states are valuable
without having to learn the effect of each action for every state.
Particularly useful when actions don't always affect the outcome.

Reference: Wang et al. "Dueling Network Architectures for Deep RL" (2016)

Example usage:
    from deep_agents.cpu import DeepDuelingDQNAgent
    from envs import LunarLanderEnv

    var env = LunarLanderEnv()
    var agent = DeepDuelingDQNAgent[obs_dim=8, num_actions=4, hidden_dim=128]()

    var metrics = agent.train(env, num_episodes=500)
    var eval_reward = agent.evaluate(env)
"""

from random import random_float64

from deep_rl.cpu import (
    LinearAdam,
    ReplayBuffer,
    relu,
    relu_grad,
    elementwise_mul,
    zeros,
)
from core import TrainingMetrics, BoxDiscreteActionEnv


# =============================================================================
# Dueling Q-Network
# =============================================================================


struct DuelingQNetwork[
    obs_dim: Int,
    num_actions: Int,
    hidden1_dim: Int = 256,
    hidden2_dim: Int = 256,
    value_hidden_dim: Int = 128,
    advantage_hidden_dim: Int = 128,
    dtype: DType = DType.float64,
]:
    """Dueling Q-Network that separates value and advantage estimation.

    Architecture:
        obs_dim -> hidden1 (relu) -> hidden2 (relu) -> [split]
                                                        |
                              +-------------------------+-------------------------+
                              |                                                   |
                       Value Stream                                       Advantage Stream
                              |                                                   |
                    value_hidden (relu)                              advantage_hidden (relu)
                              |                                                   |
                          V(s) [1]                                        A(s,a) [num_actions]
                              |                                                   |
                              +-------------------------+-------------------------+
                                                        |
                                          Q(s,a) = V(s) + (A(s,a) - mean(A))
    """

    # Shared feature layers
    var layer1: LinearAdam[Self.obs_dim, Self.hidden1_dim, Self.dtype]
    var layer2: LinearAdam[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]

    # Value stream: estimates V(s)
    var value_hidden: LinearAdam[
        Self.hidden2_dim, Self.value_hidden_dim, Self.dtype
    ]
    var value_out: LinearAdam[Self.value_hidden_dim, 1, Self.dtype]

    # Advantage stream: estimates A(s,a) for each action
    var advantage_hidden: LinearAdam[
        Self.hidden2_dim, Self.advantage_hidden_dim, Self.dtype
    ]
    var advantage_out: LinearAdam[
        Self.advantage_hidden_dim, Self.num_actions, Self.dtype
    ]

    fn __init__(out self):
        """Initialize Dueling Q-Network with Xavier initialization."""
        # Shared layers
        self.layer1 = LinearAdam[Self.obs_dim, Self.hidden1_dim, Self.dtype]()
        self.layer2 = LinearAdam[
            Self.hidden1_dim, Self.hidden2_dim, Self.dtype
        ]()

        # Value stream
        self.value_hidden = LinearAdam[
            Self.hidden2_dim, Self.value_hidden_dim, Self.dtype
        ]()
        self.value_out = LinearAdam[Self.value_hidden_dim, 1, Self.dtype]()

        # Advantage stream
        self.advantage_hidden = LinearAdam[
            Self.hidden2_dim, Self.advantage_hidden_dim, Self.dtype
        ]()
        self.advantage_out = LinearAdam[
            Self.advantage_hidden_dim, Self.num_actions, Self.dtype
        ]()

    fn forward[
        batch_size: Int
    ](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.num_actions]:
        """Forward pass: obs -> Q-values for all actions.

        Returns Q(s, a) = V(s) + (A(s,a) - mean(A)) for each action a.
        """
        # ============================
        # Shared feature extraction
        # ============================

        # Layer 1: linear + relu
        var h1_pre = self.layer1.forward[batch_size](obs)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        # Layer 2: linear + relu
        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # ============================
        # Value stream: V(s)
        # ============================

        var v_hidden_pre = self.value_hidden.forward[batch_size](h2)
        var v_hidden = relu[batch_size * Self.value_hidden_dim, Self.dtype](
            v_hidden_pre
        )
        var value = self.value_out.forward[batch_size](
            v_hidden
        )  # [batch_size, 1]

        # ============================
        # Advantage stream: A(s,a)
        # ============================

        var a_hidden_pre = self.advantage_hidden.forward[batch_size](h2)
        var a_hidden = relu[batch_size * Self.advantage_hidden_dim, Self.dtype](
            a_hidden_pre
        )
        var advantage = self.advantage_out.forward[batch_size](
            a_hidden
        )  # [batch_size, num_actions]

        # ============================
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        # ============================

        var q_values = InlineArray[
            Scalar[Self.dtype], batch_size * Self.num_actions
        ](fill=0)

        for i in range(batch_size):
            # Compute mean advantage for this sample
            var mean_adv: Scalar[Self.dtype] = 0.0
            for a in range(Self.num_actions):
                mean_adv += advantage[i * Self.num_actions + a]
            mean_adv /= Scalar[Self.dtype](Self.num_actions)

            # Q(s,a) = V(s) + (A(s,a) - mean(A))
            var v_s = value[i]
            for a in range(Self.num_actions):
                var adv_idx = i * Self.num_actions + a
                q_values[adv_idx] = v_s + (advantage[adv_idx] - mean_adv)

        return q_values^

    fn forward_with_cache[
        batch_size: Int
    ](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        mut h1_out: InlineArray[
            Scalar[Self.dtype], batch_size * Self.hidden1_dim
        ],
        mut h2_out: InlineArray[
            Scalar[Self.dtype], batch_size * Self.hidden2_dim
        ],
        mut v_hidden_out: InlineArray[
            Scalar[Self.dtype], batch_size * Self.value_hidden_dim
        ],
        mut a_hidden_out: InlineArray[
            Scalar[Self.dtype], batch_size * Self.advantage_hidden_dim
        ],
        mut value_out: InlineArray[Scalar[Self.dtype], batch_size],
        mut advantage_out: InlineArray[
            Scalar[Self.dtype], batch_size * Self.num_actions
        ],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.num_actions]:
        """Forward pass with cached activations for backward."""
        # Shared feature extraction
        var h1_pre = self.layer1.forward[batch_size](obs)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Value stream
        var v_hidden_pre = self.value_hidden.forward[batch_size](h2)
        var v_hidden = relu[batch_size * Self.value_hidden_dim, Self.dtype](
            v_hidden_pre
        )
        var value = self.value_out.forward[batch_size](v_hidden)

        # Advantage stream
        var a_hidden_pre = self.advantage_hidden.forward[batch_size](h2)
        var a_hidden = relu[batch_size * Self.advantage_hidden_dim, Self.dtype](
            a_hidden_pre
        )
        var advantage = self.advantage_out.forward[batch_size](a_hidden)

        # Store caches
        for i in range(batch_size * Self.hidden1_dim):
            h1_out[i] = h1[i]
        for i in range(batch_size * Self.hidden2_dim):
            h2_out[i] = h2[i]
        for i in range(batch_size * Self.value_hidden_dim):
            v_hidden_out[i] = v_hidden[i]
        for i in range(batch_size * Self.advantage_hidden_dim):
            a_hidden_out[i] = a_hidden[i]
        for i in range(batch_size):
            value_out[i] = value[i]
        for i in range(batch_size * Self.num_actions):
            advantage_out[i] = advantage[i]

        # Combine: Q = V + (A - mean(A))
        var q_values = InlineArray[
            Scalar[Self.dtype], batch_size * Self.num_actions
        ](fill=0)

        for i in range(batch_size):
            var mean_adv: Scalar[Self.dtype] = 0.0
            for a in range(Self.num_actions):
                mean_adv += advantage[i * Self.num_actions + a]
            mean_adv /= Scalar[Self.dtype](Self.num_actions)

            var v_s = value[i]
            for a in range(Self.num_actions):
                var adv_idx = i * Self.num_actions + a
                q_values[adv_idx] = v_s + (advantage[adv_idx] - mean_adv)

        return q_values^

    fn backward[
        batch_size: Int
    ](
        mut self,
        dq: InlineArray[Scalar[Self.dtype], batch_size * Self.num_actions],
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        h1: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        h2: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
        v_hidden: InlineArray[
            Scalar[Self.dtype], batch_size * Self.value_hidden_dim
        ],
        a_hidden: InlineArray[
            Scalar[Self.dtype], batch_size * Self.advantage_hidden_dim
        ],
    ):
        """Backward pass through Dueling Q-Network.

        Gradient flow:
            dQ -> dV, dA (from Q = V + (A - mean(A)))
            dV -> value_out -> value_hidden -> layer2
            dA -> advantage_out -> advantage_hidden -> layer2
            Combined gradients flow through shared layers.
        """
        # ============================
        # Compute gradients for V and A from dQ
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        #
        # dQ/dV = 1 for all actions -> sum over actions
        # dQ/dA_i = 1 - 1/num_actions (centering term)
        # ============================

        var dv = InlineArray[Scalar[Self.dtype], batch_size](fill=0)
        var da = InlineArray[Scalar[Self.dtype], batch_size * Self.num_actions](
            fill=0
        )

        comptime one_over_n = Scalar[Self.dtype](1.0) / Scalar[Self.dtype](
            Self.num_actions
        )

        for i in range(batch_size):
            # dV = sum of all dQ for this sample (V contributes to all Q values equally)
            var sum_dq: Scalar[Self.dtype] = 0.0
            for a in range(Self.num_actions):
                sum_dq += dq[i * Self.num_actions + a]
            dv[i] = sum_dq

            # dA_i = dQ_i - (1/n) * sum(dQ_j)
            # Because Q_i = V + A_i - mean(A) = V + A_i - (1/n)*sum(A_j)
            # So dQ_i/dA_j = delta_ij - 1/n
            for a in range(Self.num_actions):
                var idx = i * Self.num_actions + a
                da[idx] = dq[idx] - one_over_n * sum_dq

        # ============================
        # Backward through value stream
        # ============================

        var dv_hidden = self.value_out.backward[batch_size](dv, v_hidden)

        # Backprop through relu
        var relu_g_v = relu_grad[
            batch_size * Self.value_hidden_dim, Self.dtype
        ](v_hidden)
        var dv_hidden_pre = elementwise_mul[
            batch_size * Self.value_hidden_dim, Self.dtype
        ](dv_hidden, relu_g_v)

        var dh2_from_v = self.value_hidden.backward[batch_size](
            dv_hidden_pre, h2
        )

        # ============================
        # Backward through advantage stream
        # ============================

        var da_hidden = self.advantage_out.backward[batch_size](da, a_hidden)

        # Backprop through relu
        var relu_g_a = relu_grad[
            batch_size * Self.advantage_hidden_dim, Self.dtype
        ](a_hidden)
        var da_hidden_pre = elementwise_mul[
            batch_size * Self.advantage_hidden_dim, Self.dtype
        ](da_hidden, relu_g_a)

        var dh2_from_a = self.advantage_hidden.backward[batch_size](
            da_hidden_pre, h2
        )

        # ============================
        # Combine gradients and backward through shared layers
        # ============================

        # Sum gradients from both streams
        var dh2 = InlineArray[
            Scalar[Self.dtype], batch_size * Self.hidden2_dim
        ](fill=0)
        for i in range(batch_size * Self.hidden2_dim):
            dh2[i] = dh2_from_v[i] + dh2_from_a[i]

        # Backprop through relu2
        var relu_g2 = relu_grad[batch_size * Self.hidden2_dim, Self.dtype](h2)
        var dh2_pre = elementwise_mul[
            batch_size * Self.hidden2_dim, Self.dtype
        ](dh2, relu_g2)

        # Backward through layer2
        var dh1 = self.layer2.backward[batch_size](dh2_pre, h1)

        # Backprop through relu1
        var relu_g1 = relu_grad[batch_size * Self.hidden1_dim, Self.dtype](h1)
        var dh1_pre = elementwise_mul[
            batch_size * Self.hidden1_dim, Self.dtype
        ](dh1, relu_g1)

        # Backward through layer1
        _ = self.layer1.backward[batch_size](dh1_pre, obs)

    fn update_adam(
        mut self,
        lr: Scalar[Self.dtype] = 0.001,
        beta1: Scalar[Self.dtype] = 0.9,
        beta2: Scalar[Self.dtype] = 0.999,
    ):
        """Update all layers using Adam."""
        # Shared layers
        self.layer1.update_adam(lr, beta1, beta2)
        self.layer2.update_adam(lr, beta1, beta2)

        # Value stream
        self.value_hidden.update_adam(lr, beta1, beta2)
        self.value_out.update_adam(lr, beta1, beta2)

        # Advantage stream
        self.advantage_hidden.update_adam(lr, beta1, beta2)
        self.advantage_out.update_adam(lr, beta1, beta2)

    fn zero_grad(mut self):
        """Reset all gradients."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.value_hidden.zero_grad()
        self.value_out.zero_grad()
        self.advantage_hidden.zero_grad()
        self.advantage_out.zero_grad()

    fn soft_update_from(mut self, source: Self, tau: Scalar[Self.dtype]):
        """Soft update from source network: theta = tau * source + (1-tau) * theta.
        """
        self.layer1.soft_update_from(source.layer1, tau)
        self.layer2.soft_update_from(source.layer2, tau)
        self.value_hidden.soft_update_from(source.value_hidden, tau)
        self.value_out.soft_update_from(source.value_out, tau)
        self.advantage_hidden.soft_update_from(source.advantage_hidden, tau)
        self.advantage_out.soft_update_from(source.advantage_out, tau)

    fn copy_from(mut self, source: Self):
        """Hard copy from source network."""
        self.layer1.copy_from(source.layer1)
        self.layer2.copy_from(source.layer2)
        self.value_hidden.copy_from(source.value_hidden)
        self.value_out.copy_from(source.value_out)
        self.advantage_hidden.copy_from(source.advantage_hidden)
        self.advantage_out.copy_from(source.advantage_out)

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        return (
            self.layer1.num_parameters()
            + self.layer2.num_parameters()
            + self.value_hidden.num_parameters()
            + self.value_out.num_parameters()
            + self.advantage_hidden.num_parameters()
            + self.advantage_out.num_parameters()
        )

    fn print_info(self, name: String = "DuelingQNetwork"):
        """Print network architecture."""
        print(name + ":")
        print(
            "  Shared: "
            + String(Self.obs_dim)
            + " -> "
            + String(Self.hidden1_dim)
            + " (relu)"
            + " -> "
            + String(Self.hidden2_dim)
            + " (relu)"
        )
        print(
            "  Value stream: "
            + String(Self.hidden2_dim)
            + " -> "
            + String(Self.value_hidden_dim)
            + " (relu) -> 1"
        )
        print(
            "  Advantage stream: "
            + String(Self.hidden2_dim)
            + " -> "
            + String(Self.advantage_hidden_dim)
            + " (relu) -> "
            + String(Self.num_actions)
        )
        print("  Q(s,a) = V(s) + (A(s,a) - mean(A))")
        print("  Parameters: " + String(self.num_parameters()))


# =============================================================================
# Deep Dueling DQN Agent
# =============================================================================


struct DeepDuelingDQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    stream_hidden_dim: Int = 64,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    dtype: DType = DType.float64,
    double_dqn: Bool = True,  # Use Double DQN by default
]:
    """Deep Dueling DQN Agent for discrete action spaces.

    Combines Dueling Architecture with (optional) Double DQN:
    - Dueling: Separates V(s) and A(s,a) for better value decomposition
    - Double DQN: Reduces overestimation bias

    Parameters:
        obs_dim: Observation dimension.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size for shared feature layers.
        stream_hidden_dim: Hidden layer size for value/advantage streams.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        dtype: Data type for computations.
        double_dqn: If True, use Double DQN target computation.
    """

    # Dueling Q-Networks
    var q_network: DuelingQNetwork[
        Self.obs_dim,
        Self.num_actions,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.stream_hidden_dim,
        Self.stream_hidden_dim,
        Self.dtype,
    ]
    var target_network: DuelingQNetwork[
        Self.obs_dim,
        Self.num_actions,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.stream_hidden_dim,
        Self.stream_hidden_dim,
        Self.dtype,
    ]

    # Replay buffer (action_dim=1 for discrete actions)
    var buffer: ReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, Self.dtype]

    # Hyperparameters
    var gamma: Scalar[Self.dtype]
    var tau: Scalar[Self.dtype]
    var lr: Scalar[Self.dtype]
    var epsilon: Scalar[Self.dtype]
    var epsilon_min: Scalar[Self.dtype]
    var epsilon_decay: Scalar[Self.dtype]

    # Training state
    var total_steps: Int
    var total_episodes: Int

    fn __init__(
        out self,
        gamma: Scalar[Self.dtype] = 0.99,
        tau: Scalar[Self.dtype] = 0.005,
        lr: Scalar[Self.dtype] = 0.0005,
        epsilon: Scalar[Self.dtype] = 1.0,
        epsilon_min: Scalar[Self.dtype] = 0.01,
        epsilon_decay: Scalar[Self.dtype] = 0.995,
    ):
        """Initialize Deep Dueling DQN agent."""
        self.q_network = DuelingQNetwork[
            Self.obs_dim,
            Self.num_actions,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.stream_hidden_dim,
            Self.stream_hidden_dim,
            Self.dtype,
        ]()
        self.target_network = DuelingQNetwork[
            Self.obs_dim,
            Self.num_actions,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.stream_hidden_dim,
            Self.stream_hidden_dim,
            Self.dtype,
        ]()

        # Initialize target network with same weights
        self.target_network.copy_from(self.q_network)

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, Self.dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.total_episodes = 0

    fn select_action(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        training: Bool = True,
    ) -> Int:
        """Select action using epsilon-greedy policy."""
        # Epsilon-greedy exploration
        if training and random_float64() < Float64(self.epsilon):
            return (
                Int(random_float64() * Float64(Self.num_actions))
                % Self.num_actions
            )

        # Greedy action: argmax Q(s, a)
        var q_values = self.q_network.forward[1](obs)

        var best_action = 0
        var best_q = q_values[0]
        for a in range(1, Self.num_actions):
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        action: Int,
        reward: Scalar[Self.dtype],
        next_obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        done: Bool,
    ):
        """Store a transition in the replay buffer."""
        var action_arr = InlineArray[Scalar[Self.dtype], 1](fill=0)
        action_arr[0] = Scalar[Self.dtype](action)
        self.buffer.add(obs, action_arr, reward, next_obs, done)
        self.total_steps += 1

    fn train_step(mut self) -> Scalar[Self.dtype]:
        """Perform one training step. Returns loss."""
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # Sample batch
        var batch_obs = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_actions = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        var batch_rewards = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        var batch_next_obs = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_dones = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )

        var batch_actions_arr = InlineArray[
            Scalar[Self.dtype], Self.batch_size
        ](fill=0)
        self.buffer.sample[Self.batch_size](
            batch_obs,
            batch_actions_arr,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )
        for i in range(Self.batch_size):
            batch_actions[i] = batch_actions_arr[i]

        # ========================================
        # Compute target Q-values
        # ========================================

        var max_next_q = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )

        @parameter
        if Self.double_dqn:
            # Double DQN: online network selects, target evaluates
            var online_next_q = self.q_network.forward[Self.batch_size](
                batch_next_obs
            )
            var target_next_q = self.target_network.forward[Self.batch_size](
                batch_next_obs
            )

            for i in range(Self.batch_size):
                var best_action = 0
                var best_online_q = online_next_q[i * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = online_next_q[i * Self.num_actions + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a
                max_next_q[i] = target_next_q[
                    i * Self.num_actions + best_action
                ]
        else:
            # Standard DQN: max_a Q_target(s', a)
            var next_q = self.target_network.forward[Self.batch_size](
                batch_next_obs
            )
            for i in range(Self.batch_size):
                var max_q = next_q[i * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = next_q[i * Self.num_actions + a]
                    if q > max_q:
                        max_q = q
                max_next_q[i] = max_q

        # Target: y = r + gamma * (1 - done) * Q(s', a*)
        var target_values = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        for i in range(Self.batch_size):
            target_values[i] = (
                batch_rewards[i]
                + self.gamma * (1.0 - batch_dones[i]) * max_next_q[i]
            )

        # ========================================
        # Forward with cache for backward pass
        # ========================================

        var h1_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var v_hidden_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.stream_hidden_dim
        ](fill=0)
        var a_hidden_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.stream_hidden_dim
        ](fill=0)
        var value_cache = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        var advantage_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.num_actions
        ](fill=0)

        var current_q = self.q_network.forward_with_cache[Self.batch_size](
            batch_obs,
            h1_cache,
            h2_cache,
            v_hidden_cache,
            a_hidden_cache,
            value_cache,
            advantage_cache,
        )

        # ========================================
        # Compute loss and gradients
        # ========================================

        var loss: Scalar[Self.dtype] = 0.0
        var dq = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.num_actions
        ](fill=0)
        var batch_size_scalar = Scalar[Self.dtype](Self.batch_size)

        for i in range(Self.batch_size):
            var action_idx = Int(batch_actions[i])
            var q_idx = i * Self.num_actions + action_idx
            var td_error = current_q[q_idx] - target_values[i]
            loss += td_error * td_error
            dq[q_idx] = 2.0 * td_error / batch_size_scalar

        loss /= batch_size_scalar

        # ========================================
        # Backward pass and update
        # ========================================

        self.q_network.zero_grad()
        self.q_network.backward[Self.batch_size](
            dq, batch_obs, h1_cache, h2_cache, v_hidden_cache, a_hidden_cache
        )
        self.q_network.update_adam(self.lr)

        # Soft update target network
        self.target_network.soft_update_from(self.q_network, self.tau)

        return loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call once per episode)."""
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn print_info(self):
        """Print agent information."""

        @parameter
        if Self.double_dqn:
            print("Deep Dueling Double DQN Agent:")
        else:
            print("Deep Dueling DQN Agent:")
        print("  Obs dim: " + String(Self.obs_dim))
        print("  Num actions: " + String(Self.num_actions))
        print("  Hidden dim: " + String(Self.hidden_dim))
        print("  Stream hidden dim: " + String(Self.stream_hidden_dim))
        print("  Buffer capacity: " + String(Self.buffer_capacity))
        print("  Batch size: " + String(Self.batch_size))
        print("  Double DQN: " + String(Self.double_dqn))
        print("  Gamma: " + String(self.gamma)[:6])
        print("  Tau: " + String(self.tau)[:6])
        print("  LR: " + String(self.lr)[:8])
        print("  Epsilon: " + String(self.epsilon)[:5])
        print("  Epsilon min: " + String(self.epsilon_min)[:5])
        print("  Epsilon decay: " + String(self.epsilon_decay)[:6])
        self.q_network.print_info("  Q-Network")

    # ========================================================================
    # Training and Evaluation
    # ========================================================================

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = True,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the Deep Dueling DQN agent on a discrete action environment."""
        var metrics = TrainingMetrics(
            algorithm_name="Deep Dueling DQN",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("Deep Dueling DQN Training on " + environment_name)
            print("=" * 60)
            self.print_info()
            print("-" * 60)

        # Warmup phase
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

                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)
                var next_obs = _list_to_inline[Self.obs_dim, Self.dtype](
                    step_result[0]
                )

                self.store_transition(
                    obs, action, Scalar[Self.dtype](reward), next_obs, done
                )

                obs_list = env.get_obs_list()
                warmup_done += 1

        if verbose:
            print("Warmup complete. Buffer size: " + String(self.buffer.len()))
            print("-" * 60)

        # Training loop
        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var steps = 0
            var done = False

            while not done and steps < max_steps_per_episode:
                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)
                var action = self.select_action(obs, training=True)
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                var next_obs = _list_to_inline[Self.obs_dim, Self.dtype](
                    step_result[0]
                )
                self.store_transition(
                    obs, action, Scalar[Self.dtype](reward), next_obs, done
                )

                if steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                obs_list = env.get_obs_list()
                steps += 1

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
        render: Bool = False,
    ) -> Float64:
        """Evaluate the trained agent using greedy policy (no exploration)."""
        var total_reward: Float64 = 0.0

        for ep in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var done = False
            var steps = 0

            while not done and steps < max_steps_per_episode:
                if render:
                    env.render()

                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)
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


fn max(a: Int, b: Int) -> Int:
    """Return maximum of two integers."""
    return a if a > b else b
