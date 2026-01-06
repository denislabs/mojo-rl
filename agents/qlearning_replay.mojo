from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent, ReplayBuffer, DiscreteEnv, train_tabular_with_metrics, TrainingMetrics


struct QLearningReplayAgent(TabularAgent, Copyable, Movable, ImplicitlyCopyable):
    """Q-Learning agent with Experience Replay.

    Combines Q-Learning with experience replay buffer for more stable
    and sample-efficient learning.

    After each real transition:
    1. Store transition in replay buffer
    2. Sample a mini-batch from buffer
    3. Perform Q-learning updates on all batch samples

    Benefits:
    - Breaks correlation between consecutive samples
    - Reuses past experience (higher sample efficiency)
    - More stable learning
    - Precursor to DQN-style algorithms

    Parameters:
        buffer_size: Maximum number of transitions to store
        batch_size: Number of transitions to sample per update
        min_buffer_size: Minimum buffer size before learning starts
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int
    var num_states: Int

    var buffer: ReplayBuffer
    var batch_size: Int
    var min_buffer_size: Int

    fn __copyinit__(out self, existing: Self):
        self.q_table = existing.q_table
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions
        self.num_states = existing.num_states
        self.buffer = existing.buffer
        self.batch_size = existing.batch_size
        self.min_buffer_size = existing.min_buffer_size

    fn __moveinit__(out self, deinit existing: Self):
        self.q_table = existing.q_table^
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions
        self.num_states = existing.num_states
        self.buffer = existing.buffer^
        self.batch_size = existing.batch_size
        self.min_buffer_size = existing.min_buffer_size

    fn __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        buffer_size: Int = 1000,
        batch_size: Int = 32,
        min_buffer_size: Int = 100,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
    ):
        self.q_table = QTable(num_states, num_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_states = num_states

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size

    fn select_action(self, state_idx: Int) -> Int:
        """Select action using epsilon-greedy policy."""
        var rand = random_float64()
        if rand < self.epsilon:
            # random_si64 is inclusive on both ends, so use num_actions - 1
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.q_table.get_best_action(state_idx)

    fn _q_update(
        mut self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
    ):
        """Single Q-learning update."""
        var current_q = self.q_table.get(state, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(next_state)
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state, action, new_q)

    fn update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Store transition and learn from replay batch."""
        # Store transition in buffer
        self.buffer.push(state_idx, action, reward, next_state_idx, done)

        # Only start learning after buffer has enough samples
        if self.buffer.len() < self.min_buffer_size:
            return

        # Sample batch and update
        var batch = self.buffer.sample(self.batch_size)
        for i in range(len(batch)):
            var t = batch[i]
            self._q_update(t.state, t.action, t.reward, t.next_state, t.done)

    fn decay_epsilon(mut self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current epsilon value."""
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
        """Return the greedy action for a state."""
        return self.q_table.get_best_action(state_idx)

    # ========================================================================
    # Static training method
    # ========================================================================

    @staticmethod
    fn train[E: DiscreteEnv](
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 100,
        buffer_size: Int = 1000,
        batch_size: Int = 32,
        min_buffer_size: Int = 100,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> Tuple[QLearningReplayAgent, TrainingMetrics]:
        """Train a Q-Learning with Replay agent on the given environment.

        Args:
            env: The discrete environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            buffer_size: Maximum replay buffer size.
            batch_size: Mini-batch size for updates.
            min_buffer_size: Minimum buffer size before learning.
            learning_rate: Learning rate (alpha).
            discount_factor: Discount factor (gamma).
            epsilon: Initial exploration rate.
            epsilon_decay: Exploration decay rate per episode.
            epsilon_min: Minimum exploration rate.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            Tuple of (trained_agent, training_metrics).
        """
        var agent = QLearningReplayAgent(
            env.num_states(),
            env.num_actions(),
            buffer_size,
            batch_size,
            min_buffer_size,
            learning_rate,
            discount_factor,
            epsilon,
            epsilon_decay,
            epsilon_min,
        )
        var metrics = train_tabular_with_metrics(
            env,
            agent,
            num_episodes,
            max_steps_per_episode,
            verbose,
            print_every,
            algorithm_name="Q-Learning + Replay",
            environment_name=environment_name,
        )
        return (agent^, metrics^)
