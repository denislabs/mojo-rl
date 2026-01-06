from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent, DiscreteEnv, train_tabular_with_metrics, TrainingMetrics


struct SARSAAgent(TabularAgent, Copyable, Movable, ImplicitlyCopyable):
    """Tabular SARSA agent with epsilon-greedy exploration.

    SARSA is on-policy: uses Q(s',a') instead of max Q(s',a').
    For generic training, update() uses Q-learning style (max).
    Use update_sarsa() for true SARSA with next_action.
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int

    fn __copyinit__(out self, existing: Self):
        self.q_table = existing.q_table
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions

    fn __moveinit__(out self, deinit existing: Self):
        self.q_table = existing.q_table^
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions

    fn __init__(
        out self,
        num_states: Int,
        num_actions: Int,
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

    fn select_action(self, state_idx: Int) -> Int:
        var rand = random_float64()
        if rand < self.epsilon:
            # random_si64 is inclusive on both ends, so use num_actions - 1
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.q_table.get_best_action(state_idx)

    fn update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """TabularAgent interface (Q-learning style for generic training)."""
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(next_state_idx)
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

    fn update_sarsa(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        next_action: Int,
        done: Bool,
    ):
        """True SARSA: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))."""
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get(next_state_idx, next_action)
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

    fn decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
        return self.q_table.get_best_action(state_idx)

    # ========================================================================
    # Static training methods
    # ========================================================================

    @staticmethod
    fn train[E: DiscreteEnv](
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 100,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> Tuple[SARSAAgent, TrainingMetrics]:
        """Train a SARSA agent using the generic training loop.

        Note: This uses Q-learning style updates (max Q(s',a')) for compatibility
        with the generic training interface. Use train_sarsa() for true on-policy
        SARSA with Q(s',a') updates.

        Args:
            env: The discrete environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
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
        var agent = SARSAAgent(
            env.num_states(),
            env.num_actions(),
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
            algorithm_name="SARSA",
            environment_name=environment_name,
        )
        return (agent^, metrics^)

    @staticmethod
    fn train_sarsa[E: DiscreteEnv](
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 100,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> Tuple[SARSAAgent, TrainingMetrics]:
        """Train a SARSA agent with true on-policy updates.

        This uses the proper SARSA update: Q(s,a) += α * (r + γ*Q(s',a') - Q(s,a))
        where a' is the action actually selected (not the max).

        Args:
            env: The discrete environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
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
        var agent = SARSAAgent(
            env.num_states(),
            env.num_actions(),
            learning_rate,
            discount_factor,
            epsilon,
            epsilon_decay,
            epsilon_min,
        )
        var metrics = TrainingMetrics(
            algorithm_name="SARSA (on-policy)",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            var state = env.reset()
            var state_idx = env.state_to_index(state)
            var action_idx = agent.select_action(state_idx)
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var action = env.action_from_index(action_idx)
                var result = env.step(action)
                var next_state = result[0]
                var reward = result[1]
                var done = result[2]

                var next_state_idx = env.state_to_index(next_state)
                var next_action_idx = agent.select_action(next_state_idx)

                # True SARSA update with next action
                agent.update_sarsa(state_idx, action_idx, reward, next_state_idx, next_action_idx, done)

                total_reward += reward
                steps += 1
                state_idx = next_state_idx
                action_idx = next_action_idx

                if done:
                    break

            agent.decay_epsilon()
            metrics.log_episode(episode, total_reward, steps, agent.get_epsilon())

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return (agent^, metrics^)
