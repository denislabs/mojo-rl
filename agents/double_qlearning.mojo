from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent, DiscreteEnv, train_tabular_with_metrics, TrainingMetrics


struct DoubleQLearningAgent(TabularAgent, Copyable, Movable, ImplicitlyCopyable):
    """Double Q-Learning agent.

    Uses two Q-tables to reduce overestimation bias.
    """

    var q_table1: QTable
    var q_table2: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int

    fn __copyinit__(out self, existing: Self):
        self.q_table1 = existing.q_table1
        self.q_table2 = existing.q_table2
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions

    fn __moveinit__(out self, deinit existing: Self):
        self.q_table1 = existing.q_table1^
        self.q_table2 = existing.q_table2^
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
        self.q_table1 = QTable(num_states, num_actions)
        self.q_table2 = QTable(num_states, num_actions)
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
            var best_action = 0
            var best_value = self.q_table1.get(state_idx, 0) + self.q_table2.get(state_idx, 0)
            for i in range(1, self.num_actions):
                var value = self.q_table1.get(state_idx, i) + self.q_table2.get(state_idx, i)
                if value > best_value:
                    best_value = value
                    best_action = i
            return best_action

    fn update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Randomly update Q1 or Q2."""
        if random_float64() < 0.5:
            var current_q = self.q_table1.get(state_idx, action)
            var target: Float64
            if done:
                target = reward
            else:
                var best_action = self.q_table1.get_best_action(next_state_idx)
                target = reward + self.discount_factor * self.q_table2.get(next_state_idx, best_action)
            var new_q = current_q + self.learning_rate * (target - current_q)
            self.q_table1.set(state_idx, action, new_q)
        else:
            var current_q = self.q_table2.get(state_idx, action)
            var target: Float64
            if done:
                target = reward
            else:
                var best_action = self.q_table2.get_best_action(next_state_idx)
                target = reward + self.discount_factor * self.q_table1.get(next_state_idx, best_action)
            var new_q = current_q + self.learning_rate * (target - current_q)
            self.q_table2.set(state_idx, action, new_q)

    fn decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
        var best_action = 0
        var best_value = self.q_table1.get(state_idx, 0) + self.q_table2.get(state_idx, 0)
        for i in range(1, self.num_actions):
            var value = self.q_table1.get(state_idx, i) + self.q_table2.get(state_idx, i)
            if value > best_value:
                best_value = value
                best_action = i
        return best_action

    # ========================================================================
    # Static training method
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
    ) -> Tuple[DoubleQLearningAgent, TrainingMetrics]:
        """Train a Double Q-Learning agent on the given environment.

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
        var agent = DoubleQLearningAgent(
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
            algorithm_name="Double Q-Learning",
            environment_name=environment_name,
        )
        return (agent^, metrics^)
