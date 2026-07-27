from std.random import random_si64, random_float64
from .qlearning import QTable
from mojo_rl.core import (
    train_tabular,
    evaluate_tabular,
    TabularAgent,
    DiscreteEnv,
    RenderableEnv,
    TrainingMetrics,
)


struct ExpectedSARSAAgent(Copyable, ImplicitlyCopyable, Movable, TabularAgent):
    """Tabular Expected SARSA agent with epsilon-greedy exploration.

    Expected SARSA uses the expected value over all possible next actions,
    weighted by the policy probabilities:

    Q(s,a) += alpha * (r + gamma * E[Q(s',a')] - Q(s,a))

    where E[Q(s',a')] = sum over a' of pi(a'|s') * Q(s',a')

    For epsilon-greedy:
    E[Q(s',a')] = (1-epsilon) * max_a Q(s',a) + (epsilon/|A|) * sum_a Q(s',a)
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int

    def __init__(out self, *, copy: Self):
        self.q_table = copy.q_table
        self.learning_rate = copy.learning_rate
        self.discount_factor = copy.discount_factor
        self.epsilon = copy.epsilon
        self.epsilon_decay = copy.epsilon_decay
        self.epsilon_min = copy.epsilon_min
        self.num_actions = copy.num_actions

    def __init__(out self, *, deinit move: Self):
        self.q_table = move.q_table^
        self.learning_rate = move.learning_rate
        self.discount_factor = move.discount_factor
        self.epsilon = move.epsilon
        self.epsilon_decay = move.epsilon_decay
        self.epsilon_min = move.epsilon_min
        self.num_actions = move.num_actions

    def __init__(
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

    def select_action(self, state_idx: Int) -> Int:
        """Select action using epsilon-greedy policy."""
        var rand = random_float64()
        if rand < self.epsilon:
            return Int(random_si64(0, Int64(self.num_actions - 1)))
        else:
            return self.q_table.get_best_action(state_idx)

    def _get_expected_value(self, state_idx: Int) -> Float64:
        """Compute expected Q-value under epsilon-greedy policy."""
        var max_q = self.q_table.get_max_value(state_idx)
        var sum_q: Float64 = 0.0

        for a in range(self.num_actions):
            sum_q += self.q_table.get(state_idx, a)

        var greedy_prob = 1.0 - self.epsilon
        var explore_prob = self.epsilon / Float64(self.num_actions)

        return greedy_prob * max_q + explore_prob * sum_q

    def update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Update Q-value using Expected SARSA."""
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64

        if done:
            target = reward
        else:
            var expected_q = self._get_expected_value(next_state_idx)
            target = reward + self.discount_factor * expected_q

        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

    def decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_epsilon(self) -> Float64:
        return self.epsilon

    def get_best_action(self, state_idx: Int) -> Int:
        return self.q_table.get_best_action(state_idx)

    def train[
        E: DiscreteEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 100,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train on `env` — delegates to the shared `train_tabular`
        episode loop (core/tabular_training.mojo)."""
        return train_tabular(
            self,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            verbose=verbose,
            print_every=print_every,
            algorithm_name="Expected SARSA",
            environment_name=environment_name,
        )

    def evaluate[
        E: DiscreteEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Greedy eval — delegates to the shared `evaluate_tabular` loop
        (core/tabular_training.mojo)."""
        return evaluate_tabular(
            self,
            env,
            num_episodes=num_episodes,
            render=render,
            frame_delay_ms=frame_delay_ms,
        )
