from mojo_rl.core import (
    train_tabular,
    evaluate_tabular,
    TabularAgent,
    DiscreteEnv,
    RenderableEnv,
    TrainingMetrics,
)
from std.random import random_si64, random_float64


struct QTable(Copyable, ImplicitlyCopyable, Movable):
    """Q-table for tabular Q-learning.

    Uses flat array storage for better cache locality and performance.
    Layout: data[state * num_actions + action]
    """

    var data: List[Float64]
    var num_states: Int
    var num_actions: Int

    def __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        initial_value: Float64 = 0.0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        var total_size = num_states * num_actions
        self.data = List[Float64](capacity=total_size)
        for _ in range(total_size):
            self.data.append(initial_value)

    def __init__(out self, *, copy: Self):
        self.num_states = copy.num_states
        self.num_actions = copy.num_actions
        self.data = copy.data.copy()

    def __init__(out self, *, deinit move: Self):
        self.data = move.data^
        self.num_states = move.num_states
        self.num_actions = move.num_actions

    @always_inline
    def _index(self, state: Int, action: Int) -> Int:
        """Compute flat index from state and action."""
        return state * self.num_actions + action

    @always_inline
    def get(self, state: Int, action: Int) -> Float64:
        return self.data[self._index(state, action)]

    @always_inline
    def set(mut self, state: Int, action: Int, value: Float64):
        self.data[self._index(state, action)] = value

    @always_inline
    def get_max_value(self, state: Int) -> Float64:
        var base_idx = state * self.num_actions
        var max_val = self.data[base_idx]
        for i in range(1, self.num_actions):
            var val = self.data[base_idx + i]
            if val > max_val:
                max_val = val
        return max_val

    @always_inline
    def get_best_action(self, state: Int) -> Int:
        var base_idx = state * self.num_actions
        var best_action = 0
        var best_value = self.data[base_idx]
        for i in range(1, self.num_actions):
            var val = self.data[base_idx + i]
            if val > best_value:
                best_value = val
                best_action = i
        return best_action


struct QLearningAgent(Copyable, ImplicitlyCopyable, Movable, TabularAgent):
    """Tabular Q-Learning agent with epsilon-greedy exploration."""

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int

    def __init__(out self, *, copy: Self):
        self.q_table = copy.q_table.copy()
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

    @always_inline
    def select_action(self, state_idx: Int) -> Int:
        """Select action using epsilon-greedy policy."""
        var rand = random_float64()
        if rand < self.epsilon:
            # random_si64 is inclusive on both ends, so use num_actions - 1
            return Int(random_si64(0, Int64(self.num_actions - 1)))
        else:
            return self.q_table.get_best_action(state_idx)

    @always_inline
    def update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))."""
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(
                next_state_idx
            )
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

    @always_inline
    def decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @always_inline
    def get_epsilon(self) -> Float64:
        return self.epsilon

    @always_inline
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
            algorithm_name="Q-Learning",
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
