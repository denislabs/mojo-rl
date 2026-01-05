from core import TabularAgent
from random import random_si64, random_float64


@fieldwise_init
struct QTable:
    """Q-table for tabular Q-learning."""

    var data: List[List[Float64]]
    var num_states: Int
    var num_actions: Int

    fn __init__(out self, num_states: Int, num_actions: Int, initial_value: Float64 = 0.0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.data = List[List[Float64]]()
        for _ in range(num_states):
            var row = List[Float64]()
            for _ in range(num_actions):
                row.append(initial_value)
            self.data.append(row^)

    fn get(self, state: Int, action: Int) -> Float64:
        return self.data[state][action]

    fn set(mut self, state: Int, action: Int, value: Float64):
        self.data[state][action] = value

    fn get_max_value(self, state: Int) -> Float64:
        var max_val = self.data[state][0]
        for i in range(1, self.num_actions):
            if self.data[state][i] > max_val:
                max_val = self.data[state][i]
        return max_val

    fn get_best_action(self, state: Int) -> Int:
        var best_action = 0
        var best_value = self.data[state][0]
        for i in range(1, self.num_actions):
            if self.data[state][i] > best_value:
                best_value = self.data[state][i]
                best_action = i
        return best_action


struct QLearningAgent(TabularAgent):
    """Tabular Q-Learning agent with epsilon-greedy exploration."""

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int

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
        """Select action using epsilon-greedy policy."""
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
        """Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))"""
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(next_state_idx)
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

    fn decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
        return self.q_table.get_best_action(state_idx)
