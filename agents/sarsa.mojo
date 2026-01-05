from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent


struct SARSAAgent(TabularAgent):
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
        """True SARSA: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))"""
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
