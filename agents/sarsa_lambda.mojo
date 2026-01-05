from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent


struct SARSALambdaAgent(TabularAgent):
    """SARSA(λ) agent with eligibility traces.

    Eligibility traces unify TD and Monte Carlo methods by maintaining
    a trace of recently visited state-action pairs. When a TD error occurs,
    all eligible states are updated proportionally to their trace.

    Two types of traces:
    - Accumulating traces: e(s,a) += 1 when visited
    - Replacing traces: e(s,a) = 1 when visited (used here)

    Update rules:
    δ = R + γ * Q(S', A') - Q(S, A)           # TD error
    e(S, A) = 1                                # Set trace for current pair
    For all s, a:
        Q(s, a) += α * δ * e(s, a)            # Update proportional to trace
        e(s, a) *= γ * λ                       # Decay all traces

    λ controls the trace decay:
    - λ=0: TD(0), only immediate backup
    - λ=1: Monte Carlo (if episodic with γ=1)
    - 0<λ<1: Intermediate, elegant unification

    Benefits:
    - More efficient credit assignment than TD(0)
    - Online updates (no need to wait for episode end like n-step)
    - Smooth interpolation between TD and MC
    """

    var q_table: QTable
    var eligibility: QTable  # Eligibility traces e(s,a)
    var learning_rate: Float64
    var discount_factor: Float64
    var lambda_: Float64  # Trace decay parameter
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int
    var num_states: Int

    fn __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        lambda_: Float64 = 0.9,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
    ):
        self.q_table = QTable(num_states, num_actions)
        self.eligibility = QTable(num_states, num_actions, initial_value=0.0)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_states = num_states

    fn _reset_traces(mut self):
        """Reset all eligibility traces to zero."""
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.eligibility.set(s, a, 0.0)

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
        """Update Q-values using SARSA(λ) with eligibility traces.

        This uses replacing traces: e(s,a) = 1 for current state-action.
        """
        # Select next action (on-policy)
        var next_action = self.select_action(next_state_idx)

        # Compute TD error
        var current_q = self.q_table.get(state_idx, action)
        var next_q: Float64 = 0.0
        if not done:
            next_q = self.q_table.get(next_state_idx, next_action)
        var td_error = reward + self.discount_factor * next_q - current_q

        # Update eligibility trace for current state-action (replacing traces)
        self.eligibility.set(state_idx, action, 1.0)

        # Update all Q-values and decay traces
        for s in range(self.num_states):
            for a in range(self.num_actions):
                var trace = self.eligibility.get(s, a)
                if trace > 0.0001:  # Only update non-zero traces for efficiency
                    # Update Q-value
                    var q = self.q_table.get(s, a)
                    var new_q = q + self.learning_rate * td_error * trace
                    self.q_table.set(s, a, new_q)

                    # Decay trace
                    var new_trace = self.discount_factor * self.lambda_ * trace
                    self.eligibility.set(s, a, new_trace)

        # Reset traces at end of episode
        if done:
            self._reset_traces()

    fn decay_epsilon(mut self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current epsilon value."""
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
        """Return the greedy action for a state."""
        return self.q_table.get_best_action(state_idx)
