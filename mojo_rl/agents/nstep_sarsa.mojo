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


struct NStepSARSAAgent(Copyable, ImplicitlyCopyable, Movable, TabularAgent):
    """N-step SARSA agent with epsilon-greedy exploration.

    N-step methods bridge the gap between TD(0) and Monte Carlo:
    - n=1: TD(0) / regular SARSA
    - n=infinity: Monte Carlo

    The n-step return is:
    G_t:t+n = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})

    Update rule:
    Q(S_t, A_t) += α * (G_t:t+n - Q(S_t, A_t))

    Benefits:
    - Faster credit assignment than TD(0)
    - Lower variance than Monte Carlo
    - Tunable bias-variance tradeoff via n
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int
    var n: Int  # Number of steps

    # Episode buffers for n-step updates
    var states: List[Int]
    var actions: List[Int]
    var rewards: List[Float64]
    var t: Int  # Current timestep in episode
    var T: Int  # Terminal timestep (infinity until episode ends)

    def __init__(out self, *, copy: Self):
        self.q_table = copy.q_table
        self.learning_rate = copy.learning_rate
        self.discount_factor = copy.discount_factor
        self.epsilon = copy.epsilon
        self.epsilon_decay = copy.epsilon_decay
        self.epsilon_min = copy.epsilon_min
        self.num_actions = copy.num_actions
        self.n = copy.n
        self.states = copy.states.copy()
        self.actions = copy.actions.copy()
        self.rewards = copy.rewards.copy()
        self.t = copy.t
        self.T = copy.T

    def __init__(out self, *, deinit take: Self):
        self.q_table = take.q_table^
        self.learning_rate = take.learning_rate
        self.discount_factor = take.discount_factor
        self.epsilon = take.epsilon
        self.epsilon_decay = take.epsilon_decay
        self.epsilon_min = take.epsilon_min
        self.num_actions = take.num_actions
        self.n = take.n
        self.states = take.states^
        self.actions = take.actions^
        self.rewards = take.rewards^
        self.t = take.t
        self.T = take.T

    def __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        n: Int = 3,
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
        self.n = n

        # Initialize buffers
        self.states = List[Int]()
        self.actions = List[Int]()
        self.rewards = List[Float64]()
        self.t = 0
        self.T = 1000000  # Large number representing infinity

    def _reset_episode(mut self):
        """Reset episode buffers."""
        self.states = List[Int]()
        self.actions = List[Int]()
        self.rewards = List[Float64]()
        self.t = 0
        self.T = 1000000

    def select_action(self, state_idx: Int) -> Int:
        """Select action using epsilon-greedy policy."""
        var rand = random_float64()
        if rand < self.epsilon:
            # random_si64 is inclusive on both ends, so use num_actions - 1
            return Int(random_si64(0, Int64(self.num_actions - 1)))
        else:
            return self.q_table.get_best_action(state_idx)

    def update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Store transition and perform n-step update when possible.

        This implements the online n-step SARSA algorithm.
        Updates are delayed by n steps to accumulate returns.
        """
        # Store transition
        if self.t == 0:
            # First step: store initial state and action
            self.states.append(state_idx)
            self.actions.append(action)

        # Store reward and next state/action
        self.rewards.append(reward)

        if done:
            self.T = self.t + 1
        else:
            # Store next state and select next action
            self.states.append(next_state_idx)
            var next_action = self.select_action(next_state_idx)
            self.actions.append(next_action)

        # Compute tau: the time whose estimate is being updated
        var tau = self.t - self.n + 1

        if tau >= 0:
            self._update_at_tau(tau)

        self.t += 1

        # At end of episode, update remaining states
        if done:
            for remaining_tau in range(max(0, tau + 1), self.T):
                self._update_at_tau(remaining_tau)
            self._reset_episode()

    def _update_at_tau(mut self, tau: Int):
        """Perform the n-step update for time tau."""
        # Compute n-step return G
        var G: Float64 = 0.0

        # Sum discounted rewards from tau+1 to min(tau+n, T)
        var end_idx = min(tau + self.n, self.T)
        for i in range(tau + 1, end_idx + 1):
            var reward_idx = i - 1  # rewards[i-1] is R_i
            if reward_idx < len(self.rewards):
                var power = i - tau - 1
                var discount = self._power(self.discount_factor, power)
                G += discount * self.rewards[reward_idx]

        # Add bootstrap value if not at terminal state
        if tau + self.n < self.T:
            var bootstrap_idx = tau + self.n
            if bootstrap_idx < len(self.states) and bootstrap_idx < len(
                self.actions
            ):
                var bootstrap_state = self.states[bootstrap_idx]
                var bootstrap_action = self.actions[bootstrap_idx]
                var power = self.n
                var discount = self._power(self.discount_factor, power)
                G += discount * self.q_table.get(
                    bootstrap_state, bootstrap_action
                )

        # Update Q(S_tau, A_tau)
        var state = self.states[tau]
        var action = self.actions[tau]
        var current_q = self.q_table.get(state, action)
        var new_q = current_q + self.learning_rate * (G - current_q)
        self.q_table.set(state, action, new_q)

    def _power(self, base: Float64, exp: Int) -> Float64:
        """Compute base^exp for non-negative integer exponents."""
        if exp == 0:
            return 1.0
        var result: Float64 = 1.0
        for _ in range(exp):
            result *= base
        return result

    def decay_epsilon(mut self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_epsilon(self) -> Float64:
        """Return current epsilon value."""
        return self.epsilon

    def get_best_action(self, state_idx: Int) -> Int:
        """Return the greedy action for a state."""
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
            algorithm_name="N-Step SARSA",
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
