from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent, DiscreteEnv, TrainingMetrics


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

    fn __copyinit__(out self, existing: Self):
        self.q_table = existing.q_table
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions
        self.n = existing.n
        self.states = existing.states.copy()
        self.actions = existing.actions.copy()
        self.rewards = existing.rewards.copy()
        self.t = existing.t
        self.T = existing.T

    fn __moveinit__(out self, deinit existing: Self):
        self.q_table = existing.q_table^
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions
        self.n = existing.n
        self.states = existing.states^
        self.actions = existing.actions^
        self.rewards = existing.rewards^
        self.t = existing.t
        self.T = existing.T

    fn __init__(
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

    fn _reset_episode(mut self):
        """Reset episode buffers."""
        self.states = List[Int]()
        self.actions = List[Int]()
        self.rewards = List[Float64]()
        self.t = 0
        self.T = 1000000

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

    fn _update_at_tau(mut self, tau: Int):
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

    fn _power(self, base: Float64, exp: Int) -> Float64:
        """Compute base^exp for non-negative integer exponents."""
        if exp == 0:
            return 1.0
        var result: Float64 = 1.0
        for _ in range(exp):
            result *= base
        return result

    fn decay_epsilon(mut self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current epsilon value."""
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
        """Return the greedy action for a state."""
        return self.q_table.get_best_action(state_idx)

    fn train[
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
        """Train the agent on the given environment.

        Args:
            env: The discrete environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="N-Step SARSA",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            var state = env.reset()
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var state_idx = env.state_to_index(state)
                var action_idx = self.select_action(state_idx)
                var action = env.action_from_index(action_idx)

                var result = env.step(action)
                var next_state = result[0]
                var reward = result[1]
                var done = result[2]

                var next_state_idx = env.state_to_index(next_state)
                self.update(state_idx, action_idx, reward, next_state_idx, done)

                total_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            self.decay_epsilon()
            metrics.log_episode(episode, total_reward, steps, self.epsilon)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[
        E: DiscreteEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The discrete environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            render: Whether to render the environment.

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var state = env.reset()
            var episode_reward: Float64 = 0.0

            for _ in range(1000):
                if render:
                    env.render()

                var state_idx = env.state_to_index(state)
                var action_idx = self.get_best_action(state_idx)
                var action = env.action_from_index(action_idx)

                var result = env.step(action)
                var next_state = result[0]
                var reward = result[1]
                var done = result[2]

                episode_reward += reward
                state = next_state

                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)
