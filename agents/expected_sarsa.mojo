from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent, DiscreteEnv, train_tabular_with_metrics, TrainingMetrics


struct ExpectedSARSAAgent(TabularAgent, Copyable, Movable, ImplicitlyCopyable):
    """Tabular Expected SARSA agent with epsilon-greedy exploration.

    Expected SARSA uses the expected value over all possible next actions,
    weighted by the policy probabilities:

    Q(s,a) += alpha * (r + gamma * E[Q(s',a')] - Q(s,a))

    where E[Q(s',a')] = sum over a' of pi(a'|s') * Q(s',a')

    For epsilon-greedy:
    E[Q(s',a')] = (1-epsilon) * max_a Q(s',a) + (epsilon/|A|) * sum_a Q(s',a)

    Benefits over SARSA:
    - Lower variance (doesn't depend on which action was actually sampled)
    - Often converges faster and to better policies
    - Interpolates between SARSA (epsilon=1) and Q-learning (epsilon=0)
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
        """Select action using epsilon-greedy policy."""
        var rand = random_float64()
        if rand < self.epsilon:
            # random_si64 is inclusive on both ends, so use num_actions - 1
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.q_table.get_best_action(state_idx)

    fn _get_expected_value(self, state_idx: Int) -> Float64:
        """Compute expected Q-value under epsilon-greedy policy.

        E[Q(s,a)] = (1-epsilon) * max_a Q(s,a) + (epsilon/|A|) * sum_a Q(s,a)
        """
        var max_q = self.q_table.get_max_value(state_idx)
        var sum_q: Float64 = 0.0

        for a in range(self.num_actions):
            sum_q += self.q_table.get(state_idx, a)

        var greedy_prob = 1.0 - self.epsilon
        var explore_prob = self.epsilon / Float64(self.num_actions)

        # Expected value = greedy_prob * max_q + explore_prob * sum_q
        # But we need to be careful: the greedy action is counted in both terms
        # Correct formula:
        # E[Q] = sum over a of pi(a|s) * Q(s,a)
        # pi(a|s) = epsilon/|A| for non-greedy, (1-epsilon) + epsilon/|A| for greedy
        #
        # Simplified: E[Q] = (1-epsilon)*max_q + (epsilon/|A|)*sum_q
        return greedy_prob * max_q + explore_prob * sum_q

    fn update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Update Q-value using Expected SARSA.

        Q(s,a) += alpha * (r + gamma * E[Q(s',a')] - Q(s,a))
        """
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64

        if done:
            target = reward
        else:
            var expected_q = self._get_expected_value(next_state_idx)
            target = reward + self.discount_factor * expected_q

        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

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
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> Tuple[ExpectedSARSAAgent, TrainingMetrics]:
        """Train an Expected SARSA agent on the given environment.

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
        var agent = ExpectedSARSAAgent(
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
            algorithm_name="Expected SARSA",
            environment_name=environment_name,
        )
        return (agent^, metrics^)
