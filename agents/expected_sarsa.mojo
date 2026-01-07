from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent, DiscreteEnv, TrainingMetrics


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
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.q_table.get_best_action(state_idx)

    fn _get_expected_value(self, state_idx: Int) -> Float64:
        """Compute expected Q-value under epsilon-greedy policy."""
        var max_q = self.q_table.get_max_value(state_idx)
        var sum_q: Float64 = 0.0

        for a in range(self.num_actions):
            sum_q += self.q_table.get(state_idx, a)

        var greedy_prob = 1.0 - self.epsilon
        var explore_prob = self.epsilon / Float64(self.num_actions)

        return greedy_prob * max_q + explore_prob * sum_q

    fn update(
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

    fn decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
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
            algorithm_name="Expected SARSA",
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
