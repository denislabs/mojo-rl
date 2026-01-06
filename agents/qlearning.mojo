from core import TabularAgent, DiscreteEnv, TrainingMetrics
from random import random_si64, random_float64


@fieldwise_init
struct QTable(Copyable, ImplicitlyCopyable, Movable):
    """Q-table for tabular Q-learning."""

    var data: List[List[Float64]]
    var num_states: Int
    var num_actions: Int

    fn copy(self) -> Self:
        """Explicit copy method."""
        var new_data = List[List[Float64]]()
        for i in range(self.num_states):
            var row = List[Float64]()
            for j in range(self.num_actions):
                row.append(self.data[i][j])
            new_data.append(row^)
        return Self(
            data=new_data^,
            num_states=self.num_states,
            num_actions=self.num_actions,
        )

    fn __copyinit__(out self, existing: Self):
        self.num_states = existing.num_states
        self.num_actions = existing.num_actions
        self.data = List[List[Float64]]()
        for i in range(existing.num_states):
            var row = List[Float64]()
            for j in range(existing.num_actions):
                row.append(existing.data[i][j])
            self.data.append(row^)

    fn __moveinit__(out self, deinit existing: Self):
        self.data = existing.data^
        self.num_states = existing.num_states
        self.num_actions = existing.num_actions

    fn __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        initial_value: Float64 = 0.0,
    ):
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


struct QLearningAgent(Copyable, ImplicitlyCopyable, Movable, TabularAgent):
    """Tabular Q-Learning agent with epsilon-greedy exploration."""

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
            target = reward + self.discount_factor * self.q_table.get_max_value(
                next_state_idx
            )
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state_idx, action, new_q)

    fn decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
        return self.q_table.get_best_action(state_idx)

    fn train[E: DiscreteEnv](
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
            algorithm_name="Q-Learning",
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

    fn evaluate[E: DiscreteEnv](
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

            for _ in range(1000):  # Max steps for evaluation
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
