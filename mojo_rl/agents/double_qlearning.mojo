from std.random import random_si64, random_float64
from .qlearning import QTable
from mojo_rl.core import (
    TabularAgent,
    DiscreteEnv,
    TrainingMetrics,
    RenderableEnv,
)


struct DoubleQLearningAgent(
    Copyable, ImplicitlyCopyable, Movable, TabularAgent
):
    """Double Q-Learning agent.

    Uses two Q-tables to reduce overestimation bias.
    """

    var q_table1: QTable
    var q_table2: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int

    def __init__(out self, *, copy: Self):
        self.q_table1 = copy.q_table1
        self.q_table2 = copy.q_table2
        self.learning_rate = copy.learning_rate
        self.discount_factor = copy.discount_factor
        self.epsilon = copy.epsilon
        self.epsilon_decay = copy.epsilon_decay
        self.epsilon_min = copy.epsilon_min
        self.num_actions = copy.num_actions

    def __init__(out self, *, deinit take: Self):
        self.q_table1 = take.q_table1^
        self.q_table2 = take.q_table2^
        self.learning_rate = take.learning_rate
        self.discount_factor = take.discount_factor
        self.epsilon = take.epsilon
        self.epsilon_decay = take.epsilon_decay
        self.epsilon_min = take.epsilon_min
        self.num_actions = take.num_actions

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
        self.q_table1 = QTable(num_states, num_actions)
        self.q_table2 = QTable(num_states, num_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions

    def select_action(self, state_idx: Int) -> Int:
        var rand = random_float64()
        if rand < self.epsilon:
            return Int(random_si64(0, Int64(self.num_actions - 1)))
        else:
            var best_action = 0
            var best_value = self.q_table1.get(
                state_idx, 0
            ) + self.q_table2.get(state_idx, 0)
            for i in range(1, self.num_actions):
                var value = self.q_table1.get(state_idx, i) + self.q_table2.get(
                    state_idx, i
                )
                if value > best_value:
                    best_value = value
                    best_action = i
            return best_action

    def update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Randomly update Q1 or Q2."""
        if random_float64() < 0.5:
            var current_q = self.q_table1.get(state_idx, action)
            var target: Float64
            if done:
                target = reward
            else:
                var best_action = self.q_table1.get_best_action(next_state_idx)
                target = reward + self.discount_factor * self.q_table2.get(
                    next_state_idx, best_action
                )
            var new_q = current_q + self.learning_rate * (target - current_q)
            self.q_table1.set(state_idx, action, new_q)
        else:
            var current_q = self.q_table2.get(state_idx, action)
            var target: Float64
            if done:
                target = reward
            else:
                var best_action = self.q_table2.get_best_action(next_state_idx)
                target = reward + self.discount_factor * self.q_table1.get(
                    next_state_idx, best_action
                )
            var new_q = current_q + self.learning_rate * (target - current_q)
            self.q_table2.set(state_idx, action, new_q)

    def decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_epsilon(self) -> Float64:
        return self.epsilon

    def get_best_action(self, state_idx: Int) -> Int:
        var best_action = 0
        var best_value = self.q_table1.get(state_idx, 0) + self.q_table2.get(
            state_idx, 0
        )
        for i in range(1, self.num_actions):
            var value = self.q_table1.get(state_idx, i) + self.q_table2.get(
                state_idx, i
            )
            if value > best_value:
                best_value = value
                best_action = i
        return best_action

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
            algorithm_name="Double Q-Learning",
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

                var result = env.step(action^)
                var next_state = result[0]
                var reward = result[1]
                var done = result[2]

                var next_state_idx = env.state_to_index(next_state)
                self.update(
                    state_idx, action_idx, Float64(reward), next_state_idx, done
                )

                total_reward += Float64(reward)
                steps += 1
                state = next_state

                if done:
                    break

            self.decay_epsilon()
            metrics.log_episode(episode, total_reward, steps, self.epsilon)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    def evaluate[
        E: DiscreteEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The discrete environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            render: Whether to render the environment (default: False).
            frame_delay_ms: Delay between frames in milliseconds (default: 16).

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for _ in range(num_episodes):
            if quit_requested:
                break
            var state = env.reset()
            var episode_reward: Float64 = 0.0

            for _ in range(1000):
                var state_idx = env.state_to_index(state)
                var action_idx = self.get_best_action(state_idx)
                var action = env.action_from_index(action_idx)

                var result = env.step(action^)
                var next_state = result[0]
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                episode_reward += Float64(reward)
                state = next_state

                if done:
                    break

            total_reward += episode_reward

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)
