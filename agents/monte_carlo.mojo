from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent, DiscreteEnv, TrainingMetrics, RenderableEnv


struct MonteCarloAgent(Copyable, ImplicitlyCopyable, Movable, TabularAgent):
    """First-visit Monte Carlo agent.

    Learns from complete episodes. update() stores transitions;
    Q-values are updated when done=True.
    """

    var q_table: QTable
    var returns_sum: QTable
    var returns_count: QTable
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int
    var episode_states: List[Int]
    var episode_actions: List[Int]
    var episode_rewards: List[Float64]

    fn __init__(out self, *, copy: Self):
        self.q_table = copy.q_table
        self.returns_sum = copy.returns_sum
        self.returns_count = copy.returns_count
        self.discount_factor = copy.discount_factor
        self.epsilon = copy.epsilon
        self.epsilon_decay = copy.epsilon_decay
        self.epsilon_min = copy.epsilon_min
        self.num_actions = copy.num_actions
        self.episode_states = copy.episode_states.copy()
        self.episode_actions = copy.episode_actions.copy()
        self.episode_rewards = copy.episode_rewards.copy()

    fn __init__(out self, *, deinit take: Self):
        self.q_table = take.q_table^
        self.returns_sum = take.returns_sum^
        self.returns_count = take.returns_count^
        self.discount_factor = take.discount_factor
        self.epsilon = take.epsilon
        self.epsilon_decay = take.epsilon_decay
        self.epsilon_min = take.epsilon_min
        self.num_actions = take.num_actions
        self.episode_states = take.episode_states^
        self.episode_actions = take.episode_actions^
        self.episode_rewards = take.episode_rewards^

    fn __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
    ):
        self.q_table = QTable(num_states, num_actions)
        self.returns_sum = QTable(num_states, num_actions)
        self.returns_count = QTable(num_states, num_actions)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.episode_states = List[Int]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

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
        """Store transition; update Q-values when episode ends."""
        self.episode_states.append(state_idx)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        if done:
            self._update_from_episode()

    fn _update_from_episode(mut self):
        """First-visit MC update from completed episode."""
        var num_steps = len(self.episode_states)
        if num_steps == 0:
            return

        var returns = List[Float64]()
        for _ in range(num_steps):
            returns.append(0.0)

        var g: Float64 = 0.0
        for i in range(num_steps - 1, -1, -1):
            g = self.episode_rewards[i] + self.discount_factor * g
            returns[i] = g

        var visited = List[Int]()
        for i in range(num_steps):
            var state_idx = self.episode_states[i]
            var action = self.episode_actions[i]
            var pair_id = state_idx * self.num_actions + action

            var is_first_visit = True
            for j in range(len(visited)):
                if visited[j] == pair_id:
                    is_first_visit = False
                    break

            if is_first_visit:
                visited.append(pair_id)
                var old_sum = self.returns_sum.get(state_idx, action)
                var old_count = self.returns_count.get(state_idx, action)
                self.returns_sum.set(state_idx, action, old_sum + returns[i])
                self.returns_count.set(state_idx, action, old_count + 1.0)
                var new_q = (old_sum + returns[i]) / (old_count + 1.0)
                self.q_table.set(state_idx, action, new_q)

        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

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
            algorithm_name="Monte Carlo",
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

    fn evaluate[
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
