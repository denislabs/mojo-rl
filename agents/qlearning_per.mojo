"""Q-Learning agent with Prioritized Experience Replay (PER).

This agent combines Q-Learning with Prioritized Experience Replay,
which samples transitions proportionally to their TD error magnitude.

Key differences from uniform replay:
- Transitions with high TD error (surprising/important) are sampled more often
- Importance sampling weights correct for the bias introduced by non-uniform sampling
- Beta annealing gradually increases IS correction over training

Reference: Schaul et al., "Prioritized Experience Replay" (2015)
"""

from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent, DiscreteEnv, TrainingMetrics
from core.replay_buffer import PrioritizedReplayBuffer, PrioritizedTransition


struct QLearningPERAgent(Copyable, Movable, TabularAgent):
    """Q-Learning agent with Prioritized Experience Replay.

    After each real transition:
    1. Store transition in priority buffer (with max priority)
    2. Sample a mini-batch weighted by priority
    3. Perform Q-learning updates with importance sampling weights
    4. Update priorities based on TD errors

    Key hyperparameters:
    - alpha: Priority exponent (0 = uniform, 1 = full prioritization).
    - beta: IS correction exponent (annealed from beta_start to 1.0).
    - buffer_size: Maximum transitions to store.
    - batch_size: Transitions sampled per update.

    Usage:
        var agent = QLearningPERAgent(
            num_states=env.num_states(),
            num_actions=env.num_actions(),
            alpha=0.6,  # Priority exponent
            beta_start=0.4,  # Initial IS weight
        )
        var metrics = agent.train(env, num_episodes=500)
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int
    var num_states: Int

    var buffer: PrioritizedReplayBuffer
    var batch_size: Int
    var min_buffer_size: Int
    var beta_start: Float64
    var total_steps: Int  # Track steps for beta annealing

    fn __copyinit__(out self, existing: Self):
        self.q_table = existing.q_table
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions
        self.num_states = existing.num_states
        # Note: PrioritizedReplayBuffer doesn't have copy, create new one
        self.buffer = PrioritizedReplayBuffer(
            existing.buffer.capacity,
            existing.buffer.alpha,
            existing.buffer.beta,
            existing.buffer.epsilon,
        )
        self.batch_size = existing.batch_size
        self.min_buffer_size = existing.min_buffer_size
        self.beta_start = existing.beta_start
        self.total_steps = existing.total_steps

    fn __moveinit__(out self, deinit existing: Self):
        self.q_table = existing.q_table^
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions
        self.num_states = existing.num_states
        self.buffer = existing.buffer^
        self.batch_size = existing.batch_size
        self.min_buffer_size = existing.min_buffer_size
        self.beta_start = existing.beta_start
        self.total_steps = existing.total_steps

    fn __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        buffer_size: Int = 10000,
        batch_size: Int = 32,
        min_buffer_size: Int = 100,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        alpha: Float64 = 0.6,
        beta_start: Float64 = 0.4,
    ):
        """Initialize Q-Learning agent with PER.

        Args:
            num_states: Number of discrete states.
            num_actions: Number of discrete actions.
            buffer_size: Maximum transitions in buffer.
            batch_size: Batch size for updates.
            min_buffer_size: Minimum buffer fill before learning.
            learning_rate: Q-learning step size.
            discount_factor: Discount for future rewards.
            epsilon: Initial exploration rate.
            epsilon_decay: Epsilon decay per episode.
            epsilon_min: Minimum epsilon value.
            alpha: Priority exponent (0=uniform, 1=full priority).
            beta_start: Initial IS correction (annealed to 1.0).
        """
        self.q_table = QTable(num_states, num_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_states = num_states

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=alpha, beta=beta_start
        )
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.beta_start = beta_start
        self.total_steps = 0

    fn select_action(self, state_idx: Int) -> Int:
        """Select action using epsilon-greedy policy."""
        var rand = random_float64()
        if rand < self.epsilon:
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.q_table.get_best_action(state_idx)

    fn _compute_td_error(
        self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
    ) -> Float64:
        """Compute TD error for a transition."""
        var current_q = self.q_table.get(state, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(
                next_state
            )
        return target - current_q

    fn _q_update_weighted(
        mut self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
        weight: Float64,
    ) -> Float64:
        """Perform weighted Q-learning update and return TD error.

        The IS weight scales the learning rate to correct for
        non-uniform sampling bias.

        Returns:
            The TD error (for priority update).
        """
        var current_q = self.q_table.get(state, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(
                next_state
            )
        var td_error = target - current_q
        # Weight the update by IS weight
        var new_q = current_q + weight * self.learning_rate * td_error
        self.q_table.set(state, action, new_q)
        return td_error

    fn update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Store transition and learn from prioritized replay batch."""
        # Store transition in buffer (gets max priority automatically)
        self.buffer.push(state_idx, action, reward, next_state_idx, done)
        self.total_steps += 1

        # Only start learning after buffer has enough samples
        if self.buffer.len() < self.min_buffer_size:
            return

        # Sample batch with importance weights
        var result = self.buffer.sample(self.batch_size)
        var indices = result[0].copy()
        var batch = result[1].copy()

        # Process batch and collect TD errors for priority updates
        var td_errors = List[Float64]()
        for i in range(len(batch)):
            var t = batch[i]
            var td_error = self._q_update_weighted(
                t.state, t.action, t.reward, t.next_state, t.done, t.weight
            )
            td_errors.append(td_error)

        # Update priorities based on TD errors
        for i in range(len(indices)):
            self.buffer.update_priority(indices[i], td_errors[i])

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
        """Train the agent with prioritized experience replay.

        Beta is annealed from beta_start to 1.0 over training to
        gradually increase importance sampling correction.

        Args:
            env: The discrete environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes.
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Q-Learning + PER",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            var state = env.reset()
            var total_reward: Float64 = 0.0
            var steps = 0

            # Anneal beta towards 1.0 based on progress
            var progress = Float64(episode) / Float64(num_episodes)
            self.buffer.anneal_beta(progress, self.beta_start)

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
