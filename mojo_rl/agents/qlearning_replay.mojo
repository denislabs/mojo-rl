from std.random import random_si64, random_float64
from .qlearning import QTable
from mojo_rl.core import (
    train_tabular,
    evaluate_tabular,
    TabularAgent,
    ReplayBuffer,
    DiscreteEnv,
    TrainingMetrics,
    RenderableEnv,
)


struct QLearningReplayAgent(
    Copyable, ImplicitlyCopyable, Movable, TabularAgent
):
    """Q-Learning agent with Experience Replay.

    Combines Q-Learning with experience replay buffer for more stable
    and sample-efficient learning.

    After each real transition:
    1. Store transition in replay buffer
    2. Sample a mini-batch from buffer
    3. Perform Q-learning updates on all batch samples

    Benefits:
    - Breaks correlation between consecutive samples
    - Reuses past experience (higher sample efficiency)
    - More stable learning
    - Precursor to DQN-style algorithms

    Args:
        q_table: QTable for storing Q-values.
        learning_rate: Learning rate for Q-learning updates.
        discount_factor: Discount factor for future rewards.
        epsilon: Initial exploration rate.
        epsilon_decay: Epsilon decay per episode.
        epsilon_min: Minimum epsilon value.
        num_actions: Number of actions.
        num_states: Number of states.
        buffer: Replay buffer for storing transitions.
        batch_size: Number of transitions to sample per update.
        min_buffer_size: Minimum buffer size before learning starts.
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int
    var num_states: Int

    var buffer: ReplayBuffer
    var batch_size: Int
    var min_buffer_size: Int

    def __init__(out self, *, copy: Self):
        self.q_table = copy.q_table
        self.learning_rate = copy.learning_rate
        self.discount_factor = copy.discount_factor
        self.epsilon = copy.epsilon
        self.epsilon_decay = copy.epsilon_decay
        self.epsilon_min = copy.epsilon_min
        self.num_actions = copy.num_actions
        self.num_states = copy.num_states
        self.buffer = copy.buffer.copy()
        self.batch_size = copy.batch_size
        self.min_buffer_size = copy.min_buffer_size

    def __init__(out self, *, deinit move: Self):
        self.q_table = move.q_table^
        self.learning_rate = move.learning_rate
        self.discount_factor = move.discount_factor
        self.epsilon = move.epsilon
        self.epsilon_decay = move.epsilon_decay
        self.epsilon_min = move.epsilon_min
        self.num_actions = move.num_actions
        self.num_states = move.num_states
        self.buffer = move.buffer^
        self.batch_size = move.batch_size
        self.min_buffer_size = move.min_buffer_size

    def __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        buffer_size: Int = 1000,
        batch_size: Int = 32,
        min_buffer_size: Int = 100,
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
        self.num_states = num_states

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size

    def select_action(self, state_idx: Int) -> Int:
        """Select action using epsilon-greedy policy."""
        var rand = random_float64()
        if rand < self.epsilon:
            # random_si64 is inclusive on both ends, so use num_actions - 1
            return Int(random_si64(0, Int64(self.num_actions - 1)))
        else:
            return self.q_table.get_best_action(state_idx)

    def _q_update(
        mut self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
    ):
        """Single Q-learning update."""
        var current_q = self.q_table.get(state, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(
                next_state
            )
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state, action, new_q)

    def update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Store transition and learn from replay batch."""
        # Store transition in buffer
        self.buffer.push(state_idx, action, reward, next_state_idx, done)

        # Only start learning after buffer has enough samples
        if self.buffer.len() < self.min_buffer_size:
            return

        # Sample batch and update
        var batch = self.buffer.sample(self.batch_size)
        for i in range(len(batch)):
            var t = batch[i]
            self._q_update(t.state, t.action, t.reward, t.next_state, t.done)

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
            algorithm_name="Q-Learning + Replay",
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
