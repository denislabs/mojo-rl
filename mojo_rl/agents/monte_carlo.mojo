from std.random import random_si64, random_float64
from .qlearning import QTable
from mojo_rl.core import (
    train_tabular,
    evaluate_tabular,
    TabularAgent,
    DiscreteEnv,
    TrainingMetrics,
    RenderableEnv,
)


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

    def __init__(out self, *, copy: Self):
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

    def __init__(out self, *, deinit take: Self):
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

    def __init__(
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

    def select_action(self, state_idx: Int) -> Int:
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
        """Store transition; update Q-values when episode ends."""
        self.episode_states.append(state_idx)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        if done:
            self._update_from_episode()

    def _update_from_episode(mut self):
        """First-visit MC update from completed episode."""
        # len(), NOT byte_length() — byte_length on a List[Int] is 8× the
        # element count, which made this loop read 8× past episode_rewards.
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

    def decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_epsilon(self) -> Float64:
        return self.epsilon

    def get_best_action(self, state_idx: Int) -> Int:
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
            algorithm_name="Monte Carlo",
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
