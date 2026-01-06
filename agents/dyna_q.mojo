from random import random_si64, random_float64
from .qlearning import QTable
from core import (
    TabularAgent,
    DiscreteEnv,
    train_tabular_with_metrics,
    TrainingMetrics,
)


struct DynaQAgent(Copyable, ImplicitlyCopyable, Movable, TabularAgent):
    """Dyna-Q agent: model-based RL with planning.

    Dyna-Q combines:
    1. Direct RL: Learn from real experience (like Q-learning)
    2. Model learning: Build a model of the environment
    3. Planning: Use the model to simulate experience and update Q-values

    After each real step:
    - Update Q-values from real experience
    - Update the model with observed transition
    - Perform n planning steps using simulated experience

    Benefits:
    - Much faster learning with limited real experience
    - Learns from both real and simulated experience
    - n_planning controls computation vs sample efficiency tradeoff
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int
    var num_states: Int
    var n_planning: Int

    # Model: flat storage for (state, action) -> (next_state, reward)
    var model_next_state: List[Int]
    var model_reward: List[Float64]

    # Track visited (state, action) pairs as a list of pair indices
    var visited_pairs: List[Int]
    var num_visited: Int

    fn __copyinit__(out self, existing: Self):
        self.q_table = existing.q_table
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions
        self.num_states = existing.num_states
        self.n_planning = existing.n_planning
        self.model_next_state = existing.model_next_state.copy()
        self.model_reward = existing.model_reward.copy()
        self.visited_pairs = existing.visited_pairs.copy()
        self.num_visited = existing.num_visited

    fn __moveinit__(out self, deinit existing: Self):
        self.q_table = existing.q_table^
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.epsilon = existing.epsilon
        self.epsilon_decay = existing.epsilon_decay
        self.epsilon_min = existing.epsilon_min
        self.num_actions = existing.num_actions
        self.num_states = existing.num_states
        self.n_planning = existing.n_planning
        self.model_next_state = existing.model_next_state^
        self.model_reward = existing.model_reward^
        self.visited_pairs = existing.visited_pairs^
        self.num_visited = existing.num_visited

    fn __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        n_planning: Int = 5,
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
        self.n_planning = n_planning

        # Initialize flat model - all initialized to -1 (unvisited)
        var total_pairs = num_states * num_actions
        self.model_next_state = List[Int]()
        self.model_reward = List[Float64]()
        for _ in range(total_pairs):
            self.model_next_state.append(-1)  # -1 = unvisited
            self.model_reward.append(0.0)

        # Track visited pairs - pre-allocate to avoid dynamic appends
        self.visited_pairs = List[Int]()
        for _ in range(total_pairs):
            self.visited_pairs.append(-1)
        self.num_visited = 0

    fn _pair_index(self, state: Int, action: Int) -> Int:
        """Convert (state, action) to flat index."""
        return state * self.num_actions + action

    fn select_action(self, state_idx: Int) -> Int:
        """Select action using epsilon-greedy policy."""
        var rand = random_float64()
        if rand < self.epsilon:
            # random_si64 is inclusive on both ends, so use num_actions - 1
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.q_table.get_best_action(state_idx)

    fn _q_update(
        mut self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
    ):
        """Q-learning update (non-terminal)."""
        var current_q = self.q_table.get(state, action)
        var max_next_q = self.q_table.get_max_value(next_state)
        var target = reward + self.discount_factor * max_next_q
        var new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table.set(state, action, new_q)

    fn update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Dyna-Q update: direct RL + model update + planning."""
        # Step 1: Direct RL - Q-learning update from real experience
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

        # Step 2: Model learning - update model with observed transition
        var idx = self._pair_index(state_idx, action)

        # Check if this pair was already visited
        if self.model_next_state[idx] == -1:
            # First time visiting this pair - store at next available slot
            self.visited_pairs[self.num_visited] = idx
            self.num_visited += 1

        self.model_next_state[idx] = next_state_idx
        self.model_reward[idx] = reward

        # Step 3: Planning - simulate experience using the model
        if self.num_visited > 0:
            for _ in range(self.n_planning):
                # Randomly select a previously visited state-action pair
                # random_si64 is inclusive on both ends, so use num_visited - 1
                var rand_idx = Int(random_si64(0, self.num_visited - 1))
                var pair_idx = self.visited_pairs[rand_idx]

                var s = pair_idx // self.num_actions
                var a = pair_idx % self.num_actions
                var ns = self.model_next_state[pair_idx]
                var r = self.model_reward[pair_idx]

                # Q-learning update with simulated experience (assume non-terminal)
                self._q_update(s, a, r, ns)

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
    fn train[
        E: DiscreteEnv
    ](
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 100,
        n_planning: Int = 5,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> Tuple[DynaQAgent, TrainingMetrics]:
        """Train a Dyna-Q agent on the given environment.

        Args:
            env: The discrete environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            n_planning: Number of planning steps per real step.
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
        var agent = DynaQAgent(
            env.num_states(),
            env.num_actions(),
            n_planning,
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
            algorithm_name="Dyna-Q",
            environment_name=environment_name,
        )
        return (agent^, metrics^)
