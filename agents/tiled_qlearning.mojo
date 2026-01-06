"""Tile-coded Q-Learning Agent for continuous state spaces.

Uses tile coding for linear function approximation, enabling efficient
learning in continuous state spaces like CartPole, MountainCar, etc.

Key advantages over naive discretization:
- Smooth generalization between nearby states
- Better sample efficiency through feature sharing
- Controllable resolution via tiles_per_dim
- Multiple tilings provide robustness

Example usage:
    from core.tile_coding import TileCoding, make_cartpole_tile_coding
    from agents.tiled_qlearning import TiledQLearningAgent

    var tc = make_cartpole_tile_coding(num_tilings=8, tiles_per_dim=8)
    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=2,
        learning_rate=0.1,
    )

    # Training loop
    var tiles = tc.get_tiles_simd4(obs)
    var action = agent.select_action(tiles)
    # ... environment step ...
    var next_tiles = tc.get_tiles_simd4(next_obs)
    agent.update(tiles, action, reward, next_tiles, done)
"""

from random import random_float64, random_si64
from core.tile_coding import TileCoding, TiledWeights


struct TiledQLearningAgent:
    """Q-Learning agent with tile coding function approximation.

    Uses linear function approximation where:
    Q(s, a) = sum of w[a][tile] for all active tiles

    Update rule (semi-gradient Q-learning):
    w[a][tile] += α/n * (r + γ * max_a' Q(s', a') - Q(s, a))

    where n = number of tilings (active tiles per state).
    """

    var weights: TiledWeights
    var num_actions: Int
    var num_tilings: Int
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        init_value: Float64 = 0.0,
    ):
        """Initialize tiled Q-learning agent.

        Args:
            tile_coding: TileCoding instance defining the feature representation
            num_actions: Number of discrete actions
            learning_rate: Learning rate α (default 0.1)
            discount_factor: Discount factor γ (default 0.99)
            epsilon: Initial exploration rate (default 1.0)
            epsilon_decay: Epsilon decay per episode (default 0.995)
            epsilon_min: Minimum epsilon (default 0.01)
            init_value: Initial weight value (0.0 or optimistic for exploration)
        """
        self.num_actions = num_actions
        self.num_tilings = tile_coding.get_num_tilings()
        self.weights = TiledWeights(
            num_tiles=tile_coding.get_num_tiles(),
            num_actions=num_actions,
            init_value=init_value,
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    fn select_action(self, tiles: List[Int]) -> Int:
        """Select action using epsilon-greedy policy.

        Args:
            tiles: Active tile indices from TileCoding.get_tiles()

        Returns:
            Selected action index
        """
        if random_float64() < self.epsilon:
            # Explore: random action (random_si64 is inclusive on both ends)
            return Int(random_si64(0, self.num_actions - 1))
        else:
            # Exploit: best action
            return self.weights.get_best_action(tiles)

    fn get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action (no exploration).

        Args:
            tiles: Active tile indices

        Returns:
            Action with highest Q-value
        """
        return self.weights.get_best_action(tiles)

    fn get_value(self, tiles: List[Int], action: Int) -> Float64:
        """Get Q-value for state-action pair.

        Args:
            tiles: Active tile indices
            action: Action index

        Returns:
            Q(s, a)
        """
        return self.weights.get_value(tiles, action)

    fn get_max_value(self, tiles: List[Int]) -> Float64:
        """Get maximum Q-value over all actions.

        Args:
            tiles: Active tile indices

        Returns:
            max_a Q(s, a)
        """
        var best_action = self.weights.get_best_action(tiles)
        return self.weights.get_value(tiles, best_action)

    fn update(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        next_tiles: List[Int],
        done: Bool,
    ):
        """Update weights using Q-learning (semi-gradient).

        TD target: r + γ * max_a' Q(s', a') for non-terminal
                   r for terminal

        Args:
            tiles: Active tiles for current state
            action: Action taken
            reward: Reward received
            next_tiles: Active tiles for next state
            done: Whether episode terminated
        """
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.get_max_value(next_tiles)

        self.weights.update(tiles, action, target, self.learning_rate)

    fn decay_epsilon(mut self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current exploration rate."""
        return self.epsilon

    fn reset(mut self):
        """Reset for new episode (no-op, kept for interface compatibility)."""
        pass


struct TiledSARSAAgent:
    """SARSA agent with tile coding function approximation.

    On-policy variant using actual next action:
    Q(s, a) += α * (r + γ * Q(s', a') - Q(s, a))
    """

    var weights: TiledWeights
    var num_actions: Int
    var num_tilings: Int
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        init_value: Float64 = 0.0,
    ):
        """Initialize tiled SARSA agent."""
        self.num_actions = num_actions
        self.num_tilings = tile_coding.get_num_tilings()
        self.weights = TiledWeights(
            num_tiles=tile_coding.get_num_tiles(),
            num_actions=num_actions,
            init_value=init_value,
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    fn select_action(self, tiles: List[Int]) -> Int:
        """Select action using epsilon-greedy policy."""
        if random_float64() < self.epsilon:
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.weights.get_best_action(tiles)

    fn get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action."""
        return self.weights.get_best_action(tiles)

    fn get_value(self, tiles: List[Int], action: Int) -> Float64:
        """Get Q-value for state-action pair."""
        return self.weights.get_value(tiles, action)

    fn update(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        next_tiles: List[Int],
        next_action: Int,
        done: Bool,
    ):
        """Update weights using SARSA (on-policy).

        Args:
            tiles: Active tiles for current state
            action: Action taken
            reward: Reward received
            next_tiles: Active tiles for next state
            next_action: Actual action selected for next state
            done: Whether episode terminated
        """
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.get_value(next_tiles, next_action)

        self.weights.update(tiles, action, target, self.learning_rate)

    fn decay_epsilon(mut self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current exploration rate."""
        return self.epsilon

    fn reset(mut self):
        """Reset for new episode."""
        pass


struct TiledSARSALambdaAgent:
    """SARSA(λ) agent with tile coding and eligibility traces.

    Combines tile coding with eligibility traces for faster credit assignment.
    Uses replacing traces (set to 1 when visited, decay otherwise).
    """

    var weights: TiledWeights
    var traces: List[List[Float64]]  # Eligibility traces [action][tile]
    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var learning_rate: Float64
    var discount_factor: Float64
    var lambda_: Float64  # Trace decay
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        lambda_: Float64 = 0.9,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        init_value: Float64 = 0.0,
    ):
        """Initialize tiled SARSA(λ) agent.

        Args:
            tile_coding: TileCoding instance
            num_actions: Number of actions
            learning_rate: Learning rate α
            discount_factor: Discount γ
            lambda_: Eligibility trace decay (0 = SARSA, 1 = Monte Carlo)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
            init_value: Initial weight value
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.weights = TiledWeights(
            num_tiles=self.num_tiles,
            num_actions=num_actions,
            init_value=init_value,
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize eligibility traces
        self.traces = List[List[Float64]]()
        for a in range(num_actions):
            var action_traces = List[Float64]()
            for t in range(self.num_tiles):
                action_traces.append(0.0)
            self.traces.append(action_traces^)

    fn select_action(self, tiles: List[Int]) -> Int:
        """Select action using epsilon-greedy policy."""
        if random_float64() < self.epsilon:
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.weights.get_best_action(tiles)

    fn get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action."""
        return self.weights.get_best_action(tiles)

    fn get_value(self, tiles: List[Int], action: Int) -> Float64:
        """Get Q-value for state-action pair."""
        return self.weights.get_value(tiles, action)

    fn update(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        next_tiles: List[Int],
        next_action: Int,
        done: Bool,
    ):
        """Update weights and traces using SARSA(λ).

        Steps:
        1. Compute TD error: δ = r + γ * Q(s', a') - Q(s, a)
        2. Update traces: e(s, a) = 1 for active tiles (replacing traces)
        3. Update all weights: w += α * δ * e
        4. Decay traces: e *= γ * λ
        """
        # Compute TD error
        var current_value = self.weights.get_value(tiles, action)
        var next_value: Float64 = 0.0
        if not done:
            next_value = self.weights.get_value(next_tiles, next_action)
        var td_error = reward + self.discount_factor * next_value - current_value

        # Set traces for current state-action (replacing traces)
        # First, set all traces for this action's active tiles to 1
        for i in range(len(tiles)):
            var tile_idx = tiles[i]
            self.traces[action][tile_idx] = 1.0

        # Update weights using traces
        var step_size = self.learning_rate / Float64(self.num_tilings)
        for a in range(self.num_actions):
            for t in range(self.num_tiles):
                if self.traces[a][t] > 0.0:
                    self.weights.weights[a][t] += step_size * td_error * self.traces[a][t]

        # Decay traces
        var decay = self.discount_factor * self.lambda_
        for a in range(self.num_actions):
            for t in range(self.num_tiles):
                self.traces[a][t] *= decay

    fn decay_epsilon(mut self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current exploration rate."""
        return self.epsilon

    fn reset(mut self):
        """Reset eligibility traces for new episode."""
        for a in range(self.num_actions):
            for t in range(self.num_tiles):
                self.traces[a][t] = 0.0
