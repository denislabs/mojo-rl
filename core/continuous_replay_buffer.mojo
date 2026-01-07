"""Continuous Experience Replay Buffer for continuous state/action RL algorithms.

This module provides replay buffer infrastructure for algorithms that work
with continuous states and actions, such as DDPG, TD3, and SAC.

The key difference from the standard ReplayBuffer is that states and actions
are stored as feature vectors (List[Float64]) rather than discrete indices (Int).

Key components:
- ContinuousTransition: Stores (features, action, reward, next_features, done)
- ContinuousReplayBuffer: Fixed-size circular buffer for continuous transitions

Usage:
    var buffer = ContinuousReplayBuffer(capacity=100000, feature_dim=10)
    buffer.push(features, action, reward, next_features, done)
    var batch = buffer.sample(batch_size=64)
"""

from random import random_si64


struct ContinuousTransition(Copyable, ImplicitlyCopyable, Movable):
    """Single transition tuple with continuous state features and action.

    Stores:
    - state: Feature vector φ(s) representing the state
    - action: Continuous action value(s)
    - reward: Scalar reward
    - next_state: Feature vector φ(s') for next state
    - done: Episode termination flag
    """

    var state: List[Float64]
    var action: Float64  # Single continuous action (for 1D action space)
    var reward: Float64
    var next_state: List[Float64]
    var done: Bool

    fn __init__(
        out self,
        var state: List[Float64],
        action: Float64,
        reward: Float64,
        var next_state: List[Float64],
        done: Bool,
    ):
        self.state = state^
        self.action = action
        self.reward = reward
        self.next_state = next_state^
        self.done = done

    fn __copyinit__(out self, existing: Self):
        self.state = existing.state.copy()
        self.action = existing.action
        self.reward = existing.reward
        self.next_state = existing.next_state.copy()
        self.done = existing.done

    fn __moveinit__(out self, deinit existing: Self):
        self.state = existing.state^
        self.action = existing.action
        self.reward = existing.reward
        self.next_state = existing.next_state^
        self.done = existing.done


struct ContinuousReplayBuffer:
    """Fixed-size circular buffer for continuous state/action experience replay.

    Unlike the standard ReplayBuffer which stores discrete state indices,
    this buffer stores continuous feature vectors for states and continuous
    action values.

    Benefits:
    - Supports continuous state spaces via feature representations
    - Supports continuous action spaces (required for DDPG, TD3, SAC)
    - Enables off-policy learning with function approximation

    Usage:
        var buffer = ContinuousReplayBuffer(capacity=100000, feature_dim=10)

        # During training:
        var features = feature_extractor.get_features(obs)
        var action = agent.select_action(features)
        var next_features = feature_extractor.get_features(next_obs)
        buffer.push(features, action, reward, next_features, done)

        # For updates:
        if buffer.len() >= batch_size:
            var batch = buffer.sample(batch_size)
            agent.update(batch)
    """

    # Storage for transitions (parallel arrays for efficiency)
    var states: List[List[Float64]]
    var actions: List[Float64]
    var rewards: List[Float64]
    var next_states: List[List[Float64]]
    var dones: List[Bool]

    var capacity: Int
    var feature_dim: Int
    var size: Int
    var position: Int  # Write position (circular)

    fn __init__(out self, capacity: Int, feature_dim: Int):
        """Initialize buffer with given capacity and feature dimension.

        Args:
            capacity: Maximum number of transitions to store
            feature_dim: Dimensionality of state feature vectors
        """
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.size = 0
        self.position = 0

        # Pre-allocate storage
        self.states = List[List[Float64]]()
        self.actions = List[Float64]()
        self.rewards = List[Float64]()
        self.next_states = List[List[Float64]]()
        self.dones = List[Bool]()

        for _ in range(capacity):
            # Create empty feature vectors (will be overwritten on push)
            var empty_features = List[Float64]()
            for _ in range(feature_dim):
                empty_features.append(0.0)
            var empty_features_copy = List[Float64]()
            for _ in range(feature_dim):
                empty_features_copy.append(0.0)

            self.states.append(empty_features^)
            self.actions.append(0.0)
            self.rewards.append(0.0)
            self.next_states.append(empty_features_copy^)
            self.dones.append(False)

    fn push(
        mut self,
        state: List[Float64],
        action: Float64,
        reward: Float64,
        next_state: List[Float64],
        done: Bool,
    ):
        """Add a transition to the buffer.

        Args:
            state: Feature vector for current state
            action: Continuous action taken
            reward: Reward received
            next_state: Feature vector for next state
            done: Whether episode terminated
        """
        # Copy state features
        for i in range(min(len(state), self.feature_dim)):
            self.states[self.position][i] = state[i]

        self.actions[self.position] = action
        self.rewards[self.position] = reward

        # Copy next_state features
        for i in range(min(len(next_state), self.feature_dim)):
            self.next_states[self.position][i] = next_state[i]

        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    fn sample(self, batch_size: Int) -> List[ContinuousTransition]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of ContinuousTransition objects.
            Returns empty list if buffer has fewer transitions than batch_size.
        """
        var batch = List[ContinuousTransition]()

        if self.size < batch_size:
            return batch^

        # Sample random indices
        for _ in range(batch_size):
            var idx = Int(random_si64(0, self.size - 1))

            # Copy state features
            var state_copy = List[Float64]()
            for i in range(self.feature_dim):
                state_copy.append(self.states[idx][i])

            # Copy next_state features
            var next_state_copy = List[Float64]()
            for i in range(self.feature_dim):
                next_state_copy.append(self.next_states[idx][i])

            batch.append(
                ContinuousTransition(
                    state_copy^,
                    self.actions[idx],
                    self.rewards[idx],
                    next_state_copy^,
                    self.dones[idx],
                )
            )

        return batch^

    fn sample_indices(self, batch_size: Int) -> List[Int]:
        """Sample random indices for batch processing.

        Useful when you want to process transitions without copying.

        Args:
            batch_size: Number of indices to sample

        Returns:
            List of random indices into the buffer.
        """
        var indices = List[Int]()

        if self.size < batch_size:
            return indices^

        for _ in range(batch_size):
            indices.append(Int(random_si64(0, self.size - 1)))

        return indices^

    fn get(self, idx: Int) -> ContinuousTransition:
        """Get transition at index.

        Args:
            idx: Index into the buffer

        Returns:
            ContinuousTransition at the given index
        """
        # Copy state features
        var state_copy = List[Float64]()
        for i in range(self.feature_dim):
            state_copy.append(self.states[idx][i])

        # Copy next_state features
        var next_state_copy = List[Float64]()
        for i in range(self.feature_dim):
            next_state_copy.append(self.next_states[idx][i])

        return ContinuousTransition(
            state_copy^,
            self.actions[idx],
            self.rewards[idx],
            next_state_copy^,
            self.dones[idx],
        )

    fn get_state(self, idx: Int) -> List[Float64]:
        """Get state features at index (without copying next_state).

        Args:
            idx: Index into the buffer

        Returns:
            State feature vector at the given index
        """
        var state_copy = List[Float64]()
        for i in range(self.feature_dim):
            state_copy.append(self.states[idx][i])
        return state_copy^

    fn get_next_state(self, idx: Int) -> List[Float64]:
        """Get next_state features at index.

        Args:
            idx: Index into the buffer

        Returns:
            Next state feature vector at the given index
        """
        var next_state_copy = List[Float64]()
        for i in range(self.feature_dim):
            next_state_copy.append(self.next_states[idx][i])
        return next_state_copy^

    fn get_action(self, idx: Int) -> Float64:
        """Get action at index."""
        return self.actions[idx]

    fn get_reward(self, idx: Int) -> Float64:
        """Get reward at index."""
        return self.rewards[idx]

    fn get_done(self, idx: Int) -> Bool:
        """Get done flag at index."""
        return self.dones[idx]

    fn len(self) -> Int:
        """Return number of transitions in buffer."""
        return self.size

    fn is_full(self) -> Bool:
        """Check if buffer is at capacity."""
        return self.size == self.capacity

    fn clear(mut self):
        """Clear all transitions from buffer."""
        self.size = 0
        self.position = 0
