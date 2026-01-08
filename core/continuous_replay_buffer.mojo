"""Continuous Experience Replay Buffer for continuous state/action RL algorithms.

This module provides replay buffer infrastructure for algorithms that work
with continuous states and actions, such as DDPG, TD3, and SAC.

The key difference from the standard ReplayBuffer is that states and actions
are stored as feature vectors (List[Float64]) rather than discrete indices (Int).

Key components:
- ContinuousTransition: Stores (features, action, reward, next_features, done)
- ContinuousReplayBuffer: Fixed-size circular buffer for continuous transitions
- PrioritizedContinuousTransition: Transition with importance sampling weight
- PrioritizedContinuousReplayBuffer: PER for continuous state/action spaces

Usage:
    # Standard buffer
    var buffer = ContinuousReplayBuffer(capacity=100000, feature_dim=10)
    buffer.push(features, action, reward, next_features, done)
    var batch = buffer.sample(batch_size=64)

    # Prioritized buffer
    var per_buffer = PrioritizedContinuousReplayBuffer(capacity=100000, feature_dim=10)
    per_buffer.push(features, action, reward, next_features, done)
    var result = per_buffer.sample(batch_size=64, beta=0.4)
    var indices = result[0]
    var batch = result[1]  # Contains .weight field for IS correction
"""

from random import random_si64, random_float64
from .sum_tree import SumTree


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


struct PrioritizedContinuousTransition(Copyable, ImplicitlyCopyable, Movable):
    """Continuous transition with importance sampling weight for PER.

    Stores:
    - state: Feature vector φ(s) representing the state
    - action: Continuous action value
    - reward: Scalar reward
    - next_state: Feature vector φ(s') for next state
    - done: Episode termination flag
    - weight: Importance sampling weight for bias correction
    """

    var state: List[Float64]
    var action: Float64
    var reward: Float64
    var next_state: List[Float64]
    var done: Bool
    var weight: Float64

    fn __init__(
        out self,
        var state: List[Float64],
        action: Float64,
        reward: Float64,
        var next_state: List[Float64],
        done: Bool,
        weight: Float64 = 1.0,
    ):
        self.state = state^
        self.action = action
        self.reward = reward
        self.next_state = next_state^
        self.done = done
        self.weight = weight

    fn __copyinit__(out self, existing: Self):
        self.state = existing.state.copy()
        self.action = existing.action
        self.reward = existing.reward
        self.next_state = existing.next_state.copy()
        self.done = existing.done
        self.weight = existing.weight

    fn __moveinit__(out self, deinit existing: Self):
        self.state = existing.state^
        self.action = existing.action
        self.reward = existing.reward
        self.next_state = existing.next_state^
        self.done = existing.done
        self.weight = existing.weight


struct PrioritizedContinuousReplayBuffer:
    """Prioritized Experience Replay for continuous state/action spaces.

    Uses sum-tree for O(log n) sampling and importance sampling weights
    to correct for the bias introduced by non-uniform sampling.

    Uses proportional prioritization:
    P(i) = p_i^α / Σ_k p_k^α
    w_i = (N * P(i))^(-β) / max_j w_j

    Where p_i = |δ_i| + ε (TD error + small constant)

    Key parameters:
    - alpha: Priority exponent (0 = uniform, 1 = full prioritization)
    - beta: IS exponent (0 = no correction, 1 = full correction)
    - epsilon: Small constant for non-zero priority

    Usage with DDPG/TD3/SAC:
        var buffer = PrioritizedContinuousReplayBuffer(
            capacity=100000,
            feature_dim=features.get_num_features(),
            alpha=0.6,
            beta=0.4
        )

        # During training
        buffer.push(features, action, reward, next_features, done)

        # Sample with IS weights
        var result = buffer.sample(batch_size, beta=current_beta)
        var indices = result[0]
        var batch = result[1]

        # Compute TD errors and update
        for i in range(len(batch)):
            var td_error = compute_td_error(batch[i])
            # Weight the gradient by batch[i].weight
            buffer.update_priority(indices[i], td_error)

    Reference: Schaul et al., "Prioritized Experience Replay" (2015)
    """

    var states: List[List[Float64]]
    var actions: List[Float64]
    var rewards: List[Float64]
    var next_states: List[List[Float64]]
    var dones: List[Bool]
    var tree: SumTree

    var capacity: Int
    var feature_dim: Int
    var size: Int
    var position: Int
    var alpha: Float64
    var beta: Float64
    var epsilon: Float64
    var max_priority: Float64

    fn __init__(
        out self,
        capacity: Int,
        feature_dim: Int,
        alpha: Float64 = 0.6,
        beta: Float64 = 0.4,
        epsilon: Float64 = 1e-6,
    ):
        """Initialize prioritized buffer for continuous spaces.

        Args:
            capacity: Maximum number of transitions to store
            feature_dim: Dimensionality of state feature vectors
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Initial IS exponent (should be annealed to 1)
            epsilon: Small constant for non-zero priority
        """
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.size = 0
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = 1.0

        self.tree = SumTree(capacity)

        # Pre-allocate storage
        self.states = List[List[Float64]]()
        self.actions = List[Float64]()
        self.rewards = List[Float64]()
        self.next_states = List[List[Float64]]()
        self.dones = List[Bool]()

        for _ in range(capacity):
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

    fn _compute_priority(self, td_error: Float64) -> Float64:
        """Compute priority from TD error: (|δ| + ε)^α."""
        var abs_error = td_error if td_error > 0 else -td_error
        var p = abs_error + self.epsilon
        return p**self.alpha

    fn push(
        mut self,
        state: List[Float64],
        action: Float64,
        reward: Float64,
        next_state: List[Float64],
        done: Bool,
    ):
        """Add a transition with max priority.

        New transitions get max priority to ensure they're sampled
        at least once before their true priority is known.
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

        # Use max priority for new transitions
        var priority = self.max_priority**self.alpha
        _ = self.tree.add(priority)

        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    fn update_priority(mut self, idx: Int, td_error: Float64):
        """Update priority for a transition after computing TD error.

        Args:
            idx: Buffer index returned from sample()
            td_error: TD error for this transition
        """
        var priority = self._compute_priority(td_error)
        self.tree.update(idx, priority)

        var abs_error = td_error if td_error > 0 else -td_error
        var raw_priority = abs_error + self.epsilon
        if raw_priority > self.max_priority:
            self.max_priority = raw_priority

    fn update_priorities(mut self, indices: List[Int], td_errors: List[Float64]):
        """Batch update priorities for multiple transitions."""
        for i in range(len(indices)):
            self.update_priority(indices[i], td_errors[i])

    fn sample(
        self, batch_size: Int, beta: Float64 = -1.0
    ) -> Tuple[List[Int], List[PrioritizedContinuousTransition]]:
        """Sample batch with importance sampling weights.

        Uses stratified sampling for better coverage.

        Args:
            batch_size: Number of transitions to sample
            beta: IS exponent. If -1, uses self.beta.

        Returns:
            (indices, transitions) where transitions include IS weights.
        """
        var indices = List[Int]()
        var batch = List[PrioritizedContinuousTransition]()

        if self.size < batch_size:
            return (indices^, batch^)

        var use_beta = beta if beta >= 0 else self.beta

        var total_priority = self.tree.total_sum()
        var segment_size = total_priority / Float64(batch_size)

        # Compute min probability for weight normalization
        var min_prob = self.tree.min_priority() / total_priority
        var max_weight = (Float64(self.size) * min_prob) ** (-use_beta)

        # Stratified sampling
        for i in range(batch_size):
            var low = segment_size * Float64(i)
            var high = segment_size * Float64(i + 1)
            var target = low + random_float64() * (high - low)

            var idx = self.tree.sample(target)
            indices.append(idx)

            # Compute IS weight
            var priority = self.tree.get(idx)
            var prob = priority / total_priority
            var weight = ((Float64(self.size) * prob) ** (-use_beta)) / max_weight

            # Copy features
            var state_copy = List[Float64]()
            for j in range(self.feature_dim):
                state_copy.append(self.states[idx][j])

            var next_state_copy = List[Float64]()
            for j in range(self.feature_dim):
                next_state_copy.append(self.next_states[idx][j])

            batch.append(
                PrioritizedContinuousTransition(
                    state_copy^,
                    self.actions[idx],
                    self.rewards[idx],
                    next_state_copy^,
                    self.dones[idx],
                    weight,
                )
            )

        return (indices^, batch^)

    fn set_beta(mut self, beta: Float64):
        """Set IS exponent (should be annealed from initial to 1.0)."""
        self.beta = beta

    fn anneal_beta(mut self, progress: Float64, beta_start: Float64 = 0.4):
        """Anneal beta from beta_start to 1.0 based on training progress.

        Args:
            progress: Training progress from 0.0 to 1.0
            beta_start: Initial beta value
        """
        self.beta = beta_start + progress * (1.0 - beta_start)

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
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
