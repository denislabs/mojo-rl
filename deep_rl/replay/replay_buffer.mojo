"""Replay Buffer for deep RL with compile-time dimensions.

Stores transitions (s, a, r, s', done) for experience replay.
Uses circular buffer with fixed capacity known at compile time.

Key components:
- ReplayBuffer: Standard uniform sampling replay buffer.
- PrioritizedReplayBuffer: PER with sum-tree for O(log n) priority sampling.

For the prioritized version, importance sampling weights are returned
alongside samples to correct for the bias introduced by non-uniform sampling.
"""

from random import random_float64


trait ReplayBufferTrait:
    """Trait for replay buffers."""

    comptime OBS_DIM: Int
    comptime ACTION_DIM: Int
    comptime DTYPE: DType

    fn __init__(out self):
        """Initialize the replay buffer."""
        ...

    fn add(
        mut self,
        obs: InlineArray[Scalar[Self.DTYPE], Self.OBS_DIM],
        action: InlineArray[Scalar[Self.DTYPE], Self.ACTION_DIM],
        reward: Scalar[Self.DTYPE],
        next_obs: InlineArray[Scalar[Self.DTYPE], Self.OBS_DIM],
        done: Bool,
    ):
        """Add a transition to the buffer."""
        ...

    fn sample[
        batch_size: Int
    ](
        self,
        mut batch_obs: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.OBS_DIM
        ],
        mut batch_actions: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.ACTION_DIM
        ],
        mut batch_rewards: InlineArray[Scalar[Self.DTYPE], batch_size],
        mut batch_next_obs: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.OBS_DIM
        ],
        mut batch_dones: InlineArray[Scalar[Self.DTYPE], batch_size],
    ):
        """Sample a batch from the buffer."""
        ...

    fn len(self) -> Int:
        """Return the current number of transitions in the buffer."""
        ...

    fn is_ready[batch_size: Int](self) -> Bool:
        """Check if the buffer has enough samples for a batch."""
        ...


struct ReplayBuffer[
    capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    dtype: DType = DType.float64,
](ReplayBufferTrait):
    """Fixed-size replay buffer with compile-time dimensions.

    Parameters:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        dtype: Data type for storage.
    """

    comptime OBS_DIM = Self.obs_dim
    comptime ACTION_DIM = Self.action_dim
    comptime DTYPE = Self.dtype

    # Storage arrays
    var obs: InlineArray[Scalar[Self.DTYPE], Self.capacity * Self.OBS_DIM]
    var actions: InlineArray[
        Scalar[Self.DTYPE], Self.capacity * Self.ACTION_DIM
    ]
    var rewards: InlineArray[Scalar[Self.DTYPE], Self.capacity]
    var next_obs: InlineArray[Scalar[Self.DTYPE], Self.capacity * Self.OBS_DIM]
    var dones: InlineArray[
        Scalar[Self.DTYPE], Self.capacity
    ]  # 1.0 if done, 0.0 otherwise

    # Buffer state
    var size: Int
    var ptr: Int  # Next write position

    fn __init__(out self):
        """Initialize empty replay buffer."""
        self.obs = InlineArray[
            Scalar[Self.DTYPE], Self.capacity * Self.OBS_DIM
        ](fill=0)
        self.actions = InlineArray[
            Scalar[Self.DTYPE], Self.capacity * Self.ACTION_DIM
        ](fill=0)
        self.rewards = InlineArray[Scalar[Self.DTYPE], Self.capacity](fill=0)
        self.next_obs = InlineArray[
            Scalar[Self.DTYPE], Self.capacity * Self.OBS_DIM
        ](fill=0)
        self.dones = InlineArray[Scalar[Self.DTYPE], Self.capacity](fill=0)
        self.size = 0
        self.ptr = 0

    fn add(
        mut self,
        obs: InlineArray[Scalar[Self.DTYPE], Self.OBS_DIM],
        action: InlineArray[Scalar[Self.DTYPE], Self.ACTION_DIM],
        reward: Scalar[Self.DTYPE],
        next_obs: InlineArray[Scalar[Self.DTYPE], Self.OBS_DIM],
        done: Bool,
    ):
        """Add a transition to the buffer."""
        # Store observation
        for i in range(Self.OBS_DIM):
            self.obs[self.ptr * Self.OBS_DIM + i] = obs[i]

        # Store action
        for i in range(Self.ACTION_DIM):
            self.actions[self.ptr * Self.ACTION_DIM + i] = action[i]

        # Store reward
        self.rewards[self.ptr] = reward

        # Store next observation
        for i in range(Self.OBS_DIM):
            self.next_obs[self.ptr * Self.OBS_DIM + i] = next_obs[i]

        # Store done flag
        self.dones[self.ptr] = Scalar[Self.DTYPE](1.0) if done else Scalar[
            Self.DTYPE
        ](0.0)

        # Update pointer and size
        self.ptr = (self.ptr + 1) % Self.capacity
        if self.size < Self.capacity:
            self.size += 1

    fn sample[
        batch_size: Int
    ](
        self,
        mut batch_obs: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.OBS_DIM
        ],
        mut batch_actions: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.ACTION_DIM
        ],
        mut batch_rewards: InlineArray[Scalar[Self.DTYPE], batch_size],
        mut batch_next_obs: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.OBS_DIM
        ],
        mut batch_dones: InlineArray[Scalar[Self.DTYPE], batch_size],
    ):
        """Sample a random batch from the buffer.

        Args:
            batch_obs: Output buffer for observations.
            batch_actions: Output buffer for actions.
            batch_rewards: Output buffer for rewards.
            batch_next_obs: Output buffer for next observations.
            batch_dones: Output buffer for done flags.
        """
        for b in range(batch_size):
            # Random index in valid range
            var idx = Int(random_float64() * Float64(self.size)) % self.size

            # Copy observation
            for i in range(Self.obs_dim):
                batch_obs[b * Self.obs_dim + i] = self.obs[
                    idx * Self.obs_dim + i
                ]

            # Copy action
            for i in range(Self.action_dim):
                batch_actions[b * Self.action_dim + i] = self.actions[
                    idx * Self.action_dim + i
                ]

            # Copy reward
            batch_rewards[b] = self.rewards[idx]

            # Copy next observation
            for i in range(Self.obs_dim):
                batch_next_obs[b * Self.obs_dim + i] = self.next_obs[
                    idx * Self.obs_dim + i
                ]

            # Copy done flag
            batch_dones[b] = self.dones[idx]

    fn len(self) -> Int:
        """Return current number of transitions stored."""
        return self.size

    fn is_ready[batch_size: Int](self) -> Bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size


struct PrioritizedReplayBuffer[
    capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    dtype: DType = DType.float64,
]:
    """Prioritized Experience Replay buffer with compile-time dimensions.

    Uses proportional prioritization with importance sampling:
    P(i) = p_i^α / Σ_k p_k^α
    w_i = (N * P(i))^(-β) / max_j w_j

    Key parameters:
    - alpha: Priority exponent (0 = uniform, 1 = full prioritization).
    - beta: IS exponent (0 = no correction, 1 = full correction).
    - epsilon: Small constant for non-zero priority.

    The sum-tree is implemented inline using InlineArray for O(log n) sampling.

    Parameters:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        dtype: Data type for storage.

    Usage:
        alias Buffer = PrioritizedReplayBuffer[10000, 4, 1]
        var buffer = Buffer(alpha=0.6, beta=0.4)

        buffer.add(obs, action, reward, next_obs, done)

        # Sample with IS weights
        var batch_obs = InlineArray[...](fill=0)
        var batch_weights = InlineArray[...](fill=0)
        var batch_indices = InlineArray[...](fill=0)
        buffer.sample[64](batch_obs, ..., batch_weights, batch_indices)

        # After computing TD errors, update priorities
        buffer.update_priorities[64](batch_indices, td_errors)

    Reference: Schaul et al., "Prioritized Experience Replay" (2015)
    """

    comptime OBS_DIM = Self.obs_dim
    comptime ACTION_DIM = Self.action_dim
    comptime DTYPE = Self.dtype

    # Storage arrays
    var obs: InlineArray[Scalar[Self.DTYPE], Self.capacity * Self.OBS_DIM]
    var actions: InlineArray[
        Scalar[Self.DTYPE], Self.capacity * Self.ACTION_DIM
    ]
    var rewards: InlineArray[Scalar[Self.DTYPE], Self.capacity]
    var next_obs: InlineArray[Scalar[Self.DTYPE], Self.capacity * Self.OBS_DIM]
    var dones: InlineArray[Scalar[Self.DTYPE], Self.capacity]

    # Sum-tree for priorities (2 * capacity - 1 nodes)
    var tree: InlineArray[Scalar[Self.DTYPE], 2 * Self.capacity - 1]

    # Buffer state
    var size: Int
    var ptr: Int
    var alpha: Scalar[Self.dtype]
    var beta: Scalar[Self.dtype]
    var epsilon: Scalar[Self.dtype]
    var max_priority: Scalar[Self.dtype]

    fn __init__(
        out self,
        alpha: Scalar[Self.dtype] = 0.6,
        beta: Scalar[Self.dtype] = 0.4,
        epsilon: Scalar[Self.dtype] = 1e-6,
    ):
        """Initialize prioritized replay buffer.

        Args:
            alpha: Priority exponent (0 = uniform, 1 = full prioritization).
            beta: Initial IS exponent (should be annealed to 1).
            epsilon: Small constant for non-zero priority.
        """
        self.obs = InlineArray[
            Scalar[Self.dtype], Self.capacity * Self.obs_dim
        ](fill=0)
        self.actions = InlineArray[
            Scalar[Self.dtype], Self.capacity * Self.action_dim
        ](fill=0)
        self.rewards = InlineArray[Scalar[Self.dtype], Self.capacity](fill=0)
        self.next_obs = InlineArray[
            Scalar[Self.dtype], Self.capacity * Self.obs_dim
        ](fill=0)
        self.dones = InlineArray[Scalar[Self.dtype], Self.capacity](fill=0)
        self.tree = InlineArray[Scalar[Self.dtype], 2 * Self.capacity - 1](
            fill=0
        )
        self.size = 0
        self.ptr = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = 1.0

    fn _leaf_to_tree_idx(self, leaf_idx: Int) -> Int:
        """Convert leaf index to tree array index."""
        return leaf_idx + Self.capacity - 1

    fn _tree_to_leaf_idx(self, tree_idx: Int) -> Int:
        """Convert tree array index to leaf index."""
        return tree_idx - Self.capacity + 1

    fn _propagate_up(mut self, mut idx: Int, change: Scalar[Self.dtype]):
        """Propagate priority change up to root."""
        while idx > 0:
            var parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    fn _update_tree(mut self, leaf_idx: Int, priority: Scalar[Self.dtype]):
        """Update priority at leaf and propagate up."""
        var tree_idx = self._leaf_to_tree_idx(leaf_idx)
        var change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate_up(tree_idx, change)

    fn _sample_tree(self, target: Scalar[Self.dtype]) -> Int:
        """Sample a leaf index proportional to priorities."""
        var idx = 0  # Start at root
        var remaining = target

        while True:
            var left = 2 * idx + 1
            var right = 2 * idx + 2

            # If we've reached a leaf node
            if left >= 2 * Self.capacity - 1:
                break

            # Go left or right based on remaining target
            if remaining <= self.tree[left]:
                idx = left
            else:
                remaining -= self.tree[left]
                idx = right

        return self._tree_to_leaf_idx(idx)

    fn _total_priority(self) -> Scalar[Self.dtype]:
        """Get total sum of all priorities (root value)."""
        return self.tree[0]

    fn _min_priority(self) -> Scalar[Self.dtype]:
        """Get minimum non-zero priority among all leaves."""
        var min_p: Scalar[Self.dtype] = 1e10
        for i in range(self.size):
            var tree_idx = self._leaf_to_tree_idx(i)
            var p = self.tree[tree_idx]
            if p > 0 and p < min_p:
                min_p = p
        return min_p if min_p < 1e10 else Scalar[Self.dtype](1.0)

    fn _compute_priority(
        self, td_error: Scalar[Self.dtype]
    ) -> Scalar[Self.dtype]:
        """Compute priority from TD error: (|δ| + ε)^α."""
        var abs_error = td_error if td_error > 0 else -td_error
        var p = abs_error + self.epsilon
        return p**self.alpha

    fn add(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        action: InlineArray[Scalar[Self.dtype], Self.action_dim],
        reward: Scalar[Self.dtype],
        next_obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        done: Bool,
    ):
        """Add a transition with max priority."""
        # Store observation
        for i in range(Self.obs_dim):
            self.obs[self.ptr * Self.obs_dim + i] = obs[i]

        # Store action
        for i in range(Self.action_dim):
            self.actions[self.ptr * Self.action_dim + i] = action[i]

        # Store reward
        self.rewards[self.ptr] = reward

        # Store next observation
        for i in range(Self.obs_dim):
            self.next_obs[self.ptr * Self.obs_dim + i] = next_obs[i]

        # Store done flag
        self.dones[self.ptr] = Scalar[Self.dtype](1.0) if done else Scalar[
            Self.dtype
        ](0.0)

        # Use max priority for new transitions
        var priority = self.max_priority**self.alpha
        self._update_tree(self.ptr, priority)

        # Update pointer and size
        self.ptr = (self.ptr + 1) % Self.capacity
        if self.size < Self.capacity:
            self.size += 1

    fn update_priority(mut self, idx: Int, td_error: Scalar[Self.dtype]):
        """Update priority for a transition after computing TD error.

        Args:
            idx: Buffer index returned from sample.
            td_error: TD error for this transition.
        """
        var priority = self._compute_priority(td_error)
        self._update_tree(idx, priority)

        var abs_error = td_error if td_error > 0 else -td_error
        var raw_priority = abs_error + self.epsilon
        if raw_priority > self.max_priority:
            self.max_priority = raw_priority

    fn update_priorities[
        batch_size: Int
    ](
        mut self,
        indices: InlineArray[Int, batch_size],
        td_errors: InlineArray[Scalar[Self.dtype], batch_size],
    ):
        """Batch update priorities for multiple transitions.

        Args:
            indices: Buffer indices returned from sample.
            td_errors: TD errors for each transition.
        """
        for i in range(batch_size):
            self.update_priority(indices[i], td_errors[i])

    fn sample[
        batch_size: Int
    ](
        self,
        mut batch_obs: InlineArray[
            Scalar[Self.dtype], batch_size * Self.obs_dim
        ],
        mut batch_actions: InlineArray[
            Scalar[Self.dtype], batch_size * Self.action_dim
        ],
        mut batch_rewards: InlineArray[Scalar[Self.dtype], batch_size],
        mut batch_next_obs: InlineArray[
            Scalar[Self.dtype], batch_size * Self.obs_dim
        ],
        mut batch_dones: InlineArray[Scalar[Self.dtype], batch_size],
        mut batch_weights: InlineArray[Scalar[Self.dtype], batch_size],
        mut batch_indices: InlineArray[Int, batch_size],
    ):
        """Sample a batch with importance sampling weights.

        Uses stratified sampling for better coverage.

        Args:
            batch_obs: Output buffer for observations.
            batch_actions: Output buffer for actions.
            batch_rewards: Output buffer for rewards.
            batch_next_obs: Output buffer for next observations.
            batch_dones: Output buffer for done flags.
            batch_weights: Output buffer for IS weights.
            batch_indices: Output buffer for indices (for priority updates).
        """
        var total_priority = self._total_priority()
        var segment_size = total_priority / Scalar[Self.dtype](batch_size)

        # Compute min probability for weight normalization
        var min_prob = self._min_priority() / total_priority
        var max_weight = (Scalar[Self.dtype](self.size) * min_prob) ** (
            -self.beta
        )

        # Stratified sampling
        for b in range(batch_size):
            var low = segment_size * Scalar[Self.dtype](b)
            var high = segment_size * Scalar[Self.dtype](b + 1)
            var target = low + random_float64().cast[Self.dtype]() * (
                high - low
            )

            var idx = self._sample_tree(target)
            batch_indices[b] = idx

            # Compute IS weight
            var priority = self.tree[self._leaf_to_tree_idx(idx)]
            var prob = priority / total_priority
            var weight = (
                (Scalar[Self.dtype](self.size) * prob) ** (-self.beta)
            ) / max_weight
            batch_weights[b] = weight

            # Copy observation
            for i in range(Self.obs_dim):
                batch_obs[b * Self.obs_dim + i] = self.obs[
                    idx * Self.obs_dim + i
                ]

            # Copy action
            for i in range(Self.action_dim):
                batch_actions[b * Self.action_dim + i] = self.actions[
                    idx * Self.action_dim + i
                ]

            # Copy reward
            batch_rewards[b] = self.rewards[idx]

            # Copy next observation
            for i in range(Self.obs_dim):
                batch_next_obs[b * Self.obs_dim + i] = self.next_obs[
                    idx * Self.obs_dim + i
                ]

            # Copy done flag
            batch_dones[b] = self.dones[idx]

    fn set_beta(mut self, beta: Scalar[Self.dtype]):
        """Set IS exponent (should be annealed from initial to 1.0)."""
        self.beta = beta

    fn anneal_beta(
        mut self,
        progress: Scalar[Self.dtype],
        beta_start: Scalar[Self.dtype] = 0.4,
    ):
        """Anneal beta from beta_start to 1.0 based on training progress.

        Args:
            progress: Training progress from 0.0 to 1.0.
            beta_start: Initial beta value.
        """
        self.beta = beta_start + progress * (
            Scalar[Self.dtype](1.0) - beta_start
        )

    fn len(self) -> Int:
        """Return current number of transitions stored."""
        return self.size

    fn is_ready[batch_size: Int](self) -> Bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size


# =============================================================================
# Heap-based Replay Buffers (for large observation/action spaces)
# =============================================================================


struct HeapReplayBuffer[
    capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    dtype: DType = DType.float64,
](ReplayBufferTrait):
    """Heap-allocated replay buffer for large observation/action spaces.

    Unlike ReplayBuffer which uses stack-allocated InlineArray, this version
    uses heap-allocated List for storage. This allows much larger buffers
    without stack overflow, at the cost of slightly slower access.

    Use this for environments with large observation spaces (e.g., 24D+)
    or when you need buffer capacities > 10,000.

    Parameters:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        dtype: Data type for storage.

    Example:
        # For BipedalWalker with 24D obs, 4D actions, 100k buffer
        alias Buffer = HeapReplayBuffer[100000, 24, 4]
        var buffer = Buffer()
    """

    comptime OBS_DIM = Self.obs_dim
    comptime ACTION_DIM = Self.action_dim
    comptime DTYPE = Self.dtype

    # Heap-allocated storage
    var obs: List[Scalar[Self.DTYPE]]
    var actions: List[Scalar[Self.DTYPE]]
    var rewards: List[Scalar[Self.DTYPE]]
    var next_obs: List[Scalar[Self.DTYPE]]
    var dones: List[Scalar[Self.DTYPE]]

    # Buffer state
    var size: Int
    var ptr: Int

    fn __init__(out self):
        """Initialize empty heap replay buffer."""
        # Pre-allocate capacity
        var obs_size = Self.capacity * Self.obs_dim
        var action_size = Self.capacity * Self.action_dim

        self.obs = List[Scalar[Self.dtype]](capacity=obs_size)
        self.actions = List[Scalar[Self.dtype]](capacity=action_size)
        self.rewards = List[Scalar[Self.dtype]](capacity=Self.capacity)
        self.next_obs = List[Scalar[Self.dtype]](capacity=obs_size)
        self.dones = List[Scalar[Self.dtype]](capacity=Self.capacity)

        # Fill with zeros
        for _ in range(obs_size):
            self.obs.append(Scalar[Self.dtype](0))
            self.next_obs.append(Scalar[Self.dtype](0))
        for _ in range(action_size):
            self.actions.append(Scalar[Self.dtype](0))
        for _ in range(Self.capacity):
            self.rewards.append(Scalar[Self.dtype](0))
            self.dones.append(Scalar[Self.dtype](0))

        self.size = 0
        self.ptr = 0

    fn add(
        mut self,
        obs: InlineArray[Scalar[Self.DTYPE], Self.OBS_DIM],
        action: InlineArray[Scalar[Self.DTYPE], Self.ACTION_DIM],
        reward: Scalar[Self.DTYPE],
        next_obs: InlineArray[Scalar[Self.DTYPE], Self.OBS_DIM],
        done: Bool,
    ):
        """Add a transition to the buffer."""
        # Store observation
        for i in range(Self.OBS_DIM):
            self.obs[self.ptr * Self.obs_dim + i] = obs[i]

        # Store action
        for i in range(Self.ACTION_DIM):
            self.actions[self.ptr * Self.ACTION_DIM + i] = action[i]

        # Store reward
        self.rewards[self.ptr] = reward

        # Store next observation
        for i in range(Self.OBS_DIM):
            self.next_obs[self.ptr * Self.OBS_DIM + i] = next_obs[i]

        # Store done flag
        self.dones[self.ptr] = Scalar[Self.DTYPE](1.0) if done else Scalar[
            Self.DTYPE
        ](0.0)

        # Update pointer and size
        self.ptr = (self.ptr + 1) % Self.capacity
        if self.size < Self.capacity:
            self.size += 1

    fn sample[
        batch_size: Int
    ](
        self,
        mut batch_obs: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.OBS_DIM
        ],
        mut batch_actions: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.ACTION_DIM
        ],
        mut batch_rewards: InlineArray[Scalar[Self.DTYPE], batch_size],
        mut batch_next_obs: InlineArray[
            Scalar[Self.DTYPE], batch_size * Self.OBS_DIM
        ],
        mut batch_dones: InlineArray[Scalar[Self.DTYPE], batch_size],
    ):
        """Sample a random batch from the buffer.

        Args:
            batch_obs: Output buffer for observations.
            batch_actions: Output buffer for actions.
            batch_rewards: Output buffer for rewards.
            batch_next_obs: Output buffer for next observations.
            batch_dones: Output buffer for done flags.
        """
        for b in range(batch_size):
            # Random index in valid range
            var idx = Int(random_float64() * Float64(self.size)) % self.size

            # Copy observation
            for i in range(Self.obs_dim):
                batch_obs[b * Self.obs_dim + i] = self.obs[
                    idx * Self.obs_dim + i
                ]

            # Copy action
            for i in range(Self.action_dim):
                batch_actions[b * Self.action_dim + i] = self.actions[
                    idx * Self.action_dim + i
                ]

            # Copy reward
            batch_rewards[b] = self.rewards[idx]

            # Copy next observation
            for i in range(Self.obs_dim):
                batch_next_obs[b * Self.obs_dim + i] = self.next_obs[
                    idx * Self.obs_dim + i
                ]

            # Copy done flag
            batch_dones[b] = self.dones[idx]

    fn len(self) -> Int:
        """Return current number of transitions stored."""
        return self.size

    fn is_ready[batch_size: Int](self) -> Bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size
