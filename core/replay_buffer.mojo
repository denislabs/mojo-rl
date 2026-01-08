"""Experience Replay Buffer for off-policy RL algorithms."""

from random import random_si64, random_float64
from .sum_tree import SumTree


struct Transition(Copyable, ImplicitlyCopyable, Movable):
    """Single transition tuple (s, a, r, s', done)."""

    var state: Int
    var action: Int
    var reward: Float64
    var next_state: Int
    var done: Bool

    fn __init__(
        out self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    fn __copyinit__(out self, existing: Self):
        self.state = existing.state
        self.action = existing.action
        self.reward = existing.reward
        self.next_state = existing.next_state
        self.done = existing.done

    fn __moveinit__(out self, deinit existing: Self):
        self.state = existing.state
        self.action = existing.action
        self.reward = existing.reward
        self.next_state = existing.next_state
        self.done = existing.done


struct ReplayBuffer(Copyable, ImplicitlyCopyable, Movable):
    """Fixed-size circular buffer for experience replay.

    Experience replay breaks temporal correlations in training data
    by storing transitions and sampling random mini-batches.

    Benefits:
    - Reduces correlation between consecutive samples
    - Allows reuse of past experiences (sample efficiency)
    - Stabilizes training for off-policy methods
    - Essential component for DQN and other deep RL algorithms

    Usage:
        var buffer = ReplayBuffer(capacity=10000)
        buffer.push(state, action, reward, next_state, done)
        var batch = buffer.sample(batch_size=32)
    """

    var states: List[Int]
    var actions: List[Int]
    var rewards: List[Float64]
    var next_states: List[Int]
    var dones: List[Bool]

    var capacity: Int
    var size: Int
    var position: Int  # Write position (circular)

    fn __init__(out self, capacity: Int):
        """Initialize buffer with given capacity."""
        self.capacity = capacity
        self.size = 0
        self.position = 0

        # Pre-allocate storage
        self.states = List[Int]()
        self.actions = List[Int]()
        self.rewards = List[Float64]()
        self.next_states = List[Int]()
        self.dones = List[Bool]()

        for _ in range(capacity):
            self.states.append(0)
            self.actions.append(0)
            self.rewards.append(0.0)
            self.next_states.append(0)
            self.dones.append(False)

    fn copy(self) -> Self:
        """Explicit copy method."""
        var new_buffer = Self(self.capacity)
        for i in range(self.size):
            new_buffer.push(
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
            )
        return new_buffer^

    fn __copyinit__(out self, existing: Self):
        self.states = existing.states.copy()
        self.actions = existing.actions.copy()
        self.rewards = existing.rewards.copy()
        self.next_states = existing.next_states.copy()
        self.dones = existing.dones.copy()
        self.capacity = existing.capacity
        self.size = existing.size
        self.position = existing.position

    fn __moveinit__(out self, deinit existing: Self):
        self.states = existing.states^
        self.actions = existing.actions^
        self.rewards = existing.rewards^
        self.next_states = existing.next_states^
        self.dones = existing.dones^
        self.capacity = existing.capacity
        self.size = existing.size
        self.position = existing.position

    fn push(
        mut self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
    ):
        """Add a transition to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    fn sample(self, batch_size: Int) -> List[Transition]:
        """Sample a random batch of transitions.

        Returns empty list if buffer has fewer transitions than batch_size.
        """
        var batch = List[Transition]()

        if self.size < batch_size:
            return batch^

        # Sample random indices
        for _ in range(batch_size):
            # random_si64 is inclusive on both ends, so use size - 1
            var idx = Int(random_si64(0, self.size - 1))
            batch.append(
                Transition(
                    self.states[idx],
                    self.actions[idx],
                    self.rewards[idx],
                    self.next_states[idx],
                    self.dones[idx],
                )
            )

        return batch^

    fn sample_indices(self, batch_size: Int) -> List[Int]:
        """Sample random indices for batch processing.

        Useful when you want to process transitions without copying.
        """
        var indices = List[Int]()

        if self.size < batch_size:
            return indices^

        for _ in range(batch_size):
            # random_si64 is inclusive on both ends, so use size - 1
            indices.append(Int(random_si64(0, self.size - 1)))

        return indices^

    fn get(self, idx: Int) -> Transition:
        """Get transition at index."""
        return Transition(
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

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


struct PrioritizedTransition(Copyable, ImplicitlyCopyable, Movable):
    """Transition with importance sampling weight for PER."""

    var state: Int
    var action: Int
    var reward: Float64
    var next_state: Int
    var done: Bool
    var weight: Float64  # Importance sampling weight

    fn __init__(
        out self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
        weight: Float64 = 1.0,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.weight = weight

    fn __copyinit__(out self, existing: Self):
        self.state = existing.state
        self.action = existing.action
        self.reward = existing.reward
        self.next_state = existing.next_state
        self.done = existing.done
        self.weight = existing.weight

    fn __moveinit__(out self, deinit existing: Self):
        self.state = existing.state
        self.action = existing.action
        self.reward = existing.reward
        self.next_state = existing.next_state
        self.done = existing.done
        self.weight = existing.weight


struct PrioritizedReplayBuffer(Movable):
    """Prioritized Experience Replay buffer with sum-tree.

    Samples transitions with probability proportional to their TD error.
    Important transitions (high TD error) are replayed more frequently.

    Uses proportional prioritization with importance sampling:
    P(i) = p_i^α / Σ_k p_k^α
    w_i = (N * P(i))^(-β) / max_j w_j  (normalized IS weights)

    Where p_i = |δ_i| + ε (TD error + small constant)

    Key parameters:
    - alpha: Priority exponent (0 = uniform, 1 = full prioritization)
    - beta: Importance sampling exponent (0 = no correction, 1 = full correction)
    - epsilon: Small constant to ensure non-zero priority

    Benefits over uniform sampling:
    - Focuses learning on surprising/important transitions
    - Faster learning in many environments
    - Used in Rainbow DQN and other advanced algorithms

    Reference: Schaul et al., "Prioritized Experience Replay" (2015)

    Usage:
        var buffer = PrioritizedReplayBuffer(capacity=10000)

        # Store transitions
        buffer.push(state, action, reward, next_state, done)

        # Sample with importance weights
        var result = buffer.sample(batch_size=32, beta=0.4)
        var indices = result[0]
        var batch = result[1]  # List[PrioritizedTransition] with .weight field

        # Update priorities after computing TD errors
        for i in range(len(indices)):
            buffer.update_priority(indices[i], td_errors[i])
    """

    var states: List[Int]
    var actions: List[Int]
    var rewards: List[Float64]
    var next_states: List[Int]
    var dones: List[Bool]
    var tree: SumTree  # Sum-tree for O(log n) sampling

    var capacity: Int
    var size: Int
    var position: Int
    var alpha: Float64  # Priority exponent (0 = uniform, 1 = full prioritization)
    var beta: Float64  # Importance sampling exponent (annealed from beta_start to 1)
    var epsilon: Float64  # Small constant for non-zero priority
    var max_priority: Float64  # Track max priority for new transitions

    fn __init__(
        out self,
        capacity: Int,
        alpha: Float64 = 0.6,
        beta: Float64 = 0.4,
        epsilon: Float64 = 1e-6,
    ):
        """Initialize prioritized buffer with sum-tree.

        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Initial importance sampling exponent (annealed to 1)
            epsilon: Small constant to ensure non-zero priority
        """
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initial max priority

        # Initialize sum-tree
        self.tree = SumTree(capacity)

        # Pre-allocate storage
        self.states = List[Int]()
        self.actions = List[Int]()
        self.rewards = List[Float64]()
        self.next_states = List[Int]()
        self.dones = List[Bool]()

        for _ in range(capacity):
            self.states.append(0)
            self.actions.append(0)
            self.rewards.append(0.0)
            self.next_states.append(0)
            self.dones.append(False)

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.states = existing.states^
        self.actions = existing.actions^
        self.rewards = existing.rewards^
        self.next_states = existing.next_states^
        self.dones = existing.dones^
        self.tree = existing.tree^
        self.capacity = existing.capacity
        self.size = existing.size
        self.position = existing.position
        self.alpha = existing.alpha
        self.beta = existing.beta
        self.epsilon = existing.epsilon
        self.max_priority = existing.max_priority

    fn _compute_priority(self, td_error: Float64) -> Float64:
        """Compute priority from TD error: (|δ| + ε)^α."""
        var abs_error = td_error if td_error > 0 else -td_error
        var p = abs_error + self.epsilon
        return p**self.alpha

    fn push(
        mut self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
    ):
        """Add a transition with max priority (ensures it gets sampled).

        New transitions are given max priority to ensure they're sampled
        at least once before their true priority is known.
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
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
            td_error: TD error |Q(s,a) - (r + γ max_a' Q(s',a'))|
        """
        var priority = self._compute_priority(td_error)
        self.tree.update(idx, priority)

        # Update max priority
        var abs_error = td_error if td_error > 0 else -td_error
        var raw_priority = abs_error + self.epsilon
        if raw_priority > self.max_priority:
            self.max_priority = raw_priority

    fn update_priorities(mut self, indices: List[Int], td_errors: List[Float64]):
        """Batch update priorities for multiple transitions.

        Args:
            indices: Buffer indices returned from sample()
            td_errors: TD errors for each transition
        """
        for i in range(len(indices)):
            self.update_priority(indices[i], td_errors[i])

    fn sample(
        self, batch_size: Int, beta: Float64 = -1.0
    ) -> Tuple[List[Int], List[PrioritizedTransition]]:
        """Sample batch based on priorities with importance sampling weights.

        Uses stratified sampling: divides priority range into batch_size segments
        and samples one transition from each segment.

        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling exponent. If -1, uses self.beta.
                  Should be annealed from initial value to 1 during training.

        Returns:
            (indices, transitions) where transitions include IS weights.
            Weights are normalized so max weight = 1.
        """
        var indices = List[Int]()
        var batch = List[PrioritizedTransition]()

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
            # Sample uniformly within segment
            var low = segment_size * Float64(i)
            var high = segment_size * Float64(i + 1)
            var target = low + random_float64() * (high - low)

            var idx = self.tree.sample(target)
            indices.append(idx)

            # Compute importance sampling weight
            var priority = self.tree.get(idx)
            var prob = priority / total_priority
            var weight = ((Float64(self.size) * prob) ** (-use_beta)) / max_weight

            batch.append(
                PrioritizedTransition(
                    self.states[idx],
                    self.actions[idx],
                    self.rewards[idx],
                    self.next_states[idx],
                    self.dones[idx],
                    weight,
                )
            )

        return (indices^, batch^)

    fn set_beta(mut self, beta: Float64):
        """Set importance sampling exponent.

        Should be annealed from initial value (e.g., 0.4) to 1.0
        over the course of training.
        """
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
