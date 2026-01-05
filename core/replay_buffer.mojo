"""Experience Replay Buffer for off-policy RL algorithms."""

from random import random_si64


struct Transition(Copyable, Movable, ImplicitlyCopyable):
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


struct ReplayBuffer:
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


struct PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Samples transitions with probability proportional to their TD error.
    Important transitions (high TD error) are replayed more frequently.

    Uses proportional prioritization:
    P(i) = p_i^α / Σ_k p_k^α

    Where p_i = |δ_i| + ε (TD error + small constant)

    Benefits over uniform sampling:
    - Focuses learning on surprising/important transitions
    - Faster learning in many environments
    - Used in Rainbow DQN and other advanced algorithms
    """

    var states: List[Int]
    var actions: List[Int]
    var rewards: List[Float64]
    var next_states: List[Int]
    var dones: List[Bool]
    var priorities: List[Float64]

    var capacity: Int
    var size: Int
    var position: Int
    var alpha: Float64  # Priority exponent (0 = uniform, 1 = full prioritization)
    var epsilon: Float64  # Small constant for non-zero priority

    fn __init__(out self, capacity: Int, alpha: Float64 = 0.6, epsilon: Float64 = 0.0001):
        """Initialize prioritized buffer."""
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.alpha = alpha
        self.epsilon = epsilon

        # Pre-allocate storage
        self.states = List[Int]()
        self.actions = List[Int]()
        self.rewards = List[Float64]()
        self.next_states = List[Int]()
        self.dones = List[Bool]()
        self.priorities = List[Float64]()

        for _ in range(capacity):
            self.states.append(0)
            self.actions.append(0)
            self.rewards.append(0.0)
            self.next_states.append(0)
            self.dones.append(False)
            self.priorities.append(0.0)

    fn push(
        mut self,
        state: Int,
        action: Int,
        reward: Float64,
        next_state: Int,
        done: Bool,
        priority: Float64 = 1.0,
    ):
        """Add a transition with given priority."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Use max priority for new transitions (ensures they get sampled at least once)
        var max_priority = priority
        if self.size > 0:
            for i in range(self.size):
                if self.priorities[i] > max_priority:
                    max_priority = self.priorities[i]
        self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    fn _compute_priority(self, td_error: Float64) -> Float64:
        """Compute priority from TD error."""
        var abs_error = td_error if td_error > 0 else -td_error
        var p = abs_error + self.epsilon
        # Raise to power alpha
        var result = p
        if self.alpha != 1.0:
            # Simple power approximation for non-integer alpha
            result = p  # For simplicity, use linear priority
        return result

    fn update_priority(mut self, idx: Int, td_error: Float64):
        """Update priority for a transition after learning."""
        self.priorities[idx] = self._compute_priority(td_error)

    fn sample(self, batch_size: Int) -> Tuple[List[Int], List[Transition]]:
        """Sample batch based on priorities.

        Returns (indices, transitions) so priorities can be updated after learning.
        """
        var indices = List[Int]()
        var batch = List[Transition]()

        if self.size < batch_size:
            return (indices^, batch^)

        # Compute total priority
        var total_priority: Float64 = 0.0
        for i in range(self.size):
            total_priority += self.priorities[i]

        # Sample proportionally
        for _ in range(batch_size):
            var rand = random_si64(0, 1000000)
            var target = Float64(rand) / 1000000.0 * total_priority

            var cumsum: Float64 = 0.0
            var selected_idx = 0
            for i in range(self.size):
                cumsum += self.priorities[i]
                if cumsum >= target:
                    selected_idx = i
                    break

            indices.append(selected_idx)
            batch.append(
                Transition(
                    self.states[selected_idx],
                    self.actions[selected_idx],
                    self.rewards[selected_idx],
                    self.next_states[selected_idx],
                    self.dones[selected_idx],
                )
            )

        return (indices^, batch^)

    fn len(self) -> Int:
        return self.size
