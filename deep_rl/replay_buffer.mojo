"""Replay Buffer for deep RL with compile-time dimensions.

Stores transitions (s, a, r, s', done) for experience replay.
Uses circular buffer with fixed capacity known at compile time.
"""

from random import random_float64


struct ReplayBuffer[
    capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    dtype: DType = DType.float64,
]:
    """Fixed-size replay buffer with compile-time dimensions.

    Parameters:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        dtype: Data type for storage.
    """

    # Storage arrays
    var obs: InlineArray[Scalar[Self.dtype], Self.capacity * Self.obs_dim]
    var actions: InlineArray[Scalar[Self.dtype], Self.capacity * Self.action_dim]
    var rewards: InlineArray[Scalar[Self.dtype], Self.capacity]
    var next_obs: InlineArray[Scalar[Self.dtype], Self.capacity * Self.obs_dim]
    var dones: InlineArray[Scalar[Self.dtype], Self.capacity]  # 1.0 if done, 0.0 otherwise

    # Buffer state
    var size: Int
    var ptr: Int  # Next write position

    fn __init__(out self):
        """Initialize empty replay buffer."""
        self.obs = InlineArray[Scalar[Self.dtype], Self.capacity * Self.obs_dim](fill=0)
        self.actions = InlineArray[Scalar[Self.dtype], Self.capacity * Self.action_dim](fill=0)
        self.rewards = InlineArray[Scalar[Self.dtype], Self.capacity](fill=0)
        self.next_obs = InlineArray[Scalar[Self.dtype], Self.capacity * Self.obs_dim](fill=0)
        self.dones = InlineArray[Scalar[Self.dtype], Self.capacity](fill=0)
        self.size = 0
        self.ptr = 0

    fn add(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        action: InlineArray[Scalar[Self.dtype], Self.action_dim],
        reward: Scalar[Self.dtype],
        next_obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        done: Bool,
    ):
        """Add a transition to the buffer."""
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
        if done:
            self.dones[self.ptr] = 1.0
        else:
            self.dones[self.ptr] = 0.0

        # Update pointer and size
        self.ptr = (self.ptr + 1) % Self.capacity
        if self.size < Self.capacity:
            self.size += 1

    fn sample[batch_size: Int](
        self,
        mut batch_obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        mut batch_actions: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        mut batch_rewards: InlineArray[Scalar[Self.dtype], batch_size],
        mut batch_next_obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        mut batch_dones: InlineArray[Scalar[Self.dtype], batch_size],
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
                batch_obs[b * Self.obs_dim + i] = self.obs[idx * Self.obs_dim + i]

            # Copy action
            for i in range(Self.action_dim):
                batch_actions[b * Self.action_dim + i] = self.actions[idx * Self.action_dim + i]

            # Copy reward
            batch_rewards[b] = self.rewards[idx]

            # Copy next observation
            for i in range(Self.obs_dim):
                batch_next_obs[b * Self.obs_dim + i] = self.next_obs[idx * Self.obs_dim + i]

            # Copy done flag
            batch_dones[b] = self.dones[idx]

    fn len(self) -> Int:
        """Return current number of transitions stored."""
        return self.size

    fn is_ready[batch_size: Int](self) -> Bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size
