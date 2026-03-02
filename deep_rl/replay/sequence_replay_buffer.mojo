"""Sequence Replay Buffer for TDMPC2.

The consistency loss in TDMPC2 requires unrolling the world model over H steps.
This buffer stores transitions and samples contiguous sequences of length H+1,
avoiding sequences that cross episode boundaries.

Interface:
  add(obs, action, reward, done)          -- continuous insertion
  sample_sequences[BATCH, H]() →
    obs[BATCH * (H+1) * OBS_DIM],          -- observations at each step
    actions[BATCH * H * ACTION_DIM],        -- actions taken
    rewards[BATCH * H],                     -- rewards received
    dones[BATCH * H],                       -- done flags
"""

from random import random_float64


struct SequenceReplayBuffer[
    capacity: Int,
    obs_dim: Int,
    action_dim: Int,
    dtype: DType = DType.float32,
]:
    """Circular replay buffer that samples contiguous sequences.

    Sequences of length H+1 observations (H steps) are sampled such that
    no episode boundary appears in steps 0..H-2 of the sequence.
    This ensures the world model can be unrolled across the full horizon.

    Parameters:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        dtype: Data type for storage (default: float32).

    Storage layout (all circular, indexed by ptr mod capacity):
        obs[t]:    observation at timestep t              [OBS_DIM]
        actions[t]: action taken at timestep t            [ACTION_DIM]
        rewards[t]: reward received at timestep t         [1]
        dones[t]:  whether episode ended after step t     [1] (1.0 or 0.0)
        valid[t]:  whether a full sequence can start at t [Bool]
    """

    comptime OBS_DIM: Int = Self.obs_dim
    comptime ACTION_DIM: Int = Self.action_dim

    # Storage - heap-allocated to support large capacities
    var obs: List[Scalar[Self.dtype]]      # [capacity * OBS_DIM]
    var actions: List[Scalar[Self.dtype]]  # [capacity * ACTION_DIM]
    var rewards: List[Scalar[Self.dtype]]  # [capacity]
    var dones: List[Scalar[Self.dtype]]    # [capacity] (1.0 = done)
    var episode_id: List[Int]              # tracks which episode each step belongs to

    var ptr: Int   # next write position
    var size: Int  # current number of stored transitions
    var current_episode: Int  # incremented on each done

    fn __init__(out self):
        """Initialize empty sequence replay buffer."""
        self.obs = List[Scalar[Self.dtype]](capacity=Self.capacity * Self.OBS_DIM)
        self.actions = List[Scalar[Self.dtype]](
            capacity=Self.capacity * Self.ACTION_DIM
        )
        self.rewards = List[Scalar[Self.dtype]](capacity=Self.capacity)
        self.dones = List[Scalar[Self.dtype]](capacity=Self.capacity)
        self.episode_id = List[Int](capacity=Self.capacity)

        # Pre-allocate
        for _ in range(Self.capacity * Self.OBS_DIM):
            self.obs.append(Scalar[Self.dtype](0))
        for _ in range(Self.capacity * Self.ACTION_DIM):
            self.actions.append(Scalar[Self.dtype](0))
        for _ in range(Self.capacity):
            self.rewards.append(Scalar[Self.dtype](0))
            self.dones.append(Scalar[Self.dtype](0))
            self.episode_id.append(0)

        self.ptr = 0
        self.size = 0
        self.current_episode = 0

    fn add(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.OBS_DIM],
        action: InlineArray[Scalar[Self.dtype], Self.ACTION_DIM],
        reward: Scalar[Self.dtype],
        done: Bool,
    ):
        """Add a transition to the buffer.

        Args:
            obs: Current observation [OBS_DIM].
            action: Action taken [ACTION_DIM].
            reward: Reward received.
            done: Whether the episode ended after this step.
        """
        var p = self.ptr

        # Write observation
        for i in range(Self.OBS_DIM):
            self.obs[p * Self.OBS_DIM + i] = obs[i]

        # Write action
        for i in range(Self.ACTION_DIM):
            self.actions[p * Self.ACTION_DIM + i] = action[i]

        # Write reward and done
        self.rewards[p] = reward
        self.dones[p] = Scalar[Self.dtype](1.0 if done else 0.0)
        self.episode_id[p] = self.current_episode

        # Advance pointer
        self.ptr = (p + 1) % Self.capacity
        if self.size < Self.capacity:
            self.size += 1

        # New episode starts after done
        if done:
            self.current_episode += 1

    fn _is_valid_sequence_start(self, start: Int, horizon: Int) -> Bool:
        """Check whether a sequence of length horizon starting at 'start' is valid.

        A sequence is valid if:
        1. The buffer has at least horizon+1 steps from this start (wrap-around aware).
        2. No episode boundary occurs within steps [start, start+horizon-2] (done=True
           at step t means obs[t+1] belongs to a new episode, making steps t+1..H
           invalid for the sequence starting at 'start').

        Args:
            start: Starting index in the circular buffer.
            horizon: Planning horizon H (we need H+1 observations).

        Returns:
            True if the sequence starting at 'start' is valid.
        """
        # Need horizon+1 observations → horizon transitions
        # Check episode boundaries at steps [start, start+horizon-2]
        # If dones[start+t] = 1, then obs[start+t+1] is a new episode
        # → invalid for t in [0, horizon-2]
        for t in range(horizon - 1):
            var idx = (start + t) % Self.capacity
            if Float64(self.dones[idx]) > 0.5:
                return False
        return True

    fn is_ready[min_size: Int](self) -> Bool:
        """Check if the buffer has enough samples."""
        return self.size >= min_size

    fn len(self) -> Int:
        """Return the current number of stored transitions."""
        return self.size

    fn sample_sequences[
        BATCH: Int, H: Int
    ](
        self,
        mut batch_obs: List[Scalar[Self.dtype]],
        mut batch_actions: List[Scalar[Self.dtype]],
        mut batch_rewards: List[Scalar[Self.dtype]],
        mut batch_dones: List[Scalar[Self.dtype]],
    ):
        """Sample BATCH sequences of length H from the buffer.

        Each sequence provides:
          - H+1 observations: obs[t], obs[t+1], ..., obs[t+H]
          - H actions: action[t], ..., action[t+H-1]
          - H rewards: reward[t], ..., reward[t+H-1]
          - H done flags: done[t], ..., done[t+H-1]

        Episode boundaries are respected: no sequence starts within H-1 steps
        of a done=True flag.

        Output shapes:
          batch_obs:     [BATCH * (H+1) * OBS_DIM]
          batch_actions: [BATCH * H * ACTION_DIM]
          batch_rewards: [BATCH * H]
          batch_dones:   [BATCH * H]

        Args:
            batch_obs: Pre-allocated output buffer for observations.
            batch_actions: Pre-allocated output buffer for actions.
            batch_rewards: Pre-allocated output buffer for rewards.
            batch_dones: Pre-allocated output buffer for dones.
        """
        var sampled = 0
        var max_attempts = BATCH * 100  # prevent infinite loop
        var attempts = 0

        while sampled < BATCH and attempts < max_attempts:
            attempts += 1
            # Random starting index within valid range
            var start = Int(
                random_float64() * Float64(self.size)
            ) % self.size

            # Adjust for circular buffer: start should be a valid past index
            var actual_start = (self.ptr - self.size + start) % Self.capacity
            if actual_start < 0:
                actual_start += Self.capacity

            # Check that we have enough future steps
            # We need H+1 steps: actual_start, actual_start+1, ..., actual_start+H
            if self.size < H + 1:
                continue

            # Check sequence validity (no boundary in first H-1 steps)
            if not self._is_valid_sequence_start(actual_start, H):
                continue

            # Also verify that we don't step past the end of recorded data
            var end_idx = (actual_start + H) % Self.capacity
            # The end must be within the valid recorded range
            # (a valid index is in [ptr - size, ptr) mod capacity)
            var end_age = (self.ptr - end_idx - 1 + Self.capacity) % Self.capacity
            if end_age >= self.size:
                continue  # end_idx is beyond recorded data

            # Copy the sequence
            var b = sampled
            var obs_off = b * (H + 1) * Self.OBS_DIM
            var act_off = b * H * Self.ACTION_DIM
            var rew_off = b * H
            var don_off = b * H

            for t in range(H + 1):
                var idx = (actual_start + t) % Self.capacity
                var obs_start = obs_off + t * Self.OBS_DIM
                for i in range(Self.OBS_DIM):
                    batch_obs[obs_start + i] = self.obs[idx * Self.OBS_DIM + i]

            for t in range(H):
                var idx = (actual_start + t) % Self.capacity
                var act_start = act_off + t * Self.ACTION_DIM
                for i in range(Self.ACTION_DIM):
                    batch_actions[act_start + i] = self.actions[
                        idx * Self.ACTION_DIM + i
                    ]
                batch_rewards[rew_off + t] = self.rewards[idx]
                batch_dones[don_off + t] = self.dones[idx]

            sampled += 1

        # If we couldn't fill the batch (rare at start), fill remaining with zeros
        # This should not happen in practice after warmup
        for b in range(sampled, BATCH):
            var obs_off = b * (H + 1) * Self.OBS_DIM
            var act_off = b * H * Self.ACTION_DIM
            var rew_off = b * H
            var don_off = b * H
            for i in range((H + 1) * Self.OBS_DIM):
                batch_obs[obs_off + i] = Scalar[Self.dtype](0)
            for i in range(H * Self.ACTION_DIM):
                batch_actions[act_off + i] = Scalar[Self.dtype](0)
            for i in range(H):
                batch_rewards[rew_off + i] = Scalar[Self.dtype](0)
                batch_dones[don_off + i] = Scalar[Self.dtype](0)
