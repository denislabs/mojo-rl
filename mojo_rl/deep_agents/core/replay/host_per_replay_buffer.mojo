"""Host-memory Prioritized Experience Replay buffer for GPU training.

Keeps transition data in host (pinned) memory instead of GPU device memory,
enabling much larger buffer capacities for large-observation environments
(e.g., pixel-based with 28K+ floats per transition).

Data flow:
- Store: GPU env outputs → copy to host staging → CPU ring buffer write + tree update
- Sample: CPU priority sampling → CPU gather from host ring → copy batch to GPU
- Priority update: GPU TD errors → CPU tree update (same as GPU variant)

The per-step overhead is one host↔device batch copy (~7 MB for 32×28K pixel obs),
which is negligible compared to the GPU forward/backward passes.

Usage:
    var rb = HostPrioritizedReplayBuffer[100000, 28224, 1, 32, 64](
        ctx, alpha=0.6, beta=0.4
    )
    rb.store[n_envs](ctx, prev_obs, actions, rewards, obs, dones)
    rb.sample[batch](ctx, s_obs, s_act, s_rew, s_nobs, s_done, indices, weights)
    rb.update_priorities[batch](ctx, td_errors_buf)
"""

from mojo_rl.nn.constants import dtype
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import random_float64

from .nstep_buffer import GPUReplayBufferStorable


struct HostPrioritizedReplayBuffer[
    CAPACITY: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int = 1,
    BATCH_SIZE: Int = 64,
    MAX_N_ENVS: Int = 256,
](Movable & GPUReplayBufferStorable):
    """Host-memory replay buffer with CPU-side prioritized sampling.

    Transition data is stored in HostBuffers (pinned memory) instead of
    DeviceBuffers, allowing large capacities without GPU OOM. Sampled
    batches are copied to GPU DeviceBuffers for training.

    Parameters:
        CAPACITY: Maximum transitions.
        OBS_DIM: Observation dimension.
        ACTION_DIM: Action dimension (default: 1 for discrete).
        BATCH_SIZE: Fixed batch size for pre-allocated staging buffers.
        MAX_N_ENVS: Maximum parallel environments (for store staging buffers).
    """

    # Host-resident ring buffers (the large data)
    var states_buf: HostBuffer[dtype]
    var actions_buf: HostBuffer[dtype]
    var rewards_buf: HostBuffer[dtype]
    var next_states_buf: HostBuffer[dtype]
    var dones_buf: HostBuffer[dtype]

    # Store staging: GPU env outputs copied here before CPU ring write
    var store_host_obs: HostBuffer[dtype]
    var store_host_nobs: HostBuffer[dtype]
    var store_host_act: HostBuffer[dtype]
    var store_host_rew: HostBuffer[dtype]
    var store_host_done: HostBuffer[dtype]

    # Sample staging: CPU-gathered batch copied to GPU from here
    var sample_host_obs: HostBuffer[dtype]
    var sample_host_nobs: HostBuffer[dtype]
    var sample_host_act: HostBuffer[dtype]
    var sample_host_rew: HostBuffer[dtype]
    var sample_host_done: HostBuffer[dtype]

    # CPU-side sum-tree
    var tree: List[Scalar[dtype]]

    # CPU-side tracking
    var write_idx: Int
    var size: Int
    var alpha: Scalar[dtype]
    var beta: Scalar[dtype]
    var epsilon: Scalar[dtype]
    var max_priority: Scalar[dtype]

    # Pre-allocated host buffers for indices/weights/td_errors
    var host_indices: HostBuffer[DType.int32]
    var host_weights: HostBuffer[dtype]
    var host_td_errors: HostBuffer[dtype]

    # Device buffer for weights (copied from host after sampling)
    var dev_weights: DeviceBuffer[dtype]

    def __init__(
        out self,
        ctx: DeviceContext,
        alpha: Float64 = 0.6,
        beta: Float64 = 0.4,
        epsilon: Float64 = 1e-6,
    ) raises:
        """Allocate host ring buffers and CPU sum-tree.

        Args:
            ctx: GPU device context (needed for pinned host buffer allocation).
            alpha: Priority exponent (0=uniform, 1=full prioritization).
            beta: IS correction exponent (annealed from initial to 1.0).
            epsilon: Small constant for non-zero priority.
        """
        # Host-resident ring buffers
        self.states_buf = ctx.enqueue_create_host_buffer[dtype](
            Self.CAPACITY * Self.OBS_DIM
        )
        self.actions_buf = ctx.enqueue_create_host_buffer[dtype](
            Self.CAPACITY * Self.ACTION_DIM
        )
        self.rewards_buf = ctx.enqueue_create_host_buffer[dtype](Self.CAPACITY)
        self.next_states_buf = ctx.enqueue_create_host_buffer[dtype](
            Self.CAPACITY * Self.OBS_DIM
        )
        self.dones_buf = ctx.enqueue_create_host_buffer[dtype](Self.CAPACITY)

        # Store staging buffers (sized for MAX_N_ENVS)
        self.store_host_obs = ctx.enqueue_create_host_buffer[dtype](
            Self.MAX_N_ENVS * Self.OBS_DIM
        )
        self.store_host_nobs = ctx.enqueue_create_host_buffer[dtype](
            Self.MAX_N_ENVS * Self.OBS_DIM
        )
        self.store_host_act = ctx.enqueue_create_host_buffer[dtype](
            Self.MAX_N_ENVS * Self.ACTION_DIM
        )
        self.store_host_rew = ctx.enqueue_create_host_buffer[dtype](
            Self.MAX_N_ENVS
        )
        self.store_host_done = ctx.enqueue_create_host_buffer[dtype](
            Self.MAX_N_ENVS
        )

        # Sample staging buffers (sized for BATCH_SIZE)
        self.sample_host_obs = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SIZE * Self.OBS_DIM
        )
        self.sample_host_nobs = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SIZE * Self.OBS_DIM
        )
        self.sample_host_act = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SIZE * Self.ACTION_DIM
        )
        self.sample_host_rew = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SIZE
        )
        self.sample_host_done = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SIZE
        )

        # CPU sum-tree (2*CAPACITY - 1 nodes)
        var tree_size = 2 * Self.CAPACITY - 1
        self.tree = List[Scalar[dtype]](capacity=tree_size)
        for _ in range(tree_size):
            self.tree.append(Scalar[dtype](0))

        self.write_idx = 0
        self.size = 0
        self.alpha = Scalar[dtype](alpha)
        self.beta = Scalar[dtype](beta)
        self.epsilon = Scalar[dtype](epsilon)
        self.max_priority = Scalar[dtype](1.0)

        # Pre-allocated host buffers for PER bookkeeping
        self.host_indices = ctx.enqueue_create_host_buffer[DType.int32](
            Self.BATCH_SIZE
        )
        self.host_weights = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SIZE
        )
        self.host_td_errors = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH_SIZE
        )
        self.dev_weights = ctx.enqueue_create_buffer[dtype](Self.BATCH_SIZE)

    def __init__(out self, *, deinit take: Self):
        self.states_buf = take.states_buf^
        self.actions_buf = take.actions_buf^
        self.rewards_buf = take.rewards_buf^
        self.next_states_buf = take.next_states_buf^
        self.dones_buf = take.dones_buf^
        self.store_host_obs = take.store_host_obs^
        self.store_host_nobs = take.store_host_nobs^
        self.store_host_act = take.store_host_act^
        self.store_host_rew = take.store_host_rew^
        self.store_host_done = take.store_host_done^
        self.sample_host_obs = take.sample_host_obs^
        self.sample_host_nobs = take.sample_host_nobs^
        self.sample_host_act = take.sample_host_act^
        self.sample_host_rew = take.sample_host_rew^
        self.sample_host_done = take.sample_host_done^
        self.tree = take.tree^
        self.write_idx = take.write_idx
        self.size = take.size
        self.alpha = take.alpha
        self.beta = take.beta
        self.epsilon = take.epsilon
        self.max_priority = take.max_priority
        self.host_indices = take.host_indices^
        self.host_weights = take.host_weights^
        self.host_td_errors = take.host_td_errors^
        self.dev_weights = take.dev_weights^

    # =========================================================================
    # Sum-tree helpers (CPU) — identical to GPUPrioritizedReplayBuffer
    # =========================================================================

    def _leaf_to_tree_idx(self, leaf_idx: Int) -> Int:
        return leaf_idx + Self.CAPACITY - 1

    def _tree_to_leaf_idx(self, tree_idx: Int) -> Int:
        return tree_idx - Self.CAPACITY + 1

    def _propagate_up(mut self, mut idx: Int, change: Scalar[dtype]):
        while idx > 0:
            var parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    def _update_tree(mut self, leaf_idx: Int, priority: Scalar[dtype]):
        var tree_idx = self._leaf_to_tree_idx(leaf_idx)
        var change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate_up(tree_idx, change)

    def _sample_tree(self, target: Scalar[dtype]) -> Int:
        var idx = 0
        var remaining = target
        while True:
            var left = 2 * idx + 1
            var right = 2 * idx + 2
            if left >= 2 * Self.CAPACITY - 1:
                break
            if remaining <= self.tree[left]:
                idx = left
            else:
                remaining -= self.tree[left]
                idx = right
        return self._tree_to_leaf_idx(idx)

    def _total_priority(self) -> Scalar[dtype]:
        return self.tree[0]

    def _min_priority(self) -> Scalar[dtype]:
        var min_p = Scalar[dtype](1e10)
        for i in range(self.size):
            var tree_idx = self._leaf_to_tree_idx(i)
            var p = self.tree[tree_idx]
            if p > 0 and p < min_p:
                min_p = p
        return min_p if min_p < Scalar[dtype](1e10) else Scalar[dtype](1.0)

    # =========================================================================
    # Buffer state
    # =========================================================================

    def is_ready[BATCH: Int](self) -> Bool:
        return self.size >= BATCH

    def set_beta(mut self, beta: Scalar[dtype]):
        self.beta = beta

    def anneal_beta(
        mut self,
        progress: Scalar[dtype],
        beta_start: Scalar[dtype] = 0.4,
    ):
        self.beta = beta_start + progress * (Scalar[dtype](1.0) - beta_start)

    # =========================================================================
    # Store (GPU→Host copy + CPU ring write + CPU tree update)
    # =========================================================================

    def store[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        rewards: DeviceBuffer[dtype],
        next_states: DeviceBuffer[dtype],
        dones: DeviceBuffer[dtype],
    ) raises:
        """Store N_ENVS transitions: copy GPU→host, write to ring, update tree.
        """
        # 1. GPU → Host staging copy
        ctx.enqueue_copy(self.store_host_obs, states)
        ctx.enqueue_copy(self.store_host_nobs, next_states)
        ctx.enqueue_copy(self.store_host_act, actions)
        ctx.enqueue_copy(self.store_host_rew, rewards)
        ctx.enqueue_copy(self.store_host_done, dones)
        ctx.synchronize()

        # 2. CPU: copy from staging into ring buffer at write_idx
        for e in range(N_ENVS):
            var buf_idx = (self.write_idx + e) % Self.CAPACITY
            # Observations
            var src_obs_base = e * Self.OBS_DIM
            var dst_obs_base = buf_idx * Self.OBS_DIM
            for d in range(Self.OBS_DIM):
                self.states_buf[dst_obs_base + d] = self.store_host_obs[
                    src_obs_base + d
                ]
                self.next_states_buf[dst_obs_base + d] = self.store_host_nobs[
                    src_obs_base + d
                ]
            # Actions
            var src_act_base = e * Self.ACTION_DIM
            var dst_act_base = buf_idx * Self.ACTION_DIM
            for d in range(Self.ACTION_DIM):
                self.actions_buf[dst_act_base + d] = self.store_host_act[
                    src_act_base + d
                ]
            # Scalars
            self.rewards_buf[buf_idx] = self.store_host_rew[e]
            self.dones_buf[buf_idx] = self.store_host_done[e]

        # 3. CPU: update sum-tree with max_priority for new transitions
        var priority = self.max_priority**self.alpha
        for e in range(N_ENVS):
            var leaf_idx = (self.write_idx + e) % Self.CAPACITY
            self._update_tree(leaf_idx, priority)

        # Update CPU-side tracking
        self.write_idx = (self.write_idx + N_ENVS) % Self.CAPACITY
        self.size = min(self.size + N_ENVS, Self.CAPACITY)

    # =========================================================================
    # Sample (CPU priority sampling + CPU gather + Host→GPU copy)
    # =========================================================================

    def sample[
        BATCH: Int
    ](
        mut self,
        ctx: DeviceContext,
        sampled_obs: DeviceBuffer[dtype],
        sampled_actions: DeviceBuffer[dtype],
        sampled_rewards: DeviceBuffer[dtype],
        sampled_next_obs: DeviceBuffer[dtype],
        sampled_dones: DeviceBuffer[dtype],
        indices: DeviceBuffer[DType.int32],
        weights: DeviceBuffer[dtype],
    ) raises:
        """Priority-based sampling: CPU tree → CPU gather → Host→GPU copy.

        Args:
            ctx: GPU device context.
            sampled_obs: Output obs [BATCH * OBS_DIM] on GPU.
            sampled_actions: Output actions [BATCH] on GPU.
            sampled_rewards: Output rewards [BATCH] on GPU.
            sampled_next_obs: Output next_obs [BATCH * OBS_DIM] on GPU.
            sampled_dones: Output dones [BATCH] on GPU.
            indices: Output indices [BATCH] for priority updates.
            weights: Output IS weights [BATCH] on GPU.
        """
        # --- CPU: stratified priority sampling ---
        var total_priority = self._total_priority()
        var segment_size = total_priority / Scalar[dtype](BATCH)

        # Min probability for weight normalization
        var min_prob = self._min_priority() / total_priority
        var max_weight = (Scalar[dtype](self.size) * min_prob) ** (-self.beta)

        for b in range(BATCH):
            var low = segment_size * Scalar[dtype](b)
            var high = segment_size * Scalar[dtype](b + 1)
            var target = low + Scalar[dtype](random_float64()) * (high - low)

            var idx = self._sample_tree(target)
            self.host_indices[b] = Int32(idx)

            # IS weight
            var priority = self.tree[self._leaf_to_tree_idx(idx)]
            var prob = priority / total_priority
            var weight = (
                (Scalar[dtype](self.size) * prob) ** (-self.beta)
            ) / max_weight
            self.host_weights[b] = weight

        # --- CPU: gather from host ring buffer into staging ---
        for b in range(BATCH):
            var idx = Int(self.host_indices[b])
            # Observations
            var src_obs_base = idx * Self.OBS_DIM
            var dst_obs_base = b * Self.OBS_DIM
            for d in range(Self.OBS_DIM):
                self.sample_host_obs[dst_obs_base + d] = self.states_buf[
                    src_obs_base + d
                ]
                self.sample_host_nobs[dst_obs_base + d] = self.next_states_buf[
                    src_obs_base + d
                ]
            # Actions
            var src_act_base = idx * Self.ACTION_DIM
            var dst_act_base = b * Self.ACTION_DIM
            for d in range(Self.ACTION_DIM):
                self.sample_host_act[dst_act_base + d] = self.actions_buf[
                    src_act_base + d
                ]
            # Scalars
            self.sample_host_rew[b] = self.rewards_buf[idx]
            self.sample_host_done[b] = self.dones_buf[idx]

        # --- Host → GPU: copy gathered batch ---
        ctx.enqueue_copy(sampled_obs, self.sample_host_obs)
        ctx.enqueue_copy(sampled_next_obs, self.sample_host_nobs)
        ctx.enqueue_copy(sampled_actions, self.sample_host_act)
        ctx.enqueue_copy(sampled_rewards, self.sample_host_rew)
        ctx.enqueue_copy(sampled_dones, self.sample_host_done)
        ctx.enqueue_copy(indices, self.host_indices)
        ctx.enqueue_copy(weights, self.host_weights)

    # =========================================================================
    # Priority update (GPU→CPU TD errors → CPU tree update)
    # =========================================================================

    def update_priorities[
        BATCH: Int
    ](mut self, ctx: DeviceContext, td_errors_buf: DeviceBuffer[dtype],) raises:
        """Update priorities from GPU TD errors.

        1. Copy TD errors GPU→CPU
        2. Synchronize to ensure data is ready
        3. Update sum-tree on CPU
        """
        # GPU→CPU copy
        ctx.enqueue_copy(self.host_td_errors, td_errors_buf)
        ctx.synchronize()

        # CPU: update priorities
        for b in range(BATCH):
            var idx = Int(self.host_indices[b])
            var td_error = self.host_td_errors[b]
            var abs_error = td_error if td_error > 0 else -td_error
            var raw_priority = abs_error + self.epsilon
            var priority = raw_priority**self.alpha
            self._update_tree(idx, priority)
            if raw_priority > self.max_priority:
                self.max_priority = raw_priority

    # =========================================================================
    # GPUOffPolicyState compatibility
    # =========================================================================

    def gpu_store[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        rewards_buf: DeviceBuffer[dtype],
        obs_buf: DeviceBuffer[dtype],
        dones_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """GPUOffPolicyState-compatible store."""
        self.store[N_ENVS](
            ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, dones_buf
        )

    def gpu_buffer_is_ready(self) -> Bool:
        return self.size >= Self.BATCH_SIZE
