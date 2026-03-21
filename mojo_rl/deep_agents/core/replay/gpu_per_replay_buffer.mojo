"""GPU Prioritized Experience Replay buffer.

Hybrid CPU/GPU implementation:
- Sum-tree lives on CPU (inherently serial O(log n) traversal)
- Transition data lives on GPU (parallel gather)
- Sampling: CPU priority-based index selection → GPU data gather
- Priority update: GPU→CPU TD error transfer → CPU tree update

Store: GPU kernels write transitions + CPU updates tree with max_priority.
Sample: CPU stratified sampling → indices/weights copied to GPU → GPU gather.
Update: GPU TD errors → CPU priority updates.

Usage:
    var rb = GPUPrioritizedReplayBuffer[100000, 4](ctx, alpha=0.6, beta=0.4)

    # Store N_ENVS transitions (same as GPUReplayBuffer)
    rb.store[n_envs](ctx, prev_obs, actions, rewards, obs, dones)

    # Sample with priorities (returns IS weights)
    rb.sample[batch](ctx, sampled_obs, sampled_act, sampled_rew,
                      sampled_nobs, sampled_done, indices, weights)

    # After computing TD errors on GPU, update priorities
    rb.update_priorities[batch](ctx, td_errors_buf)
"""

from mojo_rl.nn.constants import dtype, TPB
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import random_float64
from std.math import abs as math_abs
from layout import Layout, LayoutTensor

from ..kernels import (
    store_obs_parallel_kernel,
    gather_obs_parallel_kernel,
    gather_scalars_kernel,
)


struct GPUPrioritizedReplayBuffer[
    CAPACITY: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int = 1,
    BATCH_SIZE: Int = 64,
](Movable):
    """GPU-resident replay buffer with CPU-side prioritized sampling.

    Parameters:
        CAPACITY: Maximum transitions.
        OBS_DIM: Observation dimension.
        ACTION_DIM: Action dimension (default: 1 for discrete).
        BATCH_SIZE: Fixed batch size for pre-allocated host buffers.
    """

    # GPU data buffers
    var states_buf: DeviceBuffer[dtype]
    var actions_buf: DeviceBuffer[dtype]
    var rewards_buf: DeviceBuffer[dtype]
    var next_states_buf: DeviceBuffer[dtype]
    var dones_buf: DeviceBuffer[dtype]

    # CPU-side sum-tree
    var tree: List[Scalar[dtype]]

    # CPU-side tracking
    var write_idx: Int
    var size: Int
    var alpha: Scalar[dtype]
    var beta: Scalar[dtype]
    var epsilon: Scalar[dtype]
    var max_priority: Scalar[dtype]

    # Pre-allocated host buffers for CPU↔GPU transfer
    var host_indices: HostBuffer[DType.int32]
    var host_weights: HostBuffer[dtype]
    var host_td_errors: HostBuffer[dtype]

    # Device buffers for weights (copied from host after sampling)
    var dev_weights: DeviceBuffer[dtype]

    fn __init__(
        out self,
        ctx: DeviceContext,
        alpha: Float64 = 0.6,
        beta: Float64 = 0.4,
        epsilon: Float64 = 1e-6,
    ) raises:
        """Allocate GPU data buffers and CPU sum-tree.

        Args:
            ctx: GPU device context.
            alpha: Priority exponent (0=uniform, 1=full prioritization).
            beta: IS correction exponent (annealed from initial to 1.0).
            epsilon: Small constant for non-zero priority.
        """
        # GPU data buffers
        self.states_buf = ctx.enqueue_create_buffer[dtype](
            Self.CAPACITY * Self.OBS_DIM
        )
        self.actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.CAPACITY * Self.ACTION_DIM
        )
        self.rewards_buf = ctx.enqueue_create_buffer[dtype](Self.CAPACITY)
        self.next_states_buf = ctx.enqueue_create_buffer[dtype](
            Self.CAPACITY * Self.OBS_DIM
        )
        self.dones_buf = ctx.enqueue_create_buffer[dtype](Self.CAPACITY)
        ctx.enqueue_memset(self.states_buf, 0)
        ctx.enqueue_memset(self.actions_buf, 0)
        ctx.enqueue_memset(self.rewards_buf, 0)
        ctx.enqueue_memset(self.next_states_buf, 0)
        ctx.enqueue_memset(self.dones_buf, 0)

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

        # Pre-allocated host buffers
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

    fn __init__(out self, *, deinit take: Self):
        self.states_buf = take.states_buf^
        self.actions_buf = take.actions_buf^
        self.rewards_buf = take.rewards_buf^
        self.next_states_buf = take.next_states_buf^
        self.dones_buf = take.dones_buf^
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
    # Sum-tree helpers (CPU)
    # =========================================================================

    fn _leaf_to_tree_idx(self, leaf_idx: Int) -> Int:
        return leaf_idx + Self.CAPACITY - 1

    fn _tree_to_leaf_idx(self, tree_idx: Int) -> Int:
        return tree_idx - Self.CAPACITY + 1

    fn _propagate_up(mut self, mut idx: Int, change: Scalar[dtype]):
        while idx > 0:
            var parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    fn _update_tree(mut self, leaf_idx: Int, priority: Scalar[dtype]):
        var tree_idx = self._leaf_to_tree_idx(leaf_idx)
        var change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate_up(tree_idx, change)

    fn _sample_tree(self, target: Scalar[dtype]) -> Int:
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

    fn _total_priority(self) -> Scalar[dtype]:
        return self.tree[0]

    fn _min_priority(self) -> Scalar[dtype]:
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

    fn is_ready[BATCH: Int](self) -> Bool:
        return self.size >= BATCH

    fn set_beta(mut self, beta: Scalar[dtype]):
        self.beta = beta

    fn anneal_beta(
        mut self,
        progress: Scalar[dtype],
        beta_start: Scalar[dtype] = 0.4,
    ):
        self.beta = beta_start + progress * (Scalar[dtype](1.0) - beta_start)

    # =========================================================================
    # Store (GPU data + CPU tree update)
    # =========================================================================

    fn store[
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
        """Store N_ENVS transitions: GPU kernels for data, CPU for tree."""
        var states_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
        ](states.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](rewards.unsafe_ptr())
        var next_states_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
        ](next_states.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](dones.unsafe_ptr())

        var buf_states_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY, Self.OBS_DIM), MutAnyOrigin
        ](self.states_buf.unsafe_ptr())
        var buf_next_states_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY, Self.OBS_DIM), MutAnyOrigin
        ](self.next_states_buf.unsafe_ptr())
        var buf_rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.rewards_buf.unsafe_ptr())
        var buf_dones_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.dones_buf.unsafe_ptr())

        var write_idx_s = Scalar[DType.int32](self.write_idx)
        comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime OBS_BLOCKS = (Self.OBS_DIM + TPB - 1) // TPB

        # Store obs/next_obs in parallel
        @always_inline
        fn store_obs_wrapper(
            s: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
            ],
            ns: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
            ],
            bs: LayoutTensor[
                dtype,
                Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            bns: LayoutTensor[
                dtype,
                Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            widx: Scalar[DType.int32],
        ):
            store_obs_parallel_kernel[
                dtype, N_ENVS, Self.OBS_DIM, Self.CAPACITY
            ](s, ns, bs, bns, widx)

        ctx.enqueue_function[store_obs_wrapper, store_obs_wrapper](
            states_t,
            next_states_t,
            buf_states_t,
            buf_next_states_t,
            write_idx_s,
            grid_dim=(OBS_BLOCKS, N_ENVS),
            block_dim=(TPB,),
        )

        # Store actions/rewards/dones
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions.unsafe_ptr())
        var buf_actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.actions_buf.unsafe_ptr())

        @always_inline
        fn store_scalars_wrapper(
            a: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            r: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            d: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            ba: LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ],
            br: LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ],
            bd: LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ],
            widx: Scalar[DType.int32],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= N_ENVS:
                return
            var buf_idx = (Int(widx) + i) % Self.CAPACITY
            ba[buf_idx] = a[i]
            br[buf_idx] = r[i]
            bd[buf_idx] = d[i]

        ctx.enqueue_function[store_scalars_wrapper, store_scalars_wrapper](
            actions_t,
            rewards_t,
            dones_t,
            buf_actions_t,
            buf_rewards_t,
            buf_dones_t,
            write_idx_s,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # CPU: update sum-tree with max_priority for new transitions
        var priority = self.max_priority**self.alpha
        for e in range(N_ENVS):
            var leaf_idx = (self.write_idx + e) % Self.CAPACITY
            self._update_tree(leaf_idx, priority)

        # Update CPU-side tracking
        self.write_idx = (self.write_idx + N_ENVS) % Self.CAPACITY
        self.size = min(self.size + N_ENVS, Self.CAPACITY)

    # =========================================================================
    # Sample (CPU priority sampling + GPU data gather)
    # =========================================================================

    fn sample[
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
        """Priority-based sampling: CPU tree → GPU data gather.

        Args:
            ctx: GPU device context.
            sampled_obs: Output obs [BATCH * OBS_DIM].
            sampled_actions: Output actions [BATCH].
            sampled_rewards: Output rewards [BATCH].
            sampled_next_obs: Output next_obs [BATCH * OBS_DIM].
            sampled_dones: Output dones [BATCH].
            indices: Output indices [BATCH] for priority updates.
            weights: Output IS weights [BATCH].
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
            var target = low + Scalar[dtype](
                random_float64()
            ) * (high - low)

            var idx = self._sample_tree(target)
            self.host_indices[b] = Int32(idx)

            # IS weight
            var priority = self.tree[self._leaf_to_tree_idx(idx)]
            var prob = priority / total_priority
            var weight = (
                (Scalar[dtype](self.size) * prob) ** (-self.beta)
            ) / max_weight
            self.host_weights[b] = weight

        # --- CPU→GPU: copy indices and weights ---
        ctx.enqueue_copy(indices, self.host_indices)
        ctx.enqueue_copy(weights, self.host_weights)

        # --- GPU: gather transitions using indices ---
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime OBS_BLOCKS = (Self.OBS_DIM + TPB - 1) // TPB

        var indices_t = LayoutTensor[
            DType.int32, Layout.row_major(BATCH), MutAnyOrigin
        ](indices.unsafe_ptr())

        var sampled_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ](sampled_obs.unsafe_ptr())
        var sampled_next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ](sampled_next_obs.unsafe_ptr())
        var buf_states_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY, Self.OBS_DIM), MutAnyOrigin
        ](self.states_buf.unsafe_ptr())
        var buf_next_states_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY, Self.OBS_DIM), MutAnyOrigin
        ](self.next_states_buf.unsafe_ptr())

        # Gather obs/next_obs (2D parallel kernel)
        @always_inline
        fn gather_obs_wrapper(
            bs: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
            ],
            bns: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
            ],
            rbs: LayoutTensor[
                dtype,
                Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            rbns: LayoutTensor[
                dtype,
                Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            idx: LayoutTensor[
                DType.int32, Layout.row_major(BATCH), MutAnyOrigin
            ],
        ):
            gather_obs_parallel_kernel[
                dtype, BATCH, Self.OBS_DIM, Self.CAPACITY
            ](bs, bns, rbs, rbns, idx)

        ctx.enqueue_function[gather_obs_wrapper, gather_obs_wrapper](
            sampled_obs_t,
            sampled_next_obs_t,
            buf_states_t,
            buf_next_states_t,
            indices_t,
            grid_dim=(OBS_BLOCKS, BATCH),
            block_dim=(TPB,),
        )

        # Gather scalars (actions, rewards, dones)
        var sampled_actions_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](sampled_actions.unsafe_ptr())
        var sampled_rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](sampled_rewards.unsafe_ptr())
        var sampled_dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](sampled_dones.unsafe_ptr())
        var buf_actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.actions_buf.unsafe_ptr())
        var buf_rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.rewards_buf.unsafe_ptr())
        var buf_dones_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.dones_buf.unsafe_ptr())

        @always_inline
        fn gather_sc_wrapper(
            ba: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            br: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            bd: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            rba: LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ],
            rbr: LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ],
            rbd: LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ],
            idx: LayoutTensor[
                DType.int32, Layout.row_major(BATCH), MutAnyOrigin
            ],
        ):
            gather_scalars_kernel[dtype, BATCH, Self.CAPACITY](
                ba, br, bd, rba, rbr, rbd, idx
            )

        ctx.enqueue_function[gather_sc_wrapper, gather_sc_wrapper](
            sampled_actions_t,
            sampled_rewards_t,
            sampled_dones_t,
            buf_actions_t,
            buf_rewards_t,
            buf_dones_t,
            indices_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Priority update (GPU→CPU TD errors → CPU tree update)
    # =========================================================================

    fn update_priorities[
        BATCH: Int
    ](
        mut self,
        ctx: DeviceContext,
        td_errors_buf: DeviceBuffer[dtype],
    ) raises:
        """Update priorities from GPU TD errors.

        1. Copy TD errors GPU→CPU
        2. Synchronize to ensure data is ready
        3. Update sum-tree on CPU

        Args:
            ctx: GPU device context.
            td_errors_buf: TD errors computed on GPU [BATCH].
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

    fn gpu_store[
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

    fn gpu_buffer_is_ready(self) -> Bool:
        return self.size >= Self.BATCH_SIZE
