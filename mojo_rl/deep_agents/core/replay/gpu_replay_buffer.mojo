"""GPUReplayBuffer: device-side circular replay buffer for GPU RL training.

Encapsulates five DeviceBuffers (states, actions, rewards, next_states, dones)
plus CPU-side write_idx and size tracking.  Provides store[N_ENVS] and
sample[BATCH] methods that define kernel wrappers internally, keeping callers
(DQN, DDPG, TD3, SAC GPU training loops) free of replay-buffer boilerplate.

ACTION_DIM defaults to 1 (scalar action) for backward compatibility with DQN.
Set ACTION_DIM > 1 for continuous control agents (DDPG, TD3, SAC).

Usage:
    # Discrete action (DQN) — ACTION_DIM=1 default
    var rb = GPUReplayBuffer[buffer_capacity, obs_dim](ctx)

    # Continuous action (DDPG/TD3/SAC)
    var rb = GPUReplayBuffer[buffer_capacity, obs_dim, action_dim](ctx)

    # Store N_ENVS new transitions each step
    rb.store[n_envs](ctx, prev_obs_buf, actions_buf, rewards_buf, obs_buf, dones_buf)

    # Sample a training batch
    var indices_buf = ctx.enqueue_create_buffer[DType.int32](batch_size)
    rb.sample[batch_size](ctx, rng_seed,
                          sampled_obs_buf, sampled_actions_buf,
                          sampled_rewards_buf, sampled_next_obs_buf,
                          sampled_dones_buf, indices_buf)
"""

from mojo_rl.nn.constants import dtype, TPB
from .replay_buffer import HeapReplayBuffer
from std.gpu import block_dim, block_idx, thread_idx
from ..kernels import (
    store_transitions_kernel,
    store_obs_parallel_kernel,
    sample_indices_kernel,
    gather_batch_kernel,
    gather_obs_parallel_kernel,
    gather_scalars_kernel,
    gather_scalars_nd_kernel,
    store_transitions_kernel_nd,
    gather_batch_kernel_nd,
    ere_update_kernel,
    sample_indices_ere_kernel,
)

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer


from .nstep_buffer import GPUReplayBufferStorable


struct GPUReplayBuffer[CAPACITY: Int, OBS_DIM: Int, ACTION_DIM: Int = 1](
    Movable & GPUReplayBufferStorable
):
    """GPU-resident circular replay buffer.

    Five DeviceBuffers hold the circular store; write_idx and size are
    tracked on CPU (updated after each store call).

    Supports both scalar (ACTION_DIM=1, default) and multi-dimensional
    (ACTION_DIM>1) actions, enabling use by DQN (scalar) and DDPG/TD3/SAC
    (continuous vector) agents with the same interface.

    Parameters:
        CAPACITY: Maximum number of transitions in the buffer.
        OBS_DIM: Observation dimension.
        ACTION_DIM: Action dimension (default: 1 for scalar/discrete actions).
    """

    var states_buf: DeviceBuffer[dtype]
    var actions_buf: DeviceBuffer[dtype]
    var rewards_buf: DeviceBuffer[dtype]
    var next_states_buf: DeviceBuffer[dtype]
    var dones_buf: DeviceBuffer[dtype]
    var write_idx: Int
    var size: Int
    var gpu_size: DeviceBuffer[DType.int32]  # GPU-side size for graph capture
    var gpu_write_idx: DeviceBuffer[DType.int32]  # GPU-side write index for env graph capture
    # ERE (Emphasizing Recent Experience) — opt-in recency-biased sampling.
    # ere_state: [k counter, eta_pow_k]. ere_c: current c_k.
    var ere_enabled: Bool
    var ere_eta: Float32
    var ere_state: DeviceBuffer[dtype]
    var ere_c: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate all device buffers and zero-initialize.

        Args:
            ctx: GPU device context.
        """
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
        self.write_idx = 0
        self.size = 0
        self.gpu_size = ctx.enqueue_create_buffer[DType.int32](1)
        self.gpu_size.enqueue_fill(Scalar[DType.int32](0))
        self.gpu_write_idx = ctx.enqueue_create_buffer[DType.int32](1)
        self.gpu_write_idx.enqueue_fill(Scalar[DType.int32](0))
        # ERE state — disabled by default; ere_c defaults to CAPACITY (no bias).
        # ere_state[0] = k counter, ere_state[1] = eta^k (init 1.0).
        self.ere_enabled = False
        self.ere_eta = Float32(0.996)
        self.ere_state = ctx.enqueue_create_buffer[dtype](2)
        var ere_host = ctx.enqueue_create_host_buffer[dtype](2)
        ere_host[0] = Scalar[dtype](0.0)
        ere_host[1] = Scalar[dtype](1.0)
        ctx.enqueue_copy(self.ere_state, ere_host)
        self.ere_c = ctx.enqueue_create_buffer[DType.int32](1)
        self.ere_c.enqueue_fill(Scalar[DType.int32](Self.CAPACITY))

    def __init__(out self, *, deinit take: Self):
        self.states_buf = take.states_buf^
        self.actions_buf = take.actions_buf^
        self.rewards_buf = take.rewards_buf^
        self.next_states_buf = take.next_states_buf^
        self.dones_buf = take.dones_buf^
        self.write_idx = take.write_idx
        self.size = take.size
        self.gpu_size = take.gpu_size^
        self.gpu_write_idx = take.gpu_write_idx^
        self.ere_enabled = take.ere_enabled
        self.ere_eta = take.ere_eta
        self.ere_state = take.ere_state^
        self.ere_c = take.ere_c^

    def enable_ere(mut self, eta: Float32 = Float32(0.996)):
        """Enable ERE recency-biased sampling. Must be called before CUDA graph capture.

        Args:
            eta: ERE decay factor (paper default 0.996). c_k ≈ N·eta^k within
                 each K_MAX=1000 cycle; smaller eta = stronger recency bias.
        """
        self.ere_enabled = True
        self.ere_eta = eta

    def is_ready[BATCH: Int](self) -> Bool:
        """Return True if the buffer holds at least BATCH transitions."""
        return self.size >= BATCH

    def upload_from(
        mut self,
        cpu_buf: HeapReplayBuffer[
            Self.CAPACITY, Self.OBS_DIM, Self.ACTION_DIM, dtype
        ],
        ctx: DeviceContext,
    ) raises:
        """Upload CPU replay buffer contents to GPU device buffers.

        Copies all valid transitions from the CPU buffer to GPU memory.
        Call ctx.synchronize() before using the GPU buffer after this.

        Args:
            cpu_buf: CPU replay buffer with matching CAPACITY/OBS_DIM/ACTION_DIM.
            ctx: GPU device context.
        """
        if cpu_buf.size == 0:
            return

        var n = cpu_buf.size

        var h_states = ctx.enqueue_create_host_buffer[dtype](
            Self.CAPACITY * Self.OBS_DIM
        )
        for i in range(n * Self.OBS_DIM):
            h_states[i] = cpu_buf.obs[i]
        ctx.enqueue_copy(self.states_buf, h_states)

        var h_actions = ctx.enqueue_create_host_buffer[dtype](
            Self.CAPACITY * Self.ACTION_DIM
        )
        for i in range(n * Self.ACTION_DIM):
            h_actions[i] = cpu_buf.actions[i]
        ctx.enqueue_copy(self.actions_buf, h_actions)

        var h_rewards = ctx.enqueue_create_host_buffer[dtype](Self.CAPACITY)
        for i in range(n):
            h_rewards[i] = cpu_buf.rewards[i]
        ctx.enqueue_copy(self.rewards_buf, h_rewards)

        var h_next = ctx.enqueue_create_host_buffer[dtype](
            Self.CAPACITY * Self.OBS_DIM
        )
        for i in range(n * Self.OBS_DIM):
            h_next[i] = cpu_buf.next_obs[i]
        ctx.enqueue_copy(self.next_states_buf, h_next)

        var h_dones = ctx.enqueue_create_host_buffer[dtype](Self.CAPACITY)
        for i in range(n):
            h_dones[i] = cpu_buf.dones[i]
        ctx.enqueue_copy(self.dones_buf, h_dones)

        self.write_idx = cpu_buf.ptr
        self.size = cpu_buf.size
        self.gpu_size.enqueue_fill(Scalar[DType.int32](self.size))

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
        """Store N_ENVS new transitions into the circular buffer.

        One GPU kernel call writes N_ENVS slots starting at write_idx
        (wrapping around via modulo).  CPU-side write_idx and size are
        updated immediately after enqueueing.

        When ACTION_DIM == 1, uses the scalar store_transitions_kernel.
        When ACTION_DIM > 1, uses store_transitions_kernel_nd for vector actions.

        Args:
            ctx: GPU device context.
            states: Current observations [N_ENVS * OBS_DIM].
            actions: Actions taken [N_ENVS * ACTION_DIM].
            rewards: Rewards received [N_ENVS].
            next_states: Next observations [N_ENVS * OBS_DIM].
            dones: Done flags [N_ENVS].
        """
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
        var buf_rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.rewards_buf.unsafe_ptr())
        var buf_next_states_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY, Self.OBS_DIM), MutAnyOrigin
        ](self.next_states_buf.unsafe_ptr())
        var buf_dones_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.dones_buf.unsafe_ptr())

        var write_idx_s = Scalar[DType.int32](self.write_idx)

        comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime OBS_BLOCKS = (Self.OBS_DIM + TPB - 1) // TPB

        comptime if Self.ACTION_DIM == 1:
            var actions_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS), MutAnyOrigin
            ](actions.unsafe_ptr())
            var buf_actions_t = LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ](self.actions_buf.unsafe_ptr())

            # Parallel obs store: 2D grid (OBS_BLOCKS, N_ENVS)
            @parameter
            @always_inline
            def store_obs_wrapper(
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

            ctx.enqueue_function[store_obs_wrapper](
                states_t,
                next_states_t,
                buf_states_t,
                buf_next_states_t,
                write_idx_s,
                grid_dim=(OBS_BLOCKS, N_ENVS),
                block_dim=(TPB,),
            )

            # Scalar store: actions/rewards/dones (tiny, 1 thread per env)
            @parameter
            @always_inline
            def store_scalars_wrapper(
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

            ctx.enqueue_function[store_scalars_wrapper](
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
        else:
            var actions_t = LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS, Self.ACTION_DIM),
                MutAnyOrigin,
            ](actions.unsafe_ptr())
            var buf_actions_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.CAPACITY, Self.ACTION_DIM),
                MutAnyOrigin,
            ](self.actions_buf.unsafe_ptr())

            @parameter
            @always_inline
            def store_nd_wrapper(
                s: LayoutTensor[
                    dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
                ],
                a: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS, Self.ACTION_DIM),
                    MutAnyOrigin,
                ],
                r: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
                ns: LayoutTensor[
                    dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
                ],
                d: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
                bs: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                ba: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.ACTION_DIM),
                    MutAnyOrigin,
                ],
                br: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                bns: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                bd: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                widx: Scalar[DType.int32],
            ):
                store_transitions_kernel_nd[
                    dtype, N_ENVS, Self.OBS_DIM, Self.ACTION_DIM, Self.CAPACITY
                ](s, a, r, ns, d, bs, ba, br, bns, bd, widx)

            ctx.enqueue_function[store_nd_wrapper](
                states_t,
                actions_t,
                rewards_t,
                next_states_t,
                dones_t,
                buf_states_t,
                buf_actions_t,
                buf_rewards_t,
                buf_next_states_t,
                buf_dones_t,
                write_idx_s,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # Update CPU-side tracking
        self.write_idx = (self.write_idx + N_ENVS) % Self.CAPACITY
        self.size = min(self.size + N_ENVS, Self.CAPACITY)
        # Sync GPU-side counters for CUDA graph compatible sampling/storing
        self.gpu_size.enqueue_fill(Scalar[DType.int32](self.size))
        self.gpu_write_idx.enqueue_fill(Scalar[DType.int32](self.write_idx))

    def store_graph[
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
        """CUDA graph compatible store: reads write_idx from GPU memory.

        Same as store[N_ENVS] but uses gpu_write_idx DeviceBuffer instead of
        CPU-side write_idx scalar. Does NOT update CPU-side counters (caller
        must track buffer state externally).

        After storing, enqueues advance_write_idx_kernel to update gpu_write_idx
        for the next graph replay.

        Args:
            ctx: GPU device context.
            states: Current observations [N_ENVS * OBS_DIM].
            actions: Actions taken [N_ENVS * ACTION_DIM].
            rewards: Rewards received [N_ENVS].
            next_states: Next observations [N_ENVS * OBS_DIM].
            dones: Done flags [N_ENVS].
        """
        from ..kernels import advance_write_idx_kernel

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
        var buf_rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.rewards_buf.unsafe_ptr())
        var buf_next_states_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY, Self.OBS_DIM), MutAnyOrigin
        ](self.next_states_buf.unsafe_ptr())
        var buf_dones_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.dones_buf.unsafe_ptr())

        # GPU-side write index (read on device, not baked)
        var widx_t = LayoutTensor[
            DType.int32, Layout.row_major(1), MutAnyOrigin
        ](self.gpu_write_idx.unsafe_ptr())

        comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime OBS_BLOCKS = (Self.OBS_DIM + TPB - 1) // TPB

        comptime if Self.ACTION_DIM == 1:
            var actions_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS), MutAnyOrigin
            ](actions.unsafe_ptr())
            var buf_actions_t = LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ](self.actions_buf.unsafe_ptr())

            @parameter
            @always_inline
            def store_obs_graph_wrapper(
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
                widx: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
            ):
                store_obs_parallel_kernel[
                    dtype, N_ENVS, Self.OBS_DIM, Self.CAPACITY
                ](s, ns, bs, bns, rebind[Scalar[DType.int32]](widx[0]))

            ctx.enqueue_function[store_obs_graph_wrapper](
                states_t,
                next_states_t,
                buf_states_t,
                buf_next_states_t,
                widx_t,
                grid_dim=(OBS_BLOCKS, N_ENVS),
                block_dim=(TPB,),
            )

            @parameter
            @always_inline
            def store_scalars_graph_wrapper(
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
                widx: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i >= N_ENVS:
                    return
                var buf_idx = (Int(rebind[Scalar[DType.int32]](widx[0])) + i) % Self.CAPACITY
                ba[buf_idx] = a[i]
                br[buf_idx] = r[i]
                bd[buf_idx] = d[i]

            ctx.enqueue_function[store_scalars_graph_wrapper](
                actions_t,
                rewards_t,
                dones_t,
                buf_actions_t,
                buf_rewards_t,
                buf_dones_t,
                widx_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            var actions_t = LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS, Self.ACTION_DIM),
                MutAnyOrigin,
            ](actions.unsafe_ptr())
            var buf_actions_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.CAPACITY, Self.ACTION_DIM),
                MutAnyOrigin,
            ](self.actions_buf.unsafe_ptr())

            @parameter
            @always_inline
            def store_nd_graph_wrapper(
                s: LayoutTensor[
                    dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
                ],
                a: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS, Self.ACTION_DIM),
                    MutAnyOrigin,
                ],
                r: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
                ns: LayoutTensor[
                    dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
                ],
                d: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
                bs: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                ba: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.ACTION_DIM),
                    MutAnyOrigin,
                ],
                br: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                bns: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                bd: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                widx: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
            ):
                store_transitions_kernel_nd[
                    dtype, N_ENVS, Self.OBS_DIM, Self.ACTION_DIM, Self.CAPACITY
                ](s, a, r, ns, d, bs, ba, br, bns, bd, rebind[Scalar[DType.int32]](widx[0]))

            ctx.enqueue_function[store_nd_graph_wrapper](
                states_t,
                actions_t,
                rewards_t,
                next_states_t,
                dones_t,
                buf_states_t,
                buf_actions_t,
                buf_rewards_t,
                buf_next_states_t,
                buf_dones_t,
                widx_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # Advance GPU-side write index: (write_idx + N_ENVS) % CAPACITY
        comptime adv_k = advance_write_idx_kernel[N_ENVS, Self.CAPACITY]
        ctx.enqueue_function[adv_k](
            widx_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

        # Advance GPU-side size: min(size + N_ENVS, CAPACITY)
        from ..kernels import advance_gpu_size_kernel

        comptime size_k = advance_gpu_size_kernel[N_ENVS, Self.CAPACITY]
        var size_t = LayoutTensor[
            DType.int32, Layout.row_major(1), MutAnyOrigin
        ](self.gpu_size.unsafe_ptr())
        ctx.enqueue_function[size_k](
            size_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

    def sample[
        BATCH: Int
    ](
        self,
        ctx: DeviceContext,
        rng_counter: DeviceBuffer[DType.uint32],
        sampled_obs: DeviceBuffer[dtype],
        sampled_actions: DeviceBuffer[dtype],
        sampled_rewards: DeviceBuffer[dtype],
        sampled_next_obs: DeviceBuffer[dtype],
        sampled_dones: DeviceBuffer[dtype],
        indices: DeviceBuffer[DType.int32],
    ) raises:
        """Sample BATCH random transitions into the provided output buffers.

        Two GPU kernels are enqueued:
        1. sample_indices_kernel — fills `indices` with random positions.
        2. gather_batch_kernel / gather_batch_kernel_nd — gathers transitions.

        When ACTION_DIM == 1, sampled_actions layout is [BATCH].
        When ACTION_DIM > 1, sampled_actions layout is [BATCH * ACTION_DIM].

        Args:
            ctx: GPU device context.
            rng_counter: GPU-side RNG counter [1] (CUDA graph compatible).
            sampled_obs: Output observations [BATCH * OBS_DIM].
            sampled_actions: Output actions [BATCH * ACTION_DIM].
            sampled_rewards: Output rewards [BATCH].
            sampled_next_obs: Output next observations [BATCH * OBS_DIM].
            sampled_dones: Output done flags [BATCH].
            indices: Scratch buffer for sampled indices [BATCH] (DType.int32).
        """
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        var indices_t = LayoutTensor[
            DType.int32, Layout.row_major(BATCH), MutAnyOrigin
        ](indices.unsafe_ptr())
        var buf_size_t = LayoutTensor[
            DType.int32, Layout.row_major(1), MutAnyOrigin
        ](self.gpu_size.unsafe_ptr())
        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](rng_counter.unsafe_ptr())

        if self.ere_enabled:
            # ERE path: update c_k then sample from most-recent c_k transitions.
            var ere_state_t = LayoutTensor[
                dtype, Layout.row_major(2), MutAnyOrigin
            ](self.ere_state.unsafe_ptr())
            var ere_c_t = LayoutTensor[
                DType.int32, Layout.row_major(1), MutAnyOrigin
            ](self.ere_c.unsafe_ptr())
            var widx_t = LayoutTensor[
                DType.int32, Layout.row_major(1), MutAnyOrigin
            ](self.gpu_write_idx.unsafe_ptr())
            var eta_scalar = Scalar[dtype](self.ere_eta)

            comptime K_MAX = 1000
            comptime C_MIN = 5000
            comptime ere_update_k = ere_update_kernel[dtype, K_MAX, C_MIN]
            ctx.enqueue_function[ere_update_k](
                ere_state_t,
                buf_size_t,
                ere_c_t,
                eta_scalar,
                grid_dim=(1,),
                block_dim=(1,),
            )

            @parameter
            @always_inline
            def sample_ere_wrapper(
                idx: LayoutTensor[
                    DType.int32, Layout.row_major(BATCH), MutAnyOrigin
                ],
                bsize: LayoutTensor[
                    DType.int32, Layout.row_major(1), MutAnyOrigin
                ],
                widx: LayoutTensor[
                    DType.int32, Layout.row_major(1), MutAnyOrigin
                ],
                ere_c_arg: LayoutTensor[
                    DType.int32, Layout.row_major(1), MutAnyOrigin
                ],
                rng: LayoutTensor[
                    DType.uint32, Layout.row_major(1), MutAnyOrigin
                ],
            ):
                sample_indices_ere_kernel[dtype, BATCH, Self.CAPACITY](
                    idx, bsize, widx, ere_c_arg, rng
                )

            ctx.enqueue_function[sample_ere_wrapper](
                indices_t,
                buf_size_t,
                widx_t,
                ere_c_t,
                rng_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:

            @parameter
            @always_inline
            def sample_wrapper(
                idx: LayoutTensor[
                    DType.int32, Layout.row_major(BATCH), MutAnyOrigin
                ],
                bsize: LayoutTensor[
                    DType.int32, Layout.row_major(1), MutAnyOrigin
                ],
                rng: LayoutTensor[
                    DType.uint32, Layout.row_major(1), MutAnyOrigin
                ],
            ):
                sample_indices_kernel[dtype, BATCH](idx, bsize, rng)

            ctx.enqueue_function[sample_wrapper](
                indices_t,
                buf_size_t,
                rng_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        var sampled_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ](sampled_obs.unsafe_ptr())
        var sampled_rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](sampled_rewards.unsafe_ptr())
        var sampled_next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS_DIM), MutAnyOrigin
        ](sampled_next_obs.unsafe_ptr())
        var sampled_dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](sampled_dones.unsafe_ptr())

        var buf_states_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY, Self.OBS_DIM), MutAnyOrigin
        ](self.states_buf.unsafe_ptr())
        var buf_rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.rewards_buf.unsafe_ptr())
        var buf_next_states_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY, Self.OBS_DIM), MutAnyOrigin
        ](self.next_states_buf.unsafe_ptr())
        var buf_dones_t = LayoutTensor[
            dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
        ](self.dones_buf.unsafe_ptr())

        # Parallel gather: 2D grid (OBS_DIM_blocks, BATCH) — one thread per
        # element. Works for all OBS_DIM sizes.
        comptime if True:
            comptime OBS_BLOCKS = (Self.OBS_DIM + TPB - 1) // TPB

            @parameter
            @always_inline
            def gather_obs_wrapper(
                bs: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                bns: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OBS_DIM),
                    MutAnyOrigin,
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

            ctx.enqueue_function[gather_obs_wrapper](
                sampled_obs_t,
                sampled_next_obs_t,
                buf_states_t,
                buf_next_states_t,
                indices_t,
                grid_dim=(OBS_BLOCKS, BATCH),
                block_dim=(TPB,),
            )

            # Scalar fields (actions/rewards/dones) — tiny kernel
            comptime if Self.ACTION_DIM == 1:
                var sampled_actions_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ](sampled_actions.unsafe_ptr())
                var buf_actions_t = LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ](self.actions_buf.unsafe_ptr())

                @parameter
                @always_inline
                def gather_sc_wrapper(
                    ba: LayoutTensor[
                        dtype, Layout.row_major(BATCH), MutAnyOrigin
                    ],
                    br: LayoutTensor[
                        dtype, Layout.row_major(BATCH), MutAnyOrigin
                    ],
                    bd: LayoutTensor[
                        dtype, Layout.row_major(BATCH), MutAnyOrigin
                    ],
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

                ctx.enqueue_function[gather_sc_wrapper](
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
            else:
                var sampled_actions_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.ACTION_DIM),
                    MutAnyOrigin,
                ](sampled_actions.unsafe_ptr())
                var buf_actions_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.ACTION_DIM),
                    MutAnyOrigin,
                ](self.actions_buf.unsafe_ptr())

                @parameter
                @always_inline
                def gather_sc_nd_wrapper(
                    ba: LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.ACTION_DIM),
                        MutAnyOrigin,
                    ],
                    br: LayoutTensor[
                        dtype, Layout.row_major(BATCH), MutAnyOrigin
                    ],
                    bd: LayoutTensor[
                        dtype, Layout.row_major(BATCH), MutAnyOrigin
                    ],
                    rba: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.CAPACITY, Self.ACTION_DIM),
                        MutAnyOrigin,
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
                    gather_scalars_nd_kernel[
                        dtype, BATCH, Self.ACTION_DIM, Self.CAPACITY
                    ](ba, br, bd, rba, rbr, rbd, idx)

                ctx.enqueue_function[gather_sc_nd_wrapper](
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

        # For small OBS_DIM, use original monolithic kernel (one thread per sample)
        elif Self.ACTION_DIM == 1:
            var sampled_actions_t = LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ](sampled_actions.unsafe_ptr())
            var buf_actions_t = LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ](self.actions_buf.unsafe_ptr())

            @parameter
            @always_inline
            def gather_wrapper(
                bs: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                ba: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                br: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                bns: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                bd: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                rbs: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                rba: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                rbr: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                rbns: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                rbd: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                idx: LayoutTensor[
                    DType.int32, Layout.row_major(BATCH), MutAnyOrigin
                ],
            ):
                gather_batch_kernel[dtype, BATCH, Self.OBS_DIM, Self.CAPACITY](
                    bs, ba, br, bns, bd, rbs, rba, rbr, rbns, rbd, idx
                )

            ctx.enqueue_function[gather_wrapper](
                sampled_obs_t,
                sampled_actions_t,
                sampled_rewards_t,
                sampled_next_obs_t,
                sampled_dones_t,
                buf_states_t,
                buf_actions_t,
                buf_rewards_t,
                buf_next_states_t,
                buf_dones_t,
                indices_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            var sampled_actions_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.ACTION_DIM),
                MutAnyOrigin,
            ](sampled_actions.unsafe_ptr())
            var buf_actions_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.CAPACITY, Self.ACTION_DIM),
                MutAnyOrigin,
            ](self.actions_buf.unsafe_ptr())

            @parameter
            @always_inline
            def gather_nd_wrapper(
                bs: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                ba: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.ACTION_DIM),
                    MutAnyOrigin,
                ],
                br: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                bns: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                bd: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                rbs: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                rba: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.ACTION_DIM),
                    MutAnyOrigin,
                ],
                rbr: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                rbns: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.CAPACITY, Self.OBS_DIM),
                    MutAnyOrigin,
                ],
                rbd: LayoutTensor[
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
                ],
                idx: LayoutTensor[
                    DType.int32, Layout.row_major(BATCH), MutAnyOrigin
                ],
            ):
                gather_batch_kernel_nd[
                    dtype, BATCH, Self.OBS_DIM, Self.ACTION_DIM, Self.CAPACITY
                ](bs, ba, br, bns, bd, rbs, rba, rbr, rbns, rbd, idx)

            ctx.enqueue_function[gather_nd_wrapper](
                sampled_obs_t,
                sampled_actions_t,
                sampled_rewards_t,
                sampled_next_obs_t,
                sampled_dones_t,
                buf_states_t,
                buf_actions_t,
                buf_rewards_t,
                buf_next_states_t,
                buf_dones_t,
                indices_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
