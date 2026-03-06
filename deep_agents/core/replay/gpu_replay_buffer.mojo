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

from nn.constants import dtype, TPB
from .replay_buffer import HeapReplayBuffer
from ..kernels import (
    store_transitions_kernel,
    sample_indices_kernel,
    gather_batch_kernel,
    store_transitions_kernel_nd,
    gather_batch_kernel_nd,
)

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer


struct GPUReplayBuffer[CAPACITY: Int, OBS_DIM: Int, ACTION_DIM: Int = 1](
    Movable
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

    fn __init__(out self, ctx: DeviceContext) raises:
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

    fn __init__(out self, *, deinit take: Self):
        self.states_buf = take.states_buf^
        self.actions_buf = take.actions_buf^
        self.rewards_buf = take.rewards_buf^
        self.next_states_buf = take.next_states_buf^
        self.dones_buf = take.dones_buf^
        self.write_idx = take.write_idx
        self.size = take.size

    fn is_ready[BATCH: Int](self) -> Bool:
        """Return True if the buffer holds at least BATCH transitions."""
        return self.size >= BATCH

    fn upload_from(
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

        comptime if Self.ACTION_DIM == 1:
            var actions_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS), MutAnyOrigin
            ](actions.unsafe_ptr())
            var buf_actions_t = LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ](self.actions_buf.unsafe_ptr())

            @always_inline
            fn store_wrapper(
                s: LayoutTensor[
                    dtype, Layout.row_major(N_ENVS, Self.OBS_DIM), MutAnyOrigin
                ],
                a: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
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
                    dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
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
                store_transitions_kernel[
                    dtype, N_ENVS, Self.OBS_DIM, Self.CAPACITY
                ](s, a, r, ns, d, bs, ba, br, bns, bd, widx)

            ctx.enqueue_function[store_wrapper, store_wrapper](
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

            @always_inline
            fn store_nd_wrapper(
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

            ctx.enqueue_function[store_nd_wrapper, store_nd_wrapper](
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

    fn sample[
        BATCH: Int
    ](
        self,
        ctx: DeviceContext,
        rng_seed: UInt32,
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
            rng_seed: Base seed for xorshift32 RNG (vary per call).
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
        var buf_size = Scalar[DType.int32](self.size)
        var seed_s = Scalar[DType.uint32](rng_seed)

        @always_inline
        fn sample_wrapper(
            idx: LayoutTensor[
                DType.int32, Layout.row_major(BATCH), MutAnyOrigin
            ],
            bsize: Scalar[DType.int32],
            s: Scalar[DType.uint32],
        ):
            sample_indices_kernel[dtype, BATCH](idx, bsize, s)

        ctx.enqueue_function[sample_wrapper, sample_wrapper](
            indices_t,
            buf_size,
            seed_s,
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

        comptime if Self.ACTION_DIM == 1:
            var sampled_actions_t = LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ](sampled_actions.unsafe_ptr())
            var buf_actions_t = LayoutTensor[
                dtype, Layout.row_major(Self.CAPACITY), MutAnyOrigin
            ](self.actions_buf.unsafe_ptr())

            @always_inline
            fn gather_wrapper(
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

            ctx.enqueue_function[gather_wrapper, gather_wrapper](
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

            @always_inline
            fn gather_nd_wrapper(
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

            ctx.enqueue_function[gather_nd_wrapper, gather_nd_wrapper](
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
