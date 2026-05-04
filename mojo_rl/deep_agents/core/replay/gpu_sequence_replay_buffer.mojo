"""GPUSequenceReplayBuffer: GPU-resident sequence replay for TD-MPC2,
DreamerV3, MuZero.

Stores transitions from N_ENVS parallel environments in per-env circular
buffers on GPU. Samples contiguous sequences of length H directly on GPU,
avoiding costly GPU->CPU->GPU round trips.

Two parallel boundary fields:
- dones_buf:        terminated OR truncated. Used by the sample kernel to
                    reject sequences that would span an env reset (any
                    obs[t+1] from the next episode is invalid for both
                    the consistency loss and the TD bootstrap).
- terminations_buf: terminated only. Returned in the output `batch_dones`
                    so the consumer's `(1 - d) * V_next` zeros the
                    bootstrap on natural termination (Hopper falling) but
                    keeps it on time-limit truncation. Without this, V is
                    biased toward |r / (1 - γ·(1 - p_truncation))|.

Design:
- Per-env storage: [N_ENVS * PER_ENV_CAP * dim] with strided addressing
- All envs share the same write_idx/size (they step in lockstep)
- Store: one kernel writes N_ENVS transitions per call
- Sample: one kernel generates BATCH valid sequences via rejection sampling
- Boundary check uses dones_buf; output dones field uses terminations_buf

Usage:
    var rb = GPUSequenceReplayBuffer[PER_ENV_CAP, OBS_DIM, ACTION_DIM, N_ENVS](ctx)

    # Before env step: save current observations
    rb.save_obs(ctx, env_obs_buf)

    # After env step: store transitions (preferred — distinguish flags)
    rb.store_with_termination(ctx, env_act_buf, env_rew_buf,
                              env_done_buf, env_terminated_buf)

    # Or, when only one flag is available (backwards-compatible):
    rb.store(ctx, env_act_buf, env_rew_buf, env_done_buf)
    # ↑ stores the same flag in both fields — the bootstrap mask returned
    #   in batch_dones will then mirror term|trunc, biasing V on truncation.

    # Sample batch of sequences directly on GPU
    rb.sample[BATCH, H](ctx, seed, batch_obs_buf, batch_act_buf,
                         batch_rew_buf, batch_done_buf)
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.random.philox import Random as PhiloxRandom
from std.math import min, max
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import dtype, TPB


# =============================================================================
# GPU Kernels
# =============================================================================


@always_inline
def gpu_seq_store_kernel[
    dtype: DType,
    N_ENVS: Int,
    PER_ENV_CAP: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
](
    # Saved obs from before env step
    prev_obs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * OBS_DIM), MutAnyOrigin
    ],
    # Step results
    actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACTION_DIM), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # term|trunc — boundary flag, drives sequence rejection in sampling
    dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # term-only — bootstrap mask returned in batch_dones at sample time
    terminations: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Per-env circular buffer storage (flat, strided by env)
    buf_obs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP * OBS_DIM), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[
        dtype,
        Layout.row_major(N_ENVS * PER_ENV_CAP * ACTION_DIM),
        MutAnyOrigin,
    ],
    buf_rewards: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    buf_dones: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    buf_terminations: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    write_idx: Scalar[DType.int32],
):
    """Store one transition per env into per-env circular buffers.

    Thread i handles env i. Writes obs/act/rew/done/terminated at write_idx
    within that env's section of the strided buffer.
    """
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= N_ENVS:
        return

    var w = Int(write_idx)

    # Obs: buf_obs[env_idx * PER_ENV_CAP * OBS_DIM + w * OBS_DIM + d]
    var obs_base = env_idx * PER_ENV_CAP * OBS_DIM + w * OBS_DIM
    var src_obs_base = env_idx * OBS_DIM
    for d in range(OBS_DIM):
        buf_obs[obs_base + d] = prev_obs[src_obs_base + d]

    # Actions: buf_actions[env_idx * PER_ENV_CAP * ACTION_DIM + w * ACTION_DIM + d]
    var act_base = env_idx * PER_ENV_CAP * ACTION_DIM + w * ACTION_DIM
    var src_act_base = env_idx * ACTION_DIM
    for d in range(ACTION_DIM):
        buf_actions[act_base + d] = actions[src_act_base + d]

    # Scalars
    buf_rewards[env_idx * PER_ENV_CAP + w] = rewards[env_idx]
    buf_dones[env_idx * PER_ENV_CAP + w] = dones[env_idx]
    buf_terminations[env_idx * PER_ENV_CAP + w] = terminations[env_idx]


@always_inline
def gpu_seq_sample_kernel[
    dtype: DType,
    BATCH: Int,
    H: Int,
    N_ENVS: Int,
    PER_ENV_CAP: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
](
    # Per-env circular buffer storage
    buf_obs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP * OBS_DIM), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[
        dtype,
        Layout.row_major(N_ENVS * PER_ENV_CAP * ACTION_DIM),
        MutAnyOrigin,
    ],
    buf_rewards: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    # term|trunc — used to reject sequences that would cross an env reset
    buf_dones: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    # term-only — copied into the output batch_dones as the bootstrap mask
    buf_terminations: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    # Output batch buffers
    batch_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH * (H + 1) * OBS_DIM), MutAnyOrigin
    ],
    batch_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH * H * ACTION_DIM), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH * H), MutAnyOrigin
    ],
    batch_dones: LayoutTensor[dtype, Layout.row_major(BATCH * H), MutAnyOrigin],
    # Buffer state (same for all envs)
    buf_size: Scalar[DType.int32],
    buf_write_idx: Scalar[DType.int32],
    rng_seed: Scalar[DType.uint32],
):
    """Sample BATCH sequences of length H from per-env GPU replay buffers.

    Each thread produces one sequence. Envs are assigned round-robin.
    Rejection uses `buf_dones` (term|trunc) so sequences never cross a
    reset; the output `batch_dones` is filled from `buf_terminations`
    (term-only) so the consumer's `(1 - d) * V_next` masks bootstrap
    correctly without zeroing on time-limit truncation.

    Output layout matches CPU SequenceReplayBuffer.sample_sequences:
      batch_obs:     [BATCH * (H+1) * OBS_DIM]  (batch-major)
      batch_actions: [BATCH * H * ACTION_DIM]
      batch_rewards: [BATCH * H]
      batch_dones:   [BATCH * H]  (term-only — bootstrap mask)
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH:
        return

    # Assign env round-robin
    var env_idx = tid % N_ENVS
    var sz = Int(buf_size)
    var wptr = Int(buf_write_idx)

    # Rejection sampling: find valid sequence start
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(tid * 137 + 1),
        offset=0,
    )
    var start = -1

    # We need H+1 contiguous steps, so valid range is [0, sz - H - 1]
    var max_start = sz - H - 1
    if max_start < 0:
        max_start = 0

    for _attempt in range(64):
        var rand_vals = philox.step_uniform()
        var candidate = Int(
            Scalar[dtype](rand_vals[0]) * Scalar[dtype](max_start + 1)
        )
        if candidate > max_start:
            candidate = max_start

        # Convert relative index to circular buffer position
        # Oldest data is at (wptr - sz) mod PER_ENV_CAP
        var actual = (wptr - sz + candidate + PER_ENV_CAP) % PER_ENV_CAP

        # Validate: no done in steps [actual, actual+H-2]
        var valid = True
        for t in range(H - 1):
            var idx = (actual + t) % PER_ENV_CAP
            if buf_dones[env_idx * PER_ENV_CAP + idx] > Scalar[dtype](0.5):
                valid = False
                break

        if valid:
            start = actual
            break

    # Fallback: use oldest valid position
    if start < 0:
        start = (wptr - sz + PER_ENV_CAP) % PER_ENV_CAP

    # Gather sequence into output batch buffers
    var obs_out_base = tid * (H + 1) * OBS_DIM
    var act_out_base = tid * H * ACTION_DIM
    var scalar_out_base = tid * H

    # Obs: H+1 observations
    for t in range(H + 1):
        var buf_idx = (start + t) % PER_ENV_CAP
        var env_obs_base = env_idx * PER_ENV_CAP * OBS_DIM + buf_idx * OBS_DIM
        var out_off = obs_out_base + t * OBS_DIM
        for d in range(OBS_DIM):
            batch_obs[out_off + d] = buf_obs[env_obs_base + d]

    # Actions, rewards, dones: H steps
    for t in range(H):
        var buf_idx = (start + t) % PER_ENV_CAP
        # Actions
        var env_act_base = (
            env_idx * PER_ENV_CAP * ACTION_DIM + buf_idx * ACTION_DIM
        )
        var act_off = act_out_base + t * ACTION_DIM
        for d in range(ACTION_DIM):
            batch_actions[act_off + d] = buf_actions[env_act_base + d]
        # Rewards and dones (term-only bootstrap mask, not term|trunc)
        batch_rewards[scalar_out_base + t] = buf_rewards[
            env_idx * PER_ENV_CAP + buf_idx
        ]
        batch_dones[scalar_out_base + t] = buf_terminations[
            env_idx * PER_ENV_CAP + buf_idx
        ]


# =============================================================================
# Struct
# =============================================================================


struct GPUSequenceReplayBuffer[
    PER_ENV_CAP: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    N_ENVS: Int,
](Movable):
    """GPU-resident per-env circular replay buffer with sequence sampling.

    Stores transitions from N_ENVS parallel environments in strided layout.
    All envs share the same write pointer (they step in lockstep).
    Samples contiguous H-step sequences directly on GPU.

    Parameters:
        PER_ENV_CAP: Capacity per environment (circular buffer size).
        OBS_DIM: Observation dimension.
        ACTION_DIM: Action dimension.
        N_ENVS: Number of parallel environments.
    """

    # Per-env circular buffers on GPU (strided: env_idx * PER_ENV_CAP * dim)
    var obs_buf: DeviceBuffer[dtype]  # [N_ENVS * PER_ENV_CAP * OBS_DIM]
    var actions_buf: DeviceBuffer[dtype]  # [N_ENVS * PER_ENV_CAP * ACTION_DIM]
    var rewards_buf: DeviceBuffer[dtype]  # [N_ENVS * PER_ENV_CAP]
    # term|trunc — boundary flag (drives sequence-rejection in sampling)
    var dones_buf: DeviceBuffer[dtype]  # [N_ENVS * PER_ENV_CAP]
    # term-only — bootstrap mask (returned as batch_dones at sample time)
    var terminations_buf: DeviceBuffer[dtype]  # [N_ENVS * PER_ENV_CAP]

    # Temp buffer to save obs before env step overwrites them
    var prev_obs_buf: DeviceBuffer[dtype]  # [N_ENVS * OBS_DIM]

    # CPU-side tracking (same for all envs since they step in lockstep)
    var write_idx: Int
    var size: Int

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate all device buffers and zero-initialize."""
        self.obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PER_ENV_CAP * Self.OBS_DIM
        )
        self.actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PER_ENV_CAP * Self.ACTION_DIM
        )
        self.rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PER_ENV_CAP
        )
        self.dones_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PER_ENV_CAP
        )
        self.terminations_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.PER_ENV_CAP
        )
        self.prev_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.OBS_DIM
        )
        ctx.enqueue_memset(self.obs_buf, 0)
        ctx.enqueue_memset(self.actions_buf, 0)
        ctx.enqueue_memset(self.rewards_buf, 0)
        ctx.enqueue_memset(self.dones_buf, 0)
        ctx.enqueue_memset(self.terminations_buf, 0)
        ctx.enqueue_memset(self.prev_obs_buf, 0)
        self.write_idx = 0
        self.size = 0

    def __init__(out self, *, deinit take: Self):
        self.obs_buf = take.obs_buf^
        self.actions_buf = take.actions_buf^
        self.rewards_buf = take.rewards_buf^
        self.dones_buf = take.dones_buf^
        self.terminations_buf = take.terminations_buf^
        self.prev_obs_buf = take.prev_obs_buf^
        self.write_idx = take.write_idx
        self.size = take.size

    def save_obs(self, ctx: DeviceContext, env_obs: DeviceBuffer[dtype]) raises:
        """Save current observations before env step overwrites them.

        Call this BEFORE stepping the environments.

        Args:
            ctx: GPU device context.
            env_obs: Current env observations [N_ENVS * OBS_DIM].
        """
        ctx.enqueue_copy(self.prev_obs_buf, env_obs)

    def store(
        mut self,
        ctx: DeviceContext,
        actions: DeviceBuffer[dtype],
        rewards: DeviceBuffer[dtype],
        dones: DeviceBuffer[dtype],
    ) raises:
        """Store one transition per env using a single done-like flag
        (collapsed-flag form, backwards-compatible).

        The same buffer is stored as both the boundary flag (drives
        sequence rejection in sampling) and the bootstrap mask (returned
        in `batch_dones`). Callers that can distinguish termination from
        truncation should use `store_with_termination` so the bootstrap
        mask stays off on time-limit truncation.

        Args:
            ctx: GPU device context.
            actions: Actions taken [N_ENVS * ACTION_DIM].
            rewards: Rewards received [N_ENVS].
            dones: Done flags [N_ENVS].
        """
        self.store_with_termination(ctx, actions, rewards, dones, dones)

    def store_with_termination(
        mut self,
        ctx: DeviceContext,
        actions: DeviceBuffer[dtype],
        rewards: DeviceBuffer[dtype],
        dones: DeviceBuffer[dtype],
        terminations: DeviceBuffer[dtype],
    ) raises:
        """Store one transition per env with explicit term/trunc separation.

        Call this AFTER stepping the environments and after save_obs.
        Enqueues one GPU kernel (N_ENVS threads) and updates CPU counters.

        Args:
            ctx: GPU device context.
            actions: Actions taken [N_ENVS * ACTION_DIM].
            rewards: Rewards received [N_ENVS].
            dones: term|trunc flags [N_ENVS] — used for sequence-boundary
                detection so we never sample sequences that cross a reset.
            terminations: term-only flags [N_ENVS] — returned at sample
                time as `batch_dones` and consumed by the trainer as
                `(1 - terminated) * V_next`. Must be 0 on time-limit
                truncation, otherwise V is biased toward
                |r / (1 - γ·(1 - p_truncation))|.
        """
        comptime ENV_BLOCKS = (Self.N_ENVS + TPB - 1) // TPB

        var prev_obs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.OBS_DIM),
            MutAnyOrigin,
        ](self.prev_obs_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.ACTION_DIM),
            MutAnyOrigin,
        ](actions.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](rewards.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](dones.unsafe_ptr())
        var terminations_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](terminations.unsafe_ptr())

        var buf_obs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP * Self.OBS_DIM),
            MutAnyOrigin,
        ](self.obs_buf.unsafe_ptr())
        var buf_actions_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP * Self.ACTION_DIM),
            MutAnyOrigin,
        ](self.actions_buf.unsafe_ptr())
        var buf_rewards_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
            MutAnyOrigin,
        ](self.rewards_buf.unsafe_ptr())
        var buf_dones_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
            MutAnyOrigin,
        ](self.dones_buf.unsafe_ptr())
        var buf_terminations_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
            MutAnyOrigin,
        ](self.terminations_buf.unsafe_ptr())

        var write_idx_s = Scalar[DType.int32](self.write_idx)

        @parameter
        @always_inline
        def store_wrapper(
            po: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.OBS_DIM),
                MutAnyOrigin,
            ],
            a: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.ACTION_DIM),
                MutAnyOrigin,
            ],
            r: LayoutTensor[dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin],
            d: LayoutTensor[dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin],
            tm: LayoutTensor[
                dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            bo: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP * Self.OBS_DIM),
                MutAnyOrigin,
            ],
            ba: LayoutTensor[
                dtype,
                Layout.row_major(
                    Self.N_ENVS * Self.PER_ENV_CAP * Self.ACTION_DIM
                ),
                MutAnyOrigin,
            ],
            br: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
                MutAnyOrigin,
            ],
            bd: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
                MutAnyOrigin,
            ],
            btm: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
                MutAnyOrigin,
            ],
            widx: Scalar[DType.int32],
        ):
            gpu_seq_store_kernel[
                dtype,
                Self.N_ENVS,
                Self.PER_ENV_CAP,
                Self.OBS_DIM,
                Self.ACTION_DIM,
            ](po, a, r, d, tm, bo, ba, br, bd, btm, widx)

        ctx.enqueue_function[store_wrapper, store_wrapper](
            prev_obs_t,
            actions_t,
            rewards_t,
            dones_t,
            terminations_t,
            buf_obs_t,
            buf_actions_t,
            buf_rewards_t,
            buf_dones_t,
            buf_terminations_t,
            write_idx_s,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Update CPU-side tracking
        self.write_idx = (self.write_idx + 1) % Self.PER_ENV_CAP
        self.size = min(self.size + 1, Self.PER_ENV_CAP)

    def is_ready[min_size: Int](self) -> Bool:
        """Check if the buffer has enough data for sampling sequences."""
        return self.size >= min_size

    def sample[
        BATCH: Int,
        H: Int,
    ](
        self,
        ctx: DeviceContext,
        rng_seed: UInt32,
        batch_obs: DeviceBuffer[dtype],
        batch_actions: DeviceBuffer[dtype],
        batch_rewards: DeviceBuffer[dtype],
        batch_dones: DeviceBuffer[dtype],
    ) raises:
        """Sample BATCH sequences of length H directly on GPU.

        Each sequence provides H+1 observations and H steps of act/rew/done.
        Sequences do not cross episode boundaries (validated on-GPU).

        Output layout (written directly into device buffers):
          batch_obs:     [BATCH * (H+1) * OBS_DIM]
          batch_actions: [BATCH * H * ACTION_DIM]
          batch_rewards: [BATCH * H]
          batch_dones:   [BATCH * H]

        Args:
            ctx: GPU device context.
            rng_seed: Random seed (vary per call).
            batch_obs: Output device buffer for observations.
            batch_actions: Output device buffer for actions.
            batch_rewards: Output device buffer for rewards.
            batch_dones: Output device buffer for dones.
        """
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        var buf_obs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP * Self.OBS_DIM),
            MutAnyOrigin,
        ](self.obs_buf.unsafe_ptr())
        var buf_actions_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP * Self.ACTION_DIM),
            MutAnyOrigin,
        ](self.actions_buf.unsafe_ptr())
        var buf_rewards_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
            MutAnyOrigin,
        ](self.rewards_buf.unsafe_ptr())
        var buf_dones_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
            MutAnyOrigin,
        ](self.dones_buf.unsafe_ptr())
        var buf_terminations_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
            MutAnyOrigin,
        ](self.terminations_buf.unsafe_ptr())

        var batch_obs_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * (H + 1) * Self.OBS_DIM),
            MutAnyOrigin,
        ](batch_obs.unsafe_ptr())
        var batch_actions_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * H * Self.ACTION_DIM),
            MutAnyOrigin,
        ](batch_actions.unsafe_ptr())
        var batch_rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * H), MutAnyOrigin
        ](batch_rewards.unsafe_ptr())
        var batch_dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * H), MutAnyOrigin
        ](batch_dones.unsafe_ptr())

        var buf_size_s = Scalar[DType.int32](self.size)
        var buf_write_idx_s = Scalar[DType.int32](self.write_idx)
        var rng_seed_s = Scalar[DType.uint32](rng_seed)

        @parameter
        @always_inline
        def sample_wrapper(
            bo: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP * Self.OBS_DIM),
                MutAnyOrigin,
            ],
            ba: LayoutTensor[
                dtype,
                Layout.row_major(
                    Self.N_ENVS * Self.PER_ENV_CAP * Self.ACTION_DIM
                ),
                MutAnyOrigin,
            ],
            br: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
                MutAnyOrigin,
            ],
            bd: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
                MutAnyOrigin,
            ],
            btm: LayoutTensor[
                dtype,
                Layout.row_major(Self.N_ENVS * Self.PER_ENV_CAP),
                MutAnyOrigin,
            ],
            out_obs: LayoutTensor[
                dtype,
                Layout.row_major(BATCH * (H + 1) * Self.OBS_DIM),
                MutAnyOrigin,
            ],
            out_act: LayoutTensor[
                dtype,
                Layout.row_major(BATCH * H * Self.ACTION_DIM),
                MutAnyOrigin,
            ],
            out_rew: LayoutTensor[
                dtype, Layout.row_major(BATCH * H), MutAnyOrigin
            ],
            out_done: LayoutTensor[
                dtype, Layout.row_major(BATCH * H), MutAnyOrigin
            ],
            bsz: Scalar[DType.int32],
            bwi: Scalar[DType.int32],
            seed: Scalar[DType.uint32],
        ):
            gpu_seq_sample_kernel[
                dtype,
                BATCH,
                H,
                Self.N_ENVS,
                Self.PER_ENV_CAP,
                Self.OBS_DIM,
                Self.ACTION_DIM,
            ](
                bo,
                ba,
                br,
                bd,
                btm,
                out_obs,
                out_act,
                out_rew,
                out_done,
                bsz,
                bwi,
                seed,
            )

        ctx.enqueue_function[sample_wrapper, sample_wrapper](
            buf_obs_t,
            buf_actions_t,
            buf_rewards_t,
            buf_dones_t,
            buf_terminations_t,
            batch_obs_t,
            batch_actions_t,
            batch_rewards_t,
            batch_dones_t,
            buf_size_s,
            buf_write_idx_s,
            rng_seed_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
