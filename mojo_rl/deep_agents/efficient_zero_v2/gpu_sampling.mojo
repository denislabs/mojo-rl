"""GPU priority sampling + window gather + priority writeback for EZ-V2.

Replaces the CPU-side priority-weighted sampling + per-sample upload
loop in `train_step_gpu` (lines ~2335-2434, ~4082-4090 of
`efficient_zero_v2.mojo`).

Reads from `EZV2GPUReplayBuffer` (Step 4's GPU-resident replay), writes
the sampled batch directly into `EZV2GPUStateBase.batch_*_buf`. No
host round-trip during the sample/gather phase. The priority writeback
kernel also runs entirely on device.

Pipeline (4 kernels):

    1. `ezv2_cum_prio_scan_kernel`  — single-thread serial scan of the
        per-window valid mask + cumulative priority array. Linear scan
        is fine at CAP=50k (1 block × ~50 µs on Apple).
    2. `ezv2_sample_starts_kernel`  — BATCH threads, each draws a
        Philox uniform, linear-searches `cum_prio` for its sample.
        Writes `batch_start_idx_buf[BATCH]`.
    3. `ezv2_gather_window_kernel`  — BATCH × (K+1) threads, each
        gathers the per-(sample, k) slot of obs / mcts_policies /
        mcts_values / age plus per-(sample, k<K) slot of actions /
        rewards.
    4. `ezv2_cum_rewards_kernel`    — BATCH threads, sequential cumsum
        across K rewards per sample (paper App. G LSTM target).
    5. `ezv2_priority_writeback_kernel` — BATCH threads, scatter-write
        `priorities[batch_start_idx[b]] = priorities_out_buf[b]`.

Bit-exact match against the host code is verified end-to-end by
`examples/cartpole/cartpole_ezv2_full_gpu_step5.mojo`.
"""

from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype, TPB


# ═════════════════════════════════════════════════════════════════════════
# Kernel 1: cumulative priority scan
# ═════════════════════════════════════════════════════════════════════════


def ezv2_per_offset_priority_kernel[
    CAP: Int,
    K: Int,
](
    priorities: LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin],
    prio_out: LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin],
    oldest: Int,
    buf_size: Int,
):
    """Stage 1 (parallel — one thread per candidate window start).

    For each `offset` in `[0, buf_size - K)`, validate the K-step window
    `[idx, idx+K-1]` (mod CAP) by checking `dones`. If all zero, write
    `max(priorities[idx], 1e-8)` to `prio_out[offset]`; otherwise write
    `0.0`. The followup `ezv2_cum_prio_compact_kernel` reads this array
    serially to build the compacted cumulative-priority array, skipping
    zero entries.

    Splitting the K-step done check off into a parallel kernel removes
    the inner-loop work from the serial compact (Phase 3d optimization,
    2026-05-13). At buf_size ~12k, K=5 the validate cost collapses from
    ~340μs single-threaded to <30μs parallel.
    """
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if buf_size <= K or t >= buf_size - K:
        return
    var offset = t
    var idx = (oldest + offset) % CAP
    var valid = True
    for k in range(K):
        var iidx = (idx + k) % CAP
        if rebind[Scalar[dtype]](dones[iidx]) > Scalar[dtype](0.5):
            valid = False
            break
    if not valid:
        prio_out[offset] = Scalar[dtype](0.0)
    else:
        var p = rebind[Scalar[dtype]](priorities[idx])
        if p < Scalar[dtype](1.0e-8):
            p = Scalar[dtype](1.0e-8)
        prio_out[offset] = p


def ezv2_cum_prio_compact_kernel[
    CAP: Int,
    K: Int,
](
    prio_in: LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin],
    cum_prio_out: LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ],
    cand_starts_out: LayoutTensor[
        DType.int32, Layout.row_major(CAP), MutAnyOrigin
    ],
    n_valid_out: LayoutTensor[
        DType.int32, Layout.row_major(1), MutAnyOrigin
    ],
    total_prio_out: LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ],
    oldest: Int,
    buf_size: Int,
):
    """Stage 2 (serial compaction + cumulative sum).

    Walks `prio_in[0..buf_size-K)`, skips zeros, writes the running
    cumulative sum into `cum_prio_out[n_valid]` and the source slot
    index into `cand_starts_out[n_valid]`. Sets `n_valid_out[0]` and
    `total_prio_out[0]` at the end.

    `prio_in` and `cum_prio_out` may alias (same `cum_prio_buf` in the
    driver). In-place is safe because writes at iter `offset` go to
    `cum_prio_out[n_valid]` with `n_valid ≤ offset`, so writes never
    clobber subsequent reads in a single-threaded sweep.
    """
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t != 0:
        return

    var n_valid: Int = 0
    var total = Scalar[dtype](0.0)
    if buf_size > K:
        for offset in range(buf_size - K):
            var p = rebind[Scalar[dtype]](prio_in[offset])
            if p > Scalar[dtype](0.0):
                total += p
                cum_prio_out[n_valid] = total
                cand_starts_out[n_valid] = Scalar[DType.int32](
                    (oldest + offset) % CAP
                )
                n_valid += 1

    n_valid_out[0] = Scalar[DType.int32](n_valid)
    total_prio_out[0] = total


# ═════════════════════════════════════════════════════════════════════════
# Kernel 2: parallel sample-start selection
# ═════════════════════════════════════════════════════════════════════════


def ezv2_sample_starts_kernel[
    BATCH: Int,
    CAP: Int,
](
    cum_prio: LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin],
    cand_starts: LayoutTensor[
        DType.int32, Layout.row_major(CAP), MutAnyOrigin
    ],
    n_valid_in: LayoutTensor[
        DType.int32, Layout.row_major(1), MutAnyOrigin
    ],
    total_prio_in: LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ],
    batch_start_idx_out: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    rng_seed: UInt32,
):
    """One thread per batch sample. Each draws a single Philox uniform,
    multiplies by `total_prio`, then linear-searches `cum_prio` for the
    smallest index whose cumulative priority ≥ u. Writes `batch_start_idx[b]`
    = `cand_starts[picked]`.

    Linear scan in O(n_valid) per sample — at CAP=50k, BATCH=64 that's
    3.2M comparisons total per call, about 30 µs on Apple Silicon. Sum
    tree optimization deferred (same call as Phase 2 in the plan).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var n_valid = Int(rebind[Scalar[DType.int32]](n_valid_in[0]))
    var total = rebind[Scalar[dtype]](total_prio_in[0])

    # Per-thread Philox: seed = base * golden_ratio + b. This matches
    # the per-block RNG pattern used elsewhere (see gpu_mcts.mojo).
    var rng = PhiloxRandom(
        seed=(UInt64(rng_seed) * UInt64(0x9E3779B97F4A7C15)) + UInt64(b),
        offset=0,
    )
    var rand_vals = rng.step_uniform()
    var u = Scalar[dtype](rand_vals[0]) * total

    # Linear search through the cumulative priority array.
    var picked: Int = 0
    for i in range(n_valid):
        if rebind[Scalar[dtype]](cum_prio[i]) >= u:
            picked = i
            break

    batch_start_idx_out[b] = cand_starts[picked]


# ═════════════════════════════════════════════════════════════════════════
# Kernel 3: parallel window gather
# ═════════════════════════════════════════════════════════════════════════


def ezv2_gather_window_kernel[
    BATCH: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    K_ROOT: Int,
    CAP: Int,
](
    batch_start_idx: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    # Source: GPU replay buffer
    src_obs: LayoutTensor[
        dtype, Layout.row_major(CAP * OBS), MutAnyOrigin
    ],
    src_actions: LayoutTensor[
        dtype, Layout.row_major(CAP * ACT), MutAnyOrigin
    ],
    src_rewards: LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin],
    src_mcts_policies: LayoutTensor[
        dtype, Layout.row_major(CAP * ACT), MutAnyOrigin
    ],
    src_mcts_values: LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ],
    src_step_at_write: LayoutTensor[
        DType.uint32, Layout.row_major(CAP), MutAnyOrigin
    ],
    src_mcts_samp_act: LayoutTensor[
        dtype, Layout.row_major(CAP * K_ROOT * ACT), MutAnyOrigin
    ],
    src_mcts_imp_pi: LayoutTensor[
        dtype, Layout.row_major(CAP * K_ROOT), MutAnyOrigin
    ],
    # Destination: per-train_step batch buffers (per-sample-time-major)
    dst_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1) * OBS), MutAnyOrigin
    ],
    dst_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH * K * ACT), MutAnyOrigin
    ],
    dst_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ],
    dst_mcts_pol: LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1) * ACT), MutAnyOrigin
    ],
    dst_mcts_val: LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
    ],
    dst_age: LayoutTensor[
        DType.int32, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
    ],
    dst_mcts_samp_act: LayoutTensor[
        dtype,
        Layout.row_major(BATCH * (K + 1) * K_ROOT * ACT),
        MutAnyOrigin,
    ],
    dst_mcts_imp_pi: LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1) * K_ROOT), MutAnyOrigin
    ],
    current_train_step: UInt32,
):
    """One thread per (sample, k) slot. K+1 slots wide for obs / mcts_*
    / age / fullpi targets; K slots wide for actions / rewards
    (skipped at k == K).

    Layout of destination buffers matches the host fill at
    `efficient_zero_v2.mojo:2391-2417` (per-sample-time-major).

    The fullpi target writes (`dst_mcts_samp_act` + `dst_mcts_imp_pi`)
    are required by `ezv2_policy_loss_grad_continuous_fullpi_kernel`
    when `ACT_DIM==1`. Before 2026-05-14 these were not gathered on
    the GPU-sampling path, so the fullpi kernel read zero-initialized
    targets and produced `L_P ≈ -ent_scale·H_d`, zero policy gradient.
    Discrete + ACT_DIM>1 configs still get the writes (uniform cost,
    no overhead vs the kernel's other K+1 reads); their simple-best
    loss path just doesn't read these buffers.
    """
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= BATCH * (K + 1):
        return

    var b = t // (K + 1)
    var k = t % (K + 1)
    var start = Int(rebind[Scalar[DType.int32]](batch_start_idx[b]))
    var idx = (start + k) % CAP

    # K+1 fields (k = 0..K)
    for d in range(OBS):
        dst_obs[(b * (K + 1) + k) * OBS + d] = src_obs[idx * OBS + d]
    for a in range(ACT):
        dst_mcts_pol[(b * (K + 1) + k) * ACT + a] = src_mcts_policies[
            idx * ACT + a
        ]
    dst_mcts_val[b * (K + 1) + k] = src_mcts_values[idx]

    var sw_u = rebind[Scalar[DType.uint32]](src_step_at_write[idx])
    # Match the CPU `current_train_step - Int(step_at_write[idx])` then
    # clamp to 0. The cast to int32 here is the host's `Scalar[int32]`
    # cast at line 2410.
    var age64 = Int(current_train_step) - Int(sw_u)
    if age64 < 0:
        age64 = 0
    dst_age[b * (K + 1) + k] = Scalar[DType.int32](age64)

    # Fullpi targets (K_ROOT slots per (b, k)).
    var src_samp_off = idx * K_ROOT * ACT
    var dst_samp_off = (b * (K + 1) + k) * K_ROOT * ACT
    for j in range(K_ROOT * ACT):
        dst_mcts_samp_act[dst_samp_off + j] = src_mcts_samp_act[
            src_samp_off + j
        ]
    var src_pi_off = idx * K_ROOT
    var dst_pi_off = (b * (K + 1) + k) * K_ROOT
    for i in range(K_ROOT):
        dst_mcts_imp_pi[dst_pi_off + i] = src_mcts_imp_pi[
            src_pi_off + i
        ]

    # K-only fields (k = 0..K-1)
    if k < K:
        for a in range(ACT):
            dst_actions[(b * K + k) * ACT + a] = src_actions[
                idx * ACT + a
            ]
        dst_rewards[b * K + k] = src_rewards[idx]


# ═════════════════════════════════════════════════════════════════════════
# Kernel 4: per-sample cumulative rewards (LSTM target)
# ═════════════════════════════════════════════════════════════════════════


def ezv2_cum_rewards_kernel[
    BATCH: Int,
    K: Int,
](
    src_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ],
    dst_cum_rewards: LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ],
):
    """One thread per batch sample. Sequentially walks K steps,
    accumulating reward into `cum_rewards[b, k]`. Matches the host
    cumulative-reward block at `efficient_zero_v2.mojo:2426-2431`.
    Even when `use_reward_prefix=False` we still compute it — the
    upload path expects it populated; the kernel just doesn't read it
    when the LSTM head is disabled.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var cum = Scalar[dtype](0.0)
    for k in range(K):
        cum += rebind[Scalar[dtype]](src_rewards[b * K + k])
        dst_cum_rewards[b * K + k] = cum


# ═════════════════════════════════════════════════════════════════════════
# Kernel 5: priority scatter writeback
# ═════════════════════════════════════════════════════════════════════════


def ezv2_priority_writeback_kernel[
    BATCH: Int,
    CAP: Int,
](
    batch_start_idx: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    priorities_out: LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ],
    priorities_buf: LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ],
):
    """Scatter-write `priorities[batch_start_idx[b]] = priorities_out[b]`
    for b in [0, BATCH). Replaces the host loop at
    `efficient_zero_v2.mojo:4083-4087`.

    Note: BATCH samples may collide on the same start index — last write
    wins. This matches the CPU loop's behavior (linear assignment).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var idx = Int(rebind[Scalar[DType.int32]](batch_start_idx[b]))
    priorities_buf[idx] = priorities_out[b]


# ═════════════════════════════════════════════════════════════════════════
# Driver: end-to-end GPU sample + gather
# ═════════════════════════════════════════════════════════════════════════
#
# Convenience that bundles kernels 1-4. Caller still issues kernel 5
# after `train_step_gpu`'s priority refresh (kernel 5 needs
# `priorities_out_buf` which is only valid post-train).


def ezv2_gpu_sample_and_gather[
    CAP: Int,
    BATCH: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    K_ROOT: Int,
](
    ctx: DeviceContext,
    # GPU replay buffer (read-only inputs)
    priorities: DeviceBuffer[dtype],
    dones: DeviceBuffer[dtype],
    src_obs: DeviceBuffer[dtype],
    src_actions: DeviceBuffer[dtype],
    src_rewards: DeviceBuffer[dtype],
    src_mcts_policies: DeviceBuffer[dtype],
    src_mcts_values: DeviceBuffer[dtype],
    src_step_at_write: DeviceBuffer[DType.uint32],
    src_mcts_samp_act: DeviceBuffer[dtype],
    src_mcts_imp_pi: DeviceBuffer[dtype],
    # Scratch (same lifetime as caller — kernel 1 fills, kernel 2 reads)
    cum_prio_buf: DeviceBuffer[dtype],
    cand_starts_buf: DeviceBuffer[DType.int32],
    n_valid_buf: DeviceBuffer[DType.int32],
    total_prio_buf: DeviceBuffer[dtype],
    # Output: per-batch start indices (BATCH int32)
    batch_start_idx_buf: DeviceBuffer[DType.int32],
    # Destination batch buffers (caller-owned, one-per-train_step)
    dst_obs: DeviceBuffer[dtype],
    dst_actions: DeviceBuffer[dtype],
    dst_rewards: DeviceBuffer[dtype],
    dst_mcts_pol: DeviceBuffer[dtype],
    dst_mcts_val: DeviceBuffer[dtype],
    dst_age: DeviceBuffer[DType.int32],
    dst_cum_rewards: DeviceBuffer[dtype],
    dst_mcts_samp_act: DeviceBuffer[dtype],
    dst_mcts_imp_pi: DeviceBuffer[dtype],
    # Scalars
    oldest: Int,
    buf_size: Int,
    current_train_step: UInt32,
    rng_seed: UInt32,
) raises:
    """Launches kernels 1 → 2 → 3 → 4 in sequence. After return,
    `dst_*` are populated and `batch_start_idx_buf` holds the picks
    needed by kernel 5 (priority writeback) post-train.

    `src_mcts_samp_act` + `src_mcts_imp_pi` mirror the CPU state's
    full-π targets (paper Eq. 6); the gather kernel writes per-(b, k)
    windows into `dst_mcts_samp_act` + `dst_mcts_imp_pi`. Required
    for the ACT_DIM==1 fullpi loss path; harmless for other configs
    (simple-best path doesn't read these dst buffers)."""
    comptime validate = ezv2_per_offset_priority_kernel[CAP, K]
    comptime compact = ezv2_cum_prio_compact_kernel[CAP, K]
    comptime sample = ezv2_sample_starts_kernel[BATCH, CAP]
    comptime gather = ezv2_gather_window_kernel[
        BATCH, K, OBS, ACT, K_ROOT, CAP
    ]
    comptime cumr = ezv2_cum_rewards_kernel[BATCH, K]

    var prio_t = LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ](priorities.unsafe_ptr())
    var dones_t = LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ](dones.unsafe_ptr())
    var cum_t = LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ](cum_prio_buf.unsafe_ptr())
    var cs_t = LayoutTensor[
        DType.int32, Layout.row_major(CAP), MutAnyOrigin
    ](cand_starts_buf.unsafe_ptr())
    var nv_t = LayoutTensor[
        DType.int32, Layout.row_major(1), MutAnyOrigin
    ](n_valid_buf.unsafe_ptr())
    var tp_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](total_prio_buf.unsafe_ptr())
    var bsi_t = LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ](batch_start_idx_buf.unsafe_ptr())

    # Stage 1: parallel per-offset validate + priority write
    # (`cum_prio_buf` repurposed as scratch — stage 2 then compacts
    # in-place over the same buffer).
    comptime validate_threads = CAP  # upper bound; kernel early-returns
    # beyond `buf_size - K`. At CAP=50000, TPB=256 → 196 blocks.
    comptime validate_blocks = (validate_threads + TPB - 1) // TPB
    ctx.enqueue_function[validate](
        prio_t, dones_t, cum_t, oldest, buf_size,
        grid_dim=(validate_blocks,), block_dim=(TPB,),
    )

    # Stage 2: serial compaction (single thread).
    ctx.enqueue_function[compact](
        cum_t, cum_t, cs_t, nv_t, tp_t, oldest, buf_size,
        grid_dim=(1,), block_dim=(1,),
    )

    comptime sample_blocks = (BATCH + TPB - 1) // TPB
    ctx.enqueue_function[sample](
        cum_t, cs_t, nv_t, tp_t, bsi_t, rng_seed,
        grid_dim=(sample_blocks,), block_dim=(TPB,),
    )

    var sob_t = LayoutTensor[
        dtype, Layout.row_major(CAP * OBS), MutAnyOrigin
    ](src_obs.unsafe_ptr())
    var sac_t = LayoutTensor[
        dtype, Layout.row_major(CAP * ACT), MutAnyOrigin
    ](src_actions.unsafe_ptr())
    var sr_t = LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ](src_rewards.unsafe_ptr())
    var smp_t = LayoutTensor[
        dtype, Layout.row_major(CAP * ACT), MutAnyOrigin
    ](src_mcts_policies.unsafe_ptr())
    var smv_t = LayoutTensor[
        dtype, Layout.row_major(CAP), MutAnyOrigin
    ](src_mcts_values.unsafe_ptr())
    var ssw_t = LayoutTensor[
        DType.uint32, Layout.row_major(CAP), MutAnyOrigin
    ](src_step_at_write.unsafe_ptr())
    var ssa_t = LayoutTensor[
        dtype, Layout.row_major(CAP * K_ROOT * ACT), MutAnyOrigin
    ](src_mcts_samp_act.unsafe_ptr())
    var sip_t = LayoutTensor[
        dtype, Layout.row_major(CAP * K_ROOT), MutAnyOrigin
    ](src_mcts_imp_pi.unsafe_ptr())
    var dob_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1) * OBS), MutAnyOrigin
    ](dst_obs.unsafe_ptr())
    var dac_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * K * ACT), MutAnyOrigin
    ](dst_actions.unsafe_ptr())
    var drw_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ](dst_rewards.unsafe_ptr())
    var dmp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1) * ACT), MutAnyOrigin
    ](dst_mcts_pol.unsafe_ptr())
    var dmv_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
    ](dst_mcts_val.unsafe_ptr())
    var dag_t = LayoutTensor[
        DType.int32, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
    ](dst_age.unsafe_ptr())
    var dcr_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ](dst_cum_rewards.unsafe_ptr())
    var dsa_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * (K + 1) * K_ROOT * ACT),
        MutAnyOrigin,
    ](dst_mcts_samp_act.unsafe_ptr())
    var dip_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1) * K_ROOT), MutAnyOrigin
    ](dst_mcts_imp_pi.unsafe_ptr())

    comptime gather_threads = BATCH * (K + 1)
    comptime gather_blocks = (gather_threads + TPB - 1) // TPB
    ctx.enqueue_function[gather](
        bsi_t,
        sob_t, sac_t, sr_t, smp_t, smv_t, ssw_t, ssa_t, sip_t,
        dob_t, dac_t, drw_t, dmp_t, dmv_t, dag_t, dsa_t, dip_t,
        current_train_step,
        grid_dim=(gather_blocks,), block_dim=(TPB,),
    )

    comptime cumr_blocks = (BATCH + TPB - 1) // TPB
    ctx.enqueue_function[cumr](
        drw_t, dcr_t,
        grid_dim=(cumr_blocks,), block_dim=(TPB,),
    )
