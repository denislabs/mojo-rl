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


def ezv2_cum_prio_scan_kernel[
    CAP: Int,
    K: Int,
](
    priorities: LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(CAP), MutAnyOrigin],
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
    """Single-thread serial scan: identical algorithm to the CPU loop at
    `efficient_zero_v2.mojo:2349-2366`. Validates each window start by
    checking no done in [idx, idx+K-1], accumulates a cumulative
    priority over valid starts, writes both `cum_prio` and `cand_starts`
    arrays plus `n_valid` / `total_prio` scalars.
    """
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t != 0:
        return

    var n_valid: Int = 0
    var total = Scalar[dtype](0.0)
    if buf_size > K:
        for offset in range(buf_size - K):
            var idx = (oldest + offset) % CAP
            var valid = True
            for k in range(K):
                var iidx = (idx + k) % CAP
                if rebind[Scalar[dtype]](dones[iidx]) > Scalar[dtype](0.5):
                    valid = False
                    break
            if not valid:
                continue
            var p = rebind[Scalar[dtype]](priorities[idx])
            if p < Scalar[dtype](1.0e-8):
                p = Scalar[dtype](1.0e-8)
            total += p
            cum_prio_out[n_valid] = total
            cand_starts_out[n_valid] = Scalar[DType.int32](idx)
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
    current_train_step: UInt32,
):
    """One thread per (sample, k) slot. K+1 slots wide for obs / mcts_*
    / age; K slots wide for actions / rewards (skipped at k == K).

    Layout of destination buffers matches the host fill at
    `efficient_zero_v2.mojo:2391-2417` (per-sample-time-major).
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
    # Scalars
    oldest: Int,
    buf_size: Int,
    current_train_step: UInt32,
    rng_seed: UInt32,
) raises:
    """Launches kernels 1 → 2 → 3 → 4 in sequence. After return,
    `dst_*` are populated and `batch_start_idx_buf` holds the picks
    needed by kernel 5 (priority writeback) post-train."""
    comptime scan = ezv2_cum_prio_scan_kernel[CAP, K]
    comptime sample = ezv2_sample_starts_kernel[BATCH, CAP]
    comptime gather = ezv2_gather_window_kernel[BATCH, K, OBS, ACT, CAP]
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

    ctx.enqueue_function[scan](
        prio_t, dones_t, cum_t, cs_t, nv_t, tp_t, oldest, buf_size,
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

    comptime gather_threads = BATCH * (K + 1)
    comptime gather_blocks = (gather_threads + TPB - 1) // TPB
    ctx.enqueue_function[gather](
        bsi_t,
        sob_t, sac_t, sr_t, smp_t, smv_t, ssw_t,
        dob_t, dac_t, drw_t, dmp_t, dmv_t, dag_t,
        current_train_step,
        grid_dim=(gather_blocks,), block_dim=(TPB,),
    )

    comptime cumr_blocks = (BATCH + TPB - 1) // TPB
    ctx.enqueue_function[cumr](
        drw_t, dcr_t,
        grid_dim=(cumr_blocks,), block_dim=(TPB,),
    )
