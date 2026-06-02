"""GPUReplay[OBS, ACT, CAP] — device-resident circular replay buffer.

Phase C.1 — drop-in GPU equivalent of CPUReplay. SAC's GPU train_step
samples directly into device minibatch buffers, eliminating the per-step
CPU sample + 4 host→device uploads that the pre-C.1 GPU path did.

Surface mirrors `CPUReplay` but routes everything through DeviceContext:

  - `new(ctx, batch_capacity=4096)` allocates five DeviceBuffers
    [CAP, OBS] / [CAP, ACT] / [CAP] / [CAP, OBS] / [CAP], zero-fills
    them, plus 1-slot device staging buffers used by `add` and a small
    pre-sized indices buffer used by `sample`.
  - `add(ctx, obs_p, act_p, r, nxt_p, d)`: host pointers → 1-slot device
    staging → kernel write into circular slot at `pos`. CPU
    bookkeeping (`pos`, `size`) updates after enqueue.
  - `sample[BATCH](ctx, mb_s_dev, mb_a_dev, mb_r_dev, mb_sp_dev, mb_d_dev)`
    runs `_sample_indices_kernel` (PhiloxRandom seeded by `rng_seed`
    + `_rng_offset`) followed by `_gather_batch_kernel` writing into
    the caller's device minibatch buffers.
  - `is_ready[BATCH]() -> Bool` mirrors `size >= BATCH`.

RNG: each `sample` call bumps `_rng_offset` by `2 * BATCH` so
back-to-back calls draw disjoint Philox streams. Pattern mirrors
RSample's device RNG handling (no shared global state).

Bit-identity: there is no GPU bit-identity baseline. CPU SAC path is
untouched, so the `-167.572` Pendulum 30k CPU baseline stays bit-
identical by construction. GPU SAC convergence is gated by
`test_sac_pendulum_gpu_convergence.mojo` (mean10 > -200).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT, TPB
from ..training.replay_buffer import ReplayBuffer
from ..training.trainer_block import TrainerState


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


def _store_one_kernel[OBS: Int, ACT: Int, CAP: Int](
    stage_s: LayoutTensor[DT, Layout.row_major(OBS), MutAnyOrigin],
    stage_a: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    stage_r: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    stage_sp: LayoutTensor[DT, Layout.row_major(OBS), MutAnyOrigin],
    stage_d: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_sp: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    slot: Int32,
):
    """Single-block kernel: thread d writes stage[d] → buf[slot, d] for
    obs / next_obs (OBS lanes) and act (ACT lanes); thread 0 also
    writes the scalar rew / dne fields.

    Launched as grid=(1,), block=(max(OBS, ACT, 1),). Threads outside
    their field's range no-op for that field — keeps the kernel a
    single dispatch instead of three separate launches.
    """
    var d = Int(thread_idx.x)
    var s = Int(slot)
    if d < OBS:
        buf_s[s, d] = stage_s[d]
        buf_sp[s, d] = stage_sp[d]
    if d < ACT:
        buf_a[s, d] = stage_a[d]
    if d == 0:
        buf_r[s] = stage_r[0]
        buf_d[s] = stage_d[0]


def _increment_rng_offset_kernel[BATCH: Int](
    offset: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Bump the device RNG offset by `2 * BATCH` (the same stride the host
    used). Launch grid=(1,), block=(1,). Enqueued AFTER the sample kernel
    so the sample reads offset N and the next call reads N + 2·BATCH —
    bit-identical to the old host `_rng_offset += 2·BATCH`, but now the
    counter lives on-device so it advances on every CUDA-graph replay
    without host intervention."""
    if Int(thread_idx.x) == 0:
        offset[0] = offset[0] + UInt64(BATCH * 2)


def _bump_size_kernel[CAP: Int](
    size_buf: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    n: Int32,
):
    """Saturating bump of the device-resident transition count by `n`
    (clamped to CAP). Launch grid=(1,), block=(1,). Enqueued on the eager
    `add` / `add_batch` path so the device counter mirrors the host
    `self.size`. The uniform sample kernel reads THIS (live, device) count
    instead of a host scalar — which is what makes sampling correct under
    CUDA-graph capture: a baked host `size` would freeze the sample range to
    the buffer's capture-time fill (≈ warmup), starving training of all newer
    experience. Bit-identical to the host `size` on the non-captured path."""
    if Int(thread_idx.x) == 0:
        var s = Int(size_buf[0]) + Int(n)
        if s > CAP:
            s = CAP
        size_buf[0] = Scalar[DType.int32](s)


def _sample_indices_range_kernel[BATCH: Int](
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    lo: Int32,
    hi: Int32,
    seed: UInt64,
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Philox uniform → integer index in `[lo, hi)`. Host-passed range (the
    MBPO dyn train/holdout split is NOT CUDA-graph captured, so host scalars
    are fine here). Advances the same `offset_buf` counter as
    `_sample_indices_kernel`."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var lo_i = Int(lo)
    var hi_i = Int(hi)
    var span = hi_i - lo_i
    if span < 1:
        span = 1
    var idx = lo_i + Int(u * Float32(span))
    if idx >= hi_i:
        idx = hi_i - 1
    if idx < lo_i:
        idx = lo_i
    indices[i] = Scalar[DType.int32](idx)


def _sample_indices_kernel[BATCH: Int](
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    size_buf: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    seed: UInt64,
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Per-thread Philox uniform → integer index in `[0, size)`.

    Seeding follows the deep_agents `sample_indices_kernel` pattern —
    one Philox stream per lane, seed mixed with thread index to avoid
    cross-lane collisions. The Philox `offset_base` is read from the
    device buffer `offset_buf[0]` (Slice 5 — CUDA-graph capturable) and
    advanced by `_increment_rng_offset_kernel` after this kernel.

    `size` (current buffer fill) is read from the device buffer `size_buf[0]`
    — NOT a host scalar — so the sample range tracks the LIVE count on every
    CUDA-graph replay. A baked host `size` would freeze sampling to the
    capture-time fill (≈ warmup), the catastrophic-divergence bug.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var size = Int(size_buf[0])
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var idx = Int(u * Float32(size))
    if idx >= size:
        idx = size - 1
    if idx < 0:
        idx = 0
    indices[i] = Scalar[DType.int32](idx)


def _sample_indices_ere_kernel[BATCH: Int, CAP: Int](
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    size: Int32,
    write_pos: Int32,
    c_k: Int32,
    seed: UInt64,
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """ERE-biased Philox uniform: index lands uniformly within the
    most recent `min(c_k, size)` slots of the circular buffer.

    Per-thread: draw `u ~ Uniform[0, 1)`, then
    `idx = (write_pos − c_k + floor(u·c_k) + CAP) % CAP`. When
    `c_k == size`, the distribution is identical to uniform over
    the whole buffer (just rotated), so this kernel is safe to use
    when ERE is effectively off.

    Mirrors `deep_agents`' `sample_indices_ere_kernel` but takes
    `c_k`, `size`, and `write_pos` as host scalar Int32 kernel args
    (the host computes `c_k`/anneals per call).

    WARNING — NOT CUDA-graph-safe: `size`, `write_pos`, and `c_k` all
    change every call, so under capture they would be baked at capture time
    and the sample distribution would freeze (the same class of bug that the
    uniform path's device-resident `size_buf` fixes). Do NOT combine ERE
    (`enable_ere`) with `USE_TRAIN_CUDA_GRAPH`. Making ERE capturable would
    require device-resident `size`/`write_pos`/`c_k` (and an on-device
    anneal). The uniform path IS capture-safe.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var c = Int(c_k)
    var sz = Int(size)
    if c > sz:
        c = sz
    if c < 1:
        c = 1
    var offset = Int(u * Float32(c))
    if offset >= c:
        offset = c - 1
    if offset < 0:
        offset = 0
    var idx = (Int(write_pos) - c + offset + CAP) % CAP
    if idx < 0:
        idx = idx + CAP
    indices[i] = Scalar[DType.int32](idx)


def _gather_batch_kernel[
    BATCH: Int, OBS: Int, ACT: Int, CAP: Int,
](
    mb_s: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    mb_a: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
    mb_r: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    mb_sp: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    mb_d: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_sp: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
):
    """Per-thread gather: thread `i` reads `indices[i] = idx` then
    copies obs/act/rew/nxt/dne[idx] into mb[i, ...].

    Launched as grid=(ceil(BATCH/TPB),), block=(TPB,). One thread per
    batch lane; OBS/ACT loops sequential within each thread (dims are
    small in continuous control — Pendulum is 3/1).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var idx = Int(indices[i])
    for d in range(OBS):
        mb_s[i, d] = buf_s[idx, d]
        mb_sp[i, d] = buf_sp[idx, d]
    for j in range(ACT):
        mb_a[i, j] = buf_a[idx, j]
    mb_r[i] = buf_r[idx]
    mb_d[i] = buf_d[idx]


def _store_batch_kernel[
    N_ENVS: Int, OBS: Int, ACT: Int, CAP: Int,
](
    src_obs: LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin],
    src_act: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    src_rew: LayoutTensor[DT, Layout.row_major(N_ENVS), MutAnyOrigin],
    src_nxt: LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin],
    src_dne: LayoutTensor[DT, Layout.row_major(N_ENVS), MutAnyOrigin],
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_sp: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    start_pos: Int32,
):
    """Batched store: thread `e` writes env e's transition into slot
    `(start_pos + e) % CAP`. Mirrors `_store_one_kernel` shape but
    with one lane per env.

    Launched as grid=(ceil(N_ENVS/TPB),), block=(TPB,). OBS/ACT loops
    sequential within each thread.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var slot = (Int(start_pos) + e) % CAP
    for d in range(OBS):
        buf_s[slot, d] = src_obs[e, d]
        buf_sp[slot, d] = src_nxt[e, d]
    for j in range(ACT):
        buf_a[slot, j] = src_act[e, j]
    buf_r[slot] = src_rew[e]
    buf_d[slot] = src_dne[e]


# ──────────────────────────────────────────────────────────────────────
# GPUReplay struct.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct GPUReplay[OBS_: Int, ACT_: Int, CAP_: Int](ReplayBuffer):
    """Device-resident circular replay buffer.

    Stored layout:
      obs  : [CAP, OBS] row-major DeviceBuffer
      act  : [CAP, ACT] row-major DeviceBuffer
      rew  : [CAP] DeviceBuffer
      nxt  : [CAP, OBS] row-major DeviceBuffer
      dne  : [CAP] DeviceBuffer

    CPU bookkeeping: `pos` (next write slot), `size` (saturates at CAP
    after first wrap), `rng_seed`, `_rng_offset`.

    Conforms to `ReplayBuffer`: `make` / `add(Lists, ctx)` /
    `sample_into` / `count` are the trait surface; the legacy
    `new` / `add(ctx, ptrs)` / `sample` methods are retained for callers
    that pre-date the trait.

    `OBS` and `ACT` are dimensions (not buffer sizes); for Pendulum,
    `GPUReplay[3, 1, 50000]`.
    """

    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    # Device-resident circular storage.
    var obs: DeviceBuffer[DT]
    var act: DeviceBuffer[DT]
    var rew: DeviceBuffer[DT]
    var nxt: DeviceBuffer[DT]
    var dne: DeviceBuffer[DT]

    # Single-transition device staging (used by `add`).
    var stage_obs: DeviceBuffer[DT]
    var stage_act: DeviceBuffer[DT]
    var stage_rew: DeviceBuffer[DT]
    var stage_nxt: DeviceBuffer[DT]
    var stage_dne: DeviceBuffer[DT]

    # Sample-side index scratch (sized to `batch_capacity`).
    var indices: DeviceBuffer[DType.int32]

    # Host scratch for the scalar fields (rew, dne) so `add` can issue
    # a uniform host→device enqueue_copy for all five fields. 1 scalar
    # each.
    var _h_rew: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _h_dne: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # CPU bookkeeping.
    var size: Int
    var pos: Int
    var rng_seed: UInt64
    # Slice 5 — device-resident Philox offset (was a host `UInt64` bumped
    # on the CPU). A 1-elem uint64 buffer the sample kernel reads and the
    # `_increment_rng_offset_kernel` advances on-device → CUDA-graph
    # capturable. The offset sequence (k·2·BATCH) is unchanged, so the
    # sampled-index stream is bit-identical to the old host-counter path.
    var _rng_offset_dev: DeviceBuffer[DType.uint64]

    # Device-resident mirror of `size` (1-elem int32), bumped by
    # `_bump_size_kernel` on every `add` / `add_batch`. The uniform sample
    # kernel reads THIS rather than the host `size` scalar so the sample range
    # tracks the live fill under CUDA-graph capture (a host scalar would be
    # baked at capture time → sampling frozen to the warmup-era fill). The host
    # `size` is retained for the eager `count()` / `is_ready` gates.
    var _size_dev: DeviceBuffer[DType.int32]

    # Cached BATCH ceiling for `indices` buffer sizing.
    var batch_capacity: Int

    # Phase C.4 — ERE (Emphasizing Recent Experience) state. Disabled
    # by default; `enable_ere(...)` flips `ere_enabled` and sets eta /
    # c_min. The cycle counter `_ere_k` wraps at `_ere_k_max` (Wang &
    # Ross default 1000) — at each `sample` call, `_ere_eta_pow_k`
    # multiplies by `ere_eta` and `_ere_k` increments; on wrap both
    # reset. `c_k = clamp(floor(size · _ere_eta_pow_k), c_min, size)`.
    var ere_enabled: Bool
    var ere_eta: Scalar[DT]
    var _ere_k: Int
    var _ere_k_max: Int
    var _ere_eta_pow_k: Scalar[DT]
    var _ere_c_min: Int

    @staticmethod
    def new(ctx: DeviceContext, batch_capacity: Int = 4096) raises -> Self:
        """Allocate all device buffers + staging + index scratch.
        Zero-fills the circular storage. Staging buffers are not zeroed
        (every `add` rewrites them before the store kernel reads).
        """
        var s = ctx.enqueue_create_buffer[DT](Self.CAP * Self.OBS)
        var a = ctx.enqueue_create_buffer[DT](Self.CAP * Self.ACT)
        var r = ctx.enqueue_create_buffer[DT](Self.CAP)
        var sp = ctx.enqueue_create_buffer[DT](Self.CAP * Self.OBS)
        var d = ctx.enqueue_create_buffer[DT](Self.CAP)
        s.enqueue_fill(Scalar[DT](0.0))
        a.enqueue_fill(Scalar[DT](0.0))
        r.enqueue_fill(Scalar[DT](0.0))
        sp.enqueue_fill(Scalar[DT](0.0))
        d.enqueue_fill(Scalar[DT](0.0))

        var stage_s = ctx.enqueue_create_buffer[DT](Self.OBS)
        var stage_a = ctx.enqueue_create_buffer[DT](Self.ACT)
        var stage_r = ctx.enqueue_create_buffer[DT](1)
        var stage_sp = ctx.enqueue_create_buffer[DT](Self.OBS)
        var stage_d = ctx.enqueue_create_buffer[DT](1)

        var idx_buf = ctx.enqueue_create_buffer[DType.int32](batch_capacity)

        var rng_off = ctx.enqueue_create_buffer[DType.uint64](1)
        rng_off.enqueue_fill(UInt64(0))

        var size_dev = ctx.enqueue_create_buffer[DType.int32](1)
        size_dev.enqueue_fill(Int32(0))

        var hr = alloc[Scalar[DT]](1)
        var hd = alloc[Scalar[DT]](1)
        hr[0] = Scalar[DT](0.0)
        hd[0] = Scalar[DT](0.0)

        return Self(
            obs=s^, act=a^, rew=r^, nxt=sp^, dne=d^,
            stage_obs=stage_s^, stage_act=stage_a^, stage_rew=stage_r^,
            stage_nxt=stage_sp^, stage_dne=stage_d^,
            indices=idx_buf^,
            _h_rew=hr, _h_dne=hd,
            size=0, pos=0,
            rng_seed=UInt64(0xC0FFEE_DECADE_0042),
            _rng_offset_dev=rng_off^,
            _size_dev=size_dev^,
            batch_capacity=batch_capacity,
            ere_enabled=False,
            ere_eta=Scalar[DT](0.996),
            _ere_k=0,
            _ere_k_max=1000,
            _ere_eta_pow_k=Scalar[DT](1.0),
            _ere_c_min=1,
        )

    # ─── ReplayBuffer trait surface ──────────────────────────────────

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        """Trait factory. `ctx` is required (GPU storage); raises if
        None. `batch_capacity` sizes the sample-side index scratch."""
        if not ctx:
            raise Error("GPUReplay.make: ctx required for device storage")
        return Self.new(ctx.value(), batch_capacity=batch_capacity)

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Trait-surface add: stage the host Lists to device and reuse
        the pointer-based `add`. `ctx` required (raises if None)."""
        if not ctx:
            raise Error("GPUReplay.add: ctx required for device storage")
        var s_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            s.unsafe_ptr()
        )
        var a_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            a.unsafe_ptr()
        )
        var sp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            sp.unsafe_ptr()
        )
        self.add(ctx.value(), s_p, a_p, r, sp_p, d)

    def sample_into[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        """Trait-surface sampling: launch the gather into the device
        mirrors of `state.mb_*` using `state.ctx`."""
        self.sample[BATCH](
            state.ctx.value(),
            state.mb_s.dev.value(),
            state.mb_a.dev.value(),
            state.mb_r.dev.value(),
            state.mb_sp.dev.value(),
            state.mb_d.dev.value(),
        )

    def count(self) -> Int:
        return self.size

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        """Trait override: flip ERE state. Delegates to `enable_ere` /
        `disable_ere` so the generic sample block can configure recency
        bias without knowing the concrete backend."""
        if enable:
            self.enable_ere(eta=eta, c_min=c_min, k_max=k_max)
        else:
            self.disable_ere()

    def enable_ere(
        mut self,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ):
        """Enable ERE recency-biased sampling (Wang & Ross 2019).

        With ERE on, `sample[BATCH]` draws indices uniformly from the
        most recent `c_k = clamp(floor(size · η^k), c_min, size)`
        transitions; `k` cycles 0 → `k_max − 1` and resets, so over a
        full cycle the sampler smoothly anneals from "very recent
        only" back to "full buffer."

        Args:
            eta: Decay factor (paper default 0.996). Smaller = stronger
                bias toward recent.
            c_min: Lower clamp on the recent window. Recommended at
                least the trainer's BATCH so each sample has enough
                slots to draw from.
            k_max: Cycle length. After `k_max` calls `k` resets to 0
                and `η^k` resets to 1.0.

        No-op when `eta == 1.0` (no bias) but the kernel path still
        runs; for true bit-identity with the non-ERE sampler, leave
        ERE off."""
        self.ere_enabled = True
        self.ere_eta = eta
        self._ere_c_min = c_min
        self._ere_k_max = k_max
        self._ere_k = 0
        self._ere_eta_pow_k = Scalar[DT](1.0)

    def disable_ere(mut self):
        """Switch back to uniform sampling."""
        self.ere_enabled = False

    def is_ready[BATCH: Int](self) -> Bool:
        return self.size >= BATCH

    def add(
        mut self,
        ctx: DeviceContext,
        s: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        sp: UnsafePointer[Scalar[DT], MutAnyOrigin],
        d: Scalar[DT],
    ) raises:
        """Push one transition (s, a, r, s', done) into the circular
        buffer at slot `pos`.

        Sequence:
          1. Stash `r` and `d` in per-instance host scratch pointers.
          2. 5 × `enqueue_copy` host_ptr → 1-slot device staging.
          3. 1 × `_store_one_kernel` writes stage → buf[pos, ...].
          4. CPU updates `pos = (pos + 1) % CAP`, `size = min(size+1, CAP)`.

        Cost per call: 5 small D2H + 1 tiny kernel. On Apple Metal the
        D2H bytes are negligible vs. kernel launch overhead; on NVIDIA
        the launch overhead is small enough that batching this would
        only matter at very high record rates.
        """
        self._h_rew[0] = r
        self._h_dne[0] = d
        ctx.enqueue_copy(self.stage_obs, s)
        ctx.enqueue_copy(self.stage_act, a)
        ctx.enqueue_copy(self.stage_rew, self._h_rew)
        ctx.enqueue_copy(self.stage_nxt, sp)
        ctx.enqueue_copy(self.stage_dne, self._h_dne)

        var stage_s_lt = LayoutTensor[
            DT, Layout.row_major(Self.OBS), MutAnyOrigin
        ](self.stage_obs.unsafe_ptr())
        var stage_a_lt = LayoutTensor[
            DT, Layout.row_major(Self.ACT), MutAnyOrigin
        ](self.stage_act.unsafe_ptr())
        var stage_r_lt = LayoutTensor[
            DT, Layout.row_major(1), MutAnyOrigin
        ](self.stage_rew.unsafe_ptr())
        var stage_sp_lt = LayoutTensor[
            DT, Layout.row_major(Self.OBS), MutAnyOrigin
        ](self.stage_nxt.unsafe_ptr())
        var stage_d_lt = LayoutTensor[
            DT, Layout.row_major(1), MutAnyOrigin
        ](self.stage_dne.unsafe_ptr())
        var buf_s_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.obs.unsafe_ptr())
        var buf_a_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.ACT), MutAnyOrigin
        ](self.act.unsafe_ptr())
        var buf_r_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.rew.unsafe_ptr())
        var buf_sp_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.nxt.unsafe_ptr())
        var buf_d_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.dne.unsafe_ptr())

        # Single-block kernel. TPB = max(OBS, ACT) so each lane has a
        # thread; rew/dne are written by lane 0 only.
        comptime TPB = Self.OBS if Self.OBS > Self.ACT else Self.ACT
        comptime kernel = _store_one_kernel[Self.OBS, Self.ACT, Self.CAP]
        ctx.enqueue_function[kernel](
            stage_s_lt, stage_a_lt, stage_r_lt, stage_sp_lt, stage_d_lt,
            buf_s_lt, buf_a_lt, buf_r_lt, buf_sp_lt, buf_d_lt,
            Int32(self.pos),
            grid_dim=1, block_dim=TPB,
        )

        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

        # Mirror the host `size` bump onto the device counter the (possibly
        # CUDA-graph-captured) sample kernel reads.
        var size_lt = LayoutTensor[
            DType.int32, Layout.row_major(1), MutAnyOrigin
        ](self._size_dev.unsafe_ptr())
        ctx.enqueue_function[_bump_size_kernel[Self.CAP]](
            size_lt, Int32(1), grid_dim=1, block_dim=1,
        )

    def add_batch[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_nxt: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """Push `N_ENVS` transitions in one kernel launch.

        Source buffers come from the GPU env step (already device-
        resident); no host→device copies are needed. One kernel writes
        all N_ENVS slots, each at `(pos + e) % CAP`. CPU updates
        `pos = (pos + N_ENVS) % CAP` and saturates `size` at CAP.

        Used by `run_offpolicy_train_gpu_n_envs` (B.5b) — replaces the
        N successive `add` calls that an N_ENVS=N driver would
        otherwise make (one kernel + zero D2H vs N kernels + 5N D2H).
        """
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        var src_obs_lt = LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](src_obs.unsafe_ptr())
        var src_act_lt = LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.ACT), MutAnyOrigin
        ](src_act.unsafe_ptr())
        var src_rew_lt = LayoutTensor[
            DT, Layout.row_major(N_ENVS), MutAnyOrigin
        ](src_rew.unsafe_ptr())
        var src_nxt_lt = LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](src_nxt.unsafe_ptr())
        var src_dne_lt = LayoutTensor[
            DT, Layout.row_major(N_ENVS), MutAnyOrigin
        ](src_dne.unsafe_ptr())
        var buf_s_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.obs.unsafe_ptr())
        var buf_a_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.ACT), MutAnyOrigin
        ](self.act.unsafe_ptr())
        var buf_r_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.rew.unsafe_ptr())
        var buf_sp_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.nxt.unsafe_ptr())
        var buf_d_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.dne.unsafe_ptr())

        comptime n_blocks = (N_ENVS + TPB - 1) // TPB
        comptime kernel = _store_batch_kernel[
            N_ENVS, Self.OBS, Self.ACT, Self.CAP
        ]
        ctx.enqueue_function[kernel](
            src_obs_lt, src_act_lt, src_rew_lt, src_nxt_lt, src_dne_lt,
            buf_s_lt, buf_a_lt, buf_r_lt, buf_sp_lt, buf_d_lt,
            Int32(self.pos),
            grid_dim=n_blocks, block_dim=TPB,
        )
        self.pos = (self.pos + N_ENVS) % Self.CAP
        self.size += N_ENVS
        if self.size > Self.CAP:
            self.size = Self.CAP

        # Mirror the host `size` bump onto the device counter the (possibly
        # CUDA-graph-captured) sample kernel reads.
        var size_lt = LayoutTensor[
            DType.int32, Layout.row_major(1), MutAnyOrigin
        ](self._size_dev.unsafe_ptr())
        ctx.enqueue_function[_bump_size_kernel[Self.CAP]](
            size_lt, Int32(N_ENVS), grid_dim=1, block_dim=1,
        )

    def sample[BATCH: Int](
        mut self,
        ctx: DeviceContext,
        mb_s: DeviceBuffer[DT],
        mb_a: DeviceBuffer[DT],
        mb_r: DeviceBuffer[DT],
        mb_sp: DeviceBuffer[DT],
        mb_d: DeviceBuffer[DT],
    ) raises:
        """Uniform-with-replacement sample of `BATCH` transitions into
        caller-provided device minibatch buffers.

        Two kernels enqueued in order: indices (Philox) → gather. Both
        launched as `grid=(ceil(BATCH/TPB),), block=(TPB,)` with TPB=128.

        BATCH must be ≤ `batch_capacity` from construction (the
        pre-allocated `indices` scratch).
        """
        comptime assert BATCH > 0, "BATCH must be > 0"
        if BATCH > self.batch_capacity:
            raise Error(
                "GPUReplay.sample[BATCH=" + String(BATCH)
                + "] exceeds batch_capacity=" + String(self.batch_capacity)
            )

        var idx_lt = LayoutTensor[
            DType.int32, Layout.row_major(BATCH), MutAnyOrigin
        ](self.indices.unsafe_ptr())
        var off_lt = LayoutTensor[
            DType.uint64, Layout.row_major(1), MutAnyOrigin
        ](self._rng_offset_dev.unsafe_ptr())
        comptime n_blocks = (BATCH + TPB - 1) // TPB
        if self.ere_enabled:
            # Host-side compute c_k = clamp(floor(size · η^k), c_min, size).
            var c = Int(
                Scalar[DT](self.size) * self._ere_eta_pow_k
            )
            if c < self._ere_c_min:
                c = self._ere_c_min
            if c > self.size:
                c = self.size
            if c < 1:
                c = 1
            comptime ere_kernel = _sample_indices_ere_kernel[
                BATCH, Self.CAP,
            ]
            ctx.enqueue_function[ere_kernel](
                idx_lt,
                Int32(self.size),
                Int32(self.pos),
                Int32(c),
                self.rng_seed, off_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
            # Advance k, η^k; wrap at k_max.
            self._ere_k = self._ere_k + 1
            self._ere_eta_pow_k = self._ere_eta_pow_k * self.ere_eta
            if self._ere_k >= self._ere_k_max:
                self._ere_k = 0
                self._ere_eta_pow_k = Scalar[DT](1.0)
        else:
            var size_lt = LayoutTensor[
                DType.int32, Layout.row_major(1), MutAnyOrigin
            ](self._size_dev.unsafe_ptr())
            comptime indices_kernel = _sample_indices_kernel[BATCH]
            ctx.enqueue_function[indices_kernel](
                idx_lt, size_lt,
                self.rng_seed, off_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
        # Bump RNG offset on-device (after the sample kernel reads it) so
        # back-to-back calls draw disjoint streams — CUDA-graph capturable.
        comptime inc_kernel = _increment_rng_offset_kernel[BATCH]
        ctx.enqueue_function[inc_kernel](
            off_lt, grid_dim=1, block_dim=1,
        )

        var mb_s_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](mb_s.unsafe_ptr())
        var mb_a_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.ACT), MutAnyOrigin
        ](mb_a.unsafe_ptr())
        var mb_r_lt = LayoutTensor[
            DT, Layout.row_major(BATCH), MutAnyOrigin
        ](mb_r.unsafe_ptr())
        var mb_sp_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](mb_sp.unsafe_ptr())
        var mb_d_lt = LayoutTensor[
            DT, Layout.row_major(BATCH), MutAnyOrigin
        ](mb_d.unsafe_ptr())
        var buf_s_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.obs.unsafe_ptr())
        var buf_a_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.ACT), MutAnyOrigin
        ](self.act.unsafe_ptr())
        var buf_r_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.rew.unsafe_ptr())
        var buf_sp_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.nxt.unsafe_ptr())
        var buf_d_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.dne.unsafe_ptr())

        comptime gather_kernel = _gather_batch_kernel[
            BATCH, Self.OBS, Self.ACT, Self.CAP
        ]
        ctx.enqueue_function[gather_kernel](
            mb_s_lt, mb_a_lt, mb_r_lt, mb_sp_lt, mb_d_lt,
            buf_s_lt, buf_a_lt, buf_r_lt, buf_sp_lt, buf_d_lt,
            idx_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )

    def sample_range[BATCH: Int](
        mut self,
        ctx: DeviceContext,
        lo: Int,
        hi: Int,
        mb_s: DeviceBuffer[DT],
        mb_a: DeviceBuffer[DT],
        mb_r: DeviceBuffer[DT],
        mb_sp: DeviceBuffer[DT],
        mb_d: DeviceBuffer[DT],
    ) raises:
        """Like `sample`, but draws indices uniformly from `[lo, hi)`
        instead of the whole live buffer — used by MBPO dynamics training to
        enforce a fixed train/holdout split. Not CUDA-graph captured, so the
        host `lo`/`hi` range is fine. Ignores ERE."""
        comptime assert BATCH > 0, "BATCH must be > 0"
        if BATCH > self.batch_capacity:
            raise Error(
                "GPUReplay.sample_range[BATCH=" + String(BATCH)
                + "] exceeds batch_capacity=" + String(self.batch_capacity)
            )
        var idx_lt = LayoutTensor[
            DType.int32, Layout.row_major(BATCH), MutAnyOrigin
        ](self.indices.unsafe_ptr())
        var off_lt = LayoutTensor[
            DType.uint64, Layout.row_major(1), MutAnyOrigin
        ](self._rng_offset_dev.unsafe_ptr())
        comptime n_blocks = (BATCH + TPB - 1) // TPB
        comptime range_kernel = _sample_indices_range_kernel[BATCH]
        ctx.enqueue_function[range_kernel](
            idx_lt, Int32(lo), Int32(hi), self.rng_seed, off_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )
        comptime inc_kernel = _increment_rng_offset_kernel[BATCH]
        ctx.enqueue_function[inc_kernel](off_lt, grid_dim=1, block_dim=1)

        var mb_s_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](mb_s.unsafe_ptr())
        var mb_a_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.ACT), MutAnyOrigin
        ](mb_a.unsafe_ptr())
        var mb_r_lt = LayoutTensor[
            DT, Layout.row_major(BATCH), MutAnyOrigin
        ](mb_r.unsafe_ptr())
        var mb_sp_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](mb_sp.unsafe_ptr())
        var mb_d_lt = LayoutTensor[
            DT, Layout.row_major(BATCH), MutAnyOrigin
        ](mb_d.unsafe_ptr())
        var buf_s_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.obs.unsafe_ptr())
        var buf_a_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.ACT), MutAnyOrigin
        ](self.act.unsafe_ptr())
        var buf_r_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.rew.unsafe_ptr())
        var buf_sp_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.nxt.unsafe_ptr())
        var buf_d_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.dne.unsafe_ptr())
        comptime gather_kernel = _gather_batch_kernel[
            BATCH, Self.OBS, Self.ACT, Self.CAP
        ]
        ctx.enqueue_function[gather_kernel](
            mb_s_lt, mb_a_lt, mb_r_lt, mb_sp_lt, mb_d_lt,
            buf_s_lt, buf_a_lt, buf_r_lt, buf_sp_lt, buf_d_lt,
            idx_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )
