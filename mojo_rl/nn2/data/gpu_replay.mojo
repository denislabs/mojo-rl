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

from ..constants import DT


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


def _sample_indices_kernel[BATCH: Int](
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH), MutAnyOrigin
    ],
    size: Int32,
    seed: UInt64,
    offset_base: UInt64,
):
    """Per-thread Philox uniform → integer index in `[0, size)`.

    Seeding follows the deep_agents `sample_indices_kernel` pattern —
    one Philox stream per lane, seed mixed with thread index to avoid
    cross-lane collisions. `offset_base` is bumped on the CPU between
    calls so back-to-back samples draw fresh streams.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var idx = Int(u * Float32(size))
    if idx >= Int(size):
        idx = Int(size) - 1
    if idx < 0:
        idx = 0
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


# ──────────────────────────────────────────────────────────────────────
# GPUReplay struct.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct GPUReplay[OBS: Int, ACT: Int, CAP: Int](
    Movable & ImplicitlyDestructible
):
    """Device-resident circular replay buffer.

    Stored layout:
      obs  : [CAP, OBS] row-major DeviceBuffer
      act  : [CAP, ACT] row-major DeviceBuffer
      rew  : [CAP] DeviceBuffer
      nxt  : [CAP, OBS] row-major DeviceBuffer
      dne  : [CAP] DeviceBuffer

    CPU bookkeeping: `pos` (next write slot), `size` (saturates at CAP
    after first wrap), `rng_seed`, `_rng_offset`.

    `OBS` and `ACT` are dimensions (not buffer sizes); for Pendulum,
    `GPUReplay[3, 1, 50000]`.
    """

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
    var _rng_offset: UInt64

    # Cached BATCH ceiling for `indices` buffer sizing.
    var batch_capacity: Int

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
            _rng_offset=UInt64(0),
            batch_capacity=batch_capacity,
        )

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
        comptime TPB = 128
        comptime n_blocks = (BATCH + TPB - 1) // TPB
        comptime indices_kernel = _sample_indices_kernel[BATCH]
        ctx.enqueue_function[indices_kernel](
            idx_lt, Int32(self.size), self.rng_seed, self._rng_offset,
            grid_dim=n_blocks, block_dim=TPB,
        )
        # Bump RNG offset so back-to-back calls draw disjoint streams.
        self._rng_offset += UInt64(BATCH * 2)

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
