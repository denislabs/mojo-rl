"""GPUSequenceReplay[OBS, ACT, CAP] — device-resident circular sequence
replay buffer; the GPU sibling of `SequenceReplay`.

Single flat ring on device (same layout / semantics as the CPU buffer, not
the per-env layout of the deep_agents v1 buffer) so CPU and GPU are true
siblings — same start distribution, same window contents, only the RNG
differs (Philox here vs `random_float64` on CPU, so no bit-identity; there
is no GPU bit-identity baseline, mirroring `GPUReplay`).

Stored layout (4 DeviceBuffers, each CAP slots):
  obs  [CAP, OBS] row-major
  act  [CAP, ACT] row-major
  rew  [CAP]
  dne  [CAP]

Surface (conforms to `SequenceReplayBuffer`):
  - `make["gpu"](ctx)`                  — allocate + zero device storage.
  - `record(s, a, r, d)`                — host ptrs → 1-slot staging →
                                          store-one kernel at `pos`.
  - `record_batch[N_ENVS](ctx, …)`      — device sources → one store kernel
                                          writing N_ENVS lockstep slots.
  - `sample_batch_dev[B, T](ctx, …)`    — Philox window sample directly into
                                          caller device buffers (no D2H).
  - `sample_batch[B, T](host ptrs)`     — device sample then D2H copy-out via
                                          the stored `ctx`; lets a GPU buffer
                                          feed the current CPU world-model
                                          pipeline.

RNG: each `sample_batch_dev` bumps `_rng_offset` so back-to-back samples draw
disjoint Philox streams (pattern from `GPUReplay`).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT, TPB
from .sequence_replay_buffer import SequenceReplayBuffer


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


def _seq_store_one_kernel[OBS: Int, ACT: Int, CAP: Int](
    stage_s: LayoutTensor[DT, Layout.row_major(OBS), MutAnyOrigin],
    stage_a: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    stage_r: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    stage_d: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    slot: Int32,
):
    """Single-block store: thread d writes stage[d] → buf[slot, d] for obs
    (OBS lanes) and act (ACT lanes); lane 0 writes the scalar rew / dne.
    Launched grid=(1,), block=(max(OBS, ACT),)."""
    var d = Int(thread_idx.x)
    var s = Int(slot)
    if d < OBS:
        buf_s[s, d] = stage_s[d]
    if d < ACT:
        buf_a[s, d] = stage_a[d]
    if d == 0:
        buf_r[s] = stage_r[0]
        buf_d[s] = stage_d[0]


def _seq_store_batch_kernel[
    N_ENVS: Int, OBS: Int, ACT: Int, CAP: Int,
](
    src_obs: LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin],
    src_act: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    src_rew: LayoutTensor[DT, Layout.row_major(N_ENVS), MutAnyOrigin],
    src_dne: LayoutTensor[DT, Layout.row_major(N_ENVS), MutAnyOrigin],
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    start_pos: Int32,
):
    """Batched store: thread `e` writes env e's transition into slot
    `(start_pos + e) % CAP`. Launched grid=(ceil(N_ENVS/TPB),), block=(TPB,)."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var slot = (Int(start_pos) + e) % CAP
    for d in range(OBS):
        buf_s[slot, d] = src_obs[e, d]
    for j in range(ACT):
        buf_a[slot, j] = src_act[e, j]
    buf_r[slot] = src_rew[e]
    buf_d[slot] = src_dne[e]


def _increment_rng_offset_kernel[B: Int](
    offset: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Bump the device RNG offset by `2 * B` (the host stride). Launch
    grid=(1,), block=(1,); enqueued after the sample kernel so the
    offset sequence is bit-identical to the old host `_rng_offset += 2·B`
    — now device-resident for CUDA-graph capture."""
    if Int(thread_idx.x) == 0:
        offset[0] = offset[0] + UInt64(B * 2)


def _seq_sample_kernel[
    B: Int, T: Int, OBS: Int, ACT: Int, CAP: Int,
](
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    out_obs: LayoutTensor[DT, Layout.row_major(B, T + 1, OBS), MutAnyOrigin],
    out_act: LayoutTensor[DT, Layout.row_major(B, T, ACT), MutAnyOrigin],
    out_rew: LayoutTensor[DT, Layout.row_major(B, T), MutAnyOrigin],
    out_dne: LayoutTensor[DT, Layout.row_major(B, T), MutAnyOrigin],
    origin: Int32,
    n_valid: Int32,
    seed: UInt64,
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Per-thread window gather: thread `b` draws a start `s ∈ [0, n_valid)`
    via Philox, then copies the T+1 obs frames + T act/rew/dne frames from
    physical slots `(origin + s + k) % CAP`.

    Matches the CPU `SequenceReplay.sample_batch` index math exactly; only
    the RNG differs. Launched grid=(ceil(B/TPB),), block=(TPB,)."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= B:
        return

    var nv = Int(n_valid)
    if nv < 1:
        nv = 1
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(b), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var s = Int(u * Float32(nv))
    if s >= nv:
        s = nv - 1
    if s < 0:
        s = 0

    var org = Int(origin)
    # T+1 observation frames.
    for k in range(T + 1):
        var phys = (org + s + k) % CAP
        for i in range(OBS):
            out_obs[b, k, i] = buf_s[phys, i]
    # T action / reward / done frames.
    for k in range(T):
        var phys = (org + s + k) % CAP
        for j in range(ACT):
            out_act[b, k, j] = buf_a[phys, j]
        out_rew[b, k] = buf_r[phys]
        out_dne[b, k] = buf_d[phys]


# ──────────────────────────────────────────────────────────────────────
# GPUSequenceReplay struct.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct GPUSequenceReplay[OBS_: Int, ACT_: Int, CAP_: Int](
    SequenceReplayBuffer
):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    # Device-resident circular storage.
    var obs: DeviceBuffer[DT]
    var act: DeviceBuffer[DT]
    var rew: DeviceBuffer[DT]
    var dne: DeviceBuffer[DT]

    # Single-transition device staging (used by `record`).
    var stage_obs: DeviceBuffer[DT]
    var stage_act: DeviceBuffer[DT]
    var stage_rew: DeviceBuffer[DT]
    var stage_dne: DeviceBuffer[DT]

    # Host scratch for scalar staging so `record` issues uniform H2D copies.
    var _h_rew: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _h_dne: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # Stored context — needed by the host `sample_batch` bridge (no ctx arg).
    var ctx: DeviceContext

    # CPU bookkeeping.
    var size: Int
    var pos: Int
    var rng_seed: UInt64
    # Slice 5 — device-resident Philox offset (was a host `UInt64`). The
    # sample kernel reads `_rng_offset_dev[0]`; `_increment_rng_offset_kernel`
    # advances it on-device → CUDA-graph capturable. Offset sequence
    # (k·2·B) unchanged → sampled windows bit-identical to the host path.
    var _rng_offset_dev: DeviceBuffer[DType.uint64]

    @staticmethod
    def new(ctx: DeviceContext) raises -> Self:
        """Allocate + zero device storage and the 1-slot staging buffers."""
        var s = ctx.enqueue_create_buffer[DT](Self.CAP * Self.OBS)
        var a = ctx.enqueue_create_buffer[DT](Self.CAP * Self.ACT)
        var r = ctx.enqueue_create_buffer[DT](Self.CAP)
        var d = ctx.enqueue_create_buffer[DT](Self.CAP)
        s.enqueue_fill(Scalar[DT](0.0))
        a.enqueue_fill(Scalar[DT](0.0))
        r.enqueue_fill(Scalar[DT](0.0))
        d.enqueue_fill(Scalar[DT](0.0))

        var stage_s = ctx.enqueue_create_buffer[DT](Self.OBS)
        var stage_a = ctx.enqueue_create_buffer[DT](Self.ACT)
        var stage_r = ctx.enqueue_create_buffer[DT](1)
        var stage_d = ctx.enqueue_create_buffer[DT](1)

        var rng_off = ctx.enqueue_create_buffer[DType.uint64](1)
        rng_off.enqueue_fill(UInt64(0))

        var hr = alloc[Scalar[DT]](1)
        var hd = alloc[Scalar[DT]](1)
        hr[0] = Scalar[DT](0.0)
        hd[0] = Scalar[DT](0.0)

        return Self(
            obs=s^, act=a^, rew=r^, dne=d^,
            stage_obs=stage_s^, stage_act=stage_a^, stage_rew=stage_r^,
            stage_dne=stage_d^,
            _h_rew=hr, _h_dne=hd,
            ctx=ctx,
            size=0, pos=0,
            rng_seed=UInt64(0xC0FFEE_DECADE_0042),
            _rng_offset_dev=rng_off^,
        )

    # ─── SequenceReplayBuffer trait surface ──────────────────────────────

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Trait factory. `ctx` required (device storage); raises if None."""
        comptime assert target == "gpu", (
            "GPUSequenceReplay is the GPU backend; use SequenceReplay for"
            " target == \"cpu\""
        )
        if not ctx:
            raise Error("GPUSequenceReplay.make: ctx required for device storage")
        return Self.new(ctx.value())

    def count(self) -> Int:
        return self.size

    def can_sample[T: Int](self) -> Bool:
        return self.size >= T + 1

    def record(
        mut self,
        s: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Stage one host transition to device then store it at `pos`."""
        self._h_rew[0] = r
        self._h_dne[0] = d
        self.ctx.enqueue_copy(self.stage_obs, s)
        self.ctx.enqueue_copy(self.stage_act, a)
        self.ctx.enqueue_copy(self.stage_rew, self._h_rew)
        self.ctx.enqueue_copy(self.stage_dne, self._h_dne)

        var stage_s_lt = LayoutTensor[
            DT, Layout.row_major(Self.OBS), MutAnyOrigin
        ](self.stage_obs.unsafe_ptr())
        var stage_a_lt = LayoutTensor[
            DT, Layout.row_major(Self.ACT), MutAnyOrigin
        ](self.stage_act.unsafe_ptr())
        var stage_r_lt = LayoutTensor[
            DT, Layout.row_major(1), MutAnyOrigin
        ](self.stage_rew.unsafe_ptr())
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
        var buf_d_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.dne.unsafe_ptr())

        comptime TPB_S = Self.OBS if Self.OBS > Self.ACT else Self.ACT
        comptime kernel = _seq_store_one_kernel[Self.OBS, Self.ACT, Self.CAP]
        self.ctx.enqueue_function[kernel](
            stage_s_lt, stage_a_lt, stage_r_lt, stage_d_lt,
            buf_s_lt, buf_a_lt, buf_r_lt, buf_d_lt,
            Int32(self.pos),
            grid_dim=1, block_dim=TPB_S,
        )
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    def record_batch[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """Store `N_ENVS` device-resident transitions in one kernel launch
        (lockstep multi-env collection); slots `(pos + e) % CAP`."""
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
        var buf_d_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.dne.unsafe_ptr())

        comptime n_blocks = (N_ENVS + TPB - 1) // TPB
        comptime kernel = _seq_store_batch_kernel[
            N_ENVS, Self.OBS, Self.ACT, Self.CAP
        ]
        ctx.enqueue_function[kernel](
            src_obs_lt, src_act_lt, src_rew_lt, src_dne_lt,
            buf_s_lt, buf_a_lt, buf_r_lt, buf_d_lt,
            Int32(self.pos),
            grid_dim=n_blocks, block_dim=TPB,
        )
        self.pos = (self.pos + N_ENVS) % Self.CAP
        self.size += N_ENVS
        if self.size > Self.CAP:
            self.size = Self.CAP

    def sample_batch_dev[
        B: Int, T: Int,
    ](
        mut self,
        ctx: DeviceContext,
        obs_dev: DeviceBuffer[DT],
        act_dev: DeviceBuffer[DT],
        rew_dev: DeviceBuffer[DT],
        dne_dev: DeviceBuffer[DT],
    ) raises:
        """Draw `B` length-`T` windows directly into caller device buffers."""
        comptime assert B > 0 and T > 0, "B and T must be > 0"
        if self.size < T + 1:
            raise Error(
                "GPUSequenceReplay.sample_batch_dev: not enough data for a"
                " length-T window"
            )
        var origin = 0 if self.size < Self.CAP else self.pos
        var n_valid = self.size - T

        var buf_s_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.OBS), MutAnyOrigin
        ](self.obs.unsafe_ptr())
        var buf_a_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP, Self.ACT), MutAnyOrigin
        ](self.act.unsafe_ptr())
        var buf_r_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.rew.unsafe_ptr())
        var buf_d_lt = LayoutTensor[
            DT, Layout.row_major(Self.CAP), MutAnyOrigin
        ](self.dne.unsafe_ptr())
        var out_obs_lt = LayoutTensor[
            DT, Layout.row_major(B, T + 1, Self.OBS), MutAnyOrigin
        ](obs_dev.unsafe_ptr())
        var out_act_lt = LayoutTensor[
            DT, Layout.row_major(B, T, Self.ACT), MutAnyOrigin
        ](act_dev.unsafe_ptr())
        var out_rew_lt = LayoutTensor[
            DT, Layout.row_major(B, T), MutAnyOrigin
        ](rew_dev.unsafe_ptr())
        var out_dne_lt = LayoutTensor[
            DT, Layout.row_major(B, T), MutAnyOrigin
        ](dne_dev.unsafe_ptr())

        var off_lt = LayoutTensor[
            DType.uint64, Layout.row_major(1), MutAnyOrigin
        ](self._rng_offset_dev.unsafe_ptr())
        comptime n_blocks = (B + TPB - 1) // TPB
        comptime kernel = _seq_sample_kernel[
            B, T, Self.OBS, Self.ACT, Self.CAP
        ]
        ctx.enqueue_function[kernel](
            buf_s_lt, buf_a_lt, buf_r_lt, buf_d_lt,
            out_obs_lt, out_act_lt, out_rew_lt, out_dne_lt,
            Int32(origin), Int32(n_valid),
            self.rng_seed, off_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )
        # Bump the offset on-device after the sample reads it.
        comptime inc_kernel = _increment_rng_offset_kernel[B]
        ctx.enqueue_function[inc_kernel](
            off_lt, grid_dim=1, block_dim=1,
        )

    def sample_batch[
        B: Int, T: Int,
    ](
        mut self,
        obs_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        dne_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Host-output bridge: sample on-device into temp buffers, then copy
        out via the stored `ctx`. Lets a GPU buffer feed the current CPU
        world-model pipeline. Allocates per call (not the hot path)."""
        # Local copy of the handle: passing `self.ctx` while `self` is the
        # `mut` receiver of `sample_batch_dev` trips the aliasing checker.
        var c = self.ctx
        var tmp_obs = c.enqueue_create_buffer[DT](B * (T + 1) * Self.OBS)
        var tmp_act = c.enqueue_create_buffer[DT](B * T * Self.ACT)
        var tmp_rew = c.enqueue_create_buffer[DT](B * T)
        var tmp_dne = c.enqueue_create_buffer[DT](B * T)
        self.sample_batch_dev[B, T](c, tmp_obs, tmp_act, tmp_rew, tmp_dne)
        c.enqueue_copy(obs_out, tmp_obs)
        c.enqueue_copy(act_out, tmp_act)
        c.enqueue_copy(rew_out, tmp_rew)
        c.enqueue_copy(dne_out, tmp_dne)
        c.synchronize()
