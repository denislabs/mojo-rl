# +--------------------------------------------------------------------------+ #
# | ACTDeviceDataset — the whole dataset resident on the GPU, sampled there
# +--------------------------------------------------------------------------+ #
"""Draw, gather and normalize a batch entirely on the device.

The host path (`data.mojo::sample_batch` + `Tensor.upload_resident`) is 16.1 ms
of a 144.8 ms ACT iteration with the GPU idle throughout, plus two 29.5 MB
element-by-element fills into pinned memory and a 29.5 MB H2D, plus 8 of the
~28 device synchronizations a step pays. This replaces all of it with three
kernels over data that never leaves the device.

Shaped after `deep_agents/data/gpu_sequence_replay.mojo`, which does the same
job for SAC's replay: a Philox draw kernel with a DEVICE-RESIDENT offset, then
an element-parallel gather. ACT is easier than SAC in the one way that matters
— the dataset is fixed, so the device copy is uploaded once and never
invalidated.

## Why uint8 on the device

Images are `N_CAM * 3 * H * W` uint8 = 460,800 B per frame. The 50-episode
store is 15,447 rows:

    7.12 GB as uint8        fits a 32 GB card with ~25 GB to spare
    28.47 GB as float32     does NOT fit, today

So the normalize belongs in the gather kernel, not in the storage format. That
also means the upload is 4x smaller, and it is the difference between "this
dataset is resident" and "this dataset is not".

⚠ **This does not scale past about an hour of recording** — 108,000 rows is
49.8 GB. `docs/ACT_GPU_DATA_PATH.md` describes the windowed tier-2 design for
that; nothing here forecloses it, because everything below is addressed through
`_g` (the chosen row per batch slot) and a window would simply change what `_g`
indexes into.

## Parity

The device RNG is Philox and the host RNG is a xorshift, so the two samplers
CANNOT draw the same batches and no seed makes them. What is testable, and what
`tests/deep_agents/act/test_act_dataset_gpu.mojo` tests, is the part that is
supposed to agree: given the SAME `(episode, start_ts)`, the device gather must
produce the same tensors as `ACTDataset.fill_at`. That is why `gather_at`
exists as a public entry point separate from `sample`.
"""

from std.gpu import global_idx, thread_idx
from std.memory import Pointer
from std.random.philox import Random as PhiloxRandom
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.io.hdf5 import H5Dataset

from .data import ACTDataset
from .config import (
    IMAGENET_MEAN_R,
    IMAGENET_MEAN_G,
    IMAGENET_MEAN_B,
    IMAGENET_STD_R,
    IMAGENET_STD_G,
    IMAGENET_STD_B,
)

comptime U8 = DType.uint8
comptime I32 = DType.int32
comptime U64 = DType.uint64

# Rows per HDF5 read during the one-time upload. 64 x 460,800 B = 29.5 MB of
# pinned staging — big enough to amortise the per-read overhead, small enough
# not to be a second copy of the dataset in host RAM.
comptime UPLOAD_CHUNK_ROWS = 64


# ── draw ────────────────────────────────────────────────────────────────


def _act_draw_kernel[
    B: Int, K: Int
](
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    n_real: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    eps: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    ep_start: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    ep_len: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    n_split: Int32,
    seed: UInt64,
    offset_buf: LayoutTensor[U64, Layout.row_major(1), MutAnyOrigin],
):
    """Thread `b` picks an episode from the split, then a start step in it.

    Mirrors the host sampler's two draws exactly in STRUCTURE —
    `eps[rand_below(n_split)]` then `rand_below(ep_len)` — but with Philox, so
    the streams differ. That is expected and is why parity is tested through
    `gather_at`, not through a shared seed.

    ⚠ The offset comes from a DEVICE buffer and is advanced by a device kernel,
    not a host counter. Same reason `Adam._pow_dev` exists: a host counter
    freezes under CUDA-graph replay. Nothing captures this yet (split-K blocks
    it, see `_split_k_cannot_be_cuda_graph_captured`), but building it host-side
    would have to be undone later.
    """
    var b = Int(global_idx.x)
    if b >= B:
        return
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(b), offset=offset_base)
    var u0 = Float32(philox.step_uniform()[0])
    var u1 = Float32(philox.step_uniform()[0])

    var ns = Int(n_split)
    if ns < 1:
        ns = 1
    var ei = Int(u0 * Float32(ns))
    if ei >= ns:
        ei = ns - 1
    if ei < 0:
        ei = 0
    var ep = Int(eps[ei])

    var elen = Int(ep_len[ep])
    if elen < 1:
        elen = 1
    var ts = Int(u1 * Float32(elen))
    if ts >= elen:
        ts = elen - 1
    if ts < 0:
        ts = 0

    g[b] = Int32(Int(ep_start[ep]) + ts)
    var remaining = elen - ts
    n_real[b] = Int32(K if remaining > K else remaining)


def _act_advance_offset_kernel[
    B: Int
](offset: LayoutTensor[U64, Layout.row_major(1), MutAnyOrigin]):
    """Bump the RNG offset by the 2 draws per row this batch consumed."""
    if Int(thread_idx.x) == 0:
        offset[0] = offset[0] + UInt64(B * 2)


# ── gather + normalize ──────────────────────────────────────────────────


def _act_gather_images_kernel[
    B: Int, IMG_ELEMS: Int, CAM_ELEMS: Int, HW: Int
](
    src: Pointer[Scalar[U8], MutAnyOrigin],
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    out_img: LayoutTensor[DT, Layout.row_major(B * IMG_ELEMS), MutAnyOrigin],
    n_rows: Int64,
):
    """`out[b, e] = (src[g[b], e]/255 - mean_ch) * inv_std_ch`.

    One thread per output element. The channel is recovered from the flat
    offset because the layout is `[N_CAM][3][H][W]`: `cam = e / CAM_ELEMS`,
    `ch = (e % CAM_ELEMS) / HW`. Same arithmetic as the host loop in
    `data.mojo::_fill_one`, which is what makes the parity gate meaningful.

    ⚠⚠ `src` is a RAW POINTER, not a `LayoutTensor`, and that is load-bearing.
    Its extent is `n_rows * IMG_ELEMS` = **7.1 BILLION** elements for the
    50-episode store, and `LayoutTensor`'s `linear_idx_type` defaults to
    **int32** — so indexing one wraps at 2.147 G however carefully the offset
    is computed. The first version of this kernel did compute the offset in
    Int64 and still read garbage, because the wrap was downstream of the
    arithmetic, in the indexing.

    The parity gate caught it only because it was run against BOTH stores: at
    1,997 rows (0.92 GB, 920 M elements) it passes, and at 15,447 rows it does
    not. A gate on the small store alone would have called this correct.
    """
    var i = Int(global_idx.x)
    if i >= B * IMG_ELEMS:
        return
    var b = i // IMG_ELEMS
    var e = i % IMG_ELEMS
    var ch = (e % CAM_ELEMS) // HW

    var mean = Scalar[DT](IMAGENET_MEAN_R) if ch == 0 else (
        Scalar[DT](IMAGENET_MEAN_G) if ch == 1 else Scalar[DT](IMAGENET_MEAN_B)
    )
    var std = Scalar[DT](IMAGENET_STD_R) if ch == 0 else (
        Scalar[DT](IMAGENET_STD_G) if ch == 1 else Scalar[DT](IMAGENET_STD_B)
    )

    var row = Int(g[b])
    var flat = row * IMG_ELEMS + e          # Int is 64-bit on device
    var v = Scalar[DT](Int(src[unsafe_offset=flat])) / Scalar[DT](255.0)
    out_img[i] = (v - mean) * (Scalar[DT](1.0) / std)


def _act_gather_qpos_kernel[
    B: Int, QPOS: Int
](
    src: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    mean: LayoutTensor[DT, Layout.row_major(QPOS), MutAnyOrigin],
    std: LayoutTensor[DT, Layout.row_major(QPOS), MutAnyOrigin],
    out_q: LayoutTensor[DT, Layout.row_major(B * QPOS), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i >= B * QPOS:
        return
    var b = i // QPOS
    var j = i % QPOS
    var flat = Int(g[b]) * QPOS + j
    out_q[i] = (
        rebind[Scalar[DT]](src[flat]) - rebind[Scalar[DT]](mean[j])
    ) / rebind[Scalar[DT]](std[j])


def _act_gather_actions_kernel[
    B: Int, K: Int, ADIM: Int
](
    src: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    g: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    n_real: LayoutTensor[I32, Layout.row_major(B), MutAnyOrigin],
    mean: LayoutTensor[DT, Layout.row_major(ADIM), MutAnyOrigin],
    std: LayoutTensor[DT, Layout.row_major(ADIM), MutAnyOrigin],
    out_a: LayoutTensor[DT, Layout.row_major(B * K * ADIM), MutAnyOrigin],
    out_v: LayoutTensor[DT, Layout.row_major(B * K), MutAnyOrigin],
    n_rows: Int64,
):
    """Actions `[g, g+K)`, padded past the episode end.

    ⚠ The pad value is the NORMALIZED zero, `(0 - mean)/std`, NOT zero — the
    reference pads the raw action and normalizes afterwards
    (`utils.py`), and `data.mojo::_fill_one` reproduces that. Writing a plain
    0.0 here would be a silent distribution shift on every truncated chunk.
    """
    var i = Int(global_idx.x)
    if i >= B * K * ADIM:
        return
    var b = i // (K * ADIM)
    var rem = i % (K * ADIM)
    var t = rem // ADIM
    var j = rem % ADIM

    var valid = t < Int(n_real[b])
    var raw = Scalar[DT](0.0)
    if valid:
        var flat = Int64(Int(g[b]) + t) * Int64(ADIM) + Int64(j)
        raw = rebind[Scalar[DT]](src[Int(flat)])
    out_a[i] = (raw - rebind[Scalar[DT]](mean[j])) / rebind[Scalar[DT]](
        std[j]
    )
    if j == 0:
        out_v[b * K + t] = Scalar[DT](1.0) if valid else Scalar[DT](0.0)


# ── ACTDeviceDataset ────────────────────────────────────────────────────


struct ACTDeviceDataset[
    QPOS: Int, ADIM: Int, N_CAM: Int, IMG_H: Int, IMG_W: Int
](Movable):
    comptime HW = Self.IMG_H * Self.IMG_W
    comptime CAM_ELEMS = 3 * Self.HW
    comptime IMG_ELEMS = Self.N_CAM * Self.CAM_ELEMS

    var images_u8: TensorImpl[U8]
    """[n_rows, IMG_ELEMS] uint8 — the whole image set, uploaded once."""
    var qpos_raw: Tensor
    var action_raw: Tensor
    var qpos_mean: Tensor
    var qpos_std: Tensor
    var action_mean: Tensor
    var action_std: Tensor
    var ep_start: TensorImpl[I32]
    var ep_len: TensorImpl[I32]
    var train_eps: TensorImpl[I32]
    var val_eps: TensorImpl[I32]
    var n_train: Int
    var n_val: Int
    var n_rows: Int

    var g: TensorImpl[I32]
    """[B] chosen flat row per batch slot — the ONE indirection everything
    else reads through, and the seam a windowed tier-2 would replace."""
    var n_real: TensorImpl[I32]
    var rng_offset: TensorImpl[U64]
    var offset_host: UInt64
    """Host MIRROR of `rng_offset`, exact because the advance is deterministic
    (`+= 2*B` per `sample`). It exists so a caller can save and restore the
    stream WITHOUT a D2H — which validation needs: `act_so101_train_gpu`
    pins the sampler around every validation pass so each one scores the same
    batches, and without that `best_val` selects the luckiest draw rather than
    the best model."""
    var seed: UInt64

    def __init__(out self):
        self.images_u8 = TensorImpl[U8]()
        self.qpos_raw = Tensor()
        self.action_raw = Tensor()
        self.qpos_mean = Tensor()
        self.qpos_std = Tensor()
        self.action_mean = Tensor()
        self.action_std = Tensor()
        self.ep_start = TensorImpl[I32]()
        self.ep_len = TensorImpl[I32]()
        self.train_eps = TensorImpl[I32]()
        self.val_eps = TensorImpl[I32]()
        self.n_train = 0
        self.n_val = 0
        self.n_rows = 0
        self.g = TensorImpl[I32]()
        self.n_real = TensorImpl[I32]()
        self.rng_offset = TensorImpl[U64]()
        self.offset_host = 0
        self.seed = 0

    def __init__(out self, *, deinit move: Self):
        self.images_u8 = move.images_u8^
        self.qpos_raw = move.qpos_raw^
        self.action_raw = move.action_raw^
        self.qpos_mean = move.qpos_mean^
        self.qpos_std = move.qpos_std^
        self.action_mean = move.action_mean^
        self.action_std = move.action_std^
        self.ep_start = move.ep_start^
        self.ep_len = move.ep_len^
        self.train_eps = move.train_eps^
        self.val_eps = move.val_eps^
        self.n_train = move.n_train
        self.n_val = move.n_val
        self.n_rows = move.n_rows
        self.g = move.g^
        self.n_real = move.n_real^
        self.rng_offset = move.rng_offset^
        self.offset_host = move.offset_host
        self.seed = move.seed

    @staticmethod
    def upload_from[
        B: Int
    ](
        mut host: ACTDataset[
            Self.QPOS, Self.ADIM, Self.N_CAM, Self.IMG_H, Self.IMG_W
        ],
        ctx: DeviceContext,
        seed: UInt64 = 0x2545F4914F6CDD1D,
    ) raises -> Self:
        """Upload the whole dataset once. `B` sizes the per-batch scratch.

        ⚠ Images stream through a bounded pinned buffer rather than being
        materialised in host RAM — the host store is deliberately NOT resident
        (7.1 GB of RSS for a 460 KB row was the reason), so reading it all into
        a `List` first would reintroduce exactly the cost that decision
        avoided.
        """
        var d = Self()
        d.seed = seed
        d.n_rows = host.store.n_rows()

        # ── images: chunked HDF5 read -> pinned staging -> device ────────
        var total = d.n_rows * Self.IMG_ELEMS
        d.images_u8.ensure_gpu(ctx, total)
        var chunk = List[Scalar[U8]](
            length=UPLOAD_CHUNK_ROWS * Self.IMG_ELEMS, fill=0
        )
        var staged = ctx.enqueue_create_host_buffer[U8](
            UPLOAD_CHUNK_ROWS * Self.IMG_ELEMS
        )
        ctx.synchronize()
        var r = 0
        while r < d.n_rows:
            var end = r + UPLOAD_CHUNK_ROWS
            if end > d.n_rows:
                end = d.n_rows
            var rows = end - r
            var n = rows * Self.IMG_ELEMS
            host._img_dset.read_range[U8](r, end, mptr(chunk))
            for i in range(n):
                staged[i] = chunk[i]
            var dst = d.images_u8.dev.value().create_sub_buffer[U8](
                r * Self.IMG_ELEMS, n
            )
            var srcw = staged.create_sub_buffer[U8](0, n)
            ctx.enqueue_copy(dst, srcw)
            ctx.synchronize()  # `staged` is reused next chunk
            r = end

        # ── the small tables ────────────────────────────────────────────
        d.qpos_raw = _up_f32(host.qpos_raw, ctx)
        d.action_raw = _up_f32(host.action_raw, ctx)
        d.qpos_mean = _up_f32(host.qpos_mean, ctx)
        d.qpos_std = _up_f32(host.qpos_std, ctx)
        d.action_mean = _up_f32(host.action_mean, ctx)
        d.action_std = _up_f32(host.action_std, ctx)

        var n_eps = host.store.episodes.n_episodes()
        var starts = List[Scalar[I32]](length=n_eps, fill=0)
        var lens = List[Scalar[I32]](length=n_eps, fill=0)
        for e in range(n_eps):
            starts[e] = Int32(host.store.episodes.start_of(e))
            lens[e] = Int32(host.store.episodes.length_of(e))
        d.ep_start = _up_i32(starts, ctx)
        d.ep_len = _up_i32(lens, ctx)

        d.n_train = len(host.train_eps)
        d.n_val = len(host.val_eps)
        var tr = List[Scalar[I32]](length=d.n_train if d.n_train > 0 else 1, fill=0)
        for i in range(d.n_train):
            tr[i] = Int32(host.train_eps[i])
        var va = List[Scalar[I32]](length=d.n_val if d.n_val > 0 else 1, fill=0)
        for i in range(d.n_val):
            va[i] = Int32(host.val_eps[i])
        d.train_eps = _up_i32(tr, ctx)
        d.val_eps = _up_i32(va, ctx)

        d.g = TensorImpl[I32].alloc_gpu(ctx, B)
        d.n_real = TensorImpl[I32].alloc_gpu(ctx, B)
        d.rng_offset = TensorImpl[U64].alloc_gpu(ctx, 1)
        ctx.synchronize()
        return d^

    def sample[
        B: Int, K: Int
    ](
        mut self,
        val: Bool,
        mut out_qpos: Tensor,
        mut out_images: Tensor,
        mut out_actions: Tensor,
        mut out_valid: Tensor,
        ctx: DeviceContext,
    ) raises:
        """Draw + gather a batch. NO host work, NO H2D, NO synchronization."""
        comptime nb = (B + TPB - 1) // TPB
        ctx.enqueue_function[_act_draw_kernel[B, K]](
            self.g.lt["gpu", Layout.row_major(B)](),
            self.n_real.lt["gpu", Layout.row_major(B)](),
            (self.val_eps if val else self.train_eps).lt[
                "gpu", Layout.row_major(B)
            ](),
            self.ep_start.lt["gpu", Layout.row_major(B)](),
            self.ep_len.lt["gpu", Layout.row_major(B)](),
            Int32(self.n_val if val else self.n_train),
            self.seed,
            self.rng_offset.lt["gpu", Layout.row_major(1)](),
            grid_dim=nb,
            block_dim=TPB,
        )
        ctx.enqueue_function[_act_advance_offset_kernel[B]](
            self.rng_offset.lt["gpu", Layout.row_major(1)](),
            grid_dim=1,
            block_dim=1,
        )
        self.offset_host += UInt64(B * 2)
        self._gather[B, K](out_qpos, out_images, out_actions, out_valid, ctx)

    def set_offset(mut self, ctx: DeviceContext, v: UInt64) raises:
        """Pin the RNG stream. Set to a fixed value before a validation pass
        and restore afterwards, so every pass draws the SAME batches."""
        self.rng_offset.dev.value().enqueue_fill(v)
        self.offset_host = v

    def gather_at[
        B: Int, K: Int
    ](
        mut self,
        rows: List[Int],
        n_reals: List[Int],
        mut out_qpos: Tensor,
        mut out_images: Tensor,
        mut out_actions: Tensor,
        mut out_valid: Tensor,
        ctx: DeviceContext,
    ) raises:
        """Gather EXPLICIT rows — the parity entry point.

        The device RNG is Philox and the host's is a xorshift, so no seed makes
        the two samplers agree and a "same batch" gate is impossible. This is
        what IS comparable: the same `(row, n_real)` must produce the same
        tensors as `ACTDataset.fill_at`."""
        self.g.ensure(B)
        self.n_real.ensure(B)
        for b in range(B):
            self.g.data[b] = Int32(rows[b])
            self.n_real.data[b] = Int32(n_reals[b])
        self.g.upload_resident(ctx)
        self.n_real.upload_resident(ctx)
        self._gather[B, K](out_qpos, out_images, out_actions, out_valid, ctx)

    def _gather[
        B: Int, K: Int
    ](
        mut self,
        mut out_qpos: Tensor,
        mut out_images: Tensor,
        mut out_actions: Tensor,
        mut out_valid: Tensor,
        ctx: DeviceContext,
    ) raises:
        out_qpos.ensure_gpu(ctx, B * Self.QPOS)
        out_images.ensure_gpu(ctx, B * Self.IMG_ELEMS)
        out_actions.ensure_gpu(ctx, B * K * Self.ADIM)
        out_valid.ensure_gpu(ctx, B * K)

        comptime nimg = (B * Self.IMG_ELEMS + TPB - 1) // TPB
        ctx.enqueue_function[
            _act_gather_images_kernel[
                B, Self.IMG_ELEMS, Self.CAM_ELEMS, Self.HW
            ]
        ](
            self.images_u8.dev.value(),
            self.g.lt["gpu", Layout.row_major(B)](),
            out_images.lt["gpu", Layout.row_major(B * Self.IMG_ELEMS)](),
            Int64(self.n_rows),
            grid_dim=nimg,
            block_dim=TPB,
        )
        comptime nq = (B * Self.QPOS + TPB - 1) // TPB
        ctx.enqueue_function[_act_gather_qpos_kernel[B, Self.QPOS]](
            self.qpos_raw.lt["gpu", Layout.row_major(1)](),
            self.g.lt["gpu", Layout.row_major(B)](),
            self.qpos_mean.lt["gpu", Layout.row_major(Self.QPOS)](),
            self.qpos_std.lt["gpu", Layout.row_major(Self.QPOS)](),
            out_qpos.lt["gpu", Layout.row_major(B * Self.QPOS)](),
            grid_dim=nq,
            block_dim=TPB,
        )
        comptime na = (B * K * Self.ADIM + TPB - 1) // TPB
        ctx.enqueue_function[_act_gather_actions_kernel[B, K, Self.ADIM]](
            self.action_raw.lt["gpu", Layout.row_major(1)](),
            self.g.lt["gpu", Layout.row_major(B)](),
            self.n_real.lt["gpu", Layout.row_major(B)](),
            self.action_mean.lt["gpu", Layout.row_major(Self.ADIM)](),
            self.action_std.lt["gpu", Layout.row_major(Self.ADIM)](),
            out_actions.lt["gpu", Layout.row_major(B * K * Self.ADIM)](),
            out_valid.lt["gpu", Layout.row_major(B * K)](),
            Int64(self.n_rows),
            grid_dim=na,
            block_dim=TPB,
        )


def _up_f32(ref src: List[Scalar[DT]], ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.alloc(len(src))
    for i in range(len(src)):
        t.data[i] = src[i]
    t.upload_resident(ctx)
    return t^


def _up_i32(
    ref src: List[Scalar[I32]], ctx: DeviceContext
) raises -> TensorImpl[I32]:
    var t = TensorImpl[I32]()
    t.ensure(len(src))
    for i in range(len(src)):
        t.data[i] = src[i]
    t.upload_resident(ctx)
    return t^
