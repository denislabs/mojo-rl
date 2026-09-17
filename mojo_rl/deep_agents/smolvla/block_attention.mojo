# +--------------------------------------------------------------------------+ #
# | SmolVLA — attention with a static [Q_LEN, KV_LEN] block mask
# +--------------------------------------------------------------------------+ #
"""The one attention shape the denoising step needs and no existing leaf has.

    q [B, Q_LEN*DIM]   k,v [B, KV_LEN*DIM]   mask [Q_LEN, KV_LEN] additive
    out[b,i] = concat_h( softmax_j( q·kᵀ/sqrt(HEAD_DIM) + mask[i,j] ) · v )

`nn` already ships three attention leaves and none fits:

  * `ScaledDotProductAttention` — packed `[Q|K|V]` of ONE sequence.
  * `MaskedAttention` — has the static 2-D mask, but packed and `Q_LEN == KV_LEN`.
    Using it would mean recomputing the prefix's queries, which is what the KV
    cache exists to avoid.
  * `CrossAttention` — separate q/k/v and `Q_LEN != KV_LEN`, but its mask is a
    per-sample KEY PADDING mask `[B, KV_LEN]`. A key is masked for every query or
    for none, so it cannot express causality.

The denoising step needs `Q_LEN != KV_LEN` **and** a query-dependent mask: the 50
action queries attend to the whole cached prefix but only causally among
themselves.

⚠ **Only the SELF (even) layers need this.** A cross (odd) layer's suffix
attends to the entire prefix with nothing masked, so it can use `CrossAttention`
unmasked. Reaching for a mask there would be harmless and pointless; reaching
for `CrossAttention` in the even case would be silently wrong.

## The backward

`vjp` produces dQ, dK and dV. The leaf is still not a `Module` — `Module`'s
`vjp` takes one packed `TensorRefs` whose members share a dimension, and here
q is `[Q_LEN, DIM]` while k and v are `[KV_LEN, DIM]`. It stays hand-driven,
like `DecoderLayerWeights`.

⚠ **It RECOMPUTES the softmax rather than caching them in the forward**, which
is the opposite of `MaskedAttention` and deliberate. One `SmolVLADenoise`
instance drives all 16 layers through ONE leaf instance, so a forward-time
cache would be overwritten fifteen times before any backward read it — the
leaf would silently return the last layer's gradient for every layer. Being
stateless across the forward/backward boundary is what makes one instance
reusable, and it is also flash-attention's own trade: the probability matrix
never has to exist for the whole forward.

The three gradient kernels cannot all be written from one thread map — dQ
reduces over j, dK and dV reduce over i — so `vjp` materialises the
probabilities into leaf scratch first and then runs three race-free passes:

    probs   one thread per (b, h, i)      writes p[b,h,i,j]
    dV      one block  per (b, h)         dv[j] = Σ_i p[i,j]·g[i]
    dscore  one block  per (b, h)         OVERWRITES p with ds, then dQ
    dK      one block  per (b, h)         dk[j] = Σ_i ds[i,j]·q[i]

⚠ **Those four are order-dependent** — dV reads p, the third pass destroys it.
They are enqueued on one stream, which orders them; splitting them across
streams would not.

⚠ **The mask is applied exactly once**, in the probability pass: a masked (i,j)
gets p = 0, so its ds is 0 and it contributes nothing to dK or dV without any
kernel downstream re-reading the mask. A fully-masked row keeps the forward's
floored denominator, so it yields a zero gradient rather than a NaN.

⚠ **dQ, dK and dV are OVERWRITTEN, not accumulated.** Each is written exactly
once by exactly one pass, so no caller has to zero them first. A driver that
needs to sum gradients from two paths must add them itself.

⚠ A fully-masked query row would divide by zero. Rows are renormalised by a
floored denominator and produce a zero context vector instead of NaN — the same
guard, and the same constant, as `cross_attention.mojo`.
"""

from std.math import exp, sqrt
from std.gpu import global_idx, thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.tensor import Tensor


comptime BA_MASK_NEG: Scalar[DT] = Scalar[DT](-1.0e30)
comptime BA_DENOM_FLOOR: Scalar[DT] = Scalar[DT](1.0e-30)
comptime BA_ROW_BLOCK: Int = 32
"""One warp: the block size the row softmax was measured at on the Orin."""


# ⚠⚠ THE FORWARD IS THREE ELEMENT-INDEXED KERNELS, NOT ONE ROW-PER-THREAD ONE,
# AND THAT IS A MEASUREMENT. The previous `_ba_kernel` launched one thread per
# (batch, head, query) with every key and every head-dim inside the thread. Its
# own note said "a tiled BMM is not worth its complexity until this shows up in
# a profile". It showed up: nsys on the Orin put it at 160 calls per SmolVLA
# query, 1.08 s of 3.24 s, at 0.29% of the board's fp32 peak.
#
# `benchmarks/smolvla_block_attention_bench.mojo` then separated the causes in
# one run on the board (A = that kernel):
#
#     block 128 -> 32 (occupancy)            1.18-1.23x
#     branchless, no `continue`              1.27-1.36x   (NOT mask divergence:
#                                                          larger on the UNMASKED
#                                                          cross shape)
#     one-pass online softmax                no gain over branchless
#     THESE THREE KERNELS                    6.10x self, 6.55x cross
#
# The scores are tiny at this model's shapes — 15 heads x 50 queries x 185 keys
# = 138 750 floats, 555 KB — so materialising them costs nothing, and it lets
# each kernel be indexed by the element it WRITES: adjacent lanes then touch
# adjacent addresses. The row-per-thread shape is the one this repo already
# recorded as slow (`_row_per_thread_kernels_are_uncoalesced_and_tiny_grid`).
#
# ⚠ NOT BIT-IDENTICAL to the old kernel — the context is now a sum of
# normalised weights rather than a normalised sum. Held to ~2.5e-6 std units of
# a float64 reference on the board, and to `test_block_attention.mojo`'s
# GPU-vs-CPU band, which also exercises B = 2 (the benchmark is B = 1).


def _ba_scores_kernel[
    BATCH: Int, D: Int, NH: Int, Q: Int, KL: Int, H: Int
](
    q: LayoutTensor[DT, Layout.row_major(BATCH, Q * D), MutAnyOrigin],
    k: LayoutTensor[DT, Layout.row_major(BATCH, KL * D), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(Q * KL), MutAnyOrigin],
    scores: LayoutTensor[DT, Layout.row_major(BATCH * NH * Q * KL), MutAnyOrigin],
):
    """One thread per (b, h, i, j): scaled score plus the additive mask."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * NH * Q * KL:
        return
    var j = idx % KL
    var r = idx // KL
    var i = r % Q
    var r2 = r // Q
    var h = r2 % NH
    var b = r2 // NH
    var qb = b * (Q * D) + i * D + h * H
    var kb = b * (KL * D) + j * D + h * H
    var s = Scalar[DT](0)
    for d in range(H):
        s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
            Scalar[DT]
        ](k.ptr[unsafe_offset = kb + d])
    scores.ptr[unsafe_offset = idx] = s * (
        Scalar[DT](1.0) / sqrt(Scalar[DT](H))
    ) + rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])


def _ba_softmax_kernel[BATCH: Int, NH: Int, Q: Int, KL: Int](
    scores: LayoutTensor[DT, Layout.row_major(BATCH * NH * Q * KL), MutAnyOrigin],
):
    """One thread per (b, h, i): stable softmax over one CONTIGUOUS row, in place.

    ⚠ A FULLY MASKED ROW gives zero weights, not NaN: every score is ~-1e30, so
    the running max never rises above half of `BA_MASK_NEG` and the row is
    zeroed — the same zero context the CPU path produces.
    """
    var idx = Int(global_idx.x)
    if idx >= BATCH * NH * Q:
        return
    var base = idx * KL
    var mx = BA_MASK_NEG
    for j in range(KL):
        mx = max(mx, rebind[Scalar[DT]](scores.ptr[unsafe_offset = base + j]))
    var denom = Scalar[DT](0)
    for j in range(KL):
        denom += exp(rebind[Scalar[DT]](scores.ptr[unsafe_offset = base + j]) - mx)
    var inv = Scalar[DT](0)
    if mx > BA_MASK_NEG * Scalar[DT](0.5) and denom > BA_DENOM_FLOOR:
        inv = Scalar[DT](1.0) / denom
    for j in range(KL):
        scores.ptr[unsafe_offset = base + j] = exp(
            rebind[Scalar[DT]](scores.ptr[unsafe_offset = base + j]) - mx
        ) * inv


def _ba_context_kernel[
    BATCH: Int, D: Int, NH: Int, Q: Int, KL: Int, H: Int
](
    probs: LayoutTensor[DT, Layout.row_major(BATCH * NH * Q * KL), MutAnyOrigin],
    v: LayoutTensor[DT, Layout.row_major(BATCH, KL * D), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, Q * D), MutAnyOrigin],
):
    """One thread per (b, h, i, d): the d-th coordinate of one context vector."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * NH * Q * H:
        return
    var d = idx % H
    var r = idx // H
    var i = r % Q
    var r2 = r // Q
    var h = r2 % NH
    var b = r2 // NH
    var pbase = ((b * NH + h) * Q + i) * KL
    var acc = Scalar[DT](0)
    for j in range(KL):
        acc += rebind[Scalar[DT]](probs.ptr[unsafe_offset = pbase + j]) * rebind[
            Scalar[DT]
        ](v.ptr[unsafe_offset = b * (KL * D) + j * D + h * H + d])
    dst.ptr[unsafe_offset = b * (Q * D) + i * D + h * H + d] = acc


# +--------------------------------------------------------------------------+ #
# | Backward
# +--------------------------------------------------------------------------+ #


def _ba_probs_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    q: LayoutTensor[DT, Layout.row_major(BATCH, QL * DIM), MutAnyOrigin],
    k: LayoutTensor[DT, Layout.row_major(BATCH, KL * DIM), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(QL * KL), MutAnyOrigin],
    probs: LayoutTensor[
        DT, Layout.row_major(BATCH, N_HEADS * QL * KL), MutAnyOrigin
    ],
):
    """Re-run the forward's softmax, keeping the normalised weights.

    ⚠ Byte-for-byte the forward's arithmetic — same two passes, same max shift,
    same floored denominator. A backward taken against a *different* softmax is
    the gradient of a different function, and nothing downstream would say so.
    """
    var idx = Int(global_idx.x)
    if idx >= BATCH * N_HEADS * QL:
        return
    var i = idx % QL
    var r = idx // QL
    var h = r % N_HEADS
    var b = r // N_HEADS
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    var qb = b * (QL * DIM) + i * DIM + h * HD
    var pb = b * (N_HEADS * QL * KL) + h * QL * KL + i * KL

    var mx = BA_MASK_NEG
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        # every entry written here, so a masked j is 0 and not stale scratch
        probs.ptr[unsafe_offset = pb + j] = Scalar[DT](0)
        if m <= BA_MASK_NEG:
            continue
        var kb = b * (KL * DIM) + j * DIM + h * HD
        var s = Scalar[DT](0)
        for d in range(HD):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        s = s * scale + m
        if s > mx:
            mx = s

    var denom = Scalar[DT](0)
    for j in range(KL):
        var m = rebind[Scalar[DT]](mask.ptr[unsafe_offset = i * KL + j])
        if m <= BA_MASK_NEG:
            continue
        var kb = b * (KL * DIM) + j * DIM + h * HD
        var s = Scalar[DT](0)
        for d in range(HD):
            s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
                Scalar[DT]
            ](k.ptr[unsafe_offset = kb + d])
        var w = exp(s * scale + m - mx)
        denom += w
        probs.ptr[unsafe_offset = pb + j] = w

    var inv = Scalar[DT](0)
    if denom > BA_DENOM_FLOOR:
        inv = Scalar[DT](1.0) / denom
    for j in range(KL):
        probs.ptr[unsafe_offset = pb + j] = (
            rebind[Scalar[DT]](probs.ptr[unsafe_offset = pb + j]) * inv
        )


def _ba_dv_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    probs: LayoutTensor[
        DT, Layout.row_major(BATCH, N_HEADS * QL * KL), MutAnyOrigin
    ],
    grad_out: LayoutTensor[DT, Layout.row_major(BATCH, QL * DIM), MutAnyOrigin],
    dv: LayoutTensor[DT, Layout.row_major(BATCH, KL * DIM), MutAnyOrigin],
):
    """dV[j] = Σ_i p[i,j]·g[i]. One block per (b,h), grid-stride over (j,d).

    ⚠ Must run BEFORE `_ba_dscore_dq_kernel`, which overwrites `probs`.

    Every (b, j, d) is written by exactly one thread of exactly one block —
    head h owns the slice `[h*HD, (h+1)*HD)` of every token — so this is a
    plain store, not a read-modify-write.
    """
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HD
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var t = tid
    while t < KL * HD:
        var j = t // HD
        var d = t % HD
        var acc = Scalar[DT](0)
        for i in range(QL):
            var p = rebind[Scalar[DT]](
                probs.ptr[
                    unsafe_offset = b * (N_HEADS * QL * KL)
                    + h * QL * KL + i * KL + j
                ]
            )
            acc += p * rebind[Scalar[DT]](
                grad_out.ptr[
                    unsafe_offset = b * (QL * DIM) + i * DIM + h_off + d
                ]
            )
        dv.ptr[unsafe_offset = b * (KL * DIM) + j * DIM + h_off + d] = acc
        t += bs


def _ba_dscore_dq_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    k: LayoutTensor[DT, Layout.row_major(BATCH, KL * DIM), MutAnyOrigin],
    v: LayoutTensor[DT, Layout.row_major(BATCH, KL * DIM), MutAnyOrigin],
    probs: LayoutTensor[
        DT, Layout.row_major(BATCH, N_HEADS * QL * KL), MutAnyOrigin
    ],
    grad_out: LayoutTensor[DT, Layout.row_major(BATCH, QL * DIM), MutAnyOrigin],
    dq: LayoutTensor[DT, Layout.row_major(BATCH, QL * DIM), MutAnyOrigin],
):
    """Softmax backward, in place, then dQ.

        da[j] = g[i]·v[j]        dot = Σ_j p[i,j]·da[j]
        ds[i,j] = p[i,j]·(da[j] − dot)·scale        (OVERWRITES p)
        dQ[i]   = Σ_j ds[i,j]·k[j]

    One block per (b,h), grid-stride over i. A row is owned entirely by one
    thread, so the in-place overwrite of `probs` races with nothing.
    """
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HD
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))

    var i = tid
    while i < QL:
        var qb = b * (QL * DIM) + i * DIM + h_off
        var pb = b * (N_HEADS * QL * KL) + h * QL * KL + i * KL

        var dot = Scalar[DT](0)
        for j in range(KL):
            var p = rebind[Scalar[DT]](probs.ptr[unsafe_offset = pb + j])
            if p == Scalar[DT](0):
                continue
            var kb = b * (KL * DIM) + j * DIM + h_off
            var da = Scalar[DT](0)
            for d in range(HD):
                da += rebind[Scalar[DT]](
                    grad_out.ptr[unsafe_offset = qb + d]
                ) * rebind[Scalar[DT]](v.ptr[unsafe_offset = kb + d])
            dot += p * da

        for j in range(KL):
            var p = rebind[Scalar[DT]](probs.ptr[unsafe_offset = pb + j])
            var ds = Scalar[DT](0)
            if p != Scalar[DT](0):
                var kb = b * (KL * DIM) + j * DIM + h_off
                var da = Scalar[DT](0)
                for d in range(HD):
                    da += rebind[Scalar[DT]](
                        grad_out.ptr[unsafe_offset = qb + d]
                    ) * rebind[Scalar[DT]](v.ptr[unsafe_offset = kb + d])
                ds = p * (da - dot) * scale
            probs.ptr[unsafe_offset = pb + j] = ds

        for d in range(HD):
            var acc = Scalar[DT](0)
            for j in range(KL):
                acc += rebind[Scalar[DT]](
                    probs.ptr[unsafe_offset = pb + j]
                ) * rebind[Scalar[DT]](
                    k.ptr[unsafe_offset = b * (KL * DIM) + j * DIM + h_off + d]
                )
            dq.ptr[unsafe_offset = qb + d] = acc
        i += bs


def _ba_dk_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, QL: Int, KL: Int, HD: Int
](
    q: LayoutTensor[DT, Layout.row_major(BATCH, QL * DIM), MutAnyOrigin],
    dscore: LayoutTensor[
        DT, Layout.row_major(BATCH, N_HEADS * QL * KL), MutAnyOrigin
    ],
    dk: LayoutTensor[DT, Layout.row_major(BATCH, KL * DIM), MutAnyOrigin],
):
    """dK[j] = Σ_i ds[i,j]·q[i]. Reads what `_ba_dscore_dq_kernel` left in
    `probs`, so it must run after it."""
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HD
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var t = tid
    while t < KL * HD:
        var j = t // HD
        var d = t % HD
        var acc = Scalar[DT](0)
        for i in range(QL):
            var ds = rebind[Scalar[DT]](
                dscore.ptr[
                    unsafe_offset = b * (N_HEADS * QL * KL)
                    + h * QL * KL + i * KL + j
                ]
            )
            acc += ds * rebind[Scalar[DT]](
                q.ptr[unsafe_offset = b * (QL * DIM) + i * DIM + h_off + d]
            )
        dk.ptr[unsafe_offset = b * (KL * DIM) + j * DIM + h_off + d] = acc
        t += bs


struct BlockCrossAttention[
    DIM: Int, N_HEADS: Int, Q_LEN: Int, KV_LEN: Int
](Movable):
    comptime HD: Int = Self.DIM // Self.N_HEADS
    comptime QN: Int = Self.Q_LEN * Self.DIM
    comptime KN: Int = Self.KV_LEN * Self.DIM
    comptime MASK_N: Int = Self.Q_LEN * Self.KV_LEN
    comptime PN: Int = Self.N_HEADS * Self.Q_LEN * Self.KV_LEN

    var mask: Tensor
    var is_gpu: Bool
    # Backward scratch: the probability matrix, then the scores' gradient in
    # its place. Lazily sized by `vjp`, and NOT touched by `forward` — the
    # forward stays allocation-free and one instance stays reusable across the
    # sixteen layers a driver runs through it.
    var probs: Tensor
    var fwd_scores: Tensor
    """The GPU forward's materialised scores, then probabilities, in place.
    Sized on the first call and reused: `ensure_gpu` allocates only on a GROW,
    so the forward stays allocation-free across the sixteen layers and ten
    steps one instance serves. Not shared with `probs`, which `vjp` owns."""

    def __init__(out self):
        comptime assert Self.DIM % Self.N_HEADS == 0, (
            "BlockCrossAttention: DIM must be divisible by N_HEADS"
        )
        self.mask = Tensor()
        self.is_gpu = False
        self.probs = Tensor()
        self.fwd_scores = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.mask = move.mask^
        self.is_gpu = move.is_gpu
        self.probs = move.probs^
        self.fwd_scores = move.fwd_scores^

    @staticmethod
    def make[
        target: StaticString
    ](
        ref mask: List[Scalar[DT]], ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        if len(mask) != Self.MASK_N:
            raise Error(
                "BlockCrossAttention: mask must be Q_LEN*KV_LEN = "
                + String(Self.MASK_N) + ", got " + String(len(mask))
            )
        var a = Self()
        a.mask = Tensor.alloc(Self.MASK_N)
        for i in range(Self.MASK_N):
            a.mask.data[i] = mask[i]
        comptime if target != "cpu":
            a.mask.upload(ctx.value())
            a.is_gpu = True
        return a^

    def forward[
        target: StaticString, B: Int
    ](
        mut self, mut q: Tensor, mut k: Tensor, mut v: Tensor,
        mut out: Tensor, ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            out.ensure(B * Self.QN)
            var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](Self.HD))
            for b in range(B):
                for h in range(Self.N_HEADS):
                    for i in range(Self.Q_LEN):
                        var qb = b * Self.QN + i * Self.DIM + h * Self.HD
                        var mx = BA_MASK_NEG
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var s = Scalar[DT](0)
                            for d in range(Self.HD):
                                s += q.data[qb + d] * k.data[kb + d]
                            s = s * scale + m
                            if s > mx:
                                mx = s
                        var denom = Scalar[DT](0)
                        for d in range(Self.HD):
                            out.data[qb + d] = Scalar[DT](0)
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var s = Scalar[DT](0)
                            for d in range(Self.HD):
                                s += q.data[qb + d] * k.data[kb + d]
                            var w = exp(s * scale + m - mx)
                            denom += w
                            for d in range(Self.HD):
                                out.data[qb + d] += w * v.data[kb + d]
                        var inv = Scalar[DT](0)
                        if denom > BA_DENOM_FLOOR:
                            inv = Scalar[DT](1.0) / denom
                        for d in range(Self.HD):
                            out.data[qb + d] *= inv
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.QN)
            self.fwd_scores.ensure_gpu(c, B * Self.PN)
            comptime n_scores = B * Self.PN
            comptime n_rows = B * Self.N_HEADS * Self.Q_LEN
            comptime n_ctx = B * Self.N_HEADS * Self.Q_LEN * Self.HD
            # Block sizes are the ones measured on the board: TPB for the two
            # thin element kernels, one warp (32) for the row softmax.
            c.enqueue_function[
                _ba_scores_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                q.lt["gpu", Layout.row_major(B, Self.QN)](),
                k.lt["gpu", Layout.row_major(B, Self.KN)](),
                self.mask.lt["gpu", Layout.row_major(Self.MASK_N)](),
                self.fwd_scores.lt["gpu", Layout.row_major(B * Self.PN)](),
                grid_dim=(n_scores + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_softmax_kernel[B, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN]
            ](
                self.fwd_scores.lt["gpu", Layout.row_major(B * Self.PN)](),
                grid_dim=(n_rows + BA_ROW_BLOCK - 1) // BA_ROW_BLOCK,
                block_dim=BA_ROW_BLOCK,
            )
            c.enqueue_function[
                _ba_context_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                self.fwd_scores.lt["gpu", Layout.row_major(B * Self.PN)](),
                v.lt["gpu", Layout.row_major(B, Self.KN)](),
                out.lt["gpu", Layout.row_major(B, Self.QN)](),
                grid_dim=(n_ctx + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int
    ](
        mut self,
        mut q: Tensor, mut k: Tensor, mut v: Tensor,
        mut grad_out: Tensor,
        mut dq: Tensor, mut dk: Tensor, mut dv: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`dQ`, `dK` and `dV` from the forward's own inputs and its output
        gradient.

        `q`, `k` and `v` must be the SAME tensors the forward saw — this leaf
        keeps nothing from the forward and recomputes the softmax from them.
        Passing a later layer's activations would produce a plausible,
        finite, wrong gradient, so a driver has to save them per layer.

        The three outputs are OVERWRITTEN. See this file's header.
        """
        comptime if target == "cpu":
            dq.ensure(B * Self.QN)
            dk.ensure(B * Self.KN)
            dv.ensure(B * Self.KN)
            self.probs.ensure(B * Self.PN)
            for i in range(B * Self.QN):
                dq.data[i] = Scalar[DT](0)
            for i in range(B * Self.KN):
                dk.data[i] = Scalar[DT](0)
                dv.data[i] = Scalar[DT](0)

            var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](Self.HD))
            for b in range(B):
                for h in range(Self.N_HEADS):
                    # ── the forward's softmax, in the forward's arithmetic ──
                    for i in range(Self.Q_LEN):
                        var qb = b * Self.QN + i * Self.DIM + h * Self.HD
                        var pb = (
                            b * Self.PN + h * Self.Q_LEN * Self.KV_LEN
                            + i * Self.KV_LEN
                        )
                        var mx = BA_MASK_NEG
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            self.probs.data[pb + j] = Scalar[DT](0)
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var sc = Scalar[DT](0)
                            for d in range(Self.HD):
                                sc += q.data[qb + d] * k.data[kb + d]
                            sc = sc * scale + m
                            if sc > mx:
                                mx = sc
                        var denom = Scalar[DT](0)
                        for j in range(Self.KV_LEN):
                            var m = self.mask.data[i * Self.KV_LEN + j]
                            if m <= BA_MASK_NEG:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var sc = Scalar[DT](0)
                            for d in range(Self.HD):
                                sc += q.data[qb + d] * k.data[kb + d]
                            var w = exp(sc * scale + m - mx)
                            denom += w
                            self.probs.data[pb + j] = w
                        var inv = Scalar[DT](0)
                        if denom > BA_DENOM_FLOOR:
                            inv = Scalar[DT](1.0) / denom
                        for j in range(Self.KV_LEN):
                            self.probs.data[pb + j] *= inv

                    # ── the gradient. Float64 accumulators: a reduction over
                    #    KV_LEN (185 in the self case) of terms that cancel
                    #    around `dot` is where fp32 loses the most. ───────────
                    for i in range(Self.Q_LEN):
                        var qb = b * Self.QN + i * Self.DIM + h * Self.HD
                        var pb = (
                            b * Self.PN + h * Self.Q_LEN * Self.KV_LEN
                            + i * Self.KV_LEN
                        )
                        var dot = Float64(0)
                        for j in range(Self.KV_LEN):
                            var pj = Float64(self.probs.data[pb + j])
                            if pj == 0.0:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var da = Float64(0)
                            for d in range(Self.HD):
                                da += Float64(grad_out.data[qb + d]) * Float64(
                                    v.data[kb + d]
                                )
                            dot += pj * da
                        for j in range(Self.KV_LEN):
                            var pj = Float64(self.probs.data[pb + j])
                            if pj == 0.0:
                                continue
                            var kb = b * Self.KN + j * Self.DIM + h * Self.HD
                            var da = Float64(0)
                            for d in range(Self.HD):
                                da += Float64(grad_out.data[qb + d]) * Float64(
                                    v.data[kb + d]
                                )
                            var ds = pj * (da - dot) * Float64(scale)
                            for d in range(Self.HD):
                                dq.data[qb + d] = dq.data[qb + d] + Scalar[DT](
                                    ds * Float64(k.data[kb + d])
                                )
                                dk.data[kb + d] = dk.data[kb + d] + Scalar[DT](
                                    ds * Float64(q.data[qb + d])
                                )
                                dv.data[kb + d] = dv.data[kb + d] + Scalar[DT](
                                    pj * Float64(grad_out.data[qb + d])
                                )
        else:
            var c = ctx.value()
            dq.ensure_gpu(c, B * Self.QN)
            dk.ensure_gpu(c, B * Self.KN)
            dv.ensure_gpu(c, B * Self.KN)
            self.probs.ensure_gpu(c, B * Self.PN)

            comptime lay_q = Layout.row_major(B, Self.QN)
            comptime lay_k = Layout.row_major(B, Self.KN)
            comptime lay_p = Layout.row_major(B, Self.PN)
            comptime grid_bh = B * Self.N_HEADS
            # ⚠ ORDER: probs -> dV -> dscore(+dQ) -> dK. The third pass
            # destroys what the second reads. One stream orders them.
            comptime rows = B * Self.N_HEADS * Self.Q_LEN
            comptime n_blocks = (rows + TPB - 1) // TPB
            c.enqueue_function[
                _ba_probs_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                q.lt["gpu", lay_q](),
                k.lt["gpu", lay_k](),
                self.mask.lt["gpu", Layout.row_major(Self.MASK_N)](),
                self.probs.lt["gpu", lay_p](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_dv_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                self.probs.lt["gpu", lay_p](),
                grad_out.lt["gpu", lay_q](),
                dv.lt["gpu", lay_k](),
                grid_dim=grid_bh,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_dscore_dq_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                k.lt["gpu", lay_k](),
                v.lt["gpu", lay_k](),
                self.probs.lt["gpu", lay_p](),
                grad_out.lt["gpu", lay_q](),
                dq.lt["gpu", lay_q](),
                grid_dim=grid_bh,
                block_dim=TPB,
            )
            c.enqueue_function[
                _ba_dk_kernel[
                    B, Self.DIM, Self.N_HEADS, Self.Q_LEN, Self.KV_LEN, Self.HD
                ]
            ](
                q.lt["gpu", lay_q](),
                self.probs.lt["gpu", lay_p](),
                dk.lt["gpu", lay_k](),
                grid_dim=grid_bh,
                block_dim=TPB,
            )
