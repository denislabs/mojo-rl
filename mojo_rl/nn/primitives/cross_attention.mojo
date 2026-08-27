"""CrossAttention[DIM, N_HEADS, Q_LEN, KV_LEN, MASKED] — multi-head attention
with SEPARATE query and key/value streams.

The gap this fills. `ScaledDotProductAttention` is ARITY=1 over a packed
`[Q|K|V]` of ONE sequence length, produced by a single `Linear[dim, 3*dim]` on
one input — so it can only ever do self-attention where q, k and v are three
projections of the same tensor. `DecoderBlock` is a degenerate single-KV-token
case built for LeWM. DETR/ACT needs two things neither can express:

  1. **Cross-attention proper** — `Q_LEN` queries (ACT: the k=100 action
     queries) attending to a `KV_LEN` memory (ACT: 2 + camera tokens).
  2. **Positional embedding on q and k but NOT v**, re-added at every layer
     (`transformer.py:with_pos_embed`). Even at `Q_LEN == KV_LEN` the existing
     self-attention leaf cannot do this, because it derives all three
     projections from one tensor.

Setting `Q_LEN == KV_LEN` recovers DETR self-attention, so ONE leaf serves both
the encoder and both attentions of the decoder layer.

## Signature

    inputs   q  [BATCH, Q_LEN  * DIM]
             k  [BATCH, KV_LEN * DIM]
             v  [BATCH, KV_LEN * DIM]
             m  [BATCH, KV_LEN]         (MASKED=True only)
    output      [BATCH, Q_LEN  * DIM]

    out[b,i] = concat_h( softmax_j( q·kᵀ/sqrt(HEAD_DIM) + bias(m) ) · v )

`m` is a **key padding mask, 1.0 = attend / 0.0 = ignore** — the same polarity
as `ACTDataset`'s `valid`, and the INVERSE of torch's `key_padding_mask` (where
True means *ignore*). It is converted to an additive `MASK_NEG` bias internally.
Per-SAMPLE, which is what `MaskedAttention` cannot do: that leaf owns ONE
`[SEQ, SEQ]` bias shared by the whole batch, right for a causal mask and wrong
for padding, where each row of the batch pads at a different length.

⚠ A fully-masked query row would divide by zero. Rows are renormalized by a
floored denominator and produce a zero context vector instead of NaN. ACT never
gets there — `cls` and `qpos` are always unmasked (`detr_vae.py:96`) — but a
NaN that only appears on a short episode is not a failure worth discovering
during a training run.

No params. Backward is the standard attention VJP; `q`/`k`/`v` come back
through `forward_input` (the `Module` contract, as `Linear.vjp` relies on), so
only the softmax weights are cached.

CPU + GPU. The GPU path mirrors `attention.mojo`'s `_forward_gpu_bmm` /
`_vjp_gpu_bmm` structure — pack to head-major, two `batched_matmul`s around a
scalar softmax, unpack — with the packed slabs sized separately for the query
and key/value streams, and per-sample masking folded into the softmax kernel.
"""

from std.math import exp, sqrt
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.bmm import batched_matmul

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime XATTN_MASK_NEG: Scalar[DT] = Scalar[DT](-1.0e30)
"""Additive bias for a masked key. Matches `masked_attention.mojo:MASK_NEG`;
large enough that `exp(s - max)` underflows to 0 in fp32, finite so that a row
which is masked EVERYWHERE still has a defined maximum."""

comptime XATTN_DENOM_FLOOR: Scalar[DT] = Scalar[DT](1.0e-30)
"""Softmax denominator floor. Only reachable when every key of a row is
masked; turns a NaN into a zero context vector."""


# ══════════════════════════════════════════════════════════════════════════
# GPU kernels
# ══════════════════════════════════════════════════════════════════════════
# Token-major <-> head-major repacking, the masked softmax, and its JVP. The
# two matmuls are `batched_matmul`, as in `attention.mojo`. Everything is
# indexed off (BH, LEN, HEAD_DIM), where LEN is Q_LEN for the query stream and
# KV_LEN for the key/value stream — the single structural difference from the
# equal-length self-attention leaf.


def _xa_pack_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, LEN: Int, HEAD_DIM: Int, PACKED: Int
](
    packed: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(BATCH, LEN * DIM), MutAnyOrigin],
):
    """token-major `(BATCH, LEN, DIM)` -> head-major `(BH, LEN, HEAD_DIM)`."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * LEN * DIM:
        return
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % LEN
    var b = rem2 // LEN
    var bh = b * N_HEADS + h
    packed.ptr[unsafe_offset=bh * LEN * HEAD_DIM + t * HEAD_DIM + d] = rebind[
        Scalar[DT]
    ](src.ptr[unsafe_offset=b * LEN * DIM + t * DIM + h * HEAD_DIM + d])


def _xa_unpack_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, LEN: Int, HEAD_DIM: Int, PACKED: Int
](
    dst: LayoutTensor[DT, Layout.row_major(BATCH, LEN * DIM), MutAnyOrigin],
    packed: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
):
    """head-major -> token-major (the inverse of `_xa_pack_kernel`)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * LEN * DIM:
        return
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % LEN
    var b = rem2 // LEN
    var bh = b * N_HEADS + h
    dst.ptr[unsafe_offset=b * LEN * DIM + t * DIM + h * HEAD_DIM + d] = rebind[
        Scalar[DT]
    ](packed.ptr[unsafe_offset=bh * LEN * HEAD_DIM + t * HEAD_DIM + d])


def _xa_softmax_kernel[
    BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, HEAD_DIM: Int,
    MASKED: Bool, ATTN_SIZE: Int, SCORES: Int, BH: Int,
](
    scores: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    attn: LayoutTensor[DT, Layout.row_major(BATCH, ATTN_SIZE), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(BATCH, KL), MutAnyOrigin],
):
    """One block per (b, h); threads stride over query rows.

    Scale, apply the per-sample key mask, stable softmax in place, and mirror
    the weights into the cache. Identical arithmetic to the CPU path, including
    the floored denominator for a fully-masked row.
    """
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HEAD_DIM))
    var bh_off = blk * QL * KL
    var attn_base = b * ATTN_SIZE + h * QL * KL
    var i = tid
    while i < QL:
        var row = bh_off + i * KL
        var arow = attn_base + i * KL
        var mx = XATTN_MASK_NEG
        for j in range(KL):
            var sv = rebind[Scalar[DT]](
                scores.ptr[unsafe_offset=row + j]
            ) * scale
            comptime if MASKED:
                if rebind[Scalar[DT]](
                    mask.ptr[unsafe_offset=b * KL + j]
                ) < Scalar[DT](0.5):
                    sv = XATTN_MASK_NEG
            scores.ptr[unsafe_offset=row + j] = sv
            if sv > mx:
                mx = sv
        var se = Scalar[DT](0)
        for j in range(KL):
            var e = exp(
                rebind[Scalar[DT]](scores.ptr[unsafe_offset=row + j]) - mx
            )
            scores.ptr[unsafe_offset=row + j] = e
            se += e
        var denom = se if se > XATTN_DENOM_FLOOR else XATTN_DENOM_FLOOR
        var inv = Scalar[DT](1) / denom
        for j in range(KL):
            var w = rebind[Scalar[DT]](
                scores.ptr[unsafe_offset=row + j]
            ) * inv
            comptime if MASKED:
                if rebind[Scalar[DT]](
                    mask.ptr[unsafe_offset=b * KL + j]
                ) < Scalar[DT](0.5):
                    w = Scalar[DT](0)
            scores.ptr[unsafe_offset=row + j] = w
            attn.ptr[unsafe_offset=arow + j] = w
        i += bs


def _xa_softmax_jvp_kernel[
    BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, HEAD_DIM: Int,
    ATTN_SIZE: Int, SCORES: Int, BH: Int,
](
    dscore: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    dattn: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    attn: LayoutTensor[DT, Layout.row_major(BATCH, ATTN_SIZE), MutAnyOrigin],
):
    """`dscore = scale * a * (dattn - sum_k a_k * dattn_k)`.

    Masked columns have `a == 0`, so their dscore is zero without a second mask
    read — the same property the CPU path relies on.
    """
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HEAD_DIM))
    var bh_off = blk * QL * KL
    var attn_base = b * ATTN_SIZE + h * QL * KL
    var i = tid
    while i < QL:
        var row = bh_off + i * KL
        var arow = attn_base + i * KL
        var sdot = Scalar[DT](0)
        for j in range(KL):
            sdot += rebind[Scalar[DT]](attn.ptr[unsafe_offset=arow + j]) * (
                rebind[Scalar[DT]](dattn.ptr[unsafe_offset=row + j])
            )
        for j in range(KL):
            var a = rebind[Scalar[DT]](attn.ptr[unsafe_offset=arow + j])
            dscore.ptr[unsafe_offset=row + j] = (
                scale
                * a
                * (
                    rebind[Scalar[DT]](dattn.ptr[unsafe_offset=row + j])
                    - sdot
                )
            )
        i += bs


def _xa_transpose_attn_kernel[
    BATCH: Int, N_HEADS: Int, QL: Int, KL: Int, ATTN_SIZE: Int, SCORES: Int
](
    out_t: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    attn: LayoutTensor[DT, Layout.row_major(BATCH, ATTN_SIZE), MutAnyOrigin],
):
    """cache `(BH, QL, KL)` -> `(BH, KL, QL)`."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SCORES:
        return
    var j = idx % KL
    var rem = idx // KL
    var i = rem % QL
    var bh = rem // QL
    var b = bh // N_HEADS
    var h = bh % N_HEADS
    out_t.ptr[unsafe_offset=bh * KL * QL + j * QL + i] = rebind[Scalar[DT]](
        attn.ptr[unsafe_offset=b * ATTN_SIZE + h * QL * KL + i * KL + j]
    )


def _xa_transpose_scores_kernel[QL: Int, KL: Int, SCORES: Int](
    out_t: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
):
    """`(BH, QL, KL)` -> `(BH, KL, QL)`."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SCORES:
        return
    var j = idx % KL
    var rem = idx // KL
    var i = rem % QL
    var bh = rem // QL
    out_t.ptr[unsafe_offset=bh * KL * QL + j * QL + i] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset=idx]
    )


def _xa_zero_kernel[N: Int](
    g: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < N:
        g.ptr[unsafe_offset=i] = Scalar[DT](0.0)


struct CrossAttention[
    DIM: Int,
    N_HEADS: Int,
    Q_LEN: Int,
    KV_LEN: Int,
    MASKED: Bool = False,
](Module):
    comptime ARITY: Int = 4 if Self.MASKED else 3
    comptime HEAD_DIM: Int = Self.DIM // Self.N_HEADS
    comptime Q_DIM: Int = Self.Q_LEN * Self.DIM
    comptime KV_DIM: Int = Self.KV_LEN * Self.DIM
    comptime IN_DIMS = _xattn_in_dims[
        Self.ARITY, Self.Q_DIM, Self.KV_DIM, Self.KV_LEN
    ]()
    comptime OUT_DIM: Int = Self.Q_DIM

    comptime ATTN_SIZE: Int = Self.N_HEADS * Self.Q_LEN * Self.KV_LEN
    """Per-sample softmax weights — the only thing worth caching. q/k/v come
    back through `forward_input`."""

    var attn: Tensor

    # GPU scratch. Three query-width slabs, three key/value-width slabs and two
    # score slabs — enough for the backward pass to hold pdout, pq, pk, pv, dQ,
    # dK, dV and two of {dattn, dscore, attn_T, dscore_T} at once. Lazily sized;
    # unused on CPU (which uses local Lists, as the self-attention leaf does).
    var sq0: Tensor
    var sq1: Tensor
    var sq2: Tensor
    var sk0: Tensor
    var sk1: Tensor
    var sk2: Tensor
    var ss0: Tensor
    var ss1: Tensor

    def __init__(out self):
        comptime assert Self.DIM % Self.N_HEADS == 0, (
            "CrossAttention: DIM must be divisible by N_HEADS"
        )
        comptime assert Self.Q_LEN > 0 and Self.KV_LEN > 0, (
            "CrossAttention: sequence lengths must be positive"
        )
        self.attn = Tensor()
        self.sq0 = Tensor()
        self.sq1 = Tensor()
        self.sq2 = Tensor()
        self.sk0 = Tensor()
        self.sk1 = Tensor()
        self.sk2 = Tensor()
        self.ss0 = Tensor()
        self.ss1 = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.attn = move.attn^
        self.sq0 = move.sq0^
        self.sq1 = move.sq1^
        self.sq2 = move.sq2^
        self.sk0 = move.sk0^
        self.sk1 = move.sk1^
        self.sk2 = move.sk2^
        self.ss0 = move.ss0^
        self.ss1 = move.ss1^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "CrossAttention: target must be 'cpu' or 'gpu'"
        )
        comptime if target != "cpu":
            if not ctx:
                raise Error("CrossAttention.make[target='gpu']: ctx required")
        return Self()

    def _ensure_scratch_gpu[B: Int](mut self, c: DeviceContext) raises:
        comptime PQ = B * Self.N_HEADS * Self.Q_LEN * Self.HEAD_DIM
        comptime PK = B * Self.N_HEADS * Self.KV_LEN * Self.HEAD_DIM
        comptime SC = B * Self.N_HEADS * Self.Q_LEN * Self.KV_LEN
        self.sq0.ensure_gpu(c, PQ)
        self.sq1.ensure_gpu(c, PQ)
        self.sq2.ensure_gpu(c, PQ)
        self.sk0.ensure_gpu(c, PK)
        self.sk1.ensure_gpu(c, PK)
        self.sk2.ensure_gpu(c, PK)
        self.ss0.ensure_gpu(c, SC)
        self.ss1.ensure_gpu(c, SC)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target != "cpu":
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            self.attn.ensure_gpu(c, B * Self.ATTN_SIZE)
            self._forward_gpu[B](inputs, out, c)
            return

        out.ensure(B * Self.OUT_DIM)
        self.attn.ensure(B * Self.ATTN_SIZE)

        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PACK_Q = BH * QL * HD
        comptime PACK_KV = BH * KL * HD
        comptime SCORES = BH * QL * KL
        var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))

        ref qp = inputs[0].data
        ref kp = inputs[1].data
        ref vp = inputs[2].data
        ref op = out.data
        ref ap = self.attn.data

        var pq = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var pk = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var pv = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var pout = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var sc = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))

        # 1. token-major -> head-major (BH, LEN, HEAD_DIM).
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var hoff = h * HD
                for t in range(QL):
                    for d in range(HD):
                        pq[bh * QL * HD + t * HD + d] = qp[
                            b * Self.Q_DIM + t * Self.DIM + hoff + d
                        ]
                for t in range(KL):
                    for d in range(HD):
                        var src = b * Self.KV_DIM + t * Self.DIM + hoff + d
                        var dst = bh * KL * HD + t * HD + d
                        pk[dst] = kp[src]
                        pv[dst] = vp[src]

        # 2. scores = Q @ Kᵀ  (BH, Q_LEN, KV_LEN).
        var sc_tt = TileTensor(sc, row_major[BH, QL, KL]())
        var pq_tt = TileTensor(pq, row_major[BH, QL, HD]())
        var pk_tt = TileTensor(pk, row_major[BH, KL, HD]())
        batched_matmul[transpose_b=True, target="cpu"](sc_tt, pq_tt, pk_tt)

        # 3. scale, additive key-padding bias, stable softmax. Weights land in
        #    BOTH `sc` (for the attn·V product) and the cache (for backward).
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var sbase = bh * QL * KL
                var abase = b * Self.ATTN_SIZE + h * QL * KL
                for i in range(QL):
                    var row = sbase + i * KL
                    var arow = abase + i * KL
                    var mx = XATTN_MASK_NEG
                    for j in range(KL):
                        var s = sc[row + j] * scale
                        comptime if Self.MASKED:
                            if inputs[3].data[b * KL + j] < Scalar[DT](0.5):
                                s = XATTN_MASK_NEG
                        sc[row + j] = s
                        if s > mx:
                            mx = s
                    var se = Scalar[DT](0)
                    for j in range(KL):
                        var e = exp(sc[row + j] - mx)
                        sc[row + j] = e
                        se += e
                    # Floored: a row whose every key is masked would otherwise
                    # be 0/0. exp(MASK_NEG - MASK_NEG) == 1 for such a row, so
                    # the floor is only load-bearing under fp underflow, but a
                    # NaN that surfaces one training run in ten is not a
                    # tradeoff worth taking for one comparison.
                    var denom = se if se > XATTN_DENOM_FLOOR else (
                        XATTN_DENOM_FLOOR
                    )
                    var inv = Scalar[DT](1) / denom
                    for j in range(KL):
                        var w = sc[row + j] * inv
                        comptime if Self.MASKED:
                            if inputs[3].data[b * KL + j] < Scalar[DT](0.5):
                                w = Scalar[DT](0)
                        sc[row + j] = w
                        ap[arow + j] = w

        # 4. pout = attn @ V  (BH, Q_LEN, HEAD_DIM).
        var pout_tt = TileTensor(pout, row_major[BH, QL, HD]())
        var pv_tt = TileTensor(pv, row_major[BH, KL, HD]())
        batched_matmul[target="cpu"](pout_tt, sc_tt, pv_tt)

        # 5. head-major -> token-major.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var hoff = h * HD
                for t in range(QL):
                    for d in range(HD):
                        op[b * Self.Q_DIM + t * Self.DIM + hoff + d] = pout[
                            bh * QL * HD + t * HD + d
                        ]
        _ = pq^
        _ = pk^
        _ = pv^
        _ = pout^
        _ = sc^

    # ── Backward ─────────────────────────────────────────────────────────

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target != "cpu":
            self._vjp_gpu[B](
                forward_input, grad_output, grad_inputs, ctx.value()
            )
            return

        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PACK_Q = BH * QL * HD
        comptime PACK_KV = BH * KL * HD
        comptime SCORES = BH * QL * KL
        var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))

        ref gq = grad_inputs[0]
        ref gk = grad_inputs[1]
        ref gv = grad_inputs[2]
        gq.ensure(B * Self.Q_DIM)
        gk.ensure(B * Self.KV_DIM)
        gv.ensure(B * Self.KV_DIM)

        ref qp = forward_input[0].data
        ref kp = forward_input[1].data
        ref vp = forward_input[2].data
        ref gop = grad_output.data
        ref ap = self.attn.data

        var pdout = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var pq = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var pk = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var pv = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var dQ = List[Scalar[DT]](length=PACK_Q, fill=Scalar[DT](0))
        var dK = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var dV = List[Scalar[DT]](length=PACK_KV, fill=Scalar[DT](0))
        var dattn = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var dscore = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var attn_T = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var dscore_T = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))

        # 1. pack grad_output + q/k/v into head-major.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var hoff = h * HD
                for t in range(QL):
                    for d in range(HD):
                        var idx = bh * QL * HD + t * HD + d
                        var col = b * Self.Q_DIM + t * Self.DIM + hoff + d
                        pdout[idx] = gop[col]
                        pq[idx] = qp[col]
                for t in range(KL):
                    for d in range(HD):
                        var idx = bh * KL * HD + t * HD + d
                        var col = b * Self.KV_DIM + t * Self.DIM + hoff + d
                        pk[idx] = kp[col]
                        pv[idx] = vp[col]

        # 2. dattn = dout @ Vᵀ  (BH, Q_LEN, KV_LEN).
        var dattn_tt = TileTensor(dattn, row_major[BH, QL, KL]())
        var pdout_tt = TileTensor(pdout, row_major[BH, QL, HD]())
        var pv_tt = TileTensor(pv, row_major[BH, KL, HD]())
        batched_matmul[transpose_b=True, target="cpu"](
            dattn_tt, pdout_tt, pv_tt
        )

        # 3. softmax JVP: dscore = scale * a * (dattn - Σ_k a_k·dattn_k).
        #    Masked columns have a == 0, so dscore is zero there without a
        #    second mask read — the mask's own gradient is zero (it is data,
        #    not a parameter) and grad_inputs[3] is filled with zeros below.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var sbase = bh * QL * KL
                var abase = b * Self.ATTN_SIZE + h * QL * KL
                for i in range(QL):
                    var row = sbase + i * KL
                    var arow = abase + i * KL
                    var s = Scalar[DT](0)
                    for j in range(KL):
                        s += ap[arow + j] * dattn[row + j]
                    for j in range(KL):
                        var a = ap[arow + j]
                        var ds = scale * a * (dattn[row + j] - s)
                        dscore[row + j] = ds
                        attn_T[sbase + j * QL + i] = a
                        dscore_T[sbase + j * QL + i] = ds

        # 4. dV = attnᵀ @ dout   (BH, KV_LEN, HEAD_DIM).
        var attnT_tt = TileTensor(attn_T, row_major[BH, KL, QL]())
        var dV_tt = TileTensor(dV, row_major[BH, KL, HD]())
        batched_matmul[target="cpu"](dV_tt, attnT_tt, pdout_tt)

        # 5. dK = dscoreᵀ @ Q    (BH, KV_LEN, HEAD_DIM).
        var dscoreT_tt = TileTensor(dscore_T, row_major[BH, KL, QL]())
        var pq_tt = TileTensor(pq, row_major[BH, QL, HD]())
        var dK_tt = TileTensor(dK, row_major[BH, KL, HD]())
        batched_matmul[target="cpu"](dK_tt, dscoreT_tt, pq_tt)

        # 6. dQ = dscore @ K     (BH, Q_LEN, HEAD_DIM).
        var dscore_tt = TileTensor(dscore, row_major[BH, QL, KL]())
        var pk_tt = TileTensor(pk, row_major[BH, KL, HD]())
        var dQ_tt = TileTensor(dQ, row_major[BH, QL, HD]())
        batched_matmul[target="cpu"](dQ_tt, dscore_tt, pk_tt)

        # 7. unpack.
        for i in range(B * Self.Q_DIM):
            gq.data[i] = Scalar[DT](0)
        for i in range(B * Self.KV_DIM):
            gk.data[i] = Scalar[DT](0)
            gv.data[i] = Scalar[DT](0)
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var hoff = h * HD
                for t in range(QL):
                    for d in range(HD):
                        gq.data[
                            b * Self.Q_DIM + t * Self.DIM + hoff + d
                        ] = dQ[bh * QL * HD + t * HD + d]
                for t in range(KL):
                    for d in range(HD):
                        var col = b * Self.KV_DIM + t * Self.DIM + hoff + d
                        var idx = bh * KL * HD + t * HD + d
                        gk.data[col] = dK[idx]
                        gv.data[col] = dV[idx]

        comptime if Self.MASKED:
            # The mask is data, not a differentiable input. Zero rather than
            # leave the caller's reused grad slot holding a previous node's
            # values — an accumulating graph would add them in.
            ref gm = grad_inputs[3]
            gm.ensure(B * KL)
            for i in range(B * KL):
                gm.data[i] = Scalar[DT](0)

        _ = pdout^
        _ = pq^
        _ = pk^
        _ = pv^
        _ = dQ^
        _ = dK^
        _ = dV^
        _ = dattn^
        _ = dscore^
        _ = attn_T^
        _ = dscore_T^

    # ── GPU bodies ───────────────────────────────────────────────────────

    def _forward_gpu[
        B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        c: DeviceContext,
    ) raises:
        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PQ = BH * QL * HD
        comptime PK = BH * KL * HD
        comptime SC = BH * QL * KL
        self._ensure_scratch_gpu[B](c)

        comptime lay_q = Layout.row_major(B, Self.Q_DIM)
        comptime lay_kv = Layout.row_major(B, Self.KV_DIM)
        comptime lay_pq = Layout.row_major(PQ)
        comptime lay_pk = Layout.row_major(PK)
        comptime lay_s = Layout.row_major(SC)
        comptime lay_a = Layout.row_major(B, Self.ATTN_SIZE)
        comptime lay_m = Layout.row_major(B, KL)

        ref q = inputs[0]
        ref k = inputs[1]
        ref v = inputs[2]

        # 1. pack q -> sq0, k -> sk0, v -> sk1  (head-major).
        comptime qblocks = (B * QL * Self.DIM + TPB - 1) // TPB
        comptime kblocks = (B * KL * Self.DIM + TPB - 1) // TPB
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            self.sq0.lt["gpu", lay_pq](),
            q.lt["gpu", lay_q](),
            grid_dim=qblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            self.sk0.lt["gpu", lay_pk](),
            k.lt["gpu", lay_kv](),
            grid_dim=kblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            self.sk1.lt["gpu", lay_pk](),
            v.lt["gpu", lay_kv](),
            grid_dim=kblocks, block_dim=TPB,
        )

        # 2. scores(ss0) = Q @ Kt   (BH, QL, KL).
        var sc_tt = TileTensor(self.ss0.dev.value(), row_major[BH, QL, KL]())
        var pq_tt = TileTensor(self.sq0.dev.value(), row_major[BH, QL, HD]())
        var pk_tt = TileTensor(self.sk0.dev.value(), row_major[BH, KL, HD]())
        batched_matmul[transpose_b=True, target="gpu"](
            sc_tt, pq_tt, pk_tt, context=c
        )

        # 3. scale + mask + stable softmax, in place; mirror into the cache.
        #    The mask slot is only read when MASKED; the unmasked instantiation
        #    still needs SOME tensor for the parameter, so the query stream is
        #    passed as an inert stand-in rather than allocating a dummy.
        comptime sm = _xa_softmax_kernel[
            B, Self.N_HEADS, QL, KL, HD, Self.MASKED, Self.ATTN_SIZE, SC, BH
        ]
        comptime if Self.MASKED:
            ref m = inputs[3]
            c.enqueue_function[sm](
                self.ss0.lt["gpu", lay_s](),
                self.attn.lt["gpu", lay_a](),
                m.lt["gpu", lay_m](),
                grid_dim=BH, block_dim=TPB,
            )
        else:
            c.enqueue_function[sm](
                self.ss0.lt["gpu", lay_s](),
                self.attn.lt["gpu", lay_a](),
                rebind[LayoutTensor[DT, lay_m, MutAnyOrigin]](
                    self.attn.lt["gpu", lay_m]()
                ),
                grid_dim=BH, block_dim=TPB,
            )

        # 4. pout(sq1) = attn(ss0) @ V(sk1).
        var pout_tt = TileTensor(self.sq1.dev.value(), row_major[BH, QL, HD]())
        var pv_tt = TileTensor(self.sk1.dev.value(), row_major[BH, KL, HD]())
        batched_matmul[target="gpu"](pout_tt, sc_tt, pv_tt, context=c)

        # 5. unpack -> token-major output.
        c.enqueue_function[
            _xa_unpack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            out.lt["gpu", lay_q](),
            self.sq1.lt["gpu", lay_pq](),
            grid_dim=qblocks, block_dim=TPB,
        )

    def _vjp_gpu[
        B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        c: DeviceContext,
    ) raises:
        comptime QL = Self.Q_LEN
        comptime KL = Self.KV_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PQ = BH * QL * HD
        comptime PK = BH * KL * HD
        comptime SC = BH * QL * KL
        self._ensure_scratch_gpu[B](c)

        comptime lay_q = Layout.row_major(B, Self.Q_DIM)
        comptime lay_kv = Layout.row_major(B, Self.KV_DIM)
        comptime lay_pq = Layout.row_major(PQ)
        comptime lay_pk = Layout.row_major(PK)
        comptime lay_s = Layout.row_major(SC)
        comptime lay_a = Layout.row_major(B, Self.ATTN_SIZE)

        ref q = forward_input[0]
        ref k = forward_input[1]
        ref v = forward_input[2]
        ref gq = grad_inputs[0]
        ref gk = grad_inputs[1]
        ref gv = grad_inputs[2]
        gq.ensure_gpu(c, B * Self.Q_DIM)
        gk.ensure_gpu(c, B * Self.KV_DIM)
        gv.ensure_gpu(c, B * Self.KV_DIM)

        comptime qblocks = (B * QL * Self.DIM + TPB - 1) // TPB
        comptime kblocks = (B * KL * Self.DIM + TPB - 1) // TPB
        comptime sblocks = (SC + TPB - 1) // TPB

        # Slab map. sq0=pdout, sq1=pq, sq2=dQ, sk0=pk (-> dK), sk1=pv, sk2=dV,
        # ss0/ss1 = the score-shaped temporaries. Only sk0 is reused, and only
        # after its last read (step 7), so no ordering hazard: kernels on one
        # stream run in issue order.
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            self.sq0.lt["gpu", lay_pq](),
            grad_output.lt["gpu", lay_q](),
            grid_dim=qblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            self.sq1.lt["gpu", lay_pq](),
            q.lt["gpu", lay_q](),
            grid_dim=qblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            self.sk0.lt["gpu", lay_pk](),
            k.lt["gpu", lay_kv](),
            grid_dim=kblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_pack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            self.sk1.lt["gpu", lay_pk](),
            v.lt["gpu", lay_kv](),
            grid_dim=kblocks, block_dim=TPB,
        )

        var pdout_tt = TileTensor(self.sq0.dev.value(), row_major[BH, QL, HD]())
        var pq_tt = TileTensor(self.sq1.dev.value(), row_major[BH, QL, HD]())
        var pk_tt = TileTensor(self.sk0.dev.value(), row_major[BH, KL, HD]())
        var pv_tt = TileTensor(self.sk1.dev.value(), row_major[BH, KL, HD]())

        # 2. dattn(ss0) = dout @ Vt   (BH, QL, KL).
        var dattn_tt = TileTensor(self.ss0.dev.value(), row_major[BH, QL, KL]())
        batched_matmul[transpose_b=True, target="gpu"](
            dattn_tt, pdout_tt, pv_tt, context=c
        )

        # 3. dscore(ss1) = softmax JVP.
        c.enqueue_function[
            _xa_softmax_jvp_kernel[
                B, Self.N_HEADS, QL, KL, HD, Self.ATTN_SIZE, SC, BH
            ]
        ](
            self.ss1.lt["gpu", lay_s](),
            self.ss0.lt["gpu", lay_s](),
            self.attn.lt["gpu", lay_a](),
            grid_dim=BH, block_dim=TPB,
        )

        # 4. attn_T(ss0) = transpose(cache)  — ss0 free, dattn consumed.
        c.enqueue_function[
            _xa_transpose_attn_kernel[
                B, Self.N_HEADS, QL, KL, Self.ATTN_SIZE, SC
            ]
        ](
            self.ss0.lt["gpu", lay_s](),
            self.attn.lt["gpu", lay_a](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 5. dV(sk2) = attn_T(ss0) @ dout(sq0)   (BH, KL, HD).
        var attnT_tt = TileTensor(self.ss0.dev.value(), row_major[BH, KL, QL]())
        var dV_tt = TileTensor(self.sk2.dev.value(), row_major[BH, KL, HD]())
        batched_matmul[target="gpu"](dV_tt, attnT_tt, pdout_tt, context=c)

        # 6. dQ(sq2) = dscore(ss1) @ K(sk0)  — BEFORE sk0 is recycled for dK.
        var dscore_tt = TileTensor(self.ss1.dev.value(), row_major[BH, QL, KL]())
        var dQ_tt = TileTensor(self.sq2.dev.value(), row_major[BH, QL, HD]())
        batched_matmul[target="gpu"](dQ_tt, dscore_tt, pk_tt, context=c)

        # 7. dscore_T(ss0) = transpose(dscore)  — ss0 free, attn_T read at 5.
        c.enqueue_function[_xa_transpose_scores_kernel[QL, KL, SC]](
            self.ss0.lt["gpu", lay_s](),
            self.ss1.lt["gpu", lay_s](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 8. dK(sk0) = dscore_T(ss0) @ Q(sq1)  — sk0 free, pk read at 6.
        var dscoreT_tt = TileTensor(
            self.ss0.dev.value(), row_major[BH, KL, QL]()
        )
        var dK_tt = TileTensor(self.sk0.dev.value(), row_major[BH, KL, HD]())
        batched_matmul[target="gpu"](dK_tt, dscoreT_tt, pq_tt, context=c)

        # 9. unpack.
        c.enqueue_function[
            _xa_unpack_kernel[B, Self.DIM, Self.N_HEADS, QL, HD, PQ]
        ](
            gq.lt["gpu", lay_q](),
            self.sq2.lt["gpu", lay_pq](),
            grid_dim=qblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_unpack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            gk.lt["gpu", lay_kv](),
            self.sk0.lt["gpu", lay_pk](),
            grid_dim=kblocks, block_dim=TPB,
        )
        c.enqueue_function[
            _xa_unpack_kernel[B, Self.DIM, Self.N_HEADS, KL, HD, PK]
        ](
            gv.lt["gpu", lay_kv](),
            self.sk2.lt["gpu", lay_pk](),
            grid_dim=kblocks, block_dim=TPB,
        )

        comptime if Self.MASKED:
            # The mask is data, not a parameter. Zeroed rather than left alone:
            # the caller's grad slot is reused across nodes.
            ref gm = grad_inputs[3]
            gm.ensure_gpu(c, B * KL)
            c.enqueue_function[_xa_zero_kernel[B * KL]](
                gm.lt["gpu", Layout.row_major(B * KL)](),
                grid_dim=(B * KL + TPB - 1) // TPB, block_dim=TPB,
            )

    # for_each_param / for_each_state / zero_grad inherit the Module
    # reflection defaults — no `Param` fields.


# ── comptime helper ──────────────────────────────────────────────────────
# `IN_DIMS` is `InlineArray[Int, ARITY]` and ARITY varies with MASKED, so the
# array cannot be written as one literal. Mirrors `concat.mojo`'s `_total_dim`.


def _xattn_in_dims[
    ARITY: Int, Q_DIM: Int, KV_DIM: Int, KV_LEN: Int
]() -> InlineArray[Int, ARITY]:
    var a = InlineArray[Int, ARITY](fill=KV_DIM)
    a[0] = Q_DIM
    comptime if ARITY == 4:
        a[3] = KV_LEN
    return a^


# ── aliases ──────────────────────────────────────────────────────────────

comptime SelfAttentionPos[
    DIM: Int, N_HEADS: Int, SEQ: Int
] = CrossAttention[DIM, N_HEADS, SEQ, SEQ, False]
"""DETR self-attention: `q = k = x + pos`, `v = x`. The positional embedding
reaching q and k but not v is why this cannot be `ScaledDotProductAttention`."""

comptime SelfAttentionPosMasked[
    DIM: Int, N_HEADS: Int, SEQ: Int
] = CrossAttention[DIM, N_HEADS, SEQ, SEQ, True]
"""As above with a per-sample key padding mask — ACT's CVAE encoder, whose
action chunk is zero-padded past the end of an episode."""
