"""ScaledDotProductAttention[DIM, N_HEADS, SEQ_LEN, CAUSAL, USE_MAX_KERNELS].

Multi-head scaled dot-product attention as a single nn leaf, on the STORAGE
surface (transformed from legacy `nn.primitives.attention` — surface-only change;
the QKᵀ/softmax/attn·V kernels + the bmm pack/softmax/unpack glue are carried
over VERBATIM). Input is the per-token concatenated `[Q ‖ K ‖ V]` (each DIM-wide),
laid out per sample as `[all-Q tokens | all-K tokens | all-V tokens]`:

    IN_DIM  = SEQ_LEN * DIM * 3        (offsets: Q@0, K@SEQ·DIM, V@2·SEQ·DIM)
    OUT_DIM = SEQ_LEN * DIM

No params. Caches are leaf-owned `Tensor` fields (output-caching — backward reads
only the cache + grad_output, never the forward input slab; storage surface
passes `forward_input` explicitly but this leaf doesn't need it). The cache holds
[Q | K | V | scores] per sample:

    CACHE_SIZE = 3*SEQ_LEN*DIM + N_HEADS*SEQ_LEN*SEQ_LEN

`head_dim = DIM // N_HEADS`, `scale = 1/sqrt(head_dim)`. `CAUSAL=True` bounds each
query i's key loop to j ≤ i. Softmax computed with the standard max-shift.

GPU path: `USE_MAX_KERNELS=True` (default) → batched-GEMM attention (tensor
cores); `False` → serial per-(b,h) custom kernels. Bit-identical (the flag only
changes speed). CPU path mirrors the bmm forward/backward (BLAS GEMMs + scalar
softmax). Unlike legacy, scratch slabs are separate owned `Tensor` fields (one
buffer per slab) instead of a single pointer-sliced scratch — no `mptr`.
"""

from std.math import exp, sqrt
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.bmm import batched_matmul

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — custom per-(b,h) path. One block per (b,h); threads stride
# over rows (fwd / dQ) or (j,d) pairs (dV / dK). Carried VERBATIM from the
# legacy leaf. Float32 throughout (Metal has no Float64).
# ──────────────────────────────────────────────────────────────────────


def _attn_fwd_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CAUSAL: Bool, IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    K_OFF: Int, V_OFF: Int, ATTN_OFF: Int,
](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))

    # Step 1: cache this head's Q/K/V slice (for backward).
    var n_qkv = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_qkv:
        var i = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        cache.ptr[b * CACHE_SIZE + i * DIM + h_off + d] = rebind[Scalar[DT]](
            input.ptr[b * IN_DIM + i * DIM + h_off + d]
        )
        cache.ptr[b * CACHE_SIZE + K_OFF + i * DIM + h_off + d] = rebind[
            Scalar[DT]
        ](input.ptr[b * IN_DIM + K_OFF + i * DIM + h_off + d])
        cache.ptr[b * CACHE_SIZE + V_OFF + i * DIM + h_off + d] = rebind[
            Scalar[DT]
        ](input.ptr[b * IN_DIM + V_OFF + i * DIM + h_off + d])
        idx0 += bs

    # Step 2: per-row attention; each thread strides over query rows i.
    var i = tid
    while i < SEQ:
        var j_end = SEQ
        comptime if CAUSAL:
            j_end = i + 1

        var max_score = Scalar[DT](-1e30)
        for j in range(j_end):
            var s = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var q = rebind[Scalar[DT]](
                    input.ptr[b * IN_DIM + i * DIM + h_off + d]
                )
                var k = rebind[Scalar[DT]](
                    input.ptr[b * IN_DIM + K_OFF + j * DIM + h_off + d]
                )
                s += q * k
            s *= scale
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            cache.ptr[aidx] = s
            if s > max_score:
                max_score = s

        var sum_exp = Scalar[DT](0)
        for j in range(j_end):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var e = exp(rebind[Scalar[DT]](cache.ptr[aidx]) - max_score)
            cache.ptr[aidx] = e
            sum_exp += e

        var inv_sum = Scalar[DT](1) / sum_exp
        for j in range(j_end):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            cache.ptr[aidx] = rebind[Scalar[DT]](cache.ptr[aidx]) * inv_sum

        for d in range(HEAD_DIM):
            var acc = Scalar[DT](0)
            for j in range(j_end):
                var aidx = (
                    b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
                )
                var v = rebind[Scalar[DT]](
                    input.ptr[b * IN_DIM + V_OFF + j * DIM + h_off + d]
                )
                acc += rebind[Scalar[DT]](cache.ptr[aidx]) * v
            output.ptr[b * OUT_DIM + i * DIM + h_off + d] = acc
        i += bs


def _attn_zero_grad_kernel[
    BATCH: Int, IN_DIM: Int
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < BATCH * IN_DIM:
        grad_input.ptr[idx] = Scalar[DT](0)


def _attn_dV_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CAUSAL: Bool, IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    V_OFF: Int, ATTN_OFF: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # dV[j, h_off+d] = Σ_i attn[i,j] * grad_out[i, h_off+d]. Causal: i ≥ j.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var n_jd = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_jd:
        var j = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        var i_start = 0
        comptime if CAUSAL:
            i_start = j
        var acc = Scalar[DT](0)
        for i in range(i_start, SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var go = rebind[Scalar[DT]](
                grad_output.ptr[b * OUT_DIM + i * DIM + h_off + d]
            )
            acc += rebind[Scalar[DT]](cache.ptr[aidx]) * go
        var dv_idx = b * IN_DIM + V_OFF + j * DIM + h_off + d
        grad_input.ptr[dv_idx] = grad_input.ptr[dv_idx] + acc
        idx0 += bs


def _attn_dscore_dQ_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CAUSAL: Bool, IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int,
    K_OFF: Int, V_OFF: Int, ATTN_OFF: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # Per row i: dot_sum, then d_score (overwrites cache.attn), then dQ.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))

    var i = tid
    while i < SEQ:
        var j_end = SEQ
        comptime if CAUSAL:
            j_end = i + 1

        var dot_sum = Scalar[DT](0)
        for j in range(j_end):
            var d_attn = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var go = rebind[Scalar[DT]](
                    grad_output.ptr[b * OUT_DIM + i * DIM + h_off + d]
                )
                var v = rebind[Scalar[DT]](
                    cache.ptr[b * CACHE_SIZE + V_OFF + j * DIM + h_off + d]
                )
                d_attn += go * v
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            dot_sum += rebind[Scalar[DT]](cache.ptr[aidx]) * d_attn

        for j in range(j_end):
            var d_attn = Scalar[DT](0)
            for d in range(HEAD_DIM):
                var go = rebind[Scalar[DT]](
                    grad_output.ptr[b * OUT_DIM + i * DIM + h_off + d]
                )
                var v = rebind[Scalar[DT]](
                    cache.ptr[b * CACHE_SIZE + V_OFF + j * DIM + h_off + d]
                )
                d_attn += go * v
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var attn_w = rebind[Scalar[DT]](cache.ptr[aidx])
            cache.ptr[aidx] = attn_w * (d_attn - dot_sum) * scale

        for d in range(HEAD_DIM):
            var acc = Scalar[DT](0)
            for j in range(j_end):
                var aidx = (
                    b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
                )
                var d_score = rebind[Scalar[DT]](cache.ptr[aidx])
                var k = rebind[Scalar[DT]](
                    cache.ptr[b * CACHE_SIZE + K_OFF + j * DIM + h_off + d]
                )
                acc += d_score * k
            var dq_idx = b * IN_DIM + i * DIM + h_off + d
            grad_input.ptr[dq_idx] = grad_input.ptr[dq_idx] + acc
        i += bs


def _attn_dK_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CAUSAL: Bool, IN_DIM: Int, CACHE_SIZE: Int, K_OFF: Int, ATTN_OFF: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # dK[j, h_off+d] = Σ_i d_score[i,j] * Q[i, h_off+d]. Reads d_score from
    # cache.attn (dscore_dQ kernel overwrote it). Causal: i ≥ j.
    var blk = Int(block_idx.x)
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    if b >= BATCH:
        return
    var h_off = h * HEAD_DIM
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var n_jd = SEQ * HEAD_DIM
    var idx0 = tid
    while idx0 < n_jd:
        var j = idx0 // HEAD_DIM
        var d = idx0 % HEAD_DIM
        var i_start = 0
        comptime if CAUSAL:
            i_start = j
        var acc = Scalar[DT](0)
        for i in range(i_start, SEQ):
            var aidx = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
            var d_score = rebind[Scalar[DT]](cache.ptr[aidx])
            var q = rebind[Scalar[DT]](
                cache.ptr[b * CACHE_SIZE + i * DIM + h_off + d]
            )
            acc += d_score * q
        var dk_idx = b * IN_DIM + K_OFF + j * DIM + h_off + d
        grad_input.ptr[dk_idx] = grad_input.ptr[dk_idx] + acc
        idx0 += bs


# ──────────────────────────────────────────────────────────────────────
# BMM fast path — batched-GEMM attention behind USE_MAX_KERNELS. Single-
# launch batched matmuls (tensor cores) for QKᵀ and attn·V, plus
# pack/softmax/unpack glue. Carried VERBATIM. Packed layout: (BH, SEQ,
# HEAD_DIM); scores layout: (BH, SEQ, SEQ).
# ──────────────────────────────────────────────────────────────────────


def _attn_pack_qkv_fwd_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, CACHE_SIZE: Int, PACKED: Int,
](
    packed_q: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_k: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_v: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime pack_elems = BATCH * SEQ * DIM
    if idx >= pack_elems:
        return
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % SEQ
    var b = rem2 // SEQ
    var col = h * HEAD_DIM + d
    var bh = b * N_HEADS + h
    var pidx = bh * SEQ * HEAD_DIM + t * HEAD_DIM + d
    var qv = rebind[Scalar[DT]](input.ptr[b * IN_DIM + t * DIM + col])
    var kv = rebind[Scalar[DT]](input.ptr[b * IN_DIM + KOFF + t * DIM + col])
    var vv = rebind[Scalar[DT]](input.ptr[b * IN_DIM + VOFF + t * DIM + col])
    cache.ptr[b * CACHE_SIZE + t * DIM + col] = qv
    cache.ptr[b * CACHE_SIZE + KOFF + t * DIM + col] = kv
    cache.ptr[b * CACHE_SIZE + VOFF + t * DIM + col] = vv
    packed_q.ptr[pidx] = qv
    packed_k.ptr[pidx] = kv
    packed_v.ptr[pidx] = vv


def _attn_softmax_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int, CAUSAL: Bool,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    scores: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # 1 block per (b,h); threads stride over rows i. scale + stable softmax
    # in-place on `scores`; mirror weights into cache.attn (per-sample strided).
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))
    var bh_off = blk * SEQ * SEQ
    var cache_attn_base = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ
    var i = tid
    while i < SEQ:
        var j_end = SEQ
        comptime if CAUSAL:
            j_end = i + 1
        var row_off = bh_off + i * SEQ
        var cache_row = cache_attn_base + i * SEQ
        var mx = Scalar[DT](-1e30)
        for j in range(j_end):
            var s = rebind[Scalar[DT]](scores.ptr[row_off + j]) * scale
            scores.ptr[row_off + j] = s
            if s > mx:
                mx = s
        var se = Scalar[DT](0)
        for j in range(j_end):
            var e = exp(rebind[Scalar[DT]](scores.ptr[row_off + j]) - mx)
            scores.ptr[row_off + j] = e
            se += e
        comptime if CAUSAL:
            for j in range(i + 1, SEQ):
                scores.ptr[row_off + j] = Scalar[DT](0)
        var inv = Scalar[DT](1) / se
        for j in range(j_end):
            var w = rebind[Scalar[DT]](scores.ptr[row_off + j]) * inv
            scores.ptr[row_off + j] = w
            cache.ptr[cache_row + j] = w
        comptime if CAUSAL:
            for j in range(i + 1, SEQ):
                cache.ptr[cache_row + j] = Scalar[DT](0)
        i += bs


def _attn_unpack_out_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    OUT_DIM: Int, PACKED: Int,
](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    packed_out: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BATCH * SEQ * DIM
    if idx >= n:
        return
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % SEQ
    var b = rem2 // SEQ
    var bh = b * N_HEADS + h
    var pidx = bh * SEQ * HEAD_DIM + t * HEAD_DIM + d
    output.ptr[b * OUT_DIM + t * DIM + h * HEAD_DIM + d] = rebind[Scalar[DT]](
        packed_out.ptr[pidx]
    )


def _attn_pack_in_bwd_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, OUT_DIM: Int, CACHE_SIZE: Int, PACKED: Int,
](
    packed_dout: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_q: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_k: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    packed_v: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BATCH * SEQ * DIM
    if idx >= n:
        return
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % SEQ
    var b = rem2 // SEQ
    var col = h * HEAD_DIM + d
    var bh = b * N_HEADS + h
    var pidx = bh * SEQ * HEAD_DIM + t * HEAD_DIM + d
    packed_dout.ptr[pidx] = rebind[Scalar[DT]](
        grad_output.ptr[b * OUT_DIM + t * DIM + col]
    )
    packed_q.ptr[pidx] = rebind[Scalar[DT]](cache.ptr[b * CACHE_SIZE + t * DIM + col])
    packed_k.ptr[pidx] = rebind[Scalar[DT]](
        cache.ptr[b * CACHE_SIZE + KOFF + t * DIM + col]
    )
    packed_v.ptr[pidx] = rebind[Scalar[DT]](
        cache.ptr[b * CACHE_SIZE + VOFF + t * DIM + col]
    )


def _attn_softmax_jvp_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    dscore: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    dattn: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    # dscore = scale * a * (dattn - sum_k a_k*dattn_k). Causal masking is
    # implicit: cache.attn[i,j>i]=0 → dscore[i,j>i]=0.
    var blk = Int(block_idx.x)
    if blk >= BH:
        return
    var b = blk // N_HEADS
    var h = blk % N_HEADS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(HEAD_DIM)))
    var bh_off = blk * SEQ * SEQ
    var cache_attn_base = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ
    var i = tid
    while i < SEQ:
        var row_off = bh_off + i * SEQ
        var cache_row = cache_attn_base + i * SEQ
        var s = Scalar[DT](0)
        for j in range(SEQ):
            s += rebind[Scalar[DT]](cache.ptr[cache_row + j]) * rebind[
                Scalar[DT]
            ](dattn.ptr[row_off + j])
        for j in range(SEQ):
            var a = rebind[Scalar[DT]](cache.ptr[cache_row + j])
            var da = rebind[Scalar[DT]](dattn.ptr[row_off + j])
            dscore.ptr[row_off + j] = scale * a * (da - s)
        i += bs


def _attn_transpose_from_cache_kernel[
    BATCH: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    CACHE_SIZE: Int, SCORES: Int, BH: Int,
](
    attn_T: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BH * SEQ * SEQ
    if idx >= n:
        return
    comptime ATTN_OFF = 3 * SEQ * (N_HEADS * HEAD_DIM)
    var i = idx % SEQ
    var rem = idx // SEQ
    var j = rem % SEQ
    var bh = rem // SEQ
    var b = bh // N_HEADS
    var h = bh % N_HEADS
    var src = b * CACHE_SIZE + ATTN_OFF + h * SEQ * SEQ + i * SEQ + j
    attn_T.ptr[bh * SEQ * SEQ + j * SEQ + i] = rebind[Scalar[DT]](
        cache.ptr[src]
    )


def _attn_transpose_scores_kernel[
    SEQ: Int, SCORES: Int, BH: Int,
](
    dst: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BH * SEQ * SEQ
    if idx >= n:
        return
    var i = idx % SEQ
    var rem = idx // SEQ
    var j = rem % SEQ
    var bh = rem // SEQ
    dst.ptr[bh * SEQ * SEQ + j * SEQ + i] = rebind[Scalar[DT]](
        src.ptr[bh * SEQ * SEQ + i * SEQ + j]
    )


def _attn_unpack_grad_kernel[
    BATCH: Int, DIM: Int, N_HEADS: Int, SEQ: Int, HEAD_DIM: Int,
    IN_DIM: Int, PACKED: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    dQ: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    dK: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
    dV: LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime n = BATCH * SEQ * DIM
    if idx >= n:
        return
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM
    var d = idx % HEAD_DIM
    var rem = idx // HEAD_DIM
    var h = rem % N_HEADS
    var rem2 = rem // N_HEADS
    var t = rem2 % SEQ
    var b = rem2 // SEQ
    var col = h * HEAD_DIM + d
    var bh = b * N_HEADS + h
    var pidx = bh * SEQ * HEAD_DIM + t * HEAD_DIM + d
    grad_input.ptr[b * IN_DIM + t * DIM + col] = rebind[Scalar[DT]](dQ.ptr[pidx])
    grad_input.ptr[b * IN_DIM + KOFF + t * DIM + col] = rebind[Scalar[DT]](
        dK.ptr[pidx]
    )
    grad_input.ptr[b * IN_DIM + VOFF + t * DIM + col] = rebind[Scalar[DT]](
        dV.ptr[pidx]
    )


struct ScaledDotProductAttention[
    DIM: Int,
    N_HEADS: Int,
    SEQ_LEN: Int,
    CAUSAL: Bool = False,
    USE_MAX_KERNELS: Bool = True,
](Module):
    comptime ARITY: Int = 1
    comptime HEAD_DIM: Int = Self.DIM // Self.N_HEADS
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SEQ_LEN * Self.DIM * 3)
    comptime OUT_DIM = Self.SEQ_LEN * Self.DIM
    # Cache offsets (per sample).
    comptime K_OFF: Int = Self.SEQ_LEN * Self.DIM
    comptime V_OFF: Int = 2 * Self.SEQ_LEN * Self.DIM
    comptime ATTN_OFF: Int = 3 * Self.SEQ_LEN * Self.DIM
    comptime CACHE_SIZE: Int = (
        3 * Self.SEQ_LEN * Self.DIM
        + Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
    )

    # Cache (leaf-owned, output-caching) — [BATCH, CACHE_SIZE], lazy.
    var cache: Tensor
    # BMM scratch slabs (separate owned Tensors — one buffer per slab, vs the
    # legacy single pointer-sliced scratch). Lazily sized; GPU-only in practice
    # (CPU path uses local Lists). 4 packed slabs + 2 scores slabs (fwd uses 4
    # packed + 1 scores; bwd recycles them, see _vjp_gpu_bmm aliasing comments).
    var sp0: Tensor  # packed slot 0
    var sp1: Tensor  # packed slot 1
    var sp2: Tensor  # packed slot 2
    var sp3: Tensor  # packed slot 3
    var ss0: Tensor  # scores slot 0
    var ss1: Tensor  # scores slot 1

    def __init__(out self):
        comptime assert (
            Self.DIM % Self.N_HEADS == 0
        ), "ScaledDotProductAttention: DIM must be divisible by N_HEADS"
        self.cache = Tensor()
        self.sp0 = Tensor()
        self.sp1 = Tensor()
        self.sp2 = Tensor()
        self.sp3 = Tensor()
        self.ss0 = Tensor()
        self.ss1 = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "ScaledDotProductAttention: target must be 'cpu' or 'gpu'"
        )
        comptime if target != "cpu":
            if not ctx:
                raise Error(
                    "ScaledDotProductAttention.make[target='gpu']: ctx required"
                )
        return Self()

    def _ensure_scratch_gpu[BATCH: Int](mut self, c: DeviceContext) raises:
        comptime PACKED = BATCH * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BATCH * Self.N_HEADS * Self.SEQ_LEN * Self.SEQ_LEN
        self.sp0.ensure_gpu(c, PACKED)
        self.sp1.ensure_gpu(c, PACKED)
        self.sp2.ensure_gpu(c, PACKED)
        self.sp3.ensure_gpu(c, PACKED)
        self.ss0.ensure_gpu(c, SCORES)
        self.ss1.ensure_gpu(c, SCORES)

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            self._forward_cpu[B](in0, out)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            self.cache.ensure_gpu(c, B * Self.CACHE_SIZE)
            comptime if Self.USE_MAX_KERNELS:
                self._forward_gpu_bmm[B](in0, out, c)
            else:
                self._forward_gpu_custom[B](in0, out, c)

    def _forward_gpu_custom[
        B: Int
    ](mut self, mut in0: Tensor, mut out: Tensor, c: DeviceContext) raises:
        comptime lay_in = Layout.row_major(B, Self.IN_DIMS[0])
        comptime lay_out = Layout.row_major(B, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(B, Self.CACHE_SIZE)
        comptime kernel = _attn_fwd_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF,
        ]
        c.enqueue_function[kernel](
            out.lt["gpu", lay_out](),
            in0.lt["gpu", lay_in](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=B * Self.N_HEADS, block_dim=TPB,
        )

    def _forward_gpu_bmm[
        B: Int
    ](mut self, mut in0: Tensor, mut out: Tensor, c: DeviceContext) raises:
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        self._ensure_scratch_gpu[B](c)

        # Forward uses pq=sp0, pk=sp1, pv=sp2, pout=sp3, scores=ss0.
        comptime lay_in = Layout.row_major(B, Self.IN_DIMS[0])
        comptime lay_out = Layout.row_major(B, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(B, Self.CACHE_SIZE)
        comptime lay_p = Layout.row_major(PACKED)
        comptime lay_s = Layout.row_major(SCORES)

        # 1. pack QKV → (BH, SEQ, HEAD_DIM) + write cache.
        comptime pelems = B * Self.SEQ_LEN * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        comptime pack_k = _attn_pack_qkv_fwd_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIMS[0], Self.CACHE_SIZE, PACKED,
        ]
        c.enqueue_function[pack_k](
            self.sp0.lt["gpu", lay_p](),
            self.sp1.lt["gpu", lay_p](),
            self.sp2.lt["gpu", lay_p](),
            self.cache.lt["gpu", lay_c](),
            in0.lt["gpu", lay_in](),
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. scores = Q @ Kᵀ  (BH, SEQ, SEQ).
        var scores_tt = TileTensor(
            self.ss0.dev.value(), row_major[BH, Self.SEQ_LEN, Self.SEQ_LEN]()
        )
        var pq_tt = TileTensor(
            self.sp0.dev.value(), row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]()
        )
        var pk_tt = TileTensor(
            self.sp1.dev.value(), row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]()
        )
        batched_matmul[transpose_b=True, target="gpu"](
            scores_tt, pq_tt, pk_tt, context=c
        )

        # 3. scale + stable softmax in-place; mirror into cache.attn.
        comptime sm_k = _attn_softmax_kernel[
            B, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM, Self.CAUSAL,
            Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[sm_k](
            self.ss0.lt["gpu", lay_s](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=BH, block_dim=TPB,
        )

        # 4. packed_out = attn @ V.
        var pout_tt = TileTensor(
            self.sp3.dev.value(), row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]()
        )
        var pv_tt = TileTensor(
            self.sp2.dev.value(), row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]()
        )
        batched_matmul[target="gpu"](pout_tt, scores_tt, pv_tt, context=c)

        # 5. unpack → output.
        comptime up_blocks = (pelems + TPB - 1) // TPB
        comptime up_k = _attn_unpack_out_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.OUT_DIM, PACKED,
        ]
        c.enqueue_function[up_k](
            out.lt["gpu", lay_out](),
            self.sp3.lt["gpu", lay_p](),
            grid_dim=up_blocks, block_dim=TPB,
        )

    def _forward_cpu[B: Int](mut self, mut in0: Tensor, mut out: Tensor) raises:
        # Mirror the GPU bmm forward (pack → QKᵀ bmm → scalar softmax+mask →
        # attn·V bmm → unpack) with target="cpu" (Apple-Accelerate). The 2
        # GEMMs are BLAS; softmax + causal mask stay scalar. Cache layout
        # [Q|K|V|scores] is identical to the GPU path.
        out.ensure(B * Self.OUT_DIM)
        self.cache.ensure(B * Self.CACHE_SIZE)
        comptime IN = Self.IN_DIMS[0]
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        comptime SD = Self.SEQ_LEN * Self.DIM
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(Self.HEAD_DIM)))

        ref ip = in0.data
        ref op = out.data
        ref cp = self.cache.data

        # Local scratch (separate Lists, no pointer slicing).
        var pq = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pk = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pv = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pout = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var sc = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))

        # 1. Cache Q/K/V and pack into (BH, SEQ, HEAD_DIM).
        for b in range(B):
            for i in range(SD):
                cp[b * C + i] = ip[b * IN + i]
                cp[b * C + Self.K_OFF + i] = ip[b * IN + Self.K_OFF + i]
                cp[b * C + Self.V_OFF + i] = ip[b * IN + Self.V_OFF + i]
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var h_off = h * Self.HEAD_DIM
                for t in range(Self.SEQ_LEN):
                    for d in range(Self.HEAD_DIM):
                        var col = h_off + d
                        var pidx = bh * Self.SEQ_LEN * Self.HEAD_DIM + t * Self.HEAD_DIM + d
                        pq[pidx] = ip[b * IN + t * Self.DIM + col]
                        pk[pidx] = ip[b * IN + Self.K_OFF + t * Self.DIM + col]
                        pv[pidx] = ip[b * IN + Self.V_OFF + t * Self.DIM + col]

        # 2. scores = Q @ Kᵀ  (BH, SEQ, SEQ).
        var scores_tt = TileTensor(
            sc, row_major[BH, Self.SEQ_LEN, Self.SEQ_LEN]()
        )
        var pq_tt = TileTensor(pq, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        var pk_tt = TileTensor(pk, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        batched_matmul[transpose_b=True, target="cpu"](
            scores_tt, pq_tt, pk_tt
        )

        # 3. scale + stable softmax (scalar); mirror weights into cache.attn
        #    and the `sc` scores buffer (zeroed in the causal upper triangle).
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var sc_base = bh * Self.SEQ_LEN * Self.SEQ_LEN
                var cache_base = (
                    b * C + Self.ATTN_OFF + h * Self.SEQ_LEN * Self.SEQ_LEN
                )
                for i in range(Self.SEQ_LEN):
                    var j_end = Self.SEQ_LEN
                    comptime if Self.CAUSAL:
                        j_end = i + 1
                    var row = sc_base + i * Self.SEQ_LEN
                    var crow = cache_base + i * Self.SEQ_LEN
                    var mx = Scalar[DT](-1e30)
                    for j in range(j_end):
                        var s = sc[row + j] * scale
                        sc[row + j] = s
                        if s > mx:
                            mx = s
                    var se = Scalar[DT](0)
                    for j in range(j_end):
                        var e = exp(sc[row + j] - mx)
                        sc[row + j] = e
                        se += e
                    var inv = Scalar[DT](1) / se
                    for j in range(j_end):
                        var w = sc[row + j] * inv
                        sc[row + j] = w
                        cp[crow + j] = w
                    comptime if Self.CAUSAL:
                        for j in range(i + 1, Self.SEQ_LEN):
                            sc[row + j] = Scalar[DT](0)
                            cp[crow + j] = Scalar[DT](0)

        # 4. packed_out = attn @ V  (BH, SEQ, HEAD_DIM).
        var pout_tt = TileTensor(
            pout, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]()
        )
        var pv_tt = TileTensor(pv, row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        batched_matmul[target="cpu"](pout_tt, scores_tt, pv_tt)

        # 5. unpack packed_out → output.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var h_off = h * Self.HEAD_DIM
                for t in range(Self.SEQ_LEN):
                    for d in range(Self.HEAD_DIM):
                        var pidx = bh * Self.SEQ_LEN * Self.HEAD_DIM + t * Self.HEAD_DIM + d
                        op[b * OUT + t * Self.DIM + h_off + d] = pout[pidx]
        _ = pq^
        _ = pk^
        _ = pv^
        _ = pout^
        _ = sc^

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # forward_input unused — this leaf is output-caching (reads only the
        # cache + grad_output).
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            self._vjp_cpu[B](grad_output, gin)
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_DIMS[0])
            comptime if Self.USE_MAX_KERNELS:
                self._vjp_gpu_bmm[B](grad_output, gin, c)
            else:
                self._vjp_gpu_custom[B](grad_output, gin, c)

    def _vjp_gpu_custom[
        B: Int
    ](mut self, mut grad_output: Tensor, mut gin: Tensor, c: DeviceContext) raises:
        comptime lay_in = Layout.row_major(B, Self.IN_DIMS[0])
        comptime lay_out = Layout.row_major(B, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(B, Self.CACHE_SIZE)
        comptime grid_bh = B * Self.N_HEADS
        # 1) zero grad_input.
        comptime zk = _attn_zero_grad_kernel[B, Self.IN_DIMS[0]]
        comptime zn = (B * Self.IN_DIMS[0] + TPB - 1) // TPB
        c.enqueue_function[zk](
            gin.lt["gpu", lay_in](), grid_dim=zn, block_dim=TPB
        )
        # 2) dV (reads attn weights — must precede dscore_dQ overwrite).
        comptime dvk = _attn_dV_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.V_OFF, Self.ATTN_OFF,
        ]
        c.enqueue_function[dvk](
            gin.lt["gpu", lay_in](),
            grad_output.lt["gpu", lay_out](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=grid_bh, block_dim=TPB,
        )
        # 3) dscore + dQ (overwrites cache.attn with d_score).
        comptime dqk = _attn_dscore_dQ_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF,
        ]
        c.enqueue_function[dqk](
            gin.lt["gpu", lay_in](),
            grad_output.lt["gpu", lay_out](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=grid_bh, block_dim=TPB,
        )
        # 4) dK (reads d_score from cache.attn).
        comptime dkk = _attn_dK_kernel[
            B, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIMS[0], Self.CACHE_SIZE,
            Self.K_OFF, Self.ATTN_OFF,
        ]
        c.enqueue_function[dkk](
            gin.lt["gpu", lay_in](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=grid_bh, block_dim=TPB,
        )

    def _vjp_gpu_bmm[
        B: Int
    ](mut self, mut grad_output: Tensor, mut gin: Tensor, c: DeviceContext) raises:
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        comptime SL = Self.SEQ_LEN
        comptime HD = Self.HEAD_DIM
        self._ensure_scratch_gpu[B](c)

        # Scratch aliasing (slabs recycled once their producer's last read is
        # enqueued — safe: kernels on the stream run in order). Map:
        #   sp0: pdout  → (step7) dK      sp1: pq     → (step8) dQ
        #   sp2: pk                        sp3: pv     → (step5) dV
        #   ss0: dattn  → (step4) attn_T → (step6) dscore_T
        #   ss1: dscore
        comptime lay_in = Layout.row_major(B, Self.IN_DIMS[0])
        comptime lay_out = Layout.row_major(B, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(B, Self.CACHE_SIZE)
        comptime lay_p = Layout.row_major(PACKED)
        comptime lay_s = Layout.row_major(SCORES)

        comptime pelems = B * SL * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        comptime sblocks = (SCORES + TPB - 1) // TPB

        # 1. pack dout + cache Q/K/V → (BH, SEQ, HEAD_DIM).
        comptime pin_k = _attn_pack_in_bwd_kernel[
            B, Self.DIM, Self.N_HEADS, SL, HD,
            Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE, PACKED,
        ]
        c.enqueue_function[pin_k](
            self.sp0.lt["gpu", lay_p](),
            self.sp1.lt["gpu", lay_p](),
            self.sp2.lt["gpu", lay_p](),
            self.sp3.lt["gpu", lay_p](),
            grad_output.lt["gpu", lay_out](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. dattn(ss0) = dout @ Vᵀ.
        var pdout_tt = TileTensor(self.sp0.dev.value(), row_major[BH, SL, HD]())
        var pq_tt = TileTensor(self.sp1.dev.value(), row_major[BH, SL, HD]())
        var pk_tt = TileTensor(self.sp2.dev.value(), row_major[BH, SL, HD]())
        var pv_tt = TileTensor(self.sp3.dev.value(), row_major[BH, SL, HD]())
        var dattn_tt = TileTensor(self.ss0.dev.value(), row_major[BH, SL, SL]())
        batched_matmul[transpose_b=True, target="gpu"](
            dattn_tt, pdout_tt, pv_tt, context=c
        )

        # 3. softmax jvp → dscore(ss1)  (reads dattn(ss0)).
        comptime jvp_k = _attn_softmax_jvp_kernel[
            B, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[jvp_k](
            self.ss1.lt["gpu", lay_s](),
            self.ss0.lt["gpu", lay_s](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=BH, block_dim=TPB,
        )

        # 4. attn_T(ss0) = transpose(cache.attn)  (ss0 free — dattn consumed).
        comptime tac_k = _attn_transpose_from_cache_kernel[
            B, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        c.enqueue_function[tac_k](
            self.ss0.lt["gpu", lay_s](),
            self.cache.lt["gpu", lay_c](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 5. dV(sp3) = attn_T(ss0) @ dout(sp0)  (sp3 free — pv last read step 2).
        var attnT_tt = TileTensor(self.ss0.dev.value(), row_major[BH, SL, SL]())
        var dV_tt = TileTensor(self.sp3.dev.value(), row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dV_tt, attnT_tt, pdout_tt, context=c)

        # 6. dscore_T(ss0) = transpose(dscore(ss1)) (ss0 free — attn_T read s5).
        comptime ts_k = _attn_transpose_scores_kernel[SL, SCORES, BH]
        c.enqueue_function[ts_k](
            self.ss0.lt["gpu", lay_s](),
            self.ss1.lt["gpu", lay_s](),
            grid_dim=sblocks, block_dim=TPB,
        )

        # 7. dK(sp0) = dscore_T(ss0) @ Q(sp1)  (sp0 free — pdout last read s5).
        var dscoreT_tt = TileTensor(self.ss0.dev.value(), row_major[BH, SL, SL]())
        var dK_tt = TileTensor(self.sp0.dev.value(), row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dK_tt, dscoreT_tt, pq_tt, context=c)

        # 8. dQ(sp1) = dscore(ss1) @ K(sp2)  (sp1 free — pq last read step 7).
        var dscore_tt = TileTensor(self.ss1.dev.value(), row_major[BH, SL, SL]())
        var dQ_tt = TileTensor(self.sp1.dev.value(), row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dQ_tt, dscore_tt, pk_tt, context=c)

        # 9. unpack dQ(sp1)/dK(sp0)/dV(sp3) → grad_input.
        comptime ug_k = _attn_unpack_grad_kernel[
            B, Self.DIM, Self.N_HEADS, SL, HD, Self.IN_DIMS[0], PACKED,
        ]
        c.enqueue_function[ug_k](
            gin.lt["gpu", lay_in](),
            self.sp1.lt["gpu", lay_p](),
            self.sp0.lt["gpu", lay_p](),
            self.sp3.lt["gpu", lay_p](),
            grid_dim=pblocks, block_dim=TPB,
        )

    def _vjp_cpu[
        B: Int
    ](mut self, mut grad_output: Tensor, mut gin: Tensor) raises:
        # Mirror the GPU bmm backward (pack → dattn=dout·Vᵀ bmm → scalar
        # softmax-JVP → transposes → dV/dK/dQ bmms → unpack) with target="cpu".
        gin.ensure(B * Self.IN_DIMS[0])
        comptime IN = Self.IN_DIMS[0]
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        comptime SL = Self.SEQ_LEN
        comptime HD = Self.HEAD_DIM
        comptime BH = B * Self.N_HEADS
        comptime PACKED = B * SL * Self.DIM
        comptime SCORES = BH * SL * SL
        var scale = Scalar[DT](Float32(1.0) / sqrt(Float32(Self.HEAD_DIM)))

        ref gop = grad_output.data
        ref gip = gin.data
        ref cp = self.cache.data

        # Local scratch (separate Lists, no pointer slicing).
        var pdout = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pq = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pk = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var pv = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var dV = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var dK = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var dQ = List[Scalar[DT]](length=PACKED, fill=Scalar[DT](0))
        var dattn = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var dscore = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var attn_T = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))
        var dscore_T = List[Scalar[DT]](length=SCORES, fill=Scalar[DT](0))

        # 1. pack dout + cache Q/K/V → (BH, SEQ, HEAD_DIM).
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var h_off = h * HD
                for t in range(SL):
                    for d in range(HD):
                        var col = h_off + d
                        var pidx = bh * SL * HD + t * HD + d
                        pdout[pidx] = gop[b * OUT + t * Self.DIM + col]
                        pq[pidx] = cp[b * C + t * Self.DIM + col]
                        pk[pidx] = cp[b * C + Self.K_OFF + t * Self.DIM + col]
                        pv[pidx] = cp[b * C + Self.V_OFF + t * Self.DIM + col]

        # 2. dattn = dout @ Vᵀ  (BH, SEQ, SEQ).
        var dattn_tt = TileTensor(dattn, row_major[BH, SL, SL]())
        var pdout_tt = TileTensor(pdout, row_major[BH, SL, HD]())
        var pv_tt = TileTensor(pv, row_major[BH, SL, HD]())
        batched_matmul[transpose_b=True, target="cpu"](
            dattn_tt, pdout_tt, pv_tt
        )

        # 3. softmax jvp (scalar): dscore = scale*a*(dattn - Σ_k a_k*dattn_k).
        #    a is cache.attn (causal upper-triangle already zeroed) → dscore is
        #    automatically zero where masked. Also build attn_T / dscore_T.
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var sc_base = bh * SL * SL
                var cache_base = b * C + Self.ATTN_OFF + h * SL * SL
                for i in range(SL):
                    var row = sc_base + i * SL
                    var crow = cache_base + i * SL
                    var s = Scalar[DT](0)
                    for j in range(SL):
                        s += cp[crow + j] * dattn[row + j]
                    for j in range(SL):
                        var a = cp[crow + j]
                        dscore[row + j] = scale * a * (dattn[row + j] - s)
                        attn_T[sc_base + j * SL + i] = a
                        dscore_T[sc_base + j * SL + i] = dscore[row + j]

        # 4. dV = attn_T @ dout.
        var attnT_tt = TileTensor(attn_T, row_major[BH, SL, SL]())
        var dV_tt = TileTensor(dV, row_major[BH, SL, HD]())
        batched_matmul[target="cpu"](dV_tt, attnT_tt, pdout_tt)

        # 5. dK = dscore_T @ Q.
        var dscoreT_tt = TileTensor(dscore_T, row_major[BH, SL, SL]())
        var pq_tt = TileTensor(pq, row_major[BH, SL, HD]())
        var dK_tt = TileTensor(dK, row_major[BH, SL, HD]())
        batched_matmul[target="cpu"](dK_tt, dscoreT_tt, pq_tt)

        # 6. dQ = dscore @ K.
        var dscore_tt = TileTensor(dscore, row_major[BH, SL, SL]())
        var pk_tt = TileTensor(pk, row_major[BH, SL, HD]())
        var dQ_tt = TileTensor(dQ, row_major[BH, SL, HD]())
        batched_matmul[target="cpu"](dQ_tt, dscore_tt, pk_tt)

        # 7. unpack dQ/dK/dV → grad_input.
        for i in range(B * IN):
            gip[i] = 0.0
        for b in range(B):
            for h in range(Self.N_HEADS):
                var bh = b * Self.N_HEADS + h
                var h_off = h * HD
                for t in range(SL):
                    for d in range(HD):
                        var col = h_off + d
                        var pidx = bh * SL * HD + t * HD + d
                        gip[b * IN + t * Self.DIM + col] = dQ[pidx]
                        gip[b * IN + Self.K_OFF + t * Self.DIM + col] = dK[pidx]
                        gip[b * IN + Self.V_OFF + t * Self.DIM + col] = dV[pidx]
        _ = pdout^
        _ = pq^
        _ = pk^
        _ = pv^
        _ = dV^
        _ = dK^
        _ = dQ^
        _ = dattn^
        _ = dscore^
        _ = attn_T^
        _ = dscore_T^

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (no Param fields → no-op).
