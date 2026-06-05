"""ScaledDotProductAttention[DIM, N_HEADS, SEQ_LEN, CAUSAL, USE_MAX_KERNELS].

Multi-head scaled dot-product attention as a single nn2 leaf. Input is the
per-token concatenated `[Q ‖ K ‖ V]` (each `DIM`-wide), laid out per sample
as `[all-Q tokens | all-K tokens | all-V tokens]`:

    IN_DIM  = SEQ_LEN * DIM * 3        (offsets: Q@0, K@SEQ·DIM, V@2·SEQ·DIM)
    OUT_DIM = SEQ_LEN * DIM

No params. Cache is leaf-owned (its own buffer, NOT the Sequential input
slab), laid out `[Q | K | V | scores]`:

    CACHE_SIZE = 3*SEQ_LEN*DIM + N_HEADS*SEQ_LEN*SEQ_LEN

Because the op is **output-cached** (it copies Q/K/V into its own cache and
materializes the softmaxed scores there), backward reads only `self.cache`
and `grad_output` — never the forward input slab — so it is EXEMPT from the
param-grad-before-grad_input aliasing invariant (and has no params anyway).
A future fused rewrite must preserve that property or reintroduce the trap.

`head_dim = DIM // N_HEADS`, `scale = 1/sqrt(head_dim)`. `causal=True` bounds
each query i's key loop to j ≤ i. Softmax is computed in fp32 with the
standard max-shift for stability (CPU accumulates in Float64).

GPU path: `USE_MAX_KERNELS=True` (default) → batched-GEMM attention (Wave C
6d, tensor cores); `False` → serial per-(b,h) custom kernels (6c). The two
are bit-identical (see tests/nn2/test_attention_bmm_parity.mojo); the flag
only changes speed. CPU path (forward + 3-pass vjp, 6a/6b) ignores the flag.
Docs: docs/NN2_TRANSFORMER_PORT.md.
"""

from std.math import exp, sqrt
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.bmm import batched_matmul

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — custom per-(b,h) path (Wave C 6c). One block per (b,h);
# threads stride over rows (fwd / dQ) or (j,d) pairs (dV / dK). Ported
# from gen-1 nn/autodiff/primitives/attention.mojo. Float32 throughout
# (Metal has no Float64). No intra-block barrier needed: the forward's
# score/softmax/output for row i touch only cache.attn[h,i,·] (the thread
# owning row i), and read Q/K/V from `input` directly, not from cache.
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
# BMM fast path (Wave C 6d) — batched-GEMM attention behind USE_MAX_KERNELS.
# Replaces the serial per-(b,h) custom kernels with single-launch batched
# matmuls (tensor cores) for QKᵀ and attn·V, plus pack/softmax/unpack glue.
# Ported from gen-1 nn/autodiff/primitives/attention.mojo. Same cache layout
# [Q|K|V|scores] as the custom path, so the two are interchangeable (the
# regression test pins them bit-close). Packed layout: (BH, SEQ, HEAD_DIM);
# scores layout: (BH, SEQ, SEQ).
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

    # Cache (leaf-owned, output-caching).
    var cache: List[Scalar[DT]]                 # [BATCH, CACHE_SIZE]
    var cache_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.DIM % Self.N_HEADS == 0
        ), "ScaledDotProductAttention: DIM must be divisible by N_HEADS"
        self.cache = List[Scalar[DT]]()
        self.cache_dev = None
        self.cache_n_batch = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "ScaledDotProductAttention: target must be 'cpu' or 'gpu'"
        )
        var a = Self()
        comptime if target == "cpu":
            a.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error(
                    "ScaledDotProductAttention.make[target='gpu']: ctx required"
                )
            var ctx_v = ctx.value()
            a.cache_dev = ctx_v.enqueue_create_buffer[DT](1)
            a.cache_n_batch = 0
            a.ts = TargetStorage.make_gpu(ctx_v)
        return a^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.CACHE_SIZE
            )
            self.cache_n_batch = batch

    @staticmethod
    def display_label() -> String:
        return String("Attention")

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["ScaledDotProductAttention", target](
            self.ts.target_tag
        )
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self._forward_cpu[BATCH](input, output_v)
        else:
            self._ensure_cache_gpu(BATCH)
            comptime if Self.USE_MAX_KERNELS:
                self._forward_gpu_bmm[BATCH](input, output_v)
            else:
                self._forward_gpu_custom[BATCH](input, output_v)

    def _forward_gpu_custom[
        BATCH: Int
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime lay_in = Layout.row_major(BATCH, Self.IN_DIMS[0])
        comptime lay_out = Layout.row_major(BATCH, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(BATCH, Self.CACHE_SIZE)
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            output_v.ptr
        )
        var in_lt = LayoutTensor[DT, lay_in, MutAnyOrigin](in_p)
        var out_lt = LayoutTensor[DT, lay_out, MutAnyOrigin](out_p)
        var c_lt = LayoutTensor[DT, lay_c, MutAnyOrigin](
            self.cache_dev.value()
        )
        comptime kernel = _attn_fwd_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF,
        ]
        self.ts.ctx.value().enqueue_function[kernel](
            out_lt, in_lt, c_lt,
            grid_dim=BATCH * Self.N_HEADS, block_dim=TPB,
        )

    def _forward_gpu_bmm[
        BATCH: Int
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime BH = BATCH * Self.N_HEADS
        comptime PACKED = BATCH * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        var ctx = self.ts.ctx.value()

        var pq = ctx.enqueue_create_buffer[DT](PACKED)
        var pk = ctx.enqueue_create_buffer[DT](PACKED)
        var pv = ctx.enqueue_create_buffer[DT](PACKED)
        var pout = ctx.enqueue_create_buffer[DT](PACKED)
        var sc = ctx.enqueue_create_buffer[DT](SCORES)

        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
        var in_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.IN_DIMS[0]), MutAnyOrigin
        ](in_p)
        var c_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ](self.cache_dev.value())

        # 1. pack QKV → (BH, SEQ, HEAD_DIM) + write cache.
        comptime pelems = BATCH * Self.SEQ_LEN * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        var pq_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pq.unsafe_ptr())
        var pk_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pk.unsafe_ptr())
        var pv_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pv.unsafe_ptr())
        comptime pack_k = _attn_pack_qkv_fwd_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.IN_DIMS[0], Self.CACHE_SIZE, PACKED,
        ]
        ctx.enqueue_function[pack_k](
            pq_lt, pk_lt, pv_lt, c_lt, in_lt,
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. scores = Q @ Kᵀ  (BH, SEQ, SEQ).
        var scores_tt = TileTensor(sc.unsafe_ptr(), row_major[BH, Self.SEQ_LEN, Self.SEQ_LEN]())
        var pq_tt = TileTensor(pq.unsafe_ptr(), row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        var pk_tt = TileTensor(pk.unsafe_ptr(), row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        batched_matmul[transpose_b=True, target="gpu"](
            scores_tt, pq_tt, pk_tt, context=ctx
        )

        # 3. scale + stable softmax in-place; mirror into cache.attn.
        var sc_lt = LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin](sc.unsafe_ptr())
        comptime sm_k = _attn_softmax_kernel[
            BATCH, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM, Self.CAUSAL,
            Self.CACHE_SIZE, SCORES, BH,
        ]
        ctx.enqueue_function[sm_k](sc_lt, c_lt, grid_dim=BH, block_dim=TPB)

        # 4. packed_out = attn @ V.
        var pout_tt = TileTensor(pout.unsafe_ptr(), row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        var pv_tt = TileTensor(pv.unsafe_ptr(), row_major[BH, Self.SEQ_LEN, Self.HEAD_DIM]())
        batched_matmul[target="gpu"](pout_tt, scores_tt, pv_tt, context=ctx)

        # 5. unpack → output.
        var out_lt = LayoutTensor[
            DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](out_p)
        var pout_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pout.unsafe_ptr())
        comptime up_blocks = (pelems + TPB - 1) // TPB
        comptime up_k = _attn_unpack_out_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.OUT_DIM, PACKED,
        ]
        ctx.enqueue_function[up_k](
            out_lt, pout_lt, grid_dim=up_blocks, block_dim=TPB
        )

    def _forward_cpu[
        BATCH: Int
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        ensure_cpu_buffer(self.cache, BATCH * Self.CACHE_SIZE)
        var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            output_v.ptr
        )
        var cp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.cache.unsafe_ptr()
        )
        comptime IN = Self.IN_DIMS[0]
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        comptime SD = Self.SEQ_LEN * Self.DIM
        var scale = 1.0 / sqrt(Float64(Self.HEAD_DIM))

        for b in range(BATCH):
            # Cache Q, K, V (same per-token layout as input).
            for i in range(SD):
                cp[b * C + i] = ip[b * IN + i]
                cp[b * C + Self.K_OFF + i] = ip[b * IN + Self.K_OFF + i]
                cp[b * C + Self.V_OFF + i] = ip[b * IN + Self.V_OFF + i]

            for h in range(Self.N_HEADS):
                var h_off = h * Self.HEAD_DIM
                for i in range(Self.SEQ_LEN):
                    var j_end = Self.SEQ_LEN
                    comptime if Self.CAUSAL:
                        j_end = i + 1

                    var max_score: Float64 = -1e30
                    for j in range(j_end):
                        var score: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var q = Float64(ip[b * IN + i * Self.DIM + h_off + d])
                            var k = Float64(
                                ip[
                                    b * IN + Self.K_OFF + j * Self.DIM + h_off + d
                                ]
                            )
                            score += q * k
                        score *= scale
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        cp[ai] = Scalar[DT](score)
                        if score > max_score:
                            max_score = score

                    var sum_exp: Float64 = 0.0
                    for j in range(j_end):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var e = exp(Float64(cp[ai]) - max_score)
                        cp[ai] = Scalar[DT](e)
                        sum_exp += e

                    var inv = 1.0 / sum_exp
                    for j in range(j_end):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        cp[ai] = Scalar[DT](Float64(cp[ai]) * inv)

                    for d in range(Self.HEAD_DIM):
                        var acc: Float64 = 0.0
                        for j in range(j_end):
                            var ai = (
                                b * C + Self.ATTN_OFF
                                + h * Self.SEQ_LEN * Self.SEQ_LEN
                                + i * Self.SEQ_LEN + j
                            )
                            var v = Float64(
                                ip[
                                    b * IN + Self.V_OFF + j * Self.DIM + h_off + d
                                ]
                            )
                            acc += Float64(cp[ai]) * v
                        op[b * OUT + i * Self.DIM + h_off + d] = Scalar[DT](acc)

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["ScaledDotProductAttention", target](
            self.ts.target_tag
        )
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            self._vjp_cpu[BATCH](grad_output_v, grad_input_v)
        else:
            comptime if Self.USE_MAX_KERNELS:
                self._vjp_gpu_bmm[BATCH](grad_output_v, grad_input_v)
            else:
                self._vjp_gpu_custom[BATCH](grad_output_v, grad_input_v)

    def _vjp_gpu_custom[
        BATCH: Int
    ](
        mut self,
        grad_output_v: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_input_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var ctx = self.ts.ctx.value()
        comptime lay_in = Layout.row_major(BATCH, Self.IN_DIMS[0])
        comptime lay_out = Layout.row_major(BATCH, Self.OUT_DIM)
        comptime lay_c = Layout.row_major(BATCH, Self.CACHE_SIZE)
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_output_v.ptr
        )
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_input_v.ptr
        )
        var go_lt = LayoutTensor[DT, lay_out, MutAnyOrigin](go_p)
        var gi_lt = LayoutTensor[DT, lay_in, MutAnyOrigin](gi_p)
        var c_lt = LayoutTensor[DT, lay_c, MutAnyOrigin](
            self.cache_dev.value()
        )
        comptime grid_bh = BATCH * Self.N_HEADS
        # 1) zero grad_input.
        comptime zk = _attn_zero_grad_kernel[BATCH, Self.IN_DIMS[0]]
        comptime zn = (BATCH * Self.IN_DIMS[0] + TPB - 1) // TPB
        ctx.enqueue_function[zk](gi_lt, grid_dim=zn, block_dim=TPB)
        # 2) dV (reads attn weights — must precede dscore_dQ overwrite).
        comptime dvk = _attn_dV_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.V_OFF, Self.ATTN_OFF,
        ]
        ctx.enqueue_function[dvk](
            gi_lt, go_lt, c_lt, grid_dim=grid_bh, block_dim=TPB
        )
        # 3) dscore + dQ (overwrites cache.attn with d_score).
        comptime dqk = _attn_dscore_dQ_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE,
            Self.K_OFF, Self.V_OFF, Self.ATTN_OFF,
        ]
        ctx.enqueue_function[dqk](
            gi_lt, go_lt, c_lt, grid_dim=grid_bh, block_dim=TPB
        )
        # 4) dK (reads d_score from cache.attn).
        comptime dkk = _attn_dK_kernel[
            BATCH, Self.DIM, Self.N_HEADS, Self.SEQ_LEN, Self.HEAD_DIM,
            Self.CAUSAL, Self.IN_DIMS[0], Self.CACHE_SIZE,
            Self.K_OFF, Self.ATTN_OFF,
        ]
        ctx.enqueue_function[dkk](
            gi_lt, c_lt, grid_dim=grid_bh, block_dim=TPB
        )

    def _vjp_gpu_bmm[
        BATCH: Int
    ](
        mut self,
        grad_output_v: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_input_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime BH = BATCH * Self.N_HEADS
        comptime PACKED = BATCH * Self.SEQ_LEN * Self.DIM
        comptime SCORES = BH * Self.SEQ_LEN * Self.SEQ_LEN
        comptime SL = Self.SEQ_LEN
        comptime HD = Self.HEAD_DIM
        var ctx = self.ts.ctx.value()

        var pdout = ctx.enqueue_create_buffer[DT](PACKED)
        var pq = ctx.enqueue_create_buffer[DT](PACKED)
        var pk = ctx.enqueue_create_buffer[DT](PACKED)
        var pv = ctx.enqueue_create_buffer[DT](PACKED)
        var dattn = ctx.enqueue_create_buffer[DT](SCORES)
        var dscore = ctx.enqueue_create_buffer[DT](SCORES)
        var attn_T = ctx.enqueue_create_buffer[DT](SCORES)  # also reused for dscore_T
        var dQ = ctx.enqueue_create_buffer[DT](PACKED)
        var dK = ctx.enqueue_create_buffer[DT](PACKED)
        var dV = ctx.enqueue_create_buffer[DT](PACKED)

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
        var go_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin](go_p)
        var gi_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.IN_DIMS[0]), MutAnyOrigin](gi_p)
        var c_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin](
            self.cache_dev.value()
        )

        comptime pelems = BATCH * SL * Self.DIM
        comptime pblocks = (pelems + TPB - 1) // TPB
        comptime sblocks = (SCORES + TPB - 1) // TPB

        # 1. pack dout + cache Q/K/V → (BH, SEQ, HEAD_DIM).
        var pdout_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pdout.unsafe_ptr())
        var pq_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pq.unsafe_ptr())
        var pk_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pk.unsafe_ptr())
        var pv_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](pv.unsafe_ptr())
        comptime pin_k = _attn_pack_in_bwd_kernel[
            BATCH, Self.DIM, Self.N_HEADS, SL, HD,
            Self.IN_DIMS[0], Self.OUT_DIM, Self.CACHE_SIZE, PACKED,
        ]
        ctx.enqueue_function[pin_k](
            pdout_lt, pq_lt, pk_lt, pv_lt, go_lt, c_lt,
            grid_dim=pblocks, block_dim=TPB,
        )

        # 2. dattn = dout @ Vᵀ.
        var pdout_tt = TileTensor(pdout.unsafe_ptr(), row_major[BH, SL, HD]())
        var pq_tt = TileTensor(pq.unsafe_ptr(), row_major[BH, SL, HD]())
        var pk_tt = TileTensor(pk.unsafe_ptr(), row_major[BH, SL, HD]())
        var pv_tt = TileTensor(pv.unsafe_ptr(), row_major[BH, SL, HD]())
        var dattn_tt = TileTensor(dattn.unsafe_ptr(), row_major[BH, SL, SL]())
        batched_matmul[transpose_b=True, target="gpu"](
            dattn_tt, pdout_tt, pv_tt, context=ctx
        )

        # 3. softmax jvp → dscore.
        var dattn_lt = LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin](dattn.unsafe_ptr())
        var dscore_lt = LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin](dscore.unsafe_ptr())
        comptime jvp_k = _attn_softmax_jvp_kernel[
            BATCH, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        ctx.enqueue_function[jvp_k](dscore_lt, dattn_lt, c_lt, grid_dim=BH, block_dim=TPB)

        # 4. attn_T = transpose(cache.attn).
        var attnT_lt = LayoutTensor[DT, Layout.row_major(SCORES), MutAnyOrigin](attn_T.unsafe_ptr())
        comptime tac_k = _attn_transpose_from_cache_kernel[
            BATCH, Self.N_HEADS, SL, HD, Self.CACHE_SIZE, SCORES, BH,
        ]
        ctx.enqueue_function[tac_k](attnT_lt, c_lt, grid_dim=sblocks, block_dim=TPB)

        # 5. dV = attn_T @ dout.
        var attnT_tt = TileTensor(attn_T.unsafe_ptr(), row_major[BH, SL, SL]())
        var dV_tt = TileTensor(dV.unsafe_ptr(), row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dV_tt, attnT_tt, pdout_tt, context=ctx)

        # 6. dscore_T = transpose(dscore) (reuse attn_T buffer).
        comptime ts_k = _attn_transpose_scores_kernel[SL, SCORES, BH]
        ctx.enqueue_function[ts_k](attnT_lt, dscore_lt, grid_dim=sblocks, block_dim=TPB)

        # 7. dK = dscore_T @ Q.
        var dscoreT_tt = TileTensor(attn_T.unsafe_ptr(), row_major[BH, SL, SL]())
        var dK_tt = TileTensor(dK.unsafe_ptr(), row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dK_tt, dscoreT_tt, pq_tt, context=ctx)

        # 8. dQ = dscore @ K.
        var dscore_tt = TileTensor(dscore.unsafe_ptr(), row_major[BH, SL, SL]())
        var dQ_tt = TileTensor(dQ.unsafe_ptr(), row_major[BH, SL, HD]())
        batched_matmul[target="gpu"](dQ_tt, dscore_tt, pk_tt, context=ctx)

        # 9. unpack dQ/dK/dV → grad_input.
        var dQ_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](dQ.unsafe_ptr())
        var dK_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](dK.unsafe_ptr())
        var dV_lt = LayoutTensor[DT, Layout.row_major(PACKED), MutAnyOrigin](dV.unsafe_ptr())
        comptime ug_k = _attn_unpack_grad_kernel[
            BATCH, Self.DIM, Self.N_HEADS, SL, HD, Self.IN_DIMS[0], PACKED,
        ]
        ctx.enqueue_function[ug_k](
            gi_lt, dQ_lt, dK_lt, dV_lt, grid_dim=pblocks, block_dim=TPB
        )

    def _vjp_cpu[
        BATCH: Int
    ](
        mut self,
        grad_output_v: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_input_v: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_output_v.ptr
        )
        var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_input_v.ptr
        )
        var cp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.cache.unsafe_ptr()
        )
        comptime IN = Self.IN_DIMS[0]
        comptime OUT = Self.OUT_DIM
        comptime C = Self.CACHE_SIZE
        var scale = 1.0 / sqrt(Float64(Self.HEAD_DIM))

        for i in range(BATCH * IN):
            gip[i] = 0.0

        for b in range(BATCH):
            for h in range(Self.N_HEADS):
                var h_off = h * Self.HEAD_DIM

                # Step 1: dV[j] += attn[i,j] * grad_out[i].
                for i in range(Self.SEQ_LEN):
                    var j_end = Self.SEQ_LEN
                    comptime if Self.CAUSAL:
                        j_end = i + 1
                    for j in range(j_end):
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var attn_w = Float64(cp[ai])
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var dv_idx = (
                                b * IN + Self.V_OFF + j * Self.DIM + h_off + d
                            )
                            gip[dv_idx] = gip[dv_idx] + Scalar[DT](attn_w * go)

                # Step 2: softmax backward → dQ, dK.
                for i in range(Self.SEQ_LEN):
                    var j_end = Self.SEQ_LEN
                    comptime if Self.CAUSAL:
                        j_end = i + 1

                    # Pass 1: dot_sum = Σ_j attn[i,j] * (grad_out[i]·V[j]).
                    var dot_sum: Float64 = 0.0
                    for j in range(j_end):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var v = Float64(
                                cp[b * C + Self.V_OFF + j * Self.DIM + h_off + d]
                            )
                            d_attn += go * v
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        dot_sum += d_attn * Float64(cp[ai])

                    # Pass 2: d_score and propagate to dQ, dK.
                    for j in range(j_end):
                        var d_attn: Float64 = 0.0
                        for d in range(Self.HEAD_DIM):
                            var go = Float64(
                                gop[b * OUT + i * Self.DIM + h_off + d]
                            )
                            var v = Float64(
                                cp[b * C + Self.V_OFF + j * Self.DIM + h_off + d]
                            )
                            d_attn += go * v
                        var ai = (
                            b * C + Self.ATTN_OFF
                            + h * Self.SEQ_LEN * Self.SEQ_LEN
                            + i * Self.SEQ_LEN + j
                        )
                        var attn_w = Float64(cp[ai])
                        var d_score = attn_w * (d_attn - dot_sum) * scale
                        for d in range(Self.HEAD_DIM):
                            var q = Float64(
                                cp[b * C + i * Self.DIM + h_off + d]
                            )
                            var k = Float64(
                                cp[b * C + Self.K_OFF + j * Self.DIM + h_off + d]
                            )
                            var dq_idx = b * IN + i * Self.DIM + h_off + d
                            gip[dq_idx] = gip[dq_idx] + Scalar[DT](d_score * k)
                            var dk_idx = (
                                b * IN + Self.K_OFF + j * Self.DIM + h_off + d
                            )
                            gip[dk_idx] = gip[dk_idx] + Scalar[DT](d_score * q)
