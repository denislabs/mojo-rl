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

Status: CPU forward + vjp implemented (Wave C 6a/6b). GPU path is the
custom per-(b,h) kernel set (Wave C 6c); the MAX-bmm fast path behind
USE_MAX_KERNELS (6d) is deferred. Docs: docs/NN2_TRANSFORMER_PORT.md.
"""

from std.math import exp, sqrt
from std.gpu import thread_idx, block_idx, block_dim, global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

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


struct ScaledDotProductAttention[
    DIM: Int,
    N_HEADS: Int,
    SEQ_LEN: Int,
    CAUSAL: Bool = False,
    USE_MAX_KERNELS: Bool = False,
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
