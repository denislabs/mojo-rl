"""Embedding[VOCAB, EMBED_DIM] — one-hot embedding lookup (storage surface).

Transformed from legacy `nn.primitives.Embedding`. The input is a `[BATCH,
VOCAB]` one-hot (DT); output `[BATCH, EMBED_DIM] = input @ weight`. One Param
(`weight` [VOCAB, EMBED_DIM], decay=True) + a leaf-owned cache of the one-hot
input. CPU uses the naive triple loops, GPU the 3 kernels (fwd / grad_in /
grad_w), carried over verbatim.

Backward: grad_in = grad_out @ Wᵀ ; grad_W += cache_inᵀ @ grad_out.

bf16-FLOW (AMP "Step B"): `Embedding[VOCAB, EMBED_DIM]` is fp32 (unchanged),
while `Embedding[VOCAB, EMBED_DIM, DType.bfloat16]` flows its ACTIVATIONS (the
one-hot input + the embedded output + grad_output/grad_input) at bf16
(`ACT_DT == bfloat16`). The master weight/grad STAY fp32 (`Param` is always
`DT`); only a CACHED bf16 weight copy (`w_bf`, version-gated — the forward GEMM
`out = input @ W` is the SAME `max_matmul` as Linear, so it needs a bf16 weight)
is low-precision. Output is bf16, grad_W accumulates bf16→fp32 master. The fp32
(ACT_DT == DT) path is byte-for-byte the legacy NoAMP path; the bf16 path is
GPU-only.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from .linear import (
    _transpose_tiled_kernel, _T_TILE, _T_BR, _cast_f2b_kernel,
)


# ── GPU kernels: the three matmuls go through max_matmul (mirrors Linear);
#    only the input-transpose (for grad_w = inputᵀ@grad_out, since max_matmul
#    rejects transpose_a) and the grad_weight accumulate remain hand-rolled.
#    The input-transpose reuses Linear's B1' tiled transpose. ──
# Dtype-parametric (`ADT`) on the grad_output activation: the bf16 path reads a
# bf16 `gw_tmp` (the fp32-out GEMM writes DT) — gw_tmp is ALWAYS fp32 here, so
# the accumulate stays DT regardless. (Kept ADT-free: only fp32 operands.)
def _emb_accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] += src[i]


struct Embedding[VOCAB_: Int, EMBED_DIM_: Int, ADT: DType = DT](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.VOCAB_)
    comptime OUT_DIM = Self.EMBED_DIM_
    comptime W_SIZE = Self.VOCAB_ * Self.EMBED_DIM_
    # Activation-flow dtype (satisfies the Module trait). `Embedding[V, D]` =
    # fp32 (ACT_DT == DT, the legacy path); `Embedding[V, D, bfloat16]` flows
    # activations at bf16 (the AMP "Step B" memory win).
    comptime ACT_DT = Self.ADT

    var weight: Param["weight", True, Self.W_SIZE]
    var cache_in: Tensor  # [BATCH, VOCAB] one-hot (CPU grad_w)
    # GPU grad_w scratch (lazy): cache_inᵀ [VOCAB, B] (transposed in forward) +
    # gw_tmp [VOCAB, EMBED_DIM] (max_matmul output before accumulate). Both STAY
    # fp32 (the bf16 grad_w GEMM writes a fp32 output → fp32 master grad).
    var cache_inT: Tensor
    var gw_tmp: Tensor
    # bf16-flow compute scratch (lazy; used only when ACT_DT == bf16 and
    # target == "gpu"). The master weight/grad stay fp32 (`Param`); only `w_bf`
    # is bf16 — the CACHED bf16 weight for the forward GEMM `out = input @ W`,
    # recast from `weight.val` only on a `weight.val.version` bump (tracked by
    # `_w_cast_version`), so the W cast is ONCE per optimizer step. `cache_inT_bf`
    # is the transposed bf16 fwd-input (backward grad_w). No input/output/grad
    # cast — activations ALREADY flow at bf16.
    var w_bf: TensorImpl[Self.ADT]
    var cache_inT_bf: TensorImpl[Self.ADT]
    var _w_cast_version: Int  # `weight.val.version` at last bf16 weight cast

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.cache_in = Tensor()
        self.cache_inT = Tensor()
        self.gw_tmp = Tensor()
        self.w_bf = TensorImpl[Self.ADT]()
        self.cache_inT_bf = TensorImpl[Self.ADT]()
        self._w_cast_version = -1  # < any real version → first forward casts

    def _ensure_w_bf(mut self, c: DeviceContext) raises:
        """Ensure the cached bf16 weight `w_bf` reflects the current fp32
        `weight.val`. Recasts ONLY on an optimizer version bump (cast once per
        step, not per fwd/bwd). Shared by forward (the cast) and vjp (reuses it —
        no optimizer step intervenes between a fwd and its bwd). Mirrors
        `Linear._ensure_w_bf`."""
        self.w_bf.ensure_gpu(c, Self.W_SIZE)
        if self.weight.val.version != self._w_cast_version:
            c.enqueue_function[_cast_f2b_kernel[Self.W_SIZE]](
                self.weight.val.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.w_bf.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                grid_dim=(Self.W_SIZE + 255) // 256,
                block_dim=256,
            )
            self._w_cast_version = self.weight.val.version

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var e = Self()
        e.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        # Honor INIT (was a fixed `(k%11-5)*0.07` placeholder that IGNORED it).
        # Embedding tables are usually Normal-init; pass the [in,out] = [VOCAB,
        # EMBED_DIM] convention so Kaiming/Xavier are well-defined too.
        # init_weight uploads to device on GPU.
        INIT.init_weight[target](
            e.weight.val, Self.W_SIZE, Self.VOCAB_, Self.EMBED_DIM_, ctx
        )
        return e^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT here, but the compiler doesn't collapse the opaque
            # `Self.ACT_DT` param to `DT` for unification against the fp32
            # weight views — so rebind the activation refs (sound; equal here).
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(B * Self.EMBED_DIM_)
                self.cache_in.ensure(B * Self.VOCAB_)
                var input = TileTensor(in0d.data, row_major[B, Self.VOCAB_]())
                var output_v = TileTensor(
                    outd.data, row_major[B, Self.EMBED_DIM_]()
                )
                var w_v = TileTensor(
                    self.weight.val.data,
                    row_major[Self.VOCAB_, Self.EMBED_DIM_](),
                )
                var cin = TileTensor(
                    self.cache_in.data, row_major[B, Self.VOCAB_]()
                )
                for b in range(B):
                    for v in range(Self.VOCAB_):
                        cin[b, v] = input[b, v]
                    for j in range(Self.EMBED_DIM_):
                        var acc: Scalar[DT] = 0.0
                        for v in range(Self.VOCAB_):
                            acc += input[b, v] * w_v[v, j]
                        output_v[b, j] = acc
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, B * Self.EMBED_DIM_)
                self.cache_inT.ensure_gpu(c, Self.VOCAB_ * B)
                comptime lbv = Layout.row_major(B, Self.VOCAB_)
                comptime lvb = Layout.row_major(Self.VOCAB_, B)
                # out[B, ED] = input[B, VOCAB] @ weight[VOCAB, ED]
                var in_v = TileTensor(
                    in0d.dev.value(), row_major[B, Self.VOCAB_]()
                )
                var w_v = TileTensor(
                    self.weight.val.dev.value(),
                    row_major[Self.VOCAB_, Self.EMBED_DIM_](),
                )
                var out_v = TileTensor(
                    outd.dev.value(), row_major[B, Self.EMBED_DIM_]()
                )
                max_matmul[target="gpu"](out_v, in_v, w_v, c)
                # cache_inᵀ[VOCAB, B] = input[B, VOCAB]ᵀ  (for grad_w in
                # backward), via Linear's B1' tiled transpose.
                c.enqueue_function[_transpose_tiled_kernel[B, Self.VOCAB_]](
                    in0d.lt["gpu", lbv](),
                    self.cache_inT.lt["gpu", lvb](),
                    grid_dim=(
                        (Self.VOCAB_ + _T_TILE - 1) // _T_TILE,
                        (B + _T_TILE - 1) // _T_TILE,
                    ),
                    block_dim=(_T_TILE, _T_BR),
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow Embedding is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.EMBED_DIM_)
            self.cache_inT_bf.ensure_gpu(c, Self.VOCAB_ * B)
            comptime lbv = Layout.row_major(B, Self.VOCAB_)
            comptime lvb = Layout.row_major(Self.VOCAB_, B)
            # input (in0) is ALREADY bf16 — no input cast. W: cached bf16 (recast
            # only on a version bump). out[B, ED] = input[B, VOCAB] @ W[VOCAB, ED]
            # bf16-in → bf16-out GEMM (fp32 accumulation is automatic).
            self._ensure_w_bf(c)
            var in_v = TileTensor(in0.dev.value(), row_major[B, Self.VOCAB_]())
            var w_bf_v = TileTensor(
                self.w_bf.dev.value(),
                row_major[Self.VOCAB_, Self.EMBED_DIM_](),
            )
            var out_v = TileTensor(
                out.dev.value(), row_major[B, Self.EMBED_DIM_]()
            )
            max_matmul[target="gpu"](out_v, in_v, w_bf_v, c)
            # cache_inᵀ[VOCAB, B] = input[B, VOCAB]ᵀ at bf16 (for grad_w), via
            # Linear's dtype-parametric tiled transpose (bf16 in → bf16 out).
            c.enqueue_function[_transpose_tiled_kernel[B, Self.VOCAB_, Self.ADT]](
                in0.lt["gpu", lbv](),
                self.cache_inT_bf.lt["gpu", lvb](),
                grid_dim=(
                    (Self.VOCAB_ + _T_TILE - 1) // _T_TILE,
                    (B + _T_TILE - 1) // _T_TILE,
                ),
                block_dim=(_T_TILE, _T_BR),
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            comptime if target == "cpu":
                gind.ensure(B * Self.VOCAB_)
                var go_v = TileTensor(
                    god.data, row_major[B, Self.EMBED_DIM_]()
                )
                var gi_v = TileTensor(gind.data, row_major[B, Self.VOCAB_]())
                var w_v = TileTensor(
                    self.weight.val.data,
                    row_major[Self.VOCAB_, Self.EMBED_DIM_](),
                )
                for b in range(B):
                    for v in range(Self.VOCAB_):
                        var acc: Scalar[DT] = 0.0
                        for j in range(Self.EMBED_DIM_):
                            acc += go_v[b, j] * w_v[v, j]
                        gi_v[b, v] = acc
                var gw_v = TileTensor(
                    self.weight.grd.data,
                    row_major[Self.VOCAB_, Self.EMBED_DIM_](),
                )
                var cin = TileTensor(
                    self.cache_in.data, row_major[B, Self.VOCAB_]()
                )
                for v in range(Self.VOCAB_):
                    for j in range(Self.EMBED_DIM_):
                        var acc: Scalar[DT] = 0.0
                        for b in range(B):
                            acc += cin[b, v] * go_v[b, j]
                        gw_v[v, j] += acc
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, B * Self.VOCAB_)
                self.gw_tmp.ensure_gpu(c, Self.W_SIZE)
                comptime lw = Layout.row_major(Self.VOCAB_, Self.EMBED_DIM_)
                comptime lwf = Layout.row_major(Self.W_SIZE)
                # grad_in[B, VOCAB] = grad_out[B, ED] @ weight[VOCAB, ED]ᵀ
                var go_v = TileTensor(
                    god.dev.value(), row_major[B, Self.EMBED_DIM_]()
                )
                var w_v = TileTensor(
                    self.weight.val.dev.value(),
                    row_major[Self.VOCAB_, Self.EMBED_DIM_](),
                )
                var gi_v = TileTensor(
                    gind.dev.value(), row_major[B, Self.VOCAB_]()
                )
                max_matmul[transpose_b=True, target="gpu"](gi_v, go_v, w_v, c)
                # gw_tmp[VOCAB, ED] = cache_inᵀ[VOCAB, B] @ grad_out[B, ED]
                var cinT_v = TileTensor(
                    self.cache_inT.dev.value(), row_major[Self.VOCAB_, B]()
                )
                var gwtmp_v = TileTensor(
                    self.gw_tmp.dev.value(),
                    row_major[Self.VOCAB_, Self.EMBED_DIM_](),
                )
                max_matmul[target="gpu"](gwtmp_v, cinT_v, go_v, c)
                # weight.grad += gw_tmp  (accumulate, matches legacy semantics)
                comptime gw_blk = (Self.W_SIZE + TPB - 1) // TPB
                c.enqueue_function[_emb_accum_kernel[Self.W_SIZE]](
                    self.weight.grd.lt["gpu", lwf](),
                    self.gw_tmp.lt["gpu", lwf](),
                    grid_dim=gw_blk,
                    block_dim=TPB,
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow Embedding is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.VOCAB_)
            self.gw_tmp.ensure_gpu(c, Self.W_SIZE)
            comptime lwf = Layout.row_major(Self.W_SIZE)
            # grad_output is ALREADY bf16 (no cast). W reuses the forward's cached
            # bf16 cast. grad_in[B, VOCAB] = grad_out[B, ED] @ W[VOCAB, ED]ᵀ →
            # bf16 gin (bf16-in, bf16-out — gin flows at bf16).
            self._ensure_w_bf(c)
            var go_v = TileTensor(
                grad_output.dev.value(), row_major[B, Self.EMBED_DIM_]()
            )
            var w_bf_v = TileTensor(
                self.w_bf.dev.value(),
                row_major[Self.VOCAB_, Self.EMBED_DIM_](),
            )
            var gi_v = TileTensor(gin.dev.value(), row_major[B, Self.VOCAB_]())
            max_matmul[transpose_b=True, target="gpu"](gi_v, go_v, w_bf_v, c)
            # gw_tmp[VOCAB, ED] = cache_inᵀ_bf[VOCAB, B] @ grad_out[B, ED]:
            # bf16-in → FP32-out GEMM into the fp32 gw_tmp.
            var cinT_v = TileTensor(
                self.cache_inT_bf.dev.value(), row_major[Self.VOCAB_, B]()
            )
            var gwtmp_v = TileTensor(
                self.gw_tmp.dev.value(),
                row_major[Self.VOCAB_, Self.EMBED_DIM_](),
            )
            max_matmul[target="gpu"](gwtmp_v, cinT_v, go_v, c)
            # weight.grad (fp32 master) += gw_tmp (fp32)
            comptime gw_blk = (Self.W_SIZE + TPB - 1) // TPB
            c.enqueue_function[_emb_accum_kernel[Self.W_SIZE]](
                self.weight.grd.lt["gpu", lwf](),
                self.gw_tmp.lt["gpu", lwf](),
                grid_dim=gw_blk,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
