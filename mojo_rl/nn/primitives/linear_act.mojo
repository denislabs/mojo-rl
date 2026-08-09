"""LinearAct[IN, OUT, OP] — fused Linear + bias + activation (storage surface).

The GENERAL storage twin of legacy `LinearAct` (the parametric fused leaf;
`LinearReLU` storage is the ReLU specialization). y = OP.forward(x@W + b) with
the bias-add + activation + cache write fused into ONE epilogue kernel on
forward, and the activation-derivative rewrite of grad_output into ONE kernel on
backward — then the SAME Linear backward (Apple-fp32 cblas beta=1 dW /
transpose+accum) as `LinearReLU`. 1 node instead of 2.

Cache convention (carried VERBATIM from legacy linear_act):
  - `OP.owns_cache = False` (ReLU, Mish, …) → cache[i] = z (pre-activation =
    matmul + bias). Backward reads it as `c` and gates: grad_z = OP.backward(z, go).
  - `OP.owns_cache = True` (Tanh, Sigmoid, …) → cache[i] = y (post-activation).
    Backward reads it as `c`: grad_z = OP.backward(y, go).
The cache is an owned Tensor (storage-clean), mirroring `LinearReLU.mask`.

bf16-FLOW (AMP "Step B"): `LinearAct[IN, OUT, OP]` is fp32 (unchanged), while
`LinearAct[IN, OUT, OP, DType.bfloat16]` flows ACTIVATIONS at bf16 (`ACT_DT ==
bfloat16`) — mirrors `Linear`/`LinearReLU` bf16-flow exactly. Master weights/
grads/bias stay fp32 (`Param`); only the CACHED bf16 weight (`w_bf`, version-
gated), the cached bf16 bias (`b_a`), the activation cache (`cache`) and the
transposed bf16 fwd-input (`cacheT_bf`) are low-precision. The fused epilogue
(`_bias_act_cache_kernel`) and the backward act-gate (`_act_gate_kernel`) are
dtype-parametric (`ADT`); the `ElementOp` math itself is fp32-only (its API takes
`Scalar[DT]`), so the bf16 kernels cast the cached/grad bf16 values UP to fp32,
call `OP.forward_scalar`/`OP.backward_scalar`, then cast the result back DOWN to
bf16 for storage — elementwise activations (ReLU/Tanh/GELU/…) are numerically
fine across that round-trip. bf16-flow is GPU-only (cblas/CPU matmul is
fp32-only). The fp32 (ACT_DT == DT) path is byte-for-byte the legacy NoAMP path.

Use via one-line aliases (see linear_tanh.mojo / linear_mish.mojo / …):
    comptime LinearReLU[IN, OUT]    = LinearAct[IN, OUT, ReLUOp]
    comptime LinearTanh[IN, OUT]    = LinearAct[IN, OUT, TanhOp]
    comptime LinearSigmoid[IN, OUT] = LinearAct[IN, OUT, SigmoidOp]
    comptime LinearMish[IN, OUT]    = LinearAct[IN, OUT, MishOp]
    comptime LinearSwish[IN, OUT]   = LinearAct[IN, OUT, SwishOp]
"""

from std.sys import CompilationTarget
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from mojo_rl.nn.constants import DT, TPB, CPU_SIMD_W
from mojo_rl.nn.core.element_op import ElementOp
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor
from .linear import (
    _lin_gb_kernel,
    _transpose_tiled_kernel,
    _accum_kernel,
    _cast_f2b_kernel,
    _T_TILE,
    _T_BR,
)


def _bias_act_cache_kernel[
    B: Int, OUT: Int, OP: ElementOp, ADT: DType = DT
](
    o: LayoutTensor[ADT, Layout.row_major(B, OUT), MutAnyOrigin],
    bias: LayoutTensor[ADT, Layout.row_major(OUT), MutAnyOrigin],
    cache: LayoutTensor[ADT, Layout.row_major(B, OUT), MutAnyOrigin],
):
    """Fused epilogue: o[b,j] = OP.forward(matmul[b,j] + bias[j]).

    cache[b,j] = y (post-act) if OP.owns_cache else z (pre-act).

    Dtype-parametric (`ADT`): the fp32 path runs entirely at DT; the bf16-flow
    path holds o/bias/cache at bfloat16 but casts z UP to fp32 for `OP.forward`
    (the ElementOp API is fp32-only), then casts the result back DOWN to bf16.
    """
    var idx = Int(global_idx.x)
    if idx < B * OUT:
        var b = idx // OUT
        var j = idx % OUT
        var z = rebind[Scalar[ADT]](o[b, j]) + rebind[Scalar[ADT]](bias[j])
        comptime if ADT == DT:
            var y = OP.forward_scalar(rebind[Scalar[DT]](z))
            o[b, j] = rebind[Scalar[ADT]](y)
            comptime if OP.owns_cache:
                cache[b, j] = rebind[Scalar[ADT]](y)
            else:
                cache[b, j] = rebind[Scalar[ADT]](z)
        else:
            # bf16-flow: lift to fp32 for the activation math, store back at bf16.
            var y_f = OP.forward_scalar(z.cast[DT]())
            o[b, j] = y_f.cast[ADT]()
            comptime if OP.owns_cache:
                cache[b, j] = y_f.cast[ADT]()
            else:
                cache[b, j] = z  # z already ADT


def _act_gate_kernel[
    B: Int, OUT: Int, OP: ElementOp, ADT: DType = DT
](
    go: LayoutTensor[ADT, Layout.row_major(B, OUT), MutAnyOrigin],
    cache: LayoutTensor[ADT, Layout.row_major(B, OUT), MutAnyOrigin],
):
    """In-place activation-derivative rewrite: go[b,j] ← OP.backward(cache, go).

    Dtype-parametric (`ADT`): the bf16-flow path casts the cached value + grad UP
    to fp32 for `OP.backward` (fp32-only API), then casts the gated grad back DOWN
    to bf16 (it stays a bf16 activation flowing into the grad_w/grad_x GEMMs).
    """
    var idx = Int(global_idx.x)
    if idx < B * OUT:
        var b = idx // OUT
        var j = idx % OUT
        var c = rebind[Scalar[ADT]](cache[b, j])
        var g = rebind[Scalar[ADT]](go[b, j])
        comptime if ADT == DT:
            go[b, j] = rebind[Scalar[ADT]](
                OP.backward_scalar(
                    rebind[Scalar[DT]](c), rebind[Scalar[DT]](g)
                )
            )
        else:
            go[b, j] = OP.backward_scalar(c.cast[DT](), g.cast[DT]()).cast[ADT]()


struct LinearAct[IN_: Int, OUT_: Int, OP: ElementOp, ADT: DType = DT](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_
    # Activation-flow dtype. `LinearAct[IN, OUT, OP]` = fp32 (ACT_DT == DT, the
    # legacy path); `LinearAct[IN, OUT, OP, bfloat16]` flows activations at bf16.
    comptime ACT_DT = Self.ADT

    @staticmethod
    def display_label() -> String:
        return String("LinearAct")
    comptime B_SIZE = Self.OUT_

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    # [B, OUT] activation cache (z or y; owned). In bf16-flow it is an ACTIVATION
    # → stored at the flow dtype `Self.ADT` (bf16).
    var cache: TensorImpl[Self.ADT]
    var cacheT: Tensor
    var dW_tmp: Tensor
    # bf16-flow compute scratch (lazy; ACT_DT == bf16 && target == "gpu" only).
    # Master weights/grads/bias stay fp32 (`Param`). `w_bf` = CACHED bf16 weight
    # (recast only on a `weight.val.version` bump → ONCE per optimizer step, see
    # `_ensure_w_bf`). `b_a` = cached bf16 bias (per-forward cast). `cacheT_bf` =
    # transposed bf16 fwd-input (backward grad_w).
    var w_bf: TensorImpl[Self.ADT]
    var b_a: TensorImpl[Self.ADT]
    var cacheT_bf: TensorImpl[Self.ADT]
    var _w_cast_version: Int  # `weight.val.version` at last bf16 weight cast
    # Capture mode (set via `set_attr["capture_recast"]`): when True, the bf16
    # weight recast is UNCONDITIONAL so the cast kernel is always recorded into a
    # CUDA graph and reads the live fp32 master on every replay — the version
    # gate would skip it on replay and serve STALE weights. Off → version-gated.
    var _force_recast: Bool

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.cache = TensorImpl[Self.ADT]()
        self.cacheT = Tensor()
        self.dW_tmp = Tensor()
        self.w_bf = TensorImpl[Self.ADT]()
        self.b_a = TensorImpl[Self.ADT]()
        self.cacheT_bf = TensorImpl[Self.ADT]()
        self._w_cast_version = -1
        self._force_recast = False

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "capture_recast":
            self._force_recast = value != Scalar[DT](0.0)

    def _ensure_w_bf(mut self, c: DeviceContext) raises:
        """Ensure the cached bf16 weight `w_bf` reflects the current fp32
        `weight.val` — recasts ONLY on a `val.version` bump (once per optimizer
        step, not per fwd/bwd). Mirror of `Linear._ensure_w_bf`."""
        self.w_bf.ensure_gpu(c, Self.W_SIZE)
        if self._force_recast or self.weight.val.version != self._w_cast_version:
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
        var l = Self()
        l.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        l.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        INIT.init_weight[target](l.weight.val, Self.W_SIZE, Self.IN_, Self.OUT_, ctx)
        INIT.init_bias[target](l.bias.val, Self.B_SIZE, ctx)
        return l^

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
            # ACT_DT IS DT here, but the checker won't collapse the opaque
            # `Self.ACT_DT` to `DT` for unification vs the fp32 weight/bias views
            # — so rebind the activation refs (sound; dtypes are equal here).
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            ref cached = rebind[Tensor](self.cache)
            comptime if target == "cpu":
                outd.ensure(B * Self.OUT_)
                cached.ensure(B * Self.OUT_)
                var x_v = TileTensor(in0d.data, row_major[B, Self.IN_]())
                var w_v = TileTensor(
                    self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
                )
                var out_v = TileTensor(outd.data, row_major[B, Self.OUT_]())
                max_matmul[target="cpu"](out_v, x_v, w_v, None)
                # fused bias + activation + cache, SIMD over the OUT dim (flat
                # pointers — no per-element TileTensor 2D-index / List access).
                var op = outd.data.unsafe_ptr()
                var bp = self.bias.val.data.unsafe_ptr()
                var cp = cached.data.unsafe_ptr()
                comptime W = CPU_SIMD_W
                for b in range(B):
                    var row = b * Self.OUT_
                    var k = 0
                    while k + W <= Self.OUT_:
                        var z = op.unsafe_load[width=W](row + k) + bp.unsafe_load[width=W](k)
                        var y = Self.OP.forward_simd[W](z)
                        op.unsafe_store(row + k, y)
                        comptime if Self.OP.owns_cache:
                            cp.unsafe_store(row + k, y)
                        else:
                            cp.unsafe_store(row + k, z)
                        k += W
                    while k < Self.OUT_:
                        var z_s = op[unsafe_offset=row + k] + bp[unsafe_offset=k]
                        var y_s = Self.OP.forward_scalar(z_s)
                        op[unsafe_offset=row + k] = y_s
                        comptime if Self.OP.owns_cache:
                            cp[unsafe_offset=row + k] = y_s
                        else:
                            cp[unsafe_offset=row + k] = z_s
                        k += 1
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, B * Self.OUT_)
                cached.ensure_gpu(c, B * Self.OUT_)
                var x_v = TileTensor(in0d.dev.value(), row_major[B, Self.IN_]())
                var w_v = TileTensor(
                    self.weight.val.dev.value(), row_major[Self.IN_, Self.OUT_]()
                )
                var out_v = TileTensor(outd.dev.value(), row_major[B, Self.OUT_]())
                max_matmul[target="gpu"](out_v, x_v, w_v, c)
                c.enqueue_function[_bias_act_cache_kernel[B, Self.OUT_, Self.OP]](
                    outd.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                    self.bias.val.lt["gpu", Layout.row_major(Self.OUT_)](),
                    cached.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                    grid_dim=(B * Self.OUT_ + TPB - 1) // TPB,
                    block_dim=TPB,
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow LinearAct is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            self.cache.ensure_gpu(c, B * Self.OUT_)
            # x (in0) is ALREADY bf16 — no input cast. W: cached bf16 (recast only
            # on a version bump). bias: cheap per-forward DT→bf16 cast.
            self._ensure_w_bf(c)
            self.b_a.ensure_gpu(c, Self.B_SIZE)
            c.enqueue_function[_cast_f2b_kernel[Self.B_SIZE]](
                self.bias.val.lt["gpu", Layout.row_major(Self.B_SIZE)](),
                self.b_a.lt["gpu", Layout.row_major(Self.B_SIZE)](),
                grid_dim=(Self.B_SIZE + 255) // 256,
                block_dim=256,
            )
            var x_v = TileTensor(in0.dev.value(), row_major[B, Self.IN_]())
            var w_bf_v = TileTensor(
                self.w_bf.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            var out_v = TileTensor(out.dev.value(), row_major[B, Self.OUT_]())
            # bf16-in → bf16-out GEMM (fp32 accumulation is automatic).
            max_matmul[target="gpu"](out_v, x_v, w_bf_v, c)
            c.enqueue_function[
                _bias_act_cache_kernel[B, Self.OUT_, Self.OP, Self.ADT]
            ](
                out.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.b_a.lt["gpu", Layout.row_major(Self.OUT_)](),
                self.cache.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                grid_dim=(B * Self.OUT_ + TPB - 1) // TPB,
                block_dim=TPB,
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
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime M = B * Self.OUT_
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            ref find = rebind[Tensor](fin)
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            ref cached = rebind[Tensor](self.cache)
            comptime if target == "cpu":
                gind.ensure(B * Self.IN_)
                # gate grad by activation derivative: grad_z = OP.backward(cache,
                # go) (SIMD flat over BATCH*OUT).
                var gp = god.data.unsafe_ptr()
                var cp = cached.data.unsafe_ptr()
                comptime W2 = CPU_SIMD_W
                var kk = 0
                while kk + W2 <= M:
                    var c = cp.unsafe_load[width=W2](kk)
                    var g = gp.unsafe_load[width=W2](kk)
                    gp.unsafe_store(kk, Self.OP.backward_simd[W2](c, g))
                    kk += W2
                while kk < M:
                    gp[unsafe_offset=kk] = Self.OP.backward_scalar(cp[unsafe_offset=kk], gp[unsafe_offset=kk])
                    kk += 1
                var go_v = TileTensor(god.data, row_major[B, Self.OUT_]())
                var gi_v = TileTensor(gind.data, row_major[B, Self.IN_]())
                var w_v = TileTensor(
                    self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
                )
                var gb_v = TileTensor(self.bias.grd.data, row_major[Self.OUT_]())
                for b in range(B):
                    for j in range(Self.OUT_):
                        gb_v[j] += go_v[b, j]
                comptime IS_APPLE_F32 = CompilationTarget.is_macos() and DT == DType.float32
                comptime if IS_APPLE_F32:
                    var cblas = get_cblas_f32_function()
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.IN_),
                        Int32(Self.OUT_),
                        Int32(B),
                        Float32(1.0),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            find.data.unsafe_ptr()
                        ),
                        Int32(Self.IN_),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            god.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),
                        Float32(1.0),
                        rebind[Pointer[Float32, MutAnyOrigin]](
                            self.weight.grd.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),
                    )
                else:
                    self.cacheT.ensure(Self.IN_ * B)
                    self.dW_tmp.ensure(Self.W_SIZE)
                    var x_v = TileTensor(find.data, row_major[B, Self.IN_]())
                    var cT_v = TileTensor(
                        self.cacheT.data, row_major[Self.IN_, B]()
                    )
                    for b in range(B):
                        for k in range(Self.IN_):
                            cT_v[k, b] = x_v[b, k]
                    var dW_v = TileTensor(
                        self.dW_tmp.data, row_major[Self.IN_, Self.OUT_]()
                    )
                    max_matmul[target="cpu"](dW_v, cT_v, go_v, None)
                    var gw_v = TileTensor(
                        self.weight.grd.data, row_major[Self.IN_, Self.OUT_]()
                    )
                    for k in range(Self.IN_):
                        for j in range(Self.OUT_):
                            gw_v[k, j] += dW_v[k, j]
                max_matmul[transpose_b=True, target="cpu"](gi_v, go_v, w_v, None)
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, B * Self.IN_)
                self.cacheT.ensure_gpu(c, Self.IN_ * B)
                self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
                # gate grad by activation derivative
                c.enqueue_function[_act_gate_kernel[B, Self.OUT_, Self.OP]](
                    god.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                    cached.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                    grid_dim=(M + TPB - 1) // TPB,
                    block_dim=TPB,
                )
                c.enqueue_function[_lin_gb_kernel[B, Self.OUT_]](
                    god.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                    self.bias.grd.lt["gpu", Layout.row_major(Self.OUT_)](),
                    grid_dim=(Self.OUT_ + TPB - 1) // TPB,
                    block_dim=TPB,
                )
                c.enqueue_function[_transpose_tiled_kernel[B, Self.IN_]](
                    find.lt["gpu", Layout.row_major(B, Self.IN_)](),
                    self.cacheT.lt["gpu", Layout.row_major(Self.IN_, B)](),
                    grid_dim=(
                        (Self.IN_ + _T_TILE - 1) // _T_TILE,
                        (B + _T_TILE - 1) // _T_TILE,
                    ),
                    block_dim=(_T_TILE, _T_BR),
                )
                var cT_v = TileTensor(
                    self.cacheT.dev.value(), row_major[Self.IN_, B]()
                )
                var go_v = TileTensor(
                    god.dev.value(), row_major[B, Self.OUT_]()
                )
                var dW_v = TileTensor(
                    self.dW_tmp.dev.value(), row_major[Self.IN_, Self.OUT_]()
                )
                max_matmul[target="gpu"](dW_v, cT_v, go_v, c)
                c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                    self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                    self.dW_tmp.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                    grid_dim=(Self.W_SIZE + TPB - 1) // TPB,
                    block_dim=TPB,
                )
                var go_v2 = TileTensor(
                    god.dev.value(), row_major[B, Self.OUT_]()
                )
                var w_v = TileTensor(
                    self.weight.val.dev.value(), row_major[Self.IN_, Self.OUT_]()
                )
                var gi_v = TileTensor(gind.dev.value(), row_major[B, Self.IN_]())
                max_matmul[transpose_b=True, target="gpu"](gi_v, go_v2, w_v, c)
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow LinearAct is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_)
            self.cacheT_bf.ensure_gpu(c, Self.IN_ * B)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
            # gate the bf16 grad by the bf16 activation cache (cast UP to fp32 for
            # OP.backward, back DOWN to bf16 — the gated grad stays bf16).
            c.enqueue_function[
                _act_gate_kernel[B, Self.OUT_, Self.OP, Self.ADT]
            ](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.cache.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                grid_dim=(M + TPB - 1) // TPB,
                block_dim=TPB,
            )
            # grad_b += colsum(go): bf16 go → fp32 master grad (fp32 accumulator).
            c.enqueue_function[_lin_gb_kernel[B, Self.OUT_, Self.ADT]](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.bias.grd.lt["gpu", Layout.row_major(Self.OUT_)](),
                grid_dim=(Self.OUT_ + TPB - 1) // TPB,
                block_dim=TPB,
            )
            # grad_w += cacheᵀ @ go: transpose the bf16 fwd-input → bf16 cacheT_bf
            # (B1' tiled), then a bf16-in → FP32-out GEMM into fp32 dW_tmp, then
            # accumulate into the fp32 master grad. W reuses the forward's cast.
            self._ensure_w_bf(c)
            c.enqueue_function[_transpose_tiled_kernel[B, Self.IN_, Self.ADT]](
                fin.lt["gpu", Layout.row_major(B, Self.IN_)](),
                self.cacheT_bf.lt["gpu", Layout.row_major(Self.IN_, B)](),
                grid_dim=(
                    (Self.IN_ + _T_TILE - 1) // _T_TILE,
                    (B + _T_TILE - 1) // _T_TILE,
                ),
                block_dim=(_T_TILE, _T_BR),
            )
            var cTb_v = TileTensor(
                self.cacheT_bf.dev.value(), row_major[Self.IN_, B]()
            )
            var gob_v = TileTensor(
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var dW_v = TileTensor(
                self.dW_tmp.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            # grad_w = cacheT_bf @ go → fp32 dW (bf16-in, fp32-out).
            max_matmul[target="gpu"](dW_v, cTb_v, gob_v, c)
            c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.dW_tmp.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                grid_dim=(Self.W_SIZE + TPB - 1) // TPB,
                block_dim=TPB,
            )
            # grad_x = go @ Wᵀ → bf16 gin (bf16-in, bf16-out — gin flows at bf16).
            var go_v2 = TileTensor(
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var wb_v = TileTensor(
                self.w_bf.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            var gi_v = TileTensor(gin.dev.value(), row_major[B, Self.IN_]())
            max_matmul[transpose_b=True, target="gpu"](gi_v, go_v2, wb_v, c)

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `weight` + `bias` Params).

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.W_SIZE](
            self.weight.val, src.weight.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.bias.val, src.bias.val, tau, ctx
        )
