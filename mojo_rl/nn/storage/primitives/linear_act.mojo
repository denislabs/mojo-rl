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

Use via one-line aliases (see linear_tanh.mojo / linear_mish.mojo / …):
    comptime LinearReLU[IN, OUT]    = LinearAct[IN, OUT, ReLUOp]
    comptime LinearTanh[IN, OUT]    = LinearAct[IN, OUT, TanhOp]
    comptime LinearSigmoid[IN, OUT] = LinearAct[IN, OUT, SigmoidOp]
    comptime LinearMish[IN, OUT]    = LinearAct[IN, OUT, MishOp]
    comptime LinearSwish[IN, OUT]   = LinearAct[IN, OUT, SwishOp]
"""

from std.sys import CompilationTarget
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from mojo_rl.nn.constants import DT, TPB, CPU_SIMD_W
from mojo_rl.nn.core.element_op import ElementOp
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..loss.sac import polyak_tensor
from .linear import (
    _lin_gb_kernel,
    _transpose_tiled_kernel,
    _accum_kernel,
    _T_TILE,
    _T_BR,
)


def _bias_act_cache_kernel[
    B: Int, OUT: Int, OP: ElementOp
](
    o: LayoutTensor[DT, Layout.row_major(B, OUT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(B, OUT), MutAnyOrigin],
):
    """Fused epilogue: o[b,j] = OP.forward(matmul[b,j] + bias[j]).

    cache[b,j] = y (post-act) if OP.owns_cache else z (pre-act).
    """
    var idx = Int(global_idx.x)
    if idx < B * OUT:
        var b = idx // OUT
        var j = idx % OUT
        var z = rebind[Scalar[DT]](o[b, j]) + rebind[Scalar[DT]](bias[j])
        var y = OP.forward_scalar(z)
        o[b, j] = y
        comptime if OP.owns_cache:
            cache[b, j] = y
        else:
            cache[b, j] = z


def _act_gate_kernel[
    B: Int, OUT: Int, OP: ElementOp
](
    go: LayoutTensor[DT, Layout.row_major(B, OUT), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(B, OUT), MutAnyOrigin],
):
    """In-place activation-derivative rewrite: go[b,j] ← OP.backward(cache, go)."""
    var idx = Int(global_idx.x)
    if idx < B * OUT:
        var b = idx // OUT
        var j = idx % OUT
        var c = rebind[Scalar[DT]](cache[b, j])
        var g = rebind[Scalar[DT]](go[b, j])
        go[b, j] = OP.backward_scalar(c, g)


struct LinearAct[IN_: Int, OUT_: Int, OP: ElementOp](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_

    @staticmethod
    def display_label() -> String:
        return String("LinearAct")
    comptime B_SIZE = Self.OUT_

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    var cache: Tensor  # [B, OUT] activation cache (z or y; owned)
    var cacheT: Tensor
    var dW_tmp: Tensor

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.cache = Tensor()
        self.cacheT = Tensor()
        self.dW_tmp = Tensor()

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
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_)
            self.cache.ensure(B * Self.OUT_)
            var x_v = TileTensor(in0.data, row_major[B, Self.IN_]())
            var w_v = TileTensor(
                self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
            )
            var out_v = TileTensor(out.data, row_major[B, Self.OUT_]())
            max_matmul[target="cpu"](out_v, x_v, w_v, None)
            # fused bias + activation + cache, SIMD over the OUT dim (flat
            # pointers — no per-element TileTensor 2D-index / List access).
            var op = out.data.unsafe_ptr()
            var bp = self.bias.val.data.unsafe_ptr()
            var cp = self.cache.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            for b in range(B):
                var row = b * Self.OUT_
                var k = 0
                while k + W <= Self.OUT_:
                    var z = op.load[width=W](row + k) + bp.load[width=W](k)
                    var y = Self.OP.forward_simd[W](z)
                    op.store(row + k, y)
                    comptime if Self.OP.owns_cache:
                        cp.store(row + k, y)
                    else:
                        cp.store(row + k, z)
                    k += W
                while k < Self.OUT_:
                    var z_s = op[row + k] + bp[k]
                    var y_s = Self.OP.forward_scalar(z_s)
                    op[row + k] = y_s
                    comptime if Self.OP.owns_cache:
                        cp[row + k] = y_s
                    else:
                        cp[row + k] = z_s
                    k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            self.cache.ensure_gpu(c, B * Self.OUT_)
            var x_v = TileTensor(in0.dev.value(), row_major[B, Self.IN_]())
            var w_v = TileTensor(
                self.weight.val.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            var out_v = TileTensor(out.dev.value(), row_major[B, Self.OUT_]())
            max_matmul[target="gpu"](out_v, x_v, w_v, c)
            c.enqueue_function[_bias_act_cache_kernel[B, Self.OUT_, Self.OP]](
                out.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.bias.val.lt["gpu", Layout.row_major(Self.OUT_)](),
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
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime M = B * Self.OUT_
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_)
            # gate grad by activation derivative: grad_z = OP.backward(cache, go)
            # (SIMD flat over BATCH*OUT).
            var gp = grad_output.data.unsafe_ptr()
            var cp = self.cache.data.unsafe_ptr()
            comptime W2 = CPU_SIMD_W
            var kk = 0
            while kk + W2 <= M:
                var c = cp.load[width=W2](kk)
                var g = gp.load[width=W2](kk)
                gp.store(kk, Self.OP.backward_simd[W2](c, g))
                kk += W2
            while kk < M:
                gp[kk] = Self.OP.backward_scalar(cp[kk], gp[kk])
                kk += 1
            var go_v = TileTensor(grad_output.data, row_major[B, Self.OUT_]())
            var gi_v = TileTensor(gin.data, row_major[B, Self.IN_]())
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
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        fin.data.unsafe_ptr()
                    ),
                    Int32(Self.IN_),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        grad_output.data.unsafe_ptr()
                    ),
                    Int32(Self.OUT_),
                    Float32(1.0),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](
                        self.weight.grd.data.unsafe_ptr()
                    ),
                    Int32(Self.OUT_),
                )
            else:
                self.cacheT.ensure(Self.IN_ * B)
                self.dW_tmp.ensure(Self.W_SIZE)
                var x_v = TileTensor(fin.data, row_major[B, Self.IN_]())
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
            gin.ensure_gpu(c, B * Self.IN_)
            self.cacheT.ensure_gpu(c, Self.IN_ * B)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
            # gate grad by activation derivative
            c.enqueue_function[_act_gate_kernel[B, Self.OUT_, Self.OP]](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.cache.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                grid_dim=(M + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[_lin_gb_kernel[B, Self.OUT_]](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.bias.grd.lt["gpu", Layout.row_major(Self.OUT_)](),
                grid_dim=(Self.OUT_ + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[_transpose_tiled_kernel[B, Self.IN_]](
                fin.lt["gpu", Layout.row_major(B, Self.IN_)](),
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
                grad_output.dev.value(), row_major[B, Self.OUT_]()
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
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var w_v = TileTensor(
                self.weight.val.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            var gi_v = TileTensor(gin.dev.value(), row_major[B, Self.IN_]())
            max_matmul[transpose_b=True, target="gpu"](gi_v, go_v2, w_v, c)

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
