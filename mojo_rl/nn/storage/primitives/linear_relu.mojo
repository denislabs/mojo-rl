"""LinearReLU[IN, OUT] — fused Linear + ReLU (storage surface).

The storage twin of legacy `LinearReLU` (= LinearAct[…, ReLUOp]). y = relu(x@W +
b) with the bias-add + ReLU + relu-mask fused into ONE epilogue kernel (vs the
unfused `Sequential[Linear, ReLU]` = matmul + bias kernel + separate ReLU + an
extra intermediate). Backward gates grad_output by the cached relu mask, then
runs the SAME Linear backward (Apple-fp32 cblas beta=1 dW). 1 node instead of 2.

Mask convention: cache[i] = 1 if pre-activation z_i > 0 else 0; grad_z = grad_y ⊙
mask. The mask is an owned Tensor (storage-clean cache).
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
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..loss.sac import polyak_tensor
from .linear import _lin_gb_kernel, _transpose_kernel, _accum_kernel


def _bias_relu_mask_kernel[
    B: Int, OUT: Int
](
    o: LayoutTensor[DT, Layout.row_major(B, OUT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(B, OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < B * OUT:
        var b = idx // OUT
        var j = idx % OUT
        var z = rebind[Scalar[DT]](o[b, j]) + rebind[Scalar[DT]](bias[j])
        mask[b, j] = Scalar[DT](1.0) if z > 0 else Scalar[DT](0.0)
        o[b, j] = z if z > 0 else Scalar[DT](0.0)


def _gate_kernel[
    N: Int
](
    go: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        go[i] = rebind[Scalar[DT]](go[i]) * rebind[Scalar[DT]](mask[i])


struct LinearReLU[IN_: Int, OUT_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_
    comptime B_SIZE = Self.OUT_

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    var mask: Tensor  # [B, OUT] relu mask (owned cache)
    var cacheT: Tensor
    var dW_tmp: Tensor

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.mask = Tensor()
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
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_)
            self.mask.ensure(B * Self.OUT_)
            var x_v = TileTensor(in0.data, row_major[B, Self.IN_]())
            var w_v = TileTensor(
                self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
            )
            var out_v = TileTensor(out.data, row_major[B, Self.OUT_]())
            max_matmul[target="cpu"](out_v, x_v, w_v, None)
            # fused bias + ReLU + mask, SIMD over the OUT dim (flat pointers —
            # no per-element TileTensor 2D-index / bounds-checked List access).
            var op = out.data.unsafe_ptr()
            var bp = self.bias.val.data.unsafe_ptr()
            var mp = self.mask.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var zero = SIMD[DT, W](0)
            var one = SIMD[DT, W](1)
            for b in range(B):
                var row = b * Self.OUT_
                var k = 0
                while k + W <= Self.OUT_:
                    var z = op.load[width=W](row + k) + bp.load[width=W](k)
                    var pos = z.gt(zero)
                    mp.store(row + k, pos.select(one, zero))
                    op.store(row + k, pos.select(z, zero))
                    k += W
                while k < Self.OUT_:
                    var z = op[row + k] + bp[k]
                    mp[row + k] = Scalar[DT](1.0) if z > 0 else Scalar[DT](0.0)
                    op[row + k] = z if z > 0 else Scalar[DT](0.0)
                    k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            self.mask.ensure_gpu(c, B * Self.OUT_)
            var x_v = TileTensor(in0.dev.value(), row_major[B, Self.IN_]())
            var w_v = TileTensor(
                self.weight.val.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            var out_v = TileTensor(out.dev.value(), row_major[B, Self.OUT_]())
            max_matmul[target="gpu"](out_v, x_v, w_v, c)
            c.enqueue_function[_bias_relu_mask_kernel[B, Self.OUT_]](
                out.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.bias.val.lt["gpu", Layout.row_major(Self.OUT_)](),
                self.mask.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                grid_dim=(B * Self.OUT_ + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
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
            # gate grad by relu mask: grad_z = grad_y ⊙ mask (SIMD flat).
            var gp = grad_output.data.unsafe_ptr()
            var mp2 = self.mask.data.unsafe_ptr()
            comptime W2 = CPU_SIMD_W
            var kk = 0
            while kk + W2 <= M:
                gp.store(kk, gp.load[width=W2](kk) * mp2.load[width=W2](kk))
                kk += W2
            while kk < M:
                gp[kk] *= mp2[kk]
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
            # gate grad by mask
            c.enqueue_function[_gate_kernel[M]](
                grad_output.lt["gpu", Layout.row_major(M)](),
                self.mask.lt["gpu", Layout.row_major(M)](),
                grid_dim=(M + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[_lin_gb_kernel[B, Self.OUT_]](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.bias.grd.lt["gpu", Layout.row_major(Self.OUT_)](),
                grid_dim=(Self.OUT_ + TPB - 1) // TPB,
                block_dim=TPB,
            )
            c.enqueue_function[_transpose_kernel[B, Self.IN_]](
                fin.lt["gpu", Layout.row_major(B, Self.IN_)](),
                self.cacheT.lt["gpu", Layout.row_major(Self.IN_, B)](),
                grid_dim=(B * Self.IN_ + TPB - 1) // TPB,
                block_dim=TPB,
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

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        self.weight.visit_with[target](visitor, ctx)
        self.bias.visit_with[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.weight.zero_grad[target](ctx)
        self.bias.zero_grad[target](ctx)
