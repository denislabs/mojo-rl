"""SkipConcat[Inner] — y = concat(x, inner(x)) (storage surface, CPU + GPU).

DenseNet/U-Net-style skip-concat: the output is the input column-stacked with the
inner module's output. Mirrors `Residual` structurally (one child) but the merge
is concat instead of add.

  IN_DIMS[0] = Inner.IN_DIMS[0]                  (passthrough)
  OUT_DIM    = Inner.IN_DIMS[0] + Inner.OUT_DIM  (input ++ inner output)

Forward:  out = [x | inner(x)]
Backward: grad_input = grad_output[:, :IN] + inner.vjp(grad_output[:, IN:])
(the input feeds both the passthrough column block and inner — its grads sum).
Reuses Parallel's concat/split kernels + Residual's add kernel.
"""

from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from .parallel import _par_concat_kernel, _par_split_kernel
from .residual import _resid_add_kernel


struct SkipConcat[Inner: Module](Module):
    comptime ARITY = 1
    comptime IN = Self.Inner.IN_DIMS[0]
    comptime OINNER = Self.Inner.OUT_DIM
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.IN_DIMS[0] + Self.Inner.OUT_DIM

    var inner: Self.Inner
    var inner_out: Tensor
    var go_pass: Tensor  # grad_output[:, :IN]   (bwd)
    var go_inner: Tensor  # grad_output[:, IN:]   (bwd, fed to inner)
    var gi_inner: Tensor  # inner's grad-input

    def __init__(out self):
        self.inner = Self.Inner()
        self.inner_out = Tensor()
        self.go_pass = Tensor()
        self.go_inner = Tensor()
        self.gi_inner = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.inner = Self.Inner.make[target, INIT](ctx)
        return s^

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        self.inner.forward[target, B](
            TensorRefs[Self.Inner.ARITY](in0), self.inner_out, ctx
        )
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            for b in range(B):
                var ob = b * Self.OUT_DIM
                for j in range(Self.IN):
                    out.data[ob + j] = in0.data[b * Self.IN + j]
                for j in range(Self.OINNER):
                    out.data[ob + Self.IN + j] = self.inner_out.data[
                        b * Self.OINNER + j
                    ]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            c.enqueue_function[_par_concat_kernel[B, Self.IN, Self.OINNER]](
                in0.lt["gpu", Layout.row_major(B, Self.IN)](),
                self.inner_out.lt["gpu", Layout.row_major(B, Self.OINNER)](),
                out.lt["gpu", Layout.row_major(B, Self.OUT_DIM)](),
                grid_dim=(B * Self.OUT_DIM + TPB - 1) // TPB,
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
        # split grad_output → go_pass [:IN], go_inner [IN:]
        comptime if target == "cpu":
            self.go_pass.ensure(B * Self.IN)
            self.go_inner.ensure(B * Self.OINNER)
            for b in range(B):
                var gb = b * Self.OUT_DIM
                for j in range(Self.IN):
                    self.go_pass.data[b * Self.IN + j] = grad_output.data[
                        gb + j
                    ]
                for j in range(Self.OINNER):
                    self.go_inner.data[b * Self.OINNER + j] = grad_output.data[
                        gb + Self.IN + j
                    ]
        else:
            var c = ctx.value()
            self.go_pass.ensure_gpu(c, B * Self.IN)
            self.go_inner.ensure_gpu(c, B * Self.OINNER)
            c.enqueue_function[_par_split_kernel[B, Self.IN, Self.OINNER]](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_DIM)](),
                self.go_pass.lt["gpu", Layout.row_major(B, Self.IN)](),
                self.go_inner.lt["gpu", Layout.row_major(B, Self.OINNER)](),
                grid_dim=(B * Self.OUT_DIM + TPB - 1) // TPB,
                block_dim=TPB,
            )
        self.inner.vjp[target, B](
            TensorRefs[Self.Inner.ARITY](fin),
            self.go_inner,
            TensorRefs[Self.Inner.ARITY](self.gi_inner),
            ctx,
        )
        # grad_input = go_pass + gi_inner
        comptime NIN = B * Self.IN
        comptime if target == "cpu":
            gin.ensure(NIN)
            var gp = gin.data.unsafe_ptr()
            var pp = self.go_pass.data.unsafe_ptr()
            var ip = self.gi_inner.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= NIN:
                gp.store(k, pp.load[width=W](k) + ip.load[width=W](k))
                k += W
            while k < NIN:
                gp[k] = pp[k] + ip[k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, NIN)
            c.enqueue_function[_resid_add_kernel[NIN]](
                self.go_pass.lt["gpu", Layout.row_major(NIN)](),
                self.gi_inner.lt["gpu", Layout.row_major(NIN)](),
                gin.lt["gpu", Layout.row_major(NIN)](),
                grid_dim=(NIN + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_param[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_state[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.inner.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.inner.polyak_from[target](src.inner, tau, ctx)
