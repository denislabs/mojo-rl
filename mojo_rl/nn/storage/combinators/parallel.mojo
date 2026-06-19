"""Parallel[A, B] — 2-branch column-concat (storage-passing, CPU + GPU).

The storage twin of legacy `nn.combinators.Parallel`. A and B share the input
dim; outputs are concatenated along columns:

  forward:  out        = [A(x) | B(x)]
  backward: grad_input = A.vjp(go[:, :OUT_A]) + B.vjp(go[:, OUT_A:])

Owns four scratch Tensors: out_a/out_b (branch outputs on fwd, reused as the
grad-output halves on bwd) and gi_a/gi_b (each branch's grad-input, summed into
grad_input). The actor's `[mu | log_std]` heads are a `Parallel[Linear, Linear]`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB, CPU_SIMD_W
from mojo_rl.nn.core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from .residual import _resid_add_kernel


def _par_concat_kernel[B: Int, OA: Int, OB: Int](
    a: LayoutTensor[DT, Layout.row_major(B, OA), MutAnyOrigin],
    bb: LayoutTensor[DT, Layout.row_major(B, OB), MutAnyOrigin],
    packed: LayoutTensor[DT, Layout.row_major(B, OA + OB), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = B * (OA + OB)
    if idx < total:
        var bi = idx // (OA + OB)
        var ji = idx % (OA + OB)
        if ji < OA:
            packed[bi, ji] = rebind[Scalar[DT]](a[bi, ji])
        else:
            packed[bi, ji] = rebind[Scalar[DT]](bb[bi, ji - OA])


def _par_split_kernel[B: Int, OA: Int, OB: Int](
    packed: LayoutTensor[DT, Layout.row_major(B, OA + OB), MutAnyOrigin],
    a: LayoutTensor[DT, Layout.row_major(B, OA), MutAnyOrigin],
    bb: LayoutTensor[DT, Layout.row_major(B, OB), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = B * (OA + OB)
    if idx < total:
        var bi = idx // (OA + OB)
        var ji = idx % (OA + OB)
        if ji < OA:
            a[bi, ji] = rebind[Scalar[DT]](packed[bi, ji])
        else:
            bb[bi, ji - OA] = rebind[Scalar[DT]](packed[bi, ji])


struct Parallel[A: Module, B: Module](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.A.IN_DIMS[0])
    comptime OUT_DIM = Self.A.OUT_DIM + Self.B.OUT_DIM
    comptime OUT_A = Self.A.OUT_DIM
    comptime OUT_B = Self.B.OUT_DIM
    comptime IN = Self.A.IN_DIMS[0]

    var branch_a: Self.A
    var branch_b: Self.B
    var out_a: Tensor
    var out_b: Tensor
    var gi_a: Tensor
    var gi_b: Tensor

    def __init__(out self):
        comptime assert (
            Self.A.IN_DIMS[0] == Self.B.IN_DIMS[0]
        ), "Parallel requires A.IN_DIMS[0] == B.IN_DIMS[0]"
        self.branch_a = Self.A()
        self.branch_b = Self.B()
        self.out_a = Tensor()
        self.out_b = Tensor()
        self.gi_a = Tensor()
        self.gi_b = Tensor()

    @staticmethod
    def make_cpu() raises -> Self:
        var p = Self()
        p.branch_a = Self.A.make_cpu()
        p.branch_b = Self.B.make_cpu()
        return p^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        var p = Self()
        p.branch_a = Self.A.make_gpu(ctx)
        p.branch_b = Self.B.make_gpu(ctx)
        return p^

    def forward[
        target: StaticString, BB: Int, o: MutOrigin
    ](
        mut self, inputs: TensorRefs[1, o], mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        self.branch_a.forward[target, BB](
            TensorRefs[Self.A.ARITY].of1(in0), self.out_a, ctx
        )
        self.branch_b.forward[target, BB](
            TensorRefs[Self.B.ARITY].of1(in0), self.out_b, ctx
        )
        comptime if target == "cpu":
            out.ensure(BB * Self.OUT_DIM)
            for b in range(BB):
                var ob = b * Self.OUT_DIM
                for j in range(Self.OUT_A):
                    out.data[ob + j] = self.out_a.data[b * Self.OUT_A + j]
                for j in range(Self.OUT_B):
                    out.data[ob + Self.OUT_A + j] = self.out_b.data[b * Self.OUT_B + j]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, BB * Self.OUT_DIM)
            c.enqueue_function[_par_concat_kernel[BB, Self.OUT_A, Self.OUT_B]](
                self.out_a.lt_gpu[Layout.row_major(BB, Self.OUT_A)](),
                self.out_b.lt_gpu[Layout.row_major(BB, Self.OUT_B)](),
                out.lt_gpu[Layout.row_major(BB, Self.OUT_DIM)](),
                grid_dim=(BB * Self.OUT_DIM + TPB - 1) // TPB, block_dim=TPB,
            )

    def vjp[
        target: StaticString, BB: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self, forward_input: TensorRefs[1, ofi], mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi], ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        # split grad_output → out_a / out_b (reused as the go halves)
        comptime if target == "cpu":
            self.out_a.ensure(BB * Self.OUT_A)
            self.out_b.ensure(BB * Self.OUT_B)
            for b in range(BB):
                var gb = b * Self.OUT_DIM
                for j in range(Self.OUT_A):
                    self.out_a.data[b * Self.OUT_A + j] = grad_output.data[gb + j]
                for j in range(Self.OUT_B):
                    self.out_b.data[b * Self.OUT_B + j] = grad_output.data[gb + Self.OUT_A + j]
        else:
            var c = ctx.value()
            self.out_a.ensure_gpu(c, BB * Self.OUT_A)
            self.out_b.ensure_gpu(c, BB * Self.OUT_B)
            c.enqueue_function[_par_split_kernel[BB, Self.OUT_A, Self.OUT_B]](
                grad_output.lt_gpu[Layout.row_major(BB, Self.OUT_DIM)](),
                self.out_a.lt_gpu[Layout.row_major(BB, Self.OUT_A)](),
                self.out_b.lt_gpu[Layout.row_major(BB, Self.OUT_B)](),
                grid_dim=(BB * Self.OUT_DIM + TPB - 1) // TPB, block_dim=TPB,
            )
        # branch backward into gi_a / gi_b
        self.branch_a.vjp[target, BB](
            TensorRefs[Self.A.ARITY].of1(fin), self.out_a,
            TensorRefs[Self.A.ARITY].of1(self.gi_a), ctx,
        )
        self.branch_b.vjp[target, BB](
            TensorRefs[Self.B.ARITY].of1(fin), self.out_b,
            TensorRefs[Self.B.ARITY].of1(self.gi_b), ctx,
        )
        # grad_input = gi_a + gi_b
        comptime NIN = BB * Self.IN
        comptime if target == "cpu":
            gin.ensure(NIN)
            var gp = gin.data.unsafe_ptr()
            var ap = self.gi_a.data.unsafe_ptr()
            var bp = self.gi_b.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var k = 0
            while k + W <= NIN:
                gp.store(k, ap.load[width=W](k) + bp.load[width=W](k))
                k += W
            while k < NIN:
                gp[k] = ap[k] + bp[k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, NIN)
            c.enqueue_function[_resid_add_kernel[NIN]](
                self.gi_a.lt_gpu[Layout.row_major(NIN)](),
                self.gi_b.lt_gpu[Layout.row_major(NIN)](),
                gin.lt_gpu[Layout.row_major(NIN)](),
                grid_dim=(NIN + TPB - 1) // TPB, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        self.branch_a.for_each_param[target](visitor, ctx)
        self.branch_b.for_each_param[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.branch_a.zero_grad[target](ctx)
        self.branch_b.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.branch_a.polyak_from[target](src.branch_a, tau, ctx)
        self.branch_b.polyak_from[target](src.branch_b, tau, ctx)

    def reinit[
        target: StaticString, INIT: Initializer
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.branch_a.reinit[target, INIT](ctx)
        self.branch_b.reinit[target, INIT](ctx)
