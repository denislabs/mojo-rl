"""Concat2[D0, D1] — binary feature-axis concatenation (storage surface).

  inputs (in0 [BATCH, D0], in1 [BATCH, D1])  →  output [BATCH, D0+D1]
  out[b, 0:D0] = in0[b] ; out[b, D0:D0+D1] = in1[b]

The SAC critic's `concat(state, action)` input. ARITY 2; fits the ComputeGraph's
binary dispatch. No params, no cache (backward is a pure slice-split of
grad_output). Higher-arity concat (>2 inputs) can be expressed as a chain of
Concat2 in the graph.

Backward: grad_in0 = grad_out[:, 0:D0] ; grad_in1 = grad_out[:, D0:D0+D1].
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor


def _dims2(d0: Int, d1: Int) -> InlineArray[Int, 2]:
    var a = InlineArray[Int, 2](fill=0)
    a[0] = d0
    a[1] = d1
    return a


def _concat_fwd_kernel[
    BATCH: Int, D0: Int, D1: Int, OUT: Int
](
    a: LayoutTensor[DT, Layout.row_major(BATCH, D0), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BATCH, D1), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT:
        return
    var bi = idx // OUT
    var c = idx % OUT
    if c < D0:
        output[bi, c] = rebind[Scalar[DT]](a[bi, c])
    else:
        output[bi, c] = rebind[Scalar[DT]](b[bi, c - D0])


def _concat_bwd_kernel[
    BATCH: Int, D0: Int, D1: Int, OUT: Int
](
    go: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    gi0: LayoutTensor[DT, Layout.row_major(BATCH, D0), MutAnyOrigin],
    gi1: LayoutTensor[DT, Layout.row_major(BATCH, D1), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT:
        return
    var bi = idx // OUT
    var c = idx % OUT
    if c < D0:
        gi0[bi, c] = rebind[Scalar[DT]](go[bi, c])
    else:
        gi1[bi, c - D0] = rebind[Scalar[DT]](go[bi, c])


struct Concat2[D0_: Int, D1_: Int](Module):
    comptime ARITY = 2
    comptime IN_DIMS = _dims2(Self.D0_, Self.D1_)
    comptime OUT_DIM = Self.D0_ + Self.D1_

    def __init__(out self):
        pass

    @staticmethod
    def make_cpu() raises -> Self:
        return Self()

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime OUT = Self.D0_ + Self.D1_
        ref a = inputs[0]
        ref b = inputs[1]
        comptime if target == "cpu":
            out.ensure(B * OUT)
            for bi in range(B):
                for c in range(Self.D0_):
                    out.data[bi * OUT + c] = a.data[bi * Self.D0_ + c]
                for c in range(Self.D1_):
                    out.data[bi * OUT + Self.D0_ + c] = b.data[bi * Self.D1_ + c]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * OUT)
            comptime nblk = (B * OUT + TPB - 1) // TPB
            c.enqueue_function[_concat_fwd_kernel[B, Self.D0_, Self.D1_, OUT]](
                a.lt_gpu[Layout.row_major(B, Self.D0_)](),
                b.lt_gpu[Layout.row_major(B, Self.D1_)](),
                out.lt_gpu[Layout.row_major(B, OUT)](),
                grid_dim=nblk, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime OUT = Self.D0_ + Self.D1_
        ref gi0 = grad_inputs[0]
        ref gi1 = grad_inputs[1]
        comptime if target == "cpu":
            gi0.ensure(B * Self.D0_)
            gi1.ensure(B * Self.D1_)
            for bi in range(B):
                for c in range(Self.D0_):
                    gi0.data[bi * Self.D0_ + c] = grad_output.data[bi * OUT + c]
                for c in range(Self.D1_):
                    gi1.data[bi * Self.D1_ + c] = grad_output.data[bi * OUT + Self.D0_ + c]
        else:
            var c = ctx.value()
            gi0.ensure_gpu(c, B * Self.D0_)
            gi1.ensure_gpu(c, B * Self.D1_)
            comptime nblk = (B * OUT + TPB - 1) // TPB
            c.enqueue_function[_concat_bwd_kernel[B, Self.D0_, Self.D1_, OUT]](
                grad_output.lt_gpu[Layout.row_major(B, OUT)](),
                gi0.lt_gpu[Layout.row_major(B, Self.D0_)](),
                gi1.lt_gpu[Layout.row_major(B, Self.D1_)](),
                grid_dim=nblk, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        pass

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        pass
