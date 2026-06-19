"""Add — binary (ARITY=2) elementwise Module conformer (CPU + GPU). z = a + b.

Each forward/vjp branches `comptime if target == "cpu"` (tracked `TileTensor`
over `.data`) `else` (device `LayoutTensor` via `lt_gpu` + a naive kernel).
The storage surface (`ref/mut Tensor`, `TensorRefs`) is identical on both
targets; the only GPU erasure is the kernel-arg `MutAnyOrigin`. Params are
`Param` (two `Tensor`s, cpu+dev).

LIFETIME NOTE: a pack subscript (`inputs[k]`) returns a TEMPORARY ref. Building
a view from `inputs[k].data` directly and using it LATER dangles (the temporary
dies at the end of the statement; a later op clobbers the stack). So each body
first binds the element to a named `ref` (`ref in0 = inputs[0]`) that lives for
the whole function, then builds views from that.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer


def _add_fwd_kernel[
    M: Int
](
    a: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < M:
        o[i] = a[i] + b[i]


def _copy_kernel[
    M: Int
](
    src: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < M:
        dst[i] = src[i]


# ── Add (binary) ───────────────────────────────────────────────────────
struct Add[DIM_: Int](Module):
    comptime ARITY = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime M = B * Self.DIM_
        ref a0 = inputs[0]
        ref a1 = inputs[1]
        comptime if target == "cpu":
            out.ensure(M)
            var a = TileTensor(a0.data, row_major[M]())
            var b = TileTensor(a1.data, row_major[M]())
            var o = TileTensor(out.data, row_major[M]())
            for i in range(M):
                o[i] = a[i] + b[i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, M)
            var al = a0.lt["gpu", Layout.row_major(M)]()
            var bl = a1.lt["gpu", Layout.row_major(M)]()
            var ol = out.lt["gpu", Layout.row_major(M)]()
            c.enqueue_function[_add_fwd_kernel[M]](
                al, bl, ol, grid_dim=(M + 255) // 256, block_dim=256
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
        comptime M = B * Self.DIM_
        ref g0 = grad_inputs[0]
        ref g1 = grad_inputs[1]
        comptime if target == "cpu":
            g0.ensure(M)
            g1.ensure(M)
            var go = TileTensor(grad_output.data, row_major[M]())
            var gi0 = TileTensor(g0.data, row_major[M]())
            var gi1 = TileTensor(g1.data, row_major[M]())
            for i in range(M):
                gi0[i] = go[i]
                gi1[i] = go[i]
        else:
            var c = ctx.value()
            g0.ensure_gpu(c, M)
            g1.ensure_gpu(c, M)
            var gol = grad_output.lt["gpu", Layout.row_major(M)]()
            var gi0l = g0.lt["gpu", Layout.row_major(M)]()
            var gi1l = g1.lt["gpu", Layout.row_major(M)]()
            comptime nblk = (M + 255) // 256
            c.enqueue_function[_copy_kernel[M]](
                gol, gi0l, grid_dim=nblk, block_dim=256
            )
            c.enqueue_function[_copy_kernel[M]](
                gol, gi1l, grid_dim=nblk, block_dim=256
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
