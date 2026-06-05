"""SPIKE — Module-wraps-ComputeGraph (LeWM Phase B prerequisite).

Path A makes `ConditionalTransformerBlock` an ARITY=2 Module whose body
is an internal ComputeGraph (per-token Tokenwise ops + Modulate/Gate +
MHA). The plan flagged this delegation pattern as "spike first": can a
struct own a `ComputeGraph` field, conform to `Module`, and delegate
forward / vjp / for_each_param to it?

`GraphBlock[D]` (ARITY=2): internal graph  x,c → lin=Linear(x) → out=lin+c.
Validates:
  1. forward delegates  (out = x@W + b + c),
  2. vjp delegates + routes grads to BOTH input slots (fd-gradcheck),
  3. for_each_param delegates (the internal Linear's params are visited).
CPU-only; the real block adds the GPU grad-copy.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators import ComputeGraph, InputSlot, Node
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.add import Add
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for


struct GraphBlock[D: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.D)
    comptime OUT_DIM = Self.D

    comptime Graph = ComputeGraph[
        Self.D,
        InputSlot["x", Self.D],
        InputSlot["c", Self.D],
        Node["lin", Linear[Self.D, Self.D], "x"],
        Node["out", Add[Self.D, 2], "lin", "c"],
    ]

    var graph: Self.Graph
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.Graph()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var b = Self()
        b.graph = Self.Graph.make[target=target, INIT=INIT](ctx=ctx)
        comptime if target == "cpu":
            b.ts = TargetStorage.make_cpu()
        else:
            b.ts = TargetStorage.make_gpu(ctx.value())
        return b^

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
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
        assert_tag_for["GraphBlock", target](self.ts.target_tag)
        var x = typed_view[BATCH, Self.D](inputs[0])
        var c = typed_view[BATCH, Self.D](inputs[1])
        var out = typed_view_mut[BATCH, Self.D](output)
        self.graph.set_input["x", BATCH](x)
        self.graph.set_input["c", BATCH](c)
        self.graph.forward[target, BATCH, POLICY=POLICY](out)

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
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
        assert_tag_for["GraphBlock", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.D](grad_output)
        var gx = typed_view_mut[BATCH, Self.D](grad_inputs[0])
        var gc = typed_view_mut[BATCH, Self.D](grad_inputs[1])
        self.graph.vjp[target, BATCH, POLICY=POLICY, mode=mode](go)
        # Copy the graph's accumulated input grads into the output tiles.
        var gx_src = self.graph.grad_input_ptr["x"]()
        var gc_src = self.graph.grad_input_ptr["c"]()
        comptime if target == "cpu":
            for b in range(BATCH):
                for i in range(Self.D):
                    gx[b, i] = gx_src[b * Self.D + i]
                    gc[b, i] = gc_src[b * Self.D + i]
        else:
            raise Error("GraphBlock: CPU-only spike")

    def for_each_param[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["GraphBlock", target](self.ts.target_tag)
        self.graph.for_each_param[target, V](prefix, visitor)


struct ParamCounter(ParamVisitor):
    var count: Int
    var total_elems: Int

    def __init__(out self):
        self.count = 0
        self.total_elems = 0

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        self.count += 1
        self.total_elems += n_elems


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def test_module_wraps_graph() raises:
    print("test_module_wraps_graph ...")
    comptime BATCH = 3
    comptime D = 4
    comptime N = BATCH * D

    var blk = GraphBlock[D].make[target="cpu", INIT=Kaiming]()

    var x = _a(N); var c = _a(N); var y = _a(N); var w = _a(N)
    var gx = _a(N); var gc = _a(N)
    for k in range(N):
        x[k] = Scalar[DT](0.1 * Float64(k + 1))
        c[k] = Scalar[DT](-0.05 * Float64(k + 1))
        w[k] = Scalar[DT](0.1 * Float64((k * 7) % 5 + 1))

    var x_t = TileTensor(x, row_major[BATCH, D]())
    var c_t = TileTensor(c, row_major[BATCH, D]())
    var y_t = TileTensor(y, row_major[BATCH, D]())
    blk.forward["cpu", BATCH](x_t, c_t, output=y_t)

    # vjp → grads to both inputs.
    var w_t = TileTensor(w, row_major[BATCH, D]())
    var gx_t = TileTensor(gx, row_major[BATCH, D]())
    var gc_t = TileTensor(gc, row_major[BATCH, D]())
    blk.vjp["cpu", BATCH](w_t, gx_t, gc_t)

    # fd-gradcheck on x and c.
    comptime EPS = Scalar[DT](1e-3)
    for which in range(2):
        var p = x if which == 0 else c
        var ga = gx if which == 0 else gc
        for k in range(N):
            var saved = p[k]
            p[k] = saved + EPS
            blk.forward["cpu", BATCH](x_t, c_t, output=y_t)
            var lp: Scalar[DT] = 0.0
            for j in range(N):
                lp += w[j] * y[j]
            p[k] = saved - EPS
            blk.forward["cpu", BATCH](x_t, c_t, output=y_t)
            var lm: Scalar[DT] = 0.0
            for j in range(N):
                lm += w[j] * y[j]
            p[k] = saved
            var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
            var ad = (ga[k] - num).__abs__()
            var ok = ad < Scalar[DT](3e-4) or (
                ad / (ga[k].__abs__() + num.__abs__() + Scalar[DT](1e-4))
            ) < Scalar[DT](2e-2)
            assert_true(ok, "GraphBlock grad fd mismatch")

    # for_each_param delegation: the internal Linear has weight + bias.
    var counter = ParamCounter()
    blk.for_each_param["cpu", ParamCounter]("blk", counter)
    print("   params visited =", counter.count, " elems =", counter.total_elems)
    assert_true(counter.count >= 2, "for_each_param must reach Linear w+b")
    assert_true(counter.total_elems >= D * D, "weight elems must be present")

    _ = blk^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("SPIKE: Module wraps ComputeGraph (LeWM Phase B prereq)")
    print("=" * 70)
    test_module_wraps_graph()
    print("=" * 70)
    print("SPIKE PASSED")
    print("=" * 70)
