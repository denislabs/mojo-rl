"""ComputeGraph ARITY-4 (and fan-out) node support (CPU + GPU).

Exercises the now-generic storage ComputeGraph dispatch on a node with
ARITY > 3 (the old hard cap), plus fan-out grad accumulation and a real
multi-node DAG. The graph:

  InputSlot["x", 2]                                   x = [x0, x1]
  Node["s0", Slice[2,0,1], "x"]                       s0 = x0
  Node["s1", Slice[2,1,2], "x"]                       s1 = x1
  Node["cat", Concat[1,1,1,1], "s0","s1","s0","s1"]   out = [x0, x1, x0, x1]

`cat` is ARITY-4 AND feeds slot "s0" from positions 0 & 2 and "s1" from
positions 1 & 3 — so the reverse walk must fan-out-accumulate. With
grad_out = [g0,g1,g2,g3]:
  d(s0) = g0 + g2 ;  d(s1) = g1 + g3
  d(x)  = [g0 + g2, g1 + g3]   (routed back through the two slices)

This is exactly what the old ARITY 1/2/3 unrolled dispatch could not
express (Concat[1,1,1,1] is the tdmpc2 WM-graph loss-vector node).

Run:
  pixi run mojo run -I . tests/nn/test_compute_graph_arity4_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/test_compute_graph_arity4_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.slice import Slice
from mojo_rl.nn.primitives.concat import Concat
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node


comptime IN = 2
comptime B = 3
comptime NX = B * IN  # input elem count
comptime NO = B * 4   # output elem count
comptime GRAPH = ComputeGraph[
    InputSlot["x", IN],
    Node["s0", Slice[IN, 0, 1], "x"],
    Node["s1", Slice[IN, 1, 2], "x"],
    Node["cat", Concat[1, 1, 1, 1], "s0", "s1", "s0", "s1"],
]


def _check(
    mut g: GRAPH, o_t: Tensor, go: Tensor, x: Tensor, tol: Scalar[DT]
) raises -> Bool:
    var mo = Scalar[DT](0)
    var mdx = Scalar[DT](0)
    # out[b] = [x0, x1, x0, x1]; dx = [g0+g2, g1+g3] per batch row.
    for b in range(B):
        var x0 = x.data[b * IN + 0]
        var x1 = x.data[b * IN + 1]
        var e0 = x0
        var e1 = x1
        var e2 = x0
        var e3 = x1
        if abs(o_t.data[b * 4 + 0] - e0) > mo: mo = abs(o_t.data[b * 4 + 0] - e0)
        if abs(o_t.data[b * 4 + 1] - e1) > mo: mo = abs(o_t.data[b * 4 + 1] - e1)
        if abs(o_t.data[b * 4 + 2] - e2) > mo: mo = abs(o_t.data[b * 4 + 2] - e2)
        if abs(o_t.data[b * 4 + 3] - e3) > mo: mo = abs(o_t.data[b * 4 + 3] - e3)
        var g0 = go.data[b * 4 + 0]
        var g1 = go.data[b * 4 + 1]
        var g2 = go.data[b * 4 + 2]
        var g3 = go.data[b * 4 + 3]
        var edx0 = g0 + g2
        var edx1 = g1 + g3
        ref dx = g.grad_input["x"]()
        if abs(dx.data[b * IN + 0] - edx0) > mdx: mdx = abs(dx.data[b * IN + 0] - edx0)
        if abs(dx.data[b * IN + 1] - edx1) > mdx: mdx = abs(dx.data[b * IN + 1] - edx1)
    print("  max Δ: out", mo, " dx", mdx)
    return mo < tol and mdx < tol


def _fill(mut x: Tensor, mut go: Tensor) raises:
    for i in range(NX):
        x.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    for i in range(NO):
        go.data[i] = Scalar[DT]((i % 4) - 1) * 0.25


def test_cpu() raises:
    print("ComputeGraph ARITY-4 (Concat[1,1,1,1] + fan-out) CPU ...")
    var g = GRAPH.make["cpu", Deterministic]()
    var x = Tensor.alloc(NX)
    var go = Tensor.alloc(NO)
    _fill(x, go)
    g.set_input["x", B](x)
    var out = Tensor.alloc(NO)
    g.forward[B](out)
    g.vjp[B](go)
    var ok = _check(g, out, go, x, Scalar[DT](1e-6))
    assert_true(ok, "CG arity4 CPU")
    print("  ok")


def test_gpu() raises:
    print("ComputeGraph ARITY-4 (Concat[1,1,1,1] + fan-out) GPU ...")
    var c = DeviceContext()
    var g = GRAPH.make["gpu", Deterministic](Optional(c))
    var x = Tensor.alloc(NX)
    var go = Tensor.alloc(NO)
    _fill(x, go)
    x.upload(c); go.upload(c)
    g.set_input["x", B](x, Optional(c))
    var out = Tensor.alloc(NO)
    g.forward[B, "gpu"](out, Optional(c))
    g.vjp[B, "gpu"](go, Optional(c))
    out.download(c)
    g.grad_input["x"]().download(c)
    var ok = _check(g, out, go, x, Scalar[DT](2e-5))
    assert_true(ok, "CG arity4 GPU")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("ComputeGraph ARITY-4 node support")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
