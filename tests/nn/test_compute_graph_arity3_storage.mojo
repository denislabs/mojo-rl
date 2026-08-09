"""ComputeGraph ARITY-3 node support (CPU + GPU).

Exercises the storage ComputeGraph dispatching an ARITY-3 node (needed by
ConditionalTransformerBlock's Modulate/Gate) with the NAME-wired DX: 3
InputSlots (x, scale, shift) → one Modulate node = x*(1+scale)+shift. Checks
forward + vjp against the analytic grads:
  out    = x*(1+scale) + shift
  dx     = go*(1+scale)
  dscale = go*x
  dshift = go

Run:
  pixi run mojo run -I . tests/nn/test_compute_graph_arity3_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/test_compute_graph_arity3_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.modulate import Modulate
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node


comptime DIM = 4
comptime B = 3
comptime N = B * DIM
comptime GRAPH = ComputeGraph[
    InputSlot["x", DIM],
    InputSlot["sc", DIM],
    InputSlot["sh", DIM],
    Node["mod", Modulate[DIM], "x", "sc", "sh"],
]


def _check(mut g: GRAPH, o_t: Tensor, go: Tensor,
           x: Tensor, sc: Tensor, sh: Tensor, tol: Scalar[DT]) raises -> Bool:
    var mo = Scalar[DT](0); var mdx = Scalar[DT](0)
    var mds = Scalar[DT](0); var mdh = Scalar[DT](0)
    for i in range(N):
        var eo = x.data[i] * (Scalar[DT](1) + sc.data[i]) + sh.data[i]
        var edx = go.data[i] * (Scalar[DT](1) + sc.data[i])
        var eds = go.data[i] * x.data[i]
        var edh = go.data[i]
        if abs(o_t.data[i] - eo) > mo: mo = abs(o_t.data[i] - eo)
        if abs(g.grad_input["x"]().data[i] - edx) > mdx: mdx = abs(g.grad_input["x"]().data[i] - edx)
        if abs(g.grad_input["sc"]().data[i] - eds) > mds: mds = abs(g.grad_input["sc"]().data[i] - eds)
        if abs(g.grad_input["sh"]().data[i] - edh) > mdh: mdh = abs(g.grad_input["sh"]().data[i] - edh)
    print("  max Δ: out", mo, " dx", mdx, " dscale", mds, " dshift", mdh)
    return mo < tol and mdx < tol and mds < tol and mdh < tol


def _fill(mut x: Tensor, mut sc: Tensor, mut sh: Tensor, mut go: Tensor) raises:
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
        sc.data[i] = Scalar[DT]((i % 5) - 2) * 0.1
        sh.data[i] = Scalar[DT]((i % 3) - 1) * 0.3
        go.data[i] = Scalar[DT]((i % 4) - 1) * 0.25


def test_cpu() raises:
    print("ComputeGraph ARITY-3 (Modulate) CPU ...")
    var g = GRAPH.make["cpu", Deterministic]()
    var x = Tensor.alloc(N); var sc = Tensor.alloc(N); var sh = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    _fill(x, sc, sh, go)
    g.set_input["x", B](x)
    g.set_input["sc", B](sc)
    g.set_input["sh", B](sh)
    var out = Tensor.alloc(N)
    g.forward[B](out)
    g.vjp[B](go)
    var ok = _check(g, out, go, x, sc, sh, Scalar[DT](1e-6))
    assert_true(ok, "CG arity3 CPU")
    print("  ok")


def test_gpu() raises:
    print("ComputeGraph ARITY-3 (Modulate) GPU ...")
    var c = DeviceContext()
    var g = GRAPH.make["gpu", Deterministic](Optional(c))
    var x = Tensor.alloc(N); var sc = Tensor.alloc(N); var sh = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    _fill(x, sc, sh, go)
    x.upload(c); sc.upload(c); sh.upload(c); go.upload(c)
    g.set_input["x", B](x, Optional(c))
    g.set_input["sc", B](sc, Optional(c))
    g.set_input["sh", B](sh, Optional(c))
    var out = Tensor.alloc(N)
    g.forward[B, "gpu"](out, Optional(c))
    g.vjp[B, "gpu"](go, Optional(c))
    out.download(c)
    g.grad_input["x"]().download(c)
    g.grad_input["sc"]().download(c)
    g.grad_input["sh"]().download(c)
    var ok = _check(g, out, go, x, sc, sh, Scalar[DT](2e-5))
    assert_true(ok, "CG arity3 GPU")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("ComputeGraph ARITY-3 node support")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
