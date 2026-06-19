"""ComputeGraph ARITY-3 node support (CPU + GPU).

Extends the storage ComputeGraph to dispatch ARITY-3 nodes (needed by
ConditionalTransformerBlock's Modulate/Gate). Graph: 3 external inputs
(x, scale, shift) → one Modulate node = x*(1+scale)+shift. Checks forward +
vjp against the analytic grads:
  out    = x*(1+scale) + shift
  dx     = go*(1+scale)
  dscale = go*x
  dshift = go

Run:
  pixi run mojo run -I . tests/nn/test_compute_graph_arity3_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/test_compute_graph_arity3_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.modulate import Modulate
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph


comptime DIM = 4
comptime B = 3
comptime N = B * DIM
comptime GRAPH = ComputeGraph[3, Modulate[DIM]]


def _check(o_t: Tensor, go: Tensor, mut gin: TensorPack[3],
           x: Tensor, sc: Tensor, sh: Tensor, tol: Scalar[DT]) raises -> Bool:
    var mo = Scalar[DT](0); var mdx = Scalar[DT](0)
    var mds = Scalar[DT](0); var mdh = Scalar[DT](0)
    for i in range(N):
        var eo = x.data[i] * (Scalar[DT](1) + sc.data[i]) + sh.data[i]
        var edx = go.data[i] * (Scalar[DT](1) + sc.data[i])
        var eds = go.data[i] * x.data[i]
        var edh = go.data[i]
        if abs(o_t.data[i] - eo) > mo: mo = abs(o_t.data[i] - eo)
        if abs(gin[0].data[i] - edx) > mdx: mdx = abs(gin[0].data[i] - edx)
        if abs(gin[1].data[i] - eds) > mds: mds = abs(gin[1].data[i] - eds)
        if abs(gin[2].data[i] - edh) > mdh: mdh = abs(gin[2].data[i] - edh)
    print("  max Δ: out", mo, " dx", mdx, " dscale", mds, " dshift", mdh)
    return mo < tol and mdx < tol and mds < tol and mdh < tol


def test_cpu() raises:
    print("ComputeGraph ARITY-3 (Modulate) CPU ...")
    var g = GRAPH.make["cpu", Deterministic]()
    var edges = List[List[Int]]()
    edges.append([0, 1, 2])
    var x = Tensor.alloc(N); var sc = Tensor.alloc(N); var sh = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
        sc.data[i] = Scalar[DT]((i % 5) - 2) * 0.1
        sh.data[i] = Scalar[DT]((i % 3) - 1) * 0.3
        go.data[i] = Scalar[DT]((i % 4) - 1) * 0.25
    var inp = TensorPack[3]()
    inp[0].ensure(N); inp[1].ensure(N); inp[2].ensure(N)
    for i in range(N):
        inp[0].data[i] = x.data[i]
        inp[1].data[i] = sc.data[i]
        inp[2].data[i] = sh.data[i]
    var out = Tensor.alloc(N)
    g.forward[B](edges, inp, out, None)
    var gin = TensorPack[3]()
    g.vjp[B](edges, go, gin, None)
    var ok = _check(out, go, gin, x, sc, sh, Scalar[DT](1e-6))
    assert_true(ok, "CG arity3 CPU")
    print("  ok")


def test_gpu() raises:
    print("ComputeGraph ARITY-3 (Modulate) GPU ...")
    var c = DeviceContext()
    var g = GRAPH.make["gpu", Deterministic](Optional(c))
    var edges = List[List[Int]]()
    edges.append([0, 1, 2])
    var x = Tensor.alloc(N); var sc = Tensor.alloc(N); var sh = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
        sc.data[i] = Scalar[DT]((i % 5) - 2) * 0.1
        sh.data[i] = Scalar[DT]((i % 3) - 1) * 0.3
        go.data[i] = Scalar[DT]((i % 4) - 1) * 0.25
    var inp = TensorPack[3]()
    inp[0].ensure(N); inp[1].ensure(N); inp[2].ensure(N)
    for i in range(N):
        inp[0].data[i] = x.data[i]
        inp[1].data[i] = sc.data[i]
        inp[2].data[i] = sh.data[i]
    inp[0].upload(c); inp[1].upload(c); inp[2].upload(c); go.upload(c)
    var out = Tensor.alloc(N)
    g.forward[B, "gpu"](edges, inp, out, Optional(c))
    var gin = TensorPack[3]()
    g.vjp[B, "gpu"](edges, go, gin, Optional(c))
    out.download(c)
    gin[0].download(c); gin[1].download(c); gin[2].download(c)
    var ok = _check(out, go, gin, x, sc, sh, Scalar[DT](2e-5))
    assert_true(ok, "CG arity3 GPU")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("ComputeGraph ARITY-3 node support")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
