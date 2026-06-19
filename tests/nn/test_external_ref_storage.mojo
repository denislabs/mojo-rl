"""ExternalRef in a ComputeGraph == owning the same node (CPU + GPU).

`ExternalRef[M]` (mojo_rl/nn/storage/combinators/external_ref.mojo) is a pure
comptime MARKER for a graph slot whose module is supplied at FORWARD TIME. The
trainer (which owns the actor/critics) threads them as tracked `mut` ref args
into `ComputeGraph.forward`/`vjp` (`mut *externals`); the graph dispatches each
`ExternalRef` slot to the matching external by node order. So a
`ComputeGraph[1, ExternalRef[Linear], ReLU]` fed an external Linear must produce
the SAME forward + grad_inputs as `ComputeGraph[1, Linear, ReLU]` that OWNS the
Linear, and grad must flow into the external Linear's Param.grd.

============================ REGRESSION: GPU matmul poisoning ============================
A previous design stored the external module as a wildcard-origin
`Pointer[M, MutAnyOrigin]` FIELD. On GPU that disabled argument-exclusivity and
miscompiled the delegated matmul: the 2nd graph forward after ANY intervening
matmul produced structured garbage (owned nodes immune; CPU immune). Fixed by
threading the module as a tracked forward/vjp ARGUMENT instead of storing a
pointer. `test_gpu` guards the exact poisoning condition (an intervening direct
`lin.forward` between two graph forwards). See
docs/BUG_REPORT_gpu_matmul_wildcard_pointer_miscompile.md.

Run:
  pixi run mojo run -I . tests/nn/test_external_ref_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/test_external_ref_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.external_ref import ExternalRef
from mojo_rl.nn.storage.core.initializer import Deterministic


comptime IN = 4
comptime H = 6
comptime B = 5
comptime N = B * IN
comptime OWN = ComputeGraph[1, Linear[IN, H], ReLU[H]]
comptime EXT = ComputeGraph[1, ExternalRef[Linear[IN, H]], ReLU[H]]


def _edges() -> List[List[Int]]:
    var e = List[List[Int]]()
    e.append([0])
    e.append([1])
    return e^


def test_cpu() raises:
    print("ExternalRef CPU == owned node ...")
    var own = OWN.make["cpu", Deterministic]()
    var ext = EXT.make["cpu", Deterministic]()
    var lin = Linear[IN, H].make[
        "cpu", Deterministic
    ]()  # external, same weights
    var edges = _edges()

    var x = TensorPack[1]()
    x[0].ensure(N)
    for i in range(N):
        x[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var o_own = Tensor.alloc(B * H)
    var o_ext = Tensor.alloc(B * H)
    own.forward[B](edges, x, o_own)
    ext.forward[B](edges, x, o_ext, None, lin)  # thread `lin` as external
    var fmax: Scalar[DT] = 0
    for i in range(B * H):
        var d = abs(o_own.data[i] - o_ext.data[i])
        if d > fmax:
            fmax = d
    print("  fwd max|own-ext|", fmax)
    assert_true(fmax < 1e-7, "ExternalRef forward == owned")

    var go_own = Tensor.alloc(B * H)
    var go_ext = Tensor.alloc(B * H)
    for i in range(B * H):
        go_own.data[i] = Scalar[DT]((i % 4) - 1) * 0.3
        go_ext.data[i] = Scalar[DT]((i % 4) - 1) * 0.3
    var gi_own = TensorPack[1]()
    var gi_ext = TensorPack[1]()
    own.vjp[B](edges, go_own, gi_own)
    ext.vjp[B](edges, go_ext, gi_ext, None, lin)
    var gmax: Scalar[DT] = 0
    for i in range(N):
        var d = abs(gi_own[0].data[i] - gi_ext[0].data[i])
        if d > gmax:
            gmax = d
    print("  grad_input max|own-ext|", gmax)
    assert_true(gmax < 1e-7, "ExternalRef grad_input == owned")
    # external Linear's weight grad must have been populated
    var wgrad_nz: Scalar[DT] = 0
    for i in range(IN * H):
        wgrad_nz += abs(lin.weight.grd.data[i])
    assert_true(wgrad_nz > 0.0, "grad flowed into external Linear.grd")
    print("  ok")


def test_gpu() raises:
    print("ExternalRef GPU == owned, repeatable across intervening matmul ...")
    var c = DeviceContext()
    var own = OWN.make["gpu", Deterministic](Optional(c))
    var ext = EXT.make["gpu", Deterministic](Optional(c))
    var lin = Linear[IN, H].make["gpu", Deterministic](Optional(c))
    var edges = _edges()

    var x = TensorPack[1]()
    x[0].ensure(N)
    for i in range(N):
        x[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    x[0].upload(c)

    # ground truth = the owned graph (immune to the poisoning bug).
    var oref = Tensor.alloc_gpu(c, B * H)
    own.forward[B, "gpu"](edges, x, oref, Optional(c))
    oref.download(c)

    # ext graph forward #1 (thread `lin`).
    var o1 = Tensor.alloc_gpu(c, B * H)
    ext.forward[B, "gpu"](edges, x, o1, Optional(c), lin)
    o1.download(c)
    var m1: Scalar[DT] = 0
    for i in range(B * H):
        var d = abs(oref.data[i] - o1.data[i])
        if d > m1:
            m1 = d
    print("  fwd#1 max|own-ext|", m1)
    assert_true(m1 < 1e-5, "ExternalRef GPU forward == owned")

    # INTERVENING DIRECT MATMUL — the exact condition that used to poison fwd#2.
    var h = Tensor.alloc_gpu(c, B * H)
    lin.forward["gpu", B](TensorRefs[1](x[0]), h, Optional(c))
    h.download(c)

    # ext graph forward #2 — must stay correct.
    var o2 = Tensor.alloc_gpu(c, B * H)
    ext.forward[B, "gpu"](edges, x, o2, Optional(c), lin)
    o2.download(c)
    var m2: Scalar[DT] = 0
    for i in range(B * H):
        var d = abs(oref.data[i] - o2.data[i])
        if d > m2:
            m2 = d
    print("  fwd#2 max|own-ext| (after intervening matmul)", m2)
    assert_true(m2 < 1e-5, "ExternalRef GPU repeatable after intervening matmul")

    # vjp parity + grad into the external Linear.
    var go_own = Tensor()
    go_own.ensure(B * H)
    var go_ext = Tensor()
    go_ext.ensure(B * H)
    for i in range(B * H):
        go_own.data[i] = Scalar[DT]((i % 4) - 1) * 0.3
        go_ext.data[i] = Scalar[DT]((i % 4) - 1) * 0.3
    go_own.upload(c)
    go_ext.upload(c)
    var gi_own = TensorPack[1]()
    var gi_ext = TensorPack[1]()
    own.vjp[B, "gpu"](edges, go_own, gi_own, Optional(c))
    ext.vjp[B, "gpu"](edges, go_ext, gi_ext, Optional(c), lin)
    gi_own[0].download(c)
    gi_ext[0].download(c)
    var gmax: Scalar[DT] = 0
    for i in range(N):
        var d = abs(gi_own[0].data[i] - gi_ext[0].data[i])
        if d > gmax:
            gmax = d
    print("  grad_input max|own-ext|", gmax)
    assert_true(gmax < 1e-5, "ExternalRef GPU grad_input == owned")
    lin.weight.grd.download(c)
    var wnz: Scalar[DT] = 0
    for i in range(IN * H):
        wnz += abs(lin.weight.grd.data[i])
    assert_true(wnz > 0.0, "grad flowed into external Linear.grd (GPU)")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("ExternalRef storage gate")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
