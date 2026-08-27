"""ExternalNode in a ComputeGraph == owning the same node (CPU + GPU).

`ExternalNode[NAME, M, *IN_NAMES]` (mojo_rl/nn/storage/combinators/graph_decl.mojo)
is a pure comptime node whose module is supplied at FORWARD TIME. The trainer
(which owns the actor/critics) threads them as tracked `mut` ref args into
`ComputeGraph.forward`/`vjp` (`mut *externals`); the graph dispatches each
`ExternalNode` slot to the matching external by node order. So an
`ExternalNode["lin", Linear, "x"]` fed an external Linear must produce the SAME
forward + input-grad as a `Node["lin", Linear, "x"]` that OWNS the Linear, and
grad must flow into the external Linear's Param.grd.

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
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import InputSlot, Node, ExternalNode
from mojo_rl.nn.core.initializer import Deterministic


comptime IN = 4
comptime H = 6
comptime B = 5
comptime N = B * IN
comptime OWN = ComputeGraph[
    InputSlot["x", IN], Node["lin", Linear[IN, H], "x"], Node["relu", ReLU[H], "lin"]
]
comptime EXT = ComputeGraph[
    InputSlot["x", IN], ExternalNode["lin", Linear[IN, H], "x"], Node["relu", ReLU[H], "lin"]
]


def test_cpu() raises:
    print("ExternalNode CPU == owned node ...")
    var own = OWN.make["cpu", Deterministic]()
    var ext = EXT.make["cpu", Deterministic]()
    var lin = Linear[IN, H].make[
        "cpu", Deterministic
    ]()  # external, same weights

    var x = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    own.set_input["x", B](x)
    ext.set_input["x", B](x)
    var o_own = Tensor.alloc(B * H)
    var o_ext = Tensor.alloc(B * H)
    own.forward[B](o_own)
    ext.forward[B](o_ext, None, lin)  # thread `lin` as external
    var fmax: Scalar[DT] = 0
    for i in range(B * H):
        var d = abs(o_own.data[i] - o_ext.data[i])
        if d > fmax:
            fmax = d
    print("  fwd max|own-ext|", fmax)
    assert_true(fmax < 1e-7, "ExternalNode forward == owned")

    var go_own = Tensor.alloc(B * H)
    var go_ext = Tensor.alloc(B * H)
    for i in range(B * H):
        go_own.data[i] = Scalar[DT]((i % 4) - 1) * 0.3
        go_ext.data[i] = Scalar[DT]((i % 4) - 1) * 0.3
    own.vjp[B](go_own)
    ext.vjp[B](go_ext, None, lin)
    var gmax: Scalar[DT] = 0
    for i in range(N):
        var d = abs(own.grad_input["x"]().data[i] - ext.grad_input["x"]().data[i])
        if d > gmax:
            gmax = d
    print("  grad_input max|own-ext|", gmax)
    assert_true(gmax < 1e-7, "ExternalNode grad_input == owned")
    # external Linear's weight grad must have been populated
    var wgrad_nz: Scalar[DT] = 0
    for i in range(IN * H):
        wgrad_nz += abs(lin.weight.grd.data[i])
    assert_true(wgrad_nz > 0.0, "grad flowed into external Linear.grd")
    print("  ok")


def test_gpu() raises:
    print("ExternalNode GPU == owned, repeatable across intervening matmul ...")
    var c = DeviceContext()
    var own = OWN.make["gpu", Deterministic](Optional(c))
    var ext = EXT.make["gpu", Deterministic](Optional(c))
    var lin = Linear[IN, H].make["gpu", Deterministic](Optional(c))

    var x = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    x.upload(c)
    own.set_input["x", B](x, Optional(c))
    ext.set_input["x", B](x, Optional(c))

    # ground truth = the owned graph (immune to the poisoning bug).
    var oref = Tensor.alloc_gpu(c, B * H)
    own.forward[B, "gpu"](oref, Optional(c))
    oref.download(c)

    # ext graph forward #1 (thread `lin`).
    var o1 = Tensor.alloc_gpu(c, B * H)
    ext.forward[B, "gpu"](o1, Optional(c), lin)
    o1.download(c)
    var m1: Scalar[DT] = 0
    for i in range(B * H):
        var d = abs(oref.data[i] - o1.data[i])
        if d > m1:
            m1 = d
    print("  fwd#1 max|own-ext|", m1)
    assert_true(m1 < 1e-5, "ExternalNode GPU forward == owned")

    # INTERVENING DIRECT MATMUL — the exact condition that used to poison fwd#2.
    var h = Tensor.alloc_gpu(c, B * H)
    lin.forward["gpu", B](TensorRefs[1](x), h, Optional(c))
    h.download(c)

    # ext graph forward #2 — must stay correct.
    var o2 = Tensor.alloc_gpu(c, B * H)
    ext.forward[B, "gpu"](o2, Optional(c), lin)
    o2.download(c)
    var m2: Scalar[DT] = 0
    for i in range(B * H):
        var d = abs(oref.data[i] - o2.data[i])
        if d > m2:
            m2 = d
    print("  fwd#2 max|own-ext| (after intervening matmul)", m2)
    assert_true(m2 < 1e-5, "ExternalNode GPU repeatable after intervening matmul")

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
    own.vjp[B, "gpu"](go_own, Optional(c))
    ext.vjp[B, "gpu"](go_ext, Optional(c), lin)
    own.grad_input["x"]().download(c)
    ext.grad_input["x"]().download(c)
    var gmax: Scalar[DT] = 0
    for i in range(N):
        var d = abs(own.grad_input["x"]().data[i] - ext.grad_input["x"]().data[i])
        if d > gmax:
            gmax = d
    print("  grad_input max|own-ext|", gmax)
    assert_true(gmax < 1e-5, "ExternalNode GPU grad_input == owned")
    lin.weight.grd.download(c)
    var wnz: Scalar[DT] = 0
    for i in range(IN * H):
        wnz += abs(lin.weight.grd.data[i])
    assert_true(wnz > 0.0, "grad flowed into external Linear.grd (GPU)")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("ExternalNode storage gate")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
