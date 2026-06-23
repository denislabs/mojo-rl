"""BinaryElemAdd storage — reference gate (CPU).

out = a + b; backward passes grad_output to BOTH inputs (gi0 = gi1 = go).
Mirrors test_binmin_concat_storage.mojo. The additive TD3 target-y op.

Run: pixi run mojo run -I . tests/nn/test_binadd_storage.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.primitives.binary_elementwise import BinaryElemAdd


comptime B = 4
comptime DIM = 5


def _check_add() raises -> Bool:
    comptime M = B * DIM
    comptime TOL = Scalar[DT](1e-6)
    var op = BinaryElemAdd[DIM].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(M)
    ins[1].ensure(M)
    var go = Tensor.alloc(M)
    for i in range(M):
        ins[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.2
        ins[1].data[i] = Scalar[DT]((i % 5) - 2) * 0.25
        go.data[i] = Scalar[DT]((i % 3) + 1) * 0.4
    var out = Tensor.alloc(M)
    var g = TensorPack[2]()
    op.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), out, None)
    op.vjp["cpu", B](
        TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](g[0], g[1]), None
    )
    var ok = True
    for i in range(M):
        var ref_o = ins[0].data[i] + ins[1].data[i]
        if abs(out.data[i] - ref_o) > TOL:
            ok = False
        if abs(g[0].data[i] - go.data[i]) > TOL:
            ok = False
        if abs(g[1].data[i] - go.data[i]) > TOL:
            ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("BinaryElemAdd storage reference gate")
    print("=" * 70)
    var ok = _check_add()
    print("  add CPU:", "OK" if ok else "FAIL")
    assert_true(ok, "binadd parity")
    print("BINADD OK")
