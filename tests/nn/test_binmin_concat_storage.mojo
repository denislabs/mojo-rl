"""BinaryElemMin + Concat2 storage — reference gate (CPU + GPU).

Both are deterministic, so verify forward + both grad-inputs against a direct
reference. min: out=min(a,b), grad credited to the winner (ties → in1).
concat: out=[a|b], grads split back.

Run: pixi run -e apple mojo run -I . tests/nn/test_binmin_concat_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.primitives.binary_elementwise import BinaryElemMin
from mojo_rl.nn.primitives.concat import Concat2


comptime B = 4
comptime DIM = 5
comptime D0 = 3
comptime D1 = 4


def _check_min[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime M = B * DIM
    comptime TOL = Scalar[DT](1e-6)
    var op = BinaryElemMin[DIM].make[target, Deterministic](ctx)
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
    comptime if target == "cpu":
        op.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), out, None)
        op.vjp["cpu", B](TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](g[0], g[1]), None)
    else:
        var c = ctx.value()
        ins[0].upload(c); ins[1].upload(c); go.upload(c)
        op.forward["gpu", B](TensorRefs[2](ins[0], ins[1]), out, ctx)
        op.vjp["gpu", B](TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](g[0], g[1]), ctx)
        out.download(c); g[0].download(c); g[1].download(c)
    var ok = True
    for i in range(M):
        var a = ins[0].data[i]
        var b = ins[1].data[i]
        var win0 = a < b
        var ref_o = a if win0 else b
        if abs(out.data[i] - ref_o) > TOL: ok = False
        var ref_g0 = go.data[i] if win0 else Scalar[DT](0)
        var ref_g1 = Scalar[DT](0) if win0 else go.data[i]
        if abs(g[0].data[i] - ref_g0) > TOL: ok = False
        if abs(g[1].data[i] - ref_g1) > TOL: ok = False
    return ok


def _check_concat[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime OUT = D0 + D1
    comptime TOL = Scalar[DT](1e-6)
    var op = Concat2[D0, D1].make[target, Deterministic](ctx)
    var ins = TensorPack[2]()
    ins[0].ensure(B * D0)
    ins[1].ensure(B * D1)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * D0):
        ins[0].data[i] = Scalar[DT](i + 1) * 0.1
    for i in range(B * D1):
        ins[1].data[i] = Scalar[DT](-(i + 1)) * 0.1
    for i in range(B * OUT):
        go.data[i] = Scalar[DT]((i % 6) - 3) * 0.3
    var out = Tensor.alloc(B * OUT)
    var g = TensorPack[2]()
    comptime if target == "cpu":
        op.forward["cpu", B](TensorRefs[2](ins[0], ins[1]), out, None)
        op.vjp["cpu", B](TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](g[0], g[1]), None)
    else:
        var c = ctx.value()
        ins[0].upload(c); ins[1].upload(c); go.upload(c)
        op.forward["gpu", B](TensorRefs[2](ins[0], ins[1]), out, ctx)
        op.vjp["gpu", B](TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](g[0], g[1]), ctx)
        out.download(c); g[0].download(c); g[1].download(c)
    var ok = True
    for bi in range(B):
        for c in range(D0):
            if abs(out.data[bi * OUT + c] - ins[0].data[bi * D0 + c]) > TOL: ok = False
            if abs(g[0].data[bi * D0 + c] - go.data[bi * OUT + c]) > TOL: ok = False
        for c in range(D1):
            if abs(out.data[bi * OUT + D0 + c] - ins[1].data[bi * D1 + c]) > TOL: ok = False
            if abs(g[1].data[bi * D1 + c] - go.data[bi * OUT + D0 + c]) > TOL: ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("BinaryElemMin + Concat2 storage reference gate")
    print("=" * 70)
    var c = DeviceContext()
    var ok = True
    var a = _check_min["cpu"](None); print("  min   CPU:", "OK" if a else "FAIL"); ok = a and ok
    var b = _check_min["gpu"](Optional(c)); print("  min   GPU:", "OK" if b else "FAIL"); ok = b and ok
    var d = _check_concat["cpu"](None); print("  concat CPU:", "OK" if d else "FAIL"); ok = d and ok
    var e = _check_concat["gpu"](Optional(c)); print("  concat GPU:", "OK" if e else "FAIL"); ok = e and ok
    assert_true(ok, "binmin/concat parity")
    print("BINMIN + CONCAT OK")
