"""SkipConcat[Inner] correctness (storage surface, CPU + GPU).

vs standalone identical Inner (deterministic init):
  forward:  out[:, :IN] == x,  out[:, IN:] == Inner(x)
  backward: grad_input == grad_output[:, :IN] + Inner.vjp(grad_output[:, IN:])

Run: pixi run -e apple mojo run -I . tests/nn/test_skip_concat_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.skip_concat import SkipConcat


comptime D = 4
comptime H = 5
comptime O = 3
comptime B = 4
comptime OUT = D + O
comptime INNER = Sequential[Linear[D, H], Linear[H, O]]
comptime SC = SkipConcat[INNER]


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-4)
    var sc = SC.make[target, Deterministic](ctx)
    var inner = INNER.make[target, Deterministic](ctx)

    var x = Tensor.alloc(B * D)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.25
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.2
    # standalone inner grad-output = go[:, D:]; passthrough = go[:, :D]
    var go_inner = Tensor.alloc(B * O)
    var go_pass = List[Scalar[DT]](length=B * D, fill=Scalar[DT](0))
    for b in range(B):
        for j in range(D):
            go_pass[b * D + j] = go.data[b * OUT + j]
        for j in range(O):
            go_inner.data[b * O + j] = go.data[b * OUT + D + j]

    var out = Tensor.alloc(B * OUT)
    var io = Tensor.alloc(B * O)
    var gi = Tensor.alloc(B * D)
    var gi_in = Tensor.alloc(B * D)

    comptime if target == "gpu":
        x.upload(ctx.value()); go.upload(ctx.value()); go_inner.upload(ctx.value())

    sc.forward[target, B](TensorRefs[1](x), out, ctx)
    inner.forward[target, B](TensorRefs[1](x), io, ctx)
    sc.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
    inner.vjp[target, B](TensorRefs[1](x), go_inner, TensorRefs[1](gi_in), ctx)

    comptime if target == "gpu":
        out.download(ctx.value()); io.download(ctx.value())
        gi.download(ctx.value()); gi_in.download(ctx.value())

    var ok = True
    for b in range(B):
        for j in range(D):
            if abs(out.data[b * OUT + j] - x.data[b * D + j]) > TOL:
                ok = False
        for j in range(O):
            if abs(out.data[b * OUT + D + j] - io.data[b * O + j]) > TOL:
                ok = False
    for i in range(B * D):
        if abs(gi.data[i] - (go_pass[i] + gi_in.data[i])) > TOL:
            ok = False
    return ok


def main() raises:
    print("SkipConcat[Inner] storage correctness")
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "SkipConcat correctness")
    print("SKIP_CONCAT OK")
