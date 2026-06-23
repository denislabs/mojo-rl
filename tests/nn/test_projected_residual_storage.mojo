"""ProjectedResidual[Inner, Skip] correctness (storage surface, CPU + GPU).

vs standalone identical branches (deterministic init):
  forward:  out == Inner(x) + Skip(x)
  backward: grad_input == Inner.vjp(go) + Skip.vjp(go)

Run: pixi run -e apple mojo run -I . tests/nn/test_projected_residual_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.projected_residual import ProjectedResidual


comptime D = 5
comptime H = 4
comptime O = 3
comptime B = 4
comptime INNER = Sequential[Linear[D, H], Linear[H, O]]
comptime SKIP = Linear[D, O]
comptime PR = ProjectedResidual[INNER, SKIP]


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-4)
    var pr = PR.make[target, Deterministic](ctx)
    var inner = INNER.make[target, Deterministic](ctx)
    var skip = SKIP.make[target, Deterministic](ctx)

    var x = Tensor.alloc(B * D)
    var go = Tensor.alloc(B * O)
    var go2 = Tensor.alloc(B * O)
    var go3 = Tensor.alloc(B * O)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.25
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.2
        go2.data[i] = go.data[i]
        go3.data[i] = go.data[i]
    var out = Tensor.alloc(B * O)
    var io = Tensor.alloc(B * O)
    var so = Tensor.alloc(B * O)
    var gi = Tensor.alloc(B * D)
    var gi_in = Tensor.alloc(B * D)
    var gi_sk = Tensor.alloc(B * D)

    comptime if target == "gpu":
        x.upload(ctx.value())
        go.upload(ctx.value()); go2.upload(ctx.value()); go3.upload(ctx.value())

    pr.forward[target, B](TensorRefs[1](x), out, ctx)
    inner.forward[target, B](TensorRefs[1](x), io, ctx)
    skip.forward[target, B](TensorRefs[1](x), so, ctx)
    pr.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
    inner.vjp[target, B](TensorRefs[1](x), go2, TensorRefs[1](gi_in), ctx)
    skip.vjp[target, B](TensorRefs[1](x), go3, TensorRefs[1](gi_sk), ctx)

    comptime if target == "gpu":
        out.download(ctx.value()); io.download(ctx.value()); so.download(ctx.value())
        gi.download(ctx.value()); gi_in.download(ctx.value()); gi_sk.download(ctx.value())

    var ok = True
    for i in range(B * O):
        if abs(out.data[i] - (io.data[i] + so.data[i])) > TOL:
            ok = False
    for i in range(B * D):
        if abs(gi.data[i] - (gi_in.data[i] + gi_sk.data[i])) > TOL:
            ok = False
    return ok


def main() raises:
    print("ProjectedResidual[Inner, Skip] storage correctness")
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "ProjectedResidual correctness")
    print("PROJECTED_RESIDUAL OK")
