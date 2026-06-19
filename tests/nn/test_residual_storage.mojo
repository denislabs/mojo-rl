"""Residual[Inner] correctness (storage surface, CPU + GPU).

With the deterministic Linear init, a `Residual[Inner]` and a standalone `Inner`
share identical weights, so:
  forward:  residual_out      == inner_out      + x
  backward: residual_grad_in  == inner_grad_in  + grad_output
(Inner is pure-linear so its vjp doesn't mutate grad_output.)

Run: pixi run -e apple mojo run -I . tests/nn/test_residual_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.combinators.residual import Residual


comptime D = 6
comptime B = 4
comptime N = B * D
comptime INNER = Sequential[Linear[D, D], Linear[D, D]]
comptime RES = Residual[INNER]


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-4)
    var res = RES.make[target, Deterministic](ctx)
    var inner = INNER.make[target, Deterministic](ctx)

    var x = Tensor.alloc(N)
    var go_orig = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
        go_orig[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.2
    var go = Tensor.alloc(N)
    var go2 = Tensor.alloc(N)
    for i in range(N):
        go.data[i] = go_orig[i]
        go2.data[i] = go_orig[i]
    var out_res = Tensor.alloc(N)
    var out_in = Tensor.alloc(N)
    var gi_res = Tensor.alloc(N)
    var gi_in = Tensor.alloc(N)

    comptime if target == "gpu":
        x.upload(ctx.value()); go.upload(ctx.value()); go2.upload(ctx.value())

    res.forward[target, B](TensorRefs[1](x), out_res, ctx)
    inner.forward[target, B](TensorRefs[1](x), out_in, ctx)
    res.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi_res), ctx)
    inner.vjp[target, B](TensorRefs[1](x), go2, TensorRefs[1](gi_in), ctx)

    comptime if target == "gpu":
        out_res.download(ctx.value()); out_in.download(ctx.value())
        gi_res.download(ctx.value()); gi_in.download(ctx.value())

    var ok = True
    for i in range(N):
        if abs(out_res.data[i] - (out_in.data[i] + x.data[i])) > TOL:
            ok = False
        if abs(gi_res.data[i] - (gi_in.data[i] + go_orig[i])) > TOL:
            ok = False
    return ok


def main() raises:
    print("Residual[Inner] storage correctness")
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "Residual correctness")
    print("RESIDUAL OK")
