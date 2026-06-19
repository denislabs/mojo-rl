"""StopGradParams[Inner] correctness (storage surface, CPU + GPU).

vs standalone identical Linear (deterministic init), grads zeroed first:
  forward:  passthrough == Linear(x)
  backward: grad_input == Linear.vjp grad_input
  FREEZE:   inner param grads stay ~0 after vjp (standalone Linear's are nonzero)

Run: pixi run -e apple mojo run -I . tests/nn/test_stop_grad_params_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.combinators.stop_grad_params import StopGradParams


comptime D = 5
comptime O = 3
comptime B = 4
comptime WSZ = D * O
comptime SG = StopGradParams[Linear[D, O]]


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-4)
    var sg = SG.make[target, Deterministic](ctx)
    var lin = Linear[D, O].make[target, Deterministic](ctx)
    sg.zero_grad[target](ctx)
    lin.zero_grad[target](ctx)

    var x = Tensor.alloc(B * D)
    var go = Tensor.alloc(B * O)
    var go2 = Tensor.alloc(B * O)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.25
        go2.data[i] = go.data[i]
    var o_sg = Tensor.alloc(B * O); var o_l = Tensor.alloc(B * O)
    var gi_sg = Tensor.alloc(B * D); var gi_l = Tensor.alloc(B * D)

    comptime if target == "gpu":
        x.upload(ctx.value()); go.upload(ctx.value()); go2.upload(ctx.value())

    sg.forward[target, B](TensorRefs[1](x), o_sg, ctx)
    lin.forward[target, B](TensorRefs[1](x), o_l, ctx)
    sg.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi_sg), ctx)
    lin.vjp[target, B](TensorRefs[1](x), go2, TensorRefs[1](gi_l), ctx)

    comptime if target == "gpu":
        o_sg.download(ctx.value()); o_l.download(ctx.value())
        gi_sg.download(ctx.value()); gi_l.download(ctx.value())
        sg.inner.weight.grd.download(ctx.value())
        lin.weight.grd.download(ctx.value())

    var ok = True
    for i in range(B * O):
        if abs(o_sg.data[i] - o_l.data[i]) > TOL:
            ok = False
    for i in range(B * D):
        if abs(gi_sg.data[i] - gi_l.data[i]) > TOL:
            ok = False
    # freeze: StopGradParams inner weight grad ~0; standalone Linear's nonzero
    var sg_g: Scalar[DT] = 0.0
    var lin_g: Scalar[DT] = 0.0
    for k in range(WSZ):
        sg_g += abs(sg.inner.weight.grd.data[k])
        lin_g += abs(lin.weight.grd.data[k])
    if sg_g > TOL:
        ok = False
    if lin_g < Scalar[DT](1e-3):  # sanity: the grad WOULD have been nonzero
        ok = False
    return ok


def main() raises:
    print("StopGradParams[Inner] storage correctness")
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "StopGradParams correctness")
    print("STOP_GRAD_PARAMS OK")
