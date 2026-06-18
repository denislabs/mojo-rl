"""MSELoss storage — SAC-convention reference gate (CPU + GPU).

loss = (1/B)·Σ_b Σ_j 0.5·(l-t)² ; grad = (l-t)/B. Verifies forward (scalar),
forward_accumulate→read_accum (mean over a window), and vjp grads.

Run: pixi run -e apple mojo run -I . tests/nn/test_mse_loss_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.loss.mse_loss import MSELoss


comptime B = 4
comptime DIM = 3
comptime M = B * DIM


def _ref_loss(ref l: Tensor, ref t: Tensor) -> Scalar[DT]:
    var s: Scalar[DT] = 0.0
    for i in range(M):
        var d = l.data[i] - t.data[i]
        s += Scalar[DT](0.5) * d * d
    return s / Scalar[DT](B)


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-5)
    var loss = MSELoss[DIM].make_cpu() if target == "cpu" else MSELoss[DIM].make_gpu(ctx.value())
    var logits = Tensor.alloc(M)
    var targets = Tensor.alloc(M)
    for i in range(M):
        logits.data[i] = Scalar[DT]((i % 7) - 3) * 0.3
        targets.data[i] = Scalar[DT]((i % 5) - 2) * 0.2
    var ref_l = _ref_loss(logits, targets)
    var grad = Tensor.alloc(M)

    var got_fwd: Scalar[DT]
    var got_acc: Scalar[DT]
    comptime if target == "cpu":
        got_fwd = loss.forward["cpu", B](logits, targets, None)
        loss.reset_accum["cpu"]()
        loss.forward_accumulate["cpu", B](logits, targets, None)
        loss.forward_accumulate["cpu", B](logits, targets, None)
        got_acc = loss.read_accum["cpu"]()
        loss.vjp["cpu", B](logits, targets, grad, None)
    else:
        var c = ctx.value()
        logits.upload(c); targets.upload(c)
        got_fwd = loss.forward["gpu", B](logits, targets, Optional(c))
        loss.reset_accum["gpu"]()
        loss.forward_accumulate["gpu", B](logits, targets, Optional(c))
        loss.forward_accumulate["gpu", B](logits, targets, Optional(c))
        got_acc = loss.read_accum["gpu"](Optional(c))
        loss.vjp["gpu", B](logits, targets, grad, Optional(c))
        grad.download(c)

    var ok = True
    if abs(got_fwd - ref_l) > TOL: ok = False
    if abs(got_acc - ref_l) > TOL: ok = False   # 2 identical accums → mean == ref
    for i in range(M):
        var ref_g = (logits.data[i] - targets.data[i]) / Scalar[DT](B)
        if abs(grad.data[i] - ref_g) > TOL: ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("MSELoss storage (SAC convention) reference gate")
    print("=" * 70)
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "MSELoss parity")
    print("MSE LOSS OK")
