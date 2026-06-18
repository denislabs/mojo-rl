"""Adam storage optimizer — reference parity (CPU + GPU), multi-step.

Drives a single `Param` through N Adam steps with deterministic grads and
compares `val` against an independent reference implementation of the standard
bias-corrected Adam (+ decoupled weight decay). Exercises the on-Param moment
state (m/v) persisting across steps, the bias-correction running powers, and the
GPU kernel.

Run: pixi run -e apple mojo run -I . tests/nn/test_adam_storage.mojo
"""

from std.math import sqrt
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.param import Param
from mojo_rl.nn.storage.optimizer.adam import Adam


comptime K = 16
comptime STEPS = 6
comptime LR = Scalar[DT](0.01)
comptime B1 = Scalar[DT](0.9)
comptime B2 = Scalar[DT](0.999)
comptime EPS = Scalar[DT](1e-8)
comptime WD = Scalar[DT](0.02)


def _grad(step: Int, i: Int) -> Scalar[DT]:
    return Scalar[DT](((step * 7 + i * 3) % 11) - 5) * 0.1


def _reference(decay: Bool) -> List[Scalar[DT]]:
    var p = List[Scalar[DT]](length=K, fill=Scalar[DT](0))
    var m = List[Scalar[DT]](length=K, fill=Scalar[DT](0))
    var v = List[Scalar[DT]](length=K, fill=Scalar[DT](0))
    for i in range(K):
        p[i] = Scalar[DT](i - 8) * 0.05  # init values
    var b1p = Scalar[DT](1.0)
    var b2p = Scalar[DT](1.0)
    for step in range(1, STEPS + 1):
        b1p *= B1
        b2p *= B2
        var bc1 = Scalar[DT](1.0) - b1p
        var bc2 = Scalar[DT](1.0) - b2p
        for i in range(K):
            var pp = p[i]
            if decay:
                pp -= LR * WD * pp
            var g = _grad(step, i)
            m[i] = B1 * m[i] + (Scalar[DT](1.0) - B1) * g
            v[i] = B2 * v[i] + (Scalar[DT](1.0) - B2) * g * g
            var mhat = m[i] / bc1
            var vhat = v[i] / bc2
            p[i] = pp - LR * mhat / (sqrt(vhat) + EPS)
    return p^


def _run[target: StaticString, decay: Bool](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-6) if target == "cpu" else Scalar[DT](1e-5)
    var p = Param["w", decay, K].make_cpu() if target == "cpu" else Param["w", decay, K].make_gpu(ctx.value())
    for i in range(K):
        p.val.data[i] = Scalar[DT](i - 8) * 0.05
    var opt = Adam(lr=LR, beta1=B1, beta2=B2, eps=EPS, wd=WD)

    comptime if target == "cpu":
        for step in range(1, STEPS + 1):
            for i in range(K):
                p.grd.data[i] = _grad(step, i)
            opt.begin_step()
            p.visit_with["cpu"](opt, None)
    else:
        var c = ctx.value()
        p.val.upload(c)
        for step in range(1, STEPS + 1):
            for i in range(K):
                p.grd.data[i] = _grad(step, i)
            p.grd.upload(c)
            opt.begin_step()
            p.visit_with["gpu"](opt, Optional(c))
        p.val.download(c)

    var refp = _reference(decay)
    var ok = True
    for i in range(K):
        if abs(p.val.data[i] - refp[i]) > TOL:
            ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("Adam storage optimizer reference parity")
    print("=" * 70)
    var c = DeviceContext()
    var ok = True
    var a = _run["cpu", True](None); print("  CPU  +decay:", "OK" if a else "FAIL"); ok = a and ok
    var b = _run["cpu", False](None); print("  CPU  -decay:", "OK" if b else "FAIL"); ok = b and ok
    var d = _run["gpu", True](Optional(c)); print("  GPU  +decay:", "OK" if d else "FAIL"); ok = d and ok
    var e = _run["gpu", False](Optional(c)); print("  GPU  -decay:", "OK" if e else "FAIL"); ok = e and ok
    if ok:
        print("ADAM OK")
    else:
        print("ADAM FAIL")
