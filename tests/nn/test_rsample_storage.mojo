"""RSample storage — reference-noise gate (CPU + GPU).

RSample draws fresh reparam noise each forward, so we read back the ACTUAL drawn
z and verify the squashed-Gaussian forward (action + log_prob) and backward
(grad wrt [mu|log_std]) against an independent reference computed from that z.

Run: pixi run -e apple mojo run -I . tests/nn/test_rsample_storage.mojo
"""

from std.math import exp, log, tanh as ftanh
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.rsample import (
    RSample, _clamp_ls, LOG_STD_MIN, LOG_STD_MAX, EPS_TANH_CORR, LOG_2PI,
)


comptime ACT = 3
comptime B = 5
comptime AO = 2 * ACT
comptime OUT = ACT + 1
comptime SCALE = Scalar[DT](2.0)


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-5) if target == "cpu" else Scalar[DT](2e-5)
    var r = RSample[ACT].make[target, Deterministic](ctx)
    r.action_scale = SCALE

    var ao = Tensor.alloc(B * AO)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * AO):
        ao.data[i] = Scalar[DT]((i % 9) - 4) * 0.3   # spans clamp range on log_std
    for i in range(B * OUT):
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.15
    var out = Tensor.alloc(B * OUT)
    var gi = Tensor.alloc(B * AO)

    comptime if target == "cpu":
        r.forward["cpu", B](TensorRefs[1](ao), out, None)
        r.vjp["cpu", B](TensorRefs[1](ao), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        ao.upload(c); go.upload(c)
        r.forward["gpu", B](TensorRefs[1](ao), out, ctx)
        r.vjp["gpu", B](TensorRefs[1](ao), go, TensorRefs[1](gi), ctx)
        out.download(c); gi.download(c)
        r.z.download(c)

    ref z = r.z.data
    var ok = True
    for b in range(B):
        var lp_ref: Scalar[DT] = 0.0
        for j in range(ACT):
            var mu = ao.data[b * AO + j]
            var ls = _clamp_ls(ao.data[b * AO + ACT + j])
            var std = exp(ls)
            var zj = z[b * ACT + j]
            var y = ftanh(mu + std * zj)
            var a_ref = SCALE * y
            if abs(out.data[b * OUT + j] - a_ref) > TOL: ok = False
            var corr = SCALE * (Scalar[DT](1.0) - y * y) + EPS_TANH_CORR
            lp_ref += (
                Scalar[DT](-0.5) * zj * zj - ls
                - Scalar[DT](0.5) * LOG_2PI - log(corr)
            )
        if abs(out.data[b * OUT + ACT] - lp_ref) > TOL: ok = False
        # backward reference
        var glp = go.data[b * OUT + ACT]
        for j in range(ACT):
            var mu = ao.data[b * AO + j]
            var ls_raw = ao.data[b * AO + ACT + j]
            var ls = _clamp_ls(ls_raw)
            var clamped = (ls_raw < LOG_STD_MIN) or (ls_raw > LOG_STD_MAX)
            var std = exp(ls)
            var zj = z[b * ACT + j]
            var y = ftanh(mu + std * zj)
            var c_om = SCALE * (Scalar[DT](1.0) - y * y)
            var corr = c_om + EPS_TANH_CORR
            var ga = go.data[b * OUT + j]
            var gmu_ref = ga * c_om + glp * (Scalar[DT](2.0) * y * c_om) / corr
            if abs(gi.data[b * AO + j] - gmu_ref) > TOL: ok = False
            var gls_ref: Scalar[DT] = 0.0
            if not clamped:
                gls_ref = ga * (c_om * zj * std) + glp * (
                    Scalar[DT](-1.0) + (Scalar[DT](2.0) * y * c_om * zj * std) / corr
                )
            if abs(gi.data[b * AO + ACT + j] - gls_ref) > TOL: ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("RSample storage reference-noise gate")
    print("=" * 70)
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "RSample reference parity")
    print("RSAMPLE OK")
