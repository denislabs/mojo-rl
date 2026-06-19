"""Dropout storage — reference-mask gate (CPU + GPU).

Dropout draws a fresh Bernoulli mask each forward, so a legacy↔storage
bit-identity test is impossible across RNG. Instead: run forward, read back the
ACTUAL scaled mask the leaf drew (`cache_mask`, 0 or 1/(1-p)), and verify the
deterministic math against an independent recompute:

  - forward:  out[i] == in[i] * mask[i]  (survivors scaled by 1/(1-p), dropped 0)
  - vjp:      grad_in[i] == grad_out[i] * mask[i]  (grad gated by SAME mask)
  - mask invariant: each entry is exactly 0 or 1/(1-p)
  - eval mode: out == in (identity) AND grad_in == grad_out (identity)

This gates the porting risk (the math/plumbing); the uniform SAMPLER (host /
device PhiloxRandom) is carried over verbatim.

Run: pixi run -e apple mojo run -I . tests/nn/test_dropout_storage_parity.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.dropout import Dropout


comptime DIM = 8
comptime B = 7
comptime P = 0.5


def _check_train[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-6)
    comptime N = B * DIM
    comptime SCALE = Scalar[DT](1.0 / (1.0 - P))
    var dp = Dropout[DIM, P, 1].make[target, Deterministic](ctx)
    dp.set_training(True)

    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 11) - 5) * 0.13
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var out = Tensor.alloc(N)
    var gi = Tensor.alloc(N)

    comptime if target == "cpu":
        dp.forward["cpu", B](TensorRefs[1](x), out, None)
        dp.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        x.upload(c); go.upload(c)
        dp.forward["gpu", B](TensorRefs[1](x), out, ctx)
        dp.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        out.download(c); gi.download(c)
        dp.cache_mask.download(c)

    ref mask = dp.cache_mask.data
    var ok = True
    for i in range(N):
        var m = mask[i]
        # mask invariant: exactly 0 or SCALE
        var is_zero = abs(m) <= TOL
        var is_scale = abs(m - SCALE) <= TOL
        if not (is_zero or is_scale):
            ok = False
        # forward: out == in * mask
        if abs(out.data[i] - x.data[i] * m) > TOL:
            ok = False
        # vjp: grad_in == grad_out * mask
        if abs(gi.data[i] - go.data[i] * m) > TOL:
            ok = False
    return ok


def _check_eval[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-7)
    comptime N = B * DIM
    var dp = Dropout[DIM, P, 1].make[target, Deterministic](ctx)
    dp.set_training(False)

    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 13) - 6) * 0.17
        go.data[i] = Scalar[DT]((i % 5) - 2) * 0.31
    var out = Tensor.alloc(N)
    var gi = Tensor.alloc(N)

    comptime if target == "cpu":
        dp.forward["cpu", B](TensorRefs[1](x), out, None)
        dp.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        x.upload(c); go.upload(c)
        dp.forward["gpu", B](TensorRefs[1](x), out, ctx)
        dp.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        out.download(c); gi.download(c)

    var ok = True
    for i in range(N):
        if abs(out.data[i] - x.data[i]) > TOL:
            ok = False  # eval forward == identity
        if abs(gi.data[i] - go.data[i]) > TOL:
            ok = False  # eval vjp == identity
    return ok


def main() raises:
    print("=" * 70)
    print("Dropout storage reference-mask gate")
    print("=" * 70)
    var ok_cpu_t = _check_train["cpu"](None)
    var ok_cpu_e = _check_eval["cpu"](None)
    print("  CPU train:", "OK" if ok_cpu_t else "FAIL")
    print("  CPU eval :", "OK" if ok_cpu_e else "FAIL")
    var c = DeviceContext()
    var ok_gpu_t = _check_train["gpu"](Optional(c))
    var ok_gpu_e = _check_eval["gpu"](Optional(c))
    print("  GPU train:", "OK" if ok_gpu_t else "FAIL")
    print("  GPU eval :", "OK" if ok_gpu_e else "FAIL")
    var all_ok = ok_cpu_t and ok_cpu_e and ok_gpu_t and ok_gpu_e
    assert_true(all_ok, "Dropout reference-mask parity")
    print("DROPOUT OK")
