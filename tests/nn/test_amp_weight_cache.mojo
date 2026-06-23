"""Phase-1 AMP weight-cache invalidation (GPU).

Guards the EXACT legacy footgun: a cached bf16 weight that never invalidated, so
the net trained against a frozen cast (MNIST 97->59). This test validates the
CACHE MECHANISM (dtype-agnostic, so it holds on Apple too) via change-detection:
  (A) invalidation — after the param VERSION is bumped (what the optimizer does)
                     with NEW weights, the forward output CHANGES (recast fired).
  (B) reuse        — WITHOUT a version bump, a changed weight.val is IGNORED
                     (the cached cast is reused) → output is byte-identical.
  (C) integration  — Adam.step bumps weight.val.version (the wiring is live).

NOTE: bf16 NUMERIC correctness (bf16 GEMM ≈ fp32) is NOT asserted here because
`linalg.matmul` MIS-COMPUTES bf16 GEMMs on Apple Metal at realistic dims (verified:
inputs cast correctly + host bf16 dot = fp32, but the Metal kernel returns garbage
~148 for [8,64]@[64,128]; correct for tiny [4,8]@[8,4] and [1,4096]@[4096,1]). AMP
numeric parity is therefore a NVIDIA gate (cutlass bf16). The cache logic below is
independent of that and is exercised on whatever GPU is present.

Run: pixi run -e apple mojo run -I . tests/nn/test_amp_weight_cache.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.optimizer.adam import Adam


comptime IN = 64
comptime OUT = 128
comptime B = 8
comptime W = IN * OUT


def _maxdiff(a: Tensor, b: Tensor, n: Int) -> Scalar[DT]:
    var m: Scalar[DT] = 0
    for i in range(n):
        var d = abs(a.data[i] - b.data[i])
        if d > m:
            m = d
    return m


def _load_w(mut lin: Linear[IN, OUT, True], c: DeviceContext, seed: Int) raises:
    for k in range(W):
        lin.weight.val.data[k] = Scalar[DT](0.05 + 0.03 * Float64((k + seed) % 5))
    lin.weight.val.upload(c)


def main() raises:
    print("=" * 70)
    print("AMP weight-cache invalidation (Phase 1)")
    print("=" * 70)
    var c = DeviceContext()

    var amp = Linear[IN, OUT, True].make["gpu", Deterministic](Optional(c))
    for k in range(OUT):
        amp.bias.val.data[k] = Scalar[DT](0)
    amp.bias.val.upload(c)
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.1 + 0.05 * Float64(i % 7))
    x.upload(c)

    var y0 = Tensor.alloc(B * OUT)
    var y1 = Tensor.alloc(B * OUT)
    var y2 = Tensor.alloc(B * OUT)

    # baseline forward with W0 (force a cast via a version bump)
    _load_w(amp, c, 0)
    amp.weight.val.version += 1
    amp.forward["gpu", B](TensorRefs[1](x), y0, Optional(c))
    y0.download(c)

    # (A) invalidation: NEW weights W1 + version bump → output must CHANGE
    _load_w(amp, c, 2)
    amp.weight.val.version += 1   # what ParamVersionBump does on opt.step
    amp.forward["gpu", B](TensorRefs[1](x), y1, Optional(c))
    y1.download(c)
    var changed = _maxdiff(y1, y0, B * OUT)
    var a_ok = changed > Scalar[DT](0.01)
    print("  (A) invalidation  max|y1-y0| =", changed, "OK" if a_ok else "FAIL")

    # (B) reuse: NEW weights W2 but NO version bump → cache reused → IDENTICAL
    _load_w(amp, c, 4)
    amp.forward["gpu", B](TensorRefs[1](x), y2, Optional(c))
    y2.download(c)
    var same = _maxdiff(y2, y1, B * OUT)
    var b_ok = same == Scalar[DT](0)
    print("  (B) reuse         max|y2-y1| =", same, "OK" if b_ok else "FAIL")

    # (C) integration, per-param path: Adam.step bumps the version
    var m = Linear[IN, OUT, True].make["gpu", Deterministic](Optional(c))
    var v0 = m.weight.val.version
    var opt = Adam(lr=Scalar[DT](1e-3))
    opt.zero_grad["gpu"](m, Optional(c))
    opt.step["gpu"](m, Optional(c))
    var v1 = m.weight.val.version
    var c_ok = v1 == v0 + 1
    print("  (C) per-param     weight.version", v0, "->", v1, "OK" if c_ok else "FAIL")

    # (D) integration, ARENA path: adopt() + step bumps the version too (the
    # grouped kernel bypasses visit, so the bump rides a separate for_each_param
    # walk — this is the path NVIDIA training actually uses).
    var ma = Linear[IN, OUT, True].make["gpu", Deterministic](Optional(c))
    var optA = Adam(lr=Scalar[DT](1e-3))
    optA.adopt["gpu"](ma, Optional(c))
    var av0 = ma.weight.val.version
    optA.zero_grad["gpu"](ma, Optional(c))
    optA.step["gpu"](ma, Optional(c))
    var av1 = ma.weight.val.version
    var d_ok = av1 == av0 + 1
    print("  (D) arena         weight.version", av0, "->", av1, "OK" if d_ok else "FAIL")

    # (E) bf16 BACKWARD executes end-to-end (run-check only — Apple Metal bf16
    # GEMM numerics are garbage per the linalg bug, so parity is NVIDIA-gated;
    # here we just confirm the Phase-2 vjp path compiles + runs without crashing).
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](0.01 * Float64(i % 5))
    go.upload(c)
    var gi = Tensor.alloc(B * IN)
    amp.zero_grad["gpu"](Optional(c))
    amp.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))
    gi.download(c)
    amp.weight.grd.download(c)
    c.synchronize()
    print("  (E) bf16 vjp      executed (no crash) OK")

    assert_true(a_ok and b_ok and c_ok and d_ok, "AMP weight-cache invalidation")
    print("ALL PASSED")
