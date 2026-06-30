"""Dropout storage primitive — CPU correctness (golden) + GPU reference-mask.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). Dropout draws a fresh
Bernoulli mask each forward, so the golden fingerprints (S = Σ vᵢ,
W = Σ vᵢ·(i+1) — the weight catches sign/position errors a plain sum would
cancel) are taken from the DETERMINISTIC paths:

  - CPU train mode: the leaf's per-instance Philox counter starts at 0 and the
    SEED is a comptime param, so the train-mode mask (and hence out / grad_in)
    is fully reproducible across runs. The golden captures out, grad_in, and the
    scaled mask itself.
  - eval mode: out == in (identity) and grad_in == grad_out (identity).

The GPU section is storage-only (no legacy dep): it cannot fingerprint-match the
CPU train mask (host vs device Philox draw different streams), so it verifies the
deterministic math against an independent recompute — out == in·mask,
grad_in == grad_out·mask, mask ∈ {0, 1/(1-p)}, plus eval identity.

Run: pixi run -e apple mojo run -I . tests/nn/test_dropout_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.dropout import Dropout


comptime DIM = 8
comptime B = 7
comptime P = 0.5


def _check(name: String, data: Tensor, n: Int,
           es: Scalar[DT], ew: Scalar[DT], tol: Scalar[DT]) -> Bool:
    """Assert tensor fingerprint (Σ vᵢ, Σ vᵢ·(i+1)) matches golden (es, ew)."""
    var s: Scalar[DT] = 0
    var w: Scalar[DT] = 0
    for i in range(n):
        s += data.data[i]
        w += data.data[i] * Scalar[DT](i + 1)
    var ok = abs(s - es) < tol and abs(w - ew) < tol
    print("  ", name, "S", s, "(exp", es, ") W", w, "(exp", ew, ")", "OK" if ok else "FAIL")
    return ok


def test_dropout_cpu_golden() raises:
    print("test_dropout_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](5e-3)
    comptime N = B * DIM

    # ── train mode: deterministic given fixed SEED + counter=0 ──
    var dp = Dropout[DIM, P, 1].make["cpu", Deterministic]()
    dp.set_training(True)
    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 11) - 5) * 0.13
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var out = Tensor.alloc(N)
    var gi = Tensor.alloc(N)
    dp.forward["cpu", B](TensorRefs[1](x), out, None)
    dp.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)

    var ok = _check("train_out", out, N, 0.0, 29.12, TOL)
    ok = _check("train_gi", gi, N, -3.6000001, -59.200004, TOL) and ok
    ok = _check("train_mask", dp.cache_mask, N, 56.0, 1634.0, TOL) and ok

    # ── eval mode: identity ──
    var dpe = Dropout[DIM, P, 1].make["cpu", Deterministic]()
    dpe.set_training(False)
    var xe = Tensor.alloc(N)
    var goe = Tensor.alloc(N)
    for i in range(N):
        xe.data[i] = Scalar[DT]((i % 13) - 6) * 0.17
        goe.data[i] = Scalar[DT]((i % 5) - 2) * 0.31
    var oute = Tensor.alloc(N)
    var gie = Tensor.alloc(N)
    dpe.forward["cpu", B](TensorRefs[1](xe), oute, None)
    dpe.vjp["cpu", B](TensorRefs[1](xe), goe, TensorRefs[1](gie), None)

    ok = _check("eval_out", oute, N, -3.06, -42.159996, TOL) and ok
    ok = _check("eval_gi", gie, N, -0.62, -0.61999416, TOL) and ok

    assert_true(ok, "Dropout CPU golden")
    print("  ok")


def _gpu_refmask() raises -> Bool:
    """Storage GPU reference-mask gate (no legacy dep): math vs recompute."""
    comptime TOL = Scalar[DT](1e-6)
    comptime N = B * DIM
    comptime SCALE = Scalar[DT](1.0 / (1.0 - P))
    var c = DeviceContext()

    # ── train mode: out==in·mask, gi==go·mask, mask ∈ {0, SCALE} ──
    var dp = Dropout[DIM, P, 1].make["gpu", Deterministic](Optional(c))
    dp.set_training(True)
    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 11) - 5) * 0.13
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var out = Tensor.alloc(N)
    var gi = Tensor.alloc(N)
    x.upload(c); go.upload(c)
    dp.forward["gpu", B](TensorRefs[1](x), out, Optional(c))
    dp.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))
    out.download(c); gi.download(c)
    dp.cache_mask.download(c)
    x.download(c); go.download(c)

    ref mask = dp.cache_mask.data
    var ok = True
    for i in range(N):
        var m = mask[i]
        var is_zero = abs(m) <= TOL
        var is_scale = abs(m - SCALE) <= TOL
        if not (is_zero or is_scale):
            ok = False
        if abs(out.data[i] - x.data[i] * m) > TOL:
            ok = False
        if abs(gi.data[i] - go.data[i] * m) > TOL:
            ok = False

    # ── eval mode: identity ──
    var dpe = Dropout[DIM, P, 1].make["gpu", Deterministic](Optional(c))
    dpe.set_training(False)
    var xe = Tensor.alloc(N)
    var goe = Tensor.alloc(N)
    for i in range(N):
        xe.data[i] = Scalar[DT]((i % 13) - 6) * 0.17
        goe.data[i] = Scalar[DT]((i % 5) - 2) * 0.31
    var oute = Tensor.alloc(N)
    var gie = Tensor.alloc(N)
    xe.upload(c); goe.upload(c)
    dpe.forward["gpu", B](TensorRefs[1](xe), oute, Optional(c))
    dpe.vjp["gpu", B](TensorRefs[1](xe), goe, TensorRefs[1](gie), Optional(c))
    oute.download(c); gie.download(c)
    xe.download(c); goe.download(c)
    for i in range(N):
        if abs(oute.data[i] - xe.data[i]) > TOL:
            ok = False
        if abs(gie.data[i] - goe.data[i]) > TOL:
            ok = False
    return ok


def test_dropout_gpu_refmask() raises:
    print("test_dropout_gpu_refmask (storage GPU vs recompute) ...")
    var ok = _gpu_refmask()
    assert_true(ok, "Dropout GPU reference-mask")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Dropout storage primitive (CPU golden + GPU reference-mask)")
    print("=" * 70)
    test_dropout_cpu_golden()
    test_dropout_gpu_refmask()
    print("ALL PASSED")
