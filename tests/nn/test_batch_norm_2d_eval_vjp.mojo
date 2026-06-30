"""BatchNorm2D eval-mode (running-stat) vjp — CPU finite-difference gate.

The storage `BatchNorm2D` historically only implemented a TRAINING-mode backward
(batch-stat differentiation); `vjp` raised in eval mode. A frozen feature
extractor used as a perceptual loss must backprop through BN in EVAL mode, using
its running stats as constants. The eval-mode input gradient is then simply

    y = γ·(x − running_mean)·inv_std + β,   inv_std = 1/√(running_var+ε)
    ∂L/∂x = γ·inv_std·∂L/∂y                 (running stats are constants)

i.e. BN is *linear* in x in eval mode, so a central finite difference matches the
analytic vjp to near machine precision. This gate locks that path in (CPU).

Run: pixi run mojo run -I . tests/nn/test_batch_norm_2d_eval_vjp.mojo
"""

from std.math import sqrt, abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D


def _scalar_loss[C: Int, H: Int, W: Int, B: Int](
    mut bn: BatchNorm2D[C, H, W],
    mut x: Tensor,
    w: Tensor,
    mut out: Tensor,
) raises -> Float64:
    # L(x) = Σ_i w_i · BN_eval(x)_i  → ∂L/∂x = vjp(w).
    bn.forward["cpu", B](TensorRefs[1](x), out, None)
    var s: Float64 = 0.0
    for i in range(len(out.data)):
        s += Float64(out.data[i]) * Float64(w.data[i])
    return s


def main() raises:
    print("BatchNorm2D eval-mode vjp finite-difference gate (CPU)")
    comptime C = 3
    comptime H = 2
    comptime W = 2
    comptime B = 4
    comptime FLAT = C * H * W
    comptime N = B * FLAT

    var bn = BatchNorm2D[C, H, W].make["cpu", Deterministic](None)

    # Give the running stats non-trivial (but well-conditioned) values so the
    # eval normalization is a genuine affine map, not the identity.
    for c in range(C):
        bn.running_mean.t.data[c] = Scalar[DT](0.1 * Float64(c) - 0.1)
        bn.running_var.t.data[c] = Scalar[DT](0.5 + 0.3 * Float64(c))
    bn.set_attr["training"](Scalar[DT](0.0))  # EVAL

    var x = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT](((i * 7 + 3) % 11) - 5) * 0.17
    var w = Tensor.alloc(N)
    for i in range(N):
        w.data[i] = Scalar[DT](((i * 5 + 1) % 7) - 3) * 0.21

    var out = Tensor.alloc(N)
    bn.zero_grad["cpu"](None)
    _ = _scalar_loss[C, H, W, B](bn, x, w, out)

    # analytic input grad = vjp(w)
    var gi = Tensor.alloc(N)
    bn.vjp["cpu", B](TensorRefs[1](x), w, TensorRefs[1](gi), None)

    var eps = 1.0e-3
    var max_rel = 0.0
    for j in range(N):
        var saved = x.data[j]
        x.data[j] = saved + Scalar[DT](eps)
        var lp = _scalar_loss[C, H, W, B](bn, x, w, out)
        x.data[j] = saved - Scalar[DT](eps)
        var lm = _scalar_loss[C, H, W, B](bn, x, w, out)
        x.data[j] = saved
        var fd = (lp - lm) / (2.0 * eps)
        var an = Float64(gi.data[j])
        var rel = abs(fd - an) / (abs(fd) + abs(an) + 1.0e-6)
        if rel > max_rel:
            max_rel = rel
    print("  max rel error (analytic vs finite-diff) =", max_rel)
    var ok = max_rel < 1.0e-3
    print("  eval-mode BN vjp matches finite difference:", "OK" if ok else "FAIL")
    assert_true(ok, "BatchNorm2D eval-mode vjp finite-difference")
    print("BATCHNORM2D EVAL VJP GATE OK")
