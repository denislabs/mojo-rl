"""GaussianNLLLoss validation — analytic forward + analytic vjp + FD gradcheck.

Phase I.1.a.

Three sub-tests:

  1. **Forward analytic parity**: hand-computed loss per element matches
     module forward to ≤1e-6 across an interior + boundary case.
  2. **Vjp analytic parity**: hand-computed `d_loss/d_µ` and
     `d_loss/d_raw_logvar` (with clamp gating) match module vjp.
  3. **FD gradcheck**: per-element finite difference of `loss(logits)`
     matches the module's vjp to ≤1e-2 at eps=1e-2 (fp32 leaf tol).
     Touches both µ and raw_logvar columns; touches in-clamp + out-of-
     clamp logvars so the gating logic is exercised both ways.
"""

from std.math import exp
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.loss.gaussian_nll_loss import GaussianNLLLoss


comptime DIM = 3
comptime BATCH = 2
comptime IN = 2 * DIM       # logits: µ block then logvar block.
comptime N_LOG = BATCH * IN
comptime N_TGT = BATCH * DIM


def _clamp(v: Scalar[DT]) -> Tuple[Scalar[DT], Bool]:
    """Return (clamped, in_clamp_flag) using the default [-10, -2] bounds."""
    var lo = Scalar[DT](-10.0)
    var hi = Scalar[DT](-2.0)
    if v > hi:
        return (hi, False)
    if v < lo:
        return (lo, False)
    return (v, True)


def _ref_loss(logits_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
              tgt_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]) -> Scalar[DT]:
    var total = Scalar[DT](0.0)
    for b in range(BATCH):
        for i in range(DIM):
            var mu = logits_ptr[b * IN + i]
            var raw_lv = logits_ptr[b * IN + DIM + i]
            var y = tgt_ptr[b * DIM + i]
            var cl = _clamp(raw_lv)
            var lv = cl[0]
            var inv_var = exp(-lv)
            var d = mu - y
            total += Scalar[DT](0.5) * d * d * inv_var + Scalar[DT](0.5) * lv
    return total / Scalar[DT](BATCH)


def test_forward_analytic() raises:
    print("test_forward_analytic ...")
    var loss = GaussianNLLLoss[DIM].make[target="cpu"]()
    var logits: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_LOG)
    var tgt: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_TGT)
    # Row 0: in-clamp logvars (-5, -4, -3).
    # Row 1: out-of-clamp on both ends (-12 < -10, +1 > -2, -7 inside).
    var ls = [
        Scalar[DT](0.1), Scalar[DT](0.5), Scalar[DT](-0.3),
        Scalar[DT](-5.0), Scalar[DT](-4.0), Scalar[DT](-3.0),
        Scalar[DT](-0.2), Scalar[DT](0.7), Scalar[DT](0.0),
        Scalar[DT](-12.0), Scalar[DT](1.0), Scalar[DT](-7.0),
    ]
    var ts = [
        Scalar[DT](0.05), Scalar[DT](0.4), Scalar[DT](-0.5),
        Scalar[DT](-0.1), Scalar[DT](0.5), Scalar[DT](0.2),
    ]
    for i in range(N_LOG):
        logits[i] = ls[i]
    for i in range(N_TGT):
        tgt[i] = ts[i]

    var logits_t = TileTensor(logits, row_major[BATCH, IN]())
    var tgt_t = TileTensor(tgt, row_major[BATCH, DIM]())

    var got = loss.forward["cpu", BATCH](logits_t, tgt_t)
    var expected = _ref_loss(logits, tgt)
    print("  got=", got, " ref=", expected)
    var d = got - expected
    var ad = d if d >= Scalar[DT](0) else -d
    assert_true(
        ad < Scalar[DT](1e-6),
        "forward analytic parity failed",
    )
    print("  ok")


def test_vjp_analytic() raises:
    print("test_vjp_analytic ...")
    var loss = GaussianNLLLoss[DIM].make[target="cpu"]()
    var logits: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_LOG)
    var tgt: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_TGT)
    var grad: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_LOG)

    var ls = [
        Scalar[DT](0.1), Scalar[DT](0.5), Scalar[DT](-0.3),
        Scalar[DT](-5.0), Scalar[DT](-4.0), Scalar[DT](-3.0),
        Scalar[DT](-0.2), Scalar[DT](0.7), Scalar[DT](0.0),
        Scalar[DT](-12.0), Scalar[DT](1.0), Scalar[DT](-7.0),
    ]
    var ts = [
        Scalar[DT](0.05), Scalar[DT](0.4), Scalar[DT](-0.5),
        Scalar[DT](-0.1), Scalar[DT](0.5), Scalar[DT](0.2),
    ]
    for i in range(N_LOG):
        logits[i] = ls[i]
    for i in range(N_TGT):
        tgt[i] = ts[i]

    var logits_t = TileTensor(logits, row_major[BATCH, IN]())
    var tgt_t = TileTensor(tgt, row_major[BATCH, DIM]())
    var grad_t = TileTensor(grad, row_major[BATCH, IN]())

    _ = loss.forward["cpu", BATCH](logits_t, tgt_t)
    loss.vjp["cpu", BATCH](tgt_t, grad_t)

    var inv_b = Scalar[DT](1.0) / Scalar[DT](BATCH)
    var max_diff = Scalar[DT](0.0)
    for b in range(BATCH):
        for i in range(DIM):
            var mu = logits[b * IN + i]
            var raw_lv = logits[b * IN + DIM + i]
            var y = tgt[b * DIM + i]
            var cl = _clamp(raw_lv)
            var lv = cl[0]
            var ic = cl[1]
            var inv_v = exp(-lv)
            var d = mu - y
            var exp_g_mu = d * inv_v * inv_b
            var exp_g_lv = (
                Scalar[DT](0.5)
                - Scalar[DT](0.5) * d * d * inv_v
            ) * inv_b * (Scalar[DT](1.0) if ic else Scalar[DT](0.0))
            var got_g_mu = grad[b * IN + i]
            var got_g_lv = grad[b * IN + DIM + i]
            var d_mu = got_g_mu - exp_g_mu
            var ad_mu = d_mu if d_mu >= Scalar[DT](0) else -d_mu
            var d_lv = got_g_lv - exp_g_lv
            var ad_lv = d_lv if d_lv >= Scalar[DT](0) else -d_lv
            if ad_mu > max_diff:
                max_diff = ad_mu
            if ad_lv > max_diff:
                max_diff = ad_lv
    print("  max |grad - ref| =", max_diff)
    assert_true(
        max_diff < Scalar[DT](1e-6),
        "vjp analytic parity failed",
    )
    print("  ok")


def test_fd_gradcheck() raises:
    """FD on `loss(logits, targets)` w.r.t. each logit element.

    Uses fp32 single-leaf eps=1e-2 (see [[feedback_fd_eps_deep_chains]]).
    Crucially perturbs both µ and raw_logvar AND touches in-clamp +
    out-of-clamp logvars to verify gating direction."""
    print("test_fd_gradcheck ...")
    var eps = Scalar[DT](1e-2)
    var tol = Scalar[DT](1e-2)
    var loss = GaussianNLLLoss[DIM].make[target="cpu"]()
    var logits: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_LOG)
    var logits_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_LOG)
    var tgt: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_TGT)
    var grad: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N_LOG)

    var ls = [
        Scalar[DT](0.1), Scalar[DT](0.5), Scalar[DT](-0.3),
        Scalar[DT](-5.0), Scalar[DT](-4.0), Scalar[DT](-3.0),
        Scalar[DT](-0.2), Scalar[DT](0.7), Scalar[DT](0.0),
        Scalar[DT](-9.5), Scalar[DT](-2.5), Scalar[DT](-7.0),
    ]
    var ts = [
        Scalar[DT](0.05), Scalar[DT](0.4), Scalar[DT](-0.5),
        Scalar[DT](-0.1), Scalar[DT](0.5), Scalar[DT](0.2),
    ]
    for i in range(N_LOG):
        logits[i] = ls[i]
    for i in range(N_TGT):
        tgt[i] = ts[i]
    var tgt_t = TileTensor(tgt, row_major[BATCH, DIM]())

    var logits_t = TileTensor(logits, row_major[BATCH, IN]())
    var logits_p_t = TileTensor(logits_p, row_major[BATCH, IN]())
    var grad_t = TileTensor(grad, row_major[BATCH, IN]())

    _ = loss.forward["cpu", BATCH](logits_t, tgt_t)
    loss.vjp["cpu", BATCH](tgt_t, grad_t)

    var max_diff = Scalar[DT](0.0)
    for i in range(N_LOG):
        for j in range(N_LOG):
            logits_p[j] = logits[j]
        logits_p[i] = logits[i] + eps
        var lp = loss.forward["cpu", BATCH](logits_p_t, tgt_t)
        logits_p[i] = logits[i] - eps
        var lm = loss.forward["cpu", BATCH](logits_p_t, tgt_t)
        var fd = (lp - lm) / (Scalar[DT](2.0) * eps)
        var d = grad[i] - fd
        var ad = d if d >= Scalar[DT](0) else -d
        if ad > max_diff:
            max_diff = ad
    print("  max |grad - fd| =", max_diff, " (tol=", tol, ")")
    assert_true(
        max_diff < tol,
        "FD gradcheck failed",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("GaussianNLLLoss validation (Phase I.1.a)")
    print("=" * 70)
    test_forward_analytic()
    test_vjp_analytic()
    test_fd_gradcheck()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
