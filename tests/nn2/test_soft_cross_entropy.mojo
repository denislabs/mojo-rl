"""SoftCrossEntropyLoss (Block D-3) — integration with TwoHot targets.

`SoftCrossEntropyLoss` is a naming alias for `CrossEntropyLoss[N_CLASSES]`
(which already accepts soft target distributions). This test verifies the
typical DreamerV3 / TD-MPC2 use-case end-to-end:

  1. Build two-hot targets from raw scalar rewards via symlog bins
  2. Run `SoftCrossEntropyLoss.forward` on uniformly-zero logits
     (analytical loss = log(N_BINS), since uniform softmax)
  3. Confirm `backward` gradient = (softmax - target) / BATCH
  4. Confirm FD gradcheck agrees with analytical backward
  5. Confirm parity with `CrossEntropyLoss` (it's the same struct)
"""

from std.math import abs as fabs, log
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.loss.soft_cross_entropy import SoftCrossEntropyLoss
from mojo_rl.nn2.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn2.loss.two_hot import (
    fill_symlog_bins_ptr,
    two_hot_encode_symlog_batch_ptr,
)


def test_uniform_logits_recovers_log_n() raises:
    """With logits = 0 everywhere, softmax = 1/N → L = -mean_i log(1/N) = log(N)."""
    comptime BATCH = 4
    comptime NUM = 41

    var bins_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](NUM)
    fill_symlog_bins_ptr[NUM](bins_p)

    var rewards_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    rewards_p[0] = 0.0
    rewards_p[1] = 1.0
    rewards_p[2] = -2.5
    rewards_p[3] = 10.0

    var targets_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    two_hot_encode_symlog_batch_ptr[BATCH, NUM](rewards_p, bins_p, targets_p)

    var logits_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    for k in range(BATCH * NUM):
        logits_p[k] = 0.0

    var logits_t  = TileTensor(logits_p, row_major[BATCH, NUM]())
    var targets_t = TileTensor(targets_p, row_major[BATCH, NUM]())

    var loss = SoftCrossEntropyLoss[NUM].make["cpu"]()
    var L = loss.forward["cpu", BATCH](logits_t, targets_t)

    var expected = log(Scalar[DT](NUM))  # ln(41) ≈ 3.713572
    print("  L = ", L, "  expected log(", NUM, ") = ", expected)
    assert_true(fabs(L - expected) < Scalar[DT](1e-5), "uniform-logit loss mismatch")

    bins_p.free()
    rewards_p.free()
    targets_p.free()
    logits_p.free()
    print("  test_uniform_logits_recovers_log_n PASSED")


def test_backward_matches_softmax_minus_target() raises:
    """Backward must produce (softmax - target) / BATCH per element.
    With uniform-zero logits and two-hot targets, this is (1/N - tgt)/B."""
    comptime BATCH = 3
    comptime NUM = 7

    var bins_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](NUM)
    # Linear bins for predictable encoding.
    fill_symlog_bins_ptr[NUM](bins_p)
    var rewards_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    rewards_p[0] = 1.0
    rewards_p[1] = 2.0
    rewards_p[2] = -1.0
    var targets_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    two_hot_encode_symlog_batch_ptr[BATCH, NUM](rewards_p, bins_p, targets_p)
    var logits_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    for k in range(BATCH * NUM):
        logits_p[k] = 0.0
    var grad_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    for k in range(BATCH * NUM):
        grad_p[k] = 0.0

    var logits_t  = TileTensor(logits_p, row_major[BATCH, NUM]())
    var targets_t = TileTensor(targets_p, row_major[BATCH, NUM]())
    var grad_t    = TileTensor(grad_p, row_major[BATCH, NUM]())

    var loss = SoftCrossEntropyLoss[NUM].make["cpu"]()
    _ = loss.forward["cpu", BATCH](logits_t, targets_t)
    loss.vjp["cpu", BATCH](targets_t, grad_t)

    var inv_n = Scalar[DT](1.0) / Scalar[DT](NUM)
    var inv_b = Scalar[DT](1.0) / Scalar[DT](BATCH)
    var max_err: Scalar[DT] = 0.0
    for b in range(BATCH):
        for c in range(NUM):
            var expected = (inv_n - targets_p[b * NUM + c]) * inv_b
            var err = fabs(grad_p[b * NUM + c] - expected)
            if err > max_err:
                max_err = err
    print("  max | grad - (softmax-target)/B | = ", max_err)
    assert_true(max_err < Scalar[DT](1e-6), "backward mismatch")

    bins_p.free()
    rewards_p.free()
    targets_p.free()
    logits_p.free()
    grad_p.free()
    print("  test_backward_matches_softmax_minus_target PASSED")


def test_fd_gradcheck() raises:
    """FD gradcheck of forward against backward.

    Uses absolute tolerance — soft-CE gradients are O(1/N/B) so the
    relative metric explodes whenever (1/N - target)/B is small.
    """
    comptime BATCH = 2
    comptime NUM = 5
    comptime EPS: Scalar[DT] = 1e-2
    comptime TOL_ABS: Scalar[DT] = 1e-3

    var loss = SoftCrossEntropyLoss[NUM].make["cpu"]()
    var logits_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    var targets_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    var grad_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    # Random-ish logits, soft (non-one-hot) targets.
    for k in range(BATCH * NUM):
        logits_p[k] = Scalar[DT](0.1 + 0.07 * Float64(k))
        grad_p[k] = 0.0
    # Targets must sum to 1 per row.
    for b in range(BATCH):
        var s: Scalar[DT] = 0.0
        for c in range(NUM):
            var t = Scalar[DT](0.1 + 0.05 * Float64(b * NUM + c))
            targets_p[b * NUM + c] = t
            s += t
        # Normalise.
        for c in range(NUM):
            targets_p[b * NUM + c] /= s

    var logits_t  = TileTensor(logits_p,  row_major[BATCH, NUM]())
    var targets_t = TileTensor(targets_p, row_major[BATCH, NUM]())
    var grad_t    = TileTensor(grad_p,    row_major[BATCH, NUM]())

    _ = loss.forward["cpu", BATCH](logits_t, targets_t)
    loss.vjp["cpu", BATCH](targets_t, grad_t)

    var max_abs: Scalar[DT] = 0.0
    for b in range(BATCH):
        for c in range(NUM):
            var saved = logits_p[b * NUM + c]
            logits_p[b * NUM + c] = saved + EPS
            var Lp = loss.forward["cpu", BATCH](logits_t, targets_t)
            logits_p[b * NUM + c] = saved - EPS
            var Lm = loss.forward["cpu", BATCH](logits_t, targets_t)
            logits_p[b * NUM + c] = saved
            var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
            var an = grad_p[b * NUM + c]
            var diff = fabs(fd - an)
            if diff > max_abs:
                max_abs = diff

    print("  SoftCE FD gradcheck max_abs = ", max_abs)
    assert_true(max_abs < TOL_ABS, "FD gradcheck failed (abs)")

    logits_p.free()
    targets_p.free()
    grad_p.free()
    print("  test_fd_gradcheck PASSED")


def test_alias_parity() raises:
    """SoftCrossEntropyLoss must equal CrossEntropyLoss element-for-element
    (it's the same struct via `alias`)."""
    comptime BATCH = 2
    comptime NUM = 4
    var loss_a = SoftCrossEntropyLoss[NUM].make["cpu"]()
    var loss_b = CrossEntropyLoss[NUM].make["cpu"]()

    var logits_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    var targets_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    for k in range(BATCH * NUM):
        logits_p[k] = Scalar[DT](0.5 - 0.1 * Float64(k))
        targets_p[k] = Scalar[DT](1.0 / Float64(NUM))  # uniform
    var logits_t  = TileTensor(logits_p,  row_major[BATCH, NUM]())
    var targets_t = TileTensor(targets_p, row_major[BATCH, NUM]())

    var La = loss_a.forward["cpu", BATCH](logits_t, targets_t)
    var Lb = loss_b.forward["cpu", BATCH](logits_t, targets_t)
    assert_true(La == Lb, "alias diverges from CrossEntropyLoss")

    logits_p.free()
    targets_p.free()
    print("  test_alias_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 SoftCrossEntropyLoss tests (Block D-3)")
    print("=" * 60)
    test_uniform_logits_recovers_log_n()
    test_backward_matches_softmax_minus_target()
    test_fd_gradcheck()
    test_alias_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
