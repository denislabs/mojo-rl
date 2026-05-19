"""CrossEntropyLoss CPU tests — Phase 1.

Covers:
  - Uniform-logit: softmax is uniform; CE = log(N_CLASSES) for any one-hot
  - Confident-correct prediction: CE → ~0
  - Confident-wrong prediction: CE → large positive
  - Backward: grad_logits = (softmax - target) / BATCH
  - Backward sums to zero over classes (probability simplex constraint)
"""

from std.math import abs as fabs, log as flog
from std.memory import alloc
from std.testing import assert_equal, assert_almost_equal
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.loss import CrossEntropyLoss


def test_forward_uniform_logits() raises:
    """Uniform logits → uniform softmax → CE = log(N_CLASSES)."""
    comptime N = 4
    comptime BATCH = 2
    var loss = CrossEntropyLoss[N].make["cpu"]()

    var lg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var tg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    for k in range(BATCH * N):
        lg_buf[k] = 0.0     # uniform
        tg_buf[k] = 0.0
    # one-hot targets: sample 0 → class 1; sample 1 → class 3
    tg_buf[0 * N + 1] = 1.0
    tg_buf[1 * N + 3] = 1.0

    var logits  = TileTensor(lg_buf, row_major[BATCH, N]())
    var targets = TileTensor(tg_buf, row_major[BATCH, N]())

    var L = loss.forward["cpu", BATCH](logits, targets)
    # Expected: CE per sample = log(N); averaged still log(N) = log(4)
    assert_almost_equal(L, Scalar[DT](flog(Scalar[DT](N))), atol=1e-6)

    # Softmax cache should be uniform: each entry = 1/N
    var sm = TileTensor(loss.softmax, row_major[BATCH, N]())
    for b in range(BATCH):
        for c in range(N):
            assert_almost_equal(sm[b, c], 1.0 / Scalar[DT](N), atol=1e-6)

    lg_buf.free()
    tg_buf.free()
    print("  test_forward_uniform_logits PASSED")


def test_forward_confident_correct() raises:
    """Strongly positive logit on true class → CE → 0."""
    comptime N = 3
    comptime BATCH = 1
    var loss = CrossEntropyLoss[N].make["cpu"]()

    var lg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var tg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    # Logits = [10, 0, 0] → softmax dominated by class 0
    lg_buf[0] = 10.0; lg_buf[1] = 0.0; lg_buf[2] = 0.0
    tg_buf[0] = 1.0;  tg_buf[1] = 0.0; tg_buf[2] = 0.0

    var logits  = TileTensor(lg_buf, row_major[BATCH, N]())
    var targets = TileTensor(tg_buf, row_major[BATCH, N]())
    var L = loss.forward["cpu", BATCH](logits, targets)

    # softmax[0] ≈ 1, log_softmax[0] ≈ 0, CE ≈ 0
    assert_almost_equal(L, 0.0, atol=1e-3)

    lg_buf.free()
    tg_buf.free()
    print("  test_forward_confident_correct PASSED")


def test_forward_confident_wrong() raises:
    """Strongly negative logit on true class → CE large."""
    comptime N = 3
    comptime BATCH = 1
    var loss = CrossEntropyLoss[N].make["cpu"]()

    var lg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var tg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    # Logits = [0, 0, 10] but true class = 0
    lg_buf[0] = 0.0; lg_buf[1] = 0.0; lg_buf[2] = 10.0
    tg_buf[0] = 1.0; tg_buf[1] = 0.0; tg_buf[2] = 0.0

    var logits  = TileTensor(lg_buf, row_major[BATCH, N]())
    var targets = TileTensor(tg_buf, row_major[BATCH, N]())
    var L = loss.forward["cpu", BATCH](logits, targets)

    # CE ≈ -log(softmax[0]) ≈ -log(exp(-10)/something) ≈ 10
    # (more precisely: log(1 + 2*exp(-10)) + 10 - 0 = 10 + log(1 + 2e-10) ≈ 10)
    # so L should be roughly 10
    assert_almost_equal(L, 10.0, atol=0.1)

    lg_buf.free()
    tg_buf.free()
    print("  test_forward_confident_wrong PASSED")


def test_backward_softmax_minus_target() raises:
    """grad_logits[b, c] = (softmax[b, c] - target[b, c]) / BATCH."""
    comptime N = 3
    comptime BATCH = 2
    var loss = CrossEntropyLoss[N].make["cpu"]()

    var lg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var tg_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    for k in range(BATCH * N):
        lg_buf[k] = 0.0     # uniform softmax = 1/3
        tg_buf[k] = 0.0
    tg_buf[0 * N + 1] = 1.0    # sample 0 → class 1
    tg_buf[1 * N + 0] = 1.0    # sample 1 → class 0

    var logits  = TileTensor(lg_buf, row_major[BATCH, N]())
    var targets = TileTensor(tg_buf, row_major[BATCH, N]())
    _ = loss.forward["cpu", BATCH](logits, targets)

    var grad_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var grad_logits = TileTensor(grad_buf, row_major[BATCH, N]())
    loss.backward["cpu", BATCH](targets, grad_logits)

    # softmax is uniform 1/3. grad = (1/3 - target) / BATCH.
    # Sample 0: target = [0, 1, 0] → grad = [1/3, 1/3 - 1, 1/3] / 2 = [1/6, -1/3, 1/6]
    # Sample 1: target = [1, 0, 0] → grad = [1/3 - 1, 1/3, 1/3] / 2 = [-1/3, 1/6, 1/6]
    var third = 1.0 / 3.0
    var sixth = 1.0 / 6.0
    var minus_third = -1.0 / 3.0
    assert_almost_equal(grad_logits[0, 0], Scalar[DT](sixth),       atol=1e-6)
    assert_almost_equal(grad_logits[0, 1], Scalar[DT](minus_third), atol=1e-6)
    assert_almost_equal(grad_logits[0, 2], Scalar[DT](sixth),       atol=1e-6)
    assert_almost_equal(grad_logits[1, 0], Scalar[DT](minus_third), atol=1e-6)
    assert_almost_equal(grad_logits[1, 1], Scalar[DT](sixth),       atol=1e-6)
    assert_almost_equal(grad_logits[1, 2], Scalar[DT](sixth),       atol=1e-6)

    # Sanity: grad_logits sums to 0 over classes for each batch sample
    # (softmax sums to 1, target sums to 1, so sum(softmax - target) = 0).
    for b in range(BATCH):
        var s: Scalar[DT] = 0.0
        for c in range(N):
            s += grad_logits[b, c]
        assert_almost_equal(s, 0.0, atol=1e-6)

    lg_buf.free()
    tg_buf.free()
    grad_buf.free()
    print("  test_backward_softmax_minus_target PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 CrossEntropyLoss unit tests (CPU, Phase 1)")
    print("=" * 60)
    test_forward_uniform_logits()
    test_forward_confident_correct()
    test_forward_confident_wrong()
    test_backward_softmax_minus_target()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
