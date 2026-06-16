"""TwoHot helpers (Block D-2).

Covers:
  * `compute_bins` / `compute_symlog_bins` produce monotonic ascending
    arrays with the expected endpoints.
  * `two_hot_encode` sums to 1.0 on interior values, places weight on two
    adjacent bins, and clamps at the edges.
  * `decode_value` applied to a two-hot target round-trips back to the
    original scalar (modulo symexp clamping).
  * Batched pointer-form encode + decode agree with the InlineArray form.
"""

from std.math import abs as fabs, log as _log
from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.loss.two_hot import (
    compute_bins,
    compute_symlog_bins,
    fill_bins_ptr,
    fill_symlog_bins_ptr,
    two_hot_encode,
    two_hot_encode_batch_ptr,
    two_hot_encode_symlog_batch_ptr,
    decode_value,
    decode_value_batch_ptr,
    decode_value_batch_linear_ptr,
    symlog,
    symexp,
)


def test_compute_bins_linear() raises:
    comptime NUM = 5
    var bins = compute_bins[NUM](Scalar[DT](-2.0), Scalar[DT](2.0))
    var expected_step = Scalar[DT](1.0)
    for i in range(NUM):
        var want = Scalar[DT](-2.0) + expected_step * Scalar[DT](i)
        assert_true(fabs(bins[i] - want) < 1e-6, "linear bin mismatch")
    print("  test_compute_bins_linear PASSED")


def test_compute_bins_symlog() raises:
    comptime NUM = 41  # DreamerV3 default
    var bins = compute_symlog_bins[NUM]()
    assert_true(fabs(bins[0] - Scalar[DT](-20.0)) < 1e-6)
    assert_true(fabs(bins[NUM - 1] - Scalar[DT](20.0)) < 1e-6)
    assert_true(fabs(bins[NUM // 2]) < 1e-6)  # symmetric around 0
    print("  test_compute_bins_symlog PASSED")


def test_two_hot_basic() raises:
    """x sitting exactly between two bins should split 50/50."""
    comptime NUM = 5
    var bins = compute_bins[NUM](Scalar[DT](0.0), Scalar[DT](4.0))  # 0,1,2,3,4
    var target = InlineArray[Scalar[DT], NUM](fill=0)
    two_hot_encode[NUM](Scalar[DT](1.5), bins, target)
    # bin 1 at x=1, bin 2 at x=2. width=1. upper_weight = (2 - 1.5)/1 = 0.5.
    assert_true(fabs(target[0]) < 1e-6)
    assert_true(fabs(target[1] - 0.5) < 1e-6)
    assert_true(fabs(target[2] - 0.5) < 1e-6)
    assert_true(fabs(target[3]) < 1e-6)
    assert_true(fabs(target[4]) < 1e-6)

    var s = Scalar[DT](0.0)
    for i in range(NUM):
        s += target[i]
    assert_true(fabs(s - 1.0) < 1e-6, "two-hot should sum to 1")
    print("  test_two_hot_basic PASSED")


def test_two_hot_edge_clamp() raises:
    """x beyond v_max should clamp into the last interval."""
    comptime NUM = 4
    var bins = compute_bins[NUM](Scalar[DT](-1.0), Scalar[DT](2.0))
    var target = InlineArray[Scalar[DT], NUM](fill=0)
    two_hot_encode[NUM](Scalar[DT](100.0), bins, target)
    # x_clamped = 2.0 → last bin gets all weight.
    assert_true(fabs(target[NUM - 1] - 1.0) < 1e-6)
    assert_true(fabs(target[NUM - 2]) < 1e-6)
    print("  test_two_hot_edge_clamp PASSED")


def test_round_trip_symlog() raises:
    """Encode x → two-hot via softmax-decode round-trips to (approximately)
    x via symexp(weighted-sum-of-bins). With logits = log(target), softmax
    recovers target exactly."""
    comptime NUM = 41
    var bins = compute_symlog_bins[NUM]()
    var target = InlineArray[Scalar[DT], NUM](fill=0)

    var x = Scalar[DT](3.0)
    var x_symlog = symlog(x)
    two_hot_encode[NUM](x_symlog, bins, target)

    # Build pseudo-logits: pretend the network is perfectly confident and
    # outputs log(probability). softmax(log(p)) = p (up to constants).
    var logits = InlineArray[Scalar[DT], NUM](fill=-30.0)  # near 0 prob
    for i in range(NUM):
        if target[i] > 1e-6:
            # log p — clipped at -30 elsewhere keeps the softmax sane.
            logits[i] = _log(target[i])

    var decoded = decode_value[NUM](logits, bins)
    print("  x = ", x, "  decoded = ", decoded)
    assert_true(fabs(decoded - x) < Scalar[DT](0.05), "round-trip mismatch")
    print("  test_round_trip_symlog PASSED")


def test_batch_pointer_form() raises:
    """Pointer-batched encode must agree with the scalar form."""
    comptime BATCH = 4
    comptime NUM = 5

    var bins_arr = compute_bins[NUM](Scalar[DT](0.0), Scalar[DT](4.0))
    var bins_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](NUM)
    fill_bins_ptr[NUM](Scalar[DT](0.0), Scalar[DT](4.0), bins_ptr)
    for i in range(NUM):
        assert_true(fabs(bins_arr[i] - bins_ptr[i]) < 1e-6)

    var values_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var targets_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    for b in range(BATCH):
        values_ptr[b] = Scalar[DT](0.3 + 0.7 * Float64(b))  # 0.3, 1.0, 1.7, 2.4

    two_hot_encode_batch_ptr[BATCH, NUM](values_ptr, bins_ptr, targets_ptr)

    # Compare per-sample to scalar form.
    for b in range(BATCH):
        var ref_target = InlineArray[Scalar[DT], NUM](fill=0)
        two_hot_encode[NUM](values_ptr[b], bins_arr, ref_target)
        for i in range(NUM):
            assert_true(
                fabs(targets_ptr[b * NUM + i] - ref_target[i]) < 1e-6,
                "batched encode mismatch",
            )

    # Decode round-trip (linear bins, no symexp).
    var logits_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    for b in range(BATCH):
        for i in range(NUM):
            var t = targets_ptr[b * NUM + i]
            logits_ptr[b * NUM + i] = (
                _log(t) if t > 1e-6 else Scalar[DT](-30.0)
            )
    var decoded_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    decode_value_batch_linear_ptr[BATCH, NUM](
        logits_ptr, bins_ptr, decoded_ptr
    )
    for b in range(BATCH):
        assert_true(
            fabs(decoded_ptr[b] - values_ptr[b]) < Scalar[DT](1e-4),
            "linear decode round-trip mismatch",
        )

    bins_ptr.free()
    values_ptr.free()
    targets_ptr.free()
    logits_ptr.free()
    decoded_ptr.free()
    print("  test_batch_pointer_form PASSED")


def test_symlog_batch_encode() raises:
    """Symlog-batched encode applies symlog to each value and encodes
    against symlog-spaced bins. Decoded round-trip via symexp should
    recover original scale."""
    comptime BATCH = 3
    comptime NUM = 41
    var bins_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](NUM)
    fill_symlog_bins_ptr[NUM](bins_ptr)

    var values_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    values_ptr[0] = 0.5
    values_ptr[1] = 10.0
    values_ptr[2] = -3.0

    var targets_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    two_hot_encode_symlog_batch_ptr[BATCH, NUM](
        values_ptr, bins_ptr, targets_ptr
    )

    # Build idealized logits from the soft targets.
    var logits_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * NUM)
    for k in range(BATCH * NUM):
        var tv = targets_ptr[k]
        logits_ptr[k] = _log(tv) if tv > 1e-6 else Scalar[DT](-30.0)
    var decoded_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    decode_value_batch_ptr[BATCH, NUM](logits_ptr, bins_ptr, decoded_ptr)
    for b in range(BATCH):
        print(
            "  symlog round-trip b=", b,
            " val=", values_ptr[b],
            " decoded=", decoded_ptr[b],
        )
        # Loose tolerance: discrete bins introduce quantization. ~5% OK
        # for values within [-20, 20] symlog range.
        var rel = fabs(decoded_ptr[b] - values_ptr[b]) / (
            fabs(values_ptr[b]) + Scalar[DT](1e-3)
        )
        assert_true(rel < Scalar[DT](0.1), "symlog round-trip diverged")

    bins_ptr.free()
    values_ptr.free()
    targets_ptr.free()
    logits_ptr.free()
    decoded_ptr.free()
    print("  test_symlog_batch_encode PASSED")


def test_symlog_symexp_round_trip() raises:
    """Scalar symlog/symexp helpers must be exact inverses."""
    var vs = InlineArray[Scalar[DT], 6](fill=0)
    vs[0] = 0.0
    vs[1] = 1.0
    vs[2] = -1.0
    vs[3] = 5.0
    vs[4] = -100.0
    vs[5] = 1e-3
    for i in range(6):
        var x = vs[i]
        var rt = symexp(symlog(x))
        assert_true(fabs(rt - x) < Scalar[DT](1e-3), "symlog/symexp not invertible")
    print("  test_symlog_symexp_round_trip PASSED")


def main() raises:
    print("=" * 60)
    print("nn TwoHot tests (Block D-2)")
    print("=" * 60)
    test_compute_bins_linear()
    test_compute_bins_symlog()
    test_two_hot_basic()
    test_two_hot_edge_clamp()
    test_round_trip_symlog()
    test_batch_pointer_form()
    test_symlog_batch_encode()
    test_symlog_symexp_round_trip()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
