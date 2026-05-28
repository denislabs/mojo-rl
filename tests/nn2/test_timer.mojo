"""Test: Timer correctness + low-overhead invariants.

Verifies:
  - add_section grows times/counts/labels in lockstep
  - accumulate measures positive deltas, increments call count
  - mean_ms / total_seconds match the stored ns sum
  - reset zeros every section
  - format_report concatenates one labeled line per section
"""

from std.time import perf_counter_ns

from mojo_rl.nn2.training.timer import Timer
from mojo_rl.nn2.constants import DT


def test_add_and_accumulate() raises:
    print("test_add_and_accumulate ...")
    var timer = Timer.new()
    timer.add_section("target_y")
    timer.add_section("critic")
    timer.add_section("actor")
    if timer.n_sections() != 3:
        raise Error("expected 3 sections, got " + String(timer.n_sections()))

    # Spin a non-trivial amount of work, accumulate into section 0.
    # Use a side-effect chain perf_counter_ns()→k→…→k to defeat DCE,
    # so the loop body actually runs for measurable wall time.
    var t0 = perf_counter_ns()
    var k: Int = Int(t0 & 0xFF)  # seed from clock so loop is data-dependent
    for i in range(2_000_000):
        k = (k * 1103515245 + 12345 + i) & 0x7FFFFFFF
    timer.accumulate(0, t0)
    print("  section 0 after 2M-iter loop: " + String(timer.total_seconds(0)) + " s")

    if timer.call_count(0) != 1:
        raise Error("expected count(0) = 1")
    if timer.call_count(1) != 0:
        raise Error("expected count(1) = 0")
    if timer.total_seconds(0) <= Scalar[DT](0):
        raise Error(
            "expected non-zero total_seconds(0), got "
            + String(timer.total_seconds(0))
        )
    if timer.total_seconds(1) != Scalar[DT](0):
        raise Error("expected zero total_seconds(1)")

    # Accumulate another call into section 0 + one into section 2.
    var t1 = perf_counter_ns()
    for i in range(1_000_000):
        k = (k * 1103515245 + 12345 + i) & 0x7FFFFFFF
    timer.accumulate(0, t1)
    var t2 = perf_counter_ns()
    timer.accumulate(2, t2)

    if timer.call_count(0) != 2:
        raise Error("expected count(0) = 2 after second accumulate")
    if timer.call_count(2) != 1:
        raise Error("expected count(2) = 1")

    # Mean ms should equal total / count.
    var total_s_0 = timer.total_seconds(0)
    var mean_ms_0 = timer.mean_ms(0)
    var expected_mean_ms = total_s_0 * Scalar[DT](1000.0) / Scalar[DT](2)
    var diff = mean_ms_0 - expected_mean_ms
    if diff < Scalar[DT](-1e-6) or diff > Scalar[DT](1e-6):
        raise Error(
            "mean_ms / total_seconds mismatch: mean_ms=" + String(mean_ms_0)
            + ", expected=" + String(expected_mean_ms)
        )

    # Make sure k is used so the loop doesn't get DCE'd.
    if k <= 0:
        raise Error("loop body did not execute")
    print("  ok")


def test_reset() raises:
    print("test_reset ...")
    var timer = Timer.new()
    timer.add_section("a")
    timer.add_section("b")
    var t0 = perf_counter_ns()
    timer.accumulate(0, t0)
    timer.accumulate(1, t0)

    timer.reset()
    if timer.call_count(0) != 0 or timer.call_count(1) != 0:
        raise Error("reset should zero call counts")
    if timer.total_seconds(0) != Scalar[DT](0):
        raise Error("reset should zero total_seconds(0)")
    if timer.total_seconds(1) != Scalar[DT](0):
        raise Error("reset should zero total_seconds(1)")

    # Labels survive reset.
    if timer.n_sections() != 2:
        raise Error("reset should preserve section count")
    print("  ok")


def test_format_report() raises:
    print("test_format_report ...")
    var timer = Timer.new()
    timer.add_section("alpha")
    timer.add_section("beta")
    var t0 = perf_counter_ns()
    timer.accumulate(0, t0)

    var report = timer.format_report()
    # Sanity: two lines, each contains its label.
    var n_alpha = report.find("alpha")
    var n_beta = report.find("beta")
    if n_alpha < 0:
        raise Error("report missing 'alpha'")
    if n_beta < 0:
        raise Error("report missing 'beta'")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("Timer smoke test")
    print("=" * 60)
    test_add_and_accumulate()
    test_reset()
    test_format_report()
    print("ALL PASSED")
