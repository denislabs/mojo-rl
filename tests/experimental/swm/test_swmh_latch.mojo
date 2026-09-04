"""G12 — SWM Phase 4 gate: the classification loop must not oscillate.

The classification feeds `in_energy`, which changes the energy, which changes
what is learned, which changes the holonomy, which changes the classification.
That is a closed loop with a threshold in it, and a bare threshold chatters
whenever a residual sits near its boundary. Each flip re-admits or re-removes a
constraint from the inference, so the chatter is not cosmetic — it is the
inference alternately being forced into a cycle's fixed subspace and released
(see G11 for what that does).

This gate drives a residual stream that hovers at the outlier boundary — the
worst case, and the realistic one while an encoder is still converging — and
compares churn with and without the latch.

Validates:
  - the latch cuts verdict changes by a large factor on a boundary-hovering
    stream
  - it does NOT pin the verdict: once the evidence settles cleanly, the latched
    verdict is the correct one. A latch that could not be moved would score
    perfectly on churn and be useless, so this leg is what stops the metric
    being gamed by its own mechanism.
  - NEGATIVE CONTROL: on a stream that is unambiguous throughout, the latch
    changes nothing — both arms report the same, small number of changes. If
    the latch "helped" there too, it would be suppressing signal, not noise.

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_latch.mojo
"""

from std.testing import assert_true

from mojo_rl.experimental.swm.rng import Rng
from mojo_rl.experimental.swm.observables import (
    classify,
    class_name,
    ClassificationLatch,
    CLASS_NOMINAL,
    CLASS_ABERRANT,
    CLASS_OBSTRUCTION,
)

comptime STEPS = 400
comptime ANGLE_TOL = 0.2


def run_stream(
    hold_steps: Int, hovering: Bool, mut rng: Rng
) raises -> List[Int]:
    """Returns [changes, final verdict]. `hold_steps = 0` disables the latch.

    The world is genuinely obstructed throughout (det H = -1). What wobbles is
    the worst edge residual: near the 10x outlier boundary while `hovering`,
    then settling well below it in the last quarter.
    """
    var latch = ClassificationLatch(hold_steps, CLASS_NOMINAL)
    var final = CLASS_NOMINAL
    for t in range(STEPS):
        var ratio: Float64
        if hovering and t < 3 * STEPS // 4:
            ratio = 10.0 + rng.normal() * 1.5  # straddles the threshold
        else:
            ratio = 2.0 + rng.uniform_range(0.0, 0.5)  # unambiguously inlier
        var proposed = classify(ratio, 1.0, -1.0, 2.0, ANGLE_TOL, False)
        final = latch.update(proposed)
    var out = List[Int]()
    out.append(latch.changes)
    out.append(Int(final))
    return out^


def main() raises:
    var checks = 0

    var r1 = Rng(20260904)
    var r2 = Rng(20260904)
    var bare = run_stream(0, True, r1)
    var latched = run_stream(8, True, r2)

    print("boundary-hovering stream, ", STEPS, " steps")
    print("  no latch : changes =", bare[0], " final =", class_name(UInt8(bare[1])))
    print("  latched  : changes =", latched[0], " final =",
          class_name(UInt8(latched[1])))

    checks += 3
    assert_true(
        bare[0] > 20,
        "the control stream must actually chatter without a latch, got "
        + String(bare[0]) + " changes — otherwise this gate measures nothing",
    )
    assert_true(
        latched[0] * 3 < bare[0],
        "the latch must cut churn substantially: " + String(latched[0])
        + " vs " + String(bare[0]),
    )
    # The leg that stops the metric being gamed: a latch that never moves would
    # win on churn and be worthless.
    assert_true(
        latched[1] == Int(CLASS_OBSTRUCTION),
        "once the evidence settles the latched verdict must be OBSTRUCTION, "
        + "got " + class_name(UInt8(latched[1])) + " — a latch that cannot be "
        + "moved scores perfectly on churn and is useless",
    )

    # ---- NEGATIVE CONTROL: an unambiguous stream ---------------------------
    var r3 = Rng(777)
    var r4 = Rng(777)
    var bare_clear = run_stream(0, False, r3)
    var latched_clear = run_stream(8, False, r4)
    print("unambiguous stream")
    print("  no latch : changes =", bare_clear[0], " final =",
          class_name(UInt8(bare_clear[1])))
    print("  latched  : changes =", latched_clear[0], " final =",
          class_name(UInt8(latched_clear[1])))
    checks += 3
    assert_true(
        bare_clear[0] <= 2,
        "an unambiguous stream must not chatter even without a latch, got "
        + String(bare_clear[0]),
    )
    assert_true(
        latched_clear[0] == bare_clear[0],
        "NEGATIVE CONTROL FAILED: the latch changed the answer on an "
        + "unambiguous stream (" + String(latched_clear[0]) + " vs "
        + String(bare_clear[0]) + ") — it would be suppressing signal, "
        + "not noise",
    )
    assert_true(
        bare_clear[1] == Int(CLASS_OBSTRUCTION)
        and latched_clear[1] == Int(CLASS_OBSTRUCTION),
        "both arms must reach OBSTRUCTION on an unambiguous obstructed stream",
    )

    print()
    print("assertions compared :", checks)
    print("PASS: G12 the classification loop is latched, not pinned")
