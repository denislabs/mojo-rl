# +--------------------------------------------------------------------------+ #
# | The follower's two-phase step clamp — catch up slowly, then track
# +--------------------------------------------------------------------------+ #
"""Gate `robot/so101/arm.mojo`: `step_limit` and `is_aligned`.

    pixi run mojo run -I . tests/robot/test_so101_step_clamp.mojo

No hardware. `SO101Arm.write_goals` composes exactly these two: the phase
flips to tracking the first time every joint is within `ALIGN_TICKS` of its
goal, and `set_torque(True)` flips it back.

⚠ WHY THIS EXISTS. One 80-tick clamp was a speed limit: trial-01's follower
lagged the leader by 300 ms on shoulder_lift, 50.9 deg at worst. The fix only
helps if the phase really changes, and only stays safe if it does NOT change
while a joint is still far from its goal.
"""

from mojo_rl.robot.so101 import ALIGN_TICKS, SO101_N, is_aligned, step_limit


def main() raises:
    print("[so101-step-clamp] gate")
    var n = 0

    if step_limit(False, 80, 512) != 80:
        raise Error("catch-up must use the small clamp")
    if step_limit(True, 80, 512) != 512:
        raise Error("tracking must use the large clamp")
    if step_limit(True, 80, 0) != 80:
        raise Error("track_step_ticks=0 must keep the single clamp (old behaviour)")
    if step_limit(True, 0, 0) != 0:
        raise Error("both 0 means no clamp")
    n += 4
    print("  step_limit: 80 catching up, 512 tracking, unchanged when tracking is off")

    var goals = Array[Int32, SO101_N](fill=2000)
    var present = Array[Int32, SO101_N](fill=2000)
    if not is_aligned(goals, present, ALIGN_TICKS):
        raise Error("identical goals and positions must be aligned")
    present[3] = Int32(2000 + ALIGN_TICKS)
    if not is_aligned(goals, present, ALIGN_TICKS):
        raise Error("exactly ALIGN_TICKS away must count as aligned")
    present[3] = Int32(2000 + ALIGN_TICKS + 1)
    if is_aligned(goals, present, ALIGN_TICKS):
        raise Error("one joint ALIGN_TICKS+1 away must NOT be aligned")
    present[3] = Int32(2000)
    present[5] = Int32(2000 - ALIGN_TICKS - 1)
    if is_aligned(goals, present, ALIGN_TICKS):
        raise Error("the LAST joint, below its goal, must be checked too")
    n += 4
    print("  is_aligned: " + String(ALIGN_TICKS) + " ticks is aligned, +1 on any joint (either side) is not")

    print("  " + String(n) + " checks, 0 failures")
    print("[PASS] so101-step-clamp")
