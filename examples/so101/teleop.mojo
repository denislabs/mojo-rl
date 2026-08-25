# +--------------------------------------------------------------------------+ #
# | SO-101 teleoperation — leader drives follower, in Mojo
# +--------------------------------------------------------------------------+ #
"""Read the leader's joints, command the follower to match, at a fixed rate.

The Mojo equivalent of `lerobot-teleoperate`, and the smallest program that
proves the whole stack: libc tty -> Feetech packets -> calibrated units ->
a real arm moving.

⚠ **THIS MOVES THE FOLLOWER.** Read `docs/SO101_SERIAL_LAYER.md` §safety
before the first run. Three guards, in order of what they catch:

1. the follower's goal is set to its OWN present position *before* torque is
   enabled, so arming does not snap it to a stale `Goal_Position`;
2. every goal is clamped to the joint's calibrated `[range_min, range_max]`;
3. every goal is clamped to `present ± max_step_ticks`, so a leader that is
   far from the follower is followed by a ramp instead of a lunge.

Torque is released in a `finally`, so a raised `Error` still disarms.

⚠ **A `finally` does NOT cover an abort or a signal.** A `debug_assert` — or
a `kill` — tears the process down without unwinding, and the follower is left
holding its pose (and heating). That happened here on 2026-08-25, and the
recovery is `pixi run soarm-torque-off`. Treat the `finally` as tidiness, not
as the safety mechanism; the safety mechanism is the recovery tool and the
power switch.

    pixi run build-serial
    pixi run mojo run -I . examples/so101/teleop.mojo

Ports are the two SO-101 boards on this desk; pass different ones by editing
the comptimes below (a proper CLI belongs with the rest of the tooling, not
in the first example).
"""

from std.ffi import external_call
from std.time import perf_counter_ns

from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.utils.fmt import col, fixed

comptime FOLLOWER_PORT = "/dev/cu.usbmodem5B8E1139971"
comptime LEADER_PORT = "/dev/cu.usbmodem5B910455171"

comptime HZ = 50
comptime SECONDS = 30

comptime MAX_STEP_TICKS = 80
"""~7 degrees per 20 ms tick. This does not cap the servo's SPEED — it caps
how far ahead of the arm the goal may sit, which is what turns a large
leader/follower mismatch into a ramp. Tighter is smoother and laggier."""


def _sleep_until(deadline_ns: Int):
    """Hold the period by spinning.

    ⚠ Spinning pins a core for ~18 ms in every 20, which is indefensible on
    its face — so both alternatives were measured (15 s at 50 Hz, idle box,
    two runs each), and **both lost**:

    | wait                        | mean tick | worst tick   |
    |-----------------------------|-----------|--------------|
    | pure spin                   | 20.2 ms   | 50–63 ms     |
    | pure `usleep`               | 23.8 ms   | 147 ms       |
    | `usleep` + spin last 1.5 ms | 21.9 ms   | 30 / 107 ms  |

    macOS `usleep` overshoots far enough to miss the period outright, and the
    hybrid inherits a smaller share of the same overshoot. Keep the spin until
    something with real timer semantics is available, and do not "fix" it back
    to a sleep without re-measuring — that is what these numbers are for.

    ⚠ The worst-tick outlier is NOT the bus timeout: it survived cutting that
    from 50 ms to 20 ms unchanged. It is unexplained, it is rare, and the mean
    holds the rate — do not read a mechanism into it without measuring one.
    """
    while perf_counter_ns() < deadline_ns:
        pass


def main() raises:
    print("opening follower:", FOLLOWER_PORT)
    var follower = SO101Arm(
        String(FOLLOWER_PORT), max_step_ticks=MAX_STEP_TICKS
    )
    print("opening leader:  ", LEADER_PORT)
    var leader = SO101Arm(String(LEADER_PORT), max_step_ticks=0)

    # A control loop drops a tick rather than stalling one, so the 50 ms
    # default (lerobot's patched single-transaction timeout, right for setup)
    # is cut to one period.
    #
    # ⚠ NOT to the 1.3 ms a sync_read takes in a tight loop. That number is
    # measured with the USB pipe polled continuously; a loop that SLEEPS
    # between ticks pays host-controller latency on top, and 5 ms produced
    # `0 of 6 motors reported a position` on the very first tick. Throughput
    # measured back-to-back does not give you a duty-cycled loop's latency
    # budget.
    follower.bus.timeout_ms = 20
    leader.bus.timeout_ms = 20

    # The leader is backdriven by hand: it must NOT hold position.
    leader.set_torque(False)

    var present = InlineArray[Int32, SO101_N](fill=0)
    var got = follower.read_positions(Span(present))
    if got != SO101_N:
        raise Error(
            "teleop: follower reported only "
            + String(got)
            + " of "
            + String(SO101_N)
            + " positions — refusing to arm torque"
        )

    # Guard 1: park the goal on the CURRENT pose before arming, so enabling
    # torque holds the arm where it stands instead of driving it to whatever
    # Goal_Position the last session left behind.
    follower.set_position_mode()
    var hold = follower.max_step_ticks
    follower.max_step_ticks = 0  # goals == present; the ramp clamp is moot
    follower.write_goals(Span(present))
    follower.max_step_ticks = hold
    follower.set_torque(True)
    print("follower torque ON — hold the leader, then move it\n")

    var lead_raw = InlineArray[Int32, SO101_N](fill=0)
    var goals = InlineArray[Int32, SO101_N](fill=0)
    var period_ns = 1_000_000_000 // HZ
    var ticks = HZ * SECONDS
    var dropped = 0
    var refused = 0
    var worst_ms = 0.0
    var t_start = perf_counter_ns()

    try:
        for t in range(ticks):
            var t0 = perf_counter_ns()

            var n = leader.read_positions(Span(lead_raw))
            if n != SO101_N:
                # Guard: a partial read means a motor dropped off the bus.
                # Skipping the tick holds the last goal, which is safe;
                # commanding a half-updated pose is not.
                dropped += 1
                _sleep_until(t0 + period_ns)
                continue

            # Both arms are calibrated in the same units, so the leader's
            # DEGREES (and the gripper's 0..100) map straight onto the
            # follower's ticks. Their mid-points differ — this is exactly the
            # constant offset `tools/soarm/so101_pairing.py` reports.
            for i in range(SO101_N):
                goals[i] = follower.cal.raw_from_degrees(
                    i, leader.cal.degrees(i, lead_raw[i])
                )
            try:
                follower.write_goals(Span(goals))
            except e:
                # `write_goals` refuses a partial follower read rather than
                # commanding a half-updated pose. In a loop that is a dropped
                # tick, not a fatal error — the last goal still stands — but
                # it is COUNTED and printed, because a run that silently
                # refused half its writes must not look like a clean one.
                refused += 1

            var ms = Float64(perf_counter_ns() - t0) / 1e6
            if ms > worst_ms:
                worst_ms = ms
            if t % HZ == 0:
                var line = String("t=") + String(t // HZ) + "s "
                for i in range(SO101_N):
                    line += (
                        " "
                        + String(joint_name(i)[byte=0:4])
                        + "="
                        + col(leader.cal.degrees(i, lead_raw[i]), 8, 2)
                    )
                print(line)
            _sleep_until(t0 + period_ns)
    finally:
        # Release torque on ANY exit, including an exception. A follower left
        # holding a pose after a crash is both a safety and a thermal problem.
        follower.set_torque(False)
        print("\nfollower torque OFF")

    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
    print(
        "ran "
        + String(ticks)
        + " ticks in "
        + fixed(elapsed, 2)
        + " s   dropped="
        + String(dropped)
        + "   refused="
        + String(refused)
        + "   worst tick="
        + fixed(worst_ms, 2)
        + " ms (budget "
        + String(1000 // HZ)
        + " ms)"
    )
