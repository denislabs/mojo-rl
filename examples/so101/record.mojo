# +--------------------------------------------------------------------------+ #
# | SO-101 teleoperation, recorded into a LeRobot v3 dataset
# +--------------------------------------------------------------------------+ #
"""Drive the follower from the leader and write every tick to a dataset.

    pixi run build-opencv                       # ONCE
    # SAFE BY DEFAULT: reads everything, never energises the follower.
    pixi run mojo run -I . examples/so101/record.mojo \\
        --out /tmp/my-recording --task "Grab the green cube"

    # --arm is what actually moves the robot. Be at the desk for this.
    pixi run mojo run -I . examples/so101/record.mojo --arm \\
        --out /tmp/my-recording --task "Grab the green cube" \\
        --episodes 5 --seconds 20

    # and then, when you are happy with it:
    pixi run mojo run -I . tools/hf/push_dataset.mojo \\
        --root /tmp/my-recording --repo DenisLabs/my-recording

The Mojo equivalent of `lerobot-record`, and the last piece of the loop:
`teleop.mojo` already proved the arms, `LeRobotWriter` already proved the
format, and this is the program that puts an operator between them.

⚠⚠ **`--arm` MOVES THE FOLLOWER, AND NOTHING ELSE DOES.** Without it this
records the leader and the cameras and never energises anything. Read
`docs/SO101_SERIAL_LAYER.md` §safety before the first armed run.

⚠ **THE OPT-IN IS DELIBERATE, AND IT IS A SCAR.** This started with a
`--dry-run` flag, so the DANGEROUS behaviour was what you got by forgetting an
argument. On 2026-08-31 exactly that happened with nobody at the desk: the
follower armed, ramped **22° on shoulder_pan and 30° on wrist_roll** to reach
the leader's resting pose, and held. The guards did their job and nothing was
damaged — but a safety default that depends on remembering a flag is not a
safety default. Arming is now the thing you have to ask for.

It inherits `teleop.mojo`'s three guards verbatim, because they are the reason
that program is safe to run:

  1. the follower's goal is parked on its OWN present position *before*
     torque is enabled, so arming does not snap it to a stale `Goal_Position`;
  2. every goal is clamped to the joint's calibrated `[range_min, range_max]`;
  3. every goal is clamped to `present ± max_step_ticks`, so a leader that is
     far from the follower is followed by a ramp instead of a lunge.

⚠ **TORQUE IS ON ONLY DURING AN EPISODE.** Between episodes — while you are
reading a prompt — the follower is released. `teleop.mojo` holds torque for
its whole run; a recorder spends most of its time waiting for a human, and
holding a pose through that is both a thermal and a safety cost for nothing.

⚠ **A `finally` DOES NOT COVER AN ABORT OR A SIGNAL.** Same warning as
`teleop.mojo`, and it is not theoretical — it happened on 2026-08-25. The
recovery is `pixi run soarm-torque-off`. Treat the `finally` as tidiness; the
safety mechanism is that tool and the power switch.

## The tick is paced by the CAMERA, not by a clock

Measured (`tools/soarm/record_budget.mojo`, and the table in
`docs/SO101_RECORDING_PLAN.md`): with the cameras and the encoders each on
their own thread the loop's own work is **3.45 ms mean / 5.43 ms worst**
against a 33.3 ms budget. But a loop *clocked* at 30 Hz still lost 22 frames
in 8 s, because the camera free-runs at its own rate and any surplus fills the
ring.

So there is no `sleep_until` here. `take_blocking` on the first camera IS the
clock: one tick per frame, by construction, and the mismatch cannot exist.
The other cameras are read straight after and are within a frame of it.

⚠ **A DROPPED TICK IS NOT A DROPPED FRAME, AND BOTH ARE COUNTED.** A partial
servo read skips the tick's WRITE but the frame is still consumed, so the
video and the data stay aligned; `LeRobotWriter.close` re-checks that
alignment and refuses to write a dataset where it fails.
"""

from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.data.lerobot_write import LeRobotWriter
from mojo_rl.io.fileio import StdinReader
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name, joint_short
from mojo_rl.utils.fmt import col, fixed
from mojo_rl.vision.camera_thread import CameraReader


comptime FOLLOWER_PORT = "/dev/cu.usbmodem5B8E1139971"
comptime LEADER_PORT = "/dev/cu.usbmodem5B910455171"

comptime HZ = 30
comptime WIDTH = 640
comptime HEIGHT = 480
comptime MAX_STEP_TICKS = 80
"""~7 degrees per tick. Not a speed limit — a limit on how far ahead of the
arm the goal may sit, which is what turns a large leader/follower mismatch
into a ramp. Same value `teleop.mojo` measured with."""

comptime CAMERA_NAMES = "observation.images.front,observation.images.side"


def _split(s: String, sep: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(s.byte_length()):
        var c = chr(Int(s.as_bytes()[i]))
        if c == sep:
            out.append(cur^)
            cur = String("")
        else:
            cur += c
    out.append(cur^)
    return out^


def main() raises:
    var out_root = String("")
    var task = String("")
    var n_episodes = 5
    var seconds = 20
    var devices = List[Int]()
    var cam_names = List[String]()
    var arm = False

    var args = argv()
    for i in range(len(args)):
        var a = String(args[i])
        if a == "--out" and i + 1 < len(args):
            out_root = String(args[i + 1])
        elif a == "--task" and i + 1 < len(args):
            task = String(args[i + 1])
        elif a == "--episodes" and i + 1 < len(args):
            n_episodes = Int(String(args[i + 1]))
        elif a == "--seconds" and i + 1 < len(args):
            seconds = Int(String(args[i + 1]))
        elif a == "--devices" and i + 1 < len(args):
            var parts = _split(String(args[i + 1]), String(","))
            for k in range(len(parts)):
                if parts[k] != "":
                    devices.append(Int(parts[k]))
        elif a == "--cameras" and i + 1 < len(args):
            var parts = _split(String(args[i + 1]), String(","))
            for k in range(len(parts)):
                if parts[k] != "":
                    cam_names.append(parts[k])
        elif a == "--arm":
            arm = True
        elif a == "--dry-run":
            # Accepted and ignored: it is now the default. Kept so an old
            # command line does not silently mean the opposite of what it did.
            pass

    if out_root == "":
        raise Error("record: --out <directory> is required")
    if task == "":
        raise Error(
            "record: --task \"<what the operator is doing>\" is required — it"
            " is written into meta/tasks.parquet and a dataset without it is"
            " not trainable"
        )
    if len(devices) == 0:
        devices.append(0)
        devices.append(1)
    if len(cam_names) == 0:
        cam_names = _split(String(CAMERA_NAMES), String(","))
    if len(cam_names) != len(devices):
        raise Error(
            "record: " + String(len(devices)) + " camera devices but "
            + String(len(cam_names)) + " camera names"
        )

    print("=" * 72)
    print("SO-101 recording -> " + out_root)
    print("=" * 72)
    print("  task:     " + task)
    print(
        "  episodes: " + String(n_episodes) + " x " + String(seconds) + " s @ "
        + String(HZ) + " Hz"
    )
    var cam_line = String("")
    for i in range(len(devices)):
        if i > 0:
            cam_line += ", "
        cam_line += String(devices[i]) + "->" + cam_names[i]
    print("  cameras:  " + cam_line)
    if arm:
        print(
            "  ⚠⚠ --arm: THE FOLLOWER WILL BE ENERGISED AND WILL MOVE."
        )
    else:
        print(
            "  safe mode (no --arm): the follower is NEVER energised. The"
            " leader and the cameras are recorded; `observation.state` is the"
            " follower's measured pose, which will not follow."
        )
    print("")

    # ── cameras first: they are the slowest thing to come up ──────────
    var cams = List[CameraReader]()
    for i in range(len(devices)):
        print("opening camera " + String(devices[i]) + " ...")
        # rgb=True: the swap happens on the camera thread, which is idle
        # anyway. Inline it cost 9.8 ms worst of a 33.3 ms tick.
        var c = CameraReader(
            devices[i], WIDTH, HEIGHT, Float64(HZ), rgb=True
        )
        c.start()
        cams.append(c^)
    var n_cam = len(cams)

    # ── the arms ──────────────────────────────────────────────────────
    print("opening follower: " + String(FOLLOWER_PORT))
    var follower = SO101Arm(
        String(FOLLOWER_PORT), max_step_ticks=MAX_STEP_TICKS
    )
    print("opening leader:   " + String(LEADER_PORT))
    var leader = SO101Arm(String(LEADER_PORT), max_step_ticks=0)
    # One period, not the 50 ms setup default and not the 1.3 ms a sync_read
    # takes back-to-back: a duty-cycled loop pays host-controller latency on
    # top, and 5 ms produced "0 of 6 motors reported a position" in teleop.
    follower.bus.timeout_ms = 20
    leader.bus.timeout_ms = 20
    leader.set_torque(False)  # backdriven by hand; it must NOT hold

    var joint_names = List[String]()
    for i in range(SO101_N):
        joint_names.append(joint_name(i) + ".pos")

    var writer = LeRobotWriter(
        out_root.copy(),
        HZ,
        joint_names.copy(),
        joint_names.copy(),
        cam_names.copy(),
        HEIGHT,
        WIDTH,
    )

    var frames = List[List[UInt8]]()
    for i in range(n_cam):
        frames.append(List[UInt8](unsafe_uninit_length = cams[i].frame_bytes()))

    var present = InlineArray[Int32, SO101_N](fill=0)
    var lead_raw = InlineArray[Int32, SO101_N](fill=0)
    var goals = InlineArray[Int32, SO101_N](fill=0)

    var stdin = StdinReader()
    var kept = 0
    var total_dropped = 0
    var total_refused = 0

    try:
        var ep = 0
        while kept < n_episodes:
            # Anything typed while the last episode was recording is not an
            # answer to this question — see `discard_pending`.
            stdin.discard_pending()
            print(
                "── episode " + String(kept + 1) + "/" + String(n_episodes)
                + " ──  position the arms, then press Enter (q = finish)"
            )
            var answer = stdin.line()
            if answer == "q" or answer == "Q":
                print("  finishing early at " + String(kept) + " episode(s)")
                break

            # ⚠ DRAIN FIRST. Frames captured while the operator was reading
            # that prompt are not part of the episode; leaving them queued
            # would prepend someone walking past to every recording.
            var stale = 0
            for i in range(n_cam):
                stale += cams[i].drain()
            # ⚠ A SNAPSHOT, because `dropped()` is CUMULATIVE since start.
            # Printing the running total as "this episode's drops" reported 15
            # for an episode that dropped none of its own.
            var drops_before = 0
            for i in range(n_cam):
                drops_before += cams[i].dropped()

            # ── guard 1: park the goal on the CURRENT pose, then arm ──
            var got = follower.read_positions(Span(present))
            if got != SO101_N:
                raise Error(
                    "record: the follower reported only " + String(got)
                    + " of " + String(SO101_N)
                    + " positions — refusing to arm torque"
                )
            if arm:
                follower.set_position_mode()
                var hold = follower.max_step_ticks
                follower.max_step_ticks = 0  # goals == present
                follower.write_goals(Span(present))
                follower.max_step_ticks = hold
                follower.set_torque(True)
                print(
                    "  follower torque ON — recording " + String(seconds) + " s"
                )
            else:
                print(
                    "  not armed — recording " + String(seconds)
                    + " s (nothing will move)"
                )
            if stale > 0:
                print("  (dropped " + String(stale) + " stale frames)")

            writer.begin_episode(task.copy())
            var ticks = HZ * seconds
            var dropped = 0
            var refused = 0
            var worst_ms = 0.0
            var t_start = perf_counter_ns()
            var recorded = 0

            var worst_wait_ms = 0.0
            # Per-part worsts: one wrong guess about where a 45 ms tick went
            # is enough. These cost two clock reads each.
            var worst_swap = 0.0
            var worst_arms = 0.0
            var worst_write = 0.0
            for _ in range(ticks):
                var t_wait = perf_counter_ns()

                # ⚠ THE CLOCK. Blocking on the first camera paces the loop to
                # the capture rate exactly; see the module docstring.
                if not cams[0].take_blocking(frames[0]):
                    raise Error(
                        "record: camera " + String(devices[0]) + " stopped"
                        " delivering frames"
                    )
                for i in range(1, n_cam):
                    if not cams[i].take_blocking(frames[i]):
                        raise Error(
                            "record: camera " + String(devices[i])
                            + " stopped delivering frames"
                        )
                # ⚠ WORK IS TIMED FROM HERE, NOT FROM THE TOP OF THE TICK.
                # The tick begins by WAITING for the next frame, and once the
                # camera is the clock that wait is the period, not a cost.
                # Timing across it reported a "worst tick" of 114 ms on a run
                # that dropped nothing and held 29.9 Hz.
                var t0 = perf_counter_ns()
                var wait_ms = Float64(t0 - t_wait) / 1e6
                if wait_ms > worst_wait_ms:
                    worst_wait_ms = wait_ms
                # Frames arrive RGB24 already — see the CameraReader above.
                var t_swap = perf_counter_ns()
                var swap_ms = Float64(t_swap - t0) / 1e6
                if swap_ms > worst_swap:
                    worst_swap = swap_ms

                var n = leader.read_positions(Span(lead_raw))
                var fgot = follower.read_positions(Span(present))
                if n != SO101_N or fgot != SO101_N:
                    # A partial read means a motor dropped off the bus.
                    # Skipping the WRITE holds the last goal, which is safe;
                    # commanding a half-updated pose is not. The frame is
                    # still recorded, so video and data stay aligned.
                    dropped += 1
                else:
                    # Both arms are calibrated in the same units, so the
                    # leader's DEGREES map straight onto the follower's ticks.
                    for i in range(SO101_N):
                        goals[i] = follower.cal.raw_from_degrees(
                            i, leader.cal.degrees(i, lead_raw[i])
                        )
                    if arm:
                        try:
                            follower.write_goals(Span(goals))
                        except:
                            refused += 1

                var t_arms = perf_counter_ns()
                var arms_ms = Float64(t_arms - t_swap) / 1e6
                if arms_ms > worst_arms:
                    worst_arms = arms_ms

                # `action` is what we COMMANDED, `observation.state` is what
                # the follower MEASURED — the LeRobot convention.
                var state = List[Float64]()
                var action = List[Float64]()
                for i in range(SO101_N):
                    state.append(follower.cal.degrees(i, present[i]))
                    action.append(leader.cal.degrees(i, lead_raw[i]))
                writer.add_frame(state, action, frames)
                var write_ms = Float64(perf_counter_ns() - t_arms) / 1e6
                if write_ms > worst_write:
                    worst_write = write_ms
                recorded += 1

                var ms = Float64(perf_counter_ns() - t0) / 1e6
                if ms > worst_ms:
                    worst_ms = ms
                if recorded % (HZ * 5) == 0:
                    var line = String("    t=") + String(recorded // HZ) + "s"
                    for i in range(SO101_N):
                        line += (
                            " " + joint_short(i) + "="
                            + col(leader.cal.degrees(i, lead_raw[i]), 7, 1)
                        )
                    print(line)

            if arm:
                follower.set_torque(False)
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9

            var cdrop = 0
            for i in range(n_cam):
                cdrop += cams[i].dropped()
            cdrop -= drops_before
            print(
                "  " + String(recorded) + " frames in " + fixed(elapsed, 1)
                + " s (" + fixed(Float64(recorded) / elapsed, 1) + " Hz)"
                + "   bus-skipped=" + String(dropped)
                + "   refused=" + String(refused)
                + "   camera-drops=" + String(cdrop)
                + "   worst work=" + fixed(worst_ms, 1) + " ms"
                + " (+ worst wait " + fixed(worst_wait_ms, 1) + " ms)"
            )
            print(
                "    worst: bgr->rgb " + fixed(worst_swap, 1)
                + " ms   arms " + fixed(worst_arms, 1)
                + " ms   writer " + fixed(worst_write, 1) + " ms"
            )

            stdin.discard_pending()
            print("  keep this episode? [Y/r=redo/q=quit] ")
            var verdict = stdin.line()
            if verdict == "r" or verdict == "R":
                # ⚠ NOT IMPLEMENTED AS A ROLLBACK. `LeRobotWriter` is
                # append-only — the frames are already inside an ffmpeg pipe —
                # so a redo would need the writer to be able to discard an
                # open episode, which it cannot. Ending it and recording
                # another is honest; the operator can drop it later.
                writer.end_episode()
                kept += 1
                print(
                    "  ⚠ kept anyway: this writer cannot discard an episode"
                    " once its frames are encoded. Episode "
                    + String(kept - 1) + " is the one to ignore."
                )
            else:
                writer.end_episode()
                kept += 1
            total_dropped += dropped
            total_refused += refused
            ep += 1
            if verdict == "q" or verdict == "Q":
                break
    finally:
        # Release torque on ANY exit, including an exception.
        try:
            follower.set_torque(False)
            print("\nfollower torque OFF")
        except:
            print(
                "\n⚠ COULD NOT RELEASE FOLLOWER TORQUE — run"
                " `pixi run soarm-torque-off`"
            )
        for i in range(n_cam):
            try:
                cams[i].stop()
            except:
                pass

    if kept == 0:
        print("no episodes recorded; nothing written")
        return

    print("\nwriting dataset ...")
    writer.close()
    print(
        "  " + String(kept) + " episodes, bus-skipped ticks "
        + String(total_dropped) + ", refused writes " + String(total_refused)
    )
    print("\nnext:")
    print("  pixi run mojo run -I . examples/so101/act_so101_import_dataset.mojo"
          " --root " + out_root)
    print("  pixi run mojo run -I . tools/hf/push_dataset.mojo --root "
          + out_root + " --repo <you>/<name>")
