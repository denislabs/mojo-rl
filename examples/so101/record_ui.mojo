# +--------------------------------------------------------------------------+ #
# | SO-101 recording, with the operator looking at what they are recording
# +--------------------------------------------------------------------------+ #
"""`record.mojo` with live camera views, joint bars and buttons.

    pixi run build-imgui                        # ONCE
    pixi run build-opencv                       # ONCE

    # safe: shows everything, energises nothing
    pixi run soarm-record-ui -- --project so101-tower --dataset trial-01 \\
        --task "Grab the green cube"

    # --arm is what moves the robot
    pixi run soarm-record-ui -- --arm --project so101-tower --dataset trial-01 \\
        --task "Grab the green cube" \\
        --devices 0,1 --cameras observation.images.overhead,observation.images.wrist

`--project P --dataset D` records into `projects/P/datasets/D/` (the project
must exist); `--out DIR` still works for a recording outside any project.

⚠ EVERY FINISHED EPISODE IS SAVED AS IT ENDS (one mp4 per episode, metadata
rewritten), so a crash or a closed window loses only the episode in progress.
Add `--resume` to continue a dataset — after a crash, or to add episodes to a
finished one. Without it an existing dataset is refused, never overwritten.

## ⚠⚠ The follower is ENGAGED once, and stays engaged

With `--arm`, "engage follower" turns torque on and the follower follows the
leader continuously — between episodes too — so it can be brought to a start
pose, and ending an episode does not drop it. Only "release follower" turns
torque off, and `finish` is refused until then, so the drop is always an
explicit act done with the leader at rest. This differs from `record.mojo`,
which blocks on a terminal prompt between episodes and so cannot follow.
(Closing the window, or an error, still releases in the `finally`.)

This is what LeRobot uses Rerun for: seeing the feeds and the joint traces
while teleoperating, so a demonstration can be judged before it is kept. It is
imgui over SDL3 — the stack `examples/vision/camera_studio.mojo` already runs —
so it is not a new dependency, it is two existing ones pointed at each other.

⚠⚠ **`--arm` MOVES THE FOLLOWER, AND NOTHING ELSE DOES.** Same rule and same
reason as `record.mojo`, which has the incident that produced it written into
its header. Every safety guard here is that program's; this only adds a view.

## The record loop is still the main loop

⚠ **THE UI DOES NOT DRIVE THE TICK.** `record.mojo` is paced by
`take_blocking` on the first camera, because a wall-clock loop drifts against
a free-running camera and drops frames (measured: 22 in 8 s). Making the UI's
frame rate the clock would reintroduce exactly that, so the loop is unchanged
and the UI is drawn every `UI_EVERY`-th tick inside it.

That is affordable because of the budget: the tick's own work is **4.9 ms**
worst of a 33.3 ms period (`docs/SO101_RECORDING_PLAN.md`), so there is room
for a draw. Drawing every tick would not fit — hence 10 Hz.

⚠ **THE PREVIEW IS A COPY, AND IT IS NOT THE RECORDED FRAME.** RGB→RGBA for
the GPU is a full-frame per-pixel pass, and a per-pixel loop over a `List` is
the single most expensive thing this loop has ever done (9.8 ms when the
channel swap lived here). So it happens only on the ticks that draw, and only
for the camera being shown.
"""

from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.core.project import project_dataset_dir
from mojo_rl.data.lerobot_write import LeRobotWriter, open_recording
from mojo_rl.render.imgui import (
    IgTexture, ig_begin_child, ig_begin_panel, ig_begin_window, ig_button,
    ig_end, ig_end_child, ig_framerate, ig_last_item_rect, ig_overlay_line,
    ig_progress_bar, ig_same_line, ig_separator, ig_separator_text, ig_text,
    ig_text_colored, ig_text_disabled, imgui_shim_available,
)
from mojo_rl.render.renderer3d import Renderer3D
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name, joint_short
from mojo_rl.utils.fmt import fixed
from mojo_rl.data.lerobot_rejected import is_rejected, load_rejected_episodes, reject_episode
from mojo_rl.vision.camera_thread import CameraReader


comptime FOLLOWER_PORT = "/dev/cu.usbmodem5B8E1139971"
comptime LEADER_PORT = "/dev/cu.usbmodem5B910455171"

comptime HZ = 30
comptime WIDTH = 640
comptime HEIGHT = 480
comptime MAX_STEP_TICKS = 80

comptime WIN_W = 1180
comptime WIN_H = 720
comptime UI_EVERY = 3
"""Draw every 3rd tick — 10 Hz. See the header: the tick belongs to the
camera, and the draw has to fit in what is left of it."""

comptime TRACK_STEP_TICKS = 512
"""~45 degrees: the clamp once the follower has caught up with the leader.
See `SO101Arm.track_step_ticks` — one 80-tick clamp made the follower lag the
leader by 300 ms."""


comptime CAMERA_NAMES = "observation.images.front,observation.images.side"

comptime TRACE_CAP = 300
"""Samples kept per signal — 10 s at 30 Hz. Long enough to see the shape of a
motion, short enough that the plot is not a smear."""

comptime PLOT_H = 118.0


def joint_colour(i: Int) -> UInt32:
    """One colour per servo, for the scope traces.

    ⚠ **ImGui PACKS COLOURS ABGR, NOT RGBA** — `ig_overlay_line`'s own header
    says `0xFF00FF00` is opaque GREEN. So the literals below read
    `0xAABBGGRR`, and a value copied from a web colour picker comes out with
    red and blue swapped: a plausible-looking plot in the wrong colours, which
    is worse than an obviously broken one because nobody checks a legend.
    """
    if i == 0:
        return 0xFF3C3CE6  # red
    if i == 1:
        return 0xFF2896F0  # orange
    if i == 2:
        return 0xFF3CDCE6  # yellow
    if i == 3:
        return 0xFF6ED250  # green
    if i == 4:
        return 0xFFFA965A  # blue
    return 0xFFE66EDC  # magenta


def joint_colour_f(i: Int) -> Tuple[Float32, Float32, Float32]:
    """The same colour as 0..1 floats — `ig_text_colored` takes those, not
    the packed word. Two representations of one table is a drift risk, so
    this DERIVES from `joint_colour` rather than repeating the numbers."""
    var c = joint_colour(i)
    var r = Float32(Int(c & 0xFF)) / 255.0
    var g = Float32(Int((c >> 8) & 0xFF)) / 255.0
    var b = Float32(Int((c >> 16) & 0xFF)) / 255.0
    return (r, g, b)


def draw_traces(
    var title: String,
    ref hist: List[Float32],
    write_at: Int,
    filled: Int,
    lo: Float32,
    hi: Float32,
    w: Float32,
) raises:
    """Six coloured polylines over one plot area.

    ⚠ NOT `ig_plot_lines`. That draws ONE series per call with no colour
    control, so six of them would be six stacked monochrome sparklines — the
    thing this replaces. A bordered child reserves the rect, `ig_last_item_rect`
    says where it landed, and the lines go on the window's draw list.

    ⚠ THE BUFFER IS A RING, so it is walked from the OLDEST sample, not from
    index 0. Plotting it unrotated makes the trace jump at the write cursor,
    which reads as a real discontinuity in the robot's motion.
    """
    ig_text(title)
    _ = ig_begin_child(title + "##plot", w, PLOT_H, True)
    ig_end_child()
    var rect = ig_last_item_rect()
    var x0 = rect[0]
    var y0 = rect[1]
    var pw = rect[2]
    var ph = rect[3]
    if filled < 2 or hi <= lo:
        return

    var span = hi - lo
    var start = write_at - filled
    if start < 0:
        start += TRACE_CAP
    var dx = pw / Float32(filled - 1)
    for j in range(SO101_N):
        var col = joint_colour(j)
        for k in range(filled - 1):
            var a = (start + k) % TRACE_CAP
            var b = (start + k + 1) % TRACE_CAP
            var va = hist[a * SO101_N + j]
            var vb = hist[b * SO101_N + j]
            var ya = y0 + ph - (va - lo) / span * ph
            var yb = y0 + ph - (vb - lo) / span * ph
            ig_overlay_line(
                x0 + Float32(k) * dx, ya,
                x0 + Float32(k + 1) * dx, yb,
                col, 1.6,
            )


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


def _engage(mut follower: SO101Arm, mut present: Array[Int32, SO101_N]) raises -> Bool:
    """Torque ON with the goal parked on the follower's CURRENT pose.

    Guard 1 of `record.mojo`: enabling torque must hold the arm where it
    stands, never snap it to a stale `Goal_Position`. From then on each tick's
    goal is clamped to `present ± MAX_STEP_TICKS`, so a leader held elsewhere
    is reached by a ramp. Returns False (and leaves torque off) on a partial
    read.
    """
    if follower.read_positions(Span(present)) != SO101_N:
        return False
    follower.set_position_mode()
    var hold = follower.max_step_ticks
    follower.max_step_ticks = 0
    follower.write_goals(Span(present))
    follower.max_step_ticks = hold
    follower.set_torque(True)
    return True


def main() raises:
    if not imgui_shim_available():
        raise Error(
            "record_ui: the ImGui shim is not built — `pixi run build-imgui`"
        )

    var out_root = String("")
    var project = String("")
    var dataset = String("")
    var resume = False
    var task = String("")
    var seconds = 120
    """A CAP, not a schedule: an episode is normally ended with a button."""
    var devices = List[Int]()
    var cam_names = List[String]()
    var arm = False

    var args = argv()
    for i in range(len(args)):
        var a = String(args[i])
        if a == "--out" and i + 1 < len(args):
            out_root = String(args[i + 1])
        elif a == "--project" and i + 1 < len(args):
            project = String(args[i + 1])
        elif a == "--dataset" and i + 1 < len(args):
            dataset = String(args[i + 1])
        elif a == "--task" and i + 1 < len(args):
            # ⚠ CONSUME EVERY WORD UP TO THE NEXT FLAG. `pixi run <task> --
            # --task "Grab the green cube"` re-splits the quoted string, so
            # taking only args[i+1] silently records the task as "Grab". The
            # task string is written into meta/tasks.parquet and is what a
            # policy is conditioned on — truncating it is not cosmetic.
            task = String(args[i + 1])
            var j = i + 2
            while j < len(args) and not String(args[j]).startswith("--"):
                task += " " + String(args[j])
                j += 1
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
        elif a == "--resume":
            resume = True
        elif a == "--arm":
            arm = True

    if project.byte_length() > 0 or dataset.byte_length() > 0:
        if out_root.byte_length() > 0:
            raise Error("record_ui: give --out, or --project with --dataset, not both")
        if project.byte_length() == 0 or dataset.byte_length() == 0:
            raise Error("record_ui: --project and --dataset go together")
        out_root = project_dataset_dir(project, dataset)
    if out_root == "":
        raise Error(
            "record_ui: --project <name> --dataset <name> (or --out <directory>)"
            " is required"
        )
    if task == "":
        raise Error("record_ui: --task \"<what you are doing>\" is required")
    if len(devices) == 0:
        devices.append(0)
        devices.append(1)
    if len(cam_names) == 0:
        cam_names = _split(String(CAMERA_NAMES), String(","))
    if len(cam_names) != len(devices):
        var got = String("")
        for n in cam_names:
            got += (" | " if got.byte_length() > 0 else "") + n
        raise Error(
            "record_ui: " + String(len(devices)) + " camera device(s) but "
            + String(len(cam_names)) + " camera name(s) [" + got + "]."
            " --devices and --cameras are COMMA-separated, with no spaces:"
            " --devices 0,1 --cameras observation.images.a,observation.images.b"
        )

    var cams = List[CameraReader]()
    for i in range(len(devices)):
        print("opening camera " + String(devices[i]) + " ...")
        var c = CameraReader(
            devices[i], WIDTH, HEIGHT, Float64(HZ), rgb=True
        )
        c.start()
        cams.append(c^)
    var n_cam = len(cams)

    print("opening arms ...")
    var follower = SO101Arm(
        String(FOLLOWER_PORT),
        max_step_ticks=MAX_STEP_TICKS,
        track_step_ticks=TRACK_STEP_TICKS,
    )
    var leader = SO101Arm(String(LEADER_PORT), max_step_ticks=0)
    follower.bus.timeout_ms = 20
    leader.bus.timeout_ms = 20
    leader.set_torque(False)

    var joint_names = List[String]()
    for i in range(SO101_N):
        joint_names.append(joint_name(i) + ".pos")

    # ⚠ CHECKPOINTED: every finished episode is on disk as a complete dataset,
    # so a crash loses only the episode in progress, and `--resume` continues.
    var writer = open_recording(
        out_root.copy(), HZ, joint_names.copy(), joint_names.copy(),
        cam_names.copy(), HEIGHT, WIDTH, resume,
    )
    if resume:
        print(
            "resuming " + out_root + ": " + String(writer.n_episodes())
            + " episodes / " + String(writer.n_rows()) + " frames already recorded"
        )

    var r = Renderer3D(WIN_W, WIN_H)
    var title = String("SO-101 recorder — ") + out_root
    r.init(title)
    if not r.imgui_init():
        raise Error("record_ui: ImGui declined this device")

    var texes = List[IgTexture]()
    for i in range(n_cam):
        texes.append(IgTexture(r.device.value(), WIDTH, HEIGHT))
    var rgba = List[UInt8](unsafe_uninit_length = WIDTH * HEIGHT * 4)

    var frames = List[List[UInt8]]()
    for i in range(n_cam):
        frames.append(List[UInt8](unsafe_uninit_length = cams[i].frame_bytes()))

    var present = Array[Int32, SO101_N](fill=0)
    var lead_raw = Array[Int32, SO101_N](fill=0)
    var goals = Array[Int32, SO101_N](fill=0)

    # ⚠ FILLED EVERY TICK, not only on drawn ones. A trace sampled at the UI
    # rate would be a different signal from the one being recorded — and the
    # point of watching it is to judge the data.
    var trace_lead = List[Float32](unsafe_uninit_length = TRACE_CAP * SO101_N)
    var trace_foll = List[Float32](unsafe_uninit_length = TRACE_CAP * SO101_N)
    for i in range(len(trace_lead)):
        trace_lead[i] = 0.0
        trace_foll[i] = 0.0
    var trace_at = 0
    var trace_filled = 0
    var trace_lo = Float32(0.0)
    var trace_hi = Float32(0.0)

    var recording = False
    var engaged = False
    """Follower torque ON and following the leader. Independent of recording."""
    var ep_frames = 0
    var kept = writer.n_episodes()
    """Episodes WRITTEN, discarded ones included: it is the writer's index."""
    var rejected_list = load_rejected_episodes(out_root)
    var rejected = len(rejected_list)
    var last_rejected = kept - 1 if is_rejected(rejected_list, kept - 1) else -1
    var total_frames = writer.n_rows()
    var tick = 0
    var bus_skipped = 0
    var worst_work = 0.0
    var finish = False
    var status = String("ready")

    try:
        while not r.check_quit() and not finish:
            # ── the tick, exactly as in record.mojo ──────────────────
            if not cams[0].take_blocking(frames[0]):
                status = String("camera 0 stopped delivering frames")
                break
            var ok = True
            for i in range(1, n_cam):
                if not cams[i].take_blocking(frames[i]):
                    ok = False
            if not ok:
                status = String("a camera stopped delivering frames")
                break
            var t0 = perf_counter_ns()

            var n = leader.read_positions(Span(lead_raw))
            var fgot = follower.read_positions(Span(present))
            if n != SO101_N or fgot != SO101_N:
                bus_skipped += 1
            else:
                for i in range(SO101_N):
                    goals[i] = follower.cal.raw_from_degrees(
                        i, leader.cal.degrees(i, lead_raw[i])
                    )
                # ⚠⚠ WHENEVER ENGAGED, NOT ONLY WHILE RECORDING. The follower
                # used to follow the leader during an episode only, so it could
                # not be brought to a start pose, and every end of episode
                # released torque with the arm in the air — it fell.
                if arm and engaged:
                    try:
                        follower.write_goals(Span(goals))
                    except:
                        pass

            if recording:
                var state = List[Float64]()
                var action = List[Float64]()
                for i in range(SO101_N):
                    state.append(follower.cal.degrees(i, present[i]))
                    action.append(leader.cal.degrees(i, lead_raw[i]))
                writer.add_frame(state, action, frames)
                ep_frames += 1
                total_frames += 1
                if ep_frames >= HZ * seconds:
                    # Auto-stop at the cap. ⚠ The follower stays ENGAGED: an
                    # episode ending is not a reason for the arm to drop.
                    writer.end_episode()
                    kept += 1
                    recording = False
                    status = (
                        String("episode ") + String(kept - rejected) + " kept ("
                        + String(ep_frames) + " frames, reached --seconds)"
                    )
                    ep_frames = 0

            # ── the scope ───────────────────────────────────────────
            for j in range(SO101_N):
                var l = Float32(leader.cal.degrees(j, lead_raw[j]))
                var f = Float32(follower.cal.degrees(j, present[j]))
                trace_lead[trace_at * SO101_N + j] = l
                trace_foll[trace_at * SO101_N + j] = f
                if trace_filled == 0 and j == 0:
                    trace_lo = l
                    trace_hi = l
                if l < trace_lo:
                    trace_lo = l
                if l > trace_hi:
                    trace_hi = l
                if f < trace_lo:
                    trace_lo = f
                if f > trace_hi:
                    trace_hi = f
            trace_at = (trace_at + 1) % TRACE_CAP
            if trace_filled < TRACE_CAP:
                trace_filled += 1

            var work = Float64(perf_counter_ns() - t0) / 1e6
            if work > worst_work:
                worst_work = work
            tick += 1

            # ── the view, at UI_EVERY ────────────────────────────────
            if tick % UI_EVERY != 0:
                continue

            for i in range(n_cam):
                # RGB -> RGBA. Only on drawn ticks; see the header.
                ref src = frames[i]
                for p in range(WIDTH * HEIGHT):
                    rgba[p * 4] = src[p * 3]
                    rgba[p * 4 + 1] = src[p * 3 + 1]
                    rgba[p * 4 + 2] = src[p * 3 + 2]
                    rgba[p * 4 + 3] = 255
                _ = texes[i].upload(rgba)

            r.imgui_new_frame()

            _ = ig_begin_panel(String("recorder"), 0.0, 0.0, 340.0, Float32(WIN_H))
            ig_separator_text(String("session"))
            ig_text(String("out    ") + out_root)
            ig_text(String("task   ") + task)
            if arm:
                # ⚠ 0..1 floats, not 0..255 bytes.
                ig_text_colored(
                    String("ARMED — the follower will move"),
                    1.0, 0.25, 0.2,
                )
            else:
                ig_text_disabled(String("not armed (no --arm): nothing moves"))
            ig_text(
                String("kept   ") + String(kept - rejected) + " episode(s)"
                + ((", " + String(rejected) + " discarded") if rejected > 0 else String(""))
            )
            ig_text(String("frames ") + String(total_frames))

            ig_separator_text(String("episode"))
            if recording:
                var frac = Float32(ep_frames) / Float32(HZ * seconds)
                ig_progress_bar(
                    frac, -1.0, 0.0,
                    String(ep_frames) + " / " + String(HZ * seconds),
                )
                if ig_button(String("stop and keep"), 150.0, 30.0):
                    writer.end_episode()
                    kept += 1
                    recording = False
                    status = String("episode ") + String(kept - rejected) + " kept"
                    ep_frames = 0
                ig_same_line()
                if ig_button(String("stop and discard"), 150.0, 30.0):
                    # ⚠ Ended normally, then listed: the writer is append-only.
                    # See `lerobot_rejected.mojo`. Written to disk NOW.
                    writer.end_episode()
                    _ = reject_episode(out_root, kept)
                    last_rejected = kept
                    kept += 1
                    rejected += 1
                    recording = False
                    status = String("discarded (skipped on import)")
                    ep_frames = 0
            else:
                if ig_button(String("start episode"), 150.0, 30.0):
                    if arm and not engaged:
                        engaged = _engage(follower, present)
                    # ⚠ The episode's encoders start FIRST (one mp4 per
                    # episode now, ~a few hundred ms), THEN the drain — so the
                    # frames queued while ffmpeg started are not the
                    # episode's first frames.
                    writer.begin_episode(task.copy())
                    for i in range(n_cam):
                        _ = cams[i].drain()
                    recording = True
                    ep_frames = 0
                    status = String("recording")
                ig_same_line()
                if ig_button(String("finish"), 120.0, 30.0):
                    if engaged:
                        # ⚠ Finishing releases torque. Refused while engaged,
                        # so the drop is always an explicit, separate act.
                        status = String(
                            "release the follower first (leader at rest)"
                        )
                    else:
                        finish = True
                # ⚠ A demonstration is often judged bad only AFTER "stop and
                # keep" — the replay in your head catches the fumble. Only the
                # most recent episode, and only once.
                if kept > 0 and last_rejected != kept - 1:
                    ig_same_line()
                    if ig_button(String("discard last"), 120.0, 30.0):
                        _ = reject_episode(out_root, kept - 1)
                        last_rejected = kept - 1
                        rejected += 1
                        status = String("last episode discarded (skipped on import)")

            if arm:
                ig_separator_text(String("follower"))
                if engaged:
                    ig_text_colored(String("ENGAGED — following the leader"), 1.0, 0.25, 0.2)
                    if not recording:
                        ig_text_disabled(
                            String("rest the leader first: releasing drops the arm")
                        )
                        if ig_button(String("release follower"), 150.0, 30.0):
                            follower.set_torque(False)
                            engaged = False
                            status = String("follower released")
                else:
                    ig_text_disabled(String("released (torque off)"))
                    if ig_button(String("engage follower"), 150.0, 30.0):
                        engaged = _engage(follower, present)
                        status = (
                            String("follower engaged") if engaged
                            else String("engage refused: partial servo read")
                        )

            ig_separator_text(String("joints (deg)"))
            for i in range(SO101_N):
                var c = joint_colour_f(i)
                ig_text_colored(
                    joint_short(i) + " " + fixed(
                        leader.cal.degrees(i, lead_raw[i]), 1
                    ) + " -> " + fixed(
                        follower.cal.degrees(i, present[i]), 1
                    ),
                    c[0], c[1], c[2],
                )

            ig_separator_text(String("health"))
            var cdrop = 0
            for i in range(n_cam):
                cdrop += cams[i].dropped()
            ig_text(String("bus skipped   ") + String(bus_skipped))
            ig_text(String("camera drops  ") + String(cdrop))
            ig_text(String("worst work    ") + fixed(worst_work, 1) + " ms")
            ig_text(String("ui            ") + fixed(Float64(ig_framerate()), 0) + " fps")
            ig_separator()
            ig_text(status)
            ig_end()

            for i in range(n_cam):
                _ = ig_begin_window(
                    cam_names[i], 350.0 + Float32(i) * 410.0, 10.0, 400.0, 340.0
                )
                texes[i].image(384.0, 288.0)
                ig_end()

            _ = ig_begin_window(
                String("servos"), 350.0, 360.0, 810.0, 330.0
            )
            # ⚠ ONE SHARED SCALE for both plots. Auto-scaling each separately
            # makes the leader and the follower look aligned while they are
            # degrees apart — which is exactly the thing an operator is
            # watching these traces to catch.
            var pad = (trace_hi - trace_lo) * 0.05 + 1.0
            draw_traces(
                String("leader (commanded)"), trace_lead, trace_at,
                trace_filled, trace_lo - pad, trace_hi + pad, 780.0,
            )
            draw_traces(
                String("follower (measured)"), trace_foll, trace_at,
                trace_filled, trace_lo - pad, trace_hi + pad, 780.0,
            )
            ig_end()

            r.begin_frame()
            r.end_frame()
    finally:
        try:
            follower.set_torque(False)
        except:
            print("⚠ COULD NOT RELEASE TORQUE — run `pixi run soarm-torque-off`")
        for i in range(n_cam):
            try:
                cams[i].stop()
            except:
                pass
        for i in range(n_cam):
            texes[i].close()

    if kept == 0:
        print("no episodes kept; nothing written")
        return
    print("\nwriting dataset ...")
    writer.close()
    print(
        "  " + String(kept - rejected) + " episodes kept, " + String(rejected)
        + " discarded (listed in meta/rejected_episodes.json), "
        + String(total_frames) + " frames written"
    )

