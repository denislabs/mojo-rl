# +--------------------------------------------------------------------------+ #
# | SO-101 recording, with the operator looking at what they are recording
# +--------------------------------------------------------------------------+ #
"""`record.mojo` with live camera views, joint bars and buttons.

    pixi run build-imgui                        # ONCE
    pixi run build-opencv                       # ONCE

    # safe: shows everything, energises nothing
    pixi run mojo run -I . examples/so101/record_ui.mojo \\
        --out /tmp/my-recording --task "Grab the green cube"

    # --arm is what moves the robot
    pixi run mojo run -I . examples/so101/record_ui.mojo --arm \\
        --out /tmp/my-recording --task "Grab the green cube"

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

from mojo_rl.data.lerobot_write import LeRobotWriter
from mojo_rl.render.imgui import (
    IgTexture, ig_begin_panel, ig_begin_window, ig_button, ig_end,
    ig_framerate,
    ig_progress_bar, ig_same_line, ig_separator, ig_separator_text, ig_text,
    ig_text_colored, ig_text_disabled, imgui_shim_available,
)
from mojo_rl.render.renderer3d import Renderer3D
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name, joint_short
from mojo_rl.utils.fmt import fixed
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
    if not imgui_shim_available():
        raise Error(
            "record_ui: the ImGui shim is not built — `pixi run build-imgui`"
        )

    var out_root = String("")
    var task = String("")
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

    if out_root == "":
        raise Error("record_ui: --out <directory> is required")
    if task == "":
        raise Error("record_ui: --task \"<what you are doing>\" is required")
    if len(devices) == 0:
        devices.append(0)
        devices.append(1)
    if len(cam_names) == 0:
        cam_names = _split(String(CAMERA_NAMES), String(","))
    if len(cam_names) != len(devices):
        raise Error("record_ui: camera device/name count mismatch")

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
        String(FOLLOWER_PORT), max_step_ticks=MAX_STEP_TICKS
    )
    var leader = SO101Arm(String(LEADER_PORT), max_step_ticks=0)
    follower.bus.timeout_ms = 20
    leader.bus.timeout_ms = 20
    leader.set_torque(False)

    var joint_names = List[String]()
    for i in range(SO101_N):
        joint_names.append(joint_name(i) + ".pos")

    var writer = LeRobotWriter(
        out_root.copy(), HZ, joint_names.copy(), joint_names.copy(),
        cam_names.copy(), HEIGHT, WIDTH,
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

    var present = InlineArray[Int32, SO101_N](fill=0)
    var lead_raw = InlineArray[Int32, SO101_N](fill=0)
    var goals = InlineArray[Int32, SO101_N](fill=0)

    var recording = False
    var ep_frames = 0
    var kept = 0
    var total_frames = 0
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
                if arm and recording:
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
                    # Auto-stop at the configured length.
                    writer.end_episode()
                    kept += 1
                    recording = False
                    if arm:
                        follower.set_torque(False)
                    status = (
                        String("episode ") + String(kept) + " kept ("
                        + String(ep_frames) + " frames)"
                    )
                    ep_frames = 0

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
            ig_text(String("kept   ") + String(kept) + " episode(s)")
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
                    if arm:
                        follower.set_torque(False)
                    status = String("episode ") + String(kept) + " kept"
                    ep_frames = 0
            else:
                if ig_button(String("start episode"), 150.0, 30.0):
                    # Frames captured while the operator was setting up are
                    # not part of the episode — same drain as record.mojo.
                    for i in range(n_cam):
                        _ = cams[i].drain()
                    if arm:
                        # Guard 1: park the goal on the CURRENT pose before
                        # arming, so torque does not snap to a stale goal.
                        if follower.read_positions(Span(present)) == SO101_N:
                            follower.set_position_mode()
                            var hold = follower.max_step_ticks
                            follower.max_step_ticks = 0
                            follower.write_goals(Span(present))
                            follower.max_step_ticks = hold
                            follower.set_torque(True)
                    writer.begin_episode(task.copy())
                    recording = True
                    ep_frames = 0
                    status = String("recording")
                ig_same_line()
                if ig_button(String("finish"), 120.0, 30.0):
                    finish = True

            ig_separator_text(String("joints (leader -> follower)"))
            for i in range(SO101_N):
                ig_text(
                    joint_short(i) + "  "
                    + fixed(leader.cal.degrees(i, lead_raw[i]), 1) + "  ->  "
                    + fixed(follower.cal.degrees(i, present[i]), 1)
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
    print("  " + String(kept) + " episodes, " + String(total_frames) + " frames")

