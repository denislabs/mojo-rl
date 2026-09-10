# +--------------------------------------------------------------------------+ #
# | Where the camera is, in the robot's frame — solved with the arm as the ruler
# +--------------------------------------------------------------------------+ #
"""Capture arm-pose / marker-pose pairs and fit the camera -> base transform.

    pixi run build-opencv      # ONCE
    pixi run build-imgui       # ONCE
    pixi run build-serial      # ONCE

    pixi run mojo run -I . examples/so101/calibrate_camera_extrinsics.mojo \\
        --camera front --device 0 --marker-mm 30 \\
        --offset 0.0 -0.045 0.012

`docs/VISION_ASSESSMENT_2026_09_09.md` §2 item 1, the RIG half.
`mojo_rl/vision/extrinsics.mojo` is the solver; this is what feeds it.

## ⚠⚠ THIS PROGRAM NEVER ENERGISES THE ARM, AND HAS NO FLAG THAT DOES

Every other program in this directory has an `--arm`. This one does not, and
that is a design decision rather than an omission: the calibration wants the
gripper at a DOZEN well-spread poses including awkward ones near the workspace
edges, and the fastest, safest and most accurate way to get there is to release
the torque and move it **by hand**. It releases torque at start-up and again in
a `finally`.

⚠ A `finally` DOES NOT COVER AN ABORT OR A SIGNAL. If this dies hard, run
`pixi run soarm-torque-off`.

## What it is doing

A marker taped to the gripper is seen by two instruments at once:

| | says where the marker is | in which frame |
|---|---|---|
| the camera | `solve_pnp` on the detected corners | CAMERA |
| the arm | FK on the measured `qpos`, plus `--offset` | ROBOT BASE |

N such pairs determine the rigid transform between the frames — Kabsch, in
`fit_rigid`. ⚠ THIS IS NOT HAND-EYE CALIBRATION AND DOES NOT NEED TO BE:
`cv2.calibrateHandEye` is not in this OpenCV 5.0 build, and a FIXED camera
admits the far easier 3D-3D problem.

## ⚠⚠ THE FOUR THINGS THAT WILL SILENTLY BE WRONG

1. **`--offset` IS NOT OPTIONAL, IT IS A MEASUREMENT.** It is the marker
   CENTRE in the gripper body's own frame, in metres. Leave it at zero and you
   are asserting the marker's centre coincides with the body origin — which is
   inside the plastic. The offset ROTATES with the wrist, so it is not a
   constant bias the fit can absorb: get it wrong by 10 mm and the residual
   rises by roughly 10 mm and stays there however many poses you take.

2. **THE MARKER GOES ON THE GRIPPER, NOT THE JAW.** `--body` defaults to
   `GRIPPER_BODY_IDX`, deliberately NOT the config's `EE_BODY` — which is
   `moving_jaw`, whose pose changes when the gripper opens. A marker on the
   jaw silently encodes the jaw angle into the fit.

3. **THE TWO READINGS MUST BE SIMULTANEOUS.** A frame taken while the arm is
   moving pairs a pose with a marker that was somewhere else. Capture is
   therefore REFUSED unless the arm has been still for `STILL_TICKS` ticks;
   the panel says so.

4. **SPREAD, NOT COUNT.** Twenty poses in one corner of the workspace fit
   beautifully and determine almost nothing. The panel reports the three
   principal extents of what has been captured, and the third one is the one
   to watch: sweeping the gripper across a table gives a plane, and a brick
   picked 15 cm above that plane is then localised by a direction nobody
   measured.

## What it needs first

An INTRINSICS calibration for this camera, as a
`mojo-rl-camera-calibration` file — `examples/vision/camera_studio.mojo`
produces one. This program refuses to start without it, because `solve_pnp`
with a guessed focal length returns a pose with an unknown scale factor on it,
and a scale error in the correspondences becomes a rotation error in the fit.
"""

from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.math3d import Mat3 as Mat3Generic, Quat as QuatGeneric, Vec3 as Vec3Generic
from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model, GRIPPER_BODY_IDX
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from mojo_rl.render.imgui import (
    IgTexture,
    ig_begin_panel,
    ig_begin_window,
    ig_button,
    ig_end,
    ig_framerate,
    ig_last_item_rect,
    ig_overlay_line,
    ig_same_line,
    ig_separator,
    ig_separator_text,
    ig_text,
    ig_text_colored,
    ig_text_disabled,
    imgui_shim_available,
)
from mojo_rl.render.renderer3d import Renderer3D
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_short
from mojo_rl.robot.so101.sim_map import SimJointMap
from mojo_rl.utils.fmt import fixed
from mojo_rl.vision.calib_file import CameraCalib, read_calib, write_calib
from mojo_rl.vision.extrinsics import RigidFit, fit_rigid
from mojo_rl.vision.opencv import (
    ArucoDetector,
    DICT_4X4_50,
    SOLVEPNP_IPPE_SQUARE,
    VideoCapture,
    opencv_shim_available,
    solve_pnp,
)

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]
comptime Quatd = QuatGeneric[DType.float64]

comptime FOLLOWER_PORT = "/dev/cu.usbmodem5B8E1139971"

comptime WIN_W = 1220
comptime WIN_H = 780
comptime REQ_W = 640
comptime REQ_H = 480

comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]

comptime STILL_EPS_TICKS = 4
"""Raw servo ticks. Below this per-joint change the arm counts as still.
⚠ NOT ZERO: a released servo reports 1-2 ticks of jitter forever, so a
zero threshold means the capture button is never enabled."""

comptime STILL_TICKS = 6
"""Consecutive still ticks required. At the loop's rate this is a fraction of
a second — long enough that a hand let go of the arm, short enough not to be
annoying."""

comptime MIN_POSES = 6
"""⚠ THE SOLVER ACCEPTS 3 AND THREE PROVES NOTHING — it fits exactly, so the
residual is 0 whatever the data says. This is the number at which `rms_mm`
starts being a measurement rather than an identity."""


def _fmt3(v: Vec3d, scale: Float64, digits: Int) -> String:
    return (
        fixed(v.x * scale, digits)
        + " "
        + fixed(v.y * scale, digits)
        + " "
        + fixed(v.z * scale, digits)
    )


def main() raises:
    # ── arguments ───────────────────────────────────────────────────────────
    var cam_name = String("front")
    var device_index = 0
    var calib_path = String("")
    var marker_mm = 30.0
    var marker_id = -1
    var body = GRIPPER_BODY_IDX
    var off = Vec3d.zero()
    var port = String(FOLLOWER_PORT)
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--camera" and i + 1 < len(args):
            cam_name = String(args[i + 1])
        elif a == "--device" and i + 1 < len(args):
            device_index = Int(String(args[i + 1]))
        elif a == "--calib" and i + 1 < len(args):
            calib_path = String(args[i + 1])
        elif a == "--marker-mm" and i + 1 < len(args):
            marker_mm = Float64(String(args[i + 1]))
        elif a == "--marker-id" and i + 1 < len(args):
            marker_id = Int(String(args[i + 1]))
        elif a == "--body" and i + 1 < len(args):
            body = Int(String(args[i + 1]))
        elif a == "--port" and i + 1 < len(args):
            port = String(args[i + 1])
        elif a == "--offset" and i + 3 < len(args):
            off = Vec3d(
                Float64(String(args[i + 1])),
                Float64(String(args[i + 2])),
                Float64(String(args[i + 3])),
            )
    if calib_path == "":
        calib_path = String("scratch/camera_") + cam_name + ".txt"

    if not opencv_shim_available():
        print("OpenCV shim not built.  Run:  pixi run build-opencv")
        return
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    # ── the intrinsics, which are a PREREQUISITE and not a nicety ──────────
    var calib: CameraCalib
    try:
        calib = read_calib(calib_path)
    except e:
        print("could not read the intrinsics for camera", cam_name, "-", e)
        print("  Expected a mojo-rl-camera-calibration file at:", calib_path)
        print("  Produce one with:  pixi run camera-studio --device", device_index)
        print("  ⚠ WITHOUT IT solve_pnp has a guessed focal length, every")
        print("    marker distance carries that error, and a scale error in")
        print("    the correspondences becomes a ROTATION error in the fit.")
        return
    print("intrinsics:", calib_path)
    print("  fx", calib.fx, " fy", calib.fy, " cx", calib.cx, " cy", calib.cy)
    if calib.has_extrinsics:
        print(
            "  ⚠ this file ALREADY has extrinsics (rms",
            calib.rms_mm,
            "mm from",
            calib.poses,
            "poses) — saving will REPLACE them",
        )

    if off.x == 0.0 and off.y == 0.0 and off.z == 0.0:
        print("")
        print("⚠⚠ --offset IS ZERO, which asserts that the marker's centre is")
        print("   at the gripper body's origin — inside the plastic. Measure")
        print("   it and pass it, or every correspondence carries the error")
        print("   and it ROTATES with the wrist, so no number of poses")
        print("   averages it away.")
        print("")

    # ── the camera ─────────────────────────────────────────────────────────
    var bgr = List[UInt8]()
    var cap = VideoCapture.closed()
    try:
        cap = VideoCapture.device(device_index, REQ_W, REQ_H, 30.0)
    except e:
        print("could not open camera", device_index, "-", e)
        return
    if not cap.read(bgr):
        print("camera", device_index, "opened but produced no frame")
        cap.close()
        return
    var fw = cap.width
    var fh = cap.height
    # ⚠ AGAINST WHAT THE DEVICE GAVE, never what was requested — OpenCV
    # substitutes a resolution and reports no error when it does.
    try:
        calib.require_size(fw, fh)
    except e:
        print(e)
        cap.close()
        return
    print("camera:", device_index, "->", fw, "x", fh)

    # ── the arm, released ──────────────────────────────────────────────────
    print("opening", port, "...")
    var arm = SO101Arm(port, max_step_ticks=0)
    arm.bus.timeout_ms = 20
    arm.set_torque(False)
    print("  torque RELEASED — move the gripper by hand")

    # ── the kinematics oracle ──────────────────────────────────────────────
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    _ = env.reset()
    var sf = SoArm101Model.make_spec_fields[DType.float64]()
    var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, SO101_N)
    var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, SO101_N)
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    for i in range(SO101_N):
        lo[i] = Float64(lo_col[i])
        hi[i] = Float64(hi_col[i])
    var jmap = SimJointMap.identity(lo^, hi^)

    # ── the window ─────────────────────────────────────────────────────────
    var r = Renderer3D(WIN_W, WIN_H)
    var title = String("extrinsics — camera ") + cam_name
    r.init(title)
    if not r.imgui_init():
        print("ImGui declined this device")
        cap.close()
        return
    var tex = IgTexture(r.device.value(), fw, fh)
    var rgba = List[UInt8](unsafe_uninit_length=fw * fh * 4)
    var det = ArucoDetector(DICT_4X4_50)
    var ids = List[Int32]()
    var corners = List[Float32]()

    # ── state ──────────────────────────────────────────────────────────────
    var raw = Array[Int32, SO101_N](fill=0)
    var prev_raw = Array[Int32, SO101_N](fill=0)
    var qp = List[Float64](length=SO101_N, fill=0.0)
    var qv = List[Float64](length=SO101_N, fill=0.0)
    var still = 0

    var cam_pts = List[Float64]()
    var base_pts = List[Float64]()
    var fit = RigidFit(
        Mat3d.identity(), Vec3d.zero(), 0.0, 0.0, 0, 0,
        Array[Float64, 3](fill=0.0),
    )
    var have_fit = False
    var fit_msg = String("")
    var status = String("release the arm and show the marker")

    var k = calib.k_matrix()
    var dist = calib.dist.copy()
    var obj = List[Float64]()
    var half = marker_mm / 2000.0
    obj.append(-half); obj.append(half); obj.append(0.0)
    obj.append(half); obj.append(half); obj.append(0.0)
    obj.append(half); obj.append(-half); obj.append(0.0)
    obj.append(-half); obj.append(-half); obj.append(0.0)
    var rvec = List[Float64]()
    var tvec = List[Float64]()


    try:
        while not r.check_quit():
            # ── the camera ──────────────────────────────────────────────────
            if not cap.read(bgr):
                # ⚠ PRINTED, NOT PUT IN `status`: the loop breaks here, so the
                # panel that would have shown it is never drawn again.
                print("the camera stopped delivering frames")
                break
            var n = fw * fh
            for i in range(n):
                rgba[i * 4 + 0] = bgr[i * 3 + 2]
                rgba[i * 4 + 1] = bgr[i * 3 + 1]
                rgba[i * 4 + 2] = bgr[i * 3 + 0]
                rgba[i * 4 + 3] = 255
            _ = tex.upload(rgba)
            var n_markers = det.detect(bgr, fw, fh, 3, ids, corners)

            # ── which marker is on the gripper ─────────────────────────────
            #
            # ⚠⚠ AMBIGUITY IS REFUSED, NOT GUESSED. Another marker in frame —
            # one taped to the table, or the calibration board still lying
            # there — would otherwise be paired with the arm's pose and drag
            # the whole fit toward it, with a residual that only says "these
            # points do not agree".
            var pick = -1
            var seen_ids = String("")
            for m in range(n_markers):
                if m > 0:
                    seen_ids += ","
                seen_ids += " " + String(ids[m])
                if marker_id >= 0:
                    if Int(ids[m]) == marker_id:
                        pick = m
                elif n_markers == 1:
                    pick = m

            var have_marker = False
            var p_cam = Vec3d.zero()
            if pick >= 0:
                var img_xy = List[Float64]()
                for i in range(8):
                    img_xy.append(Float64(corners[pick * 8 + i]))
                try:
                    solve_pnp(
                        obj, img_xy, k, dist, rvec, tvec, SOLVEPNP_IPPE_SQUARE
                    )
                    p_cam = Vec3d(tvec[0], tvec[1], tvec[2])
                    have_marker = True
                except:
                    have_marker = False

            # ── the arm, and FK ────────────────────────────────────────────
            var have_arm = arm.read_positions(Span(raw)) == SO101_N
            var p_base = Vec3d.zero()
            if have_arm:
                var moved = 0
                for i in range(SO101_N):
                    var d = Int(raw[i]) - Int(prev_raw[i])
                    if d < 0:
                        d = -d
                    if d > moved:
                        moved = d
                    prev_raw[i] = raw[i]
                if moved <= STILL_EPS_TICKS:
                    still += 1
                else:
                    still = 0
                for i in range(SO101_N):
                    qp[i] = jmap.to_sim(arm.cal, i, raw[i])
                env.set_state(qp, qv)
                # ⚠⚠ `Data.xquat` IS PACKED (x, y, z, w) — W LAST, which is
                # NOT MuJoCo's (w, x, y, z), and `Quat` takes (w, x, y, z).
                # Taking the two for each other silently rotates `off` below
                # by whatever the wrist happens to be doing: no error, a
                # plausible point, and a fit that never converges below a
                # couple of centimetres.
                #
                # ⚠ THE AUTHORITY IS `forward_kinematics.mojo:134-137`, which
                # WRITES it in this order, and the identical conversion in
                # `tests/robots/test_so_arm101_camera_vs_mujoco.mojo:95-100`,
                # which is gated against `mjData.cam_xpos`. That makes this
                # the SECOND site of one rule — the shape this tree records
                # as its most recurring defect — so if a third appears, hoist
                # all three into `kinematics/`.
                var bq = Quatd(
                    Float64(env.d.xquat.data[body * 4 + 3]),
                    Float64(env.d.xquat.data[body * 4 + 0]),
                    Float64(env.d.xquat.data[body * 4 + 1]),
                    Float64(env.d.xquat.data[body * 4 + 2]),
                )
                var bp = Vec3d(
                    Float64(env.d.xpos.data[body * 3 + 0]),
                    Float64(env.d.xpos.data[body * 3 + 1]),
                    Float64(env.d.xpos.data[body * 3 + 2]),
                )
                p_base = bp + Mat3d.from_quat(bq) * off
            else:
                still = 0

            var can_capture = have_marker and have_arm and still >= STILL_TICKS

            # ── UI ─────────────────────────────────────────────────────────
            r.imgui_new_frame()
            _ = ig_begin_panel(
                String("extrinsics"), 0.0, 0.0, 330.0, Float32(WIN_H)
            )
            ig_separator_text(String("camera"))
            ig_text(String("name    ") + cam_name)
            ig_text(
                String("size    ") + String(fw) + " x " + String(fh)
            )
            ig_text(String("fx      ") + fixed(calib.fx, 1))
            ig_text(String("ui      ") + fixed(Float64(ig_framerate()), 0) + " fps")

            ig_separator_text(String("marker"))
            if n_markers == 0:
                ig_text_disabled(String("none in frame"))
            else:
                ig_text(String("seen   ") + seen_ids)
            if pick < 0 and n_markers > 1 and marker_id < 0:
                ig_text_colored(
                    String("AMBIGUOUS — pass --marker-id"), 1.0, 0.5, 0.3, 1.0
                )
            elif have_marker:
                ig_text(String("cam    ") + _fmt3(p_cam, 1000.0, 1) + " mm")
            else:
                ig_text_disabled(String("cam    -"))

            ig_separator_text(String("arm"))
            if have_arm:
                var line = String("")
                for i in range(SO101_N):
                    if i > 0:
                        line += " "
                    line += joint_short(i)
                ig_text_disabled(line)
                var vals = String("")
                for i in range(SO101_N):
                    if i > 0:
                        vals += " "
                    vals += fixed(qp[i], 2)
                ig_text(vals)
                ig_text(String("base   ") + _fmt3(p_base, 1000.0, 1) + " mm")
            else:
                ig_text_colored(String("no reply from the bus"), 1.0, 0.4, 0.3, 1.0)
                ig_text_disabled(String("base   -"))
            if still >= STILL_TICKS:
                ig_text(String("still  yes"))
            else:
                ig_text_colored(
                    String("still  MOVING — hold it"), 1.0, 0.75, 0.2, 1.0
                )

            ig_separator_text(String("captures"))
            ig_text(String("poses  ") + String(len(cam_pts) // 3))
            if ig_button(String("capture pose"), 150.0, 30.0) and can_capture:
                cam_pts.append(p_cam.x)
                cam_pts.append(p_cam.y)
                cam_pts.append(p_cam.z)
                base_pts.append(p_base.x)
                base_pts.append(p_base.y)
                base_pts.append(p_base.z)
                try:
                    fit = fit_rigid(cam_pts, base_pts)
                    have_fit = True
                    fit_msg = String("")
                except e:
                    have_fit = False
                    fit_msg = String(e)
                status = (
                    String("captured pose ") + String(len(cam_pts) // 3)
                )
            ig_same_line()
            if ig_button(String("drop worst"), 130.0, 30.0) and have_fit:
                # ⚠ ONE OUTLIER IS A MIS-TAKEN POSE, not noise: a marker
                # detected a frame late, or the arm nudged between the two
                # reads. Dropping it is right; dropping until the residual
                # looks nice is fitting the report.
                var w = fit.worst
                var keep_cam = List[Float64]()
                var keep_base = List[Float64]()
                for j in range(len(cam_pts) // 3):
                    if j == w:
                        continue
                    for c in range(3):
                        keep_cam.append(cam_pts[j * 3 + c])
                        keep_base.append(base_pts[j * 3 + c])
                cam_pts = keep_cam^
                base_pts = keep_base^
                have_fit = False
                try:
                    fit = fit_rigid(cam_pts, base_pts)
                    have_fit = True
                    fit_msg = String("")
                except e:
                    fit_msg = String(e)
                status = String("dropped pose ") + String(w)
            if ig_button(String("clear")):
                cam_pts = List[Float64]()
                base_pts = List[Float64]()
                have_fit = False
                fit_msg = String("")
                status = String("cleared")
            if not can_capture:
                if not have_marker:
                    ig_text_disabled(String("capture needs the marker"))
                elif not have_arm:
                    ig_text_disabled(String("capture needs the arm"))
                else:
                    ig_text_disabled(String("capture needs the arm STILL"))
            else:
                ig_text(String("ready — move, hold, capture"))

            ig_separator_text(String("fit"))
            if have_fit:
                ig_text(String("rms    ") + fixed(fit.rms_mm, 2) + " mm")
                ig_text(
                    String("worst  ")
                    + fixed(fit.max_mm, 2)
                    + " mm at "
                    + String(fit.worst)
                )
                # ⚠⚠ THE THIRD EXTENT IS THE ONE THAT MATTERS. See the header.
                var sp = (
                    fixed(fit.spread_mm[0], 0)
                    + " / "
                    + fixed(fit.spread_mm[1], 0)
                    + " / "
                    + fixed(fit.spread_mm[2], 0)
                )
                if fit.spread_mm[2] < 30.0:
                    ig_text_colored(
                        String("spread ") + sp + " FLAT", 1.0, 0.75, 0.2, 1.0
                    )
                    ig_text_disabled(String("move the gripper UP and DOWN too"))
                else:
                    ig_text(String("spread ") + sp + " mm")
                    ig_text_disabled(String("(mm, three principal axes)"))
                ig_text(String("origin ") + _fmt3(fit.trans, 1.0, 3) + " m")
            else:
                ig_text_disabled(String("rms    -"))
                ig_text_disabled(String("worst  -"))
                ig_text_disabled(String("spread -"))
                ig_text_disabled(String("origin -"))
            if fit_msg != "":
                ig_text_colored(fit_msg, 1.0, 0.5, 0.3, 1.0)

            var enough = have_fit and len(cam_pts) // 3 >= MIN_POSES
            if ig_button(String("save calibration"), 200.0, 30.0) and enough:
                calib.has_extrinsics = True
                calib.rot = fit.rot
                calib.trans = fit.trans
                calib.rms_mm = fit.rms_mm
                calib.poses = fit.n
                try:
                    write_calib(calib_path, calib)
                    status = String("saved to ") + calib_path
                    print("saved", calib_path)
                    print(String(fit))
                except e:
                    status = String("COULD NOT SAVE: ") + String(e)
            if not enough:
                ig_text_disabled(
                    String("save needs ") + String(MIN_POSES) + "+ poses"
                )
            ig_separator()
            ig_text(status)
            ig_end()

            # ── the view ───────────────────────────────────────────────────
            _ = ig_begin_window(
                String("view"), 340.0, 10.0, Float32(fw) + 20.0,
                Float32(fh) + 60.0,
            )
            tex.image(Float32(fw), Float32(fh))
            var rect = ig_last_item_rect()
            for m in range(n_markers):
                # ⚠ THE PICKED MARKER IS GREEN AND EVERY OTHER ONE IS AMBER,
                # because "which tag is it using" is the question a wrong fit
                # makes you ask afterwards.
                var col = UInt32(0xFF00FF00) if m == pick else UInt32(0xFF20A0FF)
                for c in range(4):
                    var c2 = (c + 1) % 4
                    ig_overlay_line(
                        rect[0] + corners[m * 8 + c * 2],
                        rect[1] + corners[m * 8 + c * 2 + 1],
                        rect[0] + corners[m * 8 + c2 * 2],
                        rect[1] + corners[m * 8 + c2 * 2 + 1],
                        col,
                        2.0,
                    )
            ig_end()

            r.begin_frame()
            r.end_frame()
    finally:
        try:
            arm.set_torque(False)
        except:
            print("⚠ COULD NOT RELEASE TORQUE — run `pixi run soarm-torque-off`")
        det.close()
        tex.close()
        cap.close()

    if have_fit:
        print("")
        print(String(fit))
        print("saved to", calib_path, "if you pressed save")
