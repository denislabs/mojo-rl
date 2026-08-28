"""Camera studio — a live camera in a window, with ArUco detection on top.

    pixi run build-imgui                  # ONCE, the UI
    pixi run build-opencv                 # ONCE, the camera + detector
    pixi run mojo run -I . examples/vision/camera_studio.mojo
    pixi run mojo run -I . examples/vision/camera_studio.mojo --device 1
    pixi run mojo run -I . examples/vision/camera_studio.mojo --file tests/fixtures/vision/capture_12f.mp4

WHAT THIS IS
============
A playground, and deliberately much less than `examples/physics3d/physics_studio.mojo`:
one source, one view, no scene, no editing, no persistence. It exists so the
vision stack can be LOOKED AT before anything depends on it — a camera that
opens, frames that arrive at a measurable rate, and markers that light up when
you hold one in front of the lens.

⚠⚠ **IT IS ALSO THE FIRST THING THAT RUNS THE VISION PATH LIVE.** Every gate
in `tests/vision/` compares against a committed file, which is the right way to
gate marshalling and says NOTHING about whether a real camera opens, negotiates
a resolution, or keeps up. Those are exactly the questions a fixture cannot
ask, and they are why this is worth building before `deploy_reach_real` learns
to read a marker.

⚠ NO CAMERA NEEDED. `--file` plays any video (it loops), so the whole UI can be
exercised on the committed 3 KB fixture. `--image` shows a still.

WHAT TO LOOK FOR
================
  * **capture ms** — the time `cap.read()` blocks. A camera that quietly
    negotiated 1920x1080 instead of the requested size shows up here as tens of
    milliseconds, long before it shows up as a control loop that misses its
    deadline.
  * **the resolution line** — what the device ACTUALLY gave, never what was
    asked for. OpenCV reports no error when it substitutes.
  * **markers** — ids and corner overlay. The pose readout needs a focal
    length, and the slider's default is a GUESS (see `FX_GUESS`), so treat the
    distance as an order of magnitude until the camera is calibrated.
"""

from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.render.imgui import (
    IgTexture,
    ig_begin_panel,
    ig_begin_window,
    ig_button,
    ig_checkbox,
    ig_combo,
    ig_end,
    ig_framerate,
    ig_last_item_rect,
    ig_overlay_line,
    ig_plot_lines,
    ig_same_line,
    ig_separator,
    ig_separator_text,
    ig_slider_float,
    ig_text,
    ig_text_colored,
    ig_text_disabled,
    imgui_shim_available,
)
from mojo_rl.render.renderer3d import Renderer3D
from mojo_rl.utils.fmt import fixed
from mojo_rl.vision.opencv import (
    ArucoDetector,
    DICT_4X4_50,
    SOLVEPNP_IPPE_SQUARE,
    VideoCapture,
    imread,
    opencv_shim_available,
    solve_pnp,
)

comptime WIN_W = 1180
comptime WIN_H = 760

comptime MAX_W = 1920
comptime MAX_H = 1080

comptime FX_GUESS: Float32 = 600.0
"""⚠ A GUESS, AND THE UI SAYS SO. A pose from an uncalibrated camera is a
distance with an unknown scale factor on it: everything is proportional to the
focal length, so a 20% error in `fx` is a 20% error in Z. Run the ChArUco
calibration (`tests/vision/`, group E) before any of this drives an arm."""

comptime MARKER_MM: Float32 = 40.0


def _plot_push(mut buf: List[Float32], v: Float32, cap: Int = 180):
    """A scrolling history for `ig_plot_lines`. Oldest first."""
    buf.append(v)
    if len(buf) > cap:
        var trimmed = List[Float32](capacity=cap)
        for i in range(len(buf) - cap, len(buf)):
            trimmed.append(buf[i])
        buf = trimmed^


def main() raises:
    # ── arguments ───────────────────────────────────────────────────────────
    var device_index = 0
    var path = String("")
    var is_image = False
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--device" and i + 1 < len(args):
            device_index = Int(String(args[i + 1]))
        elif a == "--file" and i + 1 < len(args):
            path = String(args[i + 1])
        elif a == "--image" and i + 1 < len(args):
            path = String(args[i + 1])
            is_image = True

    if not opencv_shim_available():
        print("OpenCV shim not built.  Run:  pixi run build-opencv")
        return
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    # ── the source ──────────────────────────────────────────────────────────
    # ⚠ OPEN IT BEFORE THE WINDOW. A camera that refuses to open should print
    # one line and exit, not flash a window first — and on macOS the FIRST
    # `cap_open` is what triggers the OS camera-permission prompt, which is far
    # less confusing when nothing else is on screen yet.
    var bgr = List[UInt8]()
    var frame_w: Int
    var frame_h: Int
    var source_label: String
    # ⚠ NOT `from_file("")` AS A PLACEHOLDER — that raises. See
    # `VideoCapture.closed`.
    var cap = VideoCapture.closed()

    if is_image:
        var g = imread(path, bgr)
        frame_w = g[0]
        frame_h = g[1]
        source_label = String("image: ") + path
    else:
        # ⚠ A PLAYGROUND MUST NOT GREET YOU WITH A STACK TRACE. "device 99
        # does not exist" and "another app holds the camera" are ordinary
        # outcomes here, not defects, so they print one line and exit.
        try:
            if path.byte_length() > 0:
                cap = VideoCapture.from_file(path)
                source_label = String("file: ") + path
            else:
                cap = VideoCapture.device(device_index, 1280, 720, 30.0)
                source_label = String("camera ") + String(device_index)
        except e:
            print("could not open the source:", e)
            if path.byte_length() == 0:
                print(
                    "  Try another --device N (OpenCV prints the valid range"
                    " above), or run without a camera:"
                )
                print(
                    "    mojo run -I . examples/vision/camera_studio.mojo"
                    " --file tests/fixtures/vision/capture_12f.mp4"
                )
                print(
                    "  On macOS the first run asks for camera permission;"
                    " if you dismissed it, re-enable it in"
                )
                print("  System Settings > Privacy & Security > Camera.")
            return
        # ⚠ A FILE REPORTS ITS GEOMETRY UP FRONT; A CAMERA OFTEN DOES NOT until
        # the first frame arrives. Read one now so the texture is sized from
        # what the device actually produced rather than from what it claimed.
        if not cap.read(bgr):
            print("source opened but produced no frame:", source_label)
            cap.close()
            return
        frame_w = cap.width
        frame_h = cap.height

    if frame_w <= 0:
        print("no usable source. Tried:", source_label)
        return
    if frame_w > MAX_W or frame_h > MAX_H:
        print(
            "frame is",
            frame_w,
            "x",
            frame_h,
            "— larger than this demo's",
            MAX_W,
            "x",
            MAX_H,
            "budget",
        )
        return

    print("source:", source_label, "->", frame_w, "x", frame_h)

    # ── the window ──────────────────────────────────────────────────────────
    var r = Renderer3D(WIN_W, WIN_H)
    var title = String("camera studio — ") + source_label
    r.init(title)
    if not r.imgui_init():
        print("ImGui declined this device")
        return

    var tex = IgTexture(r.device.value(), frame_w, frame_h)
    var rgba = List[UInt8](unsafe_uninit_length=frame_w * frame_h * 4)

    var det = ArucoDetector(DICT_4X4_50)
    var ids = List[Int32]()
    var corners = List[Float32]()

    var detect_on = True
    var scale = Float32(1.0)
    var fx = FX_GUESS
    var marker_mm = MARKER_MM
    var paused = False
    var n_markers: Int
    var frames = 0
    var cap_ms_hist = List[Float32]()
    var last_cap_ms = Float32(0.0)

    var obj: List[Float64]
    var img_xy: List[Float64]
    var k: List[Float64]
    var dist = List[Float64]()
    var rvec = List[Float64]()
    var tvec = List[Float64]()

    var src_names = List[String]()
    src_names.append(String("4x4_50"))
    src_names.append(String("4x4_100"))
    src_names.append(String("5x5_50"))
    src_names.append(String("6x6_250"))
    var dict_vals = List[Int]()
    dict_vals.append(0)
    dict_vals.append(1)
    dict_vals.append(4)
    dict_vals.append(10)
    var dict_sel = Int32(0)

    while not r.check_quit():
        # ── grab ────────────────────────────────────────────────────────────
        if not is_image and not paused:
            var t0 = perf_counter_ns()
            var ok = cap.read(bgr)
            if not ok:
                # ⚠ A FILE ENDS; A CAMERA FAILING LOOKS THE SAME FROM HERE.
                # Reopening is right for a file (it loops) and is also the only
                # useful response to a camera that dropped off the bus.
                cap.close()
                if path.byte_length() > 0:
                    cap = VideoCapture.from_file(path)
                else:
                    cap = VideoCapture.device(device_index, 1280, 720, 30.0)
                _ = cap.read(bgr)
            last_cap_ms = Float32(Float64(perf_counter_ns() - t0) / 1_000_000.0)
            _plot_push(cap_ms_hist, last_cap_ms)
            frames += 1

        # ── BGR HWC -> RGBA, which is what the GPU texture wants ────────────
        # ⚠ NOT `bgr_hwc_to_rgb_chw`. That one is for the ACT store's layout;
        # this is interleaved RGBA for a sampler. Two different destinations,
        # and using either in the other's place produces an image that still
        # looks like an image.
        var n = frame_w * frame_h
        for i in range(n):
            rgba[i * 4 + 0] = bgr[i * 3 + 2]
            rgba[i * 4 + 1] = bgr[i * 3 + 1]
            rgba[i * 4 + 2] = bgr[i * 3 + 0]
            rgba[i * 4 + 3] = 255
        _ = tex.upload(rgba)

        if detect_on:
            n_markers = det.detect(bgr, frame_w, frame_h, 3, ids, corners)
        else:
            n_markers = 0

        # ── UI ──────────────────────────────────────────────────────────────
        r.imgui_new_frame()

        _ = ig_begin_panel(String("controls"), 0.0, 0.0, 300.0, Float32(WIN_H))
        ig_separator_text(String("source"))
        ig_text(source_label)
        ig_text(String("actual  ") + String(frame_w) + " x " + String(frame_h))
        if not is_image:
            ig_text_disabled(
                String("requested 1280 x 720 — a camera may ignore it")
            )
        ig_text(String("frames  ") + String(frames))
        ig_text(String("capture ") + fixed(Float64(last_cap_ms), 2) + " ms")
        ig_text(String("ui      ") + fixed(Float64(ig_framerate()), 1) + " fps")
        if len(cap_ms_hist) > 2:
            ig_plot_lines(String("ms"), cap_ms_hist, 0, 0.0, 60.0, 60.0)
        if not is_image:
            if ig_button(String("pause") if not paused else String("resume")):
                paused = not paused

        ig_separator_text(String("markers"))
        _ = ig_checkbox(String("detect"), detect_on)
        if ig_combo(String("dict"), dict_sel, src_names):
            det.close()
            det = ArucoDetector(dict_vals[Int(dict_sel)])
        if detect_on:
            ig_text(String("found   ") + String(n_markers))
            for m in range(n_markers):
                ig_text(String("  id ") + String(ids[m]))
        else:
            ig_text_disabled(String("detection off"))

        ig_separator_text(String("pose (uncalibrated)"))
        # ⚠⚠ THE READOUT IS AN ORDER OF MAGNITUDE, NOT A MEASUREMENT. Z scales
        # linearly with `fx`, so this slider is a scale factor on every
        # distance below it. Group E's calibration replaces it with a number.
        ig_text_colored(
            String("fx is a GUESS — calibrate before trusting Z"),
            1.0,
            0.75,
            0.2,
            1.0,
        )
        _ = ig_slider_float(String("fx px"), fx, 200.0, 2000.0)
        _ = ig_slider_float(String("marker mm"), marker_mm, 5.0, 200.0)
        if detect_on and n_markers > 0:
            var half = Float64(marker_mm) / 2000.0
            obj = List[Float64]()
            obj.append(-half)
            obj.append(half)
            obj.append(0.0)
            obj.append(half)
            obj.append(half)
            obj.append(0.0)
            obj.append(half)
            obj.append(-half)
            obj.append(0.0)
            obj.append(-half)
            obj.append(-half)
            obj.append(0.0)
            img_xy = List[Float64]()
            for i in range(8):
                img_xy.append(Float64(corners[i]))
            k = List[Float64]()
            k.append(Float64(fx))
            k.append(0.0)
            k.append(Float64(frame_w) / 2.0)
            k.append(0.0)
            k.append(Float64(fx))
            k.append(Float64(frame_h) / 2.0)
            k.append(0.0)
            k.append(0.0)
            k.append(1.0)
            try:
                solve_pnp(
                    obj, img_xy, k, dist, rvec, tvec, SOLVEPNP_IPPE_SQUARE
                )
                ig_text(String("x ") + fixed(tvec[0] * 1000.0, 1) + " mm")
                ig_text(String("y ") + fixed(tvec[1] * 1000.0, 1) + " mm")
                ig_text(String("z ") + fixed(tvec[2] * 1000.0, 1) + " mm")
            except:
                ig_text_disabled(String("no pose"))
        ig_separator()
        _ = ig_slider_float(String("zoom"), scale, 0.25, 2.0)
        ig_end()

        # ── the image, with the detections drawn over it ────────────────────
        _ = ig_begin_window(
            String("view"),
            310.0,
            10.0,
            Float32(frame_w) * scale + 20.0,
            Float32(frame_h) * scale + 60.0,
        )
        var dw = Float32(frame_w) * scale
        var dh = Float32(frame_h) * scale
        tex.image(dw, dh)
        # ⚠ ASK IMGUI WHERE THE IMAGE WENT rather than recomputing it: the
        # window has padding, a title bar and possibly a scrollbar, and an
        # overlay drawn at a guessed origin is subtly and permanently offset.
        var rect = ig_last_item_rect()
        if detect_on and n_markers > 0:
            for m in range(n_markers):
                for c in range(4):
                    var c2 = (c + 1) % 4
                    ig_overlay_line(
                        rect[0] + corners[m * 8 + c * 2] * scale,
                        rect[1] + corners[m * 8 + c * 2 + 1] * scale,
                        rect[0] + corners[m * 8 + c2 * 2] * scale,
                        rect[1] + corners[m * 8 + c2 * 2 + 1] * scale,
                        0xFF00FF00,
                        2.0,
                    )
        ig_end()

        r.begin_frame()
        r.end_frame()

    det.close()
    tex.close()
    if not is_image:
        cap.close()
