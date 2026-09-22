"""FISHEYE INTRINSICS FOR THE RIG'S CAMERAS — headless, runs over SSH on the Jetson.

    pixi run build-opencv                                  # once
    pixi run mojo run -I . examples/vision/calibrate_fisheye.mojo \\
        --camera /dev/soarm_cam_overhead --name overhead
    pixi run mojo run -I . examples/vision/calibrate_fisheye.mojo \\
        --camera /dev/soarm_cam_wrist --name wrist
    # recalibrate later from the saved captures, anywhere, no camera:
    pixi run mojo run -I . examples/vision/calibrate_fisheye.mojo \\
        --images projects/so101-tower/cameras/overhead_views --name overhead

Writes `projects/so101-tower/cameras/camera_<name>.txt` (`vision/calib_file`,
`model fisheye`, 4 Kannala-Brandt terms, at 640x480) — what the importer's
`--undistort` and the real-robot deploy read to bring a real frame to the
sim's pinhole camera (`vision/fisheye.mojo`).

## What you do

Print the studio's ChArUco board (`pixi run python
tools/vision/make_printable_marker.py --charuco`, 5x7 squares, 30 mm / 22 mm
— measure the printed ruler; a wrong `--square-mm` calibrates a confidently
wrong camera), mount it FLAT, and move it slowly in front of the camera. The
tool captures by itself whenever the board is seen well AND the pose is new;
you move it around until the progress line says it is done:

  * everywhere in the frame — the coverage map's EDGE cells matter most: a
    fisheye's terms live out there;
  * near (the board filling a third of the frame) and far;
  * tilted: left, right, up, down, 30-45 degrees;
  * the camera stays mounted where it will be used (the wrist camera on the
    arm, the overhead on the tower) — the board moves, not the camera.

It stops at `--views` captures (60) with the outer ring of the coverage map
at least 80% covered, calibrates, and prints the verdict.

## The verdict, and why it is not `rms`

`rms` is the fit's residual against the corners it was given; a small board
seen from few directions fits beautifully with a focal length and four terms
trading off. The number that decides is `spread`: the calibration is redone
on the views resampled with replacement (`fisheye_calib.undistort_spread`)
and the undistortion map to the sim pinhole moves by that much, rms, in
640x480 pixels. On the synthetic lens of `tests/vision/test_fisheye.mojo` it
tracked the true map error within 1.5x.

    spread < 0.3 px    good
    spread < 0.6 px    usable; more edge views would tighten it
    otherwise          redo — more views, closer, at the edges

## Detection on a 2x upscale (`--detect-scale`, default 2)

⚠ MEASURED ON THE RIG: at 640x480 the fisheye makes the board's 22 mm
markers a few pixels wide, the 4x4 ArUco codes stop decoding, and a board
the studio sees with 20 corners (it opens the camera at 1280x720) came back
with 1-3. So each frame is upscaled (PIL bilinear) for DETECTION ONLY and the
corners mapped back, `u = (u' + 0.5) / s - 0.5` (pixel-centre convention).
The calibration, the saved views and the size check stay at 640x480 — the
size the dataset was recorded at. It is not the studio's 1280x720 capture:
that is a different sensor mode (16:9, likely another crop), and a
calibration of it would not describe the recorded frames.

## Headless, over SSH

No window: a progress line and an ASCII coverage map in the terminal.
`--preview` writes the last capture raw and undistorted as PNGs next to the
calibration, so the result can be checked by eye (straight edges straight)
after `scp`.

⚠ 640x480 IS NOT A PREFERENCE. It is the size the rig recorded the dataset
at (`observation.images.*` = [480, 640, 3]) and the deploy captures at; a
calibration at another size is REFUSED by `calib_file.require_size`.
"""

from std.math import sqrt, atan2, log, pi
from std.os import makedirs, listdir
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns

from noeira.io.png import save_png
from noeira.vision.calib_file import CameraCalib, write_calib
from noeira.vision.fisheye import FisheyeLens, Pinhole, UndistortMap
from noeira.vision.fisheye_calib import (
    CalibViews, calibrate_fisheye, undistort_spread,
)
from noeira.vision.opencv import (
    CharucoBoard, VideoCapture, imread, opencv_shim_available,
)
from noeira.vision.preprocess import pil_bilinear_u8

comptime W = 640
comptime H = 480
comptime SIM_FOVY = 73.7398
"""The sim cameras' fovy (`so101_tower_stand.xml`, `so_arm101_tower.xml`) —
the pinhole the preview and the spread are computed for."""
comptime MIN_CORNERS = 12
comptime GRID_X = 8
comptime GRID_Y = 6
comptime DEFAULT_VIEWS = 60
comptime EDGE_COVERAGE = 0.8
comptime COOLDOWN_S = 0.6


def _usage() -> String:
    return String(
        "usage: calibrate_fisheye.mojo (--camera PATH | --device N | --images DIR)"
        " --name NAME [--out DIR] [--views N] [--board 5x7] [--square-mm 30]"
        " [--marker-mm 22] [--preview] [--max-minutes M] [--detect-scale 2]"
    )


struct Signature(Copyable, Movable):
    """What makes two captures different poses: where the board is, how big
    it looks, how it is foreshortened, how it is turned."""

    var cx: Float64
    var cy: Float64
    var log_size: Float64
    var aspect: Float64
    var angle: Float64

    def __init__(out self, ref corners: List[Float32], n: Int):
        var mx = 0.0
        var my = 0.0
        var x0 = 1e9
        var x1 = -1e9
        var y0 = 1e9
        var y1 = -1e9
        for i in range(n):
            var x = Float64(corners[i * 2])
            var y = Float64(corners[i * 2 + 1])
            mx += x
            my += y
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
        self.cx = mx / Float64(n) / Float64(W)
        self.cy = my / Float64(n) / Float64(H)
        var bw = max(x1 - x0, 1.0)
        var bh = max(y1 - y0, 1.0)
        self.log_size = log(sqrt(bw * bh))
        self.aspect = log(bw / bh)
        self.angle = atan2(
            Float64(corners[3]) - Float64(corners[1]),
            Float64(corners[2]) - Float64(corners[0]),
        )

    def distance(self, o: Self) -> Float64:
        var da = abs(self.angle - o.angle)
        if da > pi:
            da = 2.0 * pi - da
        var d2 = (
            ((self.cx - o.cx) / 0.08) ** 2 + ((self.cy - o.cy) / 0.08) ** 2
            + ((self.log_size - o.log_size) / 0.2) ** 2
            + ((self.aspect - o.aspect) / 0.2) ** 2 + (da / 0.35) ** 2
        )
        return sqrt(d2)


struct Coverage(Movable):
    var hit: List[Bool]

    def __init__(out self):
        self.hit = List[Bool](length=GRID_X * GRID_Y, fill=False)

    def add(mut self, ref corners: List[Float32], n: Int):
        for i in range(n):
            var gx = Int(Float64(corners[i * 2]) * Float64(GRID_X) / Float64(W))
            var gy = Int(Float64(corners[i * 2 + 1]) * Float64(GRID_Y) / Float64(H))
            if gx >= 0 and gx < GRID_X and gy >= 0 and gy < GRID_Y:
                self.hit[gy * GRID_X + gx] = True

    def edge_fraction(self) -> Float64:
        var n = 0
        var k = 0
        for gy in range(GRID_Y):
            for gx in range(GRID_X):
                if gx == 0 or gy == 0 or gx == GRID_X - 1 or gy == GRID_Y - 1:
                    n += 1
                    if self.hit[gy * GRID_X + gx]:
                        k += 1
        return Float64(k) / Float64(n)

    def draw(self) -> String:
        var s = String("")
        for gy in range(GRID_Y):
            s += "    "
            for gx in range(GRID_X):
                var edge = gx == 0 or gy == 0 or gx == GRID_X - 1 or gy == GRID_Y - 1
                s += ("##" if self.hit[gy * GRID_X + gx] else (".." if edge else "  "))
            s += "\n"
        return s^


def _pad3(n: Int) -> String:
    """Zero-padded, so the saved views sort in capture order."""
    var s = String(n)
    while s.byte_length() < 3:
        s = "0" + s
    return s^


def _detect(
    ref board: CharucoBoard, ref frame: List[UInt8], channels: Int, scale: Int,
    mut up: List[UInt8], mut corners: List[Float32], mut ids: List[Int32],
) raises -> Int:
    """ChArUco detection on a `scale`x upscale, corners mapped back to the
    native 640x480 pixel grid — see the header."""
    if scale <= 1:
        return board.detect(frame, W, H, channels, corners, ids)
    pil_bilinear_u8(frame, W, H, channels, up, W * scale, H * scale)
    var n = board.detect(up, W * scale, H * scale, channels, corners, ids)
    var s = Float32(scale)
    for i in range(n * 2):
        corners[i] = (corners[i] + 0.5) / s - 0.5
    return n


def _bgr_to_rgb(ref bgr: List[UInt8], n: Int) -> List[UInt8]:
    var out = List[UInt8](length=n * 3, fill=0)
    for i in range(n):
        out[i * 3] = bgr[i * 3 + 2]
        out[i * 3 + 1] = bgr[i * 3 + 1]
        out[i * 3 + 2] = bgr[i * 3]
    return out^


def _add_view(
    mut views: CalibViews, ref board_xyz: List[Float32],
    ref corners: List[Float32], ref ids: List[Int32], n: Int,
) raises:
    """Pair the VISIBLE corners with the board's by ID, never by position."""
    var o = List[Float64]()
    var p = List[Float64]()
    for i in range(n):
        var id = Int(ids[i])
        o.append(Float64(board_xyz[id * 3]))
        o.append(Float64(board_xyz[id * 3 + 1]))
        o.append(Float64(board_xyz[id * 3 + 2]))
        p.append(Float64(corners[i * 2]))
        p.append(Float64(corners[i * 2 + 1]))
    views.add(o^, p^)


def main() raises:
    var args = argv()
    var camera = String("")
    var device = -1
    var images = String("")
    var name = String("")
    var out_dir = String("projects/so101-tower/cameras")
    var target = DEFAULT_VIEWS
    var bx = 5
    var by = 7
    var square_mm = 30.0
    var marker_mm = 22.0
    var preview = False
    var max_minutes = 20.0
    var detect_scale = 2
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--preview":
            preview = True
            i += 1
            continue
        if i + 1 >= len(args):
            raise Error(a + " takes a value\n" + _usage())
        var v = String(args[i + 1])
        if a == "--camera":
            camera = v
        elif a == "--device":
            device = Int(v)
        elif a == "--images":
            images = v
        elif a == "--name":
            name = v
        elif a == "--out":
            out_dir = v
        elif a == "--views":
            target = Int(v)
        elif a == "--board":
            var xy = v.split("x")
            bx = Int(String(xy[0]))
            by = Int(String(xy[1]))
        elif a == "--square-mm":
            square_mm = Float64(v)
        elif a == "--marker-mm":
            marker_mm = Float64(v)
        elif a == "--max-minutes":
            max_minutes = Float64(v)
        elif a == "--detect-scale":
            detect_scale = Int(v)
        else:
            raise Error("unknown option " + a + "\n" + _usage())
        i += 2
    if name.byte_length() == 0:
        raise Error("--name is required (overhead | wrist)\n" + _usage())
    var sources = (1 if camera.byte_length() > 0 else 0) + (1 if device >= 0 else 0) + (
        1 if images.byte_length() > 0 else 0
    )
    if sources != 1:
        raise Error("exactly one of --camera, --device, --images\n" + _usage())
    if not opencv_shim_available():
        raise Error("the OpenCV shim is not built — `pixi run build-opencv`")

    print("=" * 72)
    print("fisheye intrinsics —", name, "| board", bx, "x", by, "squares,",
          square_mm, "/", marker_mm, "mm")
    print("=" * 72)
    var board = CharucoBoard(bx, by, Float32(square_mm / 1000.0), Float32(marker_mm / 1000.0))
    var board_xyz = List[Float32]()
    _ = board.board_corners(board_xyz)
    var views_dir = out_dir + "/" + name + "_views"
    makedirs(views_dir, exist_ok=True)
    var views = CalibViews()
    var corners = List[Float32]()
    var ids = List[Int32]()
    var frame = List[UInt8]()
    var up = List[UInt8]()
    var last_rgb = List[UInt8]()

    if images.byte_length() > 0:
        # ── offline: every PNG in the directory, no novelty filter ──────
        var names = listdir(images)
        var files = List[String]()
        for e in names:
            if String(e).endswith(".png"):
                files.append(String(e))
        for a in range(len(files)):
            for b in range(a + 1, len(files)):
                if files[b] < files[a]:
                    files[a], files[b] = files[b], files[a]
        for f in files:
            var whc = imread(images + "/" + f, frame)
            if whc[0] != W or whc[1] != H:
                raise Error(f + " is " + String(whc[0]) + "x" + String(whc[1])
                            + ", the calibration is at " + String(W) + "x" + String(H))
            var n = _detect(board, frame, whc[2], detect_scale, up, corners, ids)
            if n >= MIN_CORNERS:
                _add_view(views, board_xyz, corners, ids, n)
                last_rgb = _bgr_to_rgb(frame, W * H)
            print("  ", f, ":", n, "corners", "" if n >= MIN_CORNERS else "(skipped)")
    else:
        # ── live: capture on novelty ────────────────────────────────────
        var cap = VideoCapture.device_path(camera, W, H, 30.0, String("MJPG")) if camera.byte_length() > 0 else VideoCapture.device(device, W, H, 30.0)
        if cap.width != W or cap.height != H:
            cap.close()
            raise Error(
                "the camera negotiated " + String(cap.width) + "x" + String(cap.height)
                + ", not " + String(W) + "x" + String(H) + " — the size the dataset was"
                " recorded at. Refusing rather than calibrating another crop."
            )
        print("  camera", camera if camera.byte_length() > 0 else String(device), "|",
              cap.width, "x", cap.height, "| move the board; captures are automatic")
        var sigs = List[Signature]()
        var cov = Coverage()
        var t0 = perf_counter_ns()
        var t_last = 0
        var frames = 0
        var last_print = 0
        try:
            while True:
                if not cap.read(frame):
                    raise Error("the camera stopped delivering frames")
                frames += 1
                var now = Int(perf_counter_ns() - t0)
                if Float64(now) / 60e9 > max_minutes:
                    print("\n  time limit reached")
                    break
                var n = _detect(board, frame, 3, detect_scale, up, corners, ids)
                var status = String("no board")
                if n >= MIN_CORNERS:
                    var sg = Signature(corners, n)
                    var dmin = 1e9
                    for s in sigs:
                        dmin = min(dmin, sg.distance(s))
                    if dmin < 1.0:
                        status = String(n) + " corners, pose already captured — move/tilt"
                    elif Float64(now - t_last) / 1e9 < COOLDOWN_S:
                        status = String(n) + " corners, hold on"
                    else:
                        _add_view(views, board_xyz, corners, ids, n)
                        cov.add(corners, n)
                        sigs.append(sg^)
                        t_last = now
                        last_rgb = _bgr_to_rgb(frame, W * H)
                        save_png(
                            views_dir + "/view_" + _pad3(views.count()) + ".png",
                            last_rgb, W, H, 3,
                        )
                        print("\n  captured view", views.count(), "(", n, "corners ) | edge coverage",
                              Int(100.0 * cov.edge_fraction()), "%")
                        print(cov.draw())
                        if views.count() >= target and cov.edge_fraction() >= EDGE_COVERAGE:
                            break
                        continue
                elif n > 0:
                    status = String(n) + " corners (need " + String(MIN_CORNERS) + ")"
                if now - last_print > 500_000_000:
                    last_print = now
                    print("\r  views", views.count(), "/", target, "| edge",
                          Int(100.0 * cov.edge_fraction()), "% |", status, "          ", end="")
        finally:
            cap.close()
        print("")

    # ── calibrate ──────────────────────────────────────────────────────
    if views.count() < 10:
        raise Error("only " + String(views.count()) + " usable views — nothing to calibrate")
    print("  calibrating on", views.count(), "views ...")
    var fit = calibrate_fisheye(views, W, H)
    var pin = Pinhole.sim(SIM_FOVY, W, H)
    var spread = undistort_spread(views, fit, pin)
    var um = UndistortMap(fit.lens, pin)
    print("  lens  :", String(fit.lens))
    print("  rms   :", fit.rms, "px over", len(fit.used), "views")
    for j in range(len(fit.dropped)):
        print("  dropped view", fit.dropped[j] + 1, ":", fit.why[j])
    print("  spread:", spread, "px (the undistortion map's rms movement under resampling)")
    print("  out-of-lens pixels in the sim pinhole:", um.n_outside)
    var verdict = String("GOOD") if spread < 0.3 else (
        String("USABLE — more edge views would tighten it") if spread < 0.6
        else String("REDO — more views, closer, at the edges")
    )
    print("  verdict:", verdict)

    var calib = CameraCalib(name, device, W, H, fit.lens.fx, fit.lens.fy, fit.lens.cx, fit.lens.cy)
    calib.model = String("fisheye")
    calib.dist = fit.lens.d_vector()
    calib.rms_px = fit.rms
    makedirs(out_dir, exist_ok=True)
    var path = out_dir + "/camera_" + name + ".txt"
    write_calib(path, calib)
    print("  wrote", path)
    if preview and len(last_rgb) == W * H * 3:
        save_png(out_dir + "/" + name + "_preview_raw.png", last_rgb, W, H, 3)
        var und = List[UInt8](length=W * H * 3, fill=0)
        um.apply_hwc(last_rgb, 0, 3, und, 0)
        save_png(out_dir + "/" + name + "_preview_undistorted.png", und, W, H, 3)
        print("  wrote", out_dir + "/" + name + "_preview_{raw,undistorted}.png")
    if spread >= 0.6:
        raise Error("calibration spread " + String(spread) + " px — redo (see the header)")
    print("=== OK ===")
