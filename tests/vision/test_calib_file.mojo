# +--------------------------------------------------------------------------+ #
# | A calibration file that cannot be applied to the wrong picture
# +--------------------------------------------------------------------------+ #
"""Gate for `mojo_rl/vision/calib_file.mojo`.

    pixi run mojo run -I . tests/vision/test_calib_file.mojo

⚠ NEEDS NOTHING — no shim, no camera, no fixture. It writes a file, reads it
back, and then tries to misuse it in the four ways the six-float format it
replaces could not detect.

⚠⚠ **MOST OF THE CHECKS HERE ARE REFUSALS, AND THAT IS THE POINT.** A format
is not judged by what it round-trips — every format round-trips. It is judged
by what it REJECTS: a calibration measured at 1280x720 handed to a 640x480
frame is off by exactly 2x in `fx` AND `cx`, produces poses that look entirely
reasonable, and was undetectable before this file existed.
"""

from std.math import abs

from mojo_rl.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from mojo_rl.vision.calib_file import CameraCalib, read_calib, write_calib

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]

comptime TMP = "/tmp/mojo_rl_calib_gate.txt"


def _write_raw(path: String, var body: String) raises:
    with open(path, "w") as f:
        f.write(body)


def _raises(path: String) -> Bool:
    try:
        _ = read_calib(path)
        return False
    except:
        return True


def main() raises:
    print("=" * 70)
    print("camera calibration file — what it refuses")
    print("=" * 70)
    var checks = 0
    var failures = 0

    # ── a full calibration, out and back ────────────────────────────────────
    var c = CameraCalib(
        String("front"), 0, 640, 480, 557.2431, 554.9127, 321.7, 243.1
    )
    c.dist.append(-0.0421)
    c.dist.append(0.0037)
    c.rms_px = 0.107
    c.has_extrinsics = True
    c.rot = Mat3d.rotation_axis(Vec3d(0.37, -0.82, 0.44).normalized(), 0.7391)
    c.trans = Vec3d(0.31, -0.22, 0.455)
    c.rms_mm = 4.32
    c.poses = 14
    write_calib(TMP, c)
    var r = read_calib(TMP)

    checks += 1
    if r.name != "front" or r.device != 0 or r.width != 640 or r.height != 480:
        print("  FAIL: identity did not survive the round trip")
        failures += 1
    # ⚠ EXACT, NOT A TOLERANCE. Both sides are float64 and the file is decimal
    # text; if this ever needs a tolerance the WRITER has lost digits, and a
    # calibration quietly rounded is the defect, not the assertion.
    var worst = 0.0
    var d = [
        r.fx - c.fx, r.fy - c.fy, r.cx - c.cx, r.cy - c.cy,
        r.rms_px - c.rms_px, r.rms_mm - c.rms_mm,
        r.trans.x - c.trans.x, r.trans.y - c.trans.y, r.trans.z - c.trans.z,
        r.rot.m00 - c.rot.m00, r.rot.m11 - c.rot.m11, r.rot.m22 - c.rot.m22,
        r.rot.m01 - c.rot.m01, r.rot.m20 - c.rot.m20,
    ]
    for i in range(len(d)):
        if abs(d[i]) > worst:
            worst = abs(d[i])
    checks += 1
    if worst != 0.0:
        print("  FAIL: round trip lost", worst, "— the writer is dropping digits")
        failures += 1
    checks += 1
    if len(r.dist) != 2 or r.dist[0] != c.dist[0] or r.dist[1] != c.dist[1]:
        print("  FAIL: the distortion vector did not survive")
        failures += 1
    checks += 1
    if not r.has_extrinsics or r.poses != 14:
        print("  FAIL: extrinsics did not survive")
        failures += 1
    print("  round trip:  exact, worst delta", worst)

    # ── the k matrix is in OpenCV's places, not ours ────────────────────────
    var k = r.k_matrix()
    checks += 1
    if (
        k[0] != r.fx or k[2] != r.cx or k[4] != r.fy or k[5] != r.cy
        or k[1] != 0.0 or k[3] != 0.0 or k[6] != 0.0 or k[7] != 0.0
        or k[8] != 1.0
    ):
        print("  FAIL: k_matrix is not row-major [fx 0 cx; 0 fy cy; 0 0 1]")
        failures += 1
    print("  k matrix:    fx/cx/fy/cy in OpenCV's slots")

    # ── the transform is the one that was stored ────────────────────────────
    var p = Vec3d(0.05, -0.02, 0.62)
    var want = c.rot * p + c.trans
    var got = r.base_from_camera(p)
    checks += 1
    var mm = (got - want).length() * 1000.0
    if mm != 0.0:
        print("  FAIL: base_from_camera moved the point by", mm, "mm")
        failures += 1
    print("  transform:   agrees to", mm, "mm")

    # ── ⚠⚠ THE FOUR MISUSES THE OLD SIX-FLOAT FILE COULD NOT SEE ────────────
    checks += 1
    var refused_size = False
    try:
        r.require_size(1280, 720)
    except:
        refused_size = True
    if not refused_size:
        print("  FAIL: a 640x480 calibration accepted a 1280x720 frame —")
        print("        fx AND cx are both off by 2x and nothing said so")
        failures += 1
    checks += 1
    # ...and it must NOT refuse the size it was measured at.
    r.require_size(640, 480)

    checks += 1
    _write_raw(TMP, String("name front\nsize 640 480\nintrinsics 1 2 3 4\n"))
    if not _raises(TMP):
        print("  FAIL: a file with no magic line was accepted")
        failures += 1

    checks += 1
    _write_raw(
        TMP,
        String("mojo-rl-camera-calibration 99\nsize 640 480\n")
        + "intrinsics 1 2 3 4\n",
    )
    if not _raises(TMP):
        print("  FAIL: a future version was accepted by this reader")
        failures += 1

    checks += 1
    _write_raw(
        TMP,
        String("mojo-rl-camera-calibration 1\nintrinsics 557 554 320 240\n"),
    )
    if not _raises(TMP):
        print("  FAIL: a calibration with no image size was accepted")
        failures += 1

    checks += 1
    # ⚠ HALF A POSE IS NOT A PARTIAL ANSWER, it is a camera at the robot's
    # origin — a number that looks like a measurement.
    _write_raw(
        TMP,
        String("mojo-rl-camera-calibration 1\nsize 640 480\n")
        + "intrinsics 557 554 320 240\n"
        + "extrinsics_rot 1 0 0 0 1 0 0 0 1\n",
    )
    if not _raises(TMP):
        print("  FAIL: a rotation with no translation was accepted")
        failures += 1
    print("  refusals:    wrong size, no magic, bad version, no size, half a pose")

    # ── intrinsics alone are a legal file, and say so ───────────────────────
    _write_raw(
        TMP,
        String("mojo-rl-camera-calibration 1\n# measured 2026-09-09\n")
        + "name side\nsize 640 480\nintrinsics 557 554 320 240\n"
        + "future_key whatever 1 2 3\n",
    )
    var only = read_calib(TMP)
    checks += 1
    if only.has_extrinsics:
        print("  FAIL: an intrinsics-only file claimed extrinsics")
        failures += 1
    checks += 1
    var refused_no_ext = False
    try:
        _ = only.base_from_camera(p)
    except:
        refused_no_ext = True
    if not refused_no_ext:
        print("  FAIL: a file with no extrinsics placed a point anyway")
        failures += 1
    checks += 1
    if only.name != "side":
        print("  FAIL: an unknown key derailed the parse")
        failures += 1
    print("  intrinsics-only: legal, says so, refuses to place a point")

    print("-" * 70)
    if failures == 0:
        print("PASS —", checks, "checks")
    else:
        print("FAIL —", failures, "of", checks, "checks")
    print("=" * 70)
    # ⚠ THE RAISE IS THE GATE — `run_tests.sh` reads the exit code only.
    if failures != 0:
        raise String("calib file: ") + String(failures) + " checks failed"
