"""Gate the OpenCV shim's lifecycle and capture groups against Python `cv2`.

    pixi run build-opencv
    pixi run mojo run -I . tests/vision/test_opencv_shim.mojo

⚠⚠ THIS GATE ASSERTS EXACT BIT EQUALITY, AND THAT IS THE POINT OF WRAPPING
RATHER THAN PORTING. The Mojo shim and Python `cv2` call the SAME dylib, so any
difference is a marshalling bug and never a numerical one — there is no
tolerance anywhere below, and there must not be. (Contrast the ACT port, which
needs 2.4e-6 because it is a reimplementation. A rewrite of OpenCV's algorithms
in Mojo would forfeit exactly this property.)

⚠ NO CAMERA IS REQUIRED. A capture path tested only against live hardware has
no gate at all: the frames are never the same twice, so there is nothing to
compare. This decodes a committed 3 KB video through both bindings instead.

Fixtures, from `tools/vision/{make_capture_fixture,dump_cv_reference}.py`:
    tests/fixtures/vision/capture_12f.mp4          the input
    tests/fixtures/vision/capture_12f_bgr.bin      cv2's decode, all frames
    tests/fixtures/vision/capture_12f_rgb_chw.bin  cv2's RGB CHW, frame 0
"""

from std.pathlib import Path

from mojo_rl.vision.opencv import (
    VideoCapture,
    bgr_hwc_to_rgb_chw,
    cv_set_num_threads,
    cv_version,
    opencv_shim_available,
)

comptime FIX = "tests/fixtures/vision/"
comptime N_FRAMES = 12
comptime W = 64
comptime H = 48
comptime C = 3
comptime FRAME_BYTES = W * H * C


def read_bin(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var out = List[UInt8](capacity=len(raw))
        for i in range(len(raw)):
            out.append(raw[i])
        return out^


def main() raises:
    print("=" * 70)
    print("OpenCV shim — lifecycle + capture, against Python cv2")
    print("=" * 70)

    if not opencv_shim_available():
        print("SKIP: shim not built. Run: pixi run build-opencv")
        return

    var failures = 0
    var checks = 0

    # ── A: the dylib loads and is the version this was scoped against ───────
    var ver = cv_version()
    checks += 1
    print("  OpenCV", ver[0], ".", ver[1])
    if ver[0] != 5:
        print("  FAIL: expected OpenCV 5 — 4.x moved solvePnP and still has")
        print("        calibrateHandEye, so the scope document does not apply")
        failures += 1

    # Identical scheduling on both sides; see the module docstring.
    cv_set_num_threads(1)

    # ── the pinned reference ────────────────────────────────────────────────
    var ref_path = String(FIX) + "capture_12f_bgr.bin"
    if not Path(ref_path).exists():
        print("SKIP: no reference. Run:")
        print("  pixi run python tools/vision/dump_cv_reference.py")
        return
    var expected = read_bin(ref_path)
    var expect_bytes = N_FRAMES * FRAME_BYTES
    checks += 1
    if len(expected) != expect_bytes:
        print(
            "  FAIL: reference is",
            len(expected),
            "bytes, expected",
            expect_bytes,
        )
        failures += 1
        return

    # ── B: geometry, before a single frame is read ──────────────────────────
    var cap = VideoCapture.from_file(String(FIX) + "capture_12f.mp4")
    checks += 1
    if cap.width != W or cap.height != H:
        print(
            "  FAIL: geometry",
            cap.width,
            "x",
            cap.height,
            "expected",
            W,
            "x",
            H,
        )
        failures += 1
    checks += 1
    if cap.frame_count != N_FRAMES:
        print("  FAIL: frame_count", cap.frame_count, "expected", N_FRAMES)
        failures += 1
    print(
        "  file:",
        cap.width,
        "x",
        cap.height,
        "@",
        cap.fps,
        "fps,",
        cap.frame_count,
        "frames",
    )

    # ── B: every frame, byte for byte ───────────────────────────────────────
    var buf = List[UInt8]()
    var n_read = 0
    var bytes_compared = 0
    var bytes_differing = 0
    var first_bad = -1
    var frame0 = List[UInt8]()

    while True:
        var ok = cap.read(buf)
        if not ok:
            break
        if n_read >= N_FRAMES:
            n_read += 1
            break
        if n_read == 0:
            frame0 = buf.copy()
        var base = n_read * FRAME_BYTES
        for i in range(FRAME_BYTES):
            bytes_compared += 1
            if buf[i] != expected[base + i]:
                bytes_differing += 1
                if first_bad < 0:
                    first_bad = base + i
        n_read += 1
    cap.close()

    # ⚠ VACUITY GUARD. "0 bytes differed" and "nothing was compared" are the
    # same output, and this tree has been bitten by that more than once. The
    # count of bytes COMPARED is printed beside the count DIFFERING, and a run
    # that read no frames fails rather than passing silently.
    checks += 1
    if n_read != N_FRAMES:
        print("  FAIL: read", n_read, "frames, expected", N_FRAMES)
        failures += 1
    checks += 1
    if bytes_compared != expect_bytes:
        print(
            "  FAIL: compared", bytes_compared, "bytes, expected", expect_bytes
        )
        failures += 1
    checks += 1
    if bytes_differing != 0:
        print(
            "  FAIL:",
            bytes_differing,
            "of",
            bytes_compared,
            "bytes differ; first at offset",
            first_bad,
        )
        failures += 1
    print(
        "  decode: compared",
        bytes_compared,
        "bytes,",
        bytes_differing,
        "differ",
    )

    # ── the ONE conversion, pinned ──────────────────────────────────────────
    # BGR HWC -> RGB CHW is silent when wrong: the size does not change, no
    # error is raised, and the image still looks like an image.
    var chw_ref = read_bin(String(FIX) + "capture_12f_rgb_chw.bin")
    var chw = List[UInt8]()
    bgr_hwc_to_rgb_chw(frame0, chw, W, H)
    var chw_compared = 0
    var chw_differing = 0
    checks += 1
    if len(chw_ref) != FRAME_BYTES:
        print("  FAIL: CHW reference is", len(chw_ref), "bytes")
        failures += 1
    else:
        for i in range(FRAME_BYTES):
            chw_compared += 1
            if chw[i] != chw_ref[i]:
                chw_differing += 1
        checks += 1
        if chw_differing != 0:
            print(
                "  FAIL:", chw_differing, "of", chw_compared, "CHW bytes differ"
            )
            failures += 1
    print(
        "  convert: compared", chw_compared, "bytes,", chw_differing, "differ"
    )

    # ── close is idempotent ─────────────────────────────────────────────────
    # A control loop's `finally` can run twice on some abort paths, and a
    # double release that crashes is worse than the abort it was cleaning up.
    cap.close()
    checks += 1

    print("-" * 70)
    if failures == 0:
        print(
            "PASS —",
            checks,
            "checks,",
            bytes_compared + chw_compared,
            "bytes compared exactly",
        )
    else:
        print("FAIL —", failures, "of", checks, "checks")
    print("=" * 70)
