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
    ArucoDetector,
    DICT_4X4_50,
    SOLVEPNP_IPPE_SQUARE,
    VideoCapture,
    bgr_hwc_to_rgb_chw,
    cv_set_num_threads,
    cv_version,
    imread,
    opencv_shim_available,
    rodrigues,
    solve_pnp,
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

    # ═══════════════════════════════════════════════════════════════════════
    # C + D — detection and pose, against cv2's pinned answer
    # ═══════════════════════════════════════════════════════════════════════
    #
    # The fixture is a marker RENDERED at a chosen pose, so there is a ground
    # truth to sanity-check against. ⚠ That truth is NOT the parity claim: the
    # assertion below is bit equality with cv2, and the pose recovery is only a
    # vacuity guard proving the image is a real detection problem.
    var img = List[UInt8]()
    var geom = imread(String(FIX) + "marker_640x480.png", img)
    var iw = geom[0]
    var ih = geom[1]
    var ic = geom[2]
    checks += 1
    if iw != 640 or ih != 480 or ic != 3:
        print("  FAIL: marker image is", iw, "x", ih, "x", ic)
        failures += 1

    # ⚠ THE DICTIONARY ARGUMENT IS ONLY GATED ACROSS FAMILIES. Measured:
    # passing DICT_4X4_100 here still detects this marker as id 7, because the
    # 4X4 families share their codes — that is ArUco, not a plumbing bug. What
    # proves `dict_id` reaches OpenCV is a cross-family swap: DICT_6X6_250
    # detects 0 markers, and the gate goes red.
    var det = ArucoDetector(DICT_4X4_50)
    var ids = List[Int32]()
    var corners = List[Float32]()
    var n_markers = det.detect(img, iw, ih, ic, ids, corners)
    det.close()

    # ⚠ VACUITY: a detector that finds nothing agrees perfectly with a detector
    # that finds nothing, so the count is asserted before anything is compared.
    checks += 1
    if n_markers != 1:
        print("  FAIL: detected", n_markers, "markers, expected 1")
        failures += 1
    checks += 1
    if n_markers > 0 and ids[0] != 7:
        print("  FAIL: marker id", ids[0], "expected 7")
        failures += 1

    # cv2's corners, pinned. Contour-level integers — ArUco's default is
    # CORNER_REFINE_NONE, which is also why pose recovery lands at ~3 mm.
    var cv2_corners = List[Float32]()
    cv2_corners.append(372.0)
    cv2_corners.append(151.0)
    cv2_corners.append(433.0)
    cv2_corners.append(158.0)
    cv2_corners.append(421.0)
    cv2_corners.append(224.0)
    cv2_corners.append(361.0)
    cv2_corners.append(219.0)
    var corner_diff = 0
    checks += 1
    for i in range(8):
        if corners[i] != cv2_corners[i]:
            corner_diff += 1
    if corner_diff != 0:
        print("  FAIL:", corner_diff, "of 8 corner floats differ from cv2")
        failures += 1
    # ⚠ `n_markers`, NOT a hardcoded 1 — a failing run must not print a count
    # it did not measure.
    print(
        "  detect:",
        n_markers,
        "marker(s), first id",
        ids[0],
        "— 8 corner floats,",
        corner_diff,
        "differ",
    )

    # ── pose ────────────────────────────────────────────────────────────────
    # ⚠ THE OBJECT-POINT ORDER MIRRORS THE DETECTOR'S CORNER ORDER. solvePnP
    # pairs them positionally; swapping either list alone gives a plausible
    # pose that is silently rotated.
    comptime HALF = 0.020  # 40 mm marker
    var obj = List[Float64]()
    obj.append(-HALF)
    obj.append(HALF)
    obj.append(0.0)
    obj.append(HALF)
    obj.append(HALF)
    obj.append(0.0)
    obj.append(HALF)
    obj.append(-HALF)
    obj.append(0.0)
    obj.append(-HALF)
    obj.append(-HALF)
    obj.append(0.0)
    var img_xy = List[Float64]()
    for i in range(8):
        img_xy.append(Float64(corners[i]))
    var k = List[Float64]()
    k.append(600.0)
    k.append(0.0)
    k.append(320.0)
    k.append(0.0)
    k.append(600.0)
    k.append(240.0)
    k.append(0.0)
    k.append(0.0)
    k.append(1.0)
    var dist = List[Float64]()
    var rvec = List[Float64]()
    var tvec = List[Float64]()
    solve_pnp(obj, img_xy, k, dist, rvec, tvec, SOLVEPNP_IPPE_SQUARE)

    # cv2's pose on the same corners, pinned to full precision.
    var cv2_rvec_0 = -2.8515701846482786
    var cv2_rvec_1 = -0.14888728701423987
    var cv2_rvec_2 = -0.5081833926583355
    var cv2_tvec_0 = 0.045446408162437946
    var cv2_tvec_1 = -0.030301192643819412
    var cv2_tvec_2 = 0.3528335128789826
    var pose_diff = 0
    checks += 1
    if rvec[0] != cv2_rvec_0:
        pose_diff += 1
    if rvec[1] != cv2_rvec_1:
        pose_diff += 1
    if rvec[2] != cv2_rvec_2:
        pose_diff += 1
    if tvec[0] != cv2_tvec_0:
        pose_diff += 1
    if tvec[1] != cv2_tvec_1:
        pose_diff += 1
    if tvec[2] != cv2_tvec_2:
        pose_diff += 1
    if pose_diff != 0:
        print("  FAIL:", pose_diff, "of 6 pose values differ from cv2")
        print("        rvec", rvec[0], rvec[1], rvec[2])
        print("        tvec", tvec[0], tvec[1], tvec[2])
        failures += 1
    print("  pose:  6 values,", pose_diff, "differ")

    # ⚠ A SANITY CHECK, NOT THE PARITY CLAIM. The marker was rendered at a
    # known translation; recovery is 2.884 mm and does NOT improve with a
    # bigger source tag, because the limit is the detector's contour-level
    # corners. Depth is a fiducial's weak axis by construction.
    var dx = tvec[0] - 0.045
    var dy = tvec[1] - (-0.030)
    var dz = tvec[2] - 0.350
    var err_mm = ((dx * dx + dy * dy + dz * dz) ** 0.5) * 1000.0
    checks += 1
    if err_mm > 5.0:
        print("  FAIL: recovered pose is", err_mm, "mm from the truth")
        failures += 1
    print("  truth: recovered within", err_mm, "mm of the rendered pose")

    # Rodrigues: a rotation matrix must be orthonormal. Cheap, and it catches a
    # row/column-major mix-up that a pose comparison alone would not.
    var r9 = List[Float64]()
    rodrigues(rvec, r9)
    var det3 = (
        r9[0] * (r9[4] * r9[8] - r9[5] * r9[7])
        - r9[1] * (r9[3] * r9[8] - r9[5] * r9[6])
        + r9[2] * (r9[3] * r9[7] - r9[4] * r9[6])
    )
    checks += 1
    if abs(det3 - 1.0) > 1e-12:
        print("  FAIL: rodrigues determinant is", det3, "not 1")
        failures += 1
    print("  rodrigues: det =", det3)

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
