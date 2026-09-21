"""Render a marker at a KNOWN pose, so detection and pose have a truth to hit.

    pixi run python tools/vision/make_marker_fixture.py

⚠ SYNTHETIC ON PURPOSE. A photograph of a printed tag has no ground truth
attached to it — you would be gating a pose estimate against another pose
estimate. Here the marker is warped into the image BY a camera matrix and a
pose we chose, so `solvePnP` has a right answer to be wrong about.

⚠⚠ THAT GROUND TRUTH IS A SANITY CHECK, NOT THE PARITY CLAIM. The gate's
assertion is that the Mojo shim and Python cv2 agree BIT FOR BIT, because both
call the same dylib. Recovering the true pose to a millimetre is a separate,
weaker check that the fixture is a real detection problem rather than a blank
image — a vacuity guard, in other words.

The pose is deliberately OFF-AXIS. A marker facing the camera head-on sits in
the two-fold ambiguity of a square fiducial, where the orientation flips
between two solutions and a gate would be flaky for reasons that are geometry,
not code.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/vision"

DICT_ID = cv2.aruco.DICT_4X4_50   # 0
MARKER_ID = 7
MARKER_M = 0.040                  # 40 mm printed tag
IMG_W, IMG_H = 640, 480

# A plausible 640x480 webcam.
K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros((1, 5), dtype=np.float64)

# Ground truth: 35 cm away, off to one side and above, tilted in two axes.
#
# ⚠⚠ THE 180 DEGREE ROLL IS NOT DECORATION, IT IS THE CONVENTION. OpenCV's
# marker frame is X right, Y UP, Z out of the tag toward the viewer; the camera
# frame is X right, Y DOWN, Z forward. So a marker facing the camera head-on is
# `rvec = (pi, 0, 0)`, NOT zero. Rendering with `rvec = 0` draws the tag's BACK
# — a mirrored marker, which is not in the dictionary and is silently rejected.
# That cost a debugging round here: the scene held 8 352 non-background pixels
# and two REJECTED candidates, and `detectMarkers` returned None.
_TILT = np.array([0.25, -0.35, 0.10], dtype=np.float64)
RVEC = cv2.Rodrigues(
    cv2.Rodrigues(_TILT)[0] @ cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
)[0].ravel()
TVEC = np.array([0.045, -0.030, 0.350], dtype=np.float64)

# ⚠ FIVE MILLIMETRES, AND THAT IS NOT SLOP — IT IS MEASURED. Recovery on this
# fixture is 2.884 mm, and it does not improve with a bigger source tag or a
# half-pixel corner correction, because the limit is the DETECTOR: ArUco's
# default is CORNER_REFINE_NONE, so corners are contour-level on a ~65 px tag.
# Depth is a fiducial's weak axis by construction (sigma_z ~ z^2/(f*s)), which
# is exactly why the plan wants two cameras at near-orthogonal angles.
TVEC_TOLERANCE_MM = 5.0

# Marker corner order is OpenCV's: clockwise from top-left, in the marker's own
# frame, Z up out of the tag. ⚠ THIS ORDER IS THE CONTRACT — solvePnP pairs
# object and image points POSITIONALLY, so reordering one without the other
# yields a plausible pose that is silently rotated.
H = MARKER_M / 2.0
OBJ = np.array([[-H,  H, 0.0],
                [ H,  H, 0.0],
                [ H, -H, 0.0],
                [-H, -H, 0.0]], dtype=np.float64)


def main() -> int:
    cv2.setNumThreads(1)
    FIX.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    # ⚠ THE WHITE QUIET ZONE IS NOT DECORATION. `generateImageMarker` returns
    # the tag with its black border and NOTHING around it; ArUco finds
    # candidates by contour, so a black border touching a non-white background
    # is a marker that simply is not detected. Without this pad the fixture
    # produced ids=None and would have gated an empty image.
    TAG, PAD = 400, 80
    tag = np.full((TAG + 2 * PAD, TAG + 2 * PAD), 255, dtype=np.uint8)
    tag[PAD:PAD + TAG, PAD:PAD + TAG] = cv2.aruco.generateImageMarker(
        dictionary, MARKER_ID, TAG)
    tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)

    # Where the tag's four corners land under the chosen pose.
    img_pts, _ = cv2.projectPoints(OBJ, RVEC, TVEC, K, DIST)
    img_pts = img_pts.reshape(4, 2).astype(np.float32)

    # Warp the flat tag onto that quad. A light grey background, not white:
    # a marker whose quiet zone merges into the border is a detection the
    # threshold step can lose, and the gate would then be testing nothing.
    # The tag's OWN corners inside the padded canvas, not the canvas corners.
    # The tag's true geometric edges sit at the OUTER pixel boundaries, half a
    # pixel outside the first and last pixel centres.
    src = np.array([[PAD - 0.5, PAD - 0.5],
                    [PAD + TAG - 0.5, PAD - 0.5],
                    [PAD + TAG - 0.5, PAD + TAG - 0.5],
                    [PAD - 0.5, PAD + TAG - 0.5]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(src, img_pts)
    scene = np.full((IMG_H, IMG_W, 3), 190, dtype=np.uint8)
    warped = cv2.warpPerspective(tag, Hm, (IMG_W, IMG_H),
                                 borderMode=cv2.BORDER_TRANSPARENT,
                                 dst=scene.copy())
    scene = warped

    out = FIX / "marker_640x480.png"
    if not cv2.imwrite(str(out), scene):
        print("imwrite failed", file=sys.stderr)
        return 1

    # Prove the fixture is solvable before pinning anything against it.
    det = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = det.detectMarkers(scene)
    if ids is None or len(ids) != 1 or int(ids[0]) != MARKER_ID:
        print(f"fixture is not detectable: ids={ids}", file=sys.stderr)
        return 1

    ok, rvec, tvec = cv2.solvePnP(
        OBJ, corners[0].reshape(4, 2).astype(np.float64), K, DIST,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        print("solvePnP failed on the fixture", file=sys.stderr)
        return 1
    err_mm = float(np.linalg.norm(tvec.ravel() - TVEC) * 1000.0)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"  detected id {int(ids[0])}, recovered tvec within {err_mm:.3f} mm")
    if err_mm > TVEC_TOLERANCE_MM:
        print(f"  ⚠ recovery worse than {TVEC_TOLERANCE_MM} mm —"
              " the fixture is not a clean detection problem", file=sys.stderr)
        return 1

    (FIX / "marker_640x480_truth.txt").write_text(
        f"dict_id {DICT_ID}\nmarker_id {MARKER_ID}\nmarker_m {MARKER_M}\n"
        f"rvec {RVEC[0]!r} {RVEC[1]!r} {RVEC[2]!r}\n"
        f"tvec {TVEC[0]} {TVEC[1]} {TVEC[2]}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
