"""Render N views of a ChArUco board from a KNOWN camera, for the calibration gate.

    pixi run python tools/vision/make_charuco_fixture.py

⚠ SYNTHETIC, AND THE CAMERA IS THE POINT. Calibrating photographs of a real
board gives an intrinsic matrix with nothing to check it against. Here the
views are projected THROUGH a chosen `K`, so `calibrateCamera` has a right
answer — and a calibration that returns something else is a defect rather than
a fact about a lens.

⚠⚠ AS EVERYWHERE IN THIS GATE, THE TRUTH IS A VACUITY GUARD, NOT THE CLAIM.
What the Mojo gate asserts is bit equality with Python cv2 on the same inputs.
Recovering `K` proves the fixture is a real calibration problem rather than a
degenerate one, which a bit-equality test alone cannot tell you.

⚠ VIEW DIVERSITY IS A CONDITIONING REQUIREMENT, NOT A STYLE CHOICE. Views that
all face the board head-on leave focal length and distortion nearly
unidentifiable; the tilts below are deliberately spread in two axes.

⚠ NO LENS DISTORTION IS RENDERED. A homography warp cannot produce it — the
image of a plane through a distorting lens is not a homography of the plane.
So the fixture's truth is `dist = 0`, and the gate uses CALIB_ZERO_TANGENT_DIST
| CALIB_FIX_K3 rather than pretending to recover coefficients nothing drew.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/vision"

DICT_ID = cv2.aruco.DICT_4X4_50
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_M, MARKER_M = 0.030, 0.022
IMG_W, IMG_H = 640, 480

# The camera the views are drawn through — what calibration must find.
K_TRUE = np.array([[600.0, 0.0, 320.0],
                   [0.0, 600.0, 240.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

# (tilt rvec, tvec). Spread in both tilt axes and in depth.
VIEWS = [
    ([0.00,  0.00, 0.0], [0.00,  0.00, 0.45]),
    ([0.35, -0.20, 0.1], [0.02, -0.01, 0.40]),
    ([-0.30, 0.25, 0.0], [-0.02, 0.02, 0.42]),
    ([0.20,  0.40, -0.2], [0.03, 0.00, 0.38]),
    ([-0.40, -0.30, 0.15], [-0.01, -0.02, 0.44]),
    ([0.45,  0.10, 0.3], [0.00,  0.03, 0.36]),
    ([-0.15, 0.45, -0.1], [0.01, -0.03, 0.47]),
    ([0.10, -0.45, 0.2], [-0.03, 0.01, 0.39]),
    ([0.30,  0.30, 0.0], [0.00,  0.00, 0.50]),
    # ⚠⚠ THE LAST TWO ARE PARTIAL VIEWS, AND THEY ARE NOT OPTIONAL.
    # `detectBoard` returns ONLY VISIBLE corners with their ids, so the caller
    # must pair them with the board's 3D points BY ID. With nine full views
    # every detection came back as ids 0..23 in order, which makes positional
    # pairing accidentally correct — measured: swapping the gate to pair
    # positionally still PASSED, bit for bit. These two views push the board
    # partly out of frame so the ids are non-contiguous (18 corners starting at
    # 1, with gaps) and offset (14 corners starting at 10), which is what makes
    # the by-id requirement load-bearing in the gate.
    ([0.30, -0.30, 0.1], [-0.070, 0.02, 0.28]),
    ([0.25,  0.15, 0.2], [0.000, -0.06, 0.27]),
]


def main() -> int:
    cv2.setNumThreads(1)
    FIX.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_M, MARKER_M, dictionary)

    # Render the board flat, with a white margin so the outer markers have the
    # quiet zone the detector needs — the same trap the marker fixture hit.
    px_per_m = 4000.0
    bw = int(round(SQUARES_X * SQUARE_M * px_per_m))
    bh = int(round(SQUARES_Y * SQUARE_M * px_per_m))
    flat = board.generateImage((bw, bh))
    flat = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)

    # The board plane's four outer corners, in board coordinates. ⚠ Y DOWN:
    # `getChessboardCorners` puts the board in the XY plane with +Y going down
    # the image of the board as generated, so the rendering frame must agree
    # with the frame calibration will use, or every view is mirrored.
    W_M, H_M = SQUARES_X * SQUARE_M, SQUARES_Y * SQUARE_M
    obj_outer = np.array([[0.0, 0.0, 0.0],
                          [W_M, 0.0, 0.0],
                          [W_M, H_M, 0.0],
                          [0.0, H_M, 0.0]], dtype=np.float64)
    src = np.array([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]],
                   dtype=np.float32)

    # Centre the board on its own middle so the tilts rotate about the board,
    # not about a corner that would swing it out of frame.
    centre = np.array([W_M / 2.0, H_M / 2.0, 0.0])

    written = []
    for i, (rv, tv) in enumerate(VIEWS):
        R = cv2.Rodrigues(np.asarray(rv, dtype=np.float64))[0]
        t = np.asarray(tv, dtype=np.float64) - R @ centre
        pts, _ = cv2.projectPoints(obj_outer, cv2.Rodrigues(R)[0], t,
                                   K_TRUE, np.zeros((1, 5)))
        dst = pts.reshape(4, 2).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        scene = np.full((IMG_H, IMG_W, 3), 190, dtype=np.uint8)
        scene = cv2.warpPerspective(flat, Hm, (IMG_W, IMG_H), dst=scene,
                                    borderMode=cv2.BORDER_TRANSPARENT)
        out = FIX / f"charuco_{i:02d}.png"
        if not cv2.imwrite(str(out), scene):
            print(f"imwrite failed for {out}", file=sys.stderr)
            return 1
        written.append(out)

    # Prove every view is usable before pinning anything against them.
    det = cv2.aruco.CharucoDetector(board)
    counts = []
    for out in written:
        img = cv2.imread(str(out), cv2.IMREAD_COLOR)
        c, ids, _, _ = det.detectBoard(img)
        n = 0 if ids is None else len(ids)
        counts.append(n)
        if n < 6:
            print(f"{out.name}: only {n} corners — not a usable view",
                  file=sys.stderr)
            return 1

    total = sum(counts)
    print(f"wrote {len(written)} views, {total} charuco corners total")
    print("  per view:", counts)
    (FIX / "charuco_truth.txt").write_text(
        f"squares_x {SQUARES_X}\nsquares_y {SQUARES_Y}\n"
        f"square_m {SQUARE_M}\nmarker_m {MARKER_M}\ndict_id {DICT_ID}\n"
        f"views {len(written)}\nimg_w {IMG_W}\nimg_h {IMG_H}\n"
        f"fx {K_TRUE[0, 0]}\nfy {K_TRUE[1, 1]}\n"
        f"cx {K_TRUE[0, 2]}\ncy {K_TRUE[1, 2]}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
