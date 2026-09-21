"""Decode the capture fixture with Python `cv2` and pin the result.

    pixi run python tools/vision/dump_cv_reference.py

⚠ WHY THIS IS COMMITTED RATHER THAN COMPUTED AT GATE TIME. The Mojo gate must
run with no Python in the process — the same discipline the ACT gates follow,
where the reference tensors are committed so the suite needs no torch. What is
pinned here is what OpenCV's decoder returns, so the gate's claim is "the Mojo
shim marshals the SAME BYTES the Python binding does".

⚠⚠ AND THAT CLAIM IS EXACT, NOT APPROXIMATE. Both sides call the SAME dylib,
so any difference is a marshalling bug and never a numerical one. There is no
tolerance anywhere in this gate, and there must not be: a tolerance here would
hide precisely the class of defect the shim can have.

Outputs, into tests/fixtures/vision/:
    capture_12f_bgr.bin      all 12 frames, BGR HWC uint8, contiguous
    capture_12f_rgb_chw.bin  frame 0 only, converted to RGB CHW
    capture_12f_meta.txt     frames, width, height, channels
    marker_640x480_cv2.txt   ids, corners, rvec, tvec — cv2's answer
    charuco_cv2.txt          per-view corner ids/xy, K, dist, rms — cv2's answer
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/vision"
SRC = FIX / "capture_12f.mp4"


def dump_marker() -> int:
    """Pin cv2's detection and pose on the marker fixture.

    ⚠ THE CORNERS ARE float32 AND THE POSE IS float64, and they are written
    with `repr` so the text round-trips EXACTLY. A shortened decimal here would
    silently turn a bit-equality gate into a tolerance one.
    """
    png = FIX / "marker_640x480.png"
    if not png.exists():
        print(f"missing {png}; run make_marker_fixture.py first", file=sys.stderr)
        return 1
    truth = dict(
        line.split(None, 1)
        for line in (FIX / "marker_640x480_truth.txt").read_text().splitlines()
    )
    dict_id = int(truth["dict_id"])
    marker_m = float(truth["marker_m"])

    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if img is None:
        print(f"cannot read {png}", file=sys.stderr)
        return 1

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id))
    corners, ids, _ = det.detectMarkers(img)
    if ids is None or len(ids) == 0:
        print("detected nothing — the gate would be vacuous", file=sys.stderr)
        return 1

    h = marker_m / 2.0
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
                   dtype=np.float64)
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
                 dtype=np.float64)
    img_pts = corners[0].reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj, img_pts, K, np.zeros((1, 5)), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        print("solvePnP failed", file=sys.stderr)
        return 1

    lines = [f"n {len(ids)}", "ids " + " ".join(str(int(i)) for i in ids.ravel())]
    c = np.asarray(corners[0], dtype=np.float32).reshape(-1)
    lines.append("corners " + " ".join(repr(float(v)) for v in c))
    lines.append("rvec " + " ".join(repr(float(v)) for v in rvec.ravel()))
    lines.append("tvec " + " ".join(repr(float(v)) for v in tvec.ravel()))
    (FIX / "marker_640x480_cv2.txt").write_text("\n".join(lines) + "\n")
    print(f"pinned marker: {len(ids)} marker(s), id {int(ids.ravel()[0])}")
    return 0


def dump_charuco() -> int:
    """Pin cv2's ChArUco detection and calibration on the 9 committed views.

    ⚠ EVERY FLOAT IS WRITTEN WITH `repr`. `calibrateCamera` is an iterative
    fit, so a shortened decimal would quietly turn the gate's bit-equality
    claim into a tolerance one.
    """
    truth_path = FIX / "charuco_truth.txt"
    if not truth_path.exists():
        print("run make_charuco_fixture.py first", file=sys.stderr)
        return 1
    t = dict(line.split(None, 1) for line in truth_path.read_text().splitlines())
    sx, sy = int(t["squares_x"]), int(t["squares_y"])
    sq, mk = float(t["square_m"]), float(t["marker_m"])
    n_views, iw, ih = int(t["views"]), int(t["img_w"]), int(t["img_h"])

    board = cv2.aruco.CharucoBoard(
        (sx, sy), sq, mk,
        cv2.aruco.getPredefinedDictionary(int(t["dict_id"])))
    det = cv2.aruco.CharucoDetector(board)
    bc = board.getChessboardCorners()

    lines = [f"views {n_views}", f"board_corners {len(bc)}"]
    obj_all, img_all, all_ids = [], [], []
    for i in range(n_views):
        img = cv2.imread(str(FIX / f"charuco_{i:02d}.png"), cv2.IMREAD_COLOR)
        if img is None:
            print(f"missing view {i}", file=sys.stderr)
            return 1
        corners, ids, _, _ = det.detectBoard(img)
        if ids is None or len(ids) < 6:
            print(f"view {i} detected too little", file=sys.stderr)
            return 1
        ids_flat = [int(v) for v in ids.ravel()]
        xy = np.asarray(corners, dtype=np.float32).reshape(-1)
        lines.append(f"view {i} n {len(ids_flat)}")
        lines.append("  ids " + " ".join(str(v) for v in ids_flat))
        lines.append("  xy " + " ".join(repr(float(v)) for v in xy))
        all_ids.append(ids_flat)
        obj_all.append(np.array([bc[k] for k in ids_flat], dtype=np.float32))
        img_all.append(corners.reshape(-1, 2).astype(np.float32))

    # ⚠ THE FLAGS ARE PART OF THE PINNED ANSWER. A homography warp cannot
    # render lens distortion, so the fixture's truth is dist = 0 and fitting a
    # full model would be fitting corner noise. The Mojo side passes the same
    # integer, which is why it is written into the file rather than assumed.
    flags = int(cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_all, img_all, (iw, ih), None, None, flags=flags)
    lines.append(f"flags {flags}")
    lines.append("K " + " ".join(repr(float(v)) for v in K.reshape(-1)))
    lines.append(f"n_dist {dist.size}")
    lines.append("dist " + " ".join(repr(float(v)) for v in dist.reshape(-1)))
    lines.append("rms " + repr(float(rms)))
    (FIX / "charuco_cv2.txt").write_text("\n".join(lines) + "\n")

    # ⚠ THE BINARIES ARE WHAT THE GATE READS; the text above is for humans.
    # A gate that parsed decimal text would be comparing whatever the
    # formatter chose to print, which is a tolerance wearing a disguise.
    counts = np.array([len(o) for o in obj_all], dtype=np.int32)
    (FIX / "charuco_cv2_counts.bin").write_bytes(counts.tobytes())
    (FIX / "charuco_cv2_ids.bin").write_bytes(
        np.concatenate([np.array(a, dtype=np.int32) for a in all_ids]).tobytes())
    (FIX / "charuco_cv2_xy.bin").write_bytes(
        np.concatenate([c.reshape(-1) for c in img_all]).astype(np.float32).tobytes())
    (FIX / "charuco_cv2_obj.bin").write_bytes(
        np.concatenate([o.reshape(-1) for o in obj_all]).astype(np.float64).tobytes())
    (FIX / "charuco_cv2_K.bin").write_bytes(
        np.asarray(K, dtype=np.float64).reshape(-1).tobytes())
    (FIX / "charuco_cv2_dist.bin").write_bytes(
        np.asarray(dist, dtype=np.float64).reshape(-1).tobytes())
    (FIX / "charuco_cv2_rms.bin").write_bytes(
        np.array([rms], dtype=np.float64).tobytes())
    print(f"pinned charuco: {n_views} views, rms {rms:.6f}, "
          f"fx {K[0, 0]:.3f} (truth {t['fx']})")
    return 0


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}; run make_capture_fixture.py first", file=sys.stderr)
        return 1

    # ⚠ SINGLE-THREADED ON BOTH SIDES. Not a performance choice: OpenCV's
    # parallel_for_ can change reduction order with the thread count, and a
    # bit-equality gate is a claim about identical inputs AND identical
    # scheduling. The Mojo side calls cv_set_num_threads(1) for the same reason.
    cv2.setNumThreads(1)

    cap = cv2.VideoCapture(str(SRC))
    if not cap.isOpened():
        print(f"cannot open {SRC}", file=sys.stderr)
        return 1

    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(f))
    cap.release()

    if not frames:
        print("decoded 0 frames — the gate would be vacuous", file=sys.stderr)
        return 1

    h, w, c = frames[0].shape
    for i, f in enumerate(frames):
        if f.shape != (h, w, c):
            print(f"frame {i} changed shape: {f.shape}", file=sys.stderr)
            return 1

    bgr = np.stack(frames)  # (N, H, W, C), BGR HWC
    (FIX / "capture_12f_bgr.bin").write_bytes(bgr.tobytes())

    # The conversion the Mojo side must reproduce exactly, pinned for frame 0.
    # BGR HWC -> RGB CHW: reverse the channel axis, then move it to the front.
    rgb_chw = np.ascontiguousarray(frames[0][:, :, ::-1].transpose(2, 0, 1))
    (FIX / "capture_12f_rgb_chw.bin").write_bytes(rgb_chw.tobytes())

    (FIX / "capture_12f_meta.txt").write_text(
        f"frames {len(frames)}\nwidth {w}\nheight {h}\nchannels {c}\n"
    )
    print(f"pinned {len(frames)} frames {w}x{h}x{c}")
    print(f"  {(FIX / 'capture_12f_bgr.bin').stat().st_size} bytes BGR HWC")
    print(f"  {(FIX / 'capture_12f_rgb_chw.bin').stat().st_size} bytes RGB CHW (frame 0)")
    rc = dump_marker()
    return rc if rc else dump_charuco()


if __name__ == "__main__":
    raise SystemExit(main())
