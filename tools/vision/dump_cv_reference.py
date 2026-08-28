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
    return dump_marker()


if __name__ == "__main__":
    raise SystemExit(main())
