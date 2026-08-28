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
    capture_12f_bgr.bin     all 12 frames, BGR HWC uint8, contiguous
    capture_12f_rgb_chw.bin frame 0 only, converted to RGB CHW
    capture_12f_meta.txt    frames, width, height, channels
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/vision"
SRC = FIX / "capture_12f.mp4"


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
