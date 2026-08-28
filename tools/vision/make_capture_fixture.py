"""Write the deterministic video the capture gate decodes.

    pixi run python tools/vision/make_capture_fixture.py

⚠ THE FILE IS COMMITTED, THIS SCRIPT IS NOT RUN BY THE GATE. Re-encoding on
every run would compare a decoder against a fresh encode of itself, which is a
gate that passes when both are wrong together. The bytes are pinned once.

⚠ AND THE PIXEL VALUES ARE NOT WHAT IS PINNED EITHER. mp4 is lossy, so what a
decoder returns is not what was written here — the gate asserts that the Mojo
shim and Python `cv2` decode the SAME FILE to the SAME BYTES, which is a claim
about our marshalling, not about the codec. The frames below are built to make
a marshalling bug loud: a channel swap, a row-stride error and an off-by-one
frame all change the answer visibly.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, N, FPS = 64, 48, 12, 10.0
OUT = Path(__file__).resolve().parents[2] / "tests/fixtures/vision/capture_12f.mp4"


def frame(i: int) -> np.ndarray:
    """BGR, HWC. Asymmetric in every axis that a bug could transpose."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # A horizontal ramp in BLUE and a vertical ramp in RED: if the channels are
    # swapped the two ramps trade axes, which no tolerance can hide.
    img[:, :, 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    img[:, :, 2] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    # GREEN carries the frame index, so reading frame k+1 for frame k fails.
    img[:, :, 1] = np.uint8(10 + i * 20)
    # A single bright corner pixel: a row-stride error moves it.
    img[0, 0] = (255, 255, 255)
    return img


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not w.isOpened():
        print("VideoWriter did not open", file=sys.stderr)
        return 1
    for i in range(N):
        w.write(frame(i))
    w.release()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {N} frames {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
