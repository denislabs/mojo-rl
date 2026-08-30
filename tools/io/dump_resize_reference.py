#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | Pillow BILINEAR reference images for the Mojo resize
# +--------------------------------------------------------------------------+ #
"""Writes raw HWC uint8 sources and their Pillow-resized outputs.

    pixi run python tools/io/dump_resize_reference.py --out /tmp/resize_ref
    pixi run mojo run -I . tests/io/test_image_resize.mojo /tmp/resize_ref

Cases are chosen for the arithmetic they exercise, not for looking like
pictures:

  480x640 -> 240x320   the ACT case: an exact 2x downscale in both axes
  480x640 -> 224x224   non-integer scales, and different ones per axis
  37x53   -> 240x320   UPSCALE, where `filterscale` clamps to 1 and `support`
                       stops tracking `scale` — a different branch
  64x64   -> 64x64     the identity, which must still round-trip
  17x9    -> 5x3       tiny and odd, so the first and last output pixels hit
                       the `xmin < 0` and `xmax > inSize` clamps

The `noise` source is uniform random (adjacent pixels uncorrelated, so a
one-pixel window shift shows up immediately); `ramp` is a smooth gradient
(where a window shift is nearly invisible and only the fixed-point rounding
separates a correct implementation from a plausible one). Both are used for
every case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CASES = [
    (480, 640, 240, 320),
    (480, 640, 224, 224),
    (37, 53, 240, 320),
    (64, 64, 64, 64),
    (17, 9, 5, 3),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/resize_ref")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(11)
    lines = []
    for ih, iw, oh, ow in CASES:
        for kind in ("noise", "ramp"):
            if kind == "noise":
                src = rng.integers(0, 256, (ih, iw, 3), dtype=np.uint8)
            else:
                yy = np.linspace(0, 255, ih)[:, None]
                xx = np.linspace(0, 255, iw)[None, :]
                src = np.stack(
                    [
                        (yy + 0 * xx),
                        (0 * yy + xx),
                        (yy * 0.5 + xx * 0.5),
                    ],
                    axis=-1,
                ).astype(np.uint8)
            dst = np.asarray(
                Image.fromarray(src).resize((ow, oh), Image.BILINEAR)
            )
            assert dst.shape == (oh, ow, 3), dst.shape
            tag = f"{kind}_{ih}x{iw}_to_{oh}x{ow}"
            (out / f"{tag}.src").write_bytes(src.tobytes())
            (out / f"{tag}.dst").write_bytes(dst.tobytes())
            lines.append(f"{tag} {ih} {iw} {oh} {ow} 3")

    (out / "cases.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} cases to {out}")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    main()
