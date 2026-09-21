"""Pillow NEAREST references for `tests/io/test_resize_nearest.mojo`.

    pixi run python tools/io/dump_resize_nearest_reference.py --out /tmp/nearest_ref

For each of many (in_h, in_w) -> (out_h, out_w) pairs, writes the source RGBA
image and what `PIL.Image.resize(..., Image.NEAREST)` makes of it.

⚠ THE SIZE PAIRS ARE THE TEST. NEAREST has no interpolation to get wrong; the
only thing that can differ is WHICH source pixel each output pixel picks, and
that only diverges where the exact coordinate lands on an integer. Two sizes
chosen by hand will agree with almost any formula — `floor((x + 0.5) * in /
out)` matches Pillow on most pairs and disagrees on 93 of 600 random ones. So
the sweep is random, seeded, and large.
"""

import argparse
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nearest_ref")
    ap.add_argument("--cases", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(1234)
    index = []
    for k in range(args.cases):
        ih = int(rng.integers(1, 40))
        iw = int(rng.integers(1, 40))
        oh = int(rng.integers(1, 40))
        ow = int(rng.integers(1, 40))
        a = rng.integers(0, 256, size=(ih, iw, 4), dtype=np.uint8)
        im = Image.fromarray(a, mode="RGBA")
        out = np.array(im.resize((ow, oh), resample=Image.NEAREST))
        with open(os.path.join(args.out, "%03d_src.raw" % k), "wb") as f:
            f.write(a.tobytes())
        with open(os.path.join(args.out, "%03d_dst.raw" % k), "wb") as f:
            f.write(out.astype(np.uint8).tobytes())
        index.append("%03d\t%d\t%d\t%d\t%d" % (k, ih, iw, oh, ow))

    with open(os.path.join(args.out, "index.tsv"), "w") as f:
        f.write("\n".join(index) + "\n")
    print("wrote %d NEAREST cases to %s" % (len(index), args.out))


if __name__ == "__main__":
    main()
