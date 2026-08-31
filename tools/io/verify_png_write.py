"""Verify PNGs written by `mojo_rl/io/png.encode_png`, with Pillow.

    python3 tools/io/verify_png_write.py <dir>

`tests/io/test_png_write.mojo` writes `<dir>/w<channels>.png` plus a
`<dir>/expected_<channels>.raw` holding the pixels it meant to write, then
calls this. Prints `PNG-WRITE-OK` and exits 0 only if every file opens in
Pillow at the expected mode and size with exactly those bytes.

⚠ A ROUND TRIP THROUGH OUR OWN DECODER IS NOT A CHECK. Both halves would
share any misunderstanding — a CRC computed over the wrong span, a colour type
written for the wrong channel count — and agree with each other while no other
program could read the file. That is the two-parsers-one-wrong-default shape
this repo keeps recording, so the verifier is a third implementation.
"""

import os
import sys

import numpy as np
from PIL import Image

MODES = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}


def main():
    d = sys.argv[1]
    checked = 0
    for ch, mode in MODES.items():
        p = os.path.join(d, "w%d.png" % ch)
        e = os.path.join(d, "expected_%d.raw" % ch)
        if not os.path.exists(p):
            raise SystemExit("missing %s" % p)
        im = Image.open(p)
        if im.mode != mode:
            raise SystemExit(
                "%s opened as mode %s, expected %s (the colour type written "
                "does not match the channel count)" % (p, im.mode, mode))
        a = np.array(im)
        if a.ndim == 2:
            a = a[:, :, None]
        want = np.frombuffer(open(e, "rb").read(), dtype=np.uint8)
        if a.size != want.size:
            raise SystemExit("%s has %d samples, expected %d"
                             % (p, a.size, want.size))
        diff = int((a.reshape(-1) != want).sum())
        if diff:
            raise SystemExit("%s differs from the source in %d bytes"
                             % (p, diff))
        checked += 1
    if checked != len(MODES):
        raise SystemExit("only %d of %d images checked" % (checked, len(MODES)))
    print("PNG-WRITE-OK %d images" % checked)


if __name__ == "__main__":
    main()
