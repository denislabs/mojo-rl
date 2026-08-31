"""Pillow's RGBA decode of every asset PNG, for `tests/io/test_png_assets.mojo`.

    pixi run python tools/io/dump_asset_png_reference.py --out /tmp/png_assets

Walks this repo's sprite trees — `assets/procgen/` and both Craftax asset
directories — and writes, per image, `<sha1-of-path>.raw`: the bytes of
`PIL.Image.open(p).convert("RGBA")`. An index file pairs each raw with its
source path.

⚠ THE SYNTHETIC FIXTURES ARE NOT ENOUGH ON THEIR OWN, and neither is this.
`make_png_fixtures.py` covers filters and depths the real assets never use;
this covers the encoders, palettes, chunk layouts and odd sizes that 1,443
files produced in the wild actually contain and that nobody would think to
synthesise. Both, or the coverage has a hole shaped like whichever was
skipped.
"""

import argparse
import hashlib
import os

DIRS = [
    "assets/procgen",
    "mojo_rl/envs/craftax_classic/assets",
    "mojo_rl/envs/craftax_full/assets",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/png_assets")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N images (0 = all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import numpy as np
    from PIL import Image

    index = []
    for d in DIRS:
        if not os.path.isdir(d):
            print("skipping missing %s" % d)
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if not f.endswith(".png"):
                    continue
                p = os.path.join(root, f)
                a = np.array(Image.open(p).convert("RGBA")).astype(np.uint8)
                key = hashlib.sha1(p.encode()).hexdigest()[:16]
                with open(os.path.join(args.out, key + ".raw"), "wb") as g:
                    g.write(a.tobytes())
                index.append("%s\t%d\t%d\t%s" % (key, a.shape[1], a.shape[0], p))
                if args.limit and len(index) >= args.limit:
                    break
            if args.limit and len(index) >= args.limit:
                break
        if args.limit and len(index) >= args.limit:
            break

    with open(os.path.join(args.out, "index.tsv"), "w") as g:
        g.write("\n".join(index) + "\n")
    print("wrote %d references to %s" % (len(index), args.out))


if __name__ == "__main__":
    main()
