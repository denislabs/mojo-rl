"""Pin PIL's exact BILINEAR output on a real dataset frame.

    pixi run python tools/vision/make_resize_fixture.py

⚠ WHY A REAL FRAME AND NOT A SYNTHETIC PATTERN. The thing being reproduced is
one library's arithmetic on photographic data — noise, soft edges and all. A
gradient or a checkerboard is exactly the input on which two different filters
are most likely to AGREE, so it would gate almost nothing.

⚠⚠ THE SOURCE IS A 320x240 CROP, NOT THE FULL 640x480 FRAME, and that is a
size decision only: at a 2x reduction PIL's coefficient table is identical for
every output pixel except the clamped edges, so a smaller image exercises the
same `ksize`, the same weights and the same boundary handling for a quarter of
the bytes. The NON-INTEGER case below is the one that would catch a
scale-dependent bug, and it could not be produced by a 640x480 source at all.

Outputs, into tests/fixtures/vision/:
    act_src_320x240.png        a real `front` camera frame, cropped
    act_resize_160x120.bin     PIL BILINEAR, exact 2x, HWC RGB uint8
    act_resize_137x101.bin     PIL BILINEAR, non-integer ratio
    act_chw_160x120.bin        the 2x output as CHW RGB — what the store holds
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/vision"
DATASET = (Path.home() / ".cache/huggingface/lerobot/DenisLabs"
           / "record-test_20260828_092736")


def main() -> int:
    src_png = FIX / "act_src_320x240.png"
    if src_png.exists():
        # ⚠ NEVER RE-EXTRACT A COMMITTED SOURCE. Re-decoding could pick a
        # different frame or a different decoder build, silently changing what
        # the pinned answers describe.
        bgr = cv2.imread(str(src_png), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"reusing {src_png.name}")
    else:
        import imageio.v3 as iio
        mp4 = sorted(DATASET.glob(
            "videos/observation.images.front/chunk-*/file-*.mp4"))
        if not mp4:
            print(f"no dataset under {DATASET}", file=sys.stderr)
            return 1
        frame = next(iter(iio.imiter(mp4[0])))
        if frame.shape[:2] != (480, 640):
            print(f"unexpected frame {frame.shape}", file=sys.stderr)
            return 1
        rgb = np.ascontiguousarray(frame[120:360, 160:480])
        FIX.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(src_png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"wrote {src_png.name} ({src_png.stat().st_size} bytes)")

    im = Image.fromarray(rgb)
    for (w, h), name in (((160, 120), "act_resize_160x120.bin"),
                         ((137, 101), "act_resize_137x101.bin")):
        out = np.asarray(im.resize((w, h), Image.BILINEAR))
        (FIX / name).write_bytes(np.ascontiguousarray(out).tobytes())
        print(f"  {name}: {w}x{h} = {out.size} bytes")
        if name.endswith("160x120.bin"):
            chw = np.ascontiguousarray(out.transpose(2, 0, 1))
            (FIX / "act_chw_160x120.bin").write_bytes(chw.tobytes())

    # A flat image resizes to itself; if it does not, the coefficients do not
    # sum to one and every other comparison is meaningless.
    flat = Image.fromarray(np.full((240, 320, 3), 137, np.uint8))
    got = np.asarray(flat.resize((160, 120), Image.BILINEAR))
    if not (got == 137).all():
        print("PIL did not preserve a flat image — refusing to pin", file=sys.stderr)
        return 1
    print("  sanity: a flat image survives the filter unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
