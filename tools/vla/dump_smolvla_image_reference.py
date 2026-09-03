#!/usr/bin/env python3
"""Dump the reference SmolVLA image preprocessing, for gating the Mojo camera path.

    pixi run -e act-ref python tools/vla/dump_smolvla_image_reference.py --out /tmp/vla_img

⚠ **SmolVLA's resize is NOT the resize this repo already implements.**
`mojo_rl/vision/preprocess.mojo` reproduces PIL BILINEAR, whose filter support
GROWS with the reduction factor (5 taps at 1.25x). SmolVLA calls
`F.interpolate(mode="bilinear", align_corners=False)`, which is a fixed 2-tap
triangle with NO antialiasing at any scale. Feeding one where the other is
expected is finite, right-shaped and wrong -- exactly the failure that file's
own header was written about. Hence a second resize, and hence this gate.

⚠ **`vla_utils.py` ships TWO functions one letter apart.**
`resize_with_pad_torch` pads CENTRED (openpi); `resize_with_pad` pads on the
LEFT and TOP (smolvla/xvla). `modeling_smolvla.py` imports the second. A 640x480
frame gets 128 blank rows ABOVE the picture, not 64 above and 64 below, so this
choice moves every patch token. The cases below include a portrait frame so the
left-pad branch is exercised too.

The transcribed part is the eight lines of ratio / int() / pad-side bookkeeping,
copied verbatim from the reference below. The arithmetic that actually resamples
is torch's own, which is the part a hand-written check could not honestly cover.

Output is `refload.mojo`'s format:

    <out>/manifest.txt      `name<TAB>d0,d1,...` per array
    <out>/<name>.bin        raw little-endian float32, C order
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F


# --- VERBATIM from lerobot/policies/common/vla_utils.py -------------------- #
# Copied, not paraphrased: a paraphrase would encode this port's reading of the
# reference, and a gate that shares the reading under test cannot see a
# misreading.  (`resize_with_pad`, the LEFT/TOP-padding variant.)
def resize_with_pad(img: torch.Tensor, height: int, width: int, *, pad_value: float) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img
# -------------------------------------------------------------------------- #


# `prepare_images`, the two lines of it that touch pixels.
def prepare_one(rgb_hwc_u8: np.ndarray, size: int) -> np.ndarray:
    chw = torch.from_numpy(rgb_hwc_u8).permute(2, 0, 1)[None].float() / 255.0
    img = resize_with_pad(chw, size, size, pad_value=0)
    img = img * 2.0 - 1.0
    return img[0].numpy()


def frame(w: int, h: int, seed: int) -> np.ndarray:
    """A frame with structure in BOTH axes and no symmetry.

    A constant or separable pattern would survive a transposed index, a flipped
    axis and a swapped pad side; this does not."""
    rng = np.random.default_rng(seed)
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    base = (y * 7 + x * 3) % 251
    out = np.empty((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = ((base + c * 83) % 256).astype(np.uint8)
    # break the analytic pattern so a smooth-image coincidence cannot pass
    out ^= rng.integers(0, 32, size=out.shape, dtype=np.uint8)
    return out


CASES = [
    # name      w    h    why
    ("land",   640, 480),   # the SO-101 camera: 512x384, 128 rows padded on TOP
    ("store",  320, 240),   # what our TrajectoryStore holds: same shape, UPSCALED
    ("port",   480, 640),   # portrait: 384x512, 128 cols padded on the LEFT
    ("exact",  512, 512),   # the early-out: no resample at all, only *2-1
    ("odd",    700, 510),   # ratio 1.3671875 -> int() truncates 373.06 to 373
]


class Dump:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.lines = []

    def add(self, name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        a.tofile(os.path.join(self.root, name + ".bin"))
        self.lines.append(f"{name}\t{','.join(str(d) for d in a.shape)}")
        print(f"  {name:14s} {str(a.shape):20s} min {a.min():+.6f} max {a.max():+.6f}")

    def close(self):
        with open(os.path.join(self.root, "manifest.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()

    d = Dump(a.out)
    print(f"torch {torch.__version__} -> {a.out}")
    for name, w, h in CASES:
        src = frame(w, h, seed=len(name) * 1000 + w)
        ratio = max(w / a.size, h / a.size)
        rh, rw = int(h / ratio), int(w / ratio)
        print(f"{name}: {w}x{h} -> {rw}x{rh}  pad left {a.size - rw}  top {a.size - rh}")
        d.add("in_" + name, src.astype(np.float32))
        d.add("out_" + name, prepare_one(src, a.size))
    d.close()
    print(f"\n{len(CASES)} cases written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
