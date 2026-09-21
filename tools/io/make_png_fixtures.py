"""PNG fixtures for `tests/io/test_png.mojo`, plus their expected pixels.

    python3 tools/io/make_png_fixtures.py <out-dir>

⚠ THIS EXISTS BECAUSE THE REAL DATA DID NOT COVER THE DECODER. CIFAR-10's
60,000 PNGs use scanline filters 0, 1, 2 and 4 — measured — and NEVER 3
(`Average`). Sabotaging the `Average` branch left `tests/nn/test_cifar10_prep`
fully green over 30,730,000 bytes. A gate on real data is not a gate on the
code; the filter type is chosen by the ENCODER, so covering all five means
writing the PNGs here rather than hoping.

Each `<name>.png` comes with `<name>.raw`, the expected pixels HWC, one byte
per sample. `.raw` is produced by **Pillow**, so the comparison is against an
independent decoder rather than against the encoder in this file.

Written:
  filter0..filter4.png   one filter type used on EVERY row
  filter_mixed.png       the five cycling row by row, which is what a real
                         encoder emits
  grey.png / grey_alpha.png / rgba.png   colour types 0, 4 and 6
  odd_size.png           17x13, so no dimension is a multiple of anything
  one_pixel.png          the degenerate case
  big_idat.png           large enough that Pillow splits IDAT across chunks
  pal8 / pal4 / pal2 / pal1        palette at depths 8, 4, 2 and 1
  pal8_trns / pal4_trns            palette with per-entry alpha (tRNS)
  reject_16bit.png       bit depth 16       -- must RAISE
  reject_interlaced.png  Adam7              -- must RAISE
  reject_trns_rgb.png    colour-key tRNS    -- must RAISE

⚠ THE PALETTE DEPTH IS CHOSEN BY THE PALETTE SIZE, not by a flag: Pillow packs
2 colours into 1 bit, 4 into 2, 16 into 4 and more into 8. Samples below a
byte are packed MSB-first and the row is rounded up to a whole byte, so a
13-pixel 4-bit row is 7 bytes with the last nibble unused — the reason the
odd-width palette cases are here. 7 of this repo's asset PNGs are sub-byte
palettes; without these fixtures that path would ship unexecuted.
"""

import os
import struct
import sys
import zlib

import numpy as np
from PIL import Image

CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_row(ft, cur, prev, bpp):
    """Encode one scanline with filter type `ft`. The inverse of the decoder."""
    out = bytearray(len(cur))
    for i in range(len(cur)):
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if ft == 0:
            v = cur[i]
        elif ft == 1:
            v = cur[i] - a
        elif ft == 2:
            v = cur[i] - b
        elif ft == 3:
            v = cur[i] - ((a + b) // 2)
        else:
            v = cur[i] - paeth(a, b, c)
        out[i] = v & 0xFF
    return bytes(out)


def write_png(path, arr, colour_type, filters):
    """`filters` is a list giving the filter type per row (cycled if short)."""
    h, w = arr.shape[0], arr.shape[1]
    ch = CHANNELS[colour_type]
    flat = arr.reshape(h, w * ch)
    bpp = ch
    raw = bytearray()
    prev = bytes(w * ch)
    for y in range(h):
        ft = filters[y % len(filters)]
        cur = bytes(flat[y].tolist())
        raw.append(ft)
        raw += filter_row(ft, cur, prev, bpp)
        prev = cur
    body = (chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + body)


def dump_raw(png_path, raw_path):
    """The expected pixels, per PILLOW — a decoder this repo does not share."""
    im = Image.open(png_path)
    a = np.array(im)
    if a.ndim == 2:
        a = a[:, :, None]
    with open(raw_path, "wb") as f:
        f.write(a.astype(np.uint8).tobytes())
    return a.shape


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)

    def emit(name, arr, colour_type, filters):
        p = os.path.join(out, name + ".png")
        write_png(p, arr, colour_type, filters)
        shape = dump_raw(p, os.path.join(out, name + ".raw"))
        print("  %-22s %s colour=%d filters=%s"
              % (name, shape, colour_type, filters))

    rgb = rng.integers(0, 256, size=(24, 31, 3), dtype=np.uint8)
    for ft in range(5):
        emit("filter%d" % ft, rgb, 2, [ft])
    emit("filter_mixed", rgb, 2, [0, 1, 2, 3, 4])

    emit("grey", rng.integers(0, 256, size=(20, 20, 1), dtype=np.uint8), 0,
         [0, 1, 2, 3, 4])
    emit("grey_alpha", rng.integers(0, 256, size=(20, 20, 2), dtype=np.uint8),
         4, [0, 1, 2, 3, 4])
    emit("rgba", rng.integers(0, 256, size=(20, 20, 4), dtype=np.uint8), 6,
         [0, 1, 2, 3, 4])
    emit("odd_size", rng.integers(0, 256, size=(13, 17, 3), dtype=np.uint8), 2,
         [4, 3, 2, 1, 0])
    emit("one_pixel", rng.integers(0, 256, size=(1, 1, 3), dtype=np.uint8), 2,
         [0])
    # Big enough that a real encoder splits IDAT across several chunks.
    emit("big_idat", rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
         2, [0, 1, 2, 3, 4])

    # ── palette, at every depth Pillow will emit ────────────────────
    def emit_palette(name, n_colours, size, transparency=None, bits=None):
        w, h = size
        pal_rgb = rng.integers(0, 256, size=(n_colours, 3), dtype=np.uint8)
        idx = rng.integers(0, n_colours, size=(h, w), dtype=np.uint8)
        im = Image.fromarray(idx, mode="P")
        flat = []
        for c in pal_rgb:
            flat.extend(int(v) for v in c)
        flat.extend([0] * (768 - len(flat)))
        im.putpalette(flat)
        p = os.path.join(out, name + ".png")
        # ⚠ PILLOW DOES NOT PACK BY ITSELF. Without `bits=`, a 2-colour
        # palette is still written at depth 8 — the first version of these
        # fixtures produced four "sub-byte" cases that were all depth 8, so
        # the packed-sample path passed the gate without ever running.
        kw = {} if bits is None else {"bits": bits}
        if transparency is not None:
            kw["transparency"] = bytes(transparency)
        im.save(p, **kw)
        with open(p, "rb") as f:
            hdr = f.read(33)
        # The reference is PILLOW'S RGBA expansion — the decoder returns RGBA
        # for a palette image, so that is what must be compared.
        a = np.array(Image.open(p).convert("RGBA"))
        with open(os.path.join(out, name + ".raw"), "wb") as f:
            f.write(a.astype(np.uint8).tobytes())
        print("  %-22s %s depth=%d colour=%d trns=%s"
              % (name, a.shape, hdr[24], hdr[25], transparency is not None))
        if bits is not None and hdr[24] != bits:
            raise SystemExit(
                "%s came out at depth %d, wanted %d — the sub-byte path would"
                " not be exercised" % (name, hdr[24], bits))

    emit_palette("pal8", 200, (19, 11), bits=8)
    emit_palette("pal4", 16, (13, 7), bits=4)
    emit_palette("pal2", 4, (11, 5), bits=2)
    emit_palette("pal1", 2, (17, 3), bits=1)
    # tRNS SHORTER than the palette: the entries it does not reach are opaque.
    emit_palette("pal8_trns", 200, (19, 11),
                 transparency=list(range(0, 250, 2))[:64], bits=8)
    emit_palette("pal4_trns", 16, (13, 7), transparency=[0, 64, 128, 255],
                 bits=4)

    # ── the ones that must be refused ───────────────────────────────
    a16 = (rng.integers(0, 65536, size=(8, 8), dtype=np.uint16))
    Image.fromarray(a16, mode="I;16").save(os.path.join(out, "reject_16bit.png"))
    # ⚠ PILLOW IGNORES `interlace=1` — the first version of this fixture came
    # out with the interlace byte at 0, so it decoded fine and the "must be
    # refused" check passed vacuously. The IHDR byte is flipped by hand
    # instead, with the chunk CRC recomputed: the decoder must refuse on the
    # HEADER, before it ever looks at the image data, which is precisely the
    # behaviour under test.
    base = os.path.join(out, "filter0.png")
    with open(base, "rb") as f:
        raw = bytearray(f.read())
    raw[28] = 1                                   # IHDR interlace method
    ihdr = bytes(raw[12:29])                      # type + 13 data bytes
    raw[29:33] = struct.pack(">I", zlib.crc32(ihdr) & 0xFFFFFFFF)
    with open(os.path.join(out, "reject_interlaced.png"), "wb") as f:
        f.write(bytes(raw))
    # Colour-key transparency on an RGB image: tRNS naming ONE colour rather
    # than a per-entry alpha table. No asset in this repo uses it, so the
    # decoder refuses it instead of shipping an untested branch.
    base = os.path.join(out, "filter0.png")
    with open(base, "rb") as f:
        raw2 = bytearray(f.read())
    trns = struct.pack(">HHH", 1, 2, 3)
    ins = 8 + 25  # after IHDR
    raw2[ins:ins] = chunk(b"tRNS", trns)
    with open(os.path.join(out, "reject_trns_rgb.png"), "wb") as f:
        f.write(bytes(raw2))

    print("  reject_16bit / reject_interlaced / reject_trns_rgb written")


if __name__ == "__main__":
    main()
