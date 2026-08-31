# +--------------------------------------------------------------------------+ #
# | A PNG decoder — 8-bit, non-interlaced
# +--------------------------------------------------------------------------+ #
"""Decode PNG bytes to pixels, without Pillow.

    var img = decode_png(bytes)      # img.pixels is HWC, `img.channels` deep

Written for `nn/datasets/cifar10.mojo`, whose 60,000 images arrive as
PNG-encoded byte arrays inside a HuggingFace parquet file. Decoding them was
the last thing in that loader needing `PIL` — and `PIL` came with `numpy` and
a subprocess, because `pyarrow`'s libarrow cannot safely load inside Mojo's
embedded interpreter.

## Scope, and what is refused

Supported: colour types **0** (grey), **2** (RGB), **4** (grey+alpha) and
**6** (RGBA) at **bit depth 8**; colour type **3** (palette) at depths **1, 2,
4 and 8**, with `tRNS` for per-entry alpha. Non-interlaced, all five scanline
filters.

⚠ **A PALETTE IMAGE COMES BACK AS RGBA**, four channels, whatever its bit
depth — matching `PIL.Image.open(...).convert("RGBA")`, which is what all
three sprite loaders in this repo asked PIL for. There is no "indices" output:
nothing here wants one, and returning indices that a caller forgot to expand
would render a picture of the palette rather than the image.

The mix was measured, not guessed: across this repo's 1,443 asset PNGs
(procgen + both Craftax sets), 994 are RGBA, 241 palette at depth 8, 68 RGB,
7 palette at depth 1/2/4, and 236 of the palette ones carry `tRNS`.

Refused BY NAME: bit depth 16, Adam7 interlacing, sub-byte GREY (which the
same unpacking would cover but which no asset here uses, so it would be
untested code), and `tRNS` on a non-palette image (a colour-key transparency
that none of these 1,443 files uses). Each raises saying which it found. A
decoder that quietly mishandled one would return an image — the wrong one —
and a training run cannot tell the difference.

## The parts that are easy to get wrong

⚠ **`IDAT` MAY BE SPLIT ACROSS ANY NUMBER OF CHUNKS**, at arbitrary byte
boundaries — the split is not aligned to anything in the zlib stream. They
must be concatenated before inflating, not inflated one at a time.

⚠ **FILTERS REFER TO THE RECONSTRUCTED PREVIOUS ROW**, not the encoded one, so
unfiltering has to proceed in order and read back what it just wrote. `Up` on
row 0 reads zeros; `Sub` and `Paeth` read zeros for pixels left of the row.

⚠ **`Average` USES INTEGER DIVISION OF A SUM THAT CAN EXCEED 255**, so it must
be computed in a wider type and only then truncated to a byte. Doing the
arithmetic in `UInt8` gives an image that looks nearly right.
"""

from .fileio import read_file_bytes
from .http import crc32, deflate, inflate_into


comptime PNG_GREY = 0
comptime PNG_RGB = 2
comptime PNG_PALETTE = 3
comptime PNG_GREY_ALPHA = 4
comptime PNG_RGBA = 6


def _channels(colour_type: Int) raises -> Int:
    """Samples per pixel AS STORED. A palette image stores one (the index);
    what comes out is four."""
    if colour_type == PNG_GREY: return 1
    if colour_type == PNG_RGB: return 3
    if colour_type == PNG_GREY_ALPHA: return 2
    if colour_type == PNG_RGBA: return 4
    if colour_type == PNG_PALETTE: return 1
    raise Error("png: unknown colour type " + String(colour_type))


struct PngImage(Movable):
    """Decoded pixels, HWC, one byte per sample."""

    var width: Int
    var height: Int
    var channels: Int
    var pixels: List[UInt8]

    def __init__(
        out self, width: Int, height: Int, channels: Int, var pixels: List[UInt8]
    ):
        self.width = width
        self.height = height
        self.channels = channels
        self.pixels = pixels^

    def __init__(out self, *, deinit move: Self):
        self.width = move.width
        self.height = move.height
        self.channels = move.channels
        self.pixels = move.pixels^


def _be32(ref b: List[UInt8], off: Int) -> Int:
    return (
        (Int(b[off]) << 24) | (Int(b[off + 1]) << 16)
        | (Int(b[off + 2]) << 8) | Int(b[off + 3])
    )


def _paeth(a: Int, b: Int, c: Int) -> Int:
    """The PNG predictor: left, above, upper-left."""
    var p = a + b - c
    var pa = abs(p - a)
    var pb = abs(p - b)
    var pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png(ref data: List[UInt8]) raises -> PngImage:
    """Decode a PNG image held in memory."""
    if len(data) < 8:
        raise Error("png: " + String(len(data)) + " bytes is not a PNG")
    if (
        data[0] != 0x89 or data[1] != 0x50 or data[2] != 0x4E or data[3] != 0x47
        or data[4] != 0x0D or data[5] != 0x0A or data[6] != 0x1A or data[7] != 0x0A
    ):
        raise Error("png: the 8-byte signature is missing")

    var width = 0
    var height = 0
    var channels = 0
    var depth = 8
    var colour = PNG_RGB
    var idat = List[UInt8]()
    var plte = List[UInt8]()
    var trns = List[UInt8]()
    var saw_ihdr = False
    var pos = 8

    while pos + 8 <= len(data):
        var length = _be32(data, pos)
        var t0 = Int(data[pos + 4])
        var t1 = Int(data[pos + 5])
        var t2 = Int(data[pos + 6])
        var t3 = Int(data[pos + 7])
        var body = pos + 8
        if body + length + 4 > len(data):
            raise Error(
                "png: a chunk declares " + String(length) + " bytes but only "
                + String(len(data) - body) + " remain"
            )

        # IHDR
        if t0 == 0x49 and t1 == 0x48 and t2 == 0x44 and t3 == 0x52:
            if length != 13:
                raise Error("png: IHDR is " + String(length) + " bytes, not 13")
            width = _be32(data, body)
            height = _be32(data, body + 4)
            depth = Int(data[body + 8])
            colour = Int(data[body + 9])
            var interlace = Int(data[body + 12])
            if colour == PNG_PALETTE:
                if depth != 1 and depth != 2 and depth != 4 and depth != 8:
                    raise Error(
                        "png: palette bit depth " + String(depth) + " is not"
                        " one of 1, 2, 4, 8"
                    )
            elif depth != 8:
                raise Error(
                    "png: bit depth " + String(depth) + " is not implemented"
                    " for colour type " + String(colour) + " (only 8)"
                )
            if interlace != 0:
                raise Error(
                    "png: Adam7 interlacing is not implemented — the image"
                    " would decode to seven wrong sub-images"
                )
            channels = _channels(colour)
            if width <= 0 or height <= 0:
                raise Error(
                    "png: a " + String(width) + "x" + String(height) + " image"
                )
            saw_ihdr = True
        # PLTE — 3 bytes per palette entry
        elif t0 == 0x50 and t1 == 0x4C and t2 == 0x54 and t3 == 0x45:
            if length % 3 != 0:
                raise Error(
                    "png: PLTE is " + String(length) + " bytes, not a multiple"
                    " of 3"
                )
            for i in range(length):
                plte.append(data[body + i])
        # tRNS — for a palette, one alpha byte per entry (short is allowed;
        # the entries it does not reach are opaque)
        elif t0 == 0x74 and t1 == 0x52 and t2 == 0x4E and t3 == 0x53:
            for i in range(length):
                trns.append(data[body + i])
        # IDAT — one zlib stream, split across chunks at arbitrary offsets
        elif t0 == 0x49 and t1 == 0x44 and t2 == 0x41 and t3 == 0x54:
            for i in range(length):
                idat.append(data[body + i])
        # IEND
        elif t0 == 0x49 and t1 == 0x45 and t2 == 0x4E and t3 == 0x44:
            break

        pos = body + length + 4  # skip the CRC

    if not saw_ihdr:
        raise Error("png: no IHDR chunk")
    if len(idat) == 0:
        raise Error("png: no IDAT data")
    if colour == PNG_PALETTE and len(plte) == 0:
        raise Error("png: a palette image with no PLTE chunk")
    if len(trns) > 0 and colour != PNG_PALETTE:
        raise Error(
            "png: tRNS on colour type " + String(colour) + " is colour-key"
            " transparency, which is not implemented (no asset in this repo"
            " uses it, so implementing it would ship untested)"
        )

    # ⚠ A SCANLINE IS A BYTE COUNT, NOT A SAMPLE COUNT. At depth 8 they
    # coincide; at 1, 2 or 4 the samples are packed MSB-first and the row is
    # rounded UP to a whole byte, so a 13-pixel 4-bit row is 7 bytes with the
    # last nibble unused. Filtering happens on those BYTES.
    var stride = (width * channels * depth + 7) // 8
    var raw = inflate_into(idat, (stride + 1) * height)
    if len(raw) != (stride + 1) * height:
        raise Error(
            "png: the image data inflated to " + String(len(raw))
            + " bytes, the header implies " + String((stride + 1) * height)
        )

    var unfiltered = List[UInt8]()
    unfiltered.resize(stride * height, 0)
    # The filter's "pixel to the left" is a whole number of BYTES, and never
    # less than one: at sub-byte depths the spec defines bpp as 1.
    var bpp = (channels * depth + 7) // 8

    for y in range(height):
        var ft = Int(raw[y * (stride + 1)])
        var src = y * (stride + 1) + 1
        var dst = y * stride
        var up = dst - stride
        for x in range(stride):
            var cur = Int(raw[src + x])
            var a = Int(unfiltered[dst + x - bpp]) if x >= bpp else 0
            var b = Int(unfiltered[up + x]) if y > 0 else 0
            var c = Int(unfiltered[up + x - bpp]) if (y > 0 and x >= bpp) else 0
            var v: Int
            if ft == 0:
                v = cur
            elif ft == 1:
                v = cur + a
            elif ft == 2:
                v = cur + b
            elif ft == 3:
                # ⚠ (a + b) can reach 510: widen, halve, THEN truncate.
                v = cur + ((a + b) // 2)
            elif ft == 4:
                v = cur + _paeth(a, b, c)
            else:
                raise Error(
                    "png: scanline " + String(y) + " uses filter "
                    + String(ft) + ", which is not one of the five defined"
                )
            unfiltered[dst + x] = UInt8(v & 0xFF)

    if colour != PNG_PALETTE:
        return PngImage(width, height, channels, unfiltered^)

    # ── palette expansion ────────────────────────────────────────────
    # Out comes RGBA, whatever the stored depth: see the module docstring on
    # why there is no index output.
    var n_entries = len(plte) // 3
    var rgba = List[UInt8]()
    rgba.resize(width * height * 4, 0)
    var mask = (1 << depth) - 1
    for y in range(height):
        var row = y * stride
        for x in range(width):
            # MSB-first: sample 0 occupies the HIGH bits of the first byte.
            var bit = x * depth
            var byte = Int(unfiltered[row + (bit >> 3)])
            var shift = 8 - depth - (bit & 7)
            var idx = (byte >> shift) & mask
            if idx >= n_entries:
                raise Error(
                    "png: palette index " + String(idx) + " at (" + String(x)
                    + ", " + String(y) + ") outside a " + String(n_entries)
                    + "-entry PLTE"
                )
            var o = (y * width + x) * 4
            rgba[o] = plte[idx * 3]
            rgba[o + 1] = plte[idx * 3 + 1]
            rgba[o + 2] = plte[idx * 3 + 2]
            # tRNS may be SHORTER than the palette; the rest are opaque.
            rgba[o + 3] = trns[idx] if idx < len(trns) else UInt8(255)

    return PngImage(width, height, 4, rgba^)


def to_rgba(ref img: PngImage) raises -> List[UInt8]:
    """The image as RGBA, expanding grey and adding an opaque alpha.

    Equivalent to `PIL.Image.open(...).convert("RGBA").tobytes()`, which is
    what `render/png_loader.mojo`, `envs/procgen/core/assets.mojo` and both
    Craftax sprite loaders each asked PIL for.
    """
    if img.channels == 4:
        return img.pixels.copy()
    var n = img.width * img.height
    var out = List[UInt8]()
    out.resize(n * 4, 0)
    for i in range(n):
        var o = i * 4
        if img.channels == 1:  # grey
            var g = img.pixels[i]
            out[o] = g
            out[o + 1] = g
            out[o + 2] = g
            out[o + 3] = 255
        elif img.channels == 2:  # grey + alpha
            var g = img.pixels[i * 2]
            out[o] = g
            out[o + 1] = g
            out[o + 2] = g
            out[o + 3] = img.pixels[i * 2 + 1]
        else:  # RGB
            out[o] = img.pixels[i * 3]
            out[o + 1] = img.pixels[i * 3 + 1]
            out[o + 2] = img.pixels[i * 3 + 2]
            out[o + 3] = 255
    return out^


def load_png_file(path: String) raises -> PngImage:
    var b = read_file_bytes(path)
    return decode_png(b)


# ═══════════════════════════════════════════════════════════════════════════
# Writing
# ═══════════════════════════════════════════════════════════════════════════


def _be32_bytes(v: Int) -> List[UInt8]:
    var out = List[UInt8]()
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))
    return out^


def _chunk(tag: String, ref data: List[UInt8]) raises -> List[UInt8]:
    """`length | type | data | CRC`, where the CRC covers TYPE AND DATA.

    ⚠ NOT THE LENGTH. A CRC computed over the length field too produces a file
    every decoder rejects — and the mistake is invisible until something reads
    it back, which is why `tests/io/test_png_write.mojo` reads every image it
    writes with BOTH this repo's decoder and Pillow.
    """
    var body = List[UInt8]()
    for i in range(tag.byte_length()):
        body.append(tag.as_bytes()[i])
    for i in range(len(data)):
        body.append(data[i])

    var out = _be32_bytes(len(data))
    for i in range(len(body)):
        out.append(body[i])
    var c = crc32(body)
    for b in _be32_bytes(c):
        out.append(b)
    return out^


def encode_png(
    ref pixels: List[UInt8], width: Int, height: Int, channels: Int,
    level: Int = 6,
) raises -> List[UInt8]:
    """Encode HWC 8-bit pixels as a PNG. 1, 2, 3 or 4 channels.

    ⚠ EVERY ROW IS WRITTEN WITH FILTER 0 (None). Filtering is an ENCODER's
    choice and none of it is required for a valid file; picking per row would
    buy maybe 30 % on these images and add the one part of the format that is
    genuinely easy to get wrong. The decoder above implements all five because
    it has to read what other encoders produced; this only has to be read.
    """
    if channels < 1 or channels > 4 or channels == 0:
        raise Error("png: cannot encode " + String(channels) + " channels")
    var colour: Int
    if channels == 1: colour = PNG_GREY
    elif channels == 2: colour = PNG_GREY_ALPHA
    elif channels == 3: colour = PNG_RGB
    else: colour = PNG_RGBA
    if len(pixels) != width * height * channels:
        raise Error(
            "png: " + String(len(pixels)) + " bytes for a " + String(width)
            + "x" + String(height) + "x" + String(channels) + " image"
        )

    var stride = width * channels
    var raw = List[UInt8]()
    raw.resize((stride + 1) * height, 0)
    for y in range(height):
        raw[y * (stride + 1)] = 0  # filter: None
        var src = y * stride
        var dst = y * (stride + 1) + 1
        for x in range(stride):
            raw[dst + x] = pixels[src + x]

    var ihdr = List[UInt8]()
    for b in _be32_bytes(width):
        ihdr.append(b)
    for b in _be32_bytes(height):
        ihdr.append(b)
    ihdr.append(UInt8(8))       # bit depth
    ihdr.append(UInt8(colour))  # colour type
    ihdr.append(UInt8(0))       # compression: deflate
    ihdr.append(UInt8(0))       # filter method: adaptive
    ihdr.append(UInt8(0))       # interlace: none

    var idat = deflate(raw, level)
    var empty = List[UInt8]()

    var out = List[UInt8]()
    for b in [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]:
        out.append(UInt8(b))
    for part in [
        _chunk(String("IHDR"), ihdr),
        _chunk(String("IDAT"), idat),
        _chunk(String("IEND"), empty),
    ]:
        for i in range(len(part)):
            out.append(part[i])
    return out^


def save_png(
    path: String, ref pixels: List[UInt8], width: Int, height: Int,
    channels: Int, level: Int = 6,
) raises:
    """Write HWC 8-bit pixels to `path` as a PNG."""
    var bytes = encode_png(pixels, width, height, channels, level)
    from .fileio import write_file_atomic
    write_file_atomic(path, bytes)
