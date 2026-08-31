# +--------------------------------------------------------------------------+ #
# | Pillow-compatible bilinear resize for 8-bit RGB
# +--------------------------------------------------------------------------+ #
"""`resize_bilinear_pil` reproduces `PIL.Image.resize(..., Image.BILINEAR)`
**bit for bit** on an HWC uint8 image.

## Why bit-exact and not merely correct

The ACT store was built by a Python converter that resized with Pillow. If the
Mojo import path resizes even slightly differently, the two stores are
different datasets: the 6.8 GB store on disk stops being reproducible, the
sha256 gate that would otherwise prove the port has nothing to compare, and any
training curve measured against the old store carries an unquantified pixel
delta. Matching Pillow exactly makes the Mojo converter a drop-in replacement
and turns "is the port right?" into a checksum.

`ffmpeg -vf scale` was the obvious alternative and is NOT equivalent — swscale
builds different filters — so the frames come out of ffmpeg at source
resolution and are resized here.

## The algorithm, transcribed from Pillow's `src/libImaging/Resample.c`

Two passes: horizontal into an intermediate, then vertical. Both quantise to
8 bits, so the intermediate rounding is part of the answer and the passes
cannot be fused.

Per output coordinate, with `scale = in/out` and `filterscale = max(scale, 1)`:

    support = 1.0 * filterscale                  (bilinear support is 1.0)
    center  = (i + 0.5) * scale
    xmin    = clamp_lo(trunc(center - support + 0.5), 0)
    xmax    = clamp_hi(trunc(center + support + 0.5), inSize) - xmin
    k[j]    = tri((j + xmin - center + 0.5) / filterscale),  tri(x)=1-|x|
    k      /= sum(k)

then, in fixed point with 22 fractional bits:

    kk[j] = trunc(0.5 + k[j] * 2^22)
    ss    = 2^21 + sum_j pixel[xmin+j] * kk[j]
    out   = clamp(ss >> 22, 0, 255)

⚠ **`trunc`, NOT `floor`.** Pillow's `(int)` casts truncate toward zero, and
`center - support + 0.5` is negative for the first few output pixels of a
downscale. `floor(-0.5)` is -1 and `trunc(-0.5)` is 0; the difference shifts
the whole first column's window by one source pixel.

⚠ **`>>` ON A NEGATIVE ACCUMULATOR IS AN ARITHMETIC SHIFT** (rounds toward
-inf), which is what Pillow's `clip8` lookup assumes. Dividing by 2^22 instead
rounds toward zero and differs by one LSB on the pixels where the accumulator
goes negative — which bilinear's non-negative weights make rare but not
impossible once the coefficient rounding is applied.

Gated bit-exactly against Pillow by `tests/io/test_image_resize.mojo`.
"""

from std.math import ceil, trunc
from std.memory import Pointer


comptime PRECISION_BITS = 32 - 8 - 2
"""22, Pillow's `PRECISION_BITS`."""


@fieldwise_init
struct _Coeffs(Movable):
    """Per-output-coordinate filter windows, in Pillow's fixed point."""

    var ksize: Int
    var bounds_min: List[Int]
    var bounds_len: List[Int]
    var kk: List[Int]
    """`out_size * ksize` integer weights, row-major by output coordinate."""


def _precompute(in_size: Int, out_size: Int) raises -> _Coeffs:
    """`precompute_coeffs` + `normalize_coeffs_8bpc` for the bilinear filter."""
    if in_size <= 0 or out_size <= 0:
        raise Error("resize: non-positive dimension")

    var scale = Float64(in_size) / Float64(out_size)
    var filterscale = scale if scale >= 1.0 else 1.0
    var support = 1.0 * filterscale
    var ksize = Int(ceil(support)) * 2 + 1

    var bmin = List[Int](unsafe_uninit_length=out_size)
    var blen = List[Int](unsafe_uninit_length=out_size)
    var kk = List[Int](unsafe_uninit_length = out_size * ksize)

    var kf = List[Float64](unsafe_uninit_length=ksize)
    for i in range(out_size):
        var center = (Float64(i) + 0.5) * scale
        var inv = 1.0 / filterscale

        var xmin = Int(trunc(center - support + 0.5))
        if xmin < 0:
            xmin = 0
        var xmax = Int(trunc(center + support + 0.5))
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin

        var ww = 0.0
        for j in range(xmax):
            var t = (Float64(j + xmin) - center + 0.5) * inv
            if t < 0.0:
                t = -t
            var w = (1.0 - t) if t < 1.0 else 0.0
            kf[j] = w
            ww += w
        for j in range(xmax):
            if ww != 0.0:
                kf[j] = kf[j] / ww
        for j in range(xmax, ksize):
            kf[j] = 0.0

        for j in range(ksize):
            var v = kf[j]
            # Pillow rounds away from zero, in both directions.
            var q: Float64
            if v < 0.0:
                q = -0.5 + v * Float64(1 << PRECISION_BITS)
            else:
                q = 0.5 + v * Float64(1 << PRECISION_BITS)
            kk[i * ksize + j] = Int(trunc(q))

        bmin[i] = xmin
        blen[i] = xmax

    return _Coeffs(ksize, bmin^, blen^, kk^)


@always_inline
def _clip8(acc: Int) -> Int:
    """Pillow's `clip8`: arithmetic shift down, then clamp to a byte."""
    var v = acc >> PRECISION_BITS
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def resize_bilinear_pil(
    src: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    in_h: Int,
    in_w: Int,
    dst: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    out_h: Int,
    out_w: Int,
    mut scratch: List[UInt8],
    channels: Int = 3,
) raises:
    """Resize `src` (HWC, `channels` bytes per pixel) into `dst` (HWC).

    `scratch` is grown as needed and reused across calls — the intermediate is
    `out_w * in_h * channels` bytes, which for a LeRobot camera is 230 KB per
    frame and would otherwise be a per-frame allocation 15,447 times over.

    A no-op resize still copies, so the caller never has to special-case it.
    """
    if channels <= 0:
        raise Error("resize: channels must be positive")

    if in_h == out_h and in_w == out_w:
        for i in range(in_h * in_w * channels):
            dst[unsafe_offset=i] = src[unsafe_offset=i]
        return

    var half = 1 << (PRECISION_BITS - 1)

    # ── horizontal pass: (in_h, in_w) -> (in_h, out_w) ────────────────
    # Pillow crops the intermediate to the rows the vertical pass will touch;
    # with a full-image box that is every row, so the crop is skipped here and
    # the vertical bounds are used unshifted.
    var need = in_h * out_w * channels
    if len(scratch) < need:
        scratch.resize(need, UInt8(0))
    var tmp = (
        scratch.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
        .as_unsafe_any_origin()
    )

    var hc = _precompute(in_w, out_w)
    for y in range(in_h):
        var srow = y * in_w * channels
        var trow = y * out_w * channels
        for x in range(out_w):
            var xmin = hc.bounds_min[x]
            var xlen = hc.bounds_len[x]
            var kbase = x * hc.ksize
            for c in range(channels):
                var acc = half
                for j in range(xlen):
                    acc += (
                        Int(src[unsafe_offset = srow + (xmin + j) * channels + c])
                        * hc.kk[kbase + j]
                    )
                tmp[unsafe_offset = trow + x * channels + c] = UInt8(_clip8(acc))

    # ── vertical pass: (in_h, out_w) -> (out_h, out_w) ────────────────
    var vc = _precompute(in_h, out_h)
    for y in range(out_h):
        var ymin = vc.bounds_min[y]
        var ylen = vc.bounds_len[y]
        var kbase = y * vc.ksize
        var drow = y * out_w * channels
        for x in range(out_w):
            for c in range(channels):
                var acc = half
                for j in range(ylen):
                    acc += (
                        Int(tmp[
                            unsafe_offset = (ymin + j) * out_w * channels
                            + x * channels + c
                        ])
                        * vc.kk[kbase + j]
                    )
                dst[unsafe_offset = drow + x * channels + c] = UInt8(_clip8(acc))


# ═══════════════════════════════════════════════════════════════════════════
# NEAREST
# ═══════════════════════════════════════════════════════════════════════════


def resize_nearest_pil(
    src: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    in_h: Int,
    in_w: Int,
    dst: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    out_h: Int,
    out_w: Int,
    channels: Int = 4,
) raises:
    """Resize `src` (HWC) into `dst`, matching `PIL.Image.NEAREST` exactly.

    Both Craftax sprite loaders resize their atlases this way, so a one-pixel
    disagreement is a different sprite sheet — visible, but not as an error.

    ⚠ **THE ARITHMETIC IS THE SPECIFICATION, NOT THE MATH.** The obvious
    `floor((x + 0.5) * in / out)` DISAGREES WITH PILLOW on 93 of 600 random
    size pairs, and `floor(x * s + 0.5 * s)` on 104 — every disagreement is a
    case where the exact product lands on an integer and double rounding
    decides which side of it. Pillow's scaling path keeps a RUNNING
    ACCUMULATOR: it starts at `0.5 * scale` and adds `scale` per output pixel,
    so the accumulated floating-point error is part of the answer. Reproducing
    the formula is not enough; the ORDER OF OPERATIONS has to match. This
    version agrees on all 600 pairs, and `tests/io/test_resize_nearest.mojo`
    holds that sweep.

    ⚠ Do not "simplify" the loop below into a multiplication.
    """
    if channels <= 0:
        raise Error("resize: channels must be positive")
    if in_h <= 0 or in_w <= 0 or out_h <= 0 or out_w <= 0:
        raise Error("resize: a zero dimension")

    var sx = Float64(in_w) / Float64(out_w)
    var sy = Float64(in_h) / Float64(out_h)

    # The column map is the same for every row, so it is computed once.
    var xmap = List[Int]()
    xmap.resize(out_w, 0)
    var xx = 0.5 * sx
    for x in range(out_w):
        var i = Int(xx)
        xmap[x] = i if i < in_w else in_w - 1
        xx += sx

    var yy = 0.5 * sy
    for y in range(out_h):
        var iy = Int(yy)
        if iy >= in_h:
            iy = in_h - 1
        yy += sy
        var src_row = iy * in_w * channels
        var dst_row = y * out_w * channels
        for x in range(out_w):
            var so = src_row + xmap[x] * channels
            var do = dst_row + x * channels
            for c in range(channels):
                dst[unsafe_offset = do + c] = src[unsafe_offset = so + c]
