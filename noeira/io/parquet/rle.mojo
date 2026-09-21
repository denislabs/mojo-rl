# +--------------------------------------------------------------------------+ #
# | The RLE / bit-packed hybrid — Parquet's level and dictionary-index encoding
# +--------------------------------------------------------------------------+ #
"""One encoding covers both definition/repetition levels and the dictionary
indices of an `RLE_DICTIONARY` data page.

    encoded-data := (rle-run | bit-packed-run)*
    run-header   := uvarint
                    header & 1 == 1  ->  bit-packed, (header >> 1) GROUPS of 8
                    header & 1 == 0  ->  RLE,        (header >> 1) repeats

An RLE run's repeated value follows the header in `ceil(bit_width / 8)` bytes,
little-endian. A bit-packed run's values are packed **LSB-first**, `bit_width`
bits each, running continuously across byte boundaries; a run of `g` groups
occupies exactly `g * bit_width` bytes.

⚠ TWO DIFFERENT BIT ORDERS EXIST IN PARQUET. This hybrid packs LSB-first; the
deprecated standalone `BIT_PACKED` encoding (enum 4, levels only, pre-2.0
writers) packs MSB-first. They are not interchangeable and a file using the
old one decodes to garbage here, so `page.mojo` rejects encoding 4 by name
rather than letting it through.

⚠ A bit-packed run always encodes a MULTIPLE OF 8 values, so the last run of a
page usually carries a few values past the page's count. Those are padding, not
data — decode the run for its byte length and keep only what the page asked
for. Emitting them shifts every subsequent column value by the overhang.
"""

from .thrift import ByteCursor


def bit_width_for(max_value: Int) -> Int:
    """Bits needed to hold `0..max_value`. Zero when there is nothing to
    encode — a required, non-repeated column has `max_def == 0` and its levels
    occupy no bytes at all."""
    var w = 0
    var v = max_value
    while v > 0:
        w += 1
        v >>= 1
    return w


def rle_decode(
    mut c: ByteCursor, bit_width: Int, count: Int, mut out: List[Int32]
) raises:
    """Append exactly `count` decoded values to `out`, advancing `c`.

    A `bit_width` of 0 consumes no bytes and yields `count` zeros: that is the
    encoding of a level stream whose only possible value is 0.
    """
    if count < 0:
        raise Error("parquet/rle: negative count")
    if bit_width == 0:
        for _ in range(count):
            out.append(Int32(0))
        return
    if bit_width > 32:
        raise Error(
            "parquet/rle: bit width " + String(bit_width) + " exceeds 32"
        )

    var mask = (UInt64(1) << UInt64(bit_width)) - 1
    var done = 0
    while done < count:
        var header = c.uvarint()
        if header & 1 == 1:
            # ── bit-packed run ─────────────────────────────────────────
            var groups = header >> 1
            var nvals = groups * 8
            var nbytes = groups * bit_width
            var start = c.pos
            c._need(nbytes)

            var buf = UInt64(0)
            var bits = 0
            var fed = 0
            for i in range(nvals):
                while bits < bit_width:
                    var b = UInt64(c.at(start + fed)) if fed < nbytes else UInt64(0)
                    fed += 1
                    buf |= b << UInt64(bits)
                    bits += 8
                var v = Int32(Int(buf & mask))
                buf >>= UInt64(bit_width)
                bits -= bit_width
                # Values past `count` are the run's 8-alignment padding.
                if done + i < count:
                    out.append(v)
            done += min(nvals, count - done)
            c.pos = start + nbytes
        else:
            # ── run-length run ─────────────────────────────────────────
            var repeats = header >> 1
            var vbytes = (bit_width + 7) // 8
            var v = 0
            for k in range(vbytes):
                v |= c.u8() << (8 * k)
            var take = min(repeats, count - done)
            for _ in range(take):
                out.append(Int32(v))
            done += take
            if repeats > take and done < count:
                raise Error("parquet/rle: run accounting desynchronised")
