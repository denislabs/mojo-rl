# +--------------------------------------------------------------------------+ #
# | Snappy raw-block decompression
# +--------------------------------------------------------------------------+ #
"""The `SNAPPY` codec, which is what `parquet-cpp-arrow` writes by default and
therefore what every LeRobot dataset uses.

This is the RAW block format (no stream framing, no CRC): a varint holding the
uncompressed length, then a sequence of elements, each introduced by a tag
byte whose low two bits pick the element kind.

    tag & 3 == 0   LITERAL. `len-1 = tag >> 2`; values 60..63 mean the real
                   `len-1` is the next 1..4 bytes, little-endian.
    tag & 3 == 1   COPY, 1-byte offset. `len = 4 + ((tag >> 2) & 7)`,
                   `offset = ((tag >> 5) << 8) | next_byte`.  (len 4..11)
    tag & 3 == 2   COPY, 2-byte offset. `len = (tag >> 2) + 1`.
    tag & 3 == 3   COPY, 4-byte offset. `len = (tag >> 2) + 1`.

⚠ **COPIES OVERLAP ON PURPOSE.** A run of 64 identical bytes is encoded as one
literal byte followed by a copy of length 63 at offset 1, so the copy reads
bytes this same copy is still writing. `memcpy` — or any vectorised move —
produces the wrong bytes here; the loop below is deliberately byte-at-a-time
and must stay that way. Dictionary-encoded Parquet pages are full of these
(the RLE runs compress into exactly this shape), so getting it wrong does not
fail loudly, it silently returns plausible numbers.
"""

from .thrift import BPtr


def snappy_uncompressed_length(src: BPtr, n: Int) raises -> Int:
    """Peek the varint preamble without decompressing."""
    var r = 0
    var shift = 0
    for i in range(min(n, 5)):
        var c = Int(src[unsafe_offset=i])
        r |= (c & 0x7F) << shift
        if c < 0x80:
            return r
        shift += 7
    raise Error("snappy: malformed length preamble")


def snappy_decompress(
    src: BPtr, n: Int, dst: BPtr, dst_cap: Int
) raises -> Int:
    """Decompress `src[0:n]` into `dst`, returning the byte count written.

    Raises if the declared length exceeds `dst_cap`, if any copy references a
    byte before the start of the output, or if the stream ends early — all
    three are the shapes a corrupt page takes, and all three would otherwise
    read or write out of bounds.
    """
    var ip = 0
    var declared = 0
    var shift = 0
    while True:
        if ip >= n:
            raise Error("snappy: truncated length preamble")
        var c = Int(src[unsafe_offset=ip])
        ip += 1
        declared |= (c & 0x7F) << shift
        if c < 0x80:
            break
        shift += 7
        if shift > 28:
            raise Error("snappy: length preamble too long")

    if declared > dst_cap:
        raise Error(
            "snappy: block declares " + String(declared) + " bytes but the"
            " destination holds " + String(dst_cap)
        )

    var op = 0
    while ip < n:
        var tag = Int(src[unsafe_offset=ip])
        ip += 1
        var kind = tag & 3

        if kind == 0:
            # ── literal ────────────────────────────────────────────────
            var length = tag >> 2
            if length >= 60:
                var extra = length - 59
                if ip + extra > n:
                    raise Error("snappy: truncated literal length")
                length = 0
                for k in range(extra):
                    length |= Int(src[unsafe_offset = ip + k]) << (8 * k)
                ip += extra
            length += 1
            if ip + length > n:
                raise Error("snappy: literal runs past the end of the block")
            if op + length > declared:
                raise Error("snappy: literal overflows the declared length")
            for k in range(length):
                dst[unsafe_offset = op + k] = src[unsafe_offset = ip + k]
            ip += length
            op += length
        else:
            # ── back-reference ─────────────────────────────────────────
            var length: Int
            var offset: Int
            if kind == 1:
                if ip >= n:
                    raise Error("snappy: truncated 1-byte copy")
                length = 4 + ((tag >> 2) & 7)
                offset = ((tag >> 5) << 8) | Int(src[unsafe_offset=ip])
                ip += 1
            elif kind == 2:
                if ip + 2 > n:
                    raise Error("snappy: truncated 2-byte copy")
                length = (tag >> 2) + 1
                offset = (
                    Int(src[unsafe_offset=ip])
                    | (Int(src[unsafe_offset = ip + 1]) << 8)
                )
                ip += 2
            else:
                if ip + 4 > n:
                    raise Error("snappy: truncated 4-byte copy")
                length = (tag >> 2) + 1
                offset = 0
                for k in range(4):
                    offset |= Int(src[unsafe_offset = ip + k]) << (8 * k)
                ip += 4

            if offset <= 0 or offset > op:
                raise Error(
                    "snappy: copy offset " + String(offset) + " reaches before"
                    " the start of " + String(op) + " decoded bytes"
                )
            if op + length > declared:
                raise Error("snappy: copy overflows the declared length")
            # ⚠ Byte-at-a-time, and overlapping BY DESIGN — see the module
            # docstring. Do not replace this with a block move.
            for k in range(length):
                dst[unsafe_offset = op + k] = dst[unsafe_offset = op - offset + k]
            op += length

    if op != declared:
        raise Error(
            "snappy: decoded " + String(op) + " bytes, header declared "
            + String(declared)
        )
    return op
