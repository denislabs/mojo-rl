# +--------------------------------------------------------------------------+ #
# | Base64, because two HuggingFace endpoints ask for it
# +--------------------------------------------------------------------------+ #
"""RFC 4648 §4 base64 — the standard alphabet, with padding.

    var s = b64_encode(bytes)          # bytes -> ASCII
    var b = b64_decode(s)              # and back

Two places in the Hub's write API need it and neither is optional: the
`preupload` endpoint wants a base64 `sample` of each file's first bytes so the
server can decide LFS-vs-regular, and the `commit` endpoint carries every
NON-LFS file as base64 `content` inline in the NDJSON body. A dataset's
`meta/info.json` goes up that way.

⚠ **THE STANDARD ALPHABET, NOT THE URL-SAFE ONE.** `+` and `/`, not `-` and
`_`. They are the same encoding with two characters swapped, which is exactly
the kind of difference that produces a valid-looking string the other side
decodes into different bytes. Nothing here needs the URL-safe variant, so it
is absent rather than optional — a flag would only be a way to get it wrong.

⚠ **PADDING IS EMITTED AND REQUIRED.** The Hub sends and expects `=` padding.
`b64_decode` refuses unpadded input rather than guessing the length, because
guessing is how three bytes become two.
"""


comptime _ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _enc_byte(i: Int) -> UInt8:
    return _ALPHABET.as_bytes()[i]


def b64_encode(ref data: List[UInt8]) raises -> String:
    """`data` as a base64 ASCII string, padded."""
    return b64_encode_n(data, len(data))


def b64_encode_n(ref data: List[UInt8], count: Int) raises -> String:
    """The first `count` bytes of `data` as base64.

    The count is separate so a caller can encode a PREFIX without copying the
    slice first — `preupload` samples the head of a file that may be 200 MB.
    """
    if count < 0 or count > len(data):
        raise Error(
            "base64: asked to encode " + String(count) + " bytes of a "
            + String(len(data)) + " byte buffer"
        )
    var out = List[UInt8]()
    out.reserve(((count + 2) // 3) * 4 + 1)

    var i = 0
    while i + 3 <= count:
        var n = (
            (Int(data[i]) << 16) | (Int(data[i + 1]) << 8) | Int(data[i + 2])
        )
        out.append(_enc_byte((n >> 18) & 63))
        out.append(_enc_byte((n >> 12) & 63))
        out.append(_enc_byte((n >> 6) & 63))
        out.append(_enc_byte(n & 63))
        i += 3

    var left = count - i
    if left == 1:
        var n = Int(data[i]) << 16
        out.append(_enc_byte((n >> 18) & 63))
        out.append(_enc_byte((n >> 12) & 63))
        out.append(UInt8(ord("=")))
        out.append(UInt8(ord("=")))
    elif left == 2:
        var n = (Int(data[i]) << 16) | (Int(data[i + 1]) << 8)
        out.append(_enc_byte((n >> 18) & 63))
        out.append(_enc_byte((n >> 12) & 63))
        out.append(_enc_byte((n >> 6) & 63))
        out.append(UInt8(ord("=")))

    out.append(0)
    return String(unsafe_from_utf8_ptr=out.unsafe_ptr())


def _dec_byte(c: Int) raises -> Int:
    if c >= ord("A") and c <= ord("Z"):
        return c - ord("A")
    if c >= ord("a") and c <= ord("z"):
        return c - ord("a") + 26
    if c >= ord("0") and c <= ord("9"):
        return c - ord("0") + 52
    if c == ord("+"):
        return 62
    if c == ord("/"):
        return 63
    raise Error(
        "base64: '" + chr(c) + "' (0x" + hex(c) + ") is not a base64 character"
    )


def b64_decode(s: String) raises -> List[UInt8]:
    """Decode a padded base64 string.

    ⚠ STRICT. Whitespace, a missing pad, or a length that is not a multiple of
    four all raise. MIME base64 wraps at 76 columns and this deliberately does
    NOT accept that — nothing in this repo produces it, and silently skipping
    unknown bytes is how a truncated body decodes into plausible garbage.
    """
    var b = s.as_bytes()
    var n = s.byte_length()
    if n % 4 != 0:
        raise Error(
            "base64: a padded string's length must be a multiple of 4, got "
            + String(n)
        )
    var out = List[UInt8]()
    if n == 0:
        return out^
    out.reserve((n // 4) * 3)

    var i = 0
    while i < n:
        var pad = 0
        var acc = 0
        for k in range(4):
            var c = Int(b[i + k])
            if c == ord("="):
                # Padding is only legal in the last group, and only as the
                # last one or two characters.
                if i + 4 != n or k < 2:
                    raise Error(
                        "base64: '=' at offset " + String(i + k)
                        + " is not trailing padding"
                    )
                pad += 1
                acc = acc << 6
            else:
                if pad != 0:
                    raise Error(
                        "base64: a character follows the padding at offset "
                        + String(i + k)
                    )
                acc = (acc << 6) | _dec_byte(c)
        out.append(UInt8((acc >> 16) & 0xFF))
        if pad < 2:
            out.append(UInt8((acc >> 8) & 0xFF))
        if pad < 1:
            out.append(UInt8(acc & 0xFF))
        i += 4
    return out^
