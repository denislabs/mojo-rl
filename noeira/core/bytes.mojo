"""Bytes to `String`, ONCE — the fix for this tree's most-repeated defect.

    var s = string_from_bytes(buf)          # List[UInt8]
    var s = string_from_byte_ptr(p, n)      # a pointer + a length

## ⚠⚠ WHY THIS EXISTS: `chr(Int(byte))` IS NOT A BYTE COPY

The obvious spelling, written seventeen times across eleven files in this tree,
is:

    var s = String()
    for i in range(n):
        s += chr(Int(buf[i]))               # WRONG

For any byte above 127 `chr` yields the CODEPOINT of that value, which UTF-8
re-encodes as TWO bytes. So "日本語" becomes mojibake and the string is no
longer the bytes that were read. It survives everywhere because almost every
value these readers see is ASCII in practice — which is exactly why it is still
here, and why it surfaces one dataset at a time.

Measured, in `TrajectoryStore`'s manifest reader:

    wrote  "Push the ünïcøde block ✓"
    read   "Push the Ã¼nÃ¯cÃ¸de block â"

## ⚠ AND IT HIDES FROM DIFFERENTIAL GATES

The corrupted reader sat between two arms of a store-vs-store comparison, so
BOTH sides were wrong identically and the gate was green. A gate cannot see a
defect in a reader that both sides of its comparison share; catching this needs
an arm whose reference does not pass through the code under test — a literal,
or an oracle in another language.

## ⚠ NUL-TERMINATION IS PART OF THE CONTRACT

`String(unsafe_from_utf8_ptr=)` reads to a NUL, so the terminator is appended
here rather than left to each call site to remember. A caller that forgets it
gets whatever follows the buffer in memory, which is the failure mode with the
longest distance between cause and symptom.

⚠ THE BYTES ARE NOT VALIDATED AS UTF-8, deliberately. These readers carry
whatever the source held — a task instruction, a tar header, a thrift column
name — and a validating constructor would turn "this file has an odd byte" into
a crash inside a parser rather than a value the caller can inspect. Round-trip
fidelity is the contract; interpretation is the caller's.
"""


def string_from_bytes(b: List[UInt8]) -> String:
    """Every byte of `b`, verbatim."""
    var o = List[UInt8]()
    for i in range(len(b)):
        o.append(b[i])
    o.append(0)
    return String(unsafe_from_utf8_ptr=o.unsafe_ptr())


def string_from_byte_span(b: List[UInt8], start: Int, end: Int) -> String:
    """`b[start:end]`, verbatim. `end` is exclusive and is CLAMPED.

    ⚠ CLAMPED RATHER THAN ABORTING, because every caller of this is a parser
    walking a buffer whose bounds came from the file it is parsing. A malformed
    length field should produce a short string the caller can reject, not take
    the process down inside the parse.
    """
    var o = List[UInt8]()
    var lo = start if start > 0 else 0
    var hi = end if end < len(b) else len(b)
    for i in range(lo, hi):
        o.append(b[i])
    o.append(0)
    return String(unsafe_from_utf8_ptr=o.unsafe_ptr())


def string_from_byte_ptr(
    p: Pointer[Scalar[DType.uint8], MutAnyOrigin], n: Int
) -> String:
    """`n` bytes from `p`, verbatim. For the readers that hold a raw buffer."""
    var o = List[UInt8]()
    for i in range(n):
        o.append(p[unsafe_offset=i])
    o.append(0)
    return String(unsafe_from_utf8_ptr=o.unsafe_ptr())
