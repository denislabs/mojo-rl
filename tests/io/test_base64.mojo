# +--------------------------------------------------------------------------+ #
# | Base64 vs an independent implementation
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/io/base64.mojo` against Python `base64`, pinned as literals.

    pixi run mojo run -I . tests/io/test_base64.mojo

⚠ THE REFERENCE IS EMBEDDED, NOT RECOMPUTED — same rule as `test_sha256.mojo`.
These strings came out of `base64.b64encode` and are pinned here, so
disagreeing with them is a defect in this repo by construction. `base64` is
Python's STANDARD LIBRARY, so there was no dependency to add in order to gate
against it, exactly as with `hashlib`.

## What the lengths buy

⚠ BASE64 BREAKS ON THE TAIL, and the tail is decided by length mod 3. A
3-divisible input never exercises padding at all, so an implementation that
emits no padding — or pads the wrong count — agrees on "abcdef" and disagrees
on almost every real file. Lengths 0..24 cover every residue mod 3 eight
times, both padding cases, and the empty input.

The byte VALUES matter too: the sequence is `(i * 7 + 3) mod 256`, which
reaches every byte value by length 256 and therefore exercises the whole
alphabet, `+` and `/` included. An encoder written against the URL-safe
alphabet agrees on short ASCII and fails here.

Also gated: `b64_decode` round-trips every vector, `b64_encode_n` agrees with
whole-buffer encoding of the same prefix, and four malformed inputs actually
raise. A decoder that accepted those would turn a truncated body into
plausible bytes.
"""

from mojo_rl.io.base64 import b64_decode, b64_encode, b64_encode_n


comptime _N = 32

comptime _LENS: InlineArray[Int, _N] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 61, 62, 63, 64, 100, 255, 256]

comptime _EXPECT: InlineArray[StaticString, _N] = [
    "",
    "Aw==",
    "Awo=",
    "AwoR",
    "AwoRGA==",
    "AwoRGB8=",
    "AwoRGB8m",
    "AwoRGB8mLQ==",
    "AwoRGB8mLTQ=",
    "AwoRGB8mLTQ7",
    "AwoRGB8mLTQ7Qg==",
    "AwoRGB8mLTQ7Qkk=",
    "AwoRGB8mLTQ7QklQ",
    "AwoRGB8mLTQ7QklQVw==",
    "AwoRGB8mLTQ7QklQV14=",
    "AwoRGB8mLTQ7QklQV15l",
    "AwoRGB8mLTQ7QklQV15lbA==",
    "AwoRGB8mLTQ7QklQV15lbHM=",
    "AwoRGB8mLTQ7QklQV15lbHN6",
    "AwoRGB8mLTQ7QklQV15lbHN6gQ==",
    "AwoRGB8mLTQ7QklQV15lbHN6gYg=",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiP",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlg==",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp0=",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp2k",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp2kq7K5wMfO1dzj6vH4/wYNFBsiKTA3PkVMU1phaG92fYSLkpmgpw==",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp2kq7K5wMfO1dzj6vH4/wYNFBsiKTA3PkVMU1phaG92fYSLkpmgp64=",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp2kq7K5wMfO1dzj6vH4/wYNFBsiKTA3PkVMU1phaG92fYSLkpmgp661",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp2kq7K5wMfO1dzj6vH4/wYNFBsiKTA3PkVMU1phaG92fYSLkpmgp661vA==",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp2kq7K5wMfO1dzj6vH4/wYNFBsiKTA3PkVMU1phaG92fYSLkpmgp661vMPK0djf5u30+wIJEBceJSwzOkFIT1ZdZGtyeYCHjpWco6qxuA==",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp2kq7K5wMfO1dzj6vH4/wYNFBsiKTA3PkVMU1phaG92fYSLkpmgp661vMPK0djf5u30+wIJEBceJSwzOkFIT1ZdZGtyeYCHjpWco6qxuL/GzdTb4unw9/4FDBMaISgvNj1ES1JZYGdudXyDipGYn6attLvCydDX3uXs8/oBCA8WHSQrMjlAR05VXGNqcXh/ho2Um6KpsLe+xczT2uHo7/b9BAsSGSAnLjU8Q0pRWF9mbXR7gomQl56lrLO6wcjP1t3k6/L5AAcOFRwjKjE4P0ZNVFtiaXB3foWMk5qhqK+2vcTL0tng5+71",
    "AwoRGB8mLTQ7QklQV15lbHN6gYiPlp2kq7K5wMfO1dzj6vH4/wYNFBsiKTA3PkVMU1phaG92fYSLkpmgp661vMPK0djf5u30+wIJEBceJSwzOkFIT1ZdZGtyeYCHjpWco6qxuL/GzdTb4unw9/4FDBMaISgvNj1ES1JZYGdudXyDipGYn6attLvCydDX3uXs8/oBCA8WHSQrMjlAR05VXGNqcXh/ho2Um6KpsLe+xczT2uHo7/b9BAsSGSAnLjU8Q0pRWF9mbXR7gomQl56lrLO6wcjP1t3k6/L5AAcOFRwjKjE4P0ZNVFtiaXB3foWMk5qhqK+2vcTL0tng5+71/A==",
]


def _message(n: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(n):
        out.append(UInt8((i * 7 + 3) & 255))
    return out^


def main() raises:
    print("[base64] gate")

    # ⚠ Materialised ONCE. A comptime InlineArray is not ImplicitlyCopyable,
    # and indexing it inside the loop would copy the whole table per iteration.
    var lens = materialize[_LENS]()
    var expect = materialize[_EXPECT]()

    # ── encode ────────────────────────────────────────────────────────
    var compared = 0
    var total_bytes = 0
    for c in range(_N):
        var n = lens[c]
        var msg = _message(n)
        var got = b64_encode(msg)
        var want = String(expect[c])
        if got != want:
            raise Error(
                "base64: encoding " + String(n) + " bytes gave\n  " + got
                + "\nexpected\n  " + want
            )
        compared += 1
        total_bytes += n
    print(
        "  encode: " + String(compared) + "/" + String(_N)
        + " vectors match, " + String(total_bytes) + " bytes encoded"
    )
    if compared != _N:
        raise Error("base64: the encode loop compared nothing")

    # ── decode round trip ─────────────────────────────────────────────
    var round_tripped = 0
    for c in range(_N):
        var n = lens[c]
        var msg = _message(n)
        var back = b64_decode(String(expect[c]))
        if len(back) != n:
            raise Error(
                "base64: decoding vector " + String(c) + " gave "
                + String(len(back)) + " bytes, expected " + String(n)
            )
        for i in range(n):
            if back[i] != msg[i]:
                raise Error(
                    "base64: decoded byte " + String(i) + " of vector "
                    + String(c) + " is " + String(Int(back[i])) + ", expected "
                    + String(Int(msg[i]))
                )
        round_tripped += 1
    print(
        "  decode: " + String(round_tripped) + "/" + String(_N) + " round trip"
    )

    # ── b64_encode_n encodes a PREFIX ─────────────────────────────────
    var big = _message(256)
    for c in range(_N):
        var n = lens[c]
        var got = b64_encode_n(big, n)
        var want = String(expect[c])
        if got != want:
            raise Error(
                "base64: b64_encode_n(buf, " + String(n) + ") gave " + got
                + ", whole-buffer encoding of that prefix gives " + want
            )
    print("  encode_n: prefix encoding agrees on all " + String(_N))

    # ── the refusals ──────────────────────────────────────────────────
    # Each must RAISE. A decoder that accepts them produces plausible bytes
    # from input that was never valid, which is worse than failing.
    var bad = List[String]()
    bad.append(String("AA-A"))      # URL-safe alphabet
    bad.append(String("AAAAA"))     # length not a multiple of 4
    bad.append(String("A=AAAAAA"))  # padding mid-string
    bad.append(String("=AAA"))      # leading pad
    bad.append(String("AA A="))     # embedded whitespace

    var refused = 0
    for i in range(len(bad)):
        var raised = False
        try:
            _ = b64_decode(bad[i])
        except:
            raised = True
        if not raised:
            raise Error(
                "base64: '" + bad[i] + "' was ACCEPTED — it must raise"
            )
        refused += 1
    print("  refusals: " + String(refused) + "/" + String(len(bad)) + " raised")
    if refused != len(bad):
        raise Error("base64: not every malformed input was refused")

    print("[PASS] base64")
