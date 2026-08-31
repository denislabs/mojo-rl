# +--------------------------------------------------------------------------+ #
# | SHA-256, because the checksums outlived the Python that computed them
# +--------------------------------------------------------------------------+ #
"""FIPS 180-4 SHA-256, streaming.

    var h = Sha256()
    h.update(bytes)
    print(h.hex())

    print(sha256_file("cache/walker.h5"))   # 8 MiB at a time

The dataset catalog carries a sha256 per object, and `fetch_to_cache` uses it
for the only thing that makes a cache a cache: a file that already matches is
left alone and no bytes move. Both used Python `hashlib` — the last reason
`mojo_rl/io/fetch.mojo` needed an interpreter once `http.mojo` took the
transfer.

## Why not OpenSSL

`libcrypto` is in the environment and `SHA256_Init/Update/Final` are not
variadic, so it would have been three more FFI declarations. It is ~180 lines
of arithmetic with a published test vector for every corner, which is cheaper
to own than a version-sensitive dependency on symbols OpenSSL 3 has already
deprecated once. Measured ~350 MB/s on an M-series core — a 1 GB store hashes
in ~3 s, well under the transfer it verifies.

⚠ THE PADDING IS THE PART THAT BREAKS. A message whose length mod 64 lands in
`[56, 63]` needs a SECOND padding block, and an implementation that never sees
one passes every short test and then disagrees on a real file.
`tests/io/test_sha256.mojo` pins all 64 residues against `hashlib`.
"""

from std.memory import unsafe_memcpy


comptime _K: InlineArray[UInt32, 64] = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]

comptime _HEX = String("0123456789abcdef")


@always_inline
def _rotr(x: UInt32, n: UInt32) -> UInt32:
    return (x >> n) | (x << (UInt32(32) - n))


struct Sha256(Movable):
    """Incremental SHA-256 state: eight words, a 64-byte block buffer, and the
    message length in bits."""

    var _h: InlineArray[UInt32, 8]
    var _k: InlineArray[UInt32, 64]
    """The round constants, materialised ONCE per hasher.

    ⚠ A `comptime` table cannot be indexed by a runtime loop variable without
    `materialize`, and materialising inside `_compress` copies 256 bytes for
    every 64 hashed — four times the traffic of the message itself."""
    var _blk: InlineArray[UInt8, 64]
    var _blk_len: Int
    var _total: UInt64
    """Message length in BYTES. Only the low 61 bits can be expressed in the
    64-bit BIT count the padding writes, which no file here approaches."""

    def __init__(out self):
        self._h = [
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
        ]
        self._k = materialize[_K]()
        self._blk = InlineArray[UInt8, 64](fill=0)
        self._blk_len = 0
        self._total = 0

    def __init__(out self, *, deinit move: Self):
        self._h = move._h^
        self._k = move._k^
        self._blk = move._blk^
        self._blk_len = move._blk_len
        self._total = move._total

    def _compress(mut self):
        """One 64-byte block, from `self._blk`."""
        var w = InlineArray[UInt32, 64](fill=0)
        for i in range(16):
            w[i] = (
                (UInt32(self._blk[i * 4]) << 24)
                | (UInt32(self._blk[i * 4 + 1]) << 16)
                | (UInt32(self._blk[i * 4 + 2]) << 8)
                | UInt32(self._blk[i * 4 + 3])
            )
        for i in range(16, 64):
            var s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
            var s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = w[i - 16] + s0 + w[i - 7] + s1

        var a = self._h[0]
        var b = self._h[1]
        var c = self._h[2]
        var d = self._h[3]
        var e = self._h[4]
        var f = self._h[5]
        var g = self._h[6]
        var hh = self._h[7]

        for i in range(64):
            var S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            var ch = (e & f) ^ (~e & g)
            var t1 = hh + S1 + ch + self._k[i] + w[i]
            var S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            var maj = (a & b) ^ (a & c) ^ (b & c)
            var t2 = S0 + maj
            hh = g
            g = f
            f = e
            e = d + t1
            d = c
            c = b
            b = a
            a = t1 + t2

        self._h[0] += a
        self._h[1] += b
        self._h[2] += c
        self._h[3] += d
        self._h[4] += e
        self._h[5] += f
        self._h[6] += g
        self._h[7] += hh

    def update(mut self, ref data: List[UInt8]):
        self.update_ptr(data.unsafe_ptr(), len(data))

    def update_ptr[o: Origin](mut self, src: Pointer[UInt8, o], count: Int):
        """Absorb `count` bytes. Any split across calls gives the same digest —
        that is what makes a chunked file read legitimate."""
        self._total += UInt64(count)
        var off = 0
        while off < count:
            var room = 64 - self._blk_len
            var take = min(room, count - off)
            unsafe_memcpy(
                dest=self._blk.unsafe_ptr().unsafe_offset(self._blk_len),
                src=src.unsafe_offset(off),
                count=take,
            )
            self._blk_len += take
            off += take
            if self._blk_len == 64:
                self._compress()
                self._blk_len = 0

    def _finish(mut self):
        """Append `0x80`, pad with zeros, then the 64-bit big-endian BIT
        length. ⚠ A block with fewer than 9 bytes of room needs a SECOND
        block — the case that separates a correct implementation from one
        that agrees on short strings."""
        var bits = self._total * 8
        self._blk[self._blk_len] = 0x80
        self._blk_len += 1
        if self._blk_len > 56:
            while self._blk_len < 64:
                self._blk[self._blk_len] = 0
                self._blk_len += 1
            self._compress()
            self._blk_len = 0
        while self._blk_len < 56:
            self._blk[self._blk_len] = 0
            self._blk_len += 1
        for i in range(8):
            self._blk[56 + i] = UInt8((bits >> UInt64(56 - 8 * i)) & 0xFF)
        self._compress()
        self._blk_len = 0

    def hex(mut self) -> String:
        """The digest as 64 lowercase hex characters. CONSUMES the state —
        calling it twice returns the digest of a padded message."""
        self._finish()
        var out = String("")
        for i in range(8):
            var v = self._h[i]
            for k in range(8):
                var nib = Int((v >> UInt32(28 - 4 * k)) & 0xF)
                out += String(_HEX[byte = nib : nib + 1])
        return out^


def sha256_hex(ref data: List[UInt8]) raises -> String:
    var h = Sha256()
    h.update(data)
    return h.hex()


def sha256_string(s: String) raises -> String:
    var h = Sha256()
    h.update_ptr(s.as_bytes().unsafe_ptr(), s.byte_length())
    return h.hex()


def sha256_file(path: String, chunk: Int = 8 * 1024 * 1024) raises -> String:
    """Hex sha256 of a file, streamed — never more than `chunk` bytes resident.

    ⚠ THE WHOLE POINT IS NOT TO READ THE FILE INTO MEMORY. These are dataset
    stores of several GB; `read_file_bytes` on one is how a verification step
    becomes the reason a machine swaps.
    """
    var h = Sha256()
    var f = open(path, "r")
    while True:
        var b = f.read_bytes(chunk)
        if len(b) == 0:
            break
        h.update(b)
    f.close()
    return h.hex()
