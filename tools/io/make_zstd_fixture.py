"""Build the zstd fixture `tests/io/test_http.mojo` streams.

    python3 tools/io/make_zstd_fixture.py <out-prefix>

Writes `<out-prefix>.bin` (the expected plaintext), `<out-prefix>.bin.zst`
(what the fixture server serves) and `<out-prefix>.trunc.zst` (the first half
of it, which is how the test produces a CUT TRANSFER on demand).

⚠ THE PAYLOAD IS INCOMPRESSIBLE ON PURPOSE. A compressible one shrinks to a
few hundred bytes, and a fixture that small cannot be cut in the middle — the
resume path, which is the only reason the streaming decoder exists, would go
untested.

`zstandard` is the independent implementation here: the gate checks that Mojo
+ libzstd reproduce what a different library produced, not that our encoder
and decoder agree with each other.
"""

import hashlib
import os
import sys

import zstandard


def main():
    prefix = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3_000_000
    data = os.urandom(n)
    with open(prefix + ".bin", "wb") as f:
        f.write(data)
    comp = zstandard.ZstdCompressor(level=1).compress(data)
    with open(prefix + ".bin.zst", "wb") as f:
        f.write(comp)
    with open(prefix + ".trunc.zst", "wb") as f:
        f.write(comp[: len(comp) // 2])
    print(
        "plain %d bytes (sha256 %s), zst %d bytes, truncated %d"
        % (n, hashlib.sha256(data).hexdigest(), len(comp), len(comp) // 2)
    )


if __name__ == "__main__":
    main()
