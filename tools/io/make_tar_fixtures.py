"""Build the archives `tests/io/test_tar.mojo` extracts.

    python3 tools/io/make_tar_fixtures.py <out-dir>

Writes, under `<out-dir>/`:

  src/                  the tree the good archives contain
  ustar.tar             GNU/ustar format
  pax.tar               PAX format — what every modern `tar` emits
  gnu.tar               GNU format, which spells a long name differently again
  ustar.tar.gz          the gzip path, for `gunzip_file`
  traversal.tar         ONE member named `../escaped.bin`
  symlink.tar           ONE symlink member

⚠ THE THREE FORMATS ARE THE POINT. A reader tested only against the archive
you happened to build passes on that one and fails on a colleague's: macOS
`bsdtar` writes PAX for every member, GNU `tar` writes its own long-name
blocks, and both put the real path in a data block the following header
repeats TRUNCATED. Reading only the 100-byte header field is a defect that
looks like success.

⚠ `traversal.tar` and `symlink.tar` MUST BE REFUSED. An archive arriving over
the network that can write `../` escapes the destination directory — the "tar
slip" bug. A gate that only checks the happy path cannot see it.

Member shapes cover the boundaries: an empty file, one exactly 512 bytes (a
whole block, so the padding is zero), one 513 (one byte into a second block),
one over the reader's 1 MiB copy chunk, and a name past the 100-byte field.
"""

import os
import shutil
import sys
import tarfile

LONG_NAME = "deeply/nested/" + ("a" * 90) + "/" + ("b" * 80) + "/long.bin"

SHAPES = [
    ("empty.bin", 0),
    ("exact_block.bin", 512),
    ("one_over.bin", 513),
    ("big.bin", (1 << 20) + 12345),
    ("sub/small.bin", 1000),
    (LONG_NAME, 777),
]


def build_src(root):
    src = os.path.join(root, "src")
    shutil.rmtree(src, ignore_errors=True)
    for name, n in SHAPES:
        p = os.path.join(src, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # Deterministic, so the Mojo side can check content without a hash
        # file: byte i of a member of length n is (i * 37 + n) & 0xFF.
        with open(p, "wb") as f:
            f.write(bytes((i * 37 + n) & 0xFF for i in range(n)))
    return src


def main():
    root = sys.argv[1]
    os.makedirs(root, exist_ok=True)
    src = build_src(root)

    for label, fmt in (
        ("ustar", tarfile.USTAR_FORMAT),
        ("pax", tarfile.PAX_FORMAT),
        ("gnu", tarfile.GNU_FORMAT),
    ):
        out = os.path.join(root, label + ".tar")
        # USTAR cannot express the long name at all; it is included in the
        # other two, which is exactly the difference being gated.
        with tarfile.open(out, "w", format=fmt) as t:
            for name, _ in SHAPES:
                if fmt is tarfile.USTAR_FORMAT and name == LONG_NAME:
                    continue
                t.add(os.path.join(src, name), arcname=name)

    with tarfile.open(os.path.join(root, "ustar.tar.gz"), "w:gz",
                      format=tarfile.USTAR_FORMAT) as t:
        for name, _ in SHAPES:
            if name == LONG_NAME:
                continue
            t.add(os.path.join(src, name), arcname=name)

    # ── the two that must be refused ────────────────────────────────
    with tarfile.open(os.path.join(root, "traversal.tar"), "w",
                      format=tarfile.USTAR_FORMAT) as t:
        t.add(os.path.join(src, "sub/small.bin"), arcname="../escaped.bin")

    with tarfile.open(os.path.join(root, "symlink.tar"), "w",
                      format=tarfile.USTAR_FORMAT) as t:
        info = tarfile.TarInfo("a_link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        t.addfile(info)

    print("tar fixtures in %s: %s" % (
        root, ", ".join(sorted(f for f in os.listdir(root) if f.endswith((".tar", ".gz"))))))


if __name__ == "__main__":
    main()
