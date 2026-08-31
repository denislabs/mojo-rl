# +--------------------------------------------------------------------------+ #
# | The tar reader, on three formats and two attacks
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/io/tar.mojo` and `gunzip_file` in `mojo_rl/io/http.mojo`.

    pixi run build-http                     # ONCE (gunzip_file lives there)
    pixi run mojo run -I . tests/io/test_tar.mojo

Self-contained: `tools/io/make_tar_fixtures.py` builds every archive from
Python's stdlib `tarfile`, which is the independent implementation here.

⚠ THREE FORMATS, BECAUSE ONE IS NOT A TEST. `ustar`, `pax` and `gnu` disagree
about exactly the thing a naive reader gets wrong: a path longer than the
100-byte header field lives in a data block of its own, and the header that
follows repeats it TRUNCATED. A reader that only looks at the header field
extracts `long.bin` under a mangled path and reports success. macOS `bsdtar`
writes PAX for every member, so "it worked on my archive" proves nothing about
a colleague's.

⚠ TWO ARCHIVES MUST BE REFUSED. `traversal.tar` holds one member named
`../escaped.bin` — the "tar slip" bug, and this reader's input arrives over
the network. `symlink.tar` holds a member type the reader does not implement,
which must RAISE rather than be skipped: a skipped member is a partial
extraction that looks complete.

Member shapes cover the block arithmetic: 0 bytes, exactly 512 (no padding),
513 (one byte into a second block), and 1 MiB + 12345 (past the reader's copy
chunk).
"""

from std.os.path import exists

from mojo_rl.io.fileio import file_size, read_file_bytes
from mojo_rl.io.http import gunzip_file
from mojo_rl.io.proc import run_capture
from mojo_rl.io.tar import untar


comptime FIX = "/tmp/mojo_rl_tar_fixtures"
def _long_name() -> String:
    """Built the way `make_tar_fixtures.py` builds it, rather than pasted.

    ⚠ A PASTED COPY OF A 190-CHARACTER PATH IS ITS OWN BUG. The first version
    of this file miscounted the run of `a`s and reported a correct extraction
    as a missing member — a gate failing on its own transcription, which is
    the most expensive kind of red.
    """
    var a = String("")
    for _ in range(90):
        a += "a"
    var b = String("")
    for _ in range(80):
        b += "b"
    return "deeply/nested/" + a + "/" + b + "/long.bin"


def _expect_bytes(n: Int) -> List[UInt8]:
    """The generator `make_tar_fixtures.py` writes: `(i * 37 + n) & 0xFF`."""
    var out = List[UInt8]()
    for i in range(n):
        out.append(UInt8((i * 37 + n) & 0xFF))
    return out^


def _check_member(dest: String, name: String, n: Int) raises:
    var path = dest + "/" + name
    if not exists(path):
        raise Error("tar: '" + name + "' was not extracted at all")
    if file_size(path) != n:
        raise Error(
            "tar: '" + name + "' is " + String(file_size(path)) + " bytes,"
            " wanted " + String(n)
        )
    if n == 0:
        return
    var got = read_file_bytes(path)
    var want = _expect_bytes(n)
    for i in range(n):
        if got[i] != want[i]:
            raise Error(
                "tar: '" + name + "' differs at byte " + String(i) + ": "
                + String(Int(got[i])) + " vs " + String(Int(want[i]))
            )


def _check_tree(dest: String, with_long: Bool) raises -> Int:
    var checked = 0
    _check_member(dest, String("empty.bin"), 0)
    _check_member(dest, String("exact_block.bin"), 512)
    _check_member(dest, String("one_over.bin"), 513)
    _check_member(dest, String("big.bin"), (1 << 20) + 12345)
    _check_member(dest, String("sub/small.bin"), 1000)
    checked += 5
    if with_long:
        _check_member(dest, _long_name(), 777)
        checked += 1
    return checked


def main() raises:
    print("=== io/tar ===")
    _ = run_capture(
        "python3 tools/io/make_tar_fixtures.py " + String(FIX)
    )
    var checks = 0

    # ── the three formats ───────────────────────────────────────────
    for fmt in ["ustar", "pax", "gnu"]:
        var dest = String(FIX) + "/out_" + fmt
        _ = run_capture("rm -rf " + dest)
        var n = untar(String(FIX) + "/" + fmt + ".tar", dest)
        # ustar cannot express the long name, so it holds one member fewer.
        var want_files = 5 if fmt == "ustar" else 6
        if n != want_files:
            raise Error(
                fmt + ".tar: extracted " + String(n) + " files, wanted "
                + String(want_files)
            )
        checks += _check_tree(dest, fmt != "ustar")
        print("  " + fmt + ": " + String(want_files) + " members, all bytes match")
    checks += 3

    # ── the gzip path ───────────────────────────────────────────────
    var tar_out = String(FIX) + "/from_gz.tar"
    var n_bytes = gunzip_file(String(FIX) + "/ustar.tar.gz", tar_out)
    if n_bytes != file_size(String(FIX) + "/ustar.tar"):
        raise Error(
            "gunzip_file produced " + String(n_bytes) + " bytes, the plain tar"
            " is " + String(file_size(String(FIX) + "/ustar.tar"))
        )
    var gz_dest = String(FIX) + "/out_gz"
    _ = run_capture("rm -rf " + gz_dest)
    _ = untar(tar_out, gz_dest)
    checks += _check_tree(gz_dest, False) + 1
    print("  gunzip_file + untar: identical to the plain archive")

    # ── the two that must be refused ────────────────────────────────
    var escaped = String(FIX) + "/escaped.bin"
    _ = run_capture("rm -f " + escaped)
    var slip_raised = False
    try:
        _ = untar(String(FIX) + "/traversal.tar", String(FIX) + "/out_slip")
    except:
        slip_raised = True
    if not slip_raised:
        raise Error("a member named '../escaped.bin' was accepted")
    if exists(escaped):
        raise Error(
            "the traversal member WROTE OUTSIDE the destination, at " + escaped
        )
    checks += 2

    var sym_raised = False
    try:
        _ = untar(String(FIX) + "/symlink.tar", String(FIX) + "/out_sym")
    except:
        sym_raised = True
    if not sym_raised:
        raise Error(
            "a symlink member was skipped silently — a partial extraction"
            " that looks complete"
        )
    checks += 1
    print("  traversal and symlink members both refused")

    print("  " + String(checks) + " checks, 0 failing")
    print("[PASS] io/tar")
