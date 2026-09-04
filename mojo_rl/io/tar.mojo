# +--------------------------------------------------------------------------+ #
# | A ustar reader — enough tar to unpack a dataset archive
# +--------------------------------------------------------------------------+ #
"""Extract a `.tar` into a directory, without Python's `tarfile`.

    var n = untar(String("cifar-10-binary.tar"), String("~/.cache/mojo_rl"))

## Scope

POSIX `ustar`: 512-byte headers, regular files (`0` / NUL) and directories
(`5`), the `prefix` field, and the two zero blocks that end the archive. That
is the whole of what `tar czf` writes for a tree of plain files, and it is
what CIFAR-10's `cifar-10-binary.tar.gz` contains.

Also handled, because every modern `tar` emits them: **PAX extended headers**
(`x`, and `g` which is skipped) and **GNU long names** (`L`). Both carry a
member's real path in a data block of their own, which the following header
then repeats truncated — so ignoring them is not an option and skipping them
silently is worse.

⚠ macOS `bsdtar` writes a PAX header for EVERY member. An extractor that
rejects them works on the archive you tested and fails on the one a colleague
made; one that skips them writes the truncated 100-byte names. Neither failure
announces itself.

Rejected BY NAME rather than skipped: long links (`K`), sparse files (`S`),
symlinks and hard links (`1` / `2`), and device nodes. Silently skipping an
entry would produce a partial extraction that looks complete, which for a
dataset means a training run on missing data.

⚠ **AN ARCHIVE IS UNTRUSTED INPUT.** A member named `../../.ssh/authorized_keys`
or `/etc/passwd` writes outside the destination — the "tar slip" bug, and this
archive arrives over the network. Every name is checked for an absolute path
and for a `..` component before anything is opened.

⚠ **THE SIZE FIELD IS OCTAL TEXT**, eleven digits then a NUL or a space, and a
header with a bad one must raise rather than read a wrong length into the next
member's data.

Memory is bounded: members are copied through a 1 MiB buffer, so a 200 MB
archive costs 1 MiB whatever its members weigh.
"""

from mojo_rl.core.bytes import string_from_bytes

from std.os import makedirs
from std.os.path import exists

from .fileio import parent_dir


comptime _BLOCK = 512
comptime _COPY_CHUNK = 1 << 20


def _field(ref block: List[UInt8], start: Int, count: Int) -> String:
    """A NUL/space-terminated ASCII field."""
    var out = String("")
    for i in range(count):
        var c = Int(block[start + i])
        if c == 0 or c == 0x20:
            break
        out += chr(c)
    return out^


def _octal(ref block: List[UInt8], start: Int, count: Int) raises -> Int:
    """An octal text field. Raises on a non-octal digit — a wrong size read as
    zero would silently desynchronise every following header."""
    var s = _field(block, start, count)
    if s.byte_length() == 0:
        return 0
    var v = 0
    var b = s.as_bytes()
    for i in range(s.byte_length()):
        var d = Int(b[i]) - 0x30
        if d < 0 or d > 7:
            raise Error("tar: '" + s + "' is not an octal field")
        v = v * 8 + d
    return v


def _pax_path(ref data: List[UInt8]) raises -> String:
    """The `path=` record of a PAX extended header, or "" if it has none.

    The format is a sequence of `LENGTH KEY=VALUE\n` records, where LENGTH
    counts its own digits, the space, the key, the `=`, the value and the
    newline. Parsing by the declared length rather than by splitting on `\n`
    is what makes a value containing a newline safe.
    """
    var pos = 0
    while pos < len(data):
        # LENGTH, in decimal, up to the space.
        var n = 0
        var digits = 0
        while pos + digits < len(data):
            var d = Int(data[pos + digits]) - 0x30
            if d < 0 or d > 9:
                break
            n = n * 10 + d
            digits += 1
        if digits == 0 or n <= digits or pos + n > len(data):
            raise Error("tar: a malformed PAX record length")
        var body_start = pos + digits + 1  # skip the space
        # ⚠⚠ BYTES — see `core/bytes.mojo`. PAX headers are UTF-8 BY THE TAR
        # SPEC, so this is the one reader in the tree where non-ASCII is not
        # merely possible but expected: a `path` record exists precisely to
        # carry a name the ustar header could not.
        var kb = List[UInt8]()
        var i = body_start
        while i < pos + n and data[i] != 0x3D:  # "="
            kb.append(data[i])
            i += 1
        var key = string_from_bytes(kb)
        if key == "path":
            var vb = List[UInt8]()
            var j = i + 1
            while j < pos + n - 1:  # the record ends with a newline
                vb.append(data[j])
                j += 1
            return string_from_bytes(vb)
        pos += n
    return String("")


def _is_safe(name: String) raises -> Bool:
    """Reject anything that could write outside the destination."""
    if name.byte_length() == 0:
        return False
    if name.startswith("/"):
        return False
    if name == ".." or name.startswith("../"):
        return False
    if "/../" in name:
        return False
    if name.endswith("/.."):
        return False
    return True


def untar(tar_path: String, dest: String, verbose: Bool = False) raises -> Int:
    """Extract `tar_path` under `dest`. Returns the number of files written."""
    makedirs(dest, exist_ok=True)
    var f = open(tar_path, "r")
    var n_files = 0
    var zero_blocks = 0
    # A path carried by a PAX or GNU-long-name block, for the NEXT member.
    var pending_name = String("")

    while True:
        var head = f.read_bytes(_BLOCK)
        if len(head) == 0:
            break
        if len(head) != _BLOCK:
            raise Error(
                "tar: a short header block (" + String(len(head))
                + " bytes) — the archive is truncated"
            )

        var all_zero = True
        for i in range(_BLOCK):
            if head[i] != 0:
                all_zero = False
                break
        if all_zero:
            # Two of these end the archive; anything after them is padding.
            zero_blocks += 1
            if zero_blocks >= 2:
                break
            continue
        zero_blocks = 0

        var name = _field(head, 0, 100)
        var prefix = _field(head, 345, 155)
        if prefix.byte_length() > 0:
            name = prefix + "/" + name
        var size = _octal(head, 124, 12)
        var typeflag = Int(head[156])

        # ── metadata blocks: they name the member that FOLLOWS ───────
        # 'x' PAX extended header, 'g' PAX global header, 'L' GNU long name.
        if typeflag == 0x78 or typeflag == 0x67 or typeflag == 0x4C:
            var meta = f.read_bytes(size)
            if len(meta) != size:
                raise Error("tar: a truncated metadata block")
            var pad0 = (_BLOCK - (size % _BLOCK)) % _BLOCK
            if pad0 > 0:
                _ = f.read_bytes(pad0)
            if typeflag == 0x78:  # x — applies to the next member only
                var p = _pax_path(meta)
                if p.byte_length() > 0:
                    pending_name = p^
            elif typeflag == 0x4C:  # L — the data IS the name, NUL-terminated
                # ⚠ BYTES — a GNU long-name record, same reason as PAX.
                var lb = List[UInt8]()
                for i in range(len(meta)):
                    if meta[i] == 0:
                        break
                    lb.append(meta[i])
                pending_name = string_from_bytes(lb)
            # 'g' is archive-wide metadata with nothing this reader needs.
            continue

        if pending_name.byte_length() > 0:
            name = pending_name
            pending_name = String("")

        # NUL and '0' both mean "regular file"; '5' is a directory.
        var is_file = typeflag == 0 or typeflag == 0x30
        var is_dir = typeflag == 0x35
        if not is_file and not is_dir:
            raise Error(
                "tar: '" + name + "' has type '" + chr(typeflag) + "', which"
                " this reader does not implement (long names, symlinks,"
                " sparse members and PAX headers are refused rather than"
                " skipped — a partial extraction looks complete)"
            )

        if not _is_safe(name):
            raise Error(
                "tar: refusing the member '" + name + "' — it escapes the"
                " destination directory"
            )

        var out_path = dest + "/" + name
        if is_dir:
            makedirs(out_path, exist_ok=True)
            # A directory member still declares size 0; nothing to skip.
            continue

        makedirs(parent_dir(out_path), exist_ok=True)
        var o = open(out_path, "w")
        var left = size
        while left > 0:
            var take = _COPY_CHUNK if left > _COPY_CHUNK else left
            var chunk = f.read_bytes(take)
            if len(chunk) != take:
                raise Error(
                    "tar: '" + name + "' ended after " + String(size - left)
                    + " of " + String(size) + " bytes"
                )
            o.write_bytes(Span(chunk))
            left -= take
        o.close()
        n_files += 1
        if verbose:
            print("    " + name + "  (" + String(size) + " bytes)")

        # ⚠ Member data is padded to a 512-byte boundary. Not skipping the
        # padding reads it as the next header, which fails as "not octal"
        # somewhere further on rather than here.
        var pad = (_BLOCK - (size % _BLOCK)) % _BLOCK
        if pad > 0:
            var skipped = f.read_bytes(pad)
            if len(skipped) != pad:
                raise Error("tar: the archive ends inside a padding block")

    f.close()
    return n_files
