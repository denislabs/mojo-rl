# +--------------------------------------------------------------------------+ #
# | Chunked, atomic whole-file I/O
# +--------------------------------------------------------------------------+ #
"""`read_file_bytes` / `write_file_atomic` — the two rules every large blob
this repo writes has to obey, in one place.

## The two rules

**Chunking.** A single `write(2)` silently stops at `0x7FFFF000` (~2 GiB) on
Linux and returns the short count. Ignoring that return is how checkpoint v2
produced TRUNCATED files that loaded without complaint at DreamerV3 size200m.
`read(2)` has the same cap. So neither direction may issue one call for the
whole payload; both loop in `_CHUNK`-sized pieces.

**Atomicity.** The payload lands in `path + ".tmp"` and is `rename(2)`d over
`path` only once fully written. `rename` is atomic within a filesystem, so a
crash mid-save cannot destroy the previous good file — which for a checkpoint
is the difference between losing one save and losing the run.

## Why it is here and not in `nn/core/checkpoint.mojo`

It was there first, and correctly. It moved when `io/safetensors.mojo` needed
the same two rules: a second transcription of a rule is a second thing to keep
in step, and this repo's most frequent defect shape by a distance is a rule
written inline twice. `checkpoint.mojo` keeps its `_write_file_bytes` /
`_read_file_bytes` names as one-line forwards, so nothing that imported them
changed.
"""

from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.os.path import exists

comptime _CHUNK = 1 << 30
"""1 GiB — comfortably below every single-syscall I/O cap."""


def rename_over(var src: String, var dst: String) raises:
    """`rename(2)`: atomic within a filesystem, which is the whole reason the
    download and the checkpoint both write `<dst>.part`/`<dst>.tmp` first."""
    var rc = external_call["rename", Int32](
        src.as_c_string_slice().unsafe_ptr(),
        dst.as_c_string_slice().unsafe_ptr(),
    )
    if rc != 0:
        raise Error("rename failed: " + src + " -> " + dst)


def remove_file(var path: String) raises:
    """`unlink(2)`. A file that was already gone is NOT an error — every
    caller here means "make sure it is not there", not "delete this"."""
    var rc = external_call["unlink", Int32](
        path.as_c_string_slice().unsafe_ptr()
    )
    if rc != 0 and exists(path):
        raise Error("cannot remove " + path)


def parent_dir(path: String) -> String:
    """The directory part of `path`, or "." when it has none."""
    var b = path.as_bytes()
    var cut = -1
    for i in range(path.byte_length()):
        if Int(b[i]) == 0x2F:  # "/"
            cut = i
    if cut <= 0:
        return String(".") if cut < 0 else String("/")
    return String(path[byte=0:cut])


def write_file_atomic(var path: String, ref content: List[UInt8]) raises:
    """Write `content` to `path` via `path + ".tmp"` + `rename(2)`."""
    var tmp = path + ".tmp"
    with open(tmp, "w") as f:
        var off = 0
        while off < len(content):
            var take = len(content) - off
            if take > _CHUNK:
                take = _CHUNK
            # A bounded slice rather than a raw pointer + a separate length:
            # the chunk bound is then CHECKED against the buffer instead of
            # asserted by the caller.
            f.write_bytes(Span(content)[off : off + take])
            off += take
    rename_over(tmp^, path^)


def read_file_bytes(path: String) raises -> List[UInt8]:
    """Read the whole file, looping until `read_bytes` returns nothing."""
    var out = List[UInt8]()
    with open(path, "r") as f:
        while True:
            var chunk = f.read_bytes(_CHUNK)
            if len(chunk) == 0:
                break
            var old = len(out)
            out.resize(old + len(chunk), 0)
            unsafe_memcpy(
                dest=out.unsafe_ptr().unsafe_offset(old),
                src=chunk.unsafe_ptr(),
                count=len(chunk),
            )
    return out^


def read_file_range(path: String, offset: Int, count: Int) raises -> List[UInt8]:
    """`count` bytes starting at `offset`. Raises on a short read — a caller
    asking for a tensor's bytes wants the tensor, not a prefix of it."""
    if offset < 0 or count < 0:
        raise Error(
            "read_file_range: negative offset/count (" + String(offset) + ", "
            + String(count) + ")"
        )
    var out = List[UInt8]()
    out.reserve(count)
    with open(path, "r") as f:
        _ = f.seek(offset)
        var got = 0
        while got < count:
            var want = count - got
            if want > _CHUNK:
                want = _CHUNK
            var chunk = f.read_bytes(want)
            if len(chunk) == 0:
                raise Error(
                    "read_file_range: '" + path + "' ended after "
                    + String(got) + " of " + String(count)
                    + " bytes from offset " + String(offset)
                )
            var old = len(out)
            out.resize(old + len(chunk), 0)
            unsafe_memcpy(
                dest=out.unsafe_ptr().unsafe_offset(old),
                src=chunk.unsafe_ptr(),
                count=len(chunk),
            )
            got += len(chunk)
    return out^


def file_size(path: String) raises -> Int:
    var f = open(path, "r")
    var n = f.seek(0, 2)
    f.close()
    return Int(n)
