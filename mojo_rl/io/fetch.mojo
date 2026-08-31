# +--------------------------------------------------------------------------+ #
# | Resumable HTTP fetch to a local cache
# +--------------------------------------------------------------------------+ #
"""Download a URL to a cache path, resumably, with a sha256 check.

    _ = fetch_to_cache(url, "cache/walker.h5", sha, size, "walker")
    _ = upload_file(presigned_url, "cache/walker.h5", "walker")

## What this used to be

A 200-line Python program held in a Mojo string and `exec`'d, for one reason
worth remembering: driving the chunk loop from Mojo through `PythonObject`
leaked every chunk until the function RETURNED — RSS looked flat while the
process footprint grew ~1:1 with the downloaded volume, 23 GB at 34 % of a
download, filling swap. Keeping the loop inside Python freed each chunk per
iteration.

`io/http.mojo` retires the whole question: the loop is now libcurl's, in C,
writing straight to a file descriptor. Nothing is allocated per chunk on
either side.

## What was kept, because it was painful to get right

* **Resume by HTTP `Range`**, with a bounded retry count, so a multi-GB
  transfer survives a dropped connection instead of starting over.
* ⚠ **A server that ignores `Range` answers 200 and sends the WHOLE file.**
  Appending that onto a partial download is silent corruption. The shim
  refuses to write those bytes at all (`range_ignored`), and this restarts
  from zero.
* **`F_NOCACHE` (macOS) / periodic `fsync` + `POSIX_FADV_DONTNEED` (Linux)**
  so a multi-GB write does not evict everything else from the page cache.
  That now lives in `native/mrl_http.c`, next to the write it applies to.
* ⚠ **Downloads to `<dest>.part` and renames only on success.** A crashed
  transfer must never leave a truncated file where the next run would treat
  it as complete.

## Timeouts

⚠ NO TOTAL TIMEOUT. A limit large enough for the slowest acceptable link
still kills a healthy transfer of a bigger file — the failure mode is a
dataset that downloads on a fast day and not on a slow one. The guard is on
THROUGHPUT instead: under 1 KB/s for 60 s is a dead connection, at any size.
"""

from std.os import makedirs
from std.os.path import exists
from std.time import sleep

from .fileio import file_size, parent_dir, remove_file, rename_over
from .http import HttpClient
from .sha256 import sha256_file as _sha256_file


comptime MAX_RETRIES = 30
comptime R2_SINGLE_PUT_LIMIT = 5 * 1000 * 1000 * 1000


def sha256_file(path: String) raises -> String:
    """Hex sha256 of a local file, streamed in 8 MiB blocks.

    Kept here as a forward so the callers that imported it from `io.fetch`
    (`data/remote.mojo`, `data/manifest.mojo`) did not have to move when the
    implementation stopped being `hashlib`.
    """
    return _sha256_file(path)


def fetch_to_cache(
    url: String,
    dest: String,
    expect_sha256: String = String(""),
    expect_size: Int = 0,
    label: String = String("fetch"),
) raises -> String:
    """Download `url` to `dest`, resuming and verifying. Returns `dest`.

    A `dest` that already exists and matches `expect_sha256` is left alone and
    no bytes move — the point of carrying the hash in the catalog.
    """
    var part = dest + ".part"
    makedirs(parent_dir(dest), exist_ok=True)

    # Already present and verified? Skip the transfer entirely — this is what
    # makes the remote store a cache rather than a toll booth.
    if exists(dest) and expect_sha256.byte_length() > 0:
        if sha256_file(dest) == expect_sha256:
            print("  [" + label + "] cached and verified: " + dest)
            return dest
        print("  [" + label + "] cached file failed its sha256 - refetching")
        remove_file(dest)

    var c = HttpClient()
    c.timeout_ms(0, 30000)  # see the module docstring: throughput, not total
    c.stall_guard(1024, 60)

    var retries = 0
    while True:
        var offset = file_size(part) if exists(part) else 0
        if expect_size > 0 and offset == expect_size:
            break  # a previous run finished the bytes but not the rename

        var failed: String
        try:
            var r = c.download(url, part, offset, label)
            if r.ok():
                break
            # ⚠ 416 on a resume means the server has nothing past `offset` —
            # the file is already whole. Any other non-2xx is a real failure,
            # and the body is the diagnosis.
            if r.status == 416 and offset > 0:
                break
            raise Error(
                "GET " + url + " -> " + String(r.status) + ": " + r.text()
            )
        except e:
            failed = String(e)

        if c.range_ignored():
            print()
            print(
                "  [" + label + "] server ignored Range - restarting from 0"
            )
            remove_file(part)
            retries += 1
            if retries > MAX_RETRIES:
                raise Error(
                    "[" + label + "] the server ignored Range " + String(retries)
                    + " times"
                )
            continue

        retries += 1
        if retries > MAX_RETRIES:
            raise Error(
                "[" + label + "] download failed after " + String(MAX_RETRIES)
                + " retries (last: " + failed + ")"
            )
        print()
        print(
            "  [" + label + "] error at " + String(offset) + " bytes -"
            " resuming (retry " + String(retries) + "/" + String(MAX_RETRIES)
            + "): " + failed
        )
        sleep(2.0)

    var got_size = file_size(part)
    if expect_size > 0 and got_size != expect_size:
        raise Error(
            "[" + label + "] downloaded " + String(got_size) + " bytes, the"
            " catalog says " + String(expect_size) + " — the transfer was"
            " truncated (the .part file is kept at " + part + ")"
        )

    if expect_sha256.byte_length() > 0:
        var got = sha256_file(part)
        if got != expect_sha256:
            raise Error(
                "[" + label + "] sha256 mismatch: got " + got + ", expected "
                + expect_sha256 + " (the .part file is kept at " + part
                + " for inspection)"
            )

    rename_over(part, String(dest))
    print("  [" + label + "] done: " + dest)
    return dest


def upload_file(
    url: String, path: String, label: String = String("upload")
) raises -> Int:
    """PUT a local file to a presigned URL. Returns the HTTP status.

    ⚠ Single-part. R2 caps a single PUT at 5 GB; past that the object needs a
    multipart upload, which `rclone` / `aws s3 cp` already implement against
    the same presigned flow — see `docs/DATA_PLATFORM_PLAN.md` §6b. This
    raises rather than silently truncating.
    """
    var size = file_size(path)
    if size > R2_SINGLE_PUT_LIMIT:
        raise Error(
            "upload_file: " + path + " is " + String(size // 1000000)
            + " MB, past R2's 5 GB single-PUT limit. Use a multipart-capable"
            " client (rclone / aws s3 cp) against the same presigned URL."
        )

    var c = HttpClient()
    c.timeout_ms(0, 30000)
    c.stall_guard(1024, 60)
    var r = c.upload(url, path, String("PUT"), label)
    if not r.ok():
        raise Error(
            "[" + label + "] PUT " + url + " -> " + String(r.status) + ": "
            + r.text()
        )
    print(
        "  [" + label + "] uploaded " + String(size // 1000000) + " MB,"
        " status " + String(r.status)
    )
    return r.status
