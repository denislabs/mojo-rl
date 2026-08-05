# +--------------------------------------------------------------------------+ #
# | Resumable HTTP fetch to a local cache
# +--------------------------------------------------------------------------+ #
"""Download a URL to a cache path, resumably, with a sha256 check.

Generalised from `nn/datasets/lewm_pusht.mojo`'s downloader rather than
rewritten, because the parts worth keeping are the ones that were painful to
get right:

* **The whole loop runs inside Python — one exec, one call from Mojo.**
  Driving it chunk-by-chunk from Mojo leaked every large per-iteration
  `PythonObject` until the function returned: RSS looked flat (macOS swapped
  the cold pages) while the process footprint grew ~1:1 with the downloaded
  volume — 23 GB footprint at 34% of a download, filling swap. In-Python each
  chunk is freed per iteration.
* **Resume by HTTP Range**, retrying a bounded number of times, so a multi-GB
  transfer survives a dropped connection instead of starting over.
* **`F_NOCACHE` (macOS) / periodic fsync + `POSIX_FADV_DONTNEED` (Linux)** so a
  multi-GB write does not evict everything else from the page cache.

Differences from the PushT helper: plain HTTP(S) via `urllib` rather than
`HfFileSystem`, no zstd decompression (R2 serves the stored bytes), and a
sha256 check on completion.

⚠ Downloads to `<dest>.part` and renames only on success. A crashed transfer
must never leave a truncated file at the real path where the next run would
treat it as complete.
"""

from std.python import Python, PythonObject


comptime _FETCH_PY: StaticString = """
def fetch_resumable(url, dest, expect_sha256, expect_size, label):
    import hashlib, os, sys, time, urllib.request

    part = dest + '.part'
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)

    # Already present and verified? Skip the transfer entirely — this is what
    # makes the remote store a cache rather than a toll booth.
    if os.path.exists(dest) and expect_sha256:
        h = hashlib.sha256()
        with open(dest, 'rb') as f:
            for blk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(blk)
        if h.hexdigest() == expect_sha256:
            print('  [%s] cached and verified: %s' % (label, dest))
            return dest
        print('  [%s] cached file failed its sha256 - refetching' % label)
        os.remove(dest)

    CHUNK = 8 * 1024 * 1024
    MAX_RETRIES = 30
    offset = os.path.getsize(part) if os.path.exists(part) else 0
    t0 = time.monotonic()
    retries = 0
    total = expect_size or 0

    while True:
        req = urllib.request.Request(url)
        if offset:
            # Resume. A server that ignores Range returns 200, and we must NOT
            # append its full body onto a partial file.
            req.add_header('Range', 'bytes=%d-' % offset)
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            if offset and resp.status != 206:
                print('  [%s] server ignored Range - restarting from 0' % label)
                offset = 0
                if os.path.exists(part):
                    os.remove(part)
                continue
            if not total:
                cl = resp.headers.get('Content-Length')
                total = (int(cl) + offset) if cl else 0

            f_out = open(part, 'ab' if offset else 'wb')
            try:
                import fcntl
                fcntl.fcntl(f_out.fileno(), fcntl.F_NOCACHE, 1)
            except Exception:
                pass  # non-macOS: the fsync+fadvise below covers it

            n = 0
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f_out.write(buf)
                offset += len(buf)
                buf = None
                n += 1
                if n % 32 == 0:
                    f_out.flush()
                    os.fsync(f_out.fileno())
                    if hasattr(os, 'posix_fadvise'):
                        os.posix_fadvise(f_out.fileno(), 0, 0,
                                         os.POSIX_FADV_DONTNEED)
                el = time.monotonic() - t0
                mbs = offset / 1e6 / el if el > 0 else 0.0
                if total:
                    pct = offset * 100 // total
                    filled = offset * 30 // total
                    eta = int((total - offset) / 1e6 / mbs) if mbs > 0 else 0
                    sys.stdout.write(
                        '\\r  [%s] [%s%s] %d%% %.0f MB/s ETA %ds   '
                        % (label, '#' * filled, '.' * (30 - filled), pct,
                           mbs, eta))
                else:
                    sys.stdout.write('\\r  [%s] %d MB %.0f MB/s   '
                                     % (label, offset // 1000000, mbs))
                sys.stdout.flush()
            f_out.close()
            resp.close()
            break
        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                raise RuntimeError(
                    '[%s] download failed after %d retries (last: %s)'
                    % (label, MAX_RETRIES, e))
            print()
            print('  [%s] error at %d bytes - resuming (retry %d/%d): %s'
                  % (label, offset, retries, MAX_RETRIES, e))
            time.sleep(2.0)
            try:
                f_out.close()
            except Exception:
                pass
            offset = os.path.getsize(part) if os.path.exists(part) else 0

    print()
    if expect_sha256:
        h = hashlib.sha256()
        with open(part, 'rb') as f:
            for blk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(blk)
        got = h.hexdigest()
        if got != expect_sha256:
            raise RuntimeError(
                '[%s] sha256 mismatch: got %s, expected %s (the .part file is'
                ' kept at %s for inspection)' % (label, got, expect_sha256, part))

    # Rename only now: a truncated file at `dest` would be treated as complete
    # by the next run.
    os.replace(part, dest)
    print('  [%s] done: %s' % (label, dest))
    return dest


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(blk)
    return h.hexdigest()


def upload_file(url, path, label):
    import os, sys, time, urllib.request
    size = os.path.getsize(path)
    t0 = time.monotonic()
    with open(path, 'rb') as f:
        req = urllib.request.Request(url, data=f, method='PUT')
        req.add_header('Content-Length', str(size))
        resp = urllib.request.urlopen(req, timeout=3600)
        code = resp.status
        resp.close()
    el = time.monotonic() - t0
    print('  [%s] uploaded %d MB in %.1fs (%.0f MB/s), status %d'
          % (label, size // 1000000, el, size / 1e6 / el if el > 0 else 0, code))
    return code
"""


def _helpers() raises -> PythonObject:
    """One `exec` of the helper module; returns its namespace dict.

    Matches `lewm_pusht.mojo`'s idiom exactly — one exec plus one call, with
    the loop living in Python (see the module docstring on the PythonObject
    leak).
    """
    var builtins = Python.import_module("builtins")
    var ns = builtins.dict()
    _ = builtins.exec(PythonObject(_FETCH_PY), ns)
    return ns^


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
    var ns = _helpers()
    var out = ns[PythonObject("fetch_resumable")](
        PythonObject(url), PythonObject(dest), PythonObject(expect_sha256),
        PythonObject(expect_size), PythonObject(label),
    )
    return String(out)


def sha256_file(path: String) raises -> String:
    """Hex sha256 of a local file, streamed in 8 MiB blocks."""
    var ns = _helpers()
    return String(ns[PythonObject("sha256_file")](PythonObject(path)))


def upload_file(
    url: String, path: String, label: String = String("upload")
) raises -> Int:
    """PUT a local file to a presigned URL.

    ⚠ Single-part. R2 caps a single PUT at 5 GB; past that the object needs a
    multipart upload, which `rclone` / `aws s3 cp` already implement against
    the same presigned flow — see `docs/DATA_PLATFORM_PLAN.md` §6b. This
    raises rather than silently truncating.
    """
    var os = Python.import_module("os")
    var size = Int(py=os.path.getsize(PythonObject(path)))
    if size > 5 * 1000 * 1000 * 1000:
        raise Error(
            "upload_file: " + path + " is " + String(size // 1000000)
            + " MB, past R2's 5 GB single-PUT limit. Use a multipart-capable"
            " client (rclone / aws s3 cp) against the same presigned URL."
        )
    var ns = _helpers()
    return Int(
        py=ns[PythonObject("upload_file")](
            PythonObject(url), PythonObject(path), PythonObject(label)
        )
    )
