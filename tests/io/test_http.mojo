# +--------------------------------------------------------------------------+ #
# | The HTTP client, against a server that does the awkward things
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/io/http.mojo` and the resume logic in `mojo_rl/io/fetch.mojo`.

    pixi run build-http                       # ONCE
    pixi run mojo run -I . tests/io/test_http.mojo

Hermetic: `tools/io/mock_http_server.py` is started on a loopback port, so
nothing here needs the network, a credential, or a live service.

⚠ THE POINT IS THE AWKWARD PATHS, not that a GET works. Each check below
corresponds to a way the transfer can be silently wrong:

* **A 404 must not touch the output file.** Writing an error page over a
  half-finished `.part` is how a truncated download becomes a "complete" one.
* **A server that ignores `Range`** answers 200 with the WHOLE body. Appending
  that to a partial file is silent corruption and the only signal is the
  status code — `/blob-norange` exists to produce that case on demand, which
  no real server will do.
* **A resumed file must equal a whole one, byte for byte.**
* **`Content-Length` must equal the bytes on disk.** It does not when content
  encoding is left on, and both download callers count bytes.
* **A zstd stream must survive a cut transfer.** `lewm_pusht` streams a 13 GB
  `.zst` into a 47 GB `.h5` and can only afford to resume at the COMPRESSED
  offset with the decompressor intact; restarting is 13 GB of transfer.
* **The request headers must actually arrive** — an `Authorization` that is
  built but never sent fails as a 401 much later, somewhere else.
"""

from std.os.path import exists
from std.time import sleep

from mojo_rl.io.fetch import fetch_to_cache
from mojo_rl.io.fileio import (
    file_size, read_file_bytes, remove_file, write_file_atomic,
)
from mojo_rl.io.http import HttpClient, http_shim_available
from mojo_rl.io.json import parse_json
from mojo_rl.io.proc import run_capture
from mojo_rl.io.sha256 import sha256_hex


comptime PORT_FILE = "/tmp/mojo_rl_http_gate_port"
comptime TMP = "/tmp/mojo_rl_http_gate"


def _blob(n: Int) -> List[UInt8]:
    """Mirror of `mock_http_server.blob` — the fixture's byte generator."""
    var out = List[UInt8]()
    for i in range(n):
        out.append(UInt8((i * 31 + 7) & 0xFF))
    return out^


def _start_server() raises -> String:
    """Launch the fixture and return its base URL."""
    try:
        remove_file(String(PORT_FILE))
    except:
        pass
    # Detached: stdout must not stay open, or `run_capture` would block until
    # the server exits.
    _ = run_capture(
        "python3 tools/io/mock_http_server.py " + String(PORT_FILE)
        + " 120 > /tmp/mojo_rl_http_gate_server.log 2>&1 &"
    )
    for _ in range(100):
        if exists(PORT_FILE):
            var f = open(String(PORT_FILE), "r")
            var port = String(f.read().strip())
            f.close()
            if port.byte_length() > 0:
                return "http://127.0.0.1:" + port
        sleep(0.1)
    raise Error(
        "the mock server never wrote " + String(PORT_FILE) + " — see"
        " /tmp/mojo_rl_http_gate_server.log"
    )


def main() raises:
    print("=== io/http ===")
    if not http_shim_available():
        raise Error(
            "the HTTP shim is not built — run `pixi run build-http` first"
        )

    var base = _start_server()
    print("  fixture at " + base)
    var checks = 0

    var c = HttpClient(10000, 5000)
    c.header(String("X-Gate"), String("mojo-rl"))
    c.bearer(String("k3y"))

    # ── 1. a JSON GET, parsed ───────────────────────────────────────
    var r = c.get(base + "/json", 200)
    var doc = parse_json(r^.take_body())
    if not doc.boolean(doc.field(doc.root(), String("ok"))):
        raise Error("/json did not come back intact")
    if doc.integer(doc.field(doc.root(), String("n"))) != 3:
        raise Error("/json n != 3")
    checks += 2

    # ── 2. the headers we set actually arrive ───────────────────────
    var rh = c.get(base + "/headers", 200)
    var hdoc = parse_json(rh^.take_body())
    var hroot = hdoc.root()
    if hdoc.string(hdoc.field(hroot, String("authorization"))) != "Bearer k3y":
        raise Error("the Authorization header did not arrive")
    if hdoc.string(hdoc.field(hroot, String("x-gate"))) != "mojo-rl":
        raise Error("a custom header did not arrive")
    if hdoc.string(hdoc.field(hroot, String("user-agent"))) != "mojo-rl/1.0":
        raise Error("the User-Agent did not arrive")
    checks += 3

    # ── 3. a 404 is a VALUE, and it carries the server's body ───────
    var r404 = c.get(base + "/missing")
    if r404.status != 404:
        raise Error("expected 404, got " + String(r404.status))
    if r404.ok():
        raise Error("ok() said a 404 was fine")
    var edoc = parse_json(r404^.take_body())
    if edoc.string(edoc.field(edoc.root(), String("error"))) != "nope":
        raise Error("the 404 body was dropped — that body IS the diagnosis")
    checks += 3

    # ── 4. `expect=` turns a status into a raise, body included ─────
    var raised = False
    var msg = String("")
    try:
        _ = c.get(base + "/missing", 200)
    except e:
        raised = True
        msg = String(e)
    if not raised:
        raise Error("expect=200 accepted a 404")
    if "nope" not in msg:
        raise Error("the raise dropped the server's body: " + msg)
    checks += 2

    # ── 5. redirects are followed ───────────────────────────────────
    var rr = c.get(base + "/redirect", 200)
    if "items" not in rr.text():
        raise Error("the 302 was not followed to /json")
    checks += 1

    # ── 6. POST round-trips a body ──────────────────────────────────
    var payload = String('{"a":1,"b":"x"}')
    var rp = c.post_json(base + "/echo", payload, 200)
    if rp.text() != payload:
        raise Error("POST body came back as: " + rp.text())
    checks += 1

    # ── 7. download to a file: bytes, size and Content-Length agree ─
    var n = 40000
    var want = _blob(n)
    var want_sha = sha256_hex(want)
    var dest = String(TMP) + "_blob.bin"
    try:
        remove_file(dest)
    except:
        pass
    var rd = c.download(base + "/blob?n=" + String(n), dest)
    if not rd.ok():
        raise Error("/blob answered " + String(rd.status))
    if file_size(dest) != n:
        raise Error(
            "wrote " + String(file_size(dest)) + " bytes, wanted " + String(n)
        )
    # ⚠ This is the check that fails when content encoding is left on: the
    # header then describes the compressed stream, the file the decompressed
    # one.
    if c.content_length() != n:
        raise Error(
            "Content-Length " + String(c.content_length()) + " != the "
            + String(n) + " bytes on disk"
        )
    var got = read_file_bytes(dest)
    if sha256_hex(got) != want_sha:
        raise Error("the downloaded bytes are not the ones the server sent")
    checks += 4

    # ── 8. a 404 must NOT create or clobber the output file ─────────
    var guard = String(TMP) + "_guard.bin"
    var prefix = List[UInt8]()
    for i in range(1000):
        prefix.append(want[i])
    write_file_atomic(guard, prefix)
    var before = sha256_hex(prefix)
    var r404f = c.download(base + "/missing", guard)
    if r404f.status != 404:
        raise Error("expected a 404 on the download route")
    if file_size(guard) != 1000:
        raise Error(
            "a 404 rewrote the output file: it is now "
            + String(file_size(guard)) + " bytes"
        )
    if sha256_hex(read_file_bytes(guard)) != before:
        raise Error("a 404 changed the output file's contents")
    if len(r404f.body) == 0:
        raise Error("the 404 body should have gone to memory instead")
    checks += 3

    # ── 9. resume: a partial file plus a Range equals the whole ─────
    var resumed = String(TMP) + "_resume.bin"
    try:
        remove_file(resumed)
    except:
        pass
    write_file_atomic(resumed, prefix)
    var rres = c.download(base + "/blob?n=" + String(n), resumed, 1000)
    if rres.status != 206:
        raise Error(
            "a resumed GET answered " + String(rres.status) + ", not 206"
        )
    if sha256_hex(read_file_bytes(resumed)) != want_sha:
        raise Error("the resumed file is not byte-identical to a whole one")
    checks += 2

    # ── 10. a server that IGNORES Range must not corrupt the file ───
    var ignored = String(TMP) + "_ignored.bin"
    try:
        remove_file(ignored)
    except:
        pass
    write_file_atomic(ignored, prefix)
    var refused = False
    try:
        _ = c.download(base + "/blob-norange?n=" + String(n), ignored, 1000)
    except:
        refused = True
    if not refused:
        raise Error(
            "a 200 answer to a Range request was accepted — those bytes would"
            " have been appended to a partial file"
        )
    if not c.range_ignored():
        raise Error("range_ignored() did not report the 200")
    if file_size(ignored) != 1000:
        raise Error(
            "the partial file grew to " + String(file_size(ignored))
            + " bytes: the whole body WAS appended"
        )
    checks += 3

    # ── 11. upload: the server sees the bytes we sent ───────────────
    var up = String(TMP) + "_upload.bin"
    write_file_atomic(up, want)
    var ru = c.upload(base + "/upload", up)
    if not ru.ok():
        raise Error("PUT answered " + String(ru.status))
    var udoc = parse_json(ru^.take_body())
    var uroot = udoc.root()
    if udoc.integer(udoc.field(uroot, String("received"))) != n:
        raise Error(
            "the server received "
            + String(udoc.integer(udoc.field(uroot, String("received"))))
            + " of " + String(n) + " bytes"
        )
    if udoc.string(udoc.field(uroot, String("sha256"))) != want_sha:
        raise Error("the uploaded bytes hash differently at the server")
    checks += 2

    # ── 12. fetch_to_cache: verify, then skip on a second call ──────
    var cached = String(TMP) + "_cached.bin"
    try:
        remove_file(cached)
    except:
        pass
    _ = fetch_to_cache(
        base + "/blob?n=" + String(n), cached, want_sha, n, String("gate")
    )
    if sha256_hex(read_file_bytes(cached)) != want_sha:
        raise Error("fetch_to_cache produced the wrong bytes")
    var mtime_probe = file_size(cached)
    _ = fetch_to_cache(
        base + "/blob?n=" + String(n), cached, want_sha, n, String("gate")
    )
    if file_size(cached) != mtime_probe:
        raise Error("the second fetch changed the file")
    checks += 2

    # ── 13. a WRONG hash must be rejected, not accepted quietly ─────
    var bad = String(TMP) + "_bad.bin"
    try:
        remove_file(bad)
    except:
        pass
    var bad_raised = False
    try:
        _ = fetch_to_cache(
            base + "/blob?n=" + String(n), bad,
            String("0" * 64), n, String("gate"),
        )
    except:
        bad_raised = True
    if not bad_raised:
        raise Error("fetch_to_cache accepted a file whose sha256 was wrong")
    if exists(bad):
        raise Error(
            "a hash failure still renamed the .part into place at " + bad
        )
    checks += 2

    # ── 14. zstd: streamed straight to disk, and resumable ──────────
    #
    # The fixture is built by an independent implementation (`zstandard`), so
    # this checks that libzstd-through-the-shim reproduces what a different
    # library produced — not that our encoder and decoder agree.
    var zpre = String(TMP) + "_zstd"
    _ = run_capture(
        "python3 tools/io/make_zstd_fixture.py " + zpre + " 3000000"
    )
    var plain = read_file_bytes(zpre + ".bin")
    var plain_sha = sha256_hex(plain)
    var zout = String(TMP) + "_zstd_out.bin"
    try:
        remove_file(zout)
    except:
        pass

    var zc = HttpClient(0, 5000)
    zc.zstd_to_file()
    var rz = zc.download(base + "/file?p=" + zpre + ".bin.zst", zout)
    if not rz.ok():
        raise Error("the zstd fixture answered " + String(rz.status))
    if file_size(zout) != len(plain):
        raise Error(
            "decompressed to " + String(file_size(zout)) + " bytes, wanted "
            + String(len(plain))
        )
    if sha256_hex(read_file_bytes(zout)) != plain_sha:
        raise Error("the decompressed bytes are not what zstandard compressed")
    checks += 2

    # A CUT transfer, then a resume at the compressed offset with the SAME
    # decompressor — the case that decides whether a dropped connection costs
    # 13 GB or nothing.
    try:
        remove_file(zout)
    except:
        pass
    var zc2 = HttpClient(0, 5000)
    zc2.zstd_to_file()
    _ = zc2.download(base + "/file?p=" + zpre + ".trunc.zst", zout)
    var cut_at = zc2.zstd_read()
    var partial = file_size(zout)
    if partial == 0 or partial >= len(plain):
        raise Error(
            "the cut transfer produced " + String(partial) + " bytes — it was"
            " meant to stop part way"
        )
    var rz2 = zc2.download(base + "/file?p=" + zpre + ".bin.zst", zout, cut_at)
    if rz2.status != 206:
        raise Error("the resume answered " + String(rz2.status) + ", not 206")
    if sha256_hex(read_file_bytes(zout)) != plain_sha:
        raise Error(
            "the resumed file differs from a whole one — the decompressor did"
            " not survive the retry"
        )
    checks += 3

    _ = c.get(base + "/__shutdown")
    print("  " + String(checks) + " checks, 0 failing")
    print("[PASS] io/http")
