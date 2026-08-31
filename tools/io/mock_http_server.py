"""A mock HTTP server for `tests/io/test_http.mojo`.

    python3 tools/io/mock_http_server.py <port-file> [<seconds>]

Binds port 0, writes the chosen port to `<port-file>` (atomically, via a
`.tmp` + rename, so the reader never sees a half-written number), and serves
the routes below until `/__shutdown` or the timeout.

⚠ IT IS A GATE FIXTURE, NOT A SERVER. It exists so the HTTP client can be
exercised with no network and no credentials: status codes, redirects, Range
resume, an upload, and — the one thing a real server will not do on demand —
a route that IGNORES `Range` and answers 200, which is the silent-corruption
case `mojo_rl/io/fetch.mojo` has to detect.

Routes
  GET  /json           200 {"ok":true,"n":3,"items":[1,2,3]}
  GET  /headers        200, echoes the request headers it received
  POST /echo           200, echoes the request body verbatim
  GET  /blob?n=N       200/206, N deterministic bytes, honours Range
  GET  /blob-norange   200 with the WHOLE body, even when Range was asked
  GET  /missing        404 {"error":"nope"}
  GET  /redirect       302 -> /json
  PUT  /upload         200 {"received":N,"sha256":"..."}
  GET  /file?p=PATH    200/206, that local file, honouring Range
  GET  /__shutdown     200, then the process exits
"""

import hashlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def blob(n):
    """The same generator the Mojo side checks against."""
    return bytes((i * 31 + 7) & 0xFF for i in range(n))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body=b"", ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/json":
            self._send(200, json.dumps(
                {"ok": True, "n": 3, "items": [1, 2, 3]}).encode())
        elif u.path == "/headers":
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            self._send(200, json.dumps(hdrs).encode())
        elif u.path in ("/blob", "/blob-norange"):
            n = int(q.get("n", ["4096"])[0])
            data = blob(n)
            rng = self.headers.get("Range")
            if rng and u.path == "/blob":
                start = int(rng.split("=")[1].split("-")[0])
                part = data[start:]
                self._send(206, part, "application/octet-stream",
                           {"Content-Range":
                            f"bytes {start}-{len(data)-1}/{len(data)}"})
            else:
                # /blob-norange deliberately ignores Range: the client must
                # refuse to append these bytes to a partial file.
                self._send(200, data, "application/octet-stream")
        elif u.path == "/file":
            # A real file with real Range support — what the zstd streaming
            # test needs, since it has to cut a transfer in the middle and
            # resume it at a compressed offset.
            path = q.get("p", [""])[0]
            if not os.path.isfile(path):
                self._send(404, json.dumps({"error": "no such file"}).encode())
                return
            data = open(path, "rb").read()
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                part = data[start:]
                self._send(206, part, "application/octet-stream",
                           {"Content-Range":
                            f"bytes {start}-{len(data)-1}/{len(data)}"})
            else:
                self._send(200, data, "application/octet-stream")
        elif u.path == "/redirect":
            self._send(302, b"redirecting", "text/plain", {"Location": "/json"})
        elif u.path == "/missing":
            self._send(404, json.dumps({"error": "nope"}).encode())
        elif u.path == "/__shutdown":
            self._send(200, b'{"bye":true}')
            # ⚠ WITHOUT THIS THE SERVER HANGS. HTTP/1.1 keep-alive leaves the
            # handler blocked reading the next request on a connection the
            # client is still holding, so `shutdown()` never gets to return.
            self.close_connection = True
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send(404, json.dumps({"error": "unknown route"}).encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path == "/echo":
            self._send(200, body, self.headers.get("Content-Type", "text/plain"))
        elif self.path == "/created":
            self._send(201, json.dumps({"id": "abc123"}).encode())
        else:
            self._send(404, json.dumps({"error": "unknown route"}).encode())

    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path == "/upload":
            self._send(200, json.dumps({
                "received": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }).encode())
        else:
            self._send(404, json.dumps({"error": "unknown route"}).encode())

    def log_message(self, *a):
        pass


def main():
    port_file = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    port = srv.server_address[1]
    # Atomic: the Mojo side polls for this file and must never read a partial
    # number.
    with open(port_file + ".tmp", "w") as f:
        f.write(str(port))
    os.replace(port_file + ".tmp", port_file)
    t = threading.Timer(timeout, srv.shutdown)
    t.daemon = True  # a live Timer would keep the process up for the full wait
    t.start()
    srv.serve_forever()
    srv.server_close()
    try:
        os.remove(port_file)
    except OSError:
        pass
    os._exit(0)  # any lingering keep-alive thread must not hold the exit


if __name__ == "__main__":
    main()
