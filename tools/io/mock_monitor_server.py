"""A recording stand-in for the rl-monitor Worker, for the sink gates.

    python3 tools/io/mock_monitor_server.py <port-file> <log-file> [<seconds>]

Binds port 0, writes the chosen port to `<port-file>` atomically, and appends
one line per request to `<log-file>`:

    <monotonic_ms> <METHOD> <path> <body-or-"<N bytes>">

⚠ IT RECORDS RATHER THAN VALIDATES. The gates' questions are about WHEN, HOW
OFTEN and WITH WHAT BYTES a client speaks — a heartbeat that fires while
payloads are flowing, or a checkpoint uploaded four times when once would do,
are defects no status code can express. So the fixture's whole job is to leave
an ordered, timestamped trace for the Mojo side to assert against.

It plays TWO roles, because the artifact flow needs both and neither may reach
the network:

  the Worker   POST /artifacts  -> {id, upload_url} pointing back at this server
               POST /artifacts/<id>/complete
               POST /runs, /ingest, /runs/<id>/ping, /runs/<id>/finish
  R2           PUT  /r2/<key>   -> stores the bytes in memory, records the sha256
               GET  /r2/<key>   -> hands them back

⚠⚠ A PUT WHOSE KEY CONTAINS `slow` SLEEPS FOR `SLOW_PUT_MS`, and that is not a
convenience. `ArtifactSink`'s supersede rule only does anything when requests
arrive FASTER THAN TRANSFERS COMPLETE — which is the real condition (a 215 MB
ACT checkpoint against a validation every N steps) and never the loopback one,
where a PUT finishes before the next offer is made. Without this the gate
measures a sink that had nothing to collapse and reports success.

⚠ `/fail/...` prefixes let a gate ask for a failure on demand, which is the one
thing a real server will not do reliably.

Routes: anything else 200s. `/__shutdown` exits.
"""

import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

LOG = None
T0 = time.monotonic()
PORT = 0

# key -> bytes, and the artifact rows the "Worker" has seen.
OBJECTS = {}
ARTIFACTS = {}
SLOW_PUT_MS = 150
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _record(self, method, note):
        ms = int((time.monotonic() - T0) * 1000)
        with LOCK:
            with open(LOG, "a") as f:
                f.write(f"{ms} {method} {self.path} {note}\n")

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _raw(self, code, blob, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        if blob:
            self.wfile.write(blob)

    def log_message(self, *a):
        pass

    # -- routes -----------------------------------------------------------

    def do_PUT(self):
        u = urlparse(self.path)
        blob = self._body()
        if "slow" in u.path:
            time.sleep(SLOW_PUT_MS / 1000.0)
        self._record("PUT", f"<{len(blob)} bytes sha={hashlib.sha256(blob).hexdigest()[:16]}>")
        if u.path.startswith("/fail/"):
            return self._json(500, {"error": "asked to fail"})
        with LOCK:
            OBJECTS[u.path] = blob
        self._raw(200, b"")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/__shutdown":
            os._exit(0)
        if u.path.startswith("/r2/"):
            with LOCK:
                blob = OBJECTS.get(u.path)
            self._record("GET", f"<{len(blob) if blob else 0} bytes>")
            if blob is None:
                return self._json(404, {"error": "no such object"})
            return self._raw(200, blob)
        self._record("GET", "")
        self._json(200, {"ok": True})

    def do_POST(self):
        u = urlparse(self.path)
        body = self._body()
        text = body.decode("utf-8", "replace")
        self._record("POST", text)

        if u.path == "/__shutdown":
            os._exit(0)

        # A registration the gate has asked to fail, so the retry path and the
        # "abandoned" accounting can be exercised without unplugging anything.
        if "/fail" in u.path:
            return self._json(500, {"error": "asked to fail"})

        if u.path == "/artifacts":
            try:
                d = json.loads(text)
            except Exception:
                return self._json(400, {"error": "bad json"})
            run_id, path = d.get("run_id", ""), d.get("path", "")
            key = f"/r2/{run_id}/{path}"
            # One row per (run, path), like the real Worker: the same id comes
            # back for a re-registration, which is what supersede relies on.
            with LOCK:
                aid = ARTIFACTS.setdefault(key, f"art-{len(ARTIFACTS) + 1}")
            return self._json(
                201,
                {
                    "id": aid,
                    "run_id": run_id,
                    "path": path,
                    "object_key": key,
                    "upload_url": f"http://127.0.0.1:{PORT}{key}",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )

        if u.path.endswith("/complete"):
            return self._json(200, {"ok": True, "status": "ready"})

        self._json(200, {"ok": True, "cmd": ""})


def main():
    global LOG, PORT
    port_file, LOG = sys.argv[1], sys.argv[2]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    open(LOG, "w").close()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    PORT = srv.server_address[1]
    tmp = port_file + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(PORT))
    os.rename(tmp, port_file)
    srv.timeout = 0.2
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        srv.handle_request()


if __name__ == "__main__":
    main()
