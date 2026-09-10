"""A recording stand-in for the rl-monitor Worker, for the heartbeat gate.

    python3 tools/io/mock_monitor_server.py <port-file> <log-file> [<seconds>]

Binds port 0, writes the chosen port to `<port-file>` atomically, answers 200
to everything, and appends one line per request to `<log-file>`:

    <monotonic_ms> <METHOD> <path> <body>

⚠ IT RECORDS RATHER THAN VALIDATES. The gate's questions are about WHEN and
HOW OFTEN the client speaks — a heartbeat that fires while payloads are
flowing is the defect, and no status code can express that. So the fixture's
whole job is to leave an ordered, timestamped trace for the Mojo side to
assert against.

Routes: anything. `/__shutdown` exits.
"""

import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = None
T0 = time.monotonic()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _record(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        ms = int((time.monotonic() - T0) * 1000)
        with open(LOG, "a") as f:
            f.write(f"{ms} {method} {self.path} {body}\n")
        payload = b'{"ok":true,"cmd":""}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path == "/__shutdown":
            self._record("POST")
            os._exit(0)
        self._record("POST")

    def do_GET(self):
        if self.path == "/__shutdown":
            os._exit(0)
        self._record("GET")

    def log_message(self, *a):
        pass


def main():
    port_file, LOG_, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3] if len(sys.argv) > 3 else 60)
    global LOG
    LOG = LOG_
    open(LOG, "w").close()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    tmp = port_file + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(srv.server_address[1]))
    os.rename(tmp, port_file)
    srv.timeout = 0.2
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        srv.handle_request()


if __name__ == "__main__":
    main()
