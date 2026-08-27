# +--------------------------------------------------------------------------+ #
# | Remote dataset catalog client (R2 via mojo-rl-monitor)
# +--------------------------------------------------------------------------+ #
"""Resolve a dataset name to bytes on disk, pulling from R2 if needed.

Design in `docs/DATA_PLATFORM_PLAN.md` §6b. The monitor Worker is a catalog
and an authorizer only: it returns metadata plus a presigned S3 URL, and the
transfer happens directly against R2. Nothing multi-GB passes through the
Worker.

Credentials come from `.env` exactly as `core/logger.mojo`'s `RemoteLogger`
does — `RL_MONITOR_URL` + `RL_MONITOR_API_KEY`, sent as
`Authorization: Bearer <key>`. Same Worker, same key, same convention.

⚠ The catalog routes live at `<base>/datasets`, NOT `<base>/api/datasets`.
`/api/*` on that Worker is guarded by the dashboard's browser-session
middleware, which answers `{"error":"Unauthorized"}` 401 to an API key;
`/runs` and `/ingest` are top-level for the same reason.

**`pull` is a cache, not a toll booth.** If the destination already exists and
its sha256 matches the catalog, no bytes move. That is why the catalog carries
the hash, and why it also carries the generating recipe (`seed` +
`source_commit`): a state-only dataset can be regenerated (walker's 10 M
transitions are 992 MiB but ~2 min of CPU) rather than fetched.
"""

from std.python import Python, PythonObject

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.io.fetch import fetch_to_cache, sha256_file, upload_file


comptime ENV_URL_KEY = "RL_MONITOR_URL"
comptime ENV_API_KEY = "RL_MONITOR_API_KEY"


struct RemoteCatalog(Movable & Deinitable):
    """Client for the monitor's `/datasets` routes."""

    var base_url: String
    var api_key: String

    def __init__(out self, var base_url: String, var api_key: String):
        self.base_url = String(base_url.removesuffix("/"))
        self.api_key = api_key^

    def __init__(out self, *, deinit move: Self):
        self.base_url = move.base_url^
        self.api_key = move.api_key^

    @staticmethod
    def from_env(path: String = String(".env")) raises -> Self:
        """Load `RL_MONITOR_URL` / `RL_MONITOR_API_KEY` from a dotenv file.

        Raises with the missing key named rather than failing later as a 401 —
        an empty key produces a bare `Authorization: Bearer ` header, which is
        indistinguishable at the server from a wrong key.
        """
        var env = load_dotenv(path)
        if ENV_URL_KEY not in env:
            raise Error(
                "RemoteCatalog.from_env: " + String(ENV_URL_KEY)
                + " missing from " + path
            )
        if ENV_API_KEY not in env:
            raise Error(
                "RemoteCatalog.from_env: " + String(ENV_API_KEY)
                + " missing from " + path
            )
        var url = env[ENV_URL_KEY]
        var key = env[ENV_API_KEY]
        if key.byte_length() == 0:
            raise Error(
                "RemoteCatalog.from_env: " + String(ENV_API_KEY)
                + " is empty in " + path
            )
        return Self(url^, key^)

    def _request(
        self,
        method: String,
        path: String,
        body: PythonObject,
        expect: Int,
    ) raises -> PythonObject:
        """One HTTP call, returning parsed JSON.

        Errors carry the STATUS and the SERVER'S BODY. That matters here: the
        monitor answers `{"error":"Unauthorized"}` from the dashboard session
        middleware, `{"error":"Missing API key"}` when no Bearer header
        arrived, and `{"error":"Invalid API key"}` when the key did not
        verify. Those three point at three different faults, so swallowing the
        body would throw away the diagnosis.
        """
        var urllib = Python.import_module("urllib.request")
        var urlerr = Python.import_module("urllib.error")
        var json_mod = Python.import_module("json")
        var url = self.base_url + path

        var req: PythonObject
        if Bool(body is not None):
            var data = json_mod.dumps(body).encode("utf-8")
            req = urllib.Request(PythonObject(url), data=data, method=PythonObject(method))
            req.add_header("Content-Type", "application/json")
        else:
            req = urllib.Request(PythonObject(url), method=PythonObject(method))
        req.add_header("User-Agent", "mojo-rl/1.0")
        req.add_header("Authorization", "Bearer " + self.api_key)

        try:
            var resp = urllib.urlopen(req, timeout=30)
            var status = Int(py=resp.status)
            var text = resp.read().decode("utf-8")
            _ = resp.close()
            if status != expect:
                raise Error(
                    method + " " + url + " -> " + String(status)
                    + " (expected " + String(expect) + "): " + String(text)
                )
            var body_text = String(text)
            if body_text.byte_length() == 0:
                return Python.none()
            return json_mod.loads(text)
        except e:
            raise Error(method + " " + url + " failed: " + String(e))

    # ── read ──────────────────────────────────────────────────────────

    def list_datasets(self) raises -> PythonObject:
        return self._request(String("GET"), String("/datasets"), Python.none(), 200)

    def describe(self, id: String) raises -> PythonObject:
        """Catalog row + a presigned download URL. Raises 409 if the upload
        never completed — a `pending` row is not served, so a crashed upload
        cannot advertise a truncated object."""
        return self._request(
            String("GET"), String("/datasets/") + id, Python.none(), 200
        )

    def pull(
        self, id: String, dest: String, label: String = String("")
    ) raises -> String:
        """Ensure `dest` holds the dataset. Returns `dest`.

        No bytes move if `dest` already matches the catalog's sha256.
        """
        var meta = self.describe(id)
        var url = String(meta["download_url"])
        var sha = String("")
        if Bool(meta["sha256"] is not None):
            sha = String(meta["sha256"])
        var size = 0
        if Bool(meta["sizeBytes"] is not None):
            size = Int(py=meta["sizeBytes"])
        var lbl = label if label.byte_length() > 0 else id
        return fetch_to_cache(url, dest, sha, size, lbl)

    # ── write ─────────────────────────────────────────────────────────

    def push(
        self,
        path: String,
        name: String,
        version: String = String("v1"),
        env_id: String = String(""),
        n_rows: Int = 0,
        n_episodes: Int = 0,
        seed: Int = 0,
        source_commit: String = String(""),
        columns: String = String(""),
    ) raises -> String:
        """Register, upload, then confirm. Returns the dataset id.

        The three steps are separate on purpose: the row stays `pending`
        between register and complete, so an interrupted upload leaves a row
        that `describe` refuses to serve rather than one advertising a
        truncated object.
        """
        var body = Python.dict()
        body["name"] = PythonObject(name)
        body["version"] = PythonObject(version)
        if env_id.byte_length() > 0:
            body["env_id"] = PythonObject(env_id)
        if n_rows > 0:
            body["n_rows"] = PythonObject(n_rows)
        if n_episodes > 0:
            body["n_episodes"] = PythonObject(n_episodes)
        if seed != 0:
            body["seed"] = PythonObject(seed)
        if source_commit.byte_length() > 0:
            body["source_commit"] = PythonObject(source_commit)
        if columns.byte_length() > 0:
            body["columns"] = PythonObject(columns)

        var reg = self._request(String("POST"), String("/datasets"), body, 201)
        var id = String(reg["id"])
        var upload_url = String(reg["upload_url"])

        _ = upload_file(upload_url, path, id)

        var os = Python.import_module("os")
        var size = Int(py=os.path.getsize(PythonObject(path)))
        var sha = sha256_file(path)

        var done = Python.dict()
        done["size_bytes"] = PythonObject(size)
        done["sha256"] = PythonObject(sha)
        _ = self._request(
            String("POST"), String("/datasets/") + id + "/complete", done, 200
        )
        return id
