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

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.io.fetch import fetch_to_cache, sha256_file, upload_file
from mojo_rl.io.fileio import file_size
from mojo_rl.io.http import HttpClient
from mojo_rl.io.json import JsonDoc, JsonWriter, parse_json


comptime ENV_URL_KEY = "RL_MONITOR_URL"
comptime ENV_API_KEY = "RL_MONITOR_API_KEY"


struct DatasetMeta(Copyable):
    """One catalog row, plus the presigned URL `describe` hands back.

    Typed rather than a `JsonDoc` handle: every field here is read by name at
    a call site, and `meta["sizeBytes"]` returning silently-absent was exactly
    the failure the `PythonObject` version could not rule out.
    """

    var id: String
    var name: String
    var status: String
    var size_bytes: Int
    var sha256: String
    var download_url: String

    def __init__(out self):
        self.id = String("")
        self.name = String("")
        self.status = String("")
        self.size_bytes = 0
        self.sha256 = String("")
        self.download_url = String("")

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        self.name = copy.name
        self.status = copy.status
        self.size_bytes = copy.size_bytes
        self.sha256 = copy.sha256
        self.download_url = copy.download_url

    def __init__(out self, *, deinit move: Self):
        self.id = move.id^
        self.name = move.name^
        self.status = move.status^
        self.size_bytes = move.size_bytes
        self.sha256 = move.sha256^
        self.download_url = move.download_url^


def _opt_string(ref doc: JsonDoc, node: Int, name: String) raises -> String:
    """A string member, or "" when absent or null. `field` already returns -1
    for a missing key, so this is the one place that decides that an absent
    optional is not an error."""
    var n = doc.field(node, name)
    if n < 0 or doc.kind_of(n) != 3:  # J_STRING
        return String("")
    return doc.string(n)


def _opt_int(ref doc: JsonDoc, node: Int, name: String) raises -> Int:
    var n = doc.field(node, name)
    if n < 0 or doc.kind_of(n) != 2:  # J_NUMBER
        return 0
    return doc.integer(n)


struct RemoteCatalog(Movable & Deinitable):
    """Client for the monitor's `/datasets` routes."""

    var base_url: String
    var api_key: String
    var _http: HttpClient
    """One client for the catalog's lifetime — the register / upload /
    complete sequence in `push` is three calls to one host."""

    def __init__(out self, var base_url: String, var api_key: String) raises:
        self.base_url = String(base_url.removesuffix("/"))
        self.api_key = api_key^
        self._http = HttpClient(30000, 10000)
        self._http.bearer(self.api_key)

    def __init__(out self, *, deinit move: Self):
        self.base_url = move.base_url^
        self.api_key = move.api_key^
        self._http = move._http^

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
        mut self,
        method: String,
        path: String,
        body: String,
        expect: Int,
    ) raises -> JsonDoc:
        """One HTTP call, returning parsed JSON.

        Errors carry the STATUS and the SERVER'S BODY. That matters here: the
        monitor answers `{"error":"Unauthorized"}` from the dashboard session
        middleware, `{"error":"Missing API key"}` when no Bearer header
        arrived, and `{"error":"Invalid API key"}` when the key did not
        verify. Those three point at three different faults, so swallowing the
        body would throw away the diagnosis — which is why `expect` is passed
        down to `HttpClient.request`, whose raise embeds the body verbatim.

        An empty `body` means "no request body", matching the old
        `body is None`.
        """
        var url = self.base_url + path
        var payload = List[UInt8]()
        var ctype = String("")
        if body.byte_length() > 0:
            for i in range(body.byte_length()):
                payload.append(body.as_bytes()[i])
            ctype = String("application/json")
        var r = self._http.request(method, url, payload^, ctype, expect)
        # ⚠ Move the body out UNCONDITIONALLY. Moving it on only one branch
        # leaves the response partially destroyed on the other, which Mojo
        # rejects as "destroyed out of the middle of a value".
        var raw = r^.take_body()
        if len(raw) == 0:
            return JsonDoc()  # 204 / empty 200: an empty doc, not a parse error
        return parse_json(raw^)

    # ── read ──────────────────────────────────────────────────────────

    def list_datasets(mut self) raises -> JsonDoc:
        """The raw catalog listing.

        Left as a document rather than a typed list: the shape of this route
        is the monitor's to decide, and a struct here would have to be kept in
        step with it for a call nothing but a smoke test makes.
        """
        return self._request(String("GET"), String("/datasets"), String(""), 200)

    def describe(mut self, id: String) raises -> DatasetMeta:
        """Catalog row + a presigned download URL. Raises 409 if the upload
        never completed — a `pending` row is not served, so a crashed upload
        cannot advertise a truncated object."""
        var doc = self._request(
            String("GET"), String("/datasets/") + id, String(""), 200
        )
        var root = doc.root()
        var m = DatasetMeta()
        m.id = _opt_string(doc, root, String("id"))
        m.name = _opt_string(doc, root, String("name"))
        m.status = _opt_string(doc, root, String("status"))
        m.size_bytes = _opt_int(doc, root, String("sizeBytes"))
        m.sha256 = _opt_string(doc, root, String("sha256"))
        m.download_url = _opt_string(doc, root, String("download_url"))
        if m.download_url.byte_length() == 0:
            raise Error(
                "describe(" + id + "): the catalog returned no download_url"
            )
        return m^

    def pull(
        mut self, id: String, dest: String, label: String = String("")
    ) raises -> String:
        """Ensure `dest` holds the dataset. Returns `dest`.

        No bytes move if `dest` already matches the catalog's sha256.
        """
        var meta = self.describe(id)
        var lbl = label if label.byte_length() > 0 else id
        return fetch_to_cache(
            meta.download_url, dest, meta.sha256, meta.size_bytes, lbl
        )

    # ── write ─────────────────────────────────────────────────────────

    def push(
        mut self,
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
        var w = JsonWriter()
        w.begin_object()
        w.member(String("name"), name)
        w.member(String("version"), version)
        if env_id.byte_length() > 0:
            w.member(String("env_id"), env_id)
        if n_rows > 0:
            w.member(String("n_rows"), n_rows)
        if n_episodes > 0:
            w.member(String("n_episodes"), n_episodes)
        if seed != 0:
            w.member(String("seed"), seed)
        if source_commit.byte_length() > 0:
            w.member(String("source_commit"), source_commit)
        if columns.byte_length() > 0:
            w.member(String("columns"), columns)
        w.end_object()

        var reg = self._request(
            String("POST"), String("/datasets"), w.done(), 201
        )
        var root = reg.root()
        var id = _opt_string(reg, root, String("id"))
        var upload_url = _opt_string(reg, root, String("upload_url"))
        if id.byte_length() == 0 or upload_url.byte_length() == 0:
            raise Error(
                "push: the registration answered without an id or an"
                " upload_url"
            )

        _ = upload_file(upload_url, path, id)

        var size = file_size(path)
        var sha = sha256_file(path)

        var done = JsonWriter()
        done.begin_object()
        done.member(String("size_bytes"), size)
        done.member(String("sha256"), sha)
        done.end_object()
        _ = self._request(
            String("POST"),
            String("/datasets/") + id + "/complete",
            done.done(),
            200,
        )
        return id^
