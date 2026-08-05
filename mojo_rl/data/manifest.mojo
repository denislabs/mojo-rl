# +--------------------------------------------------------------------------+ #
# | Manifest — what a store IS, carried inside the store
# +--------------------------------------------------------------------------+ #
"""The store's self-description: schema version, provenance, column set.

**Format**: a UTF-8 text block of `key=value` lines, stored INSIDE the `.h5`
as a `uint8` dataset named `__manifest__`.

Why a byte dataset rather than HDF5 attributes — the idiomatic place — is a
deliberate trade. Attributes need the H5A API plus variable-length string
types, a chunk of FFI we do not otherwise need. A `uint8` dataset needs
nothing beyond what the Stage-0 writer already has, keeps the store to ONE
file (no sidecar to desync), and stays readable from any tool:

    bytes(f['__manifest__'][:]).decode()          # h5py

If we later want real attributes, `schema_version` is the hinge that lets a
reader accept both.

Why key=value and not JSON: the manifest is a flat record with one repeating
field, Mojo has no JSON parser in std, and a hand-rolled one is a bug farm for
no gain. This mirrors `core/dotenv.mojo` and the checkpoint-v2 text format.

**Provenance is not decoration.** `seed` and `source_commit` let a training
box either fetch the dataset or REGENERATE it — walker's 10 M transitions are
992 MiB but only ~2 minutes of CPU. That makes remote storage a cache rather
than a single point of failure. `sha256` lets a box skip a download it already
has.

Example:

    schema_version=1
    env_id=dm_control/walker-walk
    n_rows=10000000
    n_episodes=10000
    seed=12345
    source_commit=081b53c0
    column=qpos:float32:9
    column=action:float32:6
    column=reward:float32
    column=pixels:uint8:84,84,3
"""

from .column import ColumnSpec, dtype_from_name, dtype_name


comptime MANIFEST_DATASET = "__manifest__"
comptime SCHEMA_VERSION = 1


struct Manifest(Movable & ImplicitlyDeletable):
    var schema_version: Int
    var env_id: String
    var n_rows: Int
    var n_episodes: Int
    var seed: Int
    var source_commit: String
    var columns: List[ColumnSpec]

    def __init__(out self):
        self.schema_version = SCHEMA_VERSION
        self.env_id = String("")
        self.n_rows = 0
        self.n_episodes = 0
        self.seed = 0
        self.source_commit = String("")
        self.columns = List[ColumnSpec]()

    def __init__(out self, *, deinit move: Self):
        self.schema_version = move.schema_version
        self.env_id = move.env_id^
        self.n_rows = move.n_rows
        self.n_episodes = move.n_episodes
        self.seed = move.seed
        self.source_commit = move.source_commit^
        self.columns = move.columns^

    def column(self, name: String) raises -> ColumnSpec:
        for i in range(len(self.columns)):
            if self.columns[i].name == name:
                return ColumnSpec(copy=self.columns[i])
        raise Error("data: no such column in manifest: " + name)

    def has_column(self, name: String) -> Bool:
        for i in range(len(self.columns)):
            if self.columns[i].name == name:
                return True
        return False

    def encode(self) raises -> String:
        var s = String()
        s += "schema_version=" + String(self.schema_version) + "\n"
        s += "env_id=" + self.env_id + "\n"
        s += "n_rows=" + String(self.n_rows) + "\n"
        s += "n_episodes=" + String(self.n_episodes) + "\n"
        s += "seed=" + String(self.seed) + "\n"
        s += "source_commit=" + self.source_commit + "\n"
        for i in range(len(self.columns)):
            s += "column=" + self.columns[i].describe() + "\n"
        return s^


def _split(s: String, sep: String) -> List[String]:
    """Byte-wise split on a single-byte separator. The manifest is ASCII by
    construction; mirrors `core/dotenv.mojo`'s `_split_lines`."""
    var out = List[String]()
    var cur = String()
    var bytes = s.as_bytes()
    var sb = sep.as_bytes()[0]
    for i in range(len(bytes)):
        if bytes[i] == sb:
            out.append(cur^)
            cur = String()
        else:
            cur += chr(Int(bytes[i]))
    out.append(cur^)
    return out^


def _split_once(s: String, sep: String) -> List[String]:
    """Split on the FIRST occurrence of a single-byte separator. Byte-wise for
    the same reason as `_split` — and split-once matters because a value may
    itself contain the separator (a column spec is `name:dtype:shape`)."""
    var out = List[String]()
    var bytes = s.as_bytes()
    var sb = sep.as_bytes()[0]
    var cut = -1
    for i in range(len(bytes)):
        if bytes[i] == sb:
            cut = i
            break
    if cut < 0:
        return out^
    var head = String()
    for i in range(cut):
        head += chr(Int(bytes[i]))
    var tail = String()
    for i in range(cut + 1, len(bytes)):
        tail += chr(Int(bytes[i]))
    out.append(head^)
    out.append(tail^)
    return out^


def parse_manifest(text: String) raises -> Manifest:
    var m = Manifest()
    var saw_version = False
    var lines = _split(text, String("\n"))

    for li in range(len(lines)):
        var line = String(lines[li].strip())
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var kv = _split_once(line, String("="))
        if len(kv) != 2:
            raise Error("data: malformed manifest line: " + line)
        var key = String(kv[0].strip())
        var val = String(kv[1].strip())

        if key == "schema_version":
            m.schema_version = Int(val)
            saw_version = True
            if m.schema_version > SCHEMA_VERSION:
                raise Error(
                    "data: manifest schema_version " + val
                    + " is newer than this build supports ("
                    + String(SCHEMA_VERSION) + ")"
                )
        elif key == "env_id":
            m.env_id = val^
        elif key == "n_rows":
            m.n_rows = Int(val)
        elif key == "n_episodes":
            m.n_episodes = Int(val)
        elif key == "seed":
            m.seed = Int(val)
        elif key == "source_commit":
            m.source_commit = val^
        elif key == "column":
            m.columns.append(parse_column(val))
        # Unknown keys are IGNORED on purpose: a store written by a newer
        # build with the same schema_version must stay readable.

    if not saw_version:
        raise Error("data: manifest has no schema_version line")
    return m^


def parse_column(spec: String) raises -> ColumnSpec:
    """`name:dtype[:d0,d1,...]` → ColumnSpec."""
    var parts = _split(spec, String(":"))
    if len(parts) < 2 or len(parts) > 3:
        raise Error("data: malformed column spec: " + spec)
    var name = String(String(parts[0]).strip())
    var dt = dtype_from_name(String(String(parts[1]).strip()))
    var shape = List[Int]()
    if len(parts) == 3:
        var dims = _split(String(String(parts[2]).strip()), String(","))
        for i in range(len(dims)):
            var d = String(String(dims[i]).strip())
            if d.byte_length() == 0:
                continue
            shape.append(Int(d))
    return ColumnSpec(name^, dt, shape^)
