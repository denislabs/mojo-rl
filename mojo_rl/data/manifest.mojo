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


# ═══════════════════════════════════════════════════════════════════════════
# the task table — `task=<index>\t"<text>"`
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠⚠ THE TEXT IS STORED BYTE-EXACT, AND THAT IS A HARD REQUIREMENT, NOT A
# PREFERENCE. A consumer tokenises the instruction, and the tokenisation is
# sensitive to the exact bytes: lerobot's `NewLineTaskProcessorStep` appends
# "\n" BEFORE tokenising, so "Grab the green cube" becomes six ids ending in
# 198, not five. A store that normalised the text would let an equality gate
# either fail on a cosmetic difference or — far worse — PASS while the ids
# were generated from different bytes than the store records.
#
# ⚠ WHICH COLLIDES WITH THIS FORMAT, so the text is QUOTED AND ESCAPED.
# `parse_manifest` calls `.strip()` on every value, and a line break would
# split the record outright. Quoting puts a non-whitespace character at each
# edge so `strip` cannot reach the text, and the five escapes below cover the
# rest. Everything else — spaces included — is literal, so the common case
# stays readable:
#
#     task=0	"Grab the green cube"
#
# ⚠ THE ROUND TRIP IS GATED ON ADVERSARIAL STRINGS, not on that one. An
# escaper is exactly the kind of transformation whose bug is the silent
# failure it was added to prevent — see `tests/data/test_manifest_tasks.mojo`.


struct TaskEntry(Copyable, ImplicitlyCopyable, Movable):
    """One row of the dataset's task table: `task_index` and its instruction."""

    var index: Int
    var text: String

    def __init__(out self, index: Int, text: String):
        self.index = index
        self.text = text


comptime MANIFEST_DATASET = "__manifest__"
comptime SCHEMA_VERSION = 1


def escape_task_text(s: String) -> String:
    """Byte-exact, quoted. The inverse is `unescape_task_text`.

    ⚠⚠ BUILT AS BYTES AND CONVERTED ONCE, NOT WITH `chr` PER BYTE.
    `out += chr(Int(byte))` looks equivalent and is not: for any byte above
    127 it yields the CODEPOINT of that value, which re-encodes as two bytes.
    "ünïcøde ✓ 日本語" came back as "Ã¼nÃ¯cÃ¸de â..." — mojibake, failing the
    round trip by exactly the mechanism this encoding exists to prevent.
    Caught by the hostile-string gate on its first run.

    ⚠ AND IT IS NOT ONLY HERE. `manifest._split` and `core/kv.split_on` both
    build strings that way, so a NON-ASCII value anywhere in a manifest is
    corrupted on read TODAY. Out of scope for this change — the task text is
    what had to be byte-exact — but it is a live defect in a shared reader.
    """
    var o = List[UInt8]()
    var q = UInt8(ord('"'))
    var bs = UInt8(ord("\\"))
    o.append(q)
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = b[i]
        if c == bs:
            o.append(bs)
            o.append(bs)
        elif c == q:
            o.append(bs)
            o.append(q)
        elif c == 10:
            o.append(bs)
            o.append(UInt8(ord("n")))
        elif c == 13:
            o.append(bs)
            o.append(UInt8(ord("r")))
        elif c == 9:
            o.append(bs)
            o.append(UInt8(ord("t")))
        else:
            o.append(c)
    o.append(q)
    o.append(0)
    return String(unsafe_from_utf8_ptr=o.unsafe_ptr())


def unescape_task_text(s: String) raises -> String:
    """The inverse. RAISES on anything `escape_task_text` did not write.

    ⚠ AN UNKNOWN ESCAPE IS AN ERROR, not a passthrough. Silently dropping a
    backslash returns text differing from what was stored by exactly one byte
    — the failure this encoding exists to prevent.
    """
    var b = s.as_bytes()
    var q = UInt8(ord('"'))
    var bs = UInt8(ord("\\"))
    if len(b) < 2 or b[0] != q or b[len(b) - 1] != q:
        raise Error("data: task text is not quoted: " + s)
    var o = List[UInt8]()
    var i = 1
    var end = len(b) - 1
    while i < end:
        var c = b[i]
        if c != bs:
            o.append(c)
            i += 1
            continue
        if i + 1 >= end:
            raise Error("data: task text ends in a dangling backslash: " + s)
        var n = b[i + 1]
        if n == bs:
            o.append(bs)
        elif n == q:
            o.append(q)
        elif n == UInt8(ord("n")):
            o.append(10)
        elif n == UInt8(ord("r")):
            o.append(13)
        elif n == UInt8(ord("t")):
            o.append(9)
        else:
            raise Error("data: unknown escape in task text: " + s)
        i += 2
    o.append(0)
    return String(unsafe_from_utf8_ptr=o.unsafe_ptr())


struct Manifest(Movable & Deinitable):
    var schema_version: Int
    var env_id: String
    var n_rows: Int
    var n_episodes: Int
    var seed: Int
    var source_commit: String
    var columns: List[ColumnSpec]
    var tasks: List[TaskEntry]
    """The dataset's task table, `task_index` -> instruction.

    ⚠ EMPTY MEANS "THIS STORE RECORDS NO TASKS", not "one unnamed task". A
    consumer doing multi-task work must treat an empty table as a store it
    cannot resolve indices against, and say so, rather than defaulting to 0."""

    def __init__(out self):
        self.schema_version = SCHEMA_VERSION
        self.env_id = String("")
        self.n_rows = 0
        self.n_episodes = 0
        self.seed = 0
        self.source_commit = String("")
        self.columns = List[ColumnSpec]()
        self.tasks = List[TaskEntry]()

    def __init__(out self, *, deinit move: Self):
        self.schema_version = move.schema_version
        self.env_id = move.env_id^
        self.n_rows = move.n_rows
        self.n_episodes = move.n_episodes
        self.seed = move.seed
        self.source_commit = move.source_commit^
        self.columns = move.columns^
        self.tasks = move.tasks^

    def task_text(self, index: Int) raises -> String:
        """The instruction for `task_index`, byte-exact as imported.

        ⚠ RAISES ON AN UNKNOWN INDEX rather than returning "". A consumer that
        got "" would tokenise an empty instruction and train against it.
        """
        for i in range(len(self.tasks)):
            if self.tasks[i].index == index:
                return self.tasks[i].text
        raise Error(
            "data: no task_index " + String(index) + " in this store's task"
            " table (" + String(len(self.tasks)) + " entries). An empty table"
            " means the store records no tasks at all."
        )

    def has_task(self, index: Int) -> Bool:
        for i in range(len(self.tasks)):
            if self.tasks[i].index == index:
                return True
        return False

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
        # ⚠ AFTER the columns and in INDEX ORDER as given — the encode/parse
        # round trip is a fixed point only if this order is stable.
        for i in range(len(self.tasks)):
            s += (
                "task=" + String(self.tasks[i].index) + "\t"
                + escape_task_text(self.tasks[i].text) + "\n"
            )
        return s^


def _split(s: String, sep: String) -> List[String]:
    """Byte-wise split on a single-byte separator. The manifest is ASCII by


    ⚠⚠ BYTES, NOT `chr` PER BYTE. `cur += chr(Int(bytes[i]))` is the obvious
    spelling and it CORRUPTS any value above 127: `chr` yields the CODEPOINT
    of that byte value, which re-encodes as two bytes, so "日本語" comes back
    as mojibake. This reader carried that bug for as long as it has existed —
    every non-ASCII value in every manifest read back wrong — and it surfaced
    only when a byte-exact task string was finally put through it.
    construction; mirrors `core/dotenv.mojo`'s `_split_lines`."""
    var out = List[String]()
    var cur = List[UInt8]()
    var bytes = s.as_bytes()
    var sb = sep.as_bytes()[0]
    for i in range(len(bytes)):
        if bytes[i] == sb:
            cur.append(0)
            out.append(String(unsafe_from_utf8_ptr=cur.unsafe_ptr()))
            cur = List[UInt8]()
        else:
            cur.append(bytes[i])
    cur.append(0)
    out.append(String(unsafe_from_utf8_ptr=cur.unsafe_ptr()))
    return out^


def _split_once(s: String, sep: String) -> List[String]:
    """Split on the FIRST occurrence of a single-byte separator. Byte-wise for
    the same reason as `_split` — and split-once matters because a value may


    ⚠⚠ BYTES, NOT `chr` PER BYTE. `cur += chr(Int(bytes[i]))` is the obvious
    spelling and it CORRUPTS any value above 127: `chr` yields the CODEPOINT
    of that byte value, which re-encodes as two bytes, so "日本語" comes back
    as mojibake. This reader carried that bug for as long as it has existed —
    every non-ASCII value in every manifest read back wrong — and it surfaced
    only when a byte-exact task string was finally put through it.
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
    var head = List[UInt8]()
    for i in range(cut):
        head.append(bytes[i])
    head.append(0)
    var tail = List[UInt8]()
    for i in range(cut + 1, len(bytes)):
        tail.append(bytes[i])
    tail.append(0)
    out.append(String(unsafe_from_utf8_ptr=head.unsafe_ptr()))
    out.append(String(unsafe_from_utf8_ptr=tail.unsafe_ptr()))
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
        elif key == "task":
            m.tasks.append(parse_task_entry(val))
        # Unknown keys are IGNORED on purpose: a store written by a newer
        # build with the same schema_version must stay readable.

    if not saw_version:
        raise Error("data: manifest has no schema_version line")
    return m^


def parse_task_entry(spec: String) raises -> TaskEntry:
    """`<index>\t"<escaped text>"`.

    ⚠ SPLIT ON THE FIRST TAB ONLY. The text may contain anything the escaper
    passes through, and a greedy split would truncate an instruction at its
    first tab — which the escaper turns into `\\t`, so it cannot happen today,
    and would the moment somebody "simplified" the escaper.
    """
    var cut = -1
    var b = spec.as_bytes()
    for i in range(len(b)):
        if b[i] == 9:
            cut = i
            break
    if cut < 0:
        raise Error("data: malformed task entry (no tab): " + spec)
    var idx = String(String(spec[byte=0:cut]).strip())
    var txt = String(spec[byte = cut + 1 : spec.byte_length()])
    return TaskEntry(Int(idx), unescape_task_text(txt))


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
