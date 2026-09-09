"""A run as a durable, addressable object on disk.

Run directory, `run.kv`, and every path a driver writes to. This is the
primitive one level below the project layer (`docs/PROJECT_LAYER_PLAN.md` §6),
and it exists because a run had two identities that never met: a `run_id` the
dashboard minted, and a **compile-time constant** checkpoint path on disk.

    comptime DEFAULT_CKPT = "act_so101_best_gpu.ckpt"
    var best_ckpt = String("/tmp/act_so101_best_gpu.ckpt")

⚠⚠ THE CONSTANT IS THE BUG, AND `RunContext` IS THE ONLY THING THAT REMOVES IT.
Every run of a driver overwrote the previous run's checkpoint by construction,
and nothing on disk recorded whether the run that wrote it was any good. This
struct hands out every path, so the checkpoint, the metrics CSV and the
dashboard name all derive from ONE identifier and cannot drift apart.

⚠ `run.kv` IS WRITTEN AT t=0, NOT AT THE END. A wrapper script cannot write
`status=crashed` for a process the kernel killed, and a run that only registers
itself on success leaves exactly the orphan checkpoint this exists to abolish.
It is rewritten once more at `close()` with `finished` and the terminal status.

⚠ `status` AND `outcome` ARE WRITTEN, NEVER INFERRED. A `run.kv` that still
says `running` with a `started` an hour old IS a crashed run — recoverable
information no directory listing has ever carried here.

Usage:
    var run = RunContext(project="so101", driver="examples/so101/act_gpu.mojo",
                         slug="act-reach", env="family:so101_tabletop",
                         task="so101_reach_brick", seed=12345)
    var logger = CompositeLogger(CsvLogger(run.metrics_path()), remote)
    register_run(run, logger)          # seeds the config, then POSTs /runs
    ...
    agent.save(run.checkpoint_path("best"))
    run.set_outcome("success_rate=" + String(sr))
    run.close()                        # status=done, finished=<now>
"""

from std.ffi import external_call
from std.time import perf_counter_ns

from mojo_rl.core.concurrent.thread import OpaquePtr, null_opaque

from mojo_rl.core.kv import KvWriter, kv_lines
from mojo_rl.core.logger import Logger
from mojo_rl.core.project import runs_root_for
from mojo_rl.io.fileio import file_size, write_text_atomic
from mojo_rl.io.proc import quote_arg, run_capture
from mojo_rl.io.sha256 import sha256_file, sha256_string


comptime SCHEMA_VERSION = 1
comptime DEFAULT_ROOT = "runs"
"""The flat root, used when the named project does not exist yet.

⚠⚠ `runs_root_for` (P1) IS WHAT DECIDES, AND IT FALLS BACK HERE ON PURPOSE. A
driver names its project; the project layer becomes active for it the moment
someone runs `project-init`, and until then the run lands in this flat root. No
driver has to know which world it is in, which is why the seven retrofitted in
P0d needed no second edit."""


# =============================================================================
# Wall clock — the tree has `perf_counter_ns` and nothing else
# =============================================================================


def epoch_seconds() -> Int:
    """Seconds since 1970-01-01T00:00:00Z.

    ⚠ `std.time` EXPORTS ONLY `perf_counter_ns` AND `sleep`, and `perf_counter_ns`
    is MONOTONIC — it answers "how long since some arbitrary boot moment", which
    is the wrong question for `started=`. This is `time(NULL)`.
    """
    return Int(external_call["time", Int64, OpaquePtr](null_opaque()))


def _pad(n: Int, width: Int) -> String:
    var s = String(n)
    var out = String("")
    for _ in range(width - s.byte_length()):
        out += "0"
    return out + s


def civil_from_days(days: Int) -> Tuple[Int, Int, Int]:
    """`(year, month, day)` for a count of days since the Unix epoch.

    Hinnant's `civil_from_days`, transcribed. ⚠ IT IS ERA-BASED RATHER THAN A
    LOOP OVER YEARS because the leap rule is only periodic over 400 years; a
    "365 or 366" loop gets 2100 wrong, and nothing in a training run would
    notice until a directory sorted strangely years from now.
    """
    var z = days + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    return (y + 1 if m <= 2 else y, m, d)


def iso8601_utc(epoch: Int) -> String:
    """`YYYY-MM-DDTHH:MM:SSZ`. UTC, because a run directory is compared across
    a laptop and a rented box in another timezone."""
    var days = epoch // 86400
    var rem = epoch - days * 86400
    var ymd = civil_from_days(days)
    return (
        _pad(ymd[0], 4) + "-" + _pad(ymd[1], 2) + "-" + _pad(ymd[2], 2)
        + "T" + _pad(rem // 3600, 2) + ":" + _pad((rem // 60) % 60, 2)
        + ":" + _pad(rem % 60, 2) + "Z"
    )


def date_utc(epoch: Int) -> String:
    """`YYYY-MM-DD` — the run id's first field."""
    var ymd = civil_from_days(epoch // 86400)
    return _pad(ymd[0], 4) + "-" + _pad(ymd[1], 2) + "-" + _pad(ymd[2], 2)


# =============================================================================
# The identifier
# =============================================================================


def slugify(s: String) -> String:
    """Lowercase, and every run of non-alphanumerics collapsed to one `-`.

    ⚠ THE SLUG GOES IN A URL PATH (`/runs/<id>/finish`) AND A DIRECTORY NAME, so
    it is restricted at the point it is MINTED rather than escaped at each use.
    `logger._finish_url` records the other half of that bargain.
    """
    var out = String("")
    var b = s.as_bytes()
    var dash = False
    for i in range(s.byte_length()):
        var c = Int(b[i])
        var ok = (
            (c >= 48 and c <= 57) or (c >= 97 and c <= 122)
        )
        if c >= 65 and c <= 90:
            c += 32
            ok = True
        if ok:
            out += chr(c)
            dash = False
        elif not dash and out.byte_length() > 0:
            out += "-"
            dash = True
    while out.endswith("-"):
        var cut = String(out[byte=0 : out.byte_length() - 1])
        out = cut^
    return out^


def derive_run_id(
    date: String, slug: String, host: String, pid: Int, start_ns: Int
) raises -> String:
    """`<date>_<slug>_<hash8>`, with `hash8 = sha256(host|pid|start_ns)[:8]`.

    ⚠⚠ THE HASH IS DERIVED, NOT RANDOM, AND THAT IS THE WHOLE POINT. Two runs on
    one box differ in `pid`/`start_ns`; two boxes in the same second differ in
    `host`. Being reproducible means a support question — *which box wrote
    this?* — has an answer, which a random suffix would have thrown away.

    ⚠ A COLLISION HERE IS A BUG IN THIS FUNCTION, not bad luck. The monitor
    answers 409 and the client appends a suffix, but that must be LOGGED
    LOUDLY: a silent retry would hide the derivation defect forever.
    """
    var h = sha256_string(host + "|" + String(pid) + "|" + String(start_ns))
    return date + "_" + slug + "_" + String(h[byte=0:8])


def hostname() -> String:
    """The box's name, or `unknown` if it cannot be read.

    ⚠ IT MUST NEVER RAISE. A hostname is provenance, not correctness, and a run
    that refused to start because `hostname` was missing would be this layer
    causing the outage it exists to explain.
    """
    try:
        var h = String(run_capture(String("hostname 2>/dev/null"), 256).strip())
        if h.byte_length() > 0:
            return slugify(h)
    except:
        pass
    return String("unknown")


def process_id() -> Int:
    return Int(external_call["getpid", Int32]())


def git_commit() -> String:
    """Short HEAD, or empty outside a work tree."""
    try:
        return String(
            run_capture(
                String("git rev-parse --short=8 HEAD 2>/dev/null"), 64
            ).strip()
        )
    except:
        return String("")


def git_dirty() -> Bool:
    """Whether tracked files differ from HEAD.

    ⚠ `-uno` AND A ONE-BYTE READ. Plain `--porcelain` lists every untracked
    file, which in this tree is thousands of lines through a 64 KB cap — and
    untracked files are not what `dirty=` means anyway: the question is whether
    `source_commit` describes the code that ran.
    """
    try:
        return (
            run_capture(
                String("git status --porcelain -uno 2>/dev/null | head -c 1"),
                16,
            ).byte_length()
            > 0
        )
    except:
        return False


def _mkdir_p(path: String) raises:
    _ = run_capture(String("mkdir -p ") + quote_arg(path) + " 2>&1", 4096)


def _basename_noext(path: String) -> String:
    var cut = -1
    var dot = -1
    var b = path.as_bytes()
    for i in range(path.byte_length()):
        if Int(b[i]) == 0x2F:
            cut = i
        elif Int(b[i]) == 0x2E:
            dot = i
    var start = cut + 1
    var end = dot if dot > start else path.byte_length()
    return String(path[byte=start:end])


# =============================================================================
# RunContext
# =============================================================================


struct RunContext(Movable):
    """One execution of a learning algorithm, as a directory and a record.

    ⚠ A BENCHMARK, A PROBE OR A DIAGNOSTIC IS NOT A RUN, even when it is long
    and runs on the same rented box. They produce a number for a document, not
    an artifact to promote, and pulling them in here is what would turn a
    ~70-driver retrofit into a 374-file one.
    """

    var id: String
    var dir: String
    var project: String
    var driver: String
    var slug: String
    var env: String
    var task: String
    var dataset: String
    var seed: Int
    var host: String
    var device: String
    var source_commit: String
    var dirty: Bool
    var started: Int
    var resumed_from: String
    var _status: String
    var _outcome: String
    var _tag: String
    var _finished: Int
    var _config_k: List[String]
    var _config_v: List[String]
    var _artifacts: List[String]
    var _closed: Bool

    def __init__(
        out self,
        project: String,
        driver: String,
        slug: String = String(""),
        env: String = String(""),
        task: String = String(""),
        dataset: String = String(""),
        seed: Int = 0,
        device: String = String(""),
        resumed_from: String = String(""),
        root: String = String(""),
    ) raises:
        """Mint the id, create the directory, and write `run.kv` at t=0.

        ⚠ THE DIRECTORY AND THE RECORD ARE CREATED HERE, BEFORE ANY TRAINING.
        A run that registers itself only on success is exactly the orphan
        checkpoint this layer exists to abolish.
        """
        self.started = epoch_seconds()
        self.project = project
        self.driver = driver
        self.slug = slug if slug.byte_length() > 0 else slugify(
            _basename_noext(driver)
        )
        self.env = env
        self.task = task
        self.dataset = dataset
        self.seed = seed
        self.device = device
        self.resumed_from = resumed_from
        self.host = hostname()
        self.source_commit = git_commit()
        self.dirty = git_dirty()
        self.id = derive_run_id(
            date_utc(self.started),
            self.slug,
            self.host,
            process_id(),
            perf_counter_ns(),
        )
        var base = root if root.byte_length() > 0 else runs_root_for(project)
        self.dir = base + "/" + self.id
        self._status = String("running")
        self._outcome = String("")
        self._tag = String("")
        self._finished = 0
        self._config_k = List[String]()
        self._config_v = List[String]()
        self._artifacts = List[String]()
        self._closed = False
        _mkdir_p(self.dir + "/checkpoints")
        _mkdir_p(self.dir + "/eval")
        self._flush()

    def __init__(out self, *, deinit move: Self):
        self.id = move.id^
        self.dir = move.dir^
        self.project = move.project^
        self.driver = move.driver^
        self.slug = move.slug^
        self.env = move.env^
        self.task = move.task^
        self.dataset = move.dataset^
        self.seed = move.seed
        self.host = move.host^
        self.device = move.device^
        self.source_commit = move.source_commit^
        self.dirty = move.dirty
        self.started = move.started
        self.resumed_from = move.resumed_from^
        self._status = move._status^
        self._outcome = move._outcome^
        self._tag = move._tag^
        self._finished = move._finished
        self._config_k = move._config_k^
        self._config_v = move._config_v^
        self._artifacts = move._artifacts^
        self._closed = move._closed

    # ── the paths. THIS IS THE MECHANISM. ────────────────────────────────

    def name(self) -> String:
        """The dashboard's `run_name`.

        ⚠ IT IS THE RUN ID, DELIBERATELY. Today every run of a driver posts the
        same `runName` — `"ACT SO-ARM101 (GPU)"` — which is pain 2 exactly: a
        list of identical rows. A unique name is strictly better than a
        repeated one; grouping comes back with project scoping in P2.
        """
        return self.id

    def kv_path(self) -> String:
        return self.dir + "/run.kv"

    def metrics_path(self) -> String:
        return self.dir + "/metrics.csv"

    def eval_dir(self) -> String:
        return self.dir + "/eval"

    def checkpoint_path(self, name: String) -> String:
        """`<run>/checkpoints/<name>.ckpt` — `best`, `last`, `step_400000`.

        ⚠ THIS REPLACES A `comptime` LITERAL, WHICH IS THE ENTIRE POINT. It is
        also what turns `fb_walker_all_d128.ckpt.100000 … .1200000` — 26 files
        of one run's history flattened into a shared namespace — into that
        run's own directory.
        """
        return self.dir + "/checkpoints/" + name + ".ckpt"

    # ── the record ────────────────────────────────────────────────────────

    def set_config(mut self, key: String, value: String) raises:
        for i in range(len(self._config_k)):
            if self._config_k[i] == key:
                self._config_v[i] = value
                self._flush()
                return
        self._config_k.append(key)
        self._config_v.append(value)
        self._flush()

    def set_outcome(mut self, outcome: String) raises:
        """The run's own best eval numbers, e.g. `success_rate=0.82 val_l1=0.031`."""
        self._outcome = outcome
        self._flush()

    def set_tag(mut self, tag: String) raises:
        """Free human text. ⚠ THE ONLY FIELD A HUMAN HAS TO WRITE, and the one
        that makes 218 checkpoints searchable:
        `grep -l 'tag=.*reach' runs/*/run.kv`."""
        self._tag = tag
        self._flush()

    def set_status(mut self, status: String) raises:
        """`running|done|crashed|killed`. Used by the interrupt path (P0e) to
        state a terminal status before `close()` reports the default."""
        self._status = status
        self._flush()

    def add_artifact(mut self, rel_path: String) raises:
        """Record a file this run produced, content-addressed.

        ⚠ IT HASHES THE FILE, SO IT IS NOT FREE — an ACT checkpoint is 215 MB.
        Call it once per artifact worth recording, never inside a validation
        loop that writes `best` every N steps. P3b's `ArtifactSink` is what
        moves the hashing off the training thread.
        """
        var full = self.dir + "/" + rel_path
        var digest = sha256_file(full)
        self._artifacts.append(
            rel_path + ":sha256:" + digest + ":" + String(file_size(full))
            + ":local"
        )
        self._flush()

    def close(mut self, status: String = String("")) raises:
        """Write `finished` and the terminal status. Idempotent.

        ⚠ REACHING HERE IS ITSELF THE END SIGNAL, exactly as in
        `RemoteLogger.close()`. A run the kernel killed never arrives, and a
        `run.kv` still saying `running` with an old `started` IS that run's
        record — which is why nothing here ever infers a status.
        """
        if self._closed:
            return
        self._closed = True
        if status.byte_length() > 0:
            self._status = status
        elif self._status == String("running"):
            self._status = String("done")
        self._finished = epoch_seconds()
        self._flush()

    def closed(self) -> Bool:
        return self._closed

    def status(self) -> String:
        return self._status

    # ── serialisation: ONE renderer, used at t=0 and at close ────────────

    def render(self) raises -> String:
        """`run.kv`'s text.

        ⚠ ONE RENDERER, CALLED FROM EVERY MUTATOR. The alternative — append the
        changed line — is how a file ends up with two `status=` lines and a
        reader that believes the first. `_a_rule_written_inline_twice_drifts`.
        """
        var w = KvWriter(String("run spec"))
        w.add(String("schema_version"), String(SCHEMA_VERSION))
        w.add(String("run_id"), self.id)
        w.add(String("project"), self.project)
        w.add(String("driver"), self.driver)
        w.add(String("env"), self.env)
        w.add(String("task"), self.task)
        w.add(String("dataset"), self.dataset)
        w.add(String("source_commit"), self.source_commit)
        w.add(String("dirty"), String(1) if self.dirty else String(0))
        w.add(String("seed"), String(self.seed))
        w.add(String("host"), self.host)
        w.add(String("device"), self.device)
        w.add(String("started"), iso8601_utc(self.started))
        w.add(
            String("finished"),
            iso8601_utc(self._finished) if self._finished > 0 else String(""),
        )
        w.add(String("status"), self._status)
        w.add(String("outcome"), self._outcome)
        for i in range(len(self._config_k)):
            w.add(String("config"), self._config_k[i] + ":" + self._config_v[i])
        for i in range(len(self._artifacts)):
            w.add(String("artifact"), self._artifacts[i])
        w.add(String("tag"), self._tag)
        w.add(String("resumed_from"), self.resumed_from)
        return w^.done()

    def _flush(mut self) raises:
        write_text_atomic(self.kv_path(), self.render())


# =============================================================================
# Wiring a logger to a run
# =============================================================================


def register_run[L: Logger](ref run: RunContext, mut logger: L) raises:
    """Seed the dashboard's config from the run, then announce it.

    ⚠⚠ THIS IS WHERE `register()` IS CALLED, NOT IN `RunContext.__init__`. Two
    reasons, and the second is the load-bearing one: `/runs` carries the config,
    which is assembled here; and making `RunContext` generic over `Logger` to
    hold one would put a type parameter on every driver signature and every
    struct that stores a run.
    """
    logger.set_config(String("run_id"), run.id)
    logger.set_config(String("project"), run.project)
    logger.set_config(String("driver"), run.driver)
    if run.env.byte_length() > 0:
        logger.set_config(String("env"), run.env)
    if run.task.byte_length() > 0:
        logger.set_config(String("task"), run.task)
    if run.source_commit.byte_length() > 0:
        logger.set_config(String("source_commit"), run.source_commit)
    logger.set_config(String("seed"), String(run.seed))
    logger.set_config(String("host"), run.host)
    logger.register()


# =============================================================================
# Reading a run back
# =============================================================================


comptime _KNOWN_KEYS = (
    "schema_version run_id project driver env task dataset source_commit"
    " dirty seed host device started finished status outcome config artifact"
    " tag resumed_from"
)


struct RunRecord(Movable):
    """A `run.kv` as read back. The retrieval half of `RunContext`."""

    var run_id: String
    var project: String
    var driver: String
    var env: String
    var task: String
    var dataset: String
    var source_commit: String
    var dirty: Bool
    var seed: Int
    var host: String
    var device: String
    var started: String
    var finished: String
    var status: String
    var outcome: String
    var tag: String
    var resumed_from: String
    var config: List[String]
    var artifacts: List[String]

    def __init__(out self):
        self.run_id = String("")
        self.project = String("")
        self.driver = String("")
        self.env = String("")
        self.task = String("")
        self.dataset = String("")
        self.source_commit = String("")
        self.dirty = False
        self.seed = 0
        self.host = String("")
        self.device = String("")
        self.started = String("")
        self.finished = String("")
        self.status = String("")
        self.outcome = String("")
        self.tag = String("")
        self.resumed_from = String("")
        self.config = List[String]()
        self.artifacts = List[String]()

    def is_stale(self) -> Bool:
        """`status=running` with a `finished` that was never written.

        ⚠⚠ THIS IS RECOVERABLE INFORMATION NO DIRECTORY LISTING HAS EVER
        CARRIED HERE. A run whose record still says `running` long after
        `started` **is** a crashed run — the process died before `close()`, so
        nothing local could have said otherwise.
        """
        return self.status == String("running")


def parse_run(text: String, what: String) raises -> RunRecord:
    """Parse `run.kv`.

    ⚠⚠ AN UNKNOWN KEY RAISES. This follows `tasks/spec.mojo`, not
    `data/manifest.mojo`: a manifest ignores unknown keys so a store written by
    a newer build stays readable, but a typo'd key here is a LIE about what a
    run was, and a dropped `status=` is exactly the failure this layer exists
    to prevent.
    """
    var r = RunRecord()
    var ls = kv_lines(text, what)
    for i in range(len(ls)):
        var k = ls[i].key
        var v = ls[i].value
        if k == "run_id":
            r.run_id = v
        elif k == "project":
            r.project = v
        elif k == "driver":
            r.driver = v
        elif k == "env":
            r.env = v
        elif k == "task":
            r.task = v
        elif k == "dataset":
            r.dataset = v
        elif k == "source_commit":
            r.source_commit = v
        elif k == "dirty":
            r.dirty = v == "1"
        elif k == "seed":
            r.seed = atol(v) if v.byte_length() > 0 else 0
        elif k == "host":
            r.host = v
        elif k == "device":
            r.device = v
        elif k == "started":
            r.started = v
        elif k == "finished":
            r.finished = v
        elif k == "status":
            r.status = v
        elif k == "outcome":
            r.outcome = v
        elif k == "tag":
            r.tag = v
        elif k == "resumed_from":
            r.resumed_from = v
        elif k == "config":
            r.config.append(v)
        elif k == "artifact":
            r.artifacts.append(v)
        elif k == "schema_version":
            if v != String(SCHEMA_VERSION):
                raise Error(
                    what + ": schema_version " + v + ", this build writes "
                    + String(SCHEMA_VERSION)
                )
        else:
            raise Error(
                what + ": unknown key '" + k + "' on line "
                + String(ls[i].lineno) + ". Known keys are: "
                + String(_KNOWN_KEYS) + ". A run record refuses unknown keys"
                " because a dropped status= is the failure this layer exists"
                " to prevent."
            )
    return r^


def load_run(path: String) raises -> RunRecord:
    with open(path, "r") as fh:
        return parse_run(fh.read(), path)
