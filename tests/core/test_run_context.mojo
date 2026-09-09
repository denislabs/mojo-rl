"""`RunContext` — the id, the clock, the record, and the status transitions.

Run: pixi run mojo run -I . tests/core/test_run_context.mojo

The pain this closes, stated as it is: a run had two identities that never met.
The dashboard minted a `run_id`; on disk the same run was a **compile-time
constant** —

    comptime DEFAULT_CKPT = "act_so101_best_gpu.ckpt"

— so every run of a driver overwrote the previous one's checkpoint by
construction, and nothing on disk recorded whether the run that wrote it was
any good. `checkpoints/` holds 219 flat entries as a result.

⚠ THE FILESYSTEM HALF WRITES UNDER A PER-RUN TEMP ROOT and never touches the
repo's own `runs/`. A gate that wrote where the tool writes would be one
`--root` typo away from deleting a real run.
"""

from std.time import perf_counter_ns

from mojo_rl.core.kv import kv_lines
from mojo_rl.core.logger import CompositeLogger, CsvLogger, RemoteLogger
from mojo_rl.core.run import (
    RunContext,
    civil_from_days,
    date_utc,
    derive_run_id,
    iso8601_utc,
    load_run,
    parse_run,
    register_run,
    slugify,
)
from mojo_rl.io.proc import run_capture


def _root() -> String:
    return String("/tmp/mojo_rl_run_gate_") + String(perf_counter_ns())


# =============================================================================
# The clock — pure, and gated by value
# =============================================================================


def test_iso8601_against_known_epochs() raises:
    """⚠ THE ERA-BASED CONVERSION EXISTS BECAUSE THE LEAP RULE IS ONLY PERIODIC
    OVER 400 YEARS. 2000 is a leap year and 2100 is not; a "365 or 366" loop
    gets the second one wrong, and nothing in a training run would notice."""
    var cases = [
        (0, String("1970-01-01T00:00:00Z")),
        (951782400, String("2000-02-29T00:00:00Z")),   # leap, /400
        (4107542400, String("2100-03-01T00:00:00Z")),  # NOT leap, /100
        (1000000000, String("2001-09-09T01:46:40Z")),
        (1767225600, String("2026-01-01T00:00:00Z")),
        (1788000000, String("2026-08-29T10:40:00Z")),
    ]
    var compared = 0
    var differing = 0
    for c in cases:
        compared += 1
        var got = iso8601_utc(c[0])
        if got != c[1]:
            differing += 1
            print("    epoch", c[0], "->", got, "want", c[1])
    print("  clock:", compared, "epochs compared,", differing, "differing")
    if compared != 6 or differing != 0:
        raise Error("iso8601_utc wrong on " + String(differing))
    if date_utc(1788000000) != String("2026-08-29"):
        raise Error("date_utc: " + date_utc(1788000000))


def test_civil_from_days_edges() raises:
    var d0 = civil_from_days(0)
    if d0[0] != 1970 or d0[1] != 1 or d0[2] != 1:
        raise Error("day 0 is not 1970-01-01")
    var leap = civil_from_days(11016)  # 2000-02-29
    if leap[0] != 2000 or leap[1] != 2 or leap[2] != 29:
        raise Error("2000-02-29 missed")
    print("  civil: day 0 and the 2000 leap day both exact")


# =============================================================================
# The identifier — deterministic, and derived from things that differ
# =============================================================================


def test_run_id_is_deterministic_and_separating() raises:
    """⚠⚠ DERIVED, NOT RANDOM. Two runs on one box differ in `pid`/`start_ns`;
    two boxes in the same second differ in `host`. Reproducibility is what lets
    *which box wrote this?* have an answer at all — and it is why a collision
    would be a BUG in the derivation rather than bad luck."""
    var d = String("2026-09-02")
    var s = String("act-reach")
    var a = derive_run_id(d, s, String("vast-5090-de81"), 4242, 1000)
    var again = derive_run_id(d, s, String("vast-5090-de81"), 4242, 1000)
    if a != again:
        raise Error("same inputs gave two ids: " + a + " / " + again)
    if not a.startswith("2026-09-02_act-reach_") or a.byte_length() != 29:
        raise Error("id shape wrong: " + a + " (" + String(a.byte_length()) + ")")

    var others = [
        derive_run_id(d, s, String("laptop"), 4242, 1000),        # host
        derive_run_id(d, s, String("vast-5090-de81"), 4243, 1000),  # pid
        derive_run_id(d, s, String("vast-5090-de81"), 4242, 1001),  # start_ns
    ]
    var collisions = 0
    for o in others:
        if o == a:
            collisions += 1
            print("    collided:", o)
    print("  id: 1 repeat identical, 3 one-field changes,", collisions, "collisions")
    if collisions != 0:
        raise Error("the id does not separate on " + String(collisions) + " axes")


def test_slugify() raises:
    var cases = [
        (String("act_so101_train_gpu"), String("act-so101-train-gpu")),
        (String("SAC Task GPU"), String("sac-task-gpu")),
        (String("  --weird--  "), String("weird")),
        (String("vast.ai-5090"), String("vast-ai-5090")),
    ]
    var differing = 0
    for c in cases:
        if slugify(c[0]) != c[1]:
            differing += 1
            print("    ", c[0], "->", slugify(c[0]), "want", c[1])
    print("  slugify:", len(cases), "compared,", differing, "differing")
    if differing != 0:
        raise Error("slugify wrong on " + String(differing))


# =============================================================================
# The directory and the record
# =============================================================================


def test_run_kv_exists_at_t0_and_says_running() raises:
    """⚠⚠ THE RECORD IS WRITTEN BEFORE ANY TRAINING. A run that registers itself
    only on success leaves exactly the orphan checkpoint this layer abolishes —
    so this asserts the file is on disk with `status=running` while the run is
    still, as far as anything knows, about to start."""
    var root = _root()
    var run = RunContext(
        project=String("so101"),
        driver=String("examples/so101/act_so101_train_gpu.mojo"),
        env=String("family:so101_tabletop"),
        task=String("so101_reach_brick"),
        seed=12345,
        root=root,
    )
    var rec = load_run(run.kv_path())
    if rec.status != String("running"):
        raise Error("status at t=0 is '" + rec.status + "'")
    if not rec.is_stale():
        raise Error("a running run is not reported stale")
    if rec.run_id != run.id or rec.project != String("so101"):
        raise Error("identity did not survive: " + rec.run_id)
    if rec.seed != 12345 or rec.task != String("so101_reach_brick"):
        raise Error("fields did not survive")
    if rec.finished.byte_length() != 0:
        raise Error("finished was written before close(): " + rec.finished)
    # ⚠ The slug defaults from the driver's basename — this is the path that
    # kills `comptime DEFAULT_CKPT`, so it is asserted rather than assumed.
    if run.slug != String("act-so101-train-gpu"):
        raise Error("slug: " + run.slug)
    var ck = run.checkpoint_path(String("best"))
    if ck != run.dir + "/checkpoints/best.ckpt":
        raise Error("checkpoint_path: " + ck)
    print("  t=0:", rec.status, "/ seed", rec.seed, "/ id", rec.run_id)
    _ = run^


def test_status_transitions_and_close_idempotence() raises:
    var root = _root()
    var run = RunContext(
        project=String("p"), driver=String("d.mojo"), root=root
    )
    run.set_outcome(String("success_rate=0.82 val_l1=0.031"))
    run.set_tag(String("meilleur reach à ce jour, testé 8/10"))
    run.set_config(String("lr"), String("3e-4"))
    run.set_config(String("batch"), String("64"))
    run.close()
    var done = load_run(run.kv_path())
    if done.status != String("done") or done.finished.byte_length() == 0:
        raise Error("close(): status=" + done.status + " finished=" + done.finished)
    if done.is_stale():
        raise Error("a closed run reported stale")
    if len(done.config) != 2 or done.config[0] != String("lr:3e-4"):
        raise Error("config lines: " + String(len(done.config)))
    # ⚠ NON-ASCII IN `tag=` IS THE CASE `core/kv` WAS FIXED FOR — a byte-wise
    # reader that used `chr` per byte returned mojibake for exactly this.
    if done.tag != String("meilleur reach à ce jour, testé 8/10"):
        raise Error("tag came back as: " + done.tag)
    # ⚠ `outcome` CARRIES AN `=`, which only works because split_once cuts at
    # the FIRST one.
    if done.outcome != String("success_rate=0.82 val_l1=0.031"):
        raise Error("outcome: " + done.outcome)

    var before = done.finished
    run.close(String("killed"))
    var after = load_run(run.kv_path())
    if after.status != String("done") or after.finished != before:
        raise Error("close() was not idempotent: " + after.status)
    print("  transitions: running -> done, close() idempotent, tag/outcome exact")
    _ = run^


def test_a_stated_terminal_status_survives_close() raises:
    """⚠ THE INTERRUPT PATH (P0e) STATES `killed` BEFORE `close()` REPORTS THE
    DEFAULT. A killed run filed as `done` is worse than no record at all."""
    var root = _root()
    var run = RunContext(
        project=String("p"), driver=String("d.mojo"), root=root
    )
    run.set_status(String("killed"))
    run.close()
    var rec = load_run(run.kv_path())
    if rec.status != String("killed"):
        raise Error("close() overwrote a stated status with " + rec.status)
    print("  stated end: killed survived close()")
    _ = run^


def test_an_unknown_key_raises() raises:
    """⚠⚠ THE `tasks/spec.mojo` POLICY, NOT `data/manifest.mojo`'S. A manifest
    ignores unknown keys so a store from a newer build stays readable. Here a
    typo'd key is a LIE about what a run was."""
    var good = String("schema_version=1\nrun_id=r\nstatus=done\n")
    _ = parse_run(good, String("gate"))
    var bad = String("schema_version=1\nrun_id=r\nstatuss=done\n")
    var refused = False
    try:
        _ = parse_run(bad, String("gate"))
    except:
        refused = True
    if not refused:
        raise Error("a typo'd key was accepted — status= would be silently lost")
    var future = String("schema_version=2\nrun_id=r\n")
    var refused2 = False
    try:
        _ = parse_run(future, String("gate"))
    except:
        refused2 = True
    if not refused2:
        raise Error("a future schema_version was accepted")
    print("  strictness: unknown key and future schema both refused")


def test_the_record_is_rewritten_not_appended() raises:
    """⚠ ONE RENDERER, CALLED FROM EVERY MUTATOR. Appending the changed line is
    how a file ends up with two `status=` lines and a reader that believes the
    first — `_a_rule_written_inline_twice_drifts`."""
    var root = _root()
    var run = RunContext(
        project=String("p"), driver=String("d.mojo"), root=root
    )
    run.set_outcome(String("a"))
    run.set_outcome(String("b"))
    run.set_tag(String("t"))
    run.close()
    var text: String
    with open(run.kv_path(), "r") as fh:
        text = fh.read()
    var ls = kv_lines(text, String("gate"))
    var status_lines = 0
    var outcome_lines = 0
    for i in range(len(ls)):
        if ls[i].key == "status":
            status_lines += 1
        elif ls[i].key == "outcome":
            outcome_lines += 1
    print(
        "  single-valued keys:", status_lines, "status,", outcome_lines,
        "outcome (want 1, 1) across", len(ls), "lines",
    )
    if status_lines != 1 or outcome_lines != 1:
        raise Error("the record was appended to, not rewritten")
    _ = run^


def test_register_run_seeds_the_config_then_announces() raises:
    """⚠⚠ WHY `register()` IS HERE AND NOT IN `RunContext.__init__`. `/runs`
    carries the config, which is assembled from the run — and making
    `RunContext` generic over `Logger` to hold one would put a type parameter
    on every driver signature and every struct that stores a run.

    ⚠ ASSERT BEFORE `close()`. The remote half registers itself on close, so a
    check made after passes whether or not this function did anything — the
    vacuity that `test_run_lifecycle` already caught once.
    """
    var root = _root()
    var run = RunContext(
        project=String("so101"),
        driver=String("examples/so101/act_so101_train_gpu.mojo"),
        env=String("family:so101_tabletop"),
        task=String("so101_reach_brick"),
        seed=7,
        root=root,
    )
    var remote = RemoteLogger(
        server_url=String("http://127.0.0.1:9"), run_id=run.id
    )
    var lg = CompositeLogger(CsvLogger(run.metrics_path()), remote)
    register_run(run, lg)
    if not lg.b.registered():
        raise Error("register_run did not announce the run")
    # The dashboard's config must carry the run's identity, not the driver's
    # hand-assembled guess at it.
    var payload = lg.b._register_payload()
    var want = [
        String('"run_id":"') + run.id + '"',
        String('"project":"so101"'),
        String('"task":"so101_reach_brick"'),
        String('"seed":"7"'),
    ]
    var missing = 0
    for w in want:
        if payload.find(w) < 0:
            missing += 1
            print("    absent:", w)
    print("  register_run:", len(want) - missing, "of", len(want), "config fields")
    if missing != 0:
        raise Error("register_run seeded " + String(missing) + " fields short")
    lg.close()
    run.close()
    _ = run^


def main() raises:
    print("=" * 62)
    print("RunContext — one identifier, and a record written at t=0")
    print("=" * 62)
    test_iso8601_against_known_epochs()
    test_civil_from_days_edges()
    test_run_id_is_deterministic_and_separating()
    test_slugify()
    test_run_kv_exists_at_t0_and_says_running()
    test_status_transitions_and_close_idempotence()
    test_a_stated_terminal_status_survives_close()
    test_an_unknown_key_raises()
    test_the_record_is_rewritten_not_appended()
    test_register_run_seeds_the_config_then_announces()
    # ⚠ THE PREFIX IS A LITERAL, NOT A VARIABLE. An `rm -rf` assembled from a
    # String is one empty value away from a very bad day; this one cannot
    # widen, and every root above is minted under exactly this prefix.
    _ = run_capture(String("rm -rf /tmp/mojo_rl_run_gate_* 2>&1"), 4096)
    print("[PASS] run context")
