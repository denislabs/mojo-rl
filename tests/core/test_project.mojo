"""`project.kv` — the definition, the pinned refs, and where a run lands.

Run: pixi run mojo run -I . tests/core/test_project.mojo

A project is a container and a naming authority; it owns nothing computational
(`docs/PROJECT_LAYER_PLAN.md` §3). What it must get right is the part that keeps
`run.kv` honest: a `ref=` that says WHICH `.family` and `.task` a run used, and
at which commit.

⚠⚠ THE PIN IS THE POINT. Decision 11 chose a reference over a copy on the
reversibility argument. But an unpinned reference resolves against whatever the
working tree happens to be, so a `.family` that drifted makes `run.kv` a lie
about what was trained — this layer's own failure mode, one level up.

⚠ WRITES UNDER /tmp, never the repo's `projects/`.
"""

from std.time import perf_counter_ns

from mojo_rl.core.project import (
    ProjectSpec,
    check_refs,
    load_project,
    parse_project,
    parse_ref,
    project_exists,
    runs_root_for,
)
from mojo_rl.core.run import RunContext
from mojo_rl.io.proc import run_capture


def _root() -> String:
    return String("/tmp/mojo_rl_proj_gate_") + String(perf_counter_ns())


def _mk(root: String, name: String) raises -> ProjectSpec:
    _ = run_capture(String("mkdir -p ") + root + "/" + name + " 2>&1", 4096)
    var p = ProjectSpec(name, root)
    p.created = String("2026-09-09T00:00:00Z")
    p.description = String("gate")
    return p^


# =============================================================================


def test_ref_parses_path_name_and_commit() raises:
    """⚠ SPLIT-ONCE TWICE, NOT A GREEDY SPLIT ON `:`. Only the first two colons
    are separators; a path may contain more."""
    var cases = [
        (
            String("family:so101_tabletop:mojo_rl/tasks/families/x.family@081b53c0"),
            String("family"), String("so101_tabletop"),
            String("mojo_rl/tasks/families/x.family"), String("081b53c0"),
        ),
        (   # no commit — legal, and it means "unpinned"
            String("task:reach:mojo_rl/tasks/tasks/reach.task"),
            String("task"), String("reach"),
            String("mojo_rl/tasks/tasks/reach.task"), String(""),
        ),
        (   # a path carrying its own colon
            String("dataset:teleop:s3:bucket/x.h5@abc1234"),
            String("dataset"), String("teleop"),
            String("s3:bucket/x.h5"), String("abc1234"),
        ),
        (   # ⚠ A PATH CARRYING ITS OWN `@`. This is why the commit is split
            # from the RIGHT: a HuggingFace-style ref or a user@host path has
            # one, and a left-split would eat the path and call the rest a
            # commit. Without this case the claim in `parse_ref` is untested —
            # measured: a left-split mutant passed the gate before it existed.
            String("dataset:hub:datasets/org/name@v1/file.h5@abc1234"),
            String("dataset"), String("hub"),
            String("datasets/org/name@v1/file.h5"), String("abc1234"),
        ),
    ]
    var compared = 0
    var differing = 0
    for c in cases:
        compared += 1
        var r = parse_ref(c[0], String("gate"))
        if r.kind != c[1] or r.name != c[2] or r.path != c[3] or r.commit != c[4]:
            differing += 1
            print("    ", c[0], "->", r.kind, r.name, r.path, r.commit)
        if r.encode() != c[0]:
            differing += 1
            print("    re-encode differs:", r.encode(), "vs", c[0])
    print("  refs:", compared, "parsed,", differing, "differing")
    if compared != 4 or differing != 0:
        raise Error("ref parsing wrong on " + String(differing))


def test_project_round_trips() raises:
    var root = _root()
    var p = _mk(root, String("so101"))
    p.add_ref(
        String("family"), String("so101_tabletop"),
        String("mojo_rl/tasks/families/so101_tabletop.family"), String("081b53c0"),
    )
    p.add_ref(
        String("task"), String("so101_reach_brick"),
        String("mojo_rl/tasks/tasks/so101_reach_brick.task"), String("081b53c0"),
    )
    p.datasets.append(String("teleop_reach_v1@v1"))
    p.write()

    var back = load_project(String("so101"), root)
    if back.name != String("so101") or len(back.refs) != 2:
        raise Error("round trip lost fields: " + String(len(back.refs)))
    if len(back.tasks()) != 1 or back.tasks()[0] != String("so101_reach_brick"):
        raise Error("tasks() wrong")
    var fam = back.ref_of(String("family"), String("so101_tabletop"))
    if fam.commit != String("081b53c0"):
        raise Error("pin lost: " + fam.commit)
    # ⚠ add_ref REPLACES rather than appending a second line for one name — two
    # `ref=` lines for one family is a project that disagrees with itself.
    back.add_ref(
        String("family"), String("so101_tabletop"),
        String("mojo_rl/tasks/families/so101_tabletop.family"), String("deadbee1"),
    )
    if len(back.refs) != 2:
        raise Error("add_ref appended a duplicate: " + String(len(back.refs)))
    print("  project.kv: 2 refs, 1 dataset, pin and tasks() survive")
    _ = run_capture(String("rm -rf ") + root + " 2>&1", 4096)


def test_unknown_key_and_bad_schema_raise() raises:
    """⚠ THE `tasks/spec.mojo` POLICY. A project definition is hand-edited; a
    typo'd key is a task set that silently lost a member."""
    _ = parse_project(String("schema_version=1\nname=p\n"), String("gate"))
    var refused = 0
    for bad in [
        String("schema_version=1\nname=p\ntasks=x\n"),   # typo: tasks, not ref
        String("schema_version=2\nname=p\n"),            # future schema
        String("schema_version=1\ndescription=x\n"),     # no name
    ]:
        try:
            _ = parse_project(bad, String("gate"))
        except:
            refused += 1
    print("  strictness:", refused, "of 3 refused")
    if refused != 3:
        raise Error("a malformed project was accepted")


def test_check_refs_finds_missing_and_drift() raises:
    """⚠ DRIFT IS `git log -1 -- <path>`, NOT HEAD — the question is whether
    THAT FILE changed since the project pinned it, not whether anything did."""
    var root = _root()
    var p = _mk(root, String("g"))
    # a real tracked file, pinned to a commit that certainly did not touch it
    p.add_ref(
        String("family"), String("real"),
        String("mojo_rl/tasks/families/so101_tabletop.family"),
        String("0000000"),
    )
    p.add_ref(
        String("task"), String("gone"),
        String("mojo_rl/tasks/tasks/does_not_exist.task"), String(""),
    )
    var r = check_refs(p, verbose=False)
    print(
        "  check_refs:", r.checked, "checked,", r.missing, "missing,",
        r.drifted, "drifted",
    )
    if r.checked != 2:
        raise Error("checked " + String(r.checked) + ", want 2")
    if r.missing != 1:
        raise Error("missing file not reported: " + String(r.missing))
    if r.drifted != 1:
        raise Error("a bogus pin was not reported as drift: " + String(r.drifted))
    if r.ok():
        raise Error("ok() true with a missing ref and a drifted pin")
    _ = run_capture(String("rm -rf ") + root + " 2>&1", 4096)


def test_a_run_lands_under_its_project_once_the_project_exists() raises:
    """⚠⚠ THE FALLBACK IS WHAT LETS THE SEVEN RETROFITTED DRIVERS STAY UNEDITED.
    A driver names its project; until someone runs `project-init` the run lands
    in the flat `runs/` root, and afterwards under the project — with no change
    to the driver either way."""
    var name = String("gateproj_") + String(perf_counter_ns())
    if project_exists(name):
        raise Error("a fresh project name already exists")
    var before = runs_root_for(name)
    if before != String("runs"):
        raise Error("fallback root is '" + before + "', want 'runs'")
    print("  placement: unknown project ->", before, "(the flat P0 root)")
    # The positive half needs `projects/<name>/project.kv` under the REAL root,
    # so it is asserted through the same function with a temp root instead.
    var root = _root()
    var p = _mk(root, String("so101"))
    p.write()
    var loaded = load_project(String("so101"), root)
    if loaded.runs_dir() != root + "/so101/runs":
        raise Error("runs_dir: " + loaded.runs_dir())
    if loaded.policies_dir() != root + "/so101/policies":
        raise Error("policies_dir: " + loaded.policies_dir())
    print("  placement: known project ->", loaded.runs_dir())
    _ = run_capture(String("rm -rf ") + root + " 2>&1", 4096)


def main() raises:
    print("=" * 62)
    print("project.kv — the definition, the pins, and where a run lands")
    print("=" * 62)
    test_ref_parses_path_name_and_commit()
    test_project_round_trips()
    test_unknown_key_and_bad_schema_raise()
    test_check_refs_finds_missing_and_drift()
    test_a_run_lands_under_its_project_once_the_project_exists()
    _ = run_capture(String("rm -rf /tmp/mojo_rl_proj_gate_* 2>&1"), 4096)
    print("[PASS] project")
