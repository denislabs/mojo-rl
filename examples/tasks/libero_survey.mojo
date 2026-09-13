"""LIBERO'S 130 TASKS AGAINST OUR TAXONOMY — P5, and it is a REPORT.

    pixi run mojo run -I . examples/tasks/libero_survey.mojo
    pixi run mojo run -I . examples/tasks/libero_survey.mojo <bddl_root>

`TASK_LAYER_PLAN.md` §P5: LIBERO is **not a port target** and never was; it is
"the cheapest available stress test of P1-P4 on somebody else's taxonomy".
This is that stress test, run against the real corpus in
`references/LIBERO-master`.

## ⚠⚠ THE PLAN'S P5 GATE CANNOT BE MET, AND THAT IS THE FINDING

`TASK_LAYER_IMPLEMENTATION.md` §6 says: *"Gate: LIBERO-Goal's 10 tasks load as
one family and step."* They load. **One of the ten has a goal our language can
express.** The other nine need capabilities we do not have, and this file
counts exactly which — see `docs/TASK_LAYER_IMPLEMENTATION.md` §6.1 for the
table and what each would cost.

That is not a failure of the importer. It is the stress test doing its job:
§3.6 of the design predicted LIBERO-Goal would be "one family, trivially",
and it is — one family, ten tasks, same fixtures and objects. What §3.6 did
not check is whether our GOAL LANGUAGE spans theirs. It does not.

## WHAT THE THREE COLUMNS MEAN

    parsed      the reader understood every block of the file
    family      fixtures/objects/regions translate to a `.family`
    task        the GOAL translates to a `.task` we could evaluate

⚠ `parsed` AND `task` ARE DELIBERATELY DIFFERENT COLUMNS. A file that parses
and does not translate is a capability gap, stated precisely. A file that does
not parse is a defect in the reader. Collapsing them into one number would
hide which of the two is happening.
"""

from std.os import listdir
from std.pathlib import Path
from std.sys import argv

from mojo_rl.tasks.bddl import parse_bddl, BddlProblem
from mojo_rl.tasks.libero_import import (
    translate_family, translate_task, GoalGap, classify_goal,
    resolve_family, family_todo_count,
    GAP_NONE, GAP_OBJECT_TARGET, GAP_FIXTURE_REGION, GAP_ARTICULATION,
    GAP_UNKNOWN_PRED, GAP_ARITY, gap_name,
)
from mojo_rl.tasks.libero_categories import load_libero_table, DEFAULT_TABLE_PATH


comptime DEFAULT_ROOT = "references/LIBERO-master/libero/libero/bddl_files"
# ⚠ THE PACK, IF PULLED, ELSE THE UPSTREAM TREE. `assets-pull` materialises
# the LIBERO pack at the first path; the second is the same files where the
# reference checkout keeps them. Same bytes either way (the pack is cut from
# that tree), so the survey's answer does not depend on which one it found.
comptime PACK_DIR = "mojo_rl/tasks/libero/assets"
comptime UPSTREAM_ASSETS = "references/LIBERO-master/libero/libero/assets"


def _bddl_files(root: String) raises -> List[String]:
    """Every `.bddl` under `root`, one level of suite directories deep.

    ⚠ SORTED BY SUITE THEN NAME. `listdir` hands back an arbitrary order and
    a report whose rows re-shuffle between runs cannot be diffed against the
    previous one — the same reason `physics3d/studio/panel.mojo` sorts.
    """
    var out = List[String]()
    var suites = List[String]()
    for e in listdir(root):
        suites.append(String(e))
    for i in range(len(suites)):
        for j in range(i + 1, len(suites)):
            if suites[j] < suites[i]:
                suites[i], suites[j] = suites[j], suites[i]
    for s in range(len(suites)):
        var d = root + "/" + suites[s]
        if not Path(d).is_dir():
            continue
        var names = List[String]()
        try:
            for e in listdir(d):
                var n = String(e)
                if n.endswith(".bddl"):
                    names.append(n)
        except:
            continue
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if names[j] < names[i]:
                    names[i], names[j] = names[j], names[i]
        for i in range(len(names)):
            out.append(d + "/" + names[i])
    return out^


def main() raises:
    var root = String(DEFAULT_ROOT)
    var a = argv()
    if len(a) > 1:
        root = String(a[1])

    print("=" * 74)
    print("LIBERO's corpus against our task layer — P5")
    print("=" * 74)

    if not Path(root).is_dir():
        # ⚠ SKIPPED, LOUDLY, AND NOT A PASS. `references/` is gitignored, so a
        # clone without it cannot run this. Printing "PASS" here would make a
        # missing corpus indistinguishable from a corpus that all translated.
        print("  SKIPPED: no LIBERO corpus at", root)
        print("  Put the upstream tree in references/LIBERO-master, or pass a")
        print("  bddl_files root as argv[1].")
        print("=== SKIPPED (no corpus — this is not a pass) ===")
        return

    var files = _bddl_files(root)
    print("  corpus:", len(files), "files under", root)
    if len(files) == 0:
        raise Error(
            "libero survey: found no .bddl under '" + root + "'. An empty"
            " corpus reports 0 failures, which is the shape of a vacuous run."
        )

    var table = load_libero_table(DEFAULT_TABLE_PATH)
    var pack_dir = String(PACK_DIR)
    if not Path(pack_dir).is_dir():
        pack_dir = String(UPSTREAM_ASSETS)
    var have_assets = Path(pack_dir).is_dir()

    var parsed = 0
    var fam_ok = 0
    var task_ok = 0
    var res_ok = 0
    var res_todo = 0
    var res_narrowed = 0
    var res_errors = List[String]()
    # L3: the goal TRANSLATES against the resolved family (box regions,
    # Joint terms from the table, On(obj, obj)) — the column that says
    # "a .task could be written", counted separately from the syntactic
    # classification so a missing threshold row is a named failure.
    var xl_ok = 0
    var xl_errors = List[String]()
    var gap_counts = List[Int]()
    for _ in range(6):
        gap_counts.append(0)
    var parse_errors = List[String]()

    # per-suite tallies, in file order
    var suite_names = List[String]()
    var suite_files = List[Int]()
    var suite_task = List[Int]()

    for i in range(len(files)):
        var path = files[i]
        # suite = the directory component
        var cut = -1
        for k in range(path.byte_length() - 1, -1, -1):
            if path[byte=k : k + 1] == "/":
                cut = k
                break
        var dirp = String(path[byte=:cut])
        var cut2 = -1
        for k in range(dirp.byte_length() - 1, -1, -1):
            if dirp[byte=k : k + 1] == "/":
                cut2 = k
                break
        var suite = String(dirp[byte=cut2 + 1 :])
        if len(suite_names) == 0 or suite_names[len(suite_names) - 1] != suite:
            suite_names.append(suite)
            suite_files.append(0)
            suite_task.append(0)
        suite_files[len(suite_files) - 1] += 1

        var text: String
        with open(path, "r") as f:
            text = f.read()

        var p: BddlProblem
        try:
            p = parse_bddl(text)
        except e:
            parse_errors.append(path + ": " + String(e))
            continue
        parsed += 1

        try:
            var _f = translate_family(p)
            fam_ok += 1
        except e:
            _ = e

        # L1: the RESOLVED family — every slot a file, every fixture a pose.
        if have_assets:
            try:
                var rf = resolve_family(p, table, pack_dir, res_narrowed)
                res_ok += 1
                res_todo += family_todo_count(rf)
                try:
                    var _t = translate_task(p, rf, table, pack_dir)
                    xl_ok += 1
                except e2:
                    xl_errors.append(path + ": " + String(e2))
            except e:
                res_errors.append(path + ": " + String(e))

        var gap = classify_goal(p)
        gap_counts[gap.kind] += 1
        if gap.kind == GAP_NONE:
            task_ok += 1
            suite_task[len(suite_task) - 1] += 1

    print()
    print("  parsed  :", parsed, "of", len(files))
    print("  family  :", fam_ok, "of", len(files),
          "translate to a .family (slots + regions)")
    print("  task    :", task_ok, "of", len(files),
          "have a goal our language can express")
    if have_assets:
        print("  resolved:", res_ok, "of", len(files),
              "translate with REAL asset paths and fixture poses (L1);"
              " TODO placeholders left:", res_todo)
        print("            fixture yaws taken at a narrow band's midpoint:",
              res_narrowed, "(libero_spatial's cabinet, 3.6 deg)")
        print("  written :", xl_ok, "of", len(files),
              "goals translate to a .task against the resolved family (L3)")
    else:
        print("  resolved: SKIPPED — no LIBERO assets at", PACK_DIR, "or",
              UPSTREAM_ASSETS, "(run `pixi run assets-pull libero`)")
    print()
    print("  per suite (goals we can express):")
    for i in range(len(suite_names)):
        print("     ", suite_names[i], ":", suite_task[i], "/",
              suite_files[i])
    print()
    print("  why the rest do not translate:")
    for k in range(6):
        if gap_counts[k] > 0 and k != GAP_NONE:
            print("     ", gap_counts[k], "x", gap_name(k))

    if len(parse_errors) > 0:
        print()
        print("  ⚠ PARSE FAILURES —", len(parse_errors),
              "(a reader defect, NOT a capability gap):")
        for i in range(len(parse_errors)):
            if i >= 5:
                print("      ... and", len(parse_errors) - 5, "more")
                break
            print("      ", parse_errors[i])

    print()
    # ⚠⚠ THE READER MUST HANDLE THE WHOLE CORPUS. A capability gap is a
    # finding; a parse failure is a bug, and they are asserted differently.
    if parsed != len(files):
        raise Error(
            "libero survey: " + String(len(files) - parsed) + " of "
            + String(len(files)) + " files did not PARSE. That is a defect in"
            " `tasks/bddl.mojo`, not a limit of the goal language — the two"
            " are separate columns for exactly this reason."
        )
    print("  ok: every file in the corpus parses")

    # ⚠⚠ L1's GATE: with the assets present, EVERY file resolves and NO
    # placeholder survives. A `TODO:` path composes to nothing at the first
    # `<attach>`; counting them here is what turns that into a number.
    if have_assets:
        if len(res_errors) > 0:
            print()
            print("  ⚠ RESOLUTION FAILURES —", len(res_errors), ":")
            for i in range(len(res_errors)):
                if i >= 8:
                    print("      ... and", len(res_errors) - 8, "more")
                    break
                print("      ", res_errors[i])
            raise Error(
                "libero survey: " + String(len(res_errors)) + " of "
                + String(len(files)) + " files did not RESOLVE against "
                + DEFAULT_TABLE_PATH + " — a missing category, a fixture"
                " with no ranged init, or a ranged fixture yaw. See above."
            )
        if res_todo != 0:
            raise Error(
                "libero survey: " + String(res_todo) + " TODO asset"
                " placeholders survived resolution"
            )
        print("  ok: all", res_ok, "files resolve to real assets, 0 TODO")
        if len(xl_errors) > 0:
            print()
            print("  ⚠ TRANSLATION FAILURES —", len(xl_errors), ":")
            for i in range(len(xl_errors)):
                if i >= 8:
                    print("      ... and", len(xl_errors) - 8, "more")
                    break
                print("      ", xl_errors[i])

    # ⚠ ANTI-VACUITY. "0 gaps" is also what a classifier that returns
    # GAP_NONE unconditionally reports, and "0 translated" is what one that
    # never returns it reports. The corpus has both, so both must appear.
    if task_ok == 0:
        raise Error(
            "libero survey: NOT ONE goal translated. `push_the_plate_to_the_"
            "front_of_the_stove` is `On(obj, table region with ranges)`, which"
            " our language does express — so zero means `classify_goal` is"
            " rejecting everything."
        )
    # ⚠ THE ANTI-VACUITY CHECK CHANGED SHAPE AT L3. Before it, the corpus
    # had both expressible and refused goals and both had to appear; after
    # it the language spans the corpus, so "130 of 130" is the EXPECTED
    # answer and vacuity is guarded the other way: every refusal path is
    # exercised by `tests/tasks/test_bddl.mojo` / `test_libero_goal_eval`
    # on hand-written inputs, and the WRITTEN column below must equal the
    # expressible one — a classifier that accepted everything while the
    # translator refused half would show there.
    if have_assets and xl_ok != task_ok:
        raise Error(
            "libero survey: " + String(task_ok) + " goals classify as"
            " expressible but " + String(xl_ok) + " translate — see the"
            " translation failures above"
        )
    var written = xl_ok if have_assets else task_ok
    print("  ok: expressible ==", task_ok, "== written", written)
    print()
    print("=== SURVEYED ===")
