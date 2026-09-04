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
    GAP_NONE, GAP_OBJECT_TARGET, GAP_FIXTURE_REGION, GAP_ARTICULATION,
    GAP_UNKNOWN_PRED, GAP_ARITY, gap_name,
)


comptime DEFAULT_ROOT = "references/LIBERO-master/libero/libero/bddl_files"


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

    var parsed = 0
    var fam_ok = 0
    var task_ok = 0
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
    if task_ok == len(files):
        raise Error(
            "libero survey: EVERY goal translated, which contradicts the"
            " measured corpus — 61 On, 63 In, and 27 articulation predicates"
            " we have no equivalent for. `classify_goal` is accepting"
            " everything."
        )
    print("  ok: the survey found BOTH translatable and untranslatable goals")
    print()
    print("=== SURVEYED ===")
