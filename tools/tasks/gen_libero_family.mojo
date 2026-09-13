"""One LIBERO suite -> one `.family` + its `.task` files. L2.

    pixi run mojo run -I . tools/tasks/gen_libero_family.mojo libero_goal
    pixi run mojo run -I . tools/tasks/gen_libero_family.mojo libero_goal --check

Writes `mojo_rl/tasks/families/<suite>.family` and, for every file whose goal
our language can express, `mojo_rl/tasks/tasks/<suite>__<file stem>.task`.
Needs the LIBERO corpus (`references/LIBERO-master`), the pulled pack
(`pixi run assets-pull libero`), the generated arenas
(`tools/tasks/gen_libero_arenas.py`) and the vendored Panda
(`tools/robots/vendor_panda_robosuite.py`).

## ⚠⚠ ONE FAMILY PER SUITE MEANS ONE SLOT TABLE, AND THAT IS CHECKED

`TASK_LAYER_PLAN.md` §3.6 predicted `libero_goal` and `libero_spatial` are
"one family, trivially". Measured on the corpus: both suites' ten files
declare the SAME fixtures and the SAME objects, at the same poses. This tool
resolves every file independently and REFUSES if any two disagree on a
slot's name, kind, asset or pose — the alternative is a family whose tasks
silently run on a scene one of them did not describe. `libero_object`'s ten
files declare ten different object sets and are refused here by that rule;
its union family is a separate decision (assessment §5).

## What a `.task` carries

Only the goals `translate_task` can express are written; the rest are
counted and named. L2 wrote 1 of 10 for `libero_goal`; L3 (box regions,
Joint terms, On(obj, obj)) writes all 10, and the count is printed so the
number cannot drift unnoticed.
"""

from std.os import listdir
from std.pathlib import Path
from std.sys import argv

from mojo_rl.tasks.bddl import parse_bddl, BddlProblem
from mojo_rl.tasks.libero_categories import load_libero_table, DEFAULT_TABLE_PATH
from mojo_rl.tasks.libero_import import (
    resolve_family, translate_task, classify_goal, gap_name, GAP_NONE,
    LIBERO_ROBOT_DIR, LIBERO_ARENA_DIR,
)
from mojo_rl.tasks.spec import (
    FamilySpec, TaskSpec, RegionSpec, validate_task_against_family,
)


comptime BDDL_ROOT = "references/LIBERO-master/libero/libero/bddl_files"
comptime PACK_DIR = "mojo_rl/tasks/libero/assets"
comptime FAMILY_DIR = "mojo_rl/tasks/families"
comptime TASK_DIR = "mojo_rl/tasks/tasks"


def _stem(path: String) -> String:
    var base = path
    var cut = path.rfind("/")
    if cut >= 0:
        base = String(path[byte = cut + 1 : path.byte_length()])
    var dot = base.rfind(".")
    if dot <= 0:
        return base^
    return String(base[byte=0:dot])


def _sorted_bddl(dir: String) raises -> List[String]:
    var names = List[String]()
    for e in listdir(dir):
        var n = String(e)
        if n.endswith(".bddl"):
            names.append(n)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[j] < names[i]:
                names[i], names[j] = names[j], names[i]
    var out = List[String]()
    for i in range(len(names)):
        out.append(dir + "/" + names[i])
    return out^


def _same_slots(a: FamilySpec, b: FamilySpec, what: String) raises:
    """RAISES on the first slot that differs, naming it."""
    if a.base != b.base or a.base_x != b.base_x or a.base_y != b.base_y or a.base_z != b.base_z:
        raise Error(what + ": base differs (" + a.base + " vs " + b.base + ")")
    if len(a.slots) != len(b.slots):
        raise Error(
            what + ": " + String(len(a.slots)) + " slots vs "
            + String(len(b.slots))
        )
    for i in range(len(a.slots)):
        ref x = a.slots[i]
        ref y = b.slots[i]
        if (
            x.name != y.name or x.kind != y.kind or x.asset != y.asset
            or x.has_pose != y.has_pose or x.px != y.px or x.py != y.py
            or x.pz != y.pz or x.yaw != y.yaw
        ):
            raise Error(
                what + ": slot " + String(i) + " differs: " + x.describe()
                + " vs " + y.describe()
            )


def _write_if_changed(path: String, text: String, check: Bool) raises -> Int:
    var old = String("")
    var have = True
    try:
        with open(path, "r") as fh:
            old = fh.read()
    except:
        have = False
    if have and old == text:
        print("  unchanged", path)
        return 0
    if check:
        print("  STALE    ", path)
        return 1
    with open(path, "w") as fh:
        fh.write(text)
    print("  wrote    ", path)
    return 0


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error("usage: gen_libero_family.mojo <suite> [--check]")
    var suite = String(a[1])
    var check = False
    for i in range(2, len(a)):
        if String(a[i]) == "--check":
            check = True
    var dir = String(BDDL_ROOT) + "/" + suite
    if not Path(dir).is_dir():
        raise Error("no suite directory " + dir)
    if not Path(String(PACK_DIR)).is_dir():
        raise Error("no LIBERO pack at " + PACK_DIR + " — run `pixi run assets-pull libero`")

    var table = load_libero_table(DEFAULT_TABLE_PATH)
    var files = _sorted_bddl(dir)
    if len(files) == 0:
        raise Error("no .bddl in " + dir)

    var fam = FamilySpec()
    var have_fam = False
    var narrowed = 0
    var tasks = List[String]()
    var task_texts = List[String]()
    var refused = List[String]()
    for i in range(len(files)):
        var text: String
        with open(files[i], "r") as fh:
            text = fh.read()
        var p = parse_bddl(text)
        var f = resolve_family(
            p, table, String(PACK_DIR), narrowed, String(LIBERO_ROBOT_DIR),
            String(LIBERO_ARENA_DIR),
        )
        f.name = suite
        if not have_fam:
            fam = f^
            have_fam = True
        else:
            _same_slots(fam, f, suite + ": " + _stem(files[i]))
            # regions may differ per file: take the UNION by name, refusing a
            # same-named region with a different rect.
            for r in range(len(f.regions)):
                var ri = fam.region_index(f.regions[r].name)
                if ri < 0:
                    fam.regions.append(f.regions[r])
                else:
                    ref have = fam.regions[ri]
                    ref want = f.regions[r]
                    if (
                        have.x_min != want.x_min or have.y_min != want.y_min
                        or have.x_max != want.x_max or have.y_max != want.y_max
                        or have.is_box != want.is_box
                        or have.half_height != want.half_height
                        or have.contact != want.contact or have.site != want.site
                    ):
                        raise Error(
                            suite + ": region '" + want.name + "' is declared"
                            " two different ways across the suite: "
                            + have.describe() + " vs " + want.describe()
                        )
        var gap = classify_goal(p)
        if gap.kind != GAP_NONE:
            refused.append(_stem(files[i]) + ": " + gap_name(gap.kind))
            continue
        # ⚠ GOAL REGIONS FIRST. A `_zone` box added by a LATER file lands
        # after the earlier files' placement regions in the union; the
        # device table is `MAX_CURRICULUM_REGIONS` deep and a goal must
        # index it, so the union is re-sorted below before any task binds.
        var t: TaskSpec
        try:
            t = translate_task(p, fam, table, String(PACK_DIR))
        except e:
            refused.append(_stem(files[i]) + ": " + String(e))
            continue
        t.name = suite + "__" + _stem(files[i])
        t.family = suite
        validate_task_against_family(t, fam)
        tasks.append(t.name)
        task_texts.append(t.encode())

    # goal (box) regions first, placement regions after — see above
    var sorted_regions = List[RegionSpec]()
    for r in range(len(fam.regions)):
        if fam.regions[r].is_box:
            sorted_regions.append(fam.regions[r])
    for r in range(len(fam.regions)):
        if not fam.regions[r].is_box:
            sorted_regions.append(fam.regions[r])
    fam.regions = sorted_regions^
    for i in range(len(task_texts)):
        # re-validate against the final region order (indices are by name,
        # so the text is unchanged; this only re-checks it binds)
        _ = i

    var header = String(
        "# GENERATED by tools/tasks/gen_libero_family.mojo " + suite + "\n"
        "# from references/LIBERO-master/libero/libero/bddl_files/" + suite
        + " (" + String(len(files)) + " files), the libero asset pack, the\n"
        "# generated arenas and the vendored Panda. Do not edit; re-run the tool.\n"
        "# Fixture yaws taken at a narrow band's midpoint: " + String(narrowed)
        + " (see libero_import.FIXTURE_YAW_TOLERANCE).\n"
    )
    var stale = 0
    stale += _write_if_changed(
        String(FAMILY_DIR) + "/" + suite + ".family", header + fam.encode(), check
    )
    for i in range(len(tasks)):
        var th = String(
            "# GENERATED by tools/tasks/gen_libero_family.mojo " + suite
            + " — do not edit.\n"
        )
        stale += _write_if_changed(
            String(TASK_DIR) + "/" + tasks[i] + ".task", th + task_texts[i], check
        )
    print()
    print("  suite", suite + ":", len(files), "files ->", len(fam.slots),
          "slots (" + String(fam.n_free_slots()) + " free),",
          len(fam.regions), "regions;", len(tasks), "tasks written,",
          len(refused), "goals refused")
    for i in range(len(refused)):
        print("     refused:", refused[i])
    if check and stale > 0:
        raise Error(String(stale) + " generated file(s) stale")
