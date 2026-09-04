"""BDDL -> `.family` + `.task`, AND A REFUSAL FOR EVERYTHING ELSE. P5.

    var f = translate_family(p)     # slots + regions
    var t = translate_task(p, f)    # language + goal + init + active

`bddl.mojo` reads LIBERO's format; this maps it onto ours. The interesting
half is what it will NOT map.

## ⚠⚠ IT REFUSES RATHER THAN APPROXIMATING, AND THAT IS THE WHOLE DESIGN

A goal this cannot express is a goal that must not be emitted. `spec.mojo`
already fixed this rule for our own files — *"a silently dropped `goal=` is a
task that always succeeds"* — and an importer is where the temptation to
approximate is strongest: `On(bowl, plate)` is ALMOST `On(bowl, plate_region)`,
and a policy trained against the almost is a policy evaluated on the wrong
benchmark.

`classify_goal` returns WHICH capability is missing, so the survey can count
gaps by kind instead of reporting one opaque failure total.

## THE FOUR GAPS, MEASURED ON ALL 130 FILES

| gap | what LIBERO writes | why ours cannot say it |
|---|---|---|
| `GAP_OBJECT_TARGET` | `On(bowl_1, plate_1)` | our `On`'s second argument is a REGION; there is no object-relative form |
| `GAP_FIXTURE_REGION` | `On(bowl_1, cabinet_1_top_region)` | the region has no `:ranges` — its rectangle is a `<site>` in the fixture's asset XML, which the `.bddl` does not carry |
| `GAP_ARTICULATION` | `Open(cabinet_1_middle_region)`, `Turnon(stove_1)` | reads a fixture JOINT's qpos against per-class thresholds that live in LIBERO's PYTHON (`ArticulatedObject.default_open_ranges`), not in the file |
| `GAP_UNKNOWN_PRED` | — | a predicate outside the measured seven |

⚠ `GAP_FIXTURE_REGION` IS THE CHEAPEST TO CLOSE and the one worth closing
first: our `region=` is ALREADY site-anchored (`region=table_top:site:
table_surface:x0,y0,x1,y1`), so the missing piece is only the rectangle, and
the rectangle is the site's own `size` in the asset. It needs an asset reader,
not a language change.

⚠ `GAP_ARTICULATION` IS NOT A PREDICATE WE FORGOT. It is a different KIND of
predicate: every Tier A predicate we have reads a body pose, and this reads a
JOINT. It also cannot be imported as data — `WoodenCabinet.is_open` is
`qpos < max(default_open_ranges)` with NEGATIVE ranges while `ShortCabinet`'s
is `qpos > min(...)` with positive ones, so the comparison DIRECTION is
per-class Python. Importing the threshold without the direction gives a
predicate that is exactly backwards on half the cabinets.
"""

from .bddl import BddlProblem, BddlAtom, BddlRegion
from .spec import (
    FamilySpec, TaskSpec, SlotSpec, RegionSpec, InitSpec,
    SLOT_FREE, SLOT_STATIC, SCHEMA_VERSION,
)


comptime GAP_NONE: Int = 0
comptime GAP_OBJECT_TARGET: Int = 1
comptime GAP_FIXTURE_REGION: Int = 2
comptime GAP_ARTICULATION: Int = 3
comptime GAP_UNKNOWN_PRED: Int = 4
comptime GAP_ARITY: Int = 5


def gap_name(kind: Int) -> String:
    if kind == GAP_NONE:
        return String("expressible")
    if kind == GAP_OBJECT_TARGET:
        return String(
            "On/In whose target is an OBJECT, not a region (we have no"
            " object-relative form)"
        )
    if kind == GAP_FIXTURE_REGION:
        return String(
            "On/In onto a region with no :ranges (its rect is a <site> in the"
            " fixture's asset, not in the .bddl)"
        )
    if kind == GAP_ARTICULATION:
        return String(
            "Open/Close/Turnon/Turnoff — reads a fixture JOINT, and the"
            " threshold's DIRECTION lives in LIBERO's Python"
        )
    if kind == GAP_UNKNOWN_PRED:
        return String("a predicate outside the measured seven")
    return String("a predicate with an argument count we do not expect")


struct GoalGap(Copyable, ImplicitlyCopyable, Movable):
    """Why a goal does not translate, and which term caused it."""

    var kind: Int
    var term: String

    def __init__(out self, kind: Int, var term: String):
        self.kind = kind
        self.term = term^


def classify_goal(p: BddlProblem) raises -> GoalGap:
    """`GAP_NONE` if every goal term maps, else the FIRST gap and its term.

    ⚠ THE FIRST, NOT ALL OF THEM. A goal is a conjunction and one
    untranslatable term makes the whole task untranslatable, so reporting
    further gaps in the same file would inflate the survey's counts past the
    number of TASKS blocked.
    """
    for i in range(len(p.goal)):
        ref g = p.goal[i]
        if (
            g.pred == "Open" or g.pred == "Close"
            or g.pred == "Turnon" or g.pred == "Turnoff"
        ):
            return GoalGap(GAP_ARTICULATION, g.show())
        if g.pred != "On" and g.pred != "In":
            return GoalGap(GAP_UNKNOWN_PRED, g.show())
        if len(g.args) != 2:
            return GoalGap(GAP_ARITY, g.show())
        var target = String(g.args[1])
        if p.is_object(target) or p.is_fixture(target):
            return GoalGap(GAP_OBJECT_TARGET, g.show())
        var ri = p.region_index(target)
        if ri < 0:
            return GoalGap(GAP_UNKNOWN_PRED, g.show())
        if not p.regions[ri].has_ranges:
            return GoalGap(GAP_FIXTURE_REGION, g.show())
    return GoalGap(GAP_NONE, String(""))


def translate_family(p: BddlProblem) raises -> FamilySpec:
    """Fixtures -> static slots, objects -> free slots, ranged regions ->
    regions.

    ⚠⚠ THE ASSET PATHS ARE THE CATEGORY NAME, NOT A FILE THAT EXISTS. LIBERO
    resolves `akita_black_bowl` through a Python object registry onto
    `assets/stable_scanned_objects/...`; the `.bddl` names only the category.
    Emitting a path here would be inventing one. So the slot's asset is the
    CATEGORY, and a `.family` written from this does not compose until someone
    maps categories to files — which is a deliberate, visible TODO rather than
    a path that looks right and is not.

    ⚠ A REGION WITH NO `:ranges` IS DROPPED, and the goal that names it is
    refused by `classify_goal` — the two must agree, or a task would reference
    a region the family does not declare and `validate_task_against_family`
    would reject it with a confusing message instead of the real reason.
    """
    var f = FamilySpec()
    f.schema_version = SCHEMA_VERSION
    f.name = String(p.problem)
    f.base = String("TODO:") + p.domain
    for i in range(len(p.fixtures)):
        var s = SlotSpec(
            String(p.fixtures[i].name), SLOT_STATIC,
            String("TODO:") + p.fixtures[i].category,
        )
        # ⚠ A STATIC SLOT REQUIRES A POSE and the `.bddl` gives it as an INIT
        # RANGE, not a point — `(On wooden_cabinet_1 main_table_cabinet_region)`
        # with a rectangle. The centre of that rectangle is the honest reading
        # and it is what LIBERO samples around; a fixture whose init region is
        # missing keeps (0,0,0) and the caller is told by `family_todo_count`.
        s.has_pose = True
        s.px = 0.0
        s.py = 0.0
        s.pz = 0.0
        for k in range(len(p.init)):
            ref a = p.init[k]
            if a.pred == "On" and len(a.args) == 2 and a.args[0] == s.name:
                var ri = p.region_index(String(a.args[1]))
                if ri >= 0 and p.regions[ri].has_ranges:
                    ref r = p.regions[ri]
                    s.px = 0.5 * (r.x0 + r.x1)
                    s.py = 0.5 * (r.y0 + r.y1)
                    s.pz = 0.0
        f.slots.append(s^)
    for i in range(len(p.objects)):
        var s2 = SlotSpec(
            String(p.objects[i].name), SLOT_FREE,
            String("TODO:") + p.objects[i].category,
        )
        f.slots.append(s2^)
    for i in range(len(p.regions)):
        ref r = p.regions[i]
        if not r.has_ranges:
            continue
        var rs = RegionSpec(r.composed_name(), r.composed_name())
        rs.has_rect = True
        rs.x_min = r.x0
        rs.y_min = r.y0
        rs.x_max = r.x1
        rs.y_max = r.y1
        f.regions.append(rs^)
    return f^


def translate_task(p: BddlProblem, f: FamilySpec) raises -> TaskSpec:
    """The `.task`. RAISES with the gap's name if the goal does not map."""
    var gap = classify_goal(p)
    if gap.kind != GAP_NONE:
        raise Error(
            "libero: task '" + p.problem + "' (" + p.language + ") cannot be"
            " translated — " + gap_name(gap.kind) + ". The blocking term is "
            + gap.term + ". Refused rather than approximated: a goal that is"
            " ALMOST the benchmark's is a policy evaluated on the wrong task."
        )
    var t = TaskSpec()
    t.schema_version = SCHEMA_VERSION
    t.name = String(p.problem)
    t.family = String(f.name)
    t.language = String(p.language)

    var goal = String("")
    for i in range(len(p.goal)):
        ref g = p.goal[i]
        var term = String(g.pred) + "(" + g.args[0] + ", " + g.args[1] + ")"
        goal = term^ if i == 0 else (String("And(") + goal + ", " + term + ")")
    t.goal = goal^

    # ⚠ ACTIVE = EVERY SLOT, not `:obj_of_interest`. LIBERO's field names what
    # the INSTRUCTION is about; our `active=` names what the SCENE contains,
    # and a prop that is present but not mentioned is still on the table. Using
    # obj_of_interest would park the distractors and quietly make every task
    # easier than the benchmark's.
    for i in range(len(f.slots)):
        t.active.append(String(f.slots[i].name))

    for k in range(len(p.init)):
        ref a = p.init[k]
        if a.pred != "On" or len(a.args) != 2:
            continue
        var slot = String(a.args[0])
        var ri = p.region_index(String(a.args[1]))
        if ri < 0 or not p.regions[ri].has_ranges:
            continue
        # only FREE slots take an init; a fixture's placement is its pose
        var is_free = False
        for s in range(len(f.slots)):
            if f.slots[s].name == slot and f.slots[s].kind == SLOT_FREE:
                is_free = True
        if not is_free:
            continue
        t.inits.append(InitSpec(slot, p.regions[ri].composed_name()))
    return t^


def family_todo_count(f: FamilySpec) -> Int:
    """How many asset paths are still `TODO:` placeholders.

    ⚠ EXISTS SO A CALLER CANNOT MISS THEM. A `.family` written straight out of
    a `.bddl` names CATEGORIES, not files; composing it would fail at the
    first `<attach>`. Counting them makes that a number a tool can print
    rather than a surprise at compose time."""
    var n = 0
    if f.base.startswith("TODO:"):
        n += 1
    for i in range(len(f.slots)):
        if f.slots[i].asset.startswith("TODO:"):
            n += 1
    return n
