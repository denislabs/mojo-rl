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
from .libero_categories import LiberoTable, LiberoCategory, LiberoProblem
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
    # ⚠ FAMILY SLOT ORDER, NOT `:init` ORDER. `validate_task_against_family`
    # refuses any other order because the host and device samplers walk
    # different lists and rejection sampling is order-dependent. A `.bddl`
    # lists its `:init` in authoring order, which is neither.
    var ordered = List[InitSpec]()
    for si in range(len(f.slots)):
        for k in range(len(t.inits)):
            if t.inits[k].slot == f.slots[si].name:
                ordered.append(t.inits[k])
    t.inits = ordered^
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


# ── L1: the resolved family — real assets, real fixture poses ──────────────


def _tag_attr(tag: String, name: String) -> String:
    """`name="..."` inside one opening tag, or "" if absent."""
    var needle = String(" ") + name + '="'
    var i = tag.find(needle)
    if i < 0:
        return String("")
    var start = i + needle.byte_length()
    var end = tag.find('"', start)
    if end < 0:
        return String("")
    return String(tag[byte=start:end])


def site_pos(xml: String, site: String) raises -> List[Float64]:
    """The `pos` of `<site ... name="<site>" .../>` in an asset XML.

    ⚠ A TEXT SCAN, NOT A PARSE, on purpose: the three robosuite bookkeeping
    sites (`bottom_site`, `top_site`, `horizontal_radius_site`) sit on the
    wrapper body of every LIBERO object and this is the only thing the
    importer reads out of an asset. `MujocoXMLObject.bottom_offset` reads the
    same attribute the same way (`string_to_array(site.get("pos"))`).
    RAISES if the site is absent — an object without one cannot be placed by
    LIBERO either.
    """
    var pos = 0
    while True:
        var i = xml.find("<site", pos)
        if i < 0:
            break
        var j = xml.find(">", i)
        if j < 0:
            break
        var tag = String(xml[byte=i:j])
        if _tag_attr(tag, String("name")) == site:
            var ps = _tag_attr(tag, String("pos"))
            if ps.byte_length() == 0:
                ps = String("0 0 0")
            var out = List[Float64]()
            var toks = ps.split(" ")
            for k in range(len(toks)):
                var t = String(String(toks[k]).strip())
                if t.byte_length() > 0:
                    out.append(Float64(t))
            if len(out) != 3:
                raise Error(
                    "libero: site '" + site + "' has a pos with "
                    + String(len(out)) + " numbers"
                )
            return out^
        pos = j + 1
    raise Error(
        "libero: no <site name=\"" + site + "\"> in the asset — LIBERO"
        " places every object by this site and so must we"
    )


# ⚠ A FIXTURE YAW RANGE NARROWER THAN THIS IS TAKEN AT ITS MIDPOINT. Measured
# on the corpus: 475 of 501 yaw blocks are a point; the only RANGED fixture is
# `libero_spatial`'s wooden cabinet, [2.6614, 2.7242] rad — a 3.6 degree band.
# LIBERO itself re-draws that yaw at EVERY reset, frozen init or not (a
# fixture's pose is `model.body_pos`, which `sim.get_state()` does not carry),
# so the band is reset NOISE in the benchmark, not part of a task's identity.
# A static slot has one pose; the midpoint is the mean of what LIBERO draws
# and the deviation is at most 1.8 degrees. Wider than this and it is a
# distribution the family cannot express, so it RAISES.
comptime FIXTURE_YAW_TOLERANCE: Float64 = 0.1


# ⚠ THE ROBOT AND THE ARENA ARE L2's HALF OF THE FAMILY. With `robot_xml`
# empty (`robot_dir`) the family's base is the raw arena scene (L1's shape, kept for the
# survey and the tests); with it set, the base is the vendored Panda placed
# at the problem's `base_pos`, the styled arena from `arena_dir` is the
# first static slot at the origin, the composer's own floor is off, and the
# Panda's `<option>` is inherited. Regions then anchor on the arena's
# `workspace` site, which `gen_libero_arenas.py` places at LIBERO's
# `workspace_offset` — the point every region rect is measured from.
comptime LIBERO_ARENA_DIR: String = "mojo_rl/tasks/libero/scenes"
# ⚠ IN ROBOT MODE THE OBJECTS ARE THE GENERATED COPIES, NOT THE PACK'S.
# `tools/tasks/gen_libero_objects.py` writes one attachable XML per category:
# the pack's files are byte-identical to upstream, and upstream never loads
# them directly — robosuite's `merge_assets` drops a repeated (tag, name)
# asset declaration, which `flat_stove.xml` relies on and which MuJoCo's
# `<attach>` refuses. Without the robot (L1's survey shape) the pack path is
# still emitted, so the survey needs no generated files.
comptime LIBERO_OBJECT_DIR: String = "mojo_rl/tasks/libero/objects"
comptime LIBERO_ROBOT_DIR: String = "mojo_rl/envs/robots/assets"
comptime ARENA_SLOT: String = "arena"


# LIBERO `MountedPanda.init_qpos` / `OnTheGroundPanda.init_qpos` (identical),
# `envs/robots/*_panda.py`, plus `PandaGripper.init_qpos` (0.020833,
# -0.020833). The family's `base_qpos=`; the oracle and the env read it.
comptime PANDA_INIT_QPOS: String = (
    "0,-0.161037389,0,-2.44459747,0,2.2267522,0.7853981633974483,"
    "0.020833,-0.020833"
)


def robot_path(prob: LiberoProblem, robot_dir: String) -> String:
    """`panda_robosuite.xml` for `robot=mounted`, `panda_robosuite_nomount.xml`
    for `robot=on_the_ground` — both written by
    `tools/robots/vendor_panda_robosuite.py`."""
    if prob.robot == "on_the_ground":
        return robot_dir + "/panda_robosuite_nomount.xml"
    return robot_dir + "/panda_robosuite.xml"


def arena_path(prob: LiberoProblem, arena_dir: String) -> String:
    """`<arena_dir>/<problem, lowercased>_arena.xml` — ONE spelling, shared
    with `tools/tasks/gen_libero_arenas.py`."""
    return arena_dir + "/" + prob.name.lower() + "_arena.xml"


def resolve_family(
    p: BddlProblem, table: LiberoTable, pack_dir: String,
) raises -> FamilySpec:
    var narrowed = 0
    return resolve_family(p, table, pack_dir, narrowed, String(""), String(""))


def resolve_family(
    p: BddlProblem, table: LiberoTable, pack_dir: String, mut narrowed: Int,
) raises -> FamilySpec:
    return resolve_family(p, table, pack_dir, narrowed, String(""), String(""))


def resolve_family(
    p: BddlProblem, table: LiberoTable, pack_dir: String, mut narrowed: Int,
    robot_dir: String, arena_dir: String,
) raises -> FamilySpec:
    """`translate_family` with the registry: every slot names a FILE under
    `pack_dir`, every fixture carries the pose LIBERO computes for it, and
    the workspace (`main_table - table`) is the base scene rather than a slot.

    ## The fixture pose, quoted from LIBERO

    `bddl_base_domain._add_placement_initializer` places a fixture with a
    `MultiRegionRandomSampler(z_offset=self.z_offset, rotation=yaw_rotation,
    rotation_axis="z", reference_pos=self.workspace_offset)`, and
    `base_region_sampler.sample` then does

        object_x = sample_x + reference_pos[0]
        object_y = sample_y + reference_pos[1]
        object_z = z_offset + reference_pos[2] - bottom_offset[2]

    A static slot cannot sample, so x,y are the rect's CENTRE — the mean of
    what LIBERO draws — and the yaw is the range's MIDPOINT when the range
    is narrower than `FIXTURE_YAW_TOLERANCE` (see it for why), counted in
    `narrowed`, and REFUSED when wider.

    ⚠ THE REGION SITE IS AN OBLIGATION ON L2. Table regions are emitted
    anchored on `robot_<workspace>_top`, i.e. a site named `<workspace>_top`
    in the base scene at `workspace_offset`; robosuite's `TableArena`
    re-poses the arena's `table_top` site there at load time, and the styled
    base scene L2 generates must do the same. Until then the family composes
    but its table regions bind to nothing — `family_todo_count` does not
    count that, `validate_task_against_family` does.
    """
    var prob = table.problem(p.problem)
    var f = FamilySpec()
    f.schema_version = SCHEMA_VERSION
    f.name = String(p.problem)
    f.horizon = 600
    f.control_freq = 20
    var with_robot = robot_dir.byte_length() > 0
    var anchor = String("robot_") + prob.workspace + "_top"
    if with_robot:
        f.base = robot_path(prob, robot_dir)
        f.base_x = prob.base_x
        f.base_y = prob.base_y
        f.base_z = prob.base_z
        f.floor = False
        f.inherit_option = True
        var q = PANDA_INIT_QPOS.split(",")
        for k in range(len(q)):
            f.base_qpos.append(Float64(String(q[k])))
        f.slots.append(
            SlotSpec(
                String(ARENA_SLOT), SLOT_STATIC, arena_path(prob, arena_dir),
                0.0, 0.0, 0.0,
            )
        )
        anchor = String(ARENA_SLOT) + "_workspace"
    else:
        f.base = pack_dir + "/" + prob.scene

    for i in range(len(p.fixtures)):
        var cat = table.category(String(p.fixtures[i].category))
        if cat.is_workspace():
            if cat.name != prob.workspace:
                raise Error(
                    "libero: task '" + p.problem + "' declares workspace"
                    " fixture '" + p.fixtures[i].name + " - " + cat.name
                    + "' but its problem class works on '" + prob.workspace
                    + "'"
                )
            continue
        var asset = pack_dir + "/" + cat.asset
        if with_robot:
            asset = String(LIBERO_OBJECT_DIR) + "/" + cat.name + ".xml"
        # the init line that places it, and its region
        var ri = -1
        for k in range(len(p.init)):
            ref a = p.init[k]
            if a.pred == "On" and len(a.args) == 2 and a.args[0] == p.fixtures[i].name:
                ri = p.region_index(String(a.args[1]))
        if ri < 0 or not p.regions[ri].has_ranges:
            raise Error(
                "libero: fixture '" + p.fixtures[i].name + "' in task '"
                + p.problem + "' has no `(On <fixture> <ranged region>)`"
                " init, so it has no pose"
            )
        ref r = p.regions[ri]
        var yaw = 0.0
        if r.has_yaw:
            var band = r.yaw_hi - r.yaw_lo
            if band < 0.0:
                band = -band
            if band > FIXTURE_YAW_TOLERANCE:
                raise Error(
                    "libero: fixture '" + p.fixtures[i].name + "' in task '"
                    + p.problem + "' samples its yaw in ["
                    + String(r.yaw_lo) + ", " + String(r.yaw_hi) + "], a "
                    + String(band) + " rad band; a static slot has ONE pose"
                    " and only a band under " + String(FIXTURE_YAW_TOLERANCE)
                    + " rad is taken at its midpoint. Not approximated."
                )
            if band > 0.0:
                narrowed += 1
            yaw = 0.5 * (r.yaw_lo + r.yaw_hi)
        # ⚠ THE BOOKKEEPING SITE IS READ FROM THE PACK'S FILE, never from the
        # generated copy: robosuite reads `bottom_site` off the wrapper body
        # and then discards the wrapper, and the generated copy does the same.
        var xml: String
        with open(pack_dir + "/" + cat.asset, "r") as fh:
            xml = fh.read()
        var bottom = site_pos(xml, String("bottom_site"))
        var s = SlotSpec(
            String(p.fixtures[i].name), SLOT_STATIC, asset,
            0.5 * (r.x0 + r.x1) + prob.off_x,
            0.5 * (r.y0 + r.y1) + prob.off_y,
            prob.z_offset + prob.off_z - bottom[2],
            yaw,
        )
        f.slots.append(s^)

    for i in range(len(p.objects)):
        var cat2 = table.category(String(p.objects[i].category))
        if cat2.is_workspace():
            raise Error(
                "libero: '" + p.objects[i].name + "' is declared as an OBJECT"
                " but its category '" + cat2.name + "' is a workspace"
            )
        var asset2 = pack_dir + "/" + cat2.asset
        if with_robot:
            # ⚠ `_free`: the generated copy WITH robosuite's free joint. The
            # pack's XML has none — robosuite injects it at load for
            # `:objects` and not for `:fixtures` — so a free slot on the
            # plain copy would be welded to the world and never move.
            asset2 = String(LIBERO_OBJECT_DIR) + "/" + cat2.name + "_free.xml"
        f.slots.append(SlotSpec(String(p.objects[i].name), SLOT_FREE, asset2))

    for i in range(len(p.regions)):
        ref r2 = p.regions[i]
        if not r2.has_ranges:
            continue
        var rs = RegionSpec(r2.composed_name(), anchor)
        rs.has_rect = True
        rs.x_min = r2.x0
        rs.y_min = r2.y0
        rs.x_max = r2.x1
        rs.y_max = r2.y1
        f.regions.append(rs^)
    return f^
