"""LIBERO's object registry, read from `noeira/tasks/libero/categories.kv`.

    var t = load_libero_table()                 # the shipped table
    var c = t.category("akita_black_bowl")      # asset, rotation, articulation
    var p = t.problem("LIBERO_Tabletop_Manipulation")   # scene, offsets

L1 of `docs/LIBERO_PORT_ASSESSMENT_2026_09_13.md`. The `.bddl` names a
CATEGORY and LIBERO resolves it through a Python registry
(`envs/objects/*.py`); this is that registry as data, so `libero_import` can
emit a real asset path instead of `TODO:<category>`.

## ⚠⚠ THE ARTICULATION DIRECTION IS A FIELD, NOT AN INFERENCE

`open=lt:-0.14` carries the comparison LIBERO's `is_open` makes, verbatim.
`TASK_LAYER_IMPLEMENTATION.md` §6.2 recorded why: `WoodenCabinet` opens
NEGATIVE (`qpos < -0.14`) and `ShortCabinet` opens POSITIVE (`qpos > 0.10`),
so a reader that derived the direction from the sign of a range would be
exactly backwards on one of them. The table states both; this module only
evaluates what it is given.

## What is refused

* an unknown key, an unknown `kind=`, an unknown comparison, a malformed
  `rotation=` — at parse time, by name;
* an `asset` record with no `asset=`, or a `workspace` record WITH one;
* a duplicate `category=` or `problem=`;
* a lookup of a category the table does not carry — RAISES rather than
  returning a default, because a default here is a scene missing an object.
"""

from noeira.core.kv import kv_lines, split_on, split_once


comptime DEFAULT_TABLE_PATH: String = "noeira/tasks/libero/categories.kv"

comptime KIND_ASSET: Int = 0
comptime KIND_WORKSPACE: Int = 1

# comparison ops for the articulation thresholds
# ⚠ THE COMPARISON CODES LIVE IN `predicates.mojo` — they are the wire
# format of a `Joint` term's `b` — and are re-exported here so the table's
# readers keep their import. One definition, two names for it.
from .predicates import CMP_NONE, CMP_LT, CMP_LE, CMP_GT, CMP_GE, cmp_from_name, cmp_name


def threshold_holds(op: Int, qpos: Float64, thr: Float64) -> Bool:
    """`qpos <op> thr`. The ONE place the four comparisons are spelled out."""
    if op == CMP_LT:
        return qpos < thr
    if op == CMP_LE:
        return qpos <= thr
    if op == CMP_GT:
        return qpos > thr
    if op == CMP_GE:
        return qpos >= thr
    return False


struct JointRange(Copyable, ImplicitlyCopyable, Movable):
    """`[lo, hi]`, uniformly sampled. `present()` is False for an absent row.

    LIBERO's `default_open_ranges` / `default_close_ranges` /
    `default_turnon_ranges` / `default_turnoff_ranges`, verbatim."""

    var lo: Float64
    var hi: Float64
    var has: Bool

    def __init__(out self):
        self.lo = 0.0
        self.hi = 0.0
        self.has = False

    def __init__(out self, lo: Float64, hi: Float64):
        self.lo = lo
        self.hi = hi
        self.has = True

    def present(self) -> Bool:
        return self.has

    def describe(self) -> String:
        if not self.has:
            return String("-")
        return String(self.lo) + ".." + String(self.hi)


struct Threshold(Copyable, ImplicitlyCopyable, Movable):
    """One `open=lt:-0.14` line: a comparison and a bound."""

    var op: Int
    var thr: Float64

    def __init__(out self):
        self.op = CMP_NONE
        self.thr = 0.0

    def __init__(out self, op: Int, thr: Float64):
        self.op = op
        self.thr = thr

    def present(self) -> Bool:
        return self.op != CMP_NONE

    def holds(self, qpos: Float64) -> Bool:
        return threshold_holds(self.op, qpos, self.thr)

    def describe(self) -> String:
        return cmp_name(self.op) + ":" + String(self.thr)


struct LiberoCategory(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var kind: Int
    var asset: String
    """Relative to the pack root (LIBERO's `assets/`). Empty for a workspace."""
    var rot_lo: Float64
    var rot_hi: Float64
    var axis: String
    var open: Threshold
    var close: Threshold
    var on: Threshold
    var off: Threshold

    var open_range: JointRange
    var close_range: JointRange
    var on_range: JointRange
    var off_range: JointRange
    """LIBERO's `default_*_ranges`, the list ITSELF and not the threshold.

    ⚠⚠ THE THRESHOLD AND THE RANGE ANSWER DIFFERENT QUESTIONS, and only one of
    them can be derived from the other. `open=lt:-0.14` is `is_open` — a
    PREDICATE, and it is `max(default_open_ranges)`. `open_range=-0.16,-0.14`
    is where an episode STARTS the joint: `bddl_base_domain._reset_internal`
    builds an `OpenCloseSampler` and calls `np.random.uniform` over it. Going
    range -> threshold is a max; going threshold -> range is a guess, because
    `lt:-0.14` says nothing whatever about -0.16."""

    def __init__(out self, name: String):
        self.name = name
        self.kind = KIND_ASSET
        self.asset = String("")
        self.rot_lo = 0.0
        self.rot_hi = 0.0
        self.axis = String("z")
        self.open = Threshold()
        self.close = Threshold()
        self.on = Threshold()
        self.off = Threshold()
        self.open_range = JointRange()
        self.close_range = JointRange()
        self.on_range = JointRange()
        self.off_range = JointRange()

    def is_workspace(self) -> Bool:
        return self.kind == KIND_WORKSPACE

    def is_articulated(self) -> Bool:
        return (
            self.open.present() or self.close.present()
            or self.on.present() or self.off.present()
        )


struct LiberoProblem(Copyable, Movable):
    var name: String
    var scene: String
    """Arena XML, relative to the pack root."""
    var workspace: String
    """The category that IS the arena (`table`, `floor`, ...)."""
    var off_x: Float64
    var off_y: Float64
    var off_z: Float64
    """`workspace_offset` — where the region rects are measured from."""
    var z_offset: Float64
    """`self.z_offset` of the problem class — a fixture sits at
    `z_offset + off_z - bottom_site_z`."""
    var zone_z: Float64
    """World z of a table target zone's site (`zone_z=` in the table, L3)."""
    var has_zone_z: Bool
    var floor_texture: String
    var wall_texture: String
    var base_x: Float64
    var base_y: Float64
    var base_z: Float64
    """`base_pos=` — the Panda's root body, from `set_base_xpos`."""
    var has_table: Bool
    var table_sx: Float64
    var table_sy: Float64
    var table_sz: Float64
    """`table_size=` — FULL size, present only where LIBERO re-poses the
    table (`configure_location`)."""
    var table_friction: List[Float64]
    var cameras: List[String]
    """`camera=<name>:x,y,z:qw,qx,qy,qz` lines, verbatim; the arena generator
    parses them."""
    var robot: String
    """`robot=mounted|on_the_ground` — MountedPanda (RethinkMount below the
    arm) or OnTheGroundPanda (no mount). Which vendored XML the family's
    base is."""

    def __init__(out self, name: String):
        self.name = name
        self.scene = String("")
        self.workspace = String("")
        self.off_x = 0.0
        self.off_y = 0.0
        self.off_z = 0.0
        self.z_offset = 0.0
        self.zone_z = 0.0
        self.has_zone_z = False
        self.floor_texture = String("")
        self.wall_texture = String("")
        self.base_x = 0.0
        self.base_y = 0.0
        self.base_z = 0.0
        self.has_table = False
        self.table_sx = 0.0
        self.table_sy = 0.0
        self.table_sz = 0.0
        self.table_friction = List[Float64]()
        self.cameras = List[String]()
        self.robot = String("")


struct LiberoTable(Movable & Deinitable):
    var categories: List[LiberoCategory]
    var problems: List[LiberoProblem]

    def __init__(out self):
        self.categories = List[LiberoCategory]()
        self.problems = List[LiberoProblem]()

    def __init__(out self, *, deinit move: Self):
        self.categories = move.categories^
        self.problems = move.problems^

    def category_index(self, name: String) -> Int:
        for i in range(len(self.categories)):
            if self.categories[i].name == name:
                return i
        return -1

    def has_category(self, name: String) -> Bool:
        return self.category_index(name) >= 0

    def category(self, name: String) raises -> LiberoCategory:
        var i = self.category_index(name)
        if i < 0:
            raise Error(
                "libero table: no category '" + name + "'. Add it to "
                + DEFAULT_TABLE_PATH + " from LIBERO's envs/objects/*.py —"
                " a default here would be a scene missing an object."
            )
        return self.categories[i]

    def problem(self, name: String) raises -> LiberoProblem:
        for i in range(len(self.problems)):
            if self.problems[i].name == name:
                return self.problems[i].copy()
        raise Error(
            "libero table: no problem '" + name + "'. The five in the corpus"
            " are LIBERO_{Tabletop,Kitchen_Tabletop,Living_Room_Tabletop,"
            "Study_Tabletop,Floor}_Manipulation."
        )


def _parse_threshold(val: String, what: String) raises -> Threshold:
    var parts = split_once(val, String(":"))
    if len(parts) != 2:
        raise Error(
            "libero table: " + what + " needs '<cmp>:<number>', got '" + val
            + "'"
        )
    return Threshold(
        cmp_from_name(String(String(parts[0]).strip())),
        Float64(String(String(parts[1]).strip())),
    )


def _parse_range(val: String, what: String) raises -> JointRange:
    """`lo,hi` — one of LIBERO's `default_*_ranges` lists.

    ⚠ REFUSES `hi < lo`, which `OpenCloseSampler.__init__` asserts too
    (`joint_ranges[0] <= joint_ranges[1]`). A reversed pair makes
    `np.random.uniform(low=hi, high=lo)` draw outside the interval on every
    call, which is a scene that looks sampled and is not."""
    var n = split_on(val, String(","))
    if len(n) != 2:
        raise Error(
            "libero table: " + what + " needs 'lo,hi', got '" + val + "'"
        )
    var lo = Float64(String(String(n[0]).strip()))
    var hi = Float64(String(String(n[1]).strip()))
    if hi < lo:
        raise Error(
            "libero table: " + what + " is '" + val + "' — hi < lo. LIBERO's"
            " OpenCloseSampler asserts joint_ranges[0] <= joint_ranges[1]."
        )
    return JointRange(lo, hi)


def _parse_three(val: String, what: String) raises -> List[Float64]:
    var n = split_on(val, String(","))
    if len(n) != 3:
        raise Error(
            "libero table: " + what + " needs 'x,y,z', got '" + val + "'"
        )
    var out = List[Float64]()
    for i in range(3):
        out.append(Float64(String(String(n[i]).strip())))
    return out^


def _finish_category(mut t: LiberoTable, var c: LiberoCategory) raises:
    if c.kind == KIND_WORKSPACE and c.asset.byte_length() > 0:
        raise Error(
            "libero table: workspace category '" + c.name + "' carries an"
            " asset=. A workspace is the base scene, not a slot."
        )
    if c.kind == KIND_ASSET and c.asset.byte_length() == 0:
        raise Error(
            "libero table: category '" + c.name + "' has no asset=."
        )
    if t.has_category(c.name):
        raise Error("libero table: duplicate category '" + c.name + "'")
    t.categories.append(c^)


def _finish_problem(mut t: LiberoTable, var p: LiberoProblem) raises:
    if p.scene.byte_length() == 0 or p.workspace.byte_length() == 0:
        raise Error(
            "libero table: problem '" + p.name + "' needs scene= and"
            " workspace="
        )
    if p.robot.byte_length() == 0:
        raise Error(
            "libero table: problem '" + p.name + "' needs robot=mounted or"
            " robot=on_the_ground — the two Pandas stand at different heights"
        )
    for i in range(len(t.problems)):
        if t.problems[i].name == p.name:
            raise Error("libero table: duplicate problem '" + p.name + "'")
    t.problems.append(p^)


def parse_libero_table(text: String) raises -> LiberoTable:
    """The whole table. A record starts at `category=` or `problem=` and every
    key after it belongs to that record, the `assets.kv` idiom."""
    var t = LiberoTable()
    var lines = kv_lines(text, String("libero table"))
    var saw_version = False
    var in_cat = False
    var in_prob = False
    var cat = LiberoCategory(String(""))
    var prob = LiberoProblem(String(""))
    for i in range(len(lines)):
        var key = lines[i].key
        var val = lines[i].value
        if key == "schema_version":
            if val != "1":
                raise Error("libero table: schema_version must be 1")
            saw_version = True
        elif key == "category" or key == "problem":
            if in_cat:
                _finish_category(t, cat.copy())
            if in_prob:
                _finish_problem(t, prob.copy())
            in_cat = key == "category"
            in_prob = key == "problem"
            if in_cat:
                cat = LiberoCategory(val)
            else:
                prob = LiberoProblem(val)
        elif in_cat:
            if key == "kind":
                if val == "asset":
                    cat.kind = KIND_ASSET
                elif val == "workspace":
                    cat.kind = KIND_WORKSPACE
                else:
                    raise Error(
                        "libero table: category '" + cat.name + "' has unknown"
                        " kind '" + val + "'. Known: asset, workspace."
                    )
            elif key == "asset":
                cat.asset = val
            elif key == "rotation":
                var r = split_on(val, String(","))
                if len(r) != 2:
                    raise Error(
                        "libero table: category '" + cat.name + "' rotation"
                        " needs 'lo,hi', got '" + val + "'"
                    )
                cat.rot_lo = Float64(String(String(r[0]).strip()))
                cat.rot_hi = Float64(String(String(r[1]).strip()))
            elif key == "axis":
                if val != "x" and val != "y" and val != "z":
                    raise Error(
                        "libero table: category '" + cat.name + "' axis must"
                        " be x, y or z, got '" + val + "'"
                    )
                cat.axis = val
            elif key == "open":
                cat.open = _parse_threshold(val, String("open"))
            elif key == "close":
                cat.close = _parse_threshold(val, String("close"))
            elif key == "on":
                cat.on = _parse_threshold(val, String("on"))
            elif key == "off":
                cat.off = _parse_threshold(val, String("off"))
            elif key == "open_range":
                cat.open_range = _parse_range(val, String("open_range"))
            elif key == "close_range":
                cat.close_range = _parse_range(val, String("close_range"))
            elif key == "on_range":
                cat.on_range = _parse_range(val, String("on_range"))
            elif key == "off_range":
                cat.off_range = _parse_range(val, String("off_range"))
            else:
                raise Error(
                    "libero table: unknown key '" + key + "' at line "
                    + String(lines[i].lineno) + " inside category '"
                    + cat.name + "'. Known: kind, asset, rotation, axis,"
                    " open, close, on, off, open_range, close_range,"
                    " on_range, off_range."
                )
        elif in_prob:
            if key == "scene":
                prob.scene = val
            elif key == "workspace":
                prob.workspace = val
            elif key == "workspace_offset":
                var o = _parse_three(val, String("workspace_offset"))
                prob.off_x = o[0]
                prob.off_y = o[1]
                prob.off_z = o[2]
            elif key == "z_offset":
                prob.z_offset = Float64(val)
            elif key == "zone_z":
                prob.zone_z = Float64(val)
                prob.has_zone_z = True
            elif key == "floor_texture":
                prob.floor_texture = val
            elif key == "wall_texture":
                prob.wall_texture = val
            elif key == "base_pos":
                var b = _parse_three(val, String("base_pos"))
                prob.base_x = b[0]
                prob.base_y = b[1]
                prob.base_z = b[2]
            elif key == "table_size":
                var ts = _parse_three(val, String("table_size"))
                prob.has_table = True
                prob.table_sx = ts[0]
                prob.table_sy = ts[1]
                prob.table_sz = ts[2]
            elif key == "table_friction":
                prob.table_friction = _parse_three(val, String("table_friction"))
            elif key == "robot":
                if val != "mounted" and val != "on_the_ground":
                    raise Error(
                        "libero table: problem '" + prob.name + "' robot must"
                        " be mounted or on_the_ground, got '" + val + "'"
                    )
                prob.robot = val
            elif key == "camera":
                if len(split_on(val, String(":"))) != 3:
                    raise Error(
                        "libero table: camera needs '<name>:x,y,z:qw,qx,qy,qz',"
                        " got '" + val + "'"
                    )
                prob.cameras.append(val)
            else:
                raise Error(
                    "libero table: unknown key '" + key + "' at line "
                    + String(lines[i].lineno) + " inside problem '"
                    + prob.name + "'. Known: scene, workspace,"
                    " workspace_offset, z_offset, zone_z, floor_texture,"
                    " wall_texture, base_pos, table_size, table_friction,"
                    " camera, robot."
                )
        else:
            raise Error(
                "libero table: key '" + key + "' at line "
                + String(lines[i].lineno) + " before any category= or"
                " problem= record"
            )
    if in_cat:
        _finish_category(t, cat.copy())
    if in_prob:
        _finish_problem(t, prob.copy())
    if not saw_version:
        raise Error("libero table: no schema_version line")
    # ⚠ EVERY PROBLEM'S WORKSPACE MUST BE A DECLARED WORKSPACE CATEGORY, or a
    # fixture line `(main_table - table)` would become a slot with no asset.
    for i in range(len(t.problems)):
        var wi = t.category_index(t.problems[i].workspace)
        if wi < 0 or t.categories[wi].kind != KIND_WORKSPACE:
            raise Error(
                "libero table: problem '" + t.problems[i].name + "' names"
                " workspace '" + t.problems[i].workspace + "', which is not a"
                " kind=workspace category"
            )
    return t^


def load_libero_table(path: String = DEFAULT_TABLE_PATH) raises -> LiberoTable:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return parse_libero_table(text)
