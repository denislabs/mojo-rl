"""The `.family` and `.task` documents — P1b of `docs/TASK_LAYER_PLAN.md`.

A FAMILY is the compile unit: one base scene, a constant slot table, hence
constant `nq`/`nv`/`ngeom` for every task in it. A TASK is DATA: which slots
are active, where each starts, what the goal is, what the instruction says.

    families/so101_tabletop.family   -> one GPU monomorphisation
    tasks/so101_pick_brick.task      -> no rebuild
    tasks/so101_stack_cubes.task     -> no rebuild

## The format, and why it is not JSON

`key=value` lines with repeating keys, following `data/manifest.mojo` and
checkpoint v2 — the two text formats this tree already reads AND writes.
`io/json.mojo` reads but nothing writes, and the studio must WRITE these.
MJCF is not an option either: `PHYSICS3D_STUDIO_PLAN.md` §3's rule is
"composition is MJCF because MuJoCo is the oracle", and a goal predicate is
precisely what MuJoCo cannot express, so there is no oracle to forfeit.

## ⚠⚠ AN UNKNOWN KEY RAISES. THIS IS A DELIBERATE DIVERGENCE FROM THE MANIFEST.

`data/manifest.mojo` IGNORES unknown keys so that a store written by a newer
build stays readable — right for a format embedded in data files that outlive
the code. It is wrong here. These files are hand-authored, and every key is
load-bearing:

  * a dropped `goal=`   -> a task with no success condition;
  * a dropped `active=` -> a slot silently parked, i.e. a different task;
  * a dropped `init=`   -> an object at its XML pose in every episode.

None of those fail loudly. A typo'd key must therefore be an error, not a
shrug. ⚠ The cost is that a NEWER `.task` cannot be read by an OLDER build —
which is correct, because it means something the older build cannot honour.

## ⚠ WHAT THIS FILE DOES NOT DO

* **It does not parse the goal predicate.** `goal=` is carried as TEXT.
  Predicates are P2 (`predicates.mojo`), and a half-parser here would be a
  second one to keep in step.
* **It does not touch MJCF or the scene.** Composition is P1c
  (`family.mojo`), which CALLS `physics3d/studio`'s composer. §7's dependency
  rule: `tasks/` calls the studio, never reimplements it, and `physics3d`
  never imports `tasks`.
* **It does not validate the park pose against geometry** — see `PARK` below.
"""

from mojo_rl.core.kv import kv_lines, split_on, split_once


comptime SCHEMA_VERSION: Int = 1

# A slot's kind. `free` costs 6 dofs and 7 qpos; `static` costs NEITHER.
#
# ⚠⚠ THE DISTINCTION IS THE WHOLE COST MODEL, and it is measured, not
# stylistic. `docs/TASK_LAYER_IMPLEMENTATION.md` §1.0: on an RTX 5090 at 1024
# lanes, six FREE slots cost 2.74x the bare arm and thirteen is the compile
# ceiling. A STATIC slot adds a body and a geom and no dofs at all, so it pays
# none of that. `docs/TASK_LAYER_PLAN.md` §12 asks whether static slots are
# worth having and answers "probably yes; not needed before P1" — the budget
# measurement upgraded that to "this is the lever", so it is here from day one.
comptime SLOT_FREE: Int = 0
comptime SLOT_STATIC: Int = 1


def slot_kind_from_name(s: String) raises -> Int:
    if s == "free":
        return SLOT_FREE
    if s == "static":
        return SLOT_STATIC
    raise Error(
        "tasks: unknown slot kind '" + s + "' — expected 'free' (a movable"
        " prop, 6 dofs) or 'static' (a fixture, no dofs)"
    )


def slot_kind_name(k: Int) -> String:
    return String("static") if k == SLOT_STATIC else String("free")


struct SlotSpec(Copyable, ImplicitlyCopyable, Movable):
    """`slot=<name>:<kind>:<asset>` — one instantiable object in the family."""

    var name: String
    var kind: Int
    var asset: String

    def __init__(out self, name: String, kind: Int, asset: String):
        self.name = name
        self.kind = kind
        self.asset = asset

    def describe(self) -> String:
        return self.name + ":" + slot_kind_name(self.kind) + ":" + self.asset


struct RegionSpec(Copyable, ImplicitlyCopyable, Movable):
    """`region=<name>:site:<site>[:xmin,ymin,xmax,ymax]` — a placement area.

    ⚠ RELATIVE TO A SITE, WHICH IS WHY IT TRAVELS. A region attached to a
    movable slot's site moves with that slot, so "in the box" stays true after
    the box is picked up. This is LIBERO's `:regions` mechanism and it is the
    piece that makes a symbolic goal land on real geometry.

    With no rectangle the region IS the site's own extent.
    """

    var name: String
    var site: String
    var has_rect: Bool
    var x_min: Float64
    var y_min: Float64
    var x_max: Float64
    var y_max: Float64

    def __init__(out self, name: String, site: String):
        self.name = name
        self.site = site
        self.has_rect = False
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0

    def __init__(
        out self, name: String, site: String,
        x_min: Float64, y_min: Float64, x_max: Float64, y_max: Float64,
    ):
        self.name = name
        self.site = site
        self.has_rect = True
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def describe(self) -> String:
        var s = self.name + ":site:" + self.site
        if self.has_rect:
            s += (
                ":" + String(self.x_min) + "," + String(self.y_min)
                + "," + String(self.x_max) + "," + String(self.y_max)
            )
        return s^


struct InitSpec(Copyable, ImplicitlyCopyable, Movable):
    """`init=<slot>@<region>` — a DISTRIBUTION, not a pose.

    The sampler (P2) draws from the region with rejection. Writing a pose here
    instead would make every episode identical, which is the bug that reads as
    a policy that memorised one placement.
    """

    var slot: String
    var region: String

    def __init__(out self, slot: String, region: String):
        self.slot = slot
        self.region = region

    def describe(self) -> String:
        return self.slot + "@" + self.region


struct FamilySpec(Movable & Deinitable):
    """The compile unit. Every task in the family instantiates EVERY slot."""

    var schema_version: Int
    var name: String
    var base: String
    var horizon: Int
    var control_freq: Int
    var slots: List[SlotSpec]
    var regions: List[RegionSpec]
    var park_x: Float64
    var park_y: Float64
    var park_z: Float64

    def __init__(out self):
        self.schema_version = SCHEMA_VERSION
        self.name = String("")
        self.base = String("")
        self.horizon = 0
        self.control_freq = 0
        self.slots = List[SlotSpec]()
        self.regions = List[RegionSpec]()
        # ⚠⚠ THE DEFAULT IS HIGH AND LATERAL, NOT `(0, 0, -2)`.
        #
        # `docs/TASK_LAYER_PLAN.md` §4.2 writes the park pose as
        # `park=0.0,0.0,-2.0`. Measured against MuJoCo 3.10.0 on the real
        # SO-ARM101 asset, that pose is a FOUR-CONTACT PENETRATION that ejects
        # the body to z=+36.7 within 1.2 s, because the floor is
        # `size="0 0 0.05"` — an INFINITE plane, so there is no "below" it.
        # At four contacts per slot it also overflows a 16-contact budget by
        # the fourth slot, and an overflowed budget DROPS contacts silently.
        #
        # ⚠ THIS FILE CANNOT CHECK THAT — it has no scene. `family.mojo` (P1c)
        # gates it by composing the scene and asserting the parked slots add
        # no contacts at rest, the way `tools/tasks/gen_park_scenes.py`
        # already does for the P0 probe. The default here is only a default.
        self.park_x = 10.0
        self.park_y = 0.0
        self.park_z = 50.0

    def __init__(out self, *, deinit move: Self):
        self.schema_version = move.schema_version
        self.name = move.name^
        self.base = move.base^
        self.horizon = move.horizon
        self.control_freq = move.control_freq
        self.slots = move.slots^
        self.regions = move.regions^
        self.park_x = move.park_x
        self.park_y = move.park_y
        self.park_z = move.park_z

    def slot_index(self, name: String) -> Int:
        """Index of the named slot, or -1. Slot ORDER is the observation
        layout, so this is an identity lookup and not a convenience."""
        for i in range(len(self.slots)):
            if self.slots[i].name == name:
                return i
        return -1

    def region_index(self, name: String) -> Int:
        for i in range(len(self.regions)):
            if self.regions[i].name == name:
                return i
        return -1

    def n_free_slots(self) -> Int:
        """How many slots carry a free joint — the number that costs.

        ⚠ THE ONE TO WATCH. §1.0 of the implementation doc prices this
        directly: 6 free slots is 2.74x the bare arm, 13 is the compile
        ceiling. Static slots do not appear here because they cost no dofs.
        """
        var n = 0
        for i in range(len(self.slots)):
            if self.slots[i].kind == SLOT_FREE:
                n += 1
        return n

    def encode(self) -> String:
        var s = String()
        s += "schema_version=" + String(self.schema_version) + "\n"
        s += "family=" + self.name + "\n"
        s += "base=" + self.base + "\n"
        s += "horizon=" + String(self.horizon) + "\n"
        s += "control_freq=" + String(self.control_freq) + "\n"
        s += (
            "park=" + String(self.park_x) + "," + String(self.park_y)
            + "," + String(self.park_z) + "\n"
        )
        for i in range(len(self.slots)):
            s += "slot=" + self.slots[i].describe() + "\n"
        for i in range(len(self.regions)):
            s += "region=" + self.regions[i].describe() + "\n"
        return s^


struct TaskSpec(Movable & Deinitable):
    """A binding of values into a family. Costs no rebuild."""

    var schema_version: Int
    var name: String
    var family: String
    var language: String
    var active: List[String]
    var inits: List[InitSpec]
    var goal: String
    """The success predicate, as TEXT. Parsed in P2, not here — see the module
    header. Empty is refused by `parse_task`: a task with no goal always
    succeeds, and a policy trained against it learns nothing while every curve
    looks healthy."""

    def __init__(out self):
        self.schema_version = SCHEMA_VERSION
        self.name = String("")
        self.family = String("")
        self.language = String("")
        self.active = List[String]()
        self.inits = List[InitSpec]()
        self.goal = String("")

    def __init__(out self, *, deinit move: Self):
        self.schema_version = move.schema_version
        self.name = move.name^
        self.family = move.family^
        self.language = move.language^
        self.active = move.active^
        self.inits = move.inits^
        self.goal = move.goal^

    def is_active(self, slot: String) -> Bool:
        for i in range(len(self.active)):
            if self.active[i] == slot:
                return True
        return False

    def encode(self) -> String:
        var s = String()
        s += "schema_version=" + String(self.schema_version) + "\n"
        s += "task=" + self.name + "\n"
        s += "family=" + self.family + "\n"
        s += "language=" + self.language + "\n"
        s += "goal=" + self.goal + "\n"
        for i in range(len(self.active)):
            s += "active=" + self.active[i] + "\n"
        for i in range(len(self.inits)):
            s += "init=" + self.inits[i].describe() + "\n"
        return s^


# ═══════════════════════════════════════════════════════════════════════════
# parsing
# ═══════════════════════════════════════════════════════════════════════════


def _check_version(v: Int, what: String) raises:
    if v > SCHEMA_VERSION:
        raise Error(
            what + ": schema_version " + String(v) + " is newer than this"
            " build supports (" + String(SCHEMA_VERSION) + "). Unlike the"
            " data manifest, a task spec REFUSES what it cannot honour — see"
            " the module header."
        )


def _unknown_key(key: String, lineno: Int, what: String, known: String) raises:
    raise Error(
        what + ": unknown key '" + key + "' on line " + String(lineno)
        + ". Known keys are: " + known + ". A task spec refuses unknown keys"
        " rather than ignoring them — a typo'd `goal` is a task that always"
        " succeeds, and nothing downstream would say so."
    )


def parse_slot(spec: String) raises -> SlotSpec:
    """`<name>:<kind>:<asset>`."""
    var parts = split_on(spec, String(":"))
    if len(parts) != 3:
        raise Error(
            "tasks: malformed slot '" + spec + "' — expected"
            " '<name>:<kind>:<asset>', e.g. 'brick:free:props/brick.xml'"
        )
    var name = String(String(parts[0]).strip())
    var kind = slot_kind_from_name(String(String(parts[1]).strip()))
    var asset = String(String(parts[2]).strip())
    if name.byte_length() == 0 or asset.byte_length() == 0:
        raise Error("tasks: slot has an empty name or asset: '" + spec + "'")
    return SlotSpec(name^, kind, asset^)


def parse_region(spec: String) raises -> RegionSpec:
    """`<name>:site:<site>[:xmin,ymin,xmax,ymax]`.

    ⚠ `site` IS SPELLED OUT rather than assumed, so that a later target kind
    (a body, a geom) is an added token and not a format change. Anything else
    raises today instead of being silently read as a site.
    """
    var parts = split_on(spec, String(":"))
    if len(parts) != 3 and len(parts) != 4:
        raise Error(
            "tasks: malformed region '" + spec + "' — expected"
            " '<name>:site:<site>' or '<name>:site:<site>:xmin,ymin,xmax,ymax'"
        )
    var name = String(String(parts[0]).strip())
    var kind = String(String(parts[1]).strip())
    var site = String(String(parts[2]).strip())
    if kind != "site":
        raise Error(
            "tasks: region '" + name + "' targets '" + kind + "'; only 'site'"
            " is supported. A region is site-relative so that it TRAVELS with"
            " a movable slot — see RegionSpec."
        )
    if name.byte_length() == 0 or site.byte_length() == 0:
        raise Error("tasks: region has an empty name or site: '" + spec + "'")
    if len(parts) == 3:
        return RegionSpec(name^, site^)

    var nums = split_on(String(String(parts[3]).strip()), String(","))
    if len(nums) != 4:
        raise Error(
            "tasks: region '" + name + "' rectangle needs exactly four"
            " numbers (xmin,ymin,xmax,ymax), got " + String(len(nums))
        )
    var x0 = Float64(String(String(nums[0]).strip()))
    var y0 = Float64(String(String(nums[1]).strip()))
    var x1 = Float64(String(String(nums[2]).strip()))
    var y1 = Float64(String(String(nums[3]).strip()))
    # ⚠ ORDER IS CHECKED. A reversed rectangle is not an error the sampler can
    # see — it just never accepts a draw, and bounded retries then RAISE with
    # "exhausted", which points at the sampler rather than at this line.
    if x1 <= x0 or y1 <= y0:
        raise Error(
            "tasks: region '" + name + "' has an empty or reversed rectangle"
            " (xmin,ymin must be < xmax,ymax): '" + spec + "'"
        )
    return RegionSpec(name^, site^, x0, y0, x1, y1)


def parse_init(spec: String) raises -> InitSpec:
    """`<slot>@<region>`."""
    var parts = split_once(spec, String("@"))
    if len(parts) != 2:
        raise Error(
            "tasks: malformed init '" + spec + "' — expected '<slot>@<region>',"
            " e.g. 'brick@table'"
        )
    var slot = String(String(parts[0]).strip())
    var region = String(String(parts[1]).strip())
    if slot.byte_length() == 0 or region.byte_length() == 0:
        raise Error("tasks: init has an empty slot or region: '" + spec + "'")
    return InitSpec(slot^, region^)


def parse_family(text: String) raises -> FamilySpec:
    var f = FamilySpec()
    var saw_version = False
    var lines = kv_lines(text, String("family spec"))

    for i in range(len(lines)):
        var key = lines[i].key
        var val = lines[i].value
        if key == "schema_version":
            f.schema_version = Int(val)
            _check_version(f.schema_version, String("family spec"))
            saw_version = True
        elif key == "family":
            f.name = val
        elif key == "base":
            f.base = val
        elif key == "horizon":
            f.horizon = Int(val)
        elif key == "control_freq":
            f.control_freq = Int(val)
        elif key == "park":
            var p = split_on(val, String(","))
            if len(p) != 3:
                raise Error(
                    "family spec: park needs three numbers 'x,y,z', got '"
                    + val + "'"
                )
            f.park_x = Float64(String(String(p[0]).strip()))
            f.park_y = Float64(String(String(p[1]).strip()))
            f.park_z = Float64(String(String(p[2]).strip()))
        elif key == "slot":
            f.slots.append(parse_slot(val))
        elif key == "region":
            f.regions.append(parse_region(val))
        else:
            _unknown_key(
                key, lines[i].lineno, String("family spec"),
                String("schema_version, family, base, horizon, control_freq,"
                       " park, slot, region"),
            )

    if not saw_version:
        raise Error("family spec: no schema_version line")
    if f.name.byte_length() == 0:
        raise Error("family spec: no family= name")
    if f.base.byte_length() == 0:
        raise Error("family spec: no base= scene")
    if f.horizon <= 0:
        raise Error("family spec: horizon must be > 0, got " + String(f.horizon))

    # ⚠ DUPLICATE NAMES ARE REFUSED. Slot ORDER is the observation layout and
    # the instance prefix is the identity, so two slots sharing a name is two
    # different objects addressed by one key — a silent aliasing bug in the
    # scene, in the obs and in every goal that names it.
    for i in range(len(f.slots)):
        for j in range(i + 1, len(f.slots)):
            if f.slots[i].name == f.slots[j].name:
                raise Error(
                    "family spec: duplicate slot name '" + f.slots[i].name + "'"
                )
    for i in range(len(f.regions)):
        for j in range(i + 1, len(f.regions)):
            if f.regions[i].name == f.regions[j].name:
                raise Error(
                    "family spec: duplicate region name '"
                    + f.regions[i].name + "'"
                )
    return f^


def parse_task(text: String) raises -> TaskSpec:
    var t = TaskSpec()
    var saw_version = False
    var lines = kv_lines(text, String("task spec"))

    for i in range(len(lines)):
        var key = lines[i].key
        var val = lines[i].value
        if key == "schema_version":
            t.schema_version = Int(val)
            _check_version(t.schema_version, String("task spec"))
            saw_version = True
        elif key == "task":
            t.name = val
        elif key == "family":
            t.family = val
        elif key == "language":
            t.language = val
        elif key == "goal":
            t.goal = val
        elif key == "active":
            t.active.append(val)
        elif key == "init":
            t.inits.append(parse_init(val))
        else:
            _unknown_key(
                key, lines[i].lineno, String("task spec"),
                String("schema_version, task, family, language, goal, active,"
                       " init"),
            )

    if not saw_version:
        raise Error("task spec: no schema_version line")
    if t.name.byte_length() == 0:
        raise Error("task spec: no task= name")
    if t.family.byte_length() == 0:
        raise Error("task spec: no family= name")
    # ⚠ AN EMPTY GOAL IS REFUSED, not defaulted. A task with no success
    # condition trains against a flat-zero reward and every curve looks
    # healthy while nothing is learned — the same shape as a config wired to
    # the batched env without GPU hooks (`phyics3d_env_config.HAS_GPU_HOOKS`).
    if t.goal.byte_length() == 0:
        raise Error(
            "task spec '" + t.name + "': no goal= predicate. A task with no"
            " goal always succeeds and trains against a flat-zero reward."
        )
    for i in range(len(t.active)):
        for j in range(i + 1, len(t.active)):
            if t.active[i] == t.active[j]:
                raise Error(
                    "task spec: slot '" + t.active[i] + "' listed active twice"
                )
    return t^


def validate_task_against_family(t: TaskSpec, f: FamilySpec) raises:
    """⚠ THIS IS WHAT MAKES THE BUDGET REAL — `TASK_LAYER_PLAN.md` §4.4.

    "A task cannot introduce an object the family did not declare." Without
    this check that rule is a comment: a `.task` naming an unknown slot would
    compose a scene missing it, and the failure would surface as a policy that
    cannot reach something the instruction talks about.

    Checked here, at the SPEC level, because it needs no scene and no MuJoCo —
    so it is available to the studio while a human is typing.
    """
    if t.family != f.name:
        raise Error(
            "task '" + t.name + "' declares family '" + t.family
            + "' but was validated against '" + f.name + "'"
        )
    for i in range(len(t.active)):
        if f.slot_index(t.active[i]) < 0:
            raise Error(
                "task '" + t.name + "': active slot '" + t.active[i]
                + "' is not declared by family '" + f.name + "'. A task binds"
                " values into the family's slot table; it cannot add to it."
                " If it needs a new object, that is a NEW FAMILY and a rebuild."
            )
    for i in range(len(t.inits)):
        var slot = t.inits[i].slot
        var region = t.inits[i].region
        if f.slot_index(slot) < 0:
            raise Error(
                "task '" + t.name + "': init names slot '" + slot
                + "', which family '" + f.name + "' does not declare"
            )
        # ⚠ AN INIT FOR A PARKED SLOT IS AN ERROR, NOT A NO-OP. It reads as an
        # object that should be on the table, and it would be silently parked.
        if not t.is_active(slot):
            raise Error(
                "task '" + t.name + "': init places slot '" + slot + "' but it"
                " is not listed active, so it would be PARKED and the init"
                " ignored. Add 'active=" + slot + "' or drop the init."
            )
        if f.region_index(region) < 0:
            raise Error(
                "task '" + t.name + "': init places '" + slot + "' in region '"
                + region + "', which family '" + f.name + "' does not declare"
            )
    # ⚠ AN ACTIVE SLOT WITH NO INIT is allowed and deliberate: a fixture
    # (`static`) has a pose from the scene and nothing to sample. A FREE slot
    # without an init would start at its XML pose in every episode, which is a
    # real authoring mistake, so that one is refused.
    for i in range(len(t.active)):
        var si = f.slot_index(t.active[i])
        if si < 0 or f.slots[si].kind != SLOT_FREE:
            continue
        var found = False
        for j in range(len(t.inits)):
            if t.inits[j].slot == t.active[i]:
                found = True
        if not found:
            raise Error(
                "task '" + t.name + "': free slot '" + t.active[i] + "' is"
                " active but has no init=, so it would start at its XML pose"
                " in EVERY episode — an identical placement every time, which"
                " reads as a policy that memorised one layout."
            )


def load_family(path: String) raises -> FamilySpec:
    with open(path, "r") as fh:
        return parse_family(fh.read())


def load_task(path: String) raises -> TaskSpec:
    with open(path, "r") as fh:
        return parse_task(fh.read())
