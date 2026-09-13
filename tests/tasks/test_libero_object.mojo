"""THE FIRST UNION FAMILY — eleven props pooled, seven per task, one scene.

    pixi run mojo run -I . tests/tasks/test_libero_object.mojo

`libero_goal` and `libero_spatial` are ten files that declare the SAME props,
so "one family per suite" cost nothing to believe. `libero_object` is the suite
that made it a decision: its ten files name ELEVEN props between them and
SEVEN each, so the family pools all eleven and every `.task` selects its own
with `active=` (`gen_libero_family`'s header states the trade — eleven free
joints in one compile unit, four parked per episode).

## ⚠⚠ AND A UNION MOVES A REGION'S NAME OFF ITS RECTANGLE

Five of the ten files call `(-0.145 -0.265 -0.095 -0.215)`
`target_object_region` and the other five call it `other_object_region_0`,
with the second rectangle taking whichever name is left. Nothing about the
scene changes — the task's own prop simply starts in the other spot. So the
family keys a region by its GEOMETRY and each file resolves its own names
through a `RegionAlias` list (`libero_import.RegionAlias`).

That remap is silent by construction: both names exist in the union, so
resolving by name would find a region, place five tasks' props in the wrong
halves of the floor, and leave every downstream number agreeing with itself.

## WHAT THIS ASSERTS, AND WHY EACH ONE IS HERE

0. **the union is a union** — 12 slots, 11 free, and the per-task `active=`
   sets are PROPER SUBSETS: every prop is used by at least one task and only
   the basket by all ten. If `active=` were still the whole slot table (what
   `translate_task` did before the union) all eleven would appear in all ten
   and checks 1-3 would be testing a different family than the tasks describe.

1. ⚠⚠ **THE `.bddl` IS THE ORACLE FOR WHICH RECTANGLE, NOT THE `.family`.**
   For every prop of every task this reads the rectangle out of LIBERO's own
   file and requires the RESET POSE to be inside it, measured from the arena's
   `workspace` site after forward kinematics. Nothing in the chain under test
   is consulted: not the alias, not the region table, not the region's name.
   `rectangle2xyrange` in `libero/libero/envs/utils.py` is what fixes the
   tuple order — `x_ranges = [r[0], r[2]]`, `y_ranges = [r[1], r[3]]` — so
   (x_min, y_min, x_max, y_max) is LIBERO's reading and not our guess.

2. ⚠ **THE CONTROL: resolving by NAME puts props outside those rectangles.**
   Check 1 alone would pass on a family whose eight rectangles happened to
   cover the floor. This recomputes each placement against the region the
   `.bddl` NAMES (the pre-union rule) and counts the props that then sit
   outside the rectangle LIBERO gives them. A zero here means the suite does
   not actually exercise aliasing and check 1 proves nothing about it.

3. ⚠⚠ **THE START HEIGHT IS LIBERO'S OWN, TO THE PICOMETRE** — every
   placement's z is compared against `mojo_rl/tasks/libero/init_z_libero
   _object.kv`, which `tools/tasks/libero_init_z.py` reads out of the
   `.pruned_init` files the benchmark restores at reset. This is what caught
   the missing `TABLE_Z_OFFSET`: LIBERO routes a floor region to
   `TableRegionSampler`, whose signature carries `z_offset=0.01`, and this
   tree had transcribed the FIXTURE sampler's 0.0. The control is that the
   rule without the centimetre matches NONE of the seventy.

   ⚠ AND "CONTACT-FREE" IS NOT ASSERTED HERE, unlike `test_libero_placement`.
   At LIBERO's own frozen state MuJoCo reports **204 contacts and a 3.8 cm
   penetration** of the ground plane: `bottom_site` is a margin well below the
   collision geometry, so a floor prop starts INSIDE the floor and the five
   settle steps push it out. Requiring zero here would be requiring something
   the benchmark does not do. The count is printed beside each task so a
   change in it is visible, and the four props a task does not name are still
   required to be at the park pose.

4. ⚠⚠ **THE GOAL BOX TRAVELS WITH THE BASKET.** All ten goals are
   `In(<prop>, basket_1_contain_region)` and `basket_1` is a FREE slot — the
   first region in the port attached to something that moves. `RegionSpec`'s
   header claims this ("a region attached to a movable slot's site moves with
   that slot"); here it is measured. The control is what makes it mean
   anything: the prop and the basket are moved TOGETHER, so a region frozen at
   the basket's reset pose reports False, and then the basket alone is moved
   away, which must report False for the opposite reason.

⚠ NO MuJoCo LEG. What MuJoCo settles for this family is the composed scene and
its contact count at rest, and `tools/tasks/check_family.py` already does it
(`nq 86 nv 75`, 0 contacts with `base_qpos` applied). What is new here is the
union and the alias, and MuJoCo has no opinion about either.
"""

from std.os import listdir
from std.os.path import exists

from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.tasks.bddl import parse_bddl
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
    FamilySpec, TABLE_Z_OFFSET,
)
from mojo_rl.tasks.family import scene_path, park_pos
from mojo_rl.tasks.eval import (
    region_sites, region_contact_bodies, HostState, eval_goal,
)
from mojo_rl.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses, BoundGoal,
)
from mojo_rl.tasks.sampler import sample_placements, RegionFrame, SampleReport
from mojo_rl.tasks.reset import free_slot_addresses, reset_slots
from mojo_rl.tasks.libero_object_xml import (
    LIBERO_OBJECT_MAX_CONTACTS, LIBERO_OBJECT_N_FREE_SLOTS,
)


comptime DT = DType.float64
comptime SUITE = "libero_object"
comptime FAMILY = "mojo_rl/tasks/families/libero_object.family"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime PACK = "mojo_rl/tasks/libero/assets"
comptime BDDL_DIR = (
    "references/LIBERO-master/libero/libero/bddl_files/libero_object"
)
comptime WORKSPACE_SITE = "arena_workspace"
comptime SEED: UInt64 = 7
comptime N_TASKS = 10
comptime N_ACTIVE = 8
"""The arena plus seven props. LIBERO's `:objects` gives six distractors and
the target; the basket is one of the seven."""
comptime INIT_Z_KV = "mojo_rl/tasks/libero/init_z_libero_object.kv"
comptime Z_TOL: Float64 = 1.0e-12
"""⚠ NOT EXACT EQUALITY, and not a tuned number either. The reference is a
DECIMAL STRING and `Float64(String)` in this toolchain is up to 1 ULP low on
one token in four thousand, while ours is `site_z + 0.01 - bottom_z` computed
in binary. A picometre is nine orders below the centimetre the control has to
see."""


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def _task_stems() raises -> List[String]:
    """The suite's tasks, sorted — the same order `gen_libero_family` walks,
    so a stem is also its `.bddl` name."""
    var out = List[String]()
    var want = String(SUITE) + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def _bddl_stem(task: String) -> String:
    """`libero_object__pick_up_the_x` -> `pick_up_the_x`."""
    var cut = String(SUITE).byte_length() + 2
    return String(task[byte=cut : task.byte_length()])


def _frozen_z(f: FamilySpec) raises -> List[Float64]:
    """LIBERO's own start height per FAMILY SLOT, from `init_z_<suite>.kv`.

    ⚠ EVERY FREE SLOT MUST BE IN THE FILE. A prop the reference does not name
    would silently compare against nothing."""
    var out = List[Float64](length=len(f.slots), fill=0.0)
    var have = List[Bool](length=len(f.slots), fill=False)
    var text: String
    with open(INIT_Z_KV, "r") as fh:
        text = fh.read()
    for line in text.splitlines():
        var l = String(line).strip()
        if not l.startswith("z="):
            continue
        var body = String(l[byte=2 : l.byte_length()])
        var cut = body.rfind(":")
        if cut < 0:
            raise Error("libero_object: bad line in " + INIT_Z_KV + ": " + l)
        var nm = String(body[byte=0:cut])
        var si = f.slot_index(nm)
        if si < 0:
            raise Error(
                "libero_object: " + INIT_Z_KV + " names '" + nm + "', which"
                " the family has no slot for. Regenerate both."
            )
        out[si] = Float64(String(body[byte = cut + 1 : body.byte_length()]))
        have[si] = True
    for si in range(len(f.slots)):
        if f.slots[si].kind == SLOT_FREE and not have[si]:
            raise Error(
                "libero_object: no frozen z for '" + f.slots[si].name + "' in "
                + INIT_Z_KV + " — run `pixi run libero-init-z`."
            )
    return out^


def _site_id(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if names[i] == want:
            return i
    raise Error(
        "libero_object: the composed scene has no site '" + want + "'. The"
        " floor arena is where every placement region is anchored."
    )


def main() raises:
    print("=" * 74)
    print("The first union family —", SUITE)
    print("=" * 74)
    if not exists(PACK):
        print("  SKIPPED: no LIBERO pack at", PACK)
        print("=== SKIPPED (no pack — this is not a pass) ===")
        return
    if not exists(BDDL_DIR):
        print("  SKIPPED: no LIBERO corpus at", BDDL_DIR)
        print("=== SKIPPED (no corpus — this is not a pass) ===")
        return
    var ta = Tally()

    var f = load_family(FAMILY)
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_OBJECT_MAX_CONTACTS, nmesh_verts=verts
    )
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(
                fmd, max_contacts=LIBERO_OBJECT_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nb = dims.get_nbody()
    var ns = dims.get_nsite()

    var jt = List[Int]()
    var jqn = List[Int]()
    var jvn = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jqn.append(fmd.joints[i].nq)
        jvn.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jqn, jvn)
    var rsites = region_sites(f, fmd.site_names)
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(0.02)
    var ws = _site_id(fmd.site_names, String(WORKSPACE_SITE))

    # ── 0. the union is a union ───────────────────────────────────────────
    print()
    print("--- 0. the slot table is pooled and `active=` selects ---")
    var n_free = 0
    var n_geom = 0
    for si in range(len(f.slots)):
        if f.slots[si].kind != SLOT_FREE:
            continue
        n_free += 1
        if f.slots[si].has_geom:
            n_geom += 1
    print("    ", len(f.slots), "slots,", n_free, "free,", len(f.regions),
          "regions")
    ta.check(
        n_free == LIBERO_OBJECT_N_FREE_SLOTS and n_free == 11,
        String(n_free) + " free slots, and the model def says "
        + String(LIBERO_OBJECT_N_FREE_SLOTS),
    )
    ta.check(n_geom == n_free,
             "every pooled prop carries its asset's placement sites")

    var stems = _task_stems()
    var used = List[Int](length=len(f.slots), fill=0)
    var worst_active = 0
    var n_init_total = 0
    for i in range(len(stems)):
        var t = load_task(TASK_DIR + stems[i] + ".task")
        validate_task_against_family(t, f)
        if len(t.active) != N_ACTIVE:
            worst_active += 1
        n_init_total += len(t.inits)
        for si in range(len(f.slots)):
            if t.is_active(f.slots[si].name):
                used[si] += 1
    var n_all_ten = 0
    var n_never = 0
    for si in range(len(f.slots)):
        if f.slots[si].kind != SLOT_FREE:
            continue
        if used[si] == len(stems):
            n_all_ten += 1
        if used[si] == 0:
            n_never += 1
    print("    ", len(stems), "tasks,", n_init_total, "placements total;",
          n_all_ten, "props in every task,", n_never, "in none")
    ta.check(len(stems) == N_TASKS, String(len(stems)) + " tasks on disk")
    ta.check(worst_active == 0,
             "every task activates exactly " + String(N_ACTIVE)
             + " slots (the arena and seven props)")
    ta.check(n_init_total == N_TASKS * (N_ACTIVE - 1),
             String(n_init_total) + " placements = one per active prop")
    # ⚠ THE ANTI-VACUITY LINE. Only the basket is in all ten; if `active=`
    # were the whole slot table this would be 11 and the union untested.
    ta.check(n_all_ten == 1 and n_never == 0,
             "the active sets are PROPER subsets — one prop (the basket) in"
             " all ten, none unused")

    # ── 1-3. the bddl's rectangle, its control, and the contacts ──────────
    print()
    print("--- 1-3. LIBERO's own rectangle for every prop of every task ---")
    print("    task                              props  worst inside  ncon  parked")
    # `worst_margin`: how far INSIDE its rectangle the tightest placement sits.
    var worst_margin = 0.0
    var n_outside = 0
    var n_checked = 0
    # `by_name_outside`: the control — placements that would fall outside if
    # the region were resolved by the `.bddl`'s NAME instead of its geometry.
    var by_name_outside = 0
    var n_aliased = 0
    var bad_con = 0
    var max_con = 0
    var bad_park = 0
    var worst_z = 0.0
    var n_z_checked = 0
    var n_z_bad = 0
    # `n_z_no_offset`: placements the PRE-FIX rule (no `TABLE_Z_OFFSET`) would
    # have got right. The control — it must be zero.
    var n_z_no_offset = 0
    var frozen_z = _frozen_z(f)
    for ti in range(len(stems)):
        var t = load_task(TASK_DIR + stems[ti] + ".task")
        var p = parse_bddl(_read(BDDL_DIR + "/" + _bddl_stem(stems[ti])
                                 + ".bddl"))

        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](0)
        for i in range(nv):
            d.qvel.data[i] = Scalar[DT](0)
        for i in range(len(f.base_qpos)):
            d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        var frames = List[RegionFrame]()
        for i in range(len(f.regions)):
            var s2 = rsites[i]
            frames.append(RegionFrame(
                Float64(d.site_xpos.data[s2 * 3]),
                Float64(d.site_xpos.data[s2 * 3 + 1]),
                Float64(d.site_xpos.data[s2 * 3 + 2]),
            ))
        var rep = SampleReport()
        var placed = sample_placements(t, f, frames, radii, SEED, ti, rep)
        var qpos = List[Float64]()
        for i in range(nq):
            qpos.append(Float64(d.qpos.data[i]))
        var qvel = List[Float64](length=nv, fill=0.0)
        reset_slots(t, f, placed, addrs, qpos, qvel)
        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](qpos[i])
        for i in range(nv):
            d.qvel.data[i] = Scalar[DT](qvel[i])
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        detect_contacts["cpu", DT, DynDims, 1](d, m)
        var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        if ncon != 0:
            bad_con += 1
        if ncon > max_con:
            max_con = ncon

        # ⚠ THE FRAME IS THE ARENA'S OWN WORKSPACE SITE, read after FK. Every
        # floor region in the family anchors to it, and using it directly
        # keeps this comparison clear of the region table entirely.
        var wx = Float64(d.site_xpos.data[ws * 3])
        var wy = Float64(d.site_xpos.data[ws * 3 + 1])

        var w = 1.0e9
        for k in range(len(p.init)):
            ref a = p.init[k]
            if a.pred != "On" or len(a.args) != 2:
                continue
            var prop = String(a.args[0])
            var ri = p.region_index(String(a.args[1]))
            if ri < 0:
                continue
            if not p.regions[ri].has_ranges:
                continue
            var si = f.slot_index(prop)
            if si < 0:
                raise Error(
                    "libero_object: the bddl places '" + prop + "' and the"
                    " family has no such slot — the union is stale."
                )
            var pj = -1
            for q in range(len(placed)):
                if placed[q].slot == si:
                    pj = q
            if pj < 0:
                raise Error(
                    "libero_object: '" + prop + "' is placed by "
                    + stems[ti] + "'s bddl and was not placed by the"
                    " sampler. Its `init=` is missing or inactive."
                )
            # ⚠ THE HEIGHT, AGAINST LIBERO'S FROZEN STATE — not against the
            # sampler's own formula, which would be the blind shape.
            var want_z = frozen_z[si]
            var ez = abs(placed[pj].z - want_z)
            n_z_checked += 1
            if ez > Z_TOL:
                n_z_bad += 1
            if ez > worst_z:
                worst_z = ez
            if abs(placed[pj].z - TABLE_Z_OFFSET - want_z) <= Z_TOL:
                n_z_no_offset += 1

            var dx = placed[pj].x - wx
            var dy = placed[pj].y - wy
            ref br = p.regions[ri]
            var inside = (
                dx >= br.x0 and dx <= br.x1 and dy >= br.y0 and dy <= br.y1
            )
            n_checked += 1
            if not inside:
                n_outside += 1
            else:
                var mx = dx - br.x0
                if br.x1 - dx < mx:
                    mx = br.x1 - dx
                if dy - br.y0 < mx:
                    mx = dy - br.y0
                if br.y1 - dy < mx:
                    mx = br.y1 - dy
                if mx < w:
                    w = mx

            # ── the control: the same prop against the region NAMED ──────
            #
            # The family keeps the first name it saw for each rectangle, so
            # `region_index(<bddl name>)` is the pre-union rule. Where the
            # suite swapped the two names, that rectangle is the OTHER one and
            # the prop is outside it.
            var ni = f.region_index(String(a.args[1]))
            if ni >= 0:
                ref nr = f.regions[ni]
                if (
                    nr.x_min != br.x0 or nr.y_min != br.y0
                    or nr.x_max != br.x1 or nr.y_max != br.y1
                ):
                    n_aliased += 1
                    if not (
                        dx >= nr.x_min and dx <= nr.x_max
                        and dy >= nr.y_min and dy <= nr.y_max
                    ):
                        by_name_outside += 1
        if w < worst_margin or worst_margin == 0.0:
            worst_margin = w

        # the four props this task does not name sit at the park pose
        var parked = 0
        for si in range(len(f.slots)):
            if f.slots[si].kind != SLOT_FREE:
                continue
            if t.is_active(f.slots[si].name):
                continue
            var pk = park_pos(f, si)
            var qa = addrs[si].qadr
            var off = (
                abs(qpos[qa] - pk[0]) + abs(qpos[qa + 1] - pk[1])
                + abs(qpos[qa + 2] - pk[2])
            )
            if off > 1.0e-12:
                bad_park += 1
            else:
                parked += 1

        var pad = String(stems[ti])
        while pad.byte_length() < 34:
            pad += " "
        print("    " + pad + String(len(placed)) + "      " + String(w)
              + "   " + String(ncon) + "     " + String(parked))

    print("    ", n_checked, "placements checked against the bddl;",
          n_outside, "outside their rectangle")
    ta.check(n_checked == N_TASKS * (N_ACTIVE - 1),
             String(n_checked) + " placements read out of LIBERO's own files")
    ta.check(n_outside == 0,
             "every prop starts inside the rectangle ITS OWN bddl gives it"
             " (tightest margin " + String(worst_margin) + " m)")
    print("    ", n_aliased, "placements name a region the union holds a"
          " DIFFERENT rectangle for;", by_name_outside, "of them would land"
          " outside it")
    ta.check(n_aliased > 0,
             "the suite actually exercises aliasing — " + String(n_aliased)
             + " placements whose bddl name is not the family's")
    ta.check(by_name_outside > 0,
             "THE CONTROL: resolving by name puts " + String(by_name_outside)
             + " props outside the rectangle LIBERO gives them")
    print("     contacts at reset: max", max_con, "of a",
          LIBERO_OBJECT_MAX_CONTACTS, "budget;", bad_con, "tasks nonzero."
          " LIBERO's OWN frozen state gives MuJoCo 204 and a 3.8 cm"
          " penetration — see the header.")
    ta.check(bad_park == 0,
             "every prop a task does not name is at the family's park pose")
    print("     worst |z - LIBERO's frozen z| =", worst_z, "m over",
          n_z_checked, "placements;", n_z_bad, "outside", Z_TOL)
    ta.check(n_z_checked == N_TASKS * (N_ACTIVE - 1) and n_z_bad == 0,
             "every placement starts at the height LIBERO's own"
             " `.pruned_init` holds for that prop")
    ta.check(n_z_no_offset == 0,
             "THE CONTROL: the rule WITHOUT `TABLE_Z_OFFSET` matches "
             + String(n_z_no_offset) + " of " + String(n_z_checked)
             + " — the centimetre is not decoration")

    # ── 4. the goal box travels with the basket ───────────────────────────
    print()
    print("--- 4. `In(prop, basket_1_contain_region)` when the basket moves ---")
    var nqs = List[Int]()
    for i in range(len(fmd.joints)):
        nqs.append(fmd.joints[i].nq)
    var jadr = joint_qpos_addresses(nqs)
    var site_body = List[Int]()
    var site_quat = List[Float64]()
    for i in range(len(fmd.sites)):
        site_body.append(fmd.sites[i].body_id)
        site_quat.append(fmd.sites[i].quat_x)
        site_quat.append(fmd.sites[i].quat_y)
        site_quat.append(fmd.sites[i].quat_z)
        site_quat.append(fmd.sites[i].quat_w)
    var body_parent = List[Int]()
    body_parent.append(-1)
    for i in range(len(fmd.bodies)):
        body_parent.append(fmd.bodies[i].parent)
    var rcontact = region_contact_bodies(f, fmd.body_names)

    var t0 = load_task(TASK_DIR + stems[0] + ".task")
    var g = bind_goal(
        parse_goal(t0.goal), f, fmd.body_names, fmd.site_names,
        fmd.joint_names, jadr,
    )
    require_tier_a(g, t0.name)
    var prop = String("")
    for k in range(len(t0.active)):
        var nm = String(t0.active[k])
        if t0.goal.find(nm + ",") >= 0:
            prop = nm^
    if prop.byte_length() == 0:
        raise Error("libero_object: could not read the goal's prop from "
                    + t0.goal)
    var bi = f.slot_index(String("basket_1"))
    var pi = f.slot_index(prop)
    var b_adr = addrs[bi].qadr
    var p_adr = addrs[pi].qadr

    # start from a clean reset, then LIFT the prop into the basket
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    var frames0 = List[RegionFrame]()
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    for i in range(len(f.regions)):
        var s2 = rsites[i]
        frames0.append(RegionFrame(
            Float64(d.site_xpos.data[s2 * 3]),
            Float64(d.site_xpos.data[s2 * 3 + 1]),
            Float64(d.site_xpos.data[s2 * 3 + 2]),
        ))
    var rep0 = SampleReport()
    var placed0 = sample_placements(t0, f, frames0, radii, SEED, 0, rep0)
    var qp = List[Float64]()
    for i in range(nq):
        qp.append(Float64(d.qpos.data[i]))
    var qv = List[Float64](length=nv, fill=0.0)
    reset_slots(t0, f, placed0, addrs, qp, qv)

    var verdicts = List[Bool]()
    var labels = List[String]()
    # (a) the prop on the floor, the basket where the reset put it
    verdicts.append(_holds(g, f, d, m, qp, nq, nb, ns, rsites, rcontact,
                           site_body, site_quat, body_parent))
    labels.append("the prop on the floor")
    # (b) the prop moved into the basket
    var bx = qp[b_adr]
    var by = qp[b_adr + 1]
    var bz = qp[b_adr + 2]
    qp[p_adr] = bx
    qp[p_adr + 1] = by
    qp[p_adr + 2] = bz + 0.07
    verdicts.append(_holds(g, f, d, m, qp, nq, nb, ns, rsites, rcontact,
                           site_body, site_quat, body_parent))
    labels.append("the prop lifted into the basket")
    # (c) BOTH moved together — the region has to travel
    var shift = 0.30
    qp[b_adr] = bx + shift
    qp[p_adr] = bx + shift
    verdicts.append(_holds(g, f, d, m, qp, nq, nb, ns, rsites, rcontact,
                           site_body, site_quat, body_parent))
    labels.append("both moved 0.30 m together")
    # (d) the basket alone moved away
    qp[b_adr] = bx + shift + shift
    verdicts.append(_holds(g, f, d, m, qp, nq, nb, ns, rsites, rcontact,
                           site_body, site_quat, body_parent))
    labels.append("the basket moved on alone")
    for i in range(len(verdicts)):
        print("     ", labels[i], "->", verdicts[i])
    ta.check(not verdicts[0], "not in the basket at reset")
    ta.check(verdicts[1], "in the basket when lifted into it")
    ta.check(
        verdicts[2],
        "STILL in it after both move 0.30 m — the region travels with the"
        " free slot it is attached to",
    )
    ta.check(
        not verdicts[3],
        "THE CONTROL: false once the basket alone moves on, so check (c) is"
        " not a predicate that has stopped reading the basket",
    )

    print()
    print("  checks:", ta.checks, " failures:", ta.failures)
    if ta.failures != 0:
        raise Error(String(ta.failures) + " check(s) failed")
    print("=== LIBERO_OBJECT OK ===")


def _read(path: String) raises -> String:
    with open(path, "r") as fh:
        return fh.read()


def _holds(
    g: BoundGoal,
    f: FamilySpec,
    mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims],
    qpos: List[Float64],
    nq: Int,
    nb: Int,
    ns: Int,
    rsites: List[Int],
    rcontact: List[Int],
    site_body: List[Int],
    site_quat: List[Float64],
    body_parent: List[Int],
) raises -> Bool:
    """`qpos` -> FK -> contacts -> the goal. ⚠ CONTACTS TOO: `In` on a box
    region is site-geometry only (`SiteObjectState.check_contact` is
    unconditionally True in LIBERO), but the wide `eval_goal` refuses to read
    a list it was not given, so the state is built complete."""
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](qpos[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    detect_contacts["cpu", DT, DynDims, 1](d, m)
    var xp = List[Float64]()
    for i in range(nb * 3):
        xp.append(Float64(d.xpos.data[i]))
    var xq = List[Float64]()
    for i in range(nb * 4):
        xq.append(Float64(d.xquat.data[i]))
    var sp = List[Float64]()
    for i in range(ns * 3):
        sp.append(Float64(d.site_xpos.data[i]))
    var st = HostState(xp^, xq^, sp^)
    for i in range(nq):
        st.qpos.append(qpos[i])
    st.site_body = site_body.copy()
    st.site_quat = site_quat.copy()
    st.body_parent = body_parent.copy()
    var n = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    if n > LIBERO_OBJECT_MAX_CONTACTS:
        n = LIBERO_OBJECT_MAX_CONTACTS
    st.ncon = n
    for k in range(n):
        var base = k * CONTACT_SIZE
        st.con_a.append(Int(d.contacts.data[base + CONTACT_IDX_BODY_A]))
        st.con_b.append(Int(d.contacts.data[base + CONTACT_IDX_BODY_B]))
    return eval_goal(g, f, st, rsites, rcontact)
