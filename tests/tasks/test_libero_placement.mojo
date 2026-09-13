"""WHERE A RESET PUTS THINGS — the asset's numbers, and a contact-free scene.

    pixi run mojo run -I . tests/tasks/test_libero_placement.mojo

`init=` used to place an object at `region_site_z + radius`, where `radius` was
a number the CALLER invented — `0.02` in five drivers, a config constant in
four. That one number stood for two different physical quantities: the distance
at which two objects reject each other, and the height at which one rests.

robosuite keeps them apart, and every one of LIBERO's 93 object assets declares
all three sites it needs:

    bottom_site             z   -> the resting height  (`-bottom_offset[-1]`)
    top_site                z   -> the stacking height (`top_offset[-1]`)
    horizontal_radius_site  |xy| -> the rejection radius

MEASURED across the pack: 10 distinct triples, the radius spanning 0.005 to 0.3.
So the constant was wrong by up to 15x, and `akita_black_bowl`'s `bottom_site`
at -0.06 against a hard-coded 0.02 placed every bowl 4 cm too low.

## WHAT IT ASSERTS

1. **the height is the asset's, to the bit** — every placed slot's z equals
   `region_site_z - slot.bottom_z`, recomputed here from the family's own
   `slot_geom=` numbers and the site position after FK.
2. **every `libero_spatial` task resets CONTACT-FREE.** This is the physical
   claim, and it is the one that failed before: on the stove's `cook_region` —
   whose site sits at the vertical CENTRE of a 4 cm base box — the old rule
   gave 56 contacts, and inside the cabinet's top drawer 64, which is the
   contact CAP (so contacts were also being dropped).
3. ⚠⚠ **the control: at the old height the same scene is NOT contact-free.**
   Without this, check 2 passes on a gate that is not looking at contacts at
   all. One object is lowered back to `site_z + 0.02` and the contacts must
   appear.
4. **`jinit=` draws, and draws differently per lane.** `(Open X)` in a `.bddl`
   `:init` is `np.random.uniform` over the class's `default_open_ranges`, so a
   drawer starts somewhere in [-0.16, -0.14] and not at its threshold. Every
   draw must be inside the range AND they must not all be equal — a constant
   would make every episode open it identically.

⚠ NO MuJoCo LEG HERE, AND IT IS NOT MISSING. The reference numbers are the
ASSET's own sites, read by `asset_placement_geom` from the same XML MuJoCo
loads; what MuJoCo settled is the RULE (`SiteRegionRandomSampler.sample` places
the origin at `site_z - bottom_offset[-1]`), which is transcribed in
`sampler.sample_placements` and quoted there. What a MuJoCo leg would add is a
contact count, and `libero_viewer --check` prints ours beside the pair list for
exactly that comparison — MuJoCo reports 0 for both composed scenes with the
props parked.
"""

from std.os import listdir
from std.os.path import exists

from mojo_rl.physics3d.fields import Data, Model, DynDims, DynamicsScratch
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
    INIT_TARGET_SLOT, STACK_Z_OFFSET, TABLE_Z_OFFSET, has_stacked_init,
)
from mojo_rl.tasks.gpu_eval import require_gpu_placement
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.eval import region_sites
from mojo_rl.tasks.sampler import (
    sample_placements, sample_joint_inits, RegionFrame, SampleReport,
)
from mojo_rl.tasks.reset import (
    free_slot_addresses, reset_slots,
    joint_init_addresses, joint_init_dof_addresses, apply_joint_inits,
)
from mojo_rl.tasks.libero_spatial_xml import LIBERO_SPATIAL_MAX_CONTACTS
from mojo_rl.tasks.libero_init_z import load_init_z
from mojo_rl.tasks.bddl import parse_bddl


comptime DT = DType.float64
comptime SUITE = "libero_spatial"
comptime FAMILY = "mojo_rl/tasks/families/libero_spatial.family"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime PACK = "mojo_rl/tasks/libero/assets"
comptime BDDL_DIR = (
    "references/LIBERO-master/libero/libero/bddl_files/libero_spatial"
)
comptime Z_TOL: Float64 = 1.0e-12
"""The reference is a decimal STRING and ours is binary arithmetic — see
`test_libero_object`'s note. A picometre is far below anything that matters."""
comptime OLD_RADIUS: Float64 = 0.02
"""What the caller used to pass, and what check 3 lowers an object back to."""


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


def _task_names() raises -> List[String]:
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


def main() raises:
    print("=" * 74)
    print("Where a reset puts things —", SUITE)
    print("=" * 74)
    if not exists(PACK):
        print("  SKIPPED: no LIBERO pack at", PACK)
        print("=== SKIPPED (no pack — this is not a pass) ===")
        return
    var ta = Tally()

    var f = load_family(FAMILY)
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_SPATIAL_MAX_CONTACTS, nmesh_verts=verts
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
                fmd, max_contacts=LIBERO_SPATIAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
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
        radii.append(OLD_RADIUS)

    # every free slot must carry the asset's numbers, or checks 1-3 are vacuous
    var n_geom = 0
    var n_free = 0
    for si in range(len(f.slots)):
        if f.slots[si].kind != SLOT_FREE:
            continue
        n_free += 1
        if f.slots[si].has_geom:
            n_geom += 1
    print()
    print("--- 0. the family carries the assets' placement sites ---")
    print("    ", n_geom, "of", n_free, "free slots have slot_geom=")
    ta.check(n_geom == n_free and n_free > 0,
             "every free slot's bottom_site / top_site /"
             " horizontal_radius_site is in the family")
    ta.check(f.slots[f.slot_index(String("akita_black_bowl_1"))].bottom_z != -OLD_RADIUS,
             "and the bottom_site is NOT the old constant — "
             + String(f.slots[f.slot_index(String("akita_black_bowl_1"))].bottom_z)
             + " against " + String(-OLD_RADIUS))

    var names = _task_names()
    print()
    print("--- 1-2. every task: the asset's height, and no contacts ---")
    print("    task                                          slots  worst |z - rule|  ncon")
    var worst_z = 0.0
    var n_stacks = 0
    var bad_con = 0
    var n_tasks = 0
    var zt = load_init_z(String(SUITE))
    var n_z = 0
    var n_z_bad = 0
    var worst_frozen = 0.0
    for ti in range(len(names)):
        var t = load_task(TASK_DIR + String(names[ti]) + ".task")
        validate_task_against_family(t, f)
        # ⚠ THE CORPUS' OWN NAMES. `init_z_*.kv` is keyed by the `.bddl`'s
        # region (or, for a stack, the other prop), never by the family's.
        var btext: String
        with open(String(BDDL_DIR) + "/" + String(names[ti])[
                byte = String(SUITE).byte_length() + 2
                : String(names[ti]).byte_length()] + ".bddl", "r") as bh:
            btext = bh.read()
        var bp = parse_bddl(btext)
        var frozen = List[Float64](length=len(f.slots), fill=0.0)
        var has_frozen = List[Bool](length=len(f.slots), fill=False)
        for k in range(len(bp.init)):
            ref ia = bp.init[k]
            if len(ia.args) != 2 or (ia.pred != "On" and ia.pred != "In"):
                continue
            var fsi = f.slot_index(String(ia.args[0]))
            if fsi < 0 or f.slots[fsi].kind != SLOT_FREE:
                continue
            frozen[fsi] = zt.height(String(ia.args[0]), String(ia.args[1]))
            has_frozen[fsi] = True
        var inits_inside = List[Bool](length=len(f.slots), fill=False)
        for k in range(len(t.inits)):
            var isi = f.slot_index(t.inits[k].slot)
            if isi >= 0:
                inits_inside[isi] = t.inits[k].inside
        var jq_adr = joint_init_addresses(t, fmd.joint_names, jqn)
        var jv_adr = joint_init_dof_addresses(t, fmd.joint_names, jvn)

        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](0)
        for i in range(nv):
            d.qvel.data[i] = Scalar[DT](0)
        for i in range(len(f.base_qpos)):
            d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
        # ⚠⚠ JOINT INITS BEFORE FK AND BEFORE THE FRAMES — see the note in
        # `libero_viewer.do_reset`. Opening the drawer moves the region the bowl
        # is placed into; this gate is what found it (32 contacts).
        var jvals = sample_joint_inits(t, UInt64(7), ti)
        for k in range(len(jvals)):
            d.qpos.data[jq_adr[k]] = Scalar[DT](jvals[k])
            d.qvel.data[jv_adr[k]] = Scalar[DT](0)
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        var sp = List[Float64]()
        for i in range(ns * 3):
            sp.append(Float64(d.site_xpos.data[i]))
        var frames = List[RegionFrame]()
        for i in range(len(f.regions)):
            var si2 = rsites[i]
            frames.append(RegionFrame(
                sp[si2 * 3], sp[si2 * 3 + 1], sp[si2 * 3 + 2]
            ))
        var rep = SampleReport()
        var placed = sample_placements(t, f, frames, radii, UInt64(7), ti, rep)
        var qpos = List[Float64]()
        for i in range(nq):
            qpos.append(Float64(d.qpos.data[i]))
        var qvel = List[Float64](length=nv, fill=0.0)
        reset_slots(t, f, placed, addrs, qpos, qvel)
        apply_joint_inits(t, jq_adr, jvals, qpos, qvel, jv_adr)
        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](qpos[i])
        for i in range(nv):
            d.qvel.data[i] = Scalar[DT](qvel[i])
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        detect_contacts["cpu", DT, DynDims, 1](d, m)
        var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])

        # ⚠ THE RULE, RECOMPUTED FROM THE FAMILY — not read back from the
        # sampler. Comparing the sampler's z against the sampler's own formula
        # would be the blind shape; this takes the region's site AFTER FK and
        # the slot's own `bottom_z`.
        var w = 0.0
        for pi in range(len(placed)):
            var si3 = placed[pi].slot
            var tgt = String("")
            for k in range(len(t.inits)):
                if t.inits[k].slot == f.slots[si3].name:
                    tgt = String(t.inits[k].region)
            if tgt.byte_length() == 0:
                raise Error("no init target for a placed slot")
            var want_z: Float64
            var want_x = placed[pi].x
            var want_y = placed[pi].y
            if f.init_target_kind(tgt) == INIT_TARGET_SLOT:
                # ⚠ THE STACK RULE, RECOMPUTED: robosuite's ObjectBasedSampler
                # is `z = ref_z + ref.top_z + z_offset - obj.bottom_z` at the
                # reference's own x/y (its x_ranges and y_ranges are [0, 0]).
                var rsi = f.slot_index(tgt)
                var rj = -1
                for q in range(len(placed)):
                    if placed[q].slot == rsi:
                        rj = q
                if rj < 0:
                    raise Error(
                        "the stack's reference was not placed — order_inits"
                        " should have put it first"
                    )
                want_z = (
                    placed[rj].z + f.slots[rsi].top_z + STACK_Z_OFFSET
                    - f.slots[si3].bottom_z
                )
                want_x = placed[rj].x
                want_y = placed[rj].y
                n_stacks += 1
            else:
                # ⚠ THE TABLE/FLOOR CENTIMETRE, RECOMPUTED. LIBERO routes a
                # region on the TABLE to `TableRegionSampler` (`z_offset=0.01`
                # in its own signature) and one on a FIXTURE to
                # `SiteRegionRandomSampler` (0.0) — `spec.TABLE_Z_OFFSET`
                # quotes both and records the `.pruned_init` measurement that
                # settled it. The discriminator is whether the region names a
                # contact slot, i.e. whether it is anchored to a fixture.
                var ri = f.region_index(tgt)
                var zo = 0.0
                if f.slots[si3].has_geom:
                    if f.regions[ri].contact.byte_length() == 0:
                        zo = TABLE_Z_OFFSET
                    elif not inits_inside[si3]:
                        # ⚠ `On` A FIXTURE ADDS THE FIXTURE'S OWN `top_site`
                        # and `In` does not — one line, commented out in
                        # `InSiteRegionRandomSampler`. See `InitSpec.inside`.
                        zo = f.slots[f.slot_index(f.regions[ri].contact)].top_z
                want_z = frames[ri].z + zo - f.slots[si3].bottom_z
            # ⚠⚠ AND AGAINST LIBERO'S OWN FROZEN STATE. The rule above is
            # recomputed from the family; this is the height the benchmark
            # actually restores at reset, for all three shapes at once — a
            # table region (0.9700), a fixture one (1.0100 on the stove,
            # 1.1506 inside the drawer, 1.2315 on its roof) and a STACK
            # (1.0800 on the cookie box). Nothing of ours is consulted.
            if has_frozen[si3]:
                var ef = abs(placed[pi].z - frozen[si3])
                n_z += 1
                if ef > Z_TOL:
                    n_z_bad += 1
                if ef > worst_frozen:
                    worst_frozen = ef
            var e = abs(placed[pi].z - want_z)
            var ex = abs(placed[pi].x - want_x)
            var ey = abs(placed[pi].y - want_y)
            if ex > e:
                e = ex
            if ey > e:
                e = ey
            if e > w:
                w = e
        if w > worst_z:
            worst_z = w
        if ncon != 0:
            bad_con += 1
        n_tasks += 1
        var pad = String(names[ti])
        while pad.byte_length() < 46:
            pad += " "
        print("    " + pad + String(len(placed)) + "      " + String(w)
              + "      " + String(ncon))
    ta.check(n_tasks >= 8, String(n_tasks) + " tasks checked")
    ta.check(worst_z == 0.0,
             "every placement is exactly its rule — region_site_z, plus"
             " TABLE_Z_OFFSET on a table region or the FIXTURE's top_site on"
             " an `On` fixture one (and nothing on an `In`), less"
             " slot.bottom_z; or the reference's x/y and top for a stack")
    # ⚠ ANTI-VACUITY: without a stack in the suite, the branch above is dead and
    # the check says nothing about `init=x@y`.
    print("     against LIBERO's own frozen states:", n_z, "placements, worst",
          worst_frozen, "m")
    ta.check(n_z >= 40 and n_z_bad == 0,
             "every placement is the height LIBERO's `.pruned_init` holds for"
             " that (prop, region) — the table, the stove, the drawer, the"
             " cabinet roof and the stack, all from its own states")
    ta.check(n_stacks > 0,
             String(n_stacks) + " stacked placements were checked (init= onto"
             " another free slot)")
    ta.check(bad_con == 0,
             String(n_tasks - bad_con) + " of " + String(n_tasks)
             + " tasks reset with ZERO contacts")

    # ── 3. the control: the old height is NOT contact-free ────────────────
    print()
    print("--- 3. the control: at site_z + 0.02 the contacts come back ---")
    # ⚠⚠ THE STOVE TASK, NOT THE DRAWER ONE, AND THE REASON IS THE POINT OF A
    # CONTROL. The drawer was the obvious choice and it does NOT discriminate:
    # with the drawer open its interior box is 20 cm tall, so lowering the bowl
    # 4 cm inside it still touches nothing and the control reported 0 — a
    # control that passes for the wrong reason is worse than none.
    #
    # The stove's `cook_region` site sits at the vertical CENTRE of a 4 cm base
    # box with the burner plate 2.5 cm above it, so the old height put the bowl
    # squarely inside both. That is the case the old rule actually broke.
    var drawer = String("")
    for ti in range(len(names)):
        if String(names[ti]).find("on_the_stove") >= 0:
            drawer = String(names[ti])
    ta.check(drawer.byte_length() > 0, "the stove task is in the suite")
    var t2 = load_task(TASK_DIR + drawer + ".task")
    validate_task_against_family(t2, f)
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    var jq2 = joint_init_addresses(t2, fmd.joint_names, jqn)
    var jv2 = joint_init_dof_addresses(t2, fmd.joint_names, jvn)
    var jvals2 = sample_joint_inits(t2, UInt64(7), 0)
    for k in range(len(jvals2)):
        d.qpos.data[jq2[k]] = Scalar[DT](jvals2[k])
        d.qvel.data[jv2[k]] = Scalar[DT](0)
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var sp2 = List[Float64]()
    for i in range(ns * 3):
        sp2.append(Float64(d.site_xpos.data[i]))
    var frames2 = List[RegionFrame]()
    for i in range(len(f.regions)):
        var si4 = rsites[i]
        frames2.append(RegionFrame(
            sp2[si4 * 3], sp2[si4 * 3 + 1], sp2[si4 * 3 + 2]
        ))
    var rep2 = SampleReport()
    var placed2 = sample_placements(t2, f, frames2, radii, UInt64(7), 0, rep2)
    var qpos2 = List[Float64]()
    for i in range(nq):
        qpos2.append(Float64(d.qpos.data[i]))
    var qvel2 = List[Float64](length=nv, fill=0.0)
    reset_slots(t2, f, placed2, addrs, qpos2, qvel2)
    apply_joint_inits(t2, jq2, jvals2, qpos2, qvel2, jv2)
    # ⚠⚠ THE OLD RULE, WRITTEN OUT — NOT A FIXED DROP. It used to lower each
    # prop by `-bottom_z - 0.02`, which stopped discriminating the moment the
    # fixture's `top_site` was added to the correct height: the stove bowl rose
    # 4.5 cm, so a 4 cm drop left it ABOVE where it had been and the control
    # reported 0. The rule this is a control for is `region_site_z + a
    # caller-supplied radius`, so it is recomputed from the frame each time and
    # cannot drift behind the real one again.
    var dropped = 0.0
    for pi in range(len(placed2)):
        var si5 = placed2[pi].slot
        var tgt2 = String("")
        for k in range(len(t2.inits)):
            if t2.inits[k].slot == f.slots[si5].name:
                tgt2 = String(t2.inits[k].region)
        var ri2 = f.region_index(tgt2)
        if ri2 < 0:
            continue          # a stack: the old rule had no stacking at all
        var old_z = frames2[ri2].z + OLD_RADIUS
        var drop = qpos2[addrs[si5].qadr + 2] - old_z
        qpos2[addrs[si5].qadr + 2] = old_z
        if drop > dropped:
            dropped = drop
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](qpos2[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    detect_contacts["cpu", DT, DynDims, 1](d, m)
    var old_con = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    print("     lowered every prop by", dropped, "m ->", old_con, "contacts")
    ta.check(old_con > 0,
             "the OLD height is not contact-free (" + String(old_con)
             + " contacts), so check 2 is looking at the geometry")

    # ── 4. `jinit=` is a draw ─────────────────────────────────────────────
    print()
    print("--- 4. jinit draws, inside its range, differently per lane ---")
    # ⚠ THE DRAWER TASK IS THE ONE WITH A `jinit=`; the stove task above has
    # none, so check 4 loads it separately rather than reusing `t2`.
    var dr = String("")
    for ti in range(len(names)):
        if String(names[ti]).find("in_the_top_drawer") >= 0:
            dr = String(names[ti])
    var t3 = load_task(TASK_DIR + dr + ".task")
    validate_task_against_family(t3, f)
    ta.check(len(t3.joint_inits) > 0,
             String(len(t3.joint_inits)) + " jinit line(s) on the drawer task:"
             " " + t3.joint_inits[0].describe())
    var in_range = True
    var spread = 0.0
    var first = 0.0
    for lane in range(16):
        var v = sample_joint_inits(t3, UInt64(7), lane)
        for k in range(len(v)):
            if v[k] < t3.joint_inits[k].lo or v[k] > t3.joint_inits[k].hi:
                in_range = False
            if lane == 0 and k == 0:
                first = v[k]
            if k == 0:
                var e = abs(v[k] - first)
                if e > spread:
                    spread = e
    print("     16 lanes, spread", spread, "over a range of",
          t3.joint_inits[0].hi - t3.joint_inits[0].lo)
    ta.check(in_range, "every draw is inside its declared range")
    ta.check(spread > 0.0,
             "the draws DIFFER across lanes — a constant would open the drawer"
             " identically in every episode")

    # ── 5. the device refusal ─────────────────────────────────────────────
    #
    # ⚠⚠ A STACK IS NOT DEVICE-SAMPLABLE, and the point of the check is that the
    # refusal is SELECTIVE. `require_gpu_placement` raising on everything would
    # score the same as raising on the right thing, and would take the other
    # eight tasks off the device with it.
    print()
    print("--- 5. require_gpu_placement refuses stacks, and only stacks ---")
    var n_stack_tasks = 0
    var refused = 0
    var wrongly_refused = String("")
    for ti in range(len(names)):
        var tk = load_task(TASK_DIR + String(names[ti]) + ".task")
        validate_task_against_family(tk, f)
        var stacked = has_stacked_init(tk, f)
        if stacked:
            n_stack_tasks += 1
        var raised = False
        try:
            require_gpu_placement(tk, f)
        except e:
            raised = True
        if raised and stacked:
            refused += 1
        elif raised and not stacked:
            wrongly_refused += " " + String(names[ti])
        elif stacked and not raised:
            wrongly_refused += " NOT-REFUSED:" + String(names[ti])
    print("    ", n_stack_tasks, "tasks stack;", refused, "refused;",
          len(names) - n_stack_tasks, "region-only tasks accepted")
    ta.check(n_stack_tasks == 2,
             String(n_stack_tasks) + " tasks in the suite stack one free slot"
             " on another")
    ta.check(refused == n_stack_tasks and wrongly_refused.byte_length() == 0,
             "every stacking task is refused and no other is" + wrongly_refused)

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "libero placement: " + String(ta.failures) + " of "
            + String(ta.checks) + " failed"
        )
    print("=== PASS ===")
