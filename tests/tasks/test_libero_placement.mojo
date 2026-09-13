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
)
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


comptime DT = DType.float64
comptime SUITE = "libero_spatial"
comptime FAMILY = "mojo_rl/tasks/families/libero_spatial.family"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime PACK = "mojo_rl/tasks/libero/assets"
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
    var bad_con = 0
    var n_tasks = 0
    for ti in range(len(names)):
        var t = load_task(TASK_DIR + String(names[ti]) + ".task")
        validate_task_against_family(t, f)
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
            var ri = -1
            for k in range(len(t.inits)):
                if t.inits[k].slot == f.slots[si3].name:
                    ri = f.region_index(t.inits[k].region)
            if ri < 0:
                raise Error("no init region for a placed slot")
            var want_z = frames[ri].z - f.slots[si3].bottom_z
            var e = abs(placed[pi].z - want_z)
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
             "every placement is exactly region_site_z - slot.bottom_z")
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
    # ⚠ LOWER EVERY PLACED SLOT BACK TO THE OLD RULE, in place. The drop is
    # `-bottom_z - 0.02` per slot, which for these assets is 4 cm.
    var dropped = 0.0
    for pi in range(len(placed2)):
        var si5 = placed2[pi].slot
        var drop = -f.slots[si5].bottom_z - OLD_RADIUS
        qpos2[addrs[si5].qadr + 2] -= drop
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

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "libero placement: " + String(ta.failures) + " of "
            + String(ta.checks) + " failed"
        )
    print("=== PASS ===")
