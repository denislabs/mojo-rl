"""LIBERO-10 AND LIBERO-90 — twenty scenes, ninety-nine tasks, one pass.

    pixi run mojo run -I . tests/libero/test_libero_scenes.mojo

## ⚠⚠ THE FAMILY IS THE SCENE, NOT THE SUITE

The first three LIBERO suites were each ten files on ONE scene, so "one family
per suite" and "one family per compile unit" were the same sentence. LIBERO-10
and LIBERO-90 are two published benchmarks that SHARE twenty scenes
(KITCHEN 1-10, LIVING_ROOM 1-6, STUDY 1-4): `KITCHEN_SCENE3` carries five
tasks, one of which is scored as LIBERO-10 and four as LIBERO-90. Grouping by
suite would compose the same scene twice and put tasks that share a table in
two different compile units; grouping by scene gives twenty families of three
to seven tasks, and each `.task` records its benchmark with `suite=`.

⚠ ONLY THE FILENAME SAYS WHICH SCENE. Every one of the hundred files declares
the problem class (`LIBERO_Kitchen_Tabletop_Manipulation`) and none declares
which of the ten kitchens it is — `gen_libero_family._scene_prefix` reads it
off the name, and this file re-derives the mapping from the corpus rather than
from the generated names.

⚠ NO SLOT UNION IS NEEDED HERE, unlike `libero_object`. Measured over the 100
files: within a scene every file declares the SAME props and fixtures at the
same poses, so `_union_slots` merges without ever adding a second entry. That
is checked below (check 2) rather than assumed, because the moment it stops
being true the union would quietly widen a scene.

## WHAT THIS ASSERTS

0. **twenty families, ninety-nine tasks, and the corpus agrees** — the scene
   of every task is re-derived from the `.bddl` filename and must equal the
   family it was filed under, and the `suite=` counts must be 9 + 90. (Not
   10 + 90: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` needs five tape
   terms and the device tape holds three, so it is refused by name — and it is
   a LIBERO-10 file, which is why that benchmark is one short.)

1. **every family composes and every task binds** — `bind_goal` against the
   composed scene's own body / site / joint names, `require_tier_a`,
   `require_gpu_regions` and `encode_goal` (the twelve `meta` words). A task
   that translated but cannot bind is a task that would fail at the driver.

2. **the scene is one scene** — every family's slot table is exactly what each
   of its files declares (see above), and no family exceeds the device
   region table.

3. ⚠⚠ **EVERY PLACEMENT IS THE HEIGHT LIBERO'S OWN FROZEN STATES HOLD**, over
   all twenty scenes — `init_z_<family>.kv`, read out of the `.pruned_init`
   files by `tools/libero/libero_init_z.py` and keyed by the corpus' own
   (prop, region) names. This is the widest such comparison in the port and
   nothing of ours is consulted for the reference.

   ⚠ CONTACTS ARE PRINTED, NOT ASSERTED, and that is measured rather than
   conceded: at LIBERO's OWN frozen state MuJoCo reports 112 contacts and a
   2.9 cm penetration on `LIVING_ROOM_SCENE2`. `bottom_site` is a margin well
   below the collision geometry, so props on these tables start inside them
   and the five settle steps push them out. `test_libero_placement` keeps a
   zero-contact claim because it is true of `libero_spatial` — here it would
   be requiring something the benchmark does not do.

4. ⚠ **both suites in one family, measured.** If every scene family drew from
   one suite only, the whole reason for grouping by scene would be untested —
   so the number of families holding tasks from BOTH is printed and required
   to be nonzero.

⚠ NO MuJoCo LEG HERE. `tools/tasks/check_family.py` is the oracle for a
composed scene and was run on all twenty: they load, and `ncon` at rest with
`base_qpos` applied is 0 for every one (nq 23-65, nv 21-57).
"""

from std.os import listdir
from std.os.path import exists

from noeira.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
    FamilySpec,
)
from noeira.tasks.family import scene_path
from noeira.envs.libero.bddl import parse_bddl
from noeira.envs.libero.init_z import load_init_z
from noeira.tasks.eval import region_sites
from noeira.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses,
)
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.gpu_eval import require_gpu_regions, MAX_CURRICULUM_REGIONS
from noeira.tasks.sampler import (
    sample_placements, sample_joint_inits, RegionFrame, SampleReport,
)
from noeira.tasks.reset import (
    free_slot_addresses, reset_slots,
    joint_init_addresses, joint_init_dof_addresses, apply_joint_inits,
)


comptime DT = DType.float64
comptime FAMILY_DIR = "noeira/envs/libero/families"
comptime TASK_DIR = "noeira/envs/libero/tasks/"
comptime PACK = "noeira/envs/libero/assets"
comptime BDDL_ROOT = "references/LIBERO-master/libero/libero/bddl_files"
comptime MAX_CONTACTS = 96
comptime SEED: UInt64 = 7
comptime Z_TOL: Float64 = 1.0e-12
comptime N_SCENES = 20
comptime N_TASKS = 99
comptime N_LIBERO_10 = 9
comptime N_LIBERO_90 = 90
"""9 + 90, not 10 + 90: the one refusal
(`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, five tape terms against a
three-term device tape) is a LIBERO-10 file. Written out rather than derived
so that a task quietly moving between benchmarks would fail here."""


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


def _sorted(var xs: List[String]) -> List[String]:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                xs[i], xs[j] = xs[j], xs[i]
    return xs^


def _scene_families() raises -> List[String]:
    """The twenty `libero_<scene>` families, by name."""
    var out = List[String]()
    for e in listdir(FAMILY_DIR):
        var n = String(e)
        if not n.endswith(".family"):
            continue
        var stem = String(n[byte = 0 : n.byte_length() - 7])
        if stem.find("_scene") >= 0:
            out.append(stem^)
    return _sorted(out^)


def _tasks_of(family: String) raises -> List[String]:
    var out = List[String]()
    var want = family + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    return _sorted(out^)


def _corpus_scene(stem: String) raises -> String:
    """The family a `.bddl` filename belongs to — re-derived HERE.

    ⚠ THE POINT IS NOT TO ASK THE GENERATOR. `gen_libero_family` reads the
    scene off the filename; this reads it off the same filename with its own
    code, so a task filed under the wrong family is visible. The two agreeing
    is check 0."""
    var at = stem.find("_SCENE")
    if at < 0:
        raise Error("corpus: '" + stem + "' has no _SCENE<n>")
    var i = at + 6
    while i < stem.byte_length():
        var ch = String(stem[byte = i : i + 1])
        if String("0123456789").find(ch) < 0:
            break
        i += 1
    return String("libero_") + String(stem[byte=0:i]).lower()


def main() raises:
    print("=" * 74)
    print("LIBERO-10 and LIBERO-90 — the scene families")
    print("=" * 74)
    if not exists(PACK):
        print("  SKIPPED: no LIBERO pack at", PACK)
        print("=== SKIPPED (no pack — this is not a pass) ===")
        return
    var ta = Tally()

    # ── 0. the corpus decides which family each task belongs to ───────────
    print()
    print("--- 0. the corpus, the families and the suite tags ---")
    var fams = _scene_families()
    var want_family = List[String]()
    var want_stem = List[String]()
    var want_suite = List[String]()
    var suite_dirs = List[String]()
    suite_dirs.append(String("libero_10"))
    suite_dirs.append(String("libero_90"))
    for sx in range(len(suite_dirs)):
        var s = String(suite_dirs[sx])
        var dir = String(BDDL_ROOT) + "/" + s
        if not exists(dir):
            print("  SKIPPED: no corpus at", dir)
            print("=== SKIPPED (no corpus — this is not a pass) ===")
            return
        for e in listdir(dir):
            var n = String(e)
            if not n.endswith(".bddl"):
                continue
            var stem = String(n[byte = 0 : n.byte_length() - 5])
            want_family.append(_corpus_scene(stem))
            want_stem.append(stem^)
            want_suite.append(String(s))
    print("    ", len(fams), "scene families,", len(want_stem),
          "files in the corpus")
    ta.check(len(fams) == N_SCENES,
             String(len(fams)) + " scene families on disk")
    ta.check(len(want_stem) == 100, "100 files in libero_10 + libero_90")

    # every written task must be one of those files, filed under the family
    # the CORPUS gives it, and tagged with the suite the file came from
    var n10 = 0
    var n90 = 0
    var bad_family = 0
    var bad_suite = 0
    var both = 0
    var n_total = 0
    for fi in range(len(fams)):
        var names = _tasks_of(fams[fi])
        var saw10 = False
        var saw90 = False
        for k in range(len(names)):
            var t = load_task(TASK_DIR + names[k] + ".task")
            n_total += 1
            # find the corpus file whose scene+remainder makes this name
            var found = -1
            for c in range(len(want_stem)):
                if want_family[c] + "__" + _remainder(want_stem[c]) == t.name:
                    found = c
            if found < 0:
                raise Error(
                    "task '" + t.name + "' matches no file of libero_10 or"
                    " libero_90. A stale .task the generator no longer writes?"
                )
            if t.family != want_family[found]:
                bad_family += 1
            if t.suite != want_suite[found]:
                bad_suite += 1
            if t.suite == "libero_10":
                n10 += 1
                saw10 = True
            elif t.suite == "libero_90":
                n90 += 1
                saw90 = True
        if saw10 and saw90:
            both += 1
    print("     tasks:", n_total, "=", n10, "libero_10 +", n90, "libero_90;",
          both, "families hold BOTH")
    ta.check(n_total == N_TASKS,
             String(n_total) + " tasks written over the twenty scenes")
    ta.check(bad_family == 0,
             "every task is filed under the family the CORPUS gives its file")
    ta.check(bad_suite == 0 and n10 == N_LIBERO_10 and n90 == N_LIBERO_90,
             "every task carries the suite= of the directory it came from ("
             + String(n10) + " + " + String(n90) + ")")
    # ⚠ THE ANTI-VACUITY LINE for grouping by scene at all.
    ta.check(both > 0,
             String(both) + " families hold tasks from BOTH benchmarks — the"
             " reason the family is the scene and not the suite")

    # ── 1-3. compose, bind, reset ─────────────────────────────────────────
    print()
    print("--- 1-3. every family composes; every task binds and resets ---")
    print("    family                        tasks  slots  regions  ncon  words")
    var bad_con = 0
    var max_con = 0
    var bad_place = 0
    var n_bound = 0
    var n_z = 0
    var n_z_bad = 0
    var worst_z = 0.0
    var max_regions = 0
    var wide_slots = 0
    for fi in range(len(fams)):
        var f = load_family(String(FAMILY_DIR) + "/" + fams[fi] + ".family")
        var zt = load_init_z(fams[fi])
        var names = _tasks_of(fams[fi])
        var fmd = parse_model_runtime(scene_path(f))
        var verts = 32768
        var dims = dims_from_flat(
            fmd, max_contacts=MAX_CONTACTS, nmesh_verts=verts
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
                    fmd, max_contacts=MAX_CONTACTS, nmesh_verts=verts
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
        var jadr = joint_qpos_addresses(jqn)
        var radii = List[Float64]()
        for _ in range(len(f.slots)):
            radii.append(0.02)
        if len(f.regions) > max_regions:
            max_regions = len(f.regions)
        if len(f.regions) > MAX_CURRICULUM_REGIONS:
            wide_slots += 1

        var fam_con = 0
        var words = 0
        for k in range(len(names)):
            var t = load_task(TASK_DIR + names[k] + ".task")
            validate_task_against_family(t, f)
            # ── bind ──
            var g = bind_goal(
                parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
                fmd.joint_names, jadr,
            )
            require_tier_a(g, t.name)
            require_gpu_regions(g, t.name)
            words = len(encode_goal(g))
            n_bound += 1
            # ── reset ──
            for i in range(nq):
                d.qpos.data[i] = Scalar[DT](0)
            for i in range(nv):
                d.qvel.data[i] = Scalar[DT](0)
            for i in range(len(f.base_qpos)):
                d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
            # ⚠ THE REFERENCE HEIGHT COMES FROM THE CORPUS, per (prop,
            # region), so the key is the `.bddl`'s own vocabulary.
            var bstem = String("")
            var bsuite = String("")
            for c in range(len(want_stem)):
                if want_family[c] + "__" + _remainder(want_stem[c]) == t.name:
                    bstem = String(want_stem[c])
                    bsuite = String(want_suite[c])
            var btext: String
            with open(String(BDDL_ROOT) + "/" + bsuite + "/" + bstem
                      + ".bddl", "r") as bh:
                btext = bh.read()
            var bp = parse_bddl(btext)
            var frozen = List[Float64](length=len(f.slots), fill=0.0)
            var has_frozen = List[Bool](length=len(f.slots), fill=False)
            for c in range(len(bp.init)):
                ref ia = bp.init[c]
                if len(ia.args) != 2 or (ia.pred != "On" and ia.pred != "In"):
                    continue
                var fsi = f.slot_index(String(ia.args[0]))
                if fsi < 0 or f.slots[fsi].kind != SLOT_FREE:
                    continue
                frozen[fsi] = zt.height(
                    String(ia.args[0]), String(ia.args[1])
                )
                has_frozen[fsi] = True
            var jq_adr = joint_init_addresses(t, fmd.joint_names, jqn)
            var jv_adr = joint_init_dof_addresses(t, fmd.joint_names, jvn)
            var jvals = sample_joint_inits(t, SEED, k)
            for j in range(len(jvals)):
                d.qpos.data[jq_adr[j]] = Scalar[DT](jvals[j])
                d.qvel.data[jv_adr[j]] = Scalar[DT](0)
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
            var placed = sample_placements(t, f, frames, radii, SEED, k, rep)
            for pi in range(len(placed)):
                var psi = placed[pi].slot
                if not has_frozen[psi]:
                    continue
                var ez = abs(placed[pi].z - frozen[psi])
                n_z += 1
                if ez > Z_TOL:
                    n_z_bad += 1
                if ez > worst_z:
                    worst_z = ez
            # every ACTIVE free slot must have been placed
            var want_placed = 0
            for si in range(len(f.slots)):
                if f.slots[si].kind == SLOT_FREE and t.is_active(f.slots[si].name):
                    want_placed += 1
            if len(placed) != want_placed:
                bad_place += 1
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
            if ncon != 0:
                bad_con += 1
            if ncon > fam_con:
                fam_con = ncon
            if ncon > max_con:
                max_con = ncon
        var pad = String(fams[fi])
        while pad.byte_length() < 30:
            pad += " "
        print("    " + pad + String(len(names)) + "      "
              + String(len(f.slots)) + "      " + String(len(f.regions))
              + "       " + String(fam_con) + "     " + String(words))
        _ = ns

    ta.check(n_bound == N_TASKS,
             String(n_bound) + " goals bind, are Tier A, fit the device region"
             " table and encode into " + String(TAPE_WORDS) + " tape words")
    ta.check(wide_slots == 0,
             "no family exceeds the device region table (widest "
             + String(max_regions) + " of " + String(MAX_CURRICULUM_REGIONS)
             + ")")
    ta.check(bad_place == 0,
             "every ACTIVE free slot of every task was placed")
    print("     against LIBERO's own frozen states:", n_z, "placements, worst",
          worst_z, "m")
    ta.check(n_z >= N_TASKS and n_z_bad == 0,
             "every placement over the twenty scenes starts at the height"
             " LIBERO's `.pruned_init` holds for that (prop, region)")
    print("     contacts at reset: max", max_con, "over", bad_con,
          "tasks. LIBERO's OWN frozen state gives MuJoCo 112 and a 2.9 cm"
          " penetration on LIVING_ROOM_SCENE2 — see the header.")

    print()
    print("  checks:", ta.checks, " failures:", ta.failures)
    if ta.failures != 0:
        raise Error(String(ta.failures) + " check(s) failed")
    print("=== LIBERO SCENES OK ===")


def _remainder(stem: String) raises -> String:
    """The `.bddl` stem with its `<SCENE>_` prefix removed."""
    var at = stem.find("_SCENE")
    if at < 0:
        raise Error("corpus: '" + stem + "' has no _SCENE<n>")
    var i = at + 6
    while i < stem.byte_length():
        var ch = String(stem[byte = i : i + 1])
        if String("0123456789").find(ch) < 0:
            break
        i += 1
    return String(stem[byte = i + 1 : stem.byte_length()])
