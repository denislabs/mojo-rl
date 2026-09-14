"""THE DEVICE SAMPLER AGAINST THE HOST ONE — one distribution, two writers.

    pixi run mojo run -I . tests/tasks/test_device_placement.mojo

## ⚠⚠ WHY THERE ARE TWO AT ALL, AND WHY THAT IS THE RISK

`tasks/sampler.sample_placements` places props on the HOST — it is what the
eval, the viewer and the init table use. `placement/table.place_free_slots`
places them on the DEVICE, per lane, at every reset, from a family's comptime
`PlacementTable`. Two implementations of one distribution is exactly the drift
this file exists to prevent: if they disagree, a policy trains on scenes the
eval never shows it and every other number in the system still agrees.

## ⚠⚠ THIS GATE USED TO RUN ON ONE FAMILY, AND THAT FAMILY WAS BLIND

It ran `so101_tabletop` alone — the one family whose props declare no robosuite
sites, whose regions share one site, and whose tasks never stack. The old
device hook was correct there and wrong for every LIBERO family on four counts
(`slot_geom=`, the `order_inits` walk, `TABLE_Z_OFFSET`, the fixture's
`top_site` for `On`), and this file could not have seen any of them. It now runs
every task of every LIBERO family, and section 6 COUNTS that the corpus reaches
each rule — a parity check over placements that never take a branch says
nothing about that branch.

## THE SECTIONS

1. `meta`'s init block, and `so101_tabletop`'s hand-written table vs its family
2. device vs host on `so101_tabletop`
3. an untouched `meta` parks every free slot
4. every LIBERO family: its generated table vs the family and FK, EXACTLY
5. every LIBERO task: refused by `require_device_placement`, or device == host
   on every coordinate of every lane — and the refusals are exactly the tasks
   drawing in a region an INDEPENDENT oracle finds moving
6. the corpus reaches every rule the kernel mirrors
7. the stack exemption, on a scene built to reach it — the host used to CRASH
"""

from std.os import listdir

from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
from mojo_rl.tasks.spec import (
    FamilySpec, TaskSpec, load_family, load_task, parse_family, parse_task,
    validate_task_against_family, SLOT_FREE, INIT_TARGET_SLOT,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.family_config import (
    So101TabletopConfig, So101TabletopPlacement,
)
from mojo_rl.tasks.active import init_region_words
from mojo_rl.tasks.eval import region_sites
from mojo_rl.tasks.reset import SlotAddress, free_slot_addresses
from mojo_rl.tasks.sampler import (
    sample_placements, RegionFrame, SampleReport,
)
from mojo_rl.tasks.placement.table import (
    PlacementTable, place_free_slots,
)
from mojo_rl.tasks.placement.check import (
    require_device_placement, placement_table_drift,
)
from mojo_rl.tasks.placement.libero_goal import LiberoGoalPlacement
from mojo_rl.tasks.placement.libero_kitchen_scene1 import (
    LiberoKitchenScene1Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene2 import (
    LiberoKitchenScene2Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene3 import (
    LiberoKitchenScene3Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene4 import (
    LiberoKitchenScene4Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene5 import (
    LiberoKitchenScene5Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene6 import (
    LiberoKitchenScene6Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene7 import (
    LiberoKitchenScene7Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene8 import (
    LiberoKitchenScene8Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene9 import (
    LiberoKitchenScene9Placement,
)
from mojo_rl.tasks.placement.libero_kitchen_scene10 import (
    LiberoKitchenScene10Placement,
)
from mojo_rl.tasks.placement.libero_living_room_scene1 import (
    LiberoLivingRoomScene1Placement,
)
from mojo_rl.tasks.placement.libero_living_room_scene2 import (
    LiberoLivingRoomScene2Placement,
)
from mojo_rl.tasks.placement.libero_living_room_scene3 import (
    LiberoLivingRoomScene3Placement,
)
from mojo_rl.tasks.placement.libero_living_room_scene4 import (
    LiberoLivingRoomScene4Placement,
)
from mojo_rl.tasks.placement.libero_living_room_scene5 import (
    LiberoLivingRoomScene5Placement,
)
from mojo_rl.tasks.placement.libero_living_room_scene6 import (
    LiberoLivingRoomScene6Placement,
)
from mojo_rl.tasks.placement.libero_object import LiberoObjectPlacement
from mojo_rl.tasks.placement.libero_spatial import LiberoSpatialPlacement
from mojo_rl.tasks.placement.libero_study_scene1 import (
    LiberoStudyScene1Placement,
)
from mojo_rl.tasks.placement.libero_study_scene2 import (
    LiberoStudyScene2Placement,
)
from mojo_rl.tasks.placement.libero_study_scene3 import (
    LiberoStudyScene3Placement,
)
from mojo_rl.tasks.placement.libero_study_scene4 import (
    LiberoStudyScene4Placement,
)
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_INIT_REGION_0, META_INIT_SLOTS, META_IDX_LS_EVAL,
    MODEL_JOINT_SIZE, MODEL_BODY_SIZE, MODEL_GEOM_SIZE,
)
from mojo_rl.physics3d.joint_types import JNT_FREE
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics

from layout import Layout, LayoutTensor
from mojo_rl.nn.core.tensor import TensorImpl


comptime DT = DType.float64
comptime SO101_FAMILY = "mojo_rl/tasks/families/so101_tabletop.family"
comptime FAMILY_DIR = "mojo_rl/tasks/families"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime BATCH = 8
comptime SEED = 7
comptime TOL: Float64 = 1.0e-12
comptime N_LIBERO_FAMILIES = 23
comptime N_LIBERO_TASKS = 129

# A free slot the device declines to place must be LEFT ALONE, so every word it
# may not touch starts at a value no placement writes.
comptime QPOS_SENTINEL: Float64 = 777.0
comptime QVEL_SENTINEL: Float64 = 7.0
comptime META_CANARY: Float64 = 0.25


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


struct Stats(Copyable, ImplicitlyCopyable, Movable):
    """What the parity loop compared, and which rules it went through."""

    var tasks: Int
    var refused: Int
    var lanes: Int
    var placements: Int
    var coords: Int
    var bad: Int
    var worst: Float64
    var exact: Int
    var left_alone: Int
    var left_alone_bad: Int
    var meta_touched: Int
    # rule coverage, per placement the HOST made
    var geom: Int
    var table_off: Int
    var on_fixture: Int
    var in_fixture: Int
    var stacks: Int
    var reordered_tasks: Int
    var rejections: Int
    var clamped: Int
    var exempt: Int

    def __init__(out self):
        self.tasks = 0
        self.refused = 0
        self.lanes = 0
        self.placements = 0
        self.coords = 0
        self.bad = 0
        self.worst = 0.0
        self.exact = 0
        self.left_alone = 0
        self.left_alone_bad = 0
        self.meta_touched = 0
        self.geom = 0
        self.table_off = 0
        self.on_fixture = 0
        self.in_fixture = 0
        self.stacks = 0
        self.reordered_tasks = 0
        self.rejections = 0
        self.clamped = 0
        self.exempt = 0


struct Scene(Copyable, Movable):
    """What the host sampler needs from a composed scene, plus the oracle."""

    var frames: List[RegionFrame]
    var addrs: List[SlotAddress]
    var moves: List[Bool]
    var nq: Int
    var nv: Int

    def __init__(out self):
        self.frames = List[RegionFrame]()
        self.addrs = List[SlotAddress]()
        self.moves = List[Bool]()
        self.nq = 0
        self.nv = 0


def _sorted(var xs: List[String]) -> List[String]:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                xs[i], xs[j] = xs[j], xs[i]
    return xs^


def _libero_families() raises -> List[String]:
    var out = List[String]()
    for e in listdir(FAMILY_DIR):
        var n = String(e)
        if n.startswith("libero") and n.endswith(".family"):
            out.append(String(n[byte = 0 : n.byte_length() - 7]))
    return _sorted(out^)


def _tasks_of(family: String) raises -> List[String]:
    var out = List[String]()
    var want = family + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    return _sorted(out^)


def _scene(f: FamilySpec, verts0: Int) raises -> Scene:
    """Frames, addresses and the MOVING-REGION ORACLE from the composed scene.

    ⚠⚠ THE ORACLE DOES NOT WALK THE BODY CHAIN, which is how the generator
    decides `region_moves`. A gate that re-ran the generator's walk would
    share its off-by-one — and the generator's first run HAD one (the flat body
    list has no worldbody record). Instead: FK at the reset state, then nudge
    EVERY joint (a slide or hinge by 0.05, a free joint's position by 0.05),
    FK again, and a region moves iff its site did.
    """
    var fmd = parse_model_runtime(scene_path(f))
    var verts = verts0
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var sc = Scene()
    sc.nq = dims.get_nq()
    sc.nv = dims.get_nv()
    for i in range(sc.nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var rsites = region_sites(f, fmd.site_names)
    for r in range(len(f.regions)):
        var s = rsites[r]
        sc.frames.append(RegionFrame(
            Float64(d.site_xpos.data[s * 3]),
            Float64(d.site_xpos.data[s * 3 + 1]),
            Float64(d.site_xpos.data[s * 3 + 2]),
        ))
    var jt = List[Int]()
    var jqn = List[Int]()
    var jvn = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jqn.append(fmd.joints[i].nq)
        jvn.append(fmd.joints[i].nv)
    sc.addrs = free_slot_addresses(f, fmd.joint_names, jt, jqn, jvn)

    var adr = 0
    for j in range(len(jt)):
        if jt[j] == JNT_FREE:
            d.qpos.data[adr] = d.qpos.data[adr] + Scalar[DT](0.05)
        elif jqn[j] == 1:
            d.qpos.data[adr] = d.qpos.data[adr] + Scalar[DT](0.05)
        adr += jqn[j]
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    for r in range(len(f.regions)):
        var s = rsites[r]
        sc.moves.append(
            Float64(d.site_xpos.data[s * 3]) != sc.frames[r].x
            or Float64(d.site_xpos.data[s * 3 + 1]) != sc.frames[r].y
            or Float64(d.site_xpos.data[s * 3 + 2]) != sc.frames[r].z
        )
    return sc^


def _host_radii(f: FamilySpec, fallback: Float64) -> List[Float64]:
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(fallback)
    return radii^


def _coverage(t: TaskSpec, f: FamilySpec, mut st: Stats) raises:
    """Which of the kernel's rules THIS task's placements take — per lane, so
    the counts are comparable with `st.placements`."""
    var reordered = False
    for i in range(1, len(t.inits)):
        if f.slot_index(t.inits[i].slot) < f.slot_index(t.inits[i - 1].slot):
            reordered = True
    if reordered:
        st.reordered_tasks += 1
    for i in range(len(t.inits)):
        ref it = t.inits[i]
        var si = f.slot_index(it.slot)
        if f.slots[si].has_geom:
            st.geom += BATCH
        if f.init_target_kind(it.region) == INIT_TARGET_SLOT:
            st.stacks += BATCH
            continue
        ref reg = f.regions[f.region_index(it.region)]
        if not f.slots[si].has_geom:
            continue
        if reg.contact.byte_length() == 0:
            st.table_off += BATCH
        elif it.inside:
            st.in_fixture += BATCH
        else:
            st.on_fixture += BATCH


def _parity[T: PlacementTable](
    t: TaskSpec,
    f: FamilySpec,
    frames: List[RegionFrame],
    radii: List[Float64],
    mut st: Stats,
) raises:
    """The kernel on BATCH lanes against `sample_placements` on each lane."""
    comptime NQ = T.NQ
    comptime NV = T.NV
    comptime L_Q = Layout.row_major(BATCH, NQ)
    comptime L_V = Layout.row_major(BATCH, NV)
    comptime L_M = Layout.row_major(BATCH, METADATA_SIZE)
    var words = init_region_words(t, f)
    var qs = TensorImpl[DT].alloc(BATCH * NQ)
    var vs = TensorImpl[DT].alloc(BATCH * NV)
    var ms = TensorImpl[DT].alloc(BATCH * METADATA_SIZE)
    for i in range(BATCH * NQ):
        qs.data[i] = Scalar[DT](QPOS_SENTINEL)
    for i in range(BATCH * NV):
        vs.data[i] = Scalar[DT](QVEL_SENTINEL)
    for e in range(BATCH):
        for k in range(METADATA_SIZE):
            ms.data[e * METADATA_SIZE + k] = Scalar[DT](META_CANARY)
        for j in range(META_INIT_SLOTS):
            ms.data[e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j] = Scalar[
                DT
            ](0)
        for j in range(len(words)):
            ms.data[e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j] = Scalar[
                DT
            ](words[j])
    var meta_before = List[Float64]()
    for i in range(BATCH * METADATA_SIZE):
        meta_before.append(Float64(ms.data[i]))
    var qt = qs.lt["cpu", L_Q]()
    var vt = vs.lt["cpu", L_V]()
    var mt = ms.lt["cpu", L_M]()

    for lane in range(BATCH):
        place_free_slots[T, DT, BATCH, NQ, NV](qt, vt, mt, lane, SEED)
    for i in range(BATCH * METADATA_SIZE):
        if Float64(ms.data[i]) != meta_before[i]:
            st.meta_touched += 1

    for lane in range(BATCH):
        var rep = SampleReport()
        var placed = sample_placements(
            t, f, frames, radii, UInt64(SEED), lane, rep
        )
        st.lanes += 1
        st.rejections += rep.attempts - rep.accepted
        st.clamped += rep.clamped
        st.exempt += rep.exempt
        for j in range(T.N_FREE):
            var si = T.free_slot(j)
            var qa = T.free_qadr(j)
            var da = T.free_dadr(j)
            var k = -1
            for p in range(len(placed)):
                if placed[p].slot == si:
                    k = p
            var base = lane * NQ + qa
            var vbase = lane * NV + da
            if k < 0:
                st.left_alone += 1
                var ok = True
                for w in range(7):
                    if Float64(qs.data[base + w]) != QPOS_SENTINEL:
                        ok = False
                for w in range(6):
                    if Float64(vs.data[vbase + w]) != QVEL_SENTINEL:
                        ok = False
                if not ok:
                    st.left_alone_bad += 1
                    print("      ", t.name, "lane", lane, "slot", si,
                          ": the host did not place it and the device wrote it")
                continue
            st.placements += 1
            var want = List[Float64]()
            want.append(placed[k].x)
            want.append(placed[k].y)
            want.append(placed[k].z)
            want.append(1.0)
            want.append(0.0)
            want.append(0.0)
            want.append(0.0)
            var this_bad = False
            for w in range(7):
                var got = Float64(qs.data[base + w])
                var dd = abs(got - want[w])
                st.coords += 1
                if dd == 0.0:
                    st.exact += 1
                if dd > st.worst:
                    st.worst = dd
                if dd > TOL:
                    st.bad += 1
                    this_bad = True
            for w in range(6):
                if Float64(vs.data[vbase + w]) != 0.0:
                    st.bad += 1
                    this_bad = True
            if this_bad and st.bad <= 20:
                print("      ", t.name, "lane", lane, "slot", si,
                      ": device (", Float64(qs.data[base]), ",",
                      Float64(qs.data[base + 1]), ",",
                      Float64(qs.data[base + 2]), ") host (", placed[k].x,
                      ",", placed[k].y, ",", placed[k].z, ")")


def _family[T: PlacementTable](
    name: String, mut ta: Tally, mut st: Stats, mut refused: List[String],
    mut should_refuse: List[String],
) raises:
    """Sections 4 and 5 for one generated table."""
    var f = load_family(String(FAMILY_DIR) + "/" + name + ".family")
    var sc = _scene(f, 32768)
    var radii = _host_radii(f, 0.02)
    var drift = placement_table_drift[T](
        f, sc.addrs, sc.frames, sc.moves, radii, sc.nq, sc.nv
    )
    if len(drift) > 0:
        for i in range(len(drift)):
            if i < 6:
                print("      ", name, ":", drift[i])
    ta.check(len(drift) == 0,
             name + ": the generated table matches the family, the addresses"
             " and FK exactly, and the moving-region oracle")
    var tasks = _tasks_of(name)
    for i in range(len(tasks)):
        var t = load_task(TASK_DIR + tasks[i] + ".task")
        validate_task_against_family(t, f)
        st.tasks += 1
        var on_moving = False
        for k in range(len(t.inits)):
            var r = f.region_index(t.inits[k].region)
            if r >= 0 and sc.moves[r]:
                on_moving = True
        if on_moving:
            should_refuse.append(t.name)
        var raised = False
        try:
            require_device_placement[T](t, f)
        except e:
            raised = True
            refused.append(t.name)
            print("      refused:", t.name, "—", String(e)[byte=0:110])
        if raised:
            st.refused += 1
            continue
        _coverage(t, f, st)
        _parity[T](t, f, sc.frames, radii, st)


# ── section 7's scene: a stack wider than its reference ─────────────────────
#
# Binary fractions throughout, so the parsed `.family` and this table are the
# same bits and the drift check can stay exact.
comptime SYNTH_FAMILY = String(
    "schema_version=1\nfamily=synth_stack\nbase=b.xml\nhorizon=10\n"
    "slot=arena:static:a.xml:0.0,0.0,0.0\n"
    "slot=ref:free:p.xml\nslot=wide:free:p.xml\nslot=near:free:p.xml\n"
    "slot_geom=ref:-0.015625,0.015625,0.0078125\n"
    "slot_geom=wide:-0.015625,0.015625,0.078125\n"
    "slot_geom=near:-0.015625,0.015625,0.0078125\n"
    "region=left:site:s0:-0.00390625,-0.00390625,0.00390625,0.00390625\n"
    "region=right:site:s1:0.046875,-0.00390625,0.05078125,0.00390625\n"
)
comptime SYNTH_TASK = String(
    "schema_version=1\ntask=synth_stack_t\nfamily=synth_stack\n"
    "goal=On(wide, ref)\n"
    "active=arena\nactive=ref\nactive=wide\nactive=near\n"
    "init=ref@left\ninit=wide@ref\ninit=near@right\n"
)
comptime SYNTH_Z: Float64 = 0.8125


struct SynthStackPlacement(PlacementTable):
    comptime N_SLOTS: Int = 4
    comptime N_FREE: Int = 3
    comptime N_REGIONS: Int = 2
    comptime NQ: Int = 21
    comptime NV: Int = 18

    @staticmethod
    def free_slot(j: Int) -> Int:
        return j + 1

    @staticmethod
    def free_qadr(j: Int) -> Int:
        return 7 * j

    @staticmethod
    def free_dadr(j: Int) -> Int:
        return 6 * j

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.015625)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.078125 if j == 1 else 0.0078125)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.015625)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.015625)

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](SYNTH_Z)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        return True

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.046875 if r == 1 else -0.00390625)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.00390625)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.05078125 if r == 1 else 0.00390625)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.00390625)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_moves(r: Int) -> Bool:
        return False


# ── section 7b's scene: a fixture, its top, its interior, and a table zone ──
#
# ⚠ WHAT THE CORPUS DOES NOT REACH. `In` a fixture region and an inset that
# inverts both occur in exactly one corpus placement — `libero_spatial`'s bowl
# in the top drawer — and that task is REFUSED (its region moves). No corpus
# draw overlaps an object in a different region either. So those three rules
# would otherwise be in the kernel and never compared.
comptime SYNTH_FIXTURE_FAMILY = String(
    "schema_version=1\nfamily=synth_fixture\nbase=b.xml\nhorizon=10\n"
    "slot=arena:static:a.xml:0.0,0.0,0.0\n"
    "slot=box:static:x.xml:0.0,0.0,0.75\n"
    "slot=a:free:p.xml\nslot=b:free:p.xml\nslot=c:free:p.xml\n"
    "slot_geom=box:-0.046875,0.046875,0.125\n"
    "slot_geom=a:-0.015625,0.015625,0.0078125\n"
    "slot_geom=b:-0.015625,0.015625,0.03125\n"
    "slot_geom=c:-0.015625,0.015625,0.0078125\n"
    "region=top:box:s0:-0.0625,-0.0625,0.0625,0.0625:0.0078125:box\n"
    "region=inner:box:s1:-0.0234375,-0.0234375,0.0234375,0.0234375"
    ":0.0625:box\n"
    "region=zone:site:s2:-0.00390625,-0.00390625,0.00390625,0.00390625\n"
)
comptime SYNTH_FIXTURE_TASK = String(
    "schema_version=1\ntask=synth_fixture_t\nfamily=synth_fixture\n"
    "goal=In(b, inner)\n"
    "active=arena\nactive=box\nactive=a\nactive=b\nactive=c\n"
    "init=a@top\ninit=b@inner:in\ninit=c@zone\n"
)


struct SynthFixturePlacement(PlacementTable):
    comptime N_SLOTS: Int = 5
    comptime N_FREE: Int = 3
    comptime N_REGIONS: Int = 3
    comptime NQ: Int = 21
    comptime NV: Int = 18

    @staticmethod
    def free_slot(j: Int) -> Int:
        return j + 2

    @staticmethod
    def free_qadr(j: Int) -> Int:
        return 7 * j

    @staticmethod
    def free_dadr(j: Int) -> Int:
        return 6 * j

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.015625)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.03125 if j == 1 else 0.0078125)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.015625)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.015625)

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.75 if r == 1 else SYNTH_Z)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        return True

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.0625)
        return Scalar[DTYPE](-0.0234375 if r == 1 else -0.00390625)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.0625)
        return Scalar[DTYPE](-0.0234375 if r == 1 else -0.00390625)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.0625)
        return Scalar[DTYPE](0.0234375 if r == 1 else 0.00390625)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.0625)
        return Scalar[DTYPE](0.0234375 if r == 1 else 0.00390625)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        return r != 2

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        return r != 2

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.046875 if r != 2 else 0.0)

    @staticmethod
    def region_moves(r: Int) -> Bool:
        return False


def main() raises:
    print("=== device placement vs the host sampler ===")
    var ta = Tally()

    # ── 1. the init block, and so101_tabletop's table ─────────────────────
    print("--- 1. meta's init block, and so101_tabletop's hand-written table ---")
    ta.check(
        META_IDX_INIT_REGION_0 + META_INIT_SLOTS == METADATA_SIZE,
        "the init block is contiguous and ENDS `meta` ("
        + String(META_IDX_INIT_REGION_0) + ".." + String(METADATA_SIZE - 1)
        + "), so widening it moved no other word",
    )
    ta.check(META_IDX_LS_EVAL < META_IDX_INIT_REGION_0,
             "every older word sits below the init block")

    var f = load_family(SO101_FAMILY)
    var sc = _scene(f, SO_ARM101_NMESH_VERTS)
    var radii = _host_radii(f, So101TabletopConfig.SLOT_RADIUS)
    var drift = placement_table_drift[So101TabletopPlacement](
        f, sc.addrs, sc.frames, sc.moves, radii, sc.nq, sc.nv
    )
    for i in range(len(drift)):
        print("      so101_tabletop:", drift[i])
    ta.check(len(drift) == 0,
             "So101TabletopPlacement matches the family, addresses and FK")

    # ⚠ THE RADIUS IS READ FROM THE PROP'S OWN ASSET, not from the composed
    # scene. For this family it is both the clash radius and the resting
    # height, so a value that is not the box's half-size starts every prop
    # floating or inside the table.
    var cube = parse_model_runtime("mojo_rl/tasks/assets/props/cube.xml")
    var ok_rad = False
    for i in range(len(cube.geoms)):
        if cube.geoms[i].half_x == So101TabletopConfig.SLOT_RADIUS:
            ok_rad = True
    ta.check(ok_rad,
             "SLOT_RADIUS is the prop asset's own half-size (resting height)")

    # ── 2. so101_tabletop, device vs host ─────────────────────────────────
    print()
    print("--- 2. so101_tabletop: device vs host,", BATCH, "lanes ---")
    var st0 = Stats()
    var names = List[String]()
    names.append(String("so101_reach_brick"))
    names.append(String("so101_lift_brick"))
    names.append(String("so101_gather_bricks"))
    names.append(String("so101_settle_brick"))
    for n in range(len(names)):
        var t = load_task(TASK_DIR + names[n] + ".task")
        validate_task_against_family(t, f)
        require_device_placement[So101TabletopPlacement](t, f)
        _parity[So101TabletopPlacement](t, f, sc.frames, radii, st0)
    print("      placements", st0.placements, " coordinates", st0.coords,
          " exact", st0.exact, " worst", st0.worst, " rejections",
          st0.rejections)
    if st0.placements == 0:
        raise Error(
            "device placement: so101_tabletop placed NOTHING, so the parity"
            " below compared nothing"
        )
    ta.check(st0.bad == 0 and st0.left_alone_bad == 0 and st0.meta_touched == 0,
             "so101_tabletop: every coordinate agrees, every unplaced slot is"
             " untouched, and meta is not written")

    # ── 3. an untouched meta parks everything ─────────────────────────────
    #
    # ⚠⚠ THE CASE A DRIVER FALLS INTO BY FORGETTING. `Data.__init__` uploads a
    # ZERO-FILLED `meta`, so zero is what every lane reads until something
    # writes the init words. Zero must mean "not placed".
    print()
    print("--- 3. an untouched meta leaves every free slot where it was ---")
    comptime NQ0 = So101TabletopPlacement.NQ
    comptime NV0 = So101TabletopPlacement.NV
    var zs = TensorImpl[DT].alloc(BATCH * NQ0)
    var zv = TensorImpl[DT].alloc(BATCH * NV0)
    var zm = TensorImpl[DT].alloc(BATCH * METADATA_SIZE)
    for i in range(BATCH * NQ0):
        zs.data[i] = Scalar[DT](QPOS_SENTINEL)
    for i in range(BATCH * NV0):
        zv.data[i] = Scalar[DT](QVEL_SENTINEL)
    for i in range(BATCH * METADATA_SIZE):
        zm.data[i] = Scalar[DT](0)
    var zqt = zs.lt["cpu", Layout.row_major(BATCH, NQ0)]()
    var zvt = zv.lt["cpu", Layout.row_major(BATCH, NV0)]()
    var zmt = zm.lt["cpu", Layout.row_major(BATCH, METADATA_SIZE)]()
    # ⚠ THE FIVE UNUSED TENSORS ARE SIZE-1 SCRATCH, named so they outlive the
    # call — the hook ends with `_ = joints` / `_ = bodies` / ...
    var dj = TensorImpl[DT].alloc(MODEL_JOINT_SIZE)
    var dm3 = TensorImpl[DT].alloc(BATCH * 3)
    var dm4 = TensorImpl[DT].alloc(BATCH * 4)
    var db = TensorImpl[DT].alloc(MODEL_BODY_SIZE)
    var dg = TensorImpl[DT].alloc(MODEL_GEOM_SIZE)
    for lane in range(BATCH):
        So101TabletopConfig.init_qpos_gpu[DT, BATCH, NQ0, 1, NV0, 1, 1](
            zqt, zvt,
            dj.lt["cpu", Layout.row_major(1, MODEL_JOINT_SIZE)](),
            dm3.lt["cpu", Layout.row_major(BATCH, 3)](),
            dm4.lt["cpu", Layout.row_major(BATCH, 4)](),
            db.lt["cpu", Layout.row_major(1, MODEL_BODY_SIZE)](),
            dg.lt["cpu", Layout.row_major(1, MODEL_GEOM_SIZE)](),
            zmt, lane, SEED,
        )
    var parked_ok = True
    for i in range(BATCH * NQ0):
        if Float64(zs.data[i]) != QPOS_SENTINEL:
            parked_ok = False
    for i in range(BATCH * NV0):
        if Float64(zv.data[i]) != QVEL_SENTINEL:
            parked_ok = False
    ta.check(parked_ok,
             "through So101TabletopConfig.init_qpos_gpu, a zero meta writes"
             " NO qpos or qvel word")

    # ── 4 + 5. every LIBERO family ────────────────────────────────────────
    print()
    print("--- 4 + 5. every LIBERO family: its table, then every task ---")
    var fams = _libero_families()
    ta.check(len(fams) == N_LIBERO_FAMILIES,
             String(len(fams)) + " libero*.family files, and this gate imports "
             + String(N_LIBERO_FAMILIES) + " tables")
    var st = Stats()
    var refused = List[String]()
    var should = List[String]()
    var seen = List[String]()
    for i in range(len(fams)):
        ref nm = fams[i]
        seen.append(nm)
        if nm == "libero_goal":
            _family[LiberoGoalPlacement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene1":
            _family[LiberoKitchenScene1Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene2":
            _family[LiberoKitchenScene2Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene3":
            _family[LiberoKitchenScene3Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene4":
            _family[LiberoKitchenScene4Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene5":
            _family[LiberoKitchenScene5Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene6":
            _family[LiberoKitchenScene6Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene7":
            _family[LiberoKitchenScene7Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene8":
            _family[LiberoKitchenScene8Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene9":
            _family[LiberoKitchenScene9Placement](nm, ta, st, refused, should)
        elif nm == "libero_kitchen_scene10":
            _family[LiberoKitchenScene10Placement](nm, ta, st, refused, should)
        elif nm == "libero_living_room_scene1":
            _family[LiberoLivingRoomScene1Placement](
                nm, ta, st, refused, should
            )
        elif nm == "libero_living_room_scene2":
            _family[LiberoLivingRoomScene2Placement](
                nm, ta, st, refused, should
            )
        elif nm == "libero_living_room_scene3":
            _family[LiberoLivingRoomScene3Placement](
                nm, ta, st, refused, should
            )
        elif nm == "libero_living_room_scene4":
            _family[LiberoLivingRoomScene4Placement](
                nm, ta, st, refused, should
            )
        elif nm == "libero_living_room_scene5":
            _family[LiberoLivingRoomScene5Placement](
                nm, ta, st, refused, should
            )
        elif nm == "libero_living_room_scene6":
            _family[LiberoLivingRoomScene6Placement](
                nm, ta, st, refused, should
            )
        elif nm == "libero_object":
            _family[LiberoObjectPlacement](nm, ta, st, refused, should)
        elif nm == "libero_spatial":
            _family[LiberoSpatialPlacement](nm, ta, st, refused, should)
        elif nm == "libero_study_scene1":
            _family[LiberoStudyScene1Placement](nm, ta, st, refused, should)
        elif nm == "libero_study_scene2":
            _family[LiberoStudyScene2Placement](nm, ta, st, refused, should)
        elif nm == "libero_study_scene3":
            _family[LiberoStudyScene3Placement](nm, ta, st, refused, should)
        elif nm == "libero_study_scene4":
            _family[LiberoStudyScene4Placement](nm, ta, st, refused, should)
        else:
            ta.check(False, nm + ": a LIBERO family with no table in this gate"
                     " — run `pixi run gen-placement-tables` and import it")

    print()
    print("      tasks", st.tasks, " refused", st.refused, " lanes", st.lanes,
          " placements", st.placements)
    print("      coordinates", st.coords, " exact", st.exact, " worst",
          st.worst, " differing", st.bad)
    print("      unplaced free slots", st.left_alone, " written anyway",
          st.left_alone_bad, " meta words written", st.meta_touched)
    ta.check(st.tasks == N_LIBERO_TASKS,
             String(st.tasks) + " LIBERO tasks visited (the importer's "
             + String(N_LIBERO_TASKS) + ")")
    if st.placements == 0:
        raise Error(
            "device placement: no LIBERO placement was compared, so 'no"
            " differences' below would say nothing"
        )
    ta.check(st.bad == 0,
             "the device and the host agree on every coordinate of every"
             " placement, within " + String(TOL) + " m")
    ta.check(st.left_alone_bad == 0,
             "a free slot the host does not place, the device does not write")
    ta.check(st.meta_touched == 0, "the kernel never writes meta")

    var same = len(refused) == len(should)
    for i in range(len(refused)):
        var hit = False
        for k in range(len(should)):
            if should[k] == refused[i]:
                hit = True
        if not hit:
            same = False
    ta.check(len(should) >= 1,
             String(len(should)) + " task(s) draw in a region the FK oracle"
             " finds moving — the refusal below is not vacuous")
    ta.check(same,
             "the refused tasks are EXACTLY those drawing in a moving region ("
             + String(len(refused)) + " refused)")

    # ── 6. coverage: the corpus reaches every rule ────────────────────────
    print()
    print("--- 6. the rules the CORPUS placements reach (checked after 7) ---")
    print("      slot_geom", st.geom, " table z offset", st.table_off,
          " On a fixture", st.on_fixture, " In a fixture", st.in_fixture,
          " stacks", st.stacks)
    print("      tasks whose walk is not slot order", st.reordered_tasks,
          " rejections", st.rejections, " clamped axes", st.clamped,
          " exempt overlaps", st.exempt)

    # ── 7. the stack exemption, and the host crash ────────────────────────
    #
    # ⚠⚠ THE HOST USED TO INDEX `f.regions[-1]` HERE AND MOJO ASSERTS ON THAT.
    # A stack's `of_region` is -1; a TABLE-region draw that overlapped a stack
    # evaluated `f.regions[ri_j]` and `mojo run` crashed. No corpus seed reached
    # it: a stack stands on its reference, which rejects the draw first unless
    # the stack is the WIDER of the two — so this scene makes it wider.
    print()
    print("--- 7. a draw overlapping a stack wider than its reference ---")
    var sf = parse_family(SYNTH_FAMILY)
    var stask = parse_task(SYNTH_TASK)
    validate_task_against_family(stask, sf)
    var sframes = List[RegionFrame]()
    sframes.append(RegionFrame(0.0, 0.0, SYNTH_Z))
    sframes.append(RegionFrame(0.0, 0.0, SYNTH_Z))
    var saddrs = List[SlotAddress]()
    saddrs.append(SlotAddress(-1, -1))
    for j in range(3):
        saddrs.append(SlotAddress(7 * j, 6 * j))
    var smoves = List[Bool]()
    smoves.append(False)
    smoves.append(False)
    var sradii = _host_radii(sf, 0.02)
    var sdrift = placement_table_drift[SynthStackPlacement](
        sf, saddrs, sframes, smoves, sradii, 21, 18
    )
    for i in range(len(sdrift)):
        print("      synth:", sdrift[i])
    ta.check(len(sdrift) == 0, "the synthetic table matches its family")
    require_device_placement[SynthStackPlacement](stask, sf)
    var s7 = Stats()
    _coverage(stask, sf, s7)
    _parity[SynthStackPlacement](stask, sf, sframes, sradii, s7)
    print("      placements", s7.placements, " exempt overlaps", s7.exempt,
          " worst", s7.worst)
    ta.check(s7.exempt >= BATCH,
             "every lane's 'near' draw overlaps the wide stack and is exempted"
             " (" + String(s7.exempt) + ")")
    ta.check(s7.placements == 3 * BATCH and s7.bad == 0,
             "the host places all three without crashing, and the device"
             " agrees on every coordinate")

    # ── 7b. a fixture: On its top, In its interior, beside a table zone ───
    print()
    print("--- 7b. On a fixture, In a fixture too narrow, and a table zone ---")
    var ff = parse_family(SYNTH_FIXTURE_FAMILY)
    var ftask = parse_task(SYNTH_FIXTURE_TASK)
    validate_task_against_family(ftask, ff)
    var fframes = List[RegionFrame]()
    fframes.append(RegionFrame(0.0, 0.0, SYNTH_Z))
    fframes.append(RegionFrame(0.0, 0.0, 0.75))
    fframes.append(RegionFrame(0.0, 0.0, SYNTH_Z))
    var faddrs = List[SlotAddress]()
    faddrs.append(SlotAddress(-1, -1))
    faddrs.append(SlotAddress(-1, -1))
    for j in range(3):
        faddrs.append(SlotAddress(7 * j, 6 * j))
    var fmoves = List[Bool]()
    for _ in range(3):
        fmoves.append(False)
    var fradii = _host_radii(ff, 0.02)
    var fdrift = placement_table_drift[SynthFixturePlacement](
        ff, faddrs, fframes, fmoves, fradii, 21, 18
    )
    for i in range(len(fdrift)):
        print("      synth_fixture:", fdrift[i])
    ta.check(len(fdrift) == 0, "the synthetic fixture table matches its family")
    require_device_placement[SynthFixturePlacement](ftask, ff)
    var s7b = Stats()
    _coverage(ftask, ff, s7b)
    _parity[SynthFixturePlacement](ftask, ff, fframes, fradii, s7b)
    print("      placements", s7b.placements, " On", s7b.on_fixture, " In",
          s7b.in_fixture, " clamped", s7b.clamped, " exempt", s7b.exempt,
          " worst", s7b.worst)
    ta.check(s7b.placements == 3 * BATCH and s7b.bad == 0
             and s7b.left_alone_bad == 0,
             "all three placed, and the device agrees on every coordinate")

    # ── 6 (checked). every rule is reached by SOMETHING compared above ────
    #
    # ⚠ THE SOURCE IS PRINTED BESIDE EACH COUNT, because "reached" by a
    # synthetic scene is a weaker statement than reached by the corpus, and a
    # combined total would hide which one it is.
    print()
    print("--- 6. every rule the kernel mirrors, reached (corpus + synthetic) ---")
    var sy = Stats()
    sy.geom = s7.geom + s7b.geom
    sy.table_off = s7.table_off + s7b.table_off
    sy.on_fixture = s7.on_fixture + s7b.on_fixture
    sy.in_fixture = s7.in_fixture + s7b.in_fixture
    sy.stacks = s7.stacks + s7b.stacks
    sy.reordered_tasks = s7.reordered_tasks + s7b.reordered_tasks
    sy.rejections = s7.rejections + s7b.rejections
    sy.clamped = s7.clamped + s7b.clamped
    sy.exempt = s7.exempt + s7b.exempt
    print("      corpus   : In", st.in_fixture, " clamped", st.clamped,
          " exempt", st.exempt)
    print("      synthetic: In", sy.in_fixture, " clamped", sy.clamped,
          " exempt", sy.exempt)
    ta.check(st.geom > 0, "rule 3: resting heights from slot_geom=")
    ta.check(st.table_off > 0, "rule 4: TABLE_Z_OFFSET on a table region")
    ta.check(st.on_fixture > 0, "rule 4: a fixture's top_z for On")
    ta.check(st.in_fixture + sy.in_fixture > 0,
             "rule 4: no top_z for In (corpus " + String(st.in_fixture)
             + ", synthetic " + String(sy.in_fixture) + ")")
    ta.check(st.stacks > 0, "rule 7: stacks")
    ta.check(st.reordered_tasks > 0,
             "rule 2: a walk that is not slot order (order_inits)")
    ta.check(st.rejections > 0, "rule 6: the clash test rejects draws")
    ta.check(st.clamped + sy.clamped > 0,
             "rule 5: an inset that inverts and is clamped (corpus "
             + String(st.clamped) + ", synthetic " + String(sy.clamped) + ")")
    ta.check(st.exempt + sy.exempt > 0,
             "rule 6: overlaps exempted across regions or against a stack"
             " (corpus " + String(st.exempt) + ", synthetic "
             + String(sy.exempt) + ")")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "device placement: " + String(ta.failures) + " of "
            + String(ta.checks) + " check(s) failed"
        )
    print("=== PASS ===")
