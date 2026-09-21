"""Every LIBERO family's resting contact count — the measurement behind its
batched `max_contacts`.

    pixi run mojo run -I . tools/tasks/libero_contact_budget.mojo            # all 23
    pixi run mojo run -I . tools/tasks/libero_contact_budget.mojo libero_object

## ⚠⚠ THE BATCHED ENV TRUNCATES AT `max_contacts`, SILENTLY

`broadphase_sap.mojo` and `contact_detection.mojo` stop adding contacts at the
cap and hand the truncated set to the solver. A cap below what a scene needs
drops a prop through the table on some lanes and the step still returns. The
three hand-written LIBERO model defs say of their own 64 / 96 "a budget, not a
measurement"; the twenty scene families are generated, so their number has to
come from somewhere that is not a guess.

## WHAT IS MEASURED

Per family, per task, per seed lane: the host reset every CPU driver runs
(`base_qpos`, `jinit=` draws, then `sample_placements` on the frames after FK —
`libero_viewer.reset_episode`'s order), the OSC controller anchored on it, then
`SETTLE + STEPS` control steps of `zeros(7)`, counting `ncon` after every one
under a cap (`CAP`) far above any scene's. The row is the MAXIMUM.

⚠ A RESTING COUNT, NOT A MANIPULATION COUNT. A null action never closes the
gripper on anything, so the grasp's contacts are not in it. The margin for
those is the generator's rule (`gen_libero_envs.mojo`), and it is sized from
`libero_goal`'s demonstrations, which do grasp.

⚠ A ROW THAT REACHES `CAP` IS REFUSED, NOT RECORDED — a saturated count is a
lower bound on the thing being measured.

Writes `noeira/tasks/libero/contact_budget.kv`: `<family>=<max>,<task>`.
"""

from std.os import listdir
from std.os.path import exists
from std.sys import argv

from noeira.physics3d.fields import (
    Data, Model, DynDims, DynamicsScratch, SpecFields,
)
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
    spec_fields_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from noeira.physics3d.studio.stepping import StudioIntegEll
from noeira.physics3d.dynamics.actuation import apply_actions_fields
from noeira.physics3d.dynamics.osc_pose import (
    OscPose, OscPoseConfig, ARM_DOF, OSC_ACTION_DIM,
)
from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, TaskSpec, FamilySpec,
)
from noeira.tasks.family import scene_path
from noeira.tasks.eval import region_sites
from noeira.tasks.sampler import (
    sample_placements, sample_joint_inits, RegionFrame, SampleReport,
)
from noeira.tasks.init_table import load_init_table
from noeira.tasks.reset import (
    free_slot_addresses, reset_slots, joint_init_addresses,
    joint_init_dof_addresses, apply_joint_inits,
)


comptime DT = DType.float64
comptime FAMILY_DIR = "noeira/tasks/families"
comptime TASK_DIR = "noeira/tasks/tasks/"
comptime OUT_PATH = "noeira/tasks/libero/contact_budget.kv"
comptime CAP = 512
comptime SUBSTEPS = 25
comptime SETTLE = 5
comptime STEPS = 20
comptime SEEDS = 3
comptime PROP_RADIUS: Float64 = 0.02
comptime INIT_ROWS_PER_TASK = 5


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


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if names[i] == want:
            return i
    raise Error("contact budget: no '" + want + "' in the scene")


@fieldwise_init
struct FamilyMax(Copyable, Movable):
    var name: String
    var max_ncon: Int
    var task: String
    var episodes: Int


def _run_null(
    mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims],
    mut osc: OscPose,
    mut scratch: DynamicsScratch[DT, DynDims, 1],
    mut integ: StudioIntegEll,
    sf: SpecFields[DT, DynDims],
    null_action: List[Float64],
    mut act: List[Scalar[DT]],
    nq: Int, nv: Int, timestep: Float64, what: String,
) raises -> Int:
    """`SETTLE + STEPS` control steps of `zeros(7)`; the peak `ncon`, counted
    after EVERY substep — the landing transient lasts about ten of them."""
    var peak = 0
    for _ in range(SETTLE + STEPS):
        for s in range(SUBSTEPS):
            osc.update(d, m, scratch)
            if s == 0:
                osc.set_goal(null_action, d, m)
            var ctrl = osc.run(null_action, d, m, scratch)
            for i in range(nv):
                d.qfrc.data[i] = Scalar[DT](0)
            apply_actions_fields[DT](sf, d, ctrl, act, timestep)
            integ.step["cpu"](d, m)
            var nc = Int(d.meta.data[META_IDX_NUM_CONTACTS])
            if nc >= CAP:
                raise Error(
                    what + ": ncon reached the cap " + String(CAP)
                    + " — a saturated count is not a measurement"
                )
            if nc > peak:
                peak = nc
    for i in range(nq):
        var q = Float64(d.qpos.data[i])
        if q != q:
            raise Error(what + ": qpos went NaN")
    return peak


def _measure(name: String) raises -> FamilyMax:
    var f = load_family(String(FAMILY_DIR) + "/" + name + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=CAP, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=CAP, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var scratch = DynamicsScratch[DT, DynDims, 1](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var integ = StudioIntegEll(dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var ns = dims.get_nsite()
    var nact = dims.get_nact()
    var timestep = fmd.timestep

    var jt = List[Int]()
    var jqn = List[Int]()
    var jvn = List[Int]()
    for k in range(len(fmd.joints)):
        jt.append(fmd.joints[k].jnt_type)
        jqn.append(fmd.joints[k].nq)
        jvn.append(fmd.joints[k].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jqn, jvn)
    var rsites = region_sites(f, fmd.site_names)

    # the controller, exactly as `libero_eval` builds it
    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for k in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[k].nq
        da += fmd.joints[k].nv
    var dof = List[Int]()
    var qadr = List[Int]()
    var jidx = List[Int]()
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ji = _index(fmd.joint_names, String("robot_joint") + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
        jidx.append(ji)
        var ai = _index(
            fmd.actuator_names, String("robot_torq_j") + String(j + 1)
        )
        act_idx.append(ai)
        tmin.append(fmd.actuators[ai].ctrl_min)
        tmax.append(fmd.actuators[ai].ctrl_max)
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var site_body = fmd.sites[site].body_id
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))
    var osc = OscPose(
        dof^, qadr^, jidx^, tmin^, tmax^, act_idx.copy(), site, site_body,
        ga1, ga2,
        fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
        fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
        OscPoseConfig(), nact, nq, nv,
    )
    var act = List[Scalar[DT]](
        length=nact if nact > 0 else 1, fill=Scalar[DT](0)
    )
    var null_action = List[Float64](length=OSC_ACTION_DIM, fill=0.0)
    var radii = List[Float64](length=len(f.slots), fill=PROP_RADIUS)

    var out = FamilyMax(name, 0, String(""), 0)
    var tasks = _tasks_of(name)
    for ti in range(len(tasks)):
        var t = load_task(TASK_DIR + tasks[ti] + ".task")
        validate_task_against_family(t, f)
        var jq = joint_init_addresses(t, fmd.joint_names, jqn)
        var jd = joint_init_dof_addresses(t, fmd.joint_names, jvn)
        var task_max = 0
        for lane in range(SEEDS):
            var seed = UInt64(1000 + ti)
            # ── the host reset (`libero_viewer.reset_episode`'s order) ──
            for i in range(nq):
                d.qpos.data[i] = Scalar[DT](0)
            for i in range(nv):
                d.qvel.data[i] = Scalar[DT](0)
            for i in range(len(f.base_qpos)):
                d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
            var jv = sample_joint_inits(t, seed, lane)
            for k in range(len(jv)):
                d.qpos.data[jq[k]] = Scalar[DT](jv[k])
            forward_kinematics["cpu", DT, DynDims, 1](d, m)
            var frames = List[RegionFrame]()
            for r in range(len(f.regions)):
                var si = rsites[r]
                frames.append(RegionFrame(
                    Float64(d.site_xpos.data[si * 3]),
                    Float64(d.site_xpos.data[si * 3 + 1]),
                    Float64(d.site_xpos.data[si * 3 + 2]),
                ))
            var rep = SampleReport()
            var placed = sample_placements(t, f, frames, radii, seed, lane, rep)
            var qpos = List[Float64]()
            for i in range(nq):
                qpos.append(Float64(d.qpos.data[i]))
            var qvel = List[Float64](length=nv, fill=0.0)
            reset_slots(t, f, placed, addrs, qpos, qvel)
            apply_joint_inits(t, jq, jv, qpos, qvel, jd)
            for i in range(nq):
                d.qpos.data[i] = Scalar[DT](qpos[i])
            for i in range(nv):
                d.qvel.data[i] = Scalar[DT](qvel[i])
            forward_kinematics["cpu", DT, DynDims, 1](d, m)
            osc.update(d, m, scratch)
            osc.reset(d, m)
            out.episodes += 1

            var peak = _run_null(
                d, m, osc, scratch, integ, sf, null_action, act, nq, nv,
                timestep, name + " / " + tasks[ti],
            )
            if peak > task_max:
                task_max = peak
        print("   ", tasks[ti], " max ncon", task_max)
        if task_max > out.max_ncon:
            out.max_ncon = task_max
            out.task = tasks[ti]

    # ── LIBERO's own start: the frozen rows, objects in the AIR ──────────
    # ⚠ `libero_eval`'s order: base pose, the controller anchored there, THEN
    # the row. The props fall ~7 cm in the five settle steps (its header), a
    # harder landing than the sampler's, so where a table exists it is
    # measured too — and it is the protocol a benchmark number runs under.
    var table_path = String("build/init/") + name + ".init.h5"
    if not exists(table_path):
        print("  (no frozen inits at", table_path, "— sampled resets only)")
        return out^
    var full = load_init_table(table_path, f.name, nq, nv)
    var tbl = full.prefix_per_task(INIT_ROWS_PER_TASK)
    var init_max = 0
    for row in range(tbl.n_rows()):
        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](0)
        for i in range(nv):
            d.qvel.data[i] = Scalar[DT](0)
        for i in range(len(f.base_qpos)):
            d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        osc.update(d, m, scratch)
        osc.reset(d, m)
        var qpos = List[Float64](length=nq, fill=0.0)
        var qvel = List[Float64](length=nv, fill=0.0)
        tbl.apply(row, qpos, qvel)
        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](qpos[i])
        for i in range(nv):
            d.qvel.data[i] = Scalar[DT](qvel[i])
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        out.episodes += 1
        var peak = _run_null(
            d, m, osc, scratch, integ, sf, null_action, act, nq, nv, timestep,
            name + " / frozen row " + String(row),
        )
        if peak > init_max:
            init_max = peak
        if peak > out.max_ncon:
            out.max_ncon = peak
            out.task = String("frozen row ") + String(row) + " (" + tbl.task_label(row) + ")"
    print("    frozen inits:", tbl.n_rows(), "rows, max ncon", init_max)
    return out^


def main() raises:
    var args = argv()
    var only = String("")
    if len(args) > 2:
        raise Error("usage: libero_contact_budget.mojo [family]")
    if len(args) == 2:
        only = String(args[1])
    var fams = _libero_families()
    if only != "":
        var found = False
        for i in range(len(fams)):
            if fams[i] == only:
                found = True
        if not found:
            raise Error("contact budget: no family '" + only + "'")

    var rows = List[FamilyMax]()
    for i in range(len(fams)):
        if only != "" and fams[i] != only:
            continue
        print("---", fams[i], "---")
        rows.append(_measure(fams[i]))
        print("  max", rows[len(rows) - 1].max_ncon, "over",
              rows[len(rows) - 1].episodes, "episodes, at",
              rows[len(rows) - 1].task)

    if only != "":
        print("(one family: nothing written)")
        return
    var text = String(
        "# GENERATED by tools/tasks/libero_contact_budget.mojo — do not edit.\n"
        "# <family>=<max resting ncon>,<task that reached it>\n"
        "# settle " + String(SETTLE) + " + " + String(STEPS)
        + " null control steps x " + String(SUBSTEPS) + " substeps, "
        + String(SEEDS) + " seed lanes per task, counted after every substep.\n"
    )
    for i in range(len(rows)):
        text += rows[i].name + "=" + String(rows[i].max_ncon) + "," + rows[i].task + "\n"
    with open(String(OUT_PATH), "w") as fh:
        fh.write(text)
    print("wrote", OUT_PATH)
