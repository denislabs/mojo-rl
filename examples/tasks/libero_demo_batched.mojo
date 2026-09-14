"""LIBERO'S OWN DEMONSTRATIONS, REPLAYED ON THE BATCH — a real task per lane.

    pixi run mojo run -I . examples/tasks/libero_demo_batched.mojo
    pixi run mojo run -I . examples/tasks/libero_demo_batched.mojo --demo-offset 2 --steps 100
    pixi run -e nvidia mojo run -I . examples/tasks/libero_demo_batched.mojo

Every lane runs one `libero_goal` demonstration: its task's goal tape and
active mask in `meta`, the family's region table in `curriculum`, the demo's
recorded initial state, and its recorded 7-word OSC_POSE actions fed step by
step through the batched env. Beside it, ONE CPU lane per demo runs the same
replay through `OscPose` and the host integrator — the loop
`examples/tasks/libero_demo_replay.mojo` gated against MuJoCo.

## WHAT IT CHECKS, AND WHICH CHECK IS HARD

1. ⚠⚠ **THE SUCCESS WORD, EXACTLY.** After every step, each lane's device
   `META_IDX_GOAL_HELD` is compared with the HOST evaluator (`eval.eval_goal`)
   run on that SAME lane's downloaded device state. Same state, two
   evaluators: any disagreement is an evaluator defect, not physics. This
   raises, and it is refused as vacuous unless both True and False occur.
2. **THE PHYSICS, LANE BY LANE — MEASURED, THEN BOUNDED.** Per-lane max |qpos|
   divergence, batch against CPU, after steps 1, 10, 50 and the demo's last.
   They cannot agree to the bit: the device is float32, the host float64, and
   even two CPU batch sizes differ by 1-2 ULP
   (`_the_cpu_dynamics_answer_depends_on_the_batch_size`), and a contact-rich
   replay amplifies that. The hard bound is on the first `--window` steps only.
3. **SUCCESS OVER THE DEMO, BATCH AND CPU SIDE BY SIDE** — reported, not
   gated. Open-loop replay is not guaranteed to reproduce a demo's success:
   the single-lane replay is 1.59 cm from the 2022 recording at step 30, which
   is exactly MuJoCo 3.12's own distance from it.

## MEASURED — M1 Pro, Metal, demos 0-1 of every task, 2026-09-14 (1505 s)

    success word    2613 comparisons, 399 true, 0 disagreeing
    success         batch 19 / 20, CPU 20 / 20, 19 lanes agree; where both
                    succeed the first successful step differs by 0-6
    physics         worst |dq| over the first 10 steps 2.9e-06; by step 50 up
                    to 0.11 on the drawer tasks, centimetres on the rest
    controller      no singular lane on any step

⚠ THE ONE DISAGREEMENT IS NOT DIAGNOSED. Lane 0 (`open_the_middle_drawer`,
demo 0): the CPU opens the drawer at step 127 and the batch never does; the two
are 2e-06 apart at step 50 and 1.42 apart at the end, and demo 1 of the same
task succeeds on both (126 / 127). A float32 lane diverging through a grasp on
a drawer handle is the obvious reading and it is only a reading — the
per-step curve of that lane is where to look.

## ⚠ THE PROTOCOL IS THE REPLAY GATE'S, NOT THE EVAL'S

`tools/tasks/libero_demo_replay.py` sets `states[0]` and THEN builds the
controller, so OSC's nullspace target is the demo's own start pose; the eval
protocol (`libero_eval.mojo`) anchors on the rest pose and restores the frozen
row after. Both legs here take the replay order: the batch resets, writes the
demo state, and re-anchors (`_osc_anchor`); the CPU lane builds `OscPose` on
the written state. There is no settle: `states[0]` is recorded AFTER LIBERO's
five zero-action steps.

## ⚠ THE FIXTURES ARE THE SCENE'S, ON BOTH LEGS

LIBERO re-draws each fixture's xy per episode, and the recorded demos carry
that draw in their `model_file`. A batched env has ONE model for every lane, so
the per-demo draw cannot be applied per lane — both legs use the composed
scene's fixture poses (the centre of LIBERO's band). The comparison is fair;
the success rate is a replay in a scene up to a centimetre from the recording.
"""

from std.os import listdir
from std.os.path import exists
from std.sys import argv
from std.memory.alloc import unsafe_alloc
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.io.hdf5.reader import H5File
from mojo_rl.physics3d.fields import Data, Model, DynDims, DynamicsScratch
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
    spec_fields_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.studio.stepping import StudioIntegEll
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.dynamics.osc_pose import (
    OscPose, OscPoseConfig, ARM_DOF,
)
from mojo_rl.physics3d.dynamics.osc_pose_gpu import (
    OSC_ACTION_DIM, build_osc_refs,
)
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    META_IDX_GOAL_HELD, META_IDX_NUM_CONTACTS, META_IDX_INIT_REGION_0,
    META_INIT_SLOTS, META_IDX_JINIT_0, META_JINIT_SLOTS, META_JINIT_WORDS,
    META_IDX_SHAPE_W_GOAL, META_IDX_SHAPE_W_REACH, MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
)
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, FamilySpec,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses, BoundGoal,
)
from mojo_rl.tasks.eval import (
    eval_goal, HostState, region_sites, region_contact_bodies,
)
from mojo_rl.tasks.tape import encode_goal, TAPE_WORDS
from mojo_rl.tasks.gpu_eval import region_table_words, require_gpu_regions
from mojo_rl.tasks.active import active_mask
from mojo_rl.tasks.libero_state_remap import load_state_remap
from mojo_rl.tasks.libero_goal_dims import LIBERO_GOAL_DIMS
from mojo_rl.tasks.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS
from mojo_rl.tasks.libero_goal_config import (
    LiberoGoalOscEnv, LIBERO_GOAL_FRAME_SKIP,
)


comptime H = DType.float64
comptime FAMILY = "libero_goal"
comptime FAMILY_DIR = "mojo_rl/tasks/families/"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime DEMO_DIR = "references/libero_demos/libero_goal"
comptime N_TASKS = 10
comptime DEMOS_PER_TASK = 2
comptime N_ENVS = N_TASKS * DEMOS_PER_TASK
comptime NQ = LIBERO_GOAL_DIMS.NQ
comptime NV = LIBERO_GOAL_DIMS.NV
comptime NB = LIBERO_GOAL_DIMS.NBODY
comptime NS = LIBERO_GOAL_DIMS.NSITE
comptime MC = LIBERO_GOAL_MAX_CONTACTS
comptime SUBSTEPS = LIBERO_GOAL_FRAME_SKIP
comptime ARM_WORDS = 9
"""The Panda's seven joints and two fingers lead `qpos`."""

comptime WINDOW_TOL: Float64 = 1.0e-3
"""The hard bound on the first `--window` steps, in qpos units (m / rad).
See the header: float32 against float64, through contact."""


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("libero demo batched: no '" + want + "' in the composed scene")


def _task_names() raises -> List[String]:
    var out = List[String]()
    var want = String(FAMILY) + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def _pad(s: String, n: Int) -> String:
    var out = String(s)
    if out.byte_length() > n:
        return String(out[byte = 0 : n])
    while out.byte_length() < n:
        out += " "
    return out^


def _num(x: Float64) -> String:
    """A short number for the table."""
    var t = String(x)
    var e = t.find("e")
    if e >= 0:
        # keep the exponent: `4.2968018e-07` must not print as `4.2968018`
        var cut = e if e < 5 else 5
        return String(t[byte = 0 : cut]) + String(t[byte = e : t.byte_length()])
    if t.byte_length() > 9:
        return String(t[byte = 0 : 9])
    return t^


def _make_osc(fmd_joint_names: List[String], fmd_site_names: List[String],
              fmd_actuator_names: List[String], site_body: Int,
              qadr_all: List[Int], dadr_all: List[Int],
              ctrl_min: List[Float64], ctrl_max: List[Float64],
              nact: Int) raises -> OscPose:
    """A fresh controller — built on the state already written, the replay
    gate's order."""
    var dof = List[Int]()
    var qadr = List[Int]()
    var jidx = List[Int]()
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ji = _index(fmd_joint_names, String("robot_joint") + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
        jidx.append(ji)
        var ai = _index(fmd_actuator_names, String("robot_torq_j") + String(j + 1))
        act_idx.append(ai)
        tmin.append(ctrl_min[ai])
        tmax.append(ctrl_max[ai])
    var site = _index(fmd_site_names, String("robot_grip_site"))
    var ga1 = _index(fmd_actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd_actuator_names, String("robot_gripper_finger_joint2"))
    return OscPose(
        dof^, qadr^, jidx^, tmin^, tmax^, act_idx^, site, site_body, ga1, ga2,
        ctrl_min[ga1], ctrl_max[ga1], ctrl_min[ga2], ctrl_max[ga2],
        OscPoseConfig(), nact, NQ, NV,
    )


def main() raises:
    var args = argv()
    var demo_offset = 0
    var max_steps = 0
    var window = 10
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--demo-offset" and i + 1 < len(args):
            demo_offset = Int(String(args[i + 1]))
            i += 1
        elif s == "--steps" and i + 1 < len(args):
            max_steps = Int(String(args[i + 1]))
            i += 1
        elif s == "--window" and i + 1 < len(args):
            window = Int(String(args[i + 1]))
            i += 1
        else:
            # ⚠ REFUSED, NOT IGNORED — `_a_silently_ignored_argument_runs_the_
            # wrong_experiment_for_an_hour`.
            raise Error("libero demo batched: unknown argument '" + s + "'")
        i += 1

    print("=" * 78)
    print("LIBERO demonstrations replayed on the batch —", N_ENVS, "lanes,",
          DEMOS_PER_TASK, "demos x", N_TASKS, "tasks of", FAMILY)
    print("=" * 78)

    var f = load_family(String(FAMILY_DIR) + FAMILY + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var names = _task_names()
    if len(names) != N_TASKS:
        raise Error(
            String(len(names)) + " " + FAMILY + " tasks on disk, this driver"
            " is built for " + String(N_TASKS)
        )

    # ── the demos: actions, and the first recorded state in OUR order ──────
    var remap = load_state_remap(String(FAMILY))
    var row_words = remap.row_words()
    var lane_task = List[Int]()
    var lane_demo = List[Int]()
    var lane_T = List[Int]()
    var actions = List[List[Float64]]()
    var q0 = List[List[Float64]]()
    var v0 = List[List[Float64]]()
    for ti in range(N_TASKS):
        var stem = String(String(names[ti])[byte = String(FAMILY).byte_length() + 2 : String(names[ti]).byte_length()])
        var path = String(DEMO_DIR) + "/" + stem + "_demo.hdf5"
        if not exists(path):
            raise Error(
                "no demo file at " + path + ". The demonstrations are"
                " gitignored (HF `yifengzhu-hf/LIBERO-datasets`, ~6 GB for"
                " libero_goal); fetch them into " + DEMO_DIR + "."
            )
        var h5 = H5File(path)
        for k in range(DEMOS_PER_TASK):
            var di = demo_offset + k
            var d_act = h5.open_dataset(
                String("data/demo_") + String(di) + "/actions"
            )
            var d_st = h5.open_dataset(
                String("data/demo_") + String(di) + "/states"
            )
            var T = Int(d_act.dims[0])
            var raw_a = unsafe_alloc[Scalar[H]](T * 7).as_unsafe_any_origin()
            d_act.read_all[H](raw_a)
            var al = List[Float64]()
            for r in range(T * 7):
                al.append(Float64(raw_a[unsafe_offset=r]))
            var raw_s = unsafe_alloc[Scalar[H]](row_words).as_unsafe_any_origin()
            d_st.read_range[H](0, 1, raw_s)
            var row = List[Float64]()
            for r in range(row_words):
                row.append(Float64(raw_s[unsafe_offset=r]))
            var qo = List[Float64](length=remap.nq, fill=0.0)
            var vo = List[Float64](length=remap.nv, fill=0.0)
            remap.convert_into(row, qo, vo)
            lane_task.append(ti)
            lane_demo.append(di)
            lane_T.append(T)
            actions.append(al^)
            q0.append(qo^)
            v0.append(vo^)
    var horizon = 0
    for e in range(N_ENVS):
        if max_steps > 0 and lane_T[e] > max_steps:
            lane_T[e] = max_steps
        if lane_T[e] > horizon:
            horizon = lane_T[e]
    print("  demos:", DEMO_DIR, "| demo indices", demo_offset, "..",
          demo_offset + DEMOS_PER_TASK - 1, "| longest", horizon, "steps")

    # ── goals, and the host tables the evaluator reads ────────────────────
    var nqs = List[Int]()
    for k in range(len(fmd.joints)):
        nqs.append(fmd.joints[k].nq)
    var jadr = joint_qpos_addresses(nqs)
    var goals = List[BoundGoal]()
    var tapes = List[List[Float64]]()
    var masks = List[Float64]()
    for ti in range(N_TASKS):
        var t = load_task(String(TASK_DIR) + names[ti] + ".task")
        validate_task_against_family(t, f)
        var g = bind_goal(
            parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
            fmd.joint_names, jadr,
        )
        require_tier_a(g, t.name)
        require_gpu_regions(g, t.name)
        tapes.append(encode_goal(g))
        masks.append(active_mask(t, f))
        goals.append(g^)
    var rsites = region_sites(f, fmd.site_names)
    var rcontact = region_contact_bodies(f, fmd.body_names)
    var site_body_tab = List[Int]()
    var site_quat_tab = List[Float64]()
    for k in range(len(fmd.sites)):
        site_body_tab.append(fmd.sites[k].body_id)
        site_quat_tab.append(fmd.sites[k].quat_x)
        site_quat_tab.append(fmd.sites[k].quat_y)
        site_quat_tab.append(fmd.sites[k].quat_z)
        site_quat_tab.append(fmd.sites[k].quat_w)
    var body_parent_tab = List[Int]()
    body_parent_tab.append(-1)
    for k in range(len(fmd.bodies)):
        body_parent_tab.append(fmd.bodies[k].parent)

    # ── the controller record, shared ─────────────────────────────────────
    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for k in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[k].nq
        da += fmd.joints[k].nv
    var ctrl_min = List[Float64]()
    var ctrl_max = List[Float64]()
    for k in range(len(fmd.actuators)):
        ctrl_min.append(fmd.actuators[k].ctrl_min)
        ctrl_max.append(fmd.actuators[k].ctrl_max)
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var site_body = fmd.sites[site].body_id
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
        var ai = _index(fmd.actuator_names, String("robot_torq_j") + String(j + 1))
        act_idx.append(ai)
        tmin.append(ctrl_min[ai])
        tmax.append(ctrl_max[ai])
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))
    var cfg = OscPoseConfig()
    var refs = build_osc_refs(
        dof^, qadr^, jidx^, tmin^, tmax^, act_idx^, site, site_body, ga1, ga2,
        ctrl_min[ga1], ctrl_max[ga1], ctrl_min[ga2], ctrl_max[ga2],
        cfg.kp, cfg.damping_ratio, cfg.output_max_pos, cfg.output_max_ori,
        cfg.nullspace_kp, cfg.gripper_speed,
    )

    # ══ THE BATCH ══════════════════════════════════════════════════════════
    var ctx = DeviceContext()
    var env = LiberoGoalOscEnv[N_ENVS](ctx)
    env.set_osc_refs(refs, ctx)
    var cw = region_table_words(f, rsites, rcontact)
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DT](cw[k])
    env.mf.curriculum.upload(ctx)
    env.reset_batch[N_ENVS](ctx, 1)
    ctx.synchronize()
    env.d.meta.download(ctx)
    env.d.qpos.download(ctx)
    env.d.qvel.download(ctx)
    ctx.synchronize()
    for e in range(N_ENVS):
        var mb = e * METADATA_SIZE
        var ti = lane_task[e]
        for k in range(TAPE_WORDS):
            env.d.meta.data[mb + META_IDX_TASK_PARAM_0 + k] = Scalar[DT](tapes[ti][k])
        env.d.meta.data[mb + META_IDX_TASK_ACTIVE] = Scalar[DT](masks[ti])
        # ⚠ NO DEVICE PLACEMENT AND NO JOINT DRAW: the state is the demo's.
        for k in range(META_INIT_SLOTS):
            env.d.meta.data[mb + META_IDX_INIT_REGION_0 + k] = Scalar[DT](0)
        for k in range(META_JINIT_SLOTS * META_JINIT_WORDS):
            env.d.meta.data[mb + META_IDX_JINIT_0 + k] = Scalar[DT](0)
        env.d.meta.data[mb + META_IDX_SHAPE_W_GOAL] = Scalar[DT](0)
        env.d.meta.data[mb + META_IDX_SHAPE_W_REACH] = Scalar[DT](0)
        env.d.meta.data[mb + META_IDX_GOAL_HELD] = Scalar[DT](0)
        for k in range(NQ):
            env.d.qpos.data[e * NQ + k] = Scalar[DT](q0[e][k])
        for k in range(NV):
            env.d.qvel.data[e * NV + k] = Scalar[DT](v0[e][k])
    env.d.meta.upload(ctx)
    env.d.qpos.upload(ctx)
    env.d.qvel.upload(ctx)
    ctx.synchronize()
    # ⚠ THE REPLAY ORDER: the controller anchored on the DEMO's start pose.
    env._osc_anchor(ctx)
    ctx.synchronize()

    var act_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OSC_ACTION_DIM)
    var dev_traj = List[List[Float64]]()
    var dev_success = List[Int]()
    for _ in range(N_ENVS):
        dev_traj.append(List[Float64]())
        dev_success.append(-1)
    var eval_cmp = 0
    var eval_bad = 0
    var eval_true = 0
    var singular_steps = 0
    for t in range(horizon):
        var ap = act_h.unsafe_ptr()
        for e in range(N_ENVS):
            for k in range(OSC_ACTION_DIM):
                var v = actions[e][t * 7 + k] if t < lane_T[e] else 0.0
                ap[unsafe_offset = e * OSC_ACTION_DIM + k] = Scalar[DT](v)
        ctx.enqueue_copy(env._action, act_h)
        env.step_batch[N_ENVS](ctx, UInt64(t + 1))
        env.d.qpos.download(ctx)
        env.d.xpos.download(ctx)
        env.d.xquat.download(ctx)
        env.d.site_xpos.download(ctx)
        env.d.contacts.download(ctx)
        env.d.meta.download(ctx)
        ctx.synchronize()
        if env.osc_singular_lanes(ctx) > 0:
            singular_steps += 1
        for e in range(N_ENVS):
            if t >= lane_T[e]:
                continue
            for k in range(NQ):
                dev_traj[e].append(Float64(env.d.qpos.data[e * NQ + k]))
            var st = HostState(List[Float64](), List[Float64](), List[Float64]())
            for k in range(NB * 3):
                st.xpos.append(Float64(env.d.xpos.data[e * NB * 3 + k]))
            for k in range(NB * 4):
                st.xquat.append(Float64(env.d.xquat.data[e * NB * 4 + k]))
            for k in range(NS * 3):
                st.site_xpos.append(Float64(env.d.site_xpos.data[e * NS * 3 + k]))
            for k in range(NQ):
                st.qpos.append(Float64(env.d.qpos.data[e * NQ + k]))
            st.site_body = site_body_tab.copy()
            st.site_quat = site_quat_tab.copy()
            st.body_parent = body_parent_tab.copy()
            var ncon = Int(env.d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
            if ncon > MC:
                ncon = MC
            st.ncon = ncon
            for c in range(ncon):
                var cb = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
                st.con_a.append(Int(env.d.contacts.data[cb + CONTACT_IDX_BODY_A]))
                st.con_b.append(Int(env.d.contacts.data[cb + CONTACT_IDX_BODY_B]))
            var host_says = eval_goal(goals[lane_task[e]], f, st, rsites, rcontact)
            var dev_says = Float64(
                env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD]
            ) > 0.5
            eval_cmp += 1
            if host_says:
                eval_true += 1
            if host_says != dev_says:
                eval_bad += 1
                if eval_bad <= 10:
                    print("   EVAL MISMATCH lane", e, "step", t, ": device",
                          dev_says, "host on the device state", host_says)
            if dev_says and dev_success[e] < 0:
                dev_success[e] = t
    print("  batch: stepped", horizon, "control steps on", N_ENVS, "lanes")

    # ══ THE CPU LEG — one lane at a time, the same demos ═══════════════════
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=MC, nmesh_verts=verts)
    var m = Model[H, DynDims](dims)
    while True:
        try:
            build_model_runtime[H](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=MC, nmesh_verts=verts)
            m = Model[H, DynDims](dims)
    var sf = spec_fields_runtime[H](fmd, dims, m)
    var nact = dims.get_nact()
    var cpu_success = List[Int]()
    var div_1 = List[Float64]()
    var div_10 = List[Float64]()
    var div_50 = List[Float64]()
    var div_end = List[Float64]()
    var arm_end = List[Float64]()
    var window_worst = 0.0
    for e in range(N_ENVS):
        # a fresh Data, scratch and integrator per lane: nothing carried over
        var d = Data[H, DynDims, 1](dims)
        var scratch = DynamicsScratch[H, DynDims, 1](dims)
        var integ = StudioIntegEll(dims)
        for k in range(NQ):
            d.qpos.data[k] = Scalar[H](q0[e][k])
        for k in range(NV):
            d.qvel.data[k] = Scalar[H](v0[e][k])
        forward_kinematics["cpu", H, DynDims, 1](d, m)
        var osc = _make_osc(
            fmd.joint_names, fmd.site_names, fmd.actuator_names, site_body,
            qadr_all, dadr_all, ctrl_min, ctrl_max, nact,
        )
        osc.update(d, m, scratch)
        osc.reset(d, m)
        var act = List[Scalar[H]](length=nact if nact > 0 else 1, fill=Scalar[H](0))
        var succ = -1
        var d1 = 0.0
        var d10 = 0.0
        var d50 = 0.0
        var dend = 0.0
        var aend = 0.0
        for t in range(lane_T[e]):
            var a = List[Float64]()
            for k in range(7):
                a.append(actions[e][t * 7 + k])
            for s in range(SUBSTEPS):
                osc.update(d, m, scratch)
                if s == 0:
                    osc.set_goal(a, d, m)
                var ctrl = osc.run(a, d, m, scratch)
                for k in range(NV):
                    d.qfrc.data[k] = Scalar[H](0)
                apply_actions_fields[H](sf, d, ctrl, act, fmd.timestep)
                integ.step["cpu"](d, m)
            forward_kinematics["cpu", H, DynDims, 1](d, m)
            detect_contacts["cpu", H, DynDims, 1](d, m)
            var dq = 0.0
            var darm = 0.0
            for k in range(NQ):
                var x = abs(Float64(d.qpos.data[k]) - dev_traj[e][t * NQ + k])
                if x > dq:
                    dq = x
                if k < ARM_WORDS and x > darm:
                    darm = x
            if t == 0:
                d1 = dq
            if t == 9:
                d10 = dq
            if t == 49:
                d50 = dq
            dend = dq
            aend = darm
            if t < window and dq > window_worst:
                window_worst = dq
            if succ < 0:
                var st = HostState(List[Float64](), List[Float64](), List[Float64]())
                for k in range(NB * 3):
                    st.xpos.append(Float64(d.xpos.data[k]))
                for k in range(NB * 4):
                    st.xquat.append(Float64(d.xquat.data[k]))
                for k in range(NS * 3):
                    st.site_xpos.append(Float64(d.site_xpos.data[k]))
                for k in range(NQ):
                    st.qpos.append(Float64(d.qpos.data[k]))
                st.site_body = site_body_tab.copy()
                st.site_quat = site_quat_tab.copy()
                st.body_parent = body_parent_tab.copy()
                var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
                if ncon > MC:
                    ncon = MC
                st.ncon = ncon
                for c in range(ncon):
                    st.con_a.append(Int(d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_A]))
                    st.con_b.append(Int(d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_B]))
                if eval_goal(goals[lane_task[e]], f, st, rsites, rcontact):
                    succ = t
        cpu_success.append(succ)
        div_1.append(d1)
        div_10.append(d10)
        div_50.append(d50)
        div_end.append(dend)
        arm_end.append(aend)

    # ══ THE REPORT ═════════════════════════════════════════════════════════
    print()
    print("  lane task                                demo    T  success(batch/cpu)"
          + "   |dq| @1       @10       @50       @end   arm@end")
    var n_dev = 0
    var n_cpu = 0
    var n_agree = 0
    for e in range(N_ENVS):
        var nm = String(String(names[lane_task[e]])[byte = String(FAMILY).byte_length() + 2 : String(names[lane_task[e]]).byte_length()])
        if dev_success[e] >= 0:
            n_dev += 1
        if cpu_success[e] >= 0:
            n_cpu += 1
        if (dev_success[e] >= 0) == (cpu_success[e] >= 0):
            n_agree += 1
        var sd = String(dev_success[e]) if dev_success[e] >= 0 else String("-")
        var sc = String(cpu_success[e]) if cpu_success[e] >= 0 else String("-")
        print("  " + _pad(String(e), 5) + _pad(nm, 36) + _pad(String(lane_demo[e]), 5)
              + _pad(String(lane_T[e]), 5) + _pad(sd + " / " + sc, 21)
              + _pad(_num(div_1[e]), 10)
              + _pad(_num(div_10[e]), 10)
              + _pad(_num(div_50[e]), 10)
              + _pad(_num(div_end[e]), 10)
              + _num(arm_end[e]))
    print()
    print("  success over the demo: batch", n_dev, "/", N_ENVS, "  cpu", n_cpu,
          "/", N_ENVS, "  lanes agreeing", n_agree, "/", N_ENVS)
    print("  success word: device GOAL_HELD vs host eval on the device state —",
          eval_cmp, "comparisons,", eval_true, "true,", eval_bad, "disagreeing")
    print("  physics: worst |dq| over the first", window, "steps",
          window_worst, "(bound", WINDOW_TOL, ")")
    print("  controller: control steps with a singular lane", singular_steps)

    if eval_bad != 0:
        raise Error(
            String(eval_bad) + " device GOAL_HELD words disagree with the host"
            " evaluator on the SAME state — an evaluator defect, not physics"
        )
    if eval_true == 0 or eval_true == eval_cmp:
        raise Error(
            "the success word was " + ("never" if eval_true == 0 else "always")
            + " true across " + String(eval_cmp) + " comparisons, so their"
            " agreement is vacuous"
        )
    if singular_steps != 0:
        raise Error(String(singular_steps) + " control steps had a singular lane")
    if window_worst > WINDOW_TOL:
        raise Error(
            "batch and CPU diverge by " + String(window_worst) + " within the"
            " first " + String(window) + " steps (bound "
            + String(WINDOW_TOL) + ")"
        )
    print()
    print("=== PASS ===")
