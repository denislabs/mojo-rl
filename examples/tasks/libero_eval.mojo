"""LIBERO'S OWN EVAL PROTOCOL, ON OUR ENGINE — L6's driver, and its gate.

    pixi run libero-eval                          # the null policy, all ten tasks
    pixi run libero-eval --inits 20 --steps 600   # LIBERO's own numbers
    pixi run libero-eval --task turn_on_the_stove --steps 100 --csv /tmp/e.csv

`lifelong/metric.py::evaluate_one_task_success`, transcribed:

    init_states = torch.load(<task>.pruned_init)      # the frozen fifty
    env.reset()                                       #   -> our base_qpos
    obs = env.set_init_state(init_states_)            #   -> the frozen row
    for _ in range(5): env.step(zeros(7))             #   five settle steps
    while steps < cfg.eval.max_steps:                 #   600
        obs, reward, done, info = env.step(policy(obs))
        dones[k] = dones[k] or done[k]                #   success at ANY step
    num_success += int(dones[k])

Four details of that loop are load-bearing and none of them is obvious.

## ⚠⚠ `env.reset()` HAPPENS BEFORE `set_init_state`, AND THE CONTROLLER KEEPS THE FIRST

`Robot.reset` writes `init_qpos` and constructs the OSC controller, whose
nullspace target `q0` is the configuration AT CONSTRUCTION. `set_init_state`
then overwrites `qpos` wholesale — and does not rebuild the controller. So the
arm is at the frozen row and the nullspace is pulling toward the family's rest
pose, which are NOT the same vector: LIBERO's reset adds robosuite's default
0.02-magnitude joint noise before freezing, so row 0 of `open_the_middle_drawer`
starts at joint2 = -0.188 against a rest pose of -0.161.

This driver reproduces the order — base pose, `osc.reset`, THEN the row — and
that is why `apply_row` does not re-anchor the controller. Anchoring on the
restored pose instead is a one-line change that silently removes the nullspace
force every LIBERO episode actually starts with.

## ⚠⚠ THE FIVE SETTLE STEPS ARE A SEVEN-CENTIMETRE DROP, AND IT IS THEIRS

The per-object settle column below reports 7.2 / 2.6 / 7.1 / 6.8 cm on
`libero_goal`, and essentially nothing moves over the 600 steps that follow
(settle and whole-episode agree to five decimals). That looked like a
conversion fault and is not: robosuite's placement sampler puts every object at
z = 0.97 above a table top near 0.90, so a `.pruned_init` row is a scene of
objects in the AIR, and the five zero-action steps are the landing.

MEASURED IN THEIR OWN MODEL — the recorded `model_file`, their raw rows, five
policy steps of MuJoCo:

    LIBERO   0.0716  0.0261  0.0711  0.0675   m
    ours     0.07160 0.02608 0.07112 0.06750  m

Sub-millimetre across a free fall and four landings, which is a fidelity result
in its own right. And the state itself is not merely close: the converter's own
gate drives both models from all fifty frozen inits of all ten tasks and puts
every body within 2.18e-07 m, so the objects START in exactly the same place.
Whatever they then do, they do it in LIBERO too.

## ⚠ AND THE FIXTURES MOVE UNDER THEM

`bddl_base_domain._reset_internal` re-draws every FIXTURE per reset and
`set_init_state` restores only `(time, qpos, qvel)` — so in LIBERO's own loop a
bowl frozen inside a drawer is restored against a drawer up to a centimetre
away. Measured over the fifty recorded draws of `libero_goal`, the band is
19-20 mm wide per axis and our static fixture pose is its centre
(`tools/tasks/libero_init_table.py`). The five zero-action steps are what let
the props fall the last millimetres onto whatever is actually there.

## ⚠ SUCCESS IS `ANY STEP`, AND IT IS EVALUATED AFTER A STEP, NEVER BEFORE

`dones[k] = dones[k] or done[k]` — a goal met on step 3 and abandoned on step 4
is a success, which is why `libero_demo_success.py` needs a forty-state window
to score LIBERO's own demonstrations. The at-reset and post-settle columns
below are reported SEPARATELY and are not part of the rate, because LIBERO does
not evaluate there.

## ⚠⚠ WHAT THIS FILE IS A GATE FOR: THE NULL-ACTION RATE

With no policy in the tree the action is `zeros(7)` — which is precisely
`metric.py`'s `dummy`, so the null run is a real instance of the protocol and
not a stand-in for one. The assessment's L6 gate:

    any task above 0 at reset is a bug in G3-G6, not a policy

and this file RAISES on it. `_a_goal_false_at_reset_can_be_solved_by_the_null_action`
is the memory: `so101_reach_brick` was false at reset on all eight lanes and an
untrained actor met it on 64 of 64 episodes, so the reset column alone is not
the check. Three columns are printed and all three must be zero for the null
policy: at reset, after the five settle steps, and over the whole horizon.

⚠ A NON-ZERO POST-SETTLE OR HORIZON RATE IS NOT AUTOMATICALLY A BUG — a task
whose goal is "the drawer is closed" would be solved by doing nothing, and
LIBERO has none like that in `libero_goal`. The refusal names the task and the
column so the finding can be read; it does not diagnose it.

## ⚠ THE HORIZON IS PRICED, NOT ASSUMED

600 policy steps x 25 substeps is 15 000 physics steps per episode, and this
scene is 41 dof with 240 geoms on ONE CPU lane. `--steps` exists so the horizon
is a stated number rather than a silent truncation: the summary prints the
horizon it ran and whether it is LIBERO's.
"""

from std.os import listdir
from std.random import seed as seed_rng
from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.core.logger import CsvLogger
from mojo_rl.physics3d.fields import (
    Data, Model, DynDims, DynamicsScratch, SpecFields,
)
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
    spec_fields_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
)
from mojo_rl.physics3d.studio.stepping import StudioIntegEll
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.dynamics.osc_pose import (
    OscPose, OscPoseConfig, ARM_DOF, OSC_ACTION_DIM,
)

from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, TaskSpec, FamilySpec,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses, BoundGoal,
)
from mojo_rl.tasks.eval import (
    eval_goal, HostState, region_sites, region_contact_bodies,
)
from mojo_rl.tasks.init_table import load_init_table, InitTable
from mojo_rl.tasks.eval_report import SuccessReport
from mojo_rl.tasks.reset import free_slot_addresses
from mojo_rl.tasks.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime DT = DType.float64
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime FAMILY_DIR = "mojo_rl/tasks/families/"
comptime SUBSTEPS = 25
"""`control_freq=20` against the family's 2 ms timestep. robosuite's
`Robot.control` runs the controller EVERY substep and gates only `set_goal` on
`policy_step`; `osc_pose.mojo`'s header records the same split."""
comptime SETTLE_STEPS = 5
"""`metric.py`: `for _ in range(5): obs, _, _, _ = env.step(dummy)`."""
comptime LIBERO_MAX_STEPS = 600
comptime LIBERO_N_EVAL = 20


def _pad(s: String, n: Int) -> String:
    var out = String(s)
    while out.byte_length() < n:
        out += " "
    return out^


def task_names(suite: String) raises -> List[String]:
    """Every `<suite>__*.task` on disk, sorted — the family's task list.

    ⚠ THE SAME WALK `libero_viewer.task_names` DOES, AND FOR THE SAME REASON: a
    LIBERO suite's task list is generated and grows, so a hardcoded one goes
    stale silently. Here it is also the ORDER the init table's `task_index`
    means, and `main` does not take that on trust — it compares each task's
    `language=` against the table's own label and raises if they disagree.
    """
    var out = List[String]()
    var want = suite + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if names[i] == want:
            return i
    raise Error("libero eval: no '" + want + "' in the composed scene")


def read_state(
    d: Data[DT, DynDims, 1], nb: Int, ns: Int, nq: Int,
    site_body_tab: List[Int], site_quat_tab: List[Float64],
    body_parent_tab: List[Int],
) -> HostState:
    """Everything the L3 language reads, out of `Data`. `detect_contacts` must
    have run — `On`/`In` are contact predicates and a stale list answers a
    question about the previous step."""
    var st = HostState(List[Float64](), List[Float64](), List[Float64]())
    for k in range(nb * 3):
        st.xpos.append(Float64(d.xpos.data[k]))
    for k in range(nb * 4):
        st.xquat.append(Float64(d.xquat.data[k]))
    for k in range(ns * 3):
        st.site_xpos.append(Float64(d.site_xpos.data[k]))
    for k in range(nq):
        st.qpos.append(Float64(d.qpos.data[k]))
    st.site_body = site_body_tab.copy()
    st.site_quat = site_quat_tab.copy()
    st.body_parent = body_parent_tab.copy()
    var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    if ncon > LIBERO_GOAL_MAX_CONTACTS:
        ncon = LIBERO_GOAL_MAX_CONTACTS
    st.ncon = ncon
    for k in range(ncon):
        st.con_a.append(Int(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_A]))
        st.con_b.append(Int(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_B]))
    return st^


def env_reset(
    mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims],
    mut osc: OscPose,
    mut scratch: DynamicsScratch[DT, DynDims, 1],
    f: FamilySpec, nq: Int, nv: Int,
) raises:
    """`env.reset()` — the family's rest pose, and the controller anchored ON IT.

    ⚠ THE ANCHOR IS TAKEN HERE AND NOT AFTER THE ROW. See the header: robosuite
    builds the controller in `Robot.reset` and `set_init_state` does not rebuild
    it, so `q0` is the rest pose for the whole episode.
    """
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    osc.update(d, m, scratch)
    osc.reset(d, m)


def set_init_state(
    mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims],
    mut osc: OscPose,
    mut scratch: DynamicsScratch[DT, DynDims, 1],
    tbl: InitTable, row: Int, nq: Int, nv: Int,
) raises:
    """`env.set_init_state(...)` — the frozen row, whole, then `sim.forward()`."""
    var qpos = List[Float64](length=nq, fill=0.0)
    var qvel = List[Float64](length=nv, fill=0.0)
    tbl.apply(row, qpos, qvel)
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](qpos[i])
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](qvel[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    detect_contacts["cpu", DT, DynDims, 1](d, m)
    osc.update(d, m, scratch)


def policy_step(
    mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims],
    mut osc: OscPose,
    mut scratch: DynamicsScratch[DT, DynDims, 1],
    mut integ: StudioIntegEll,
    sf: SpecFields[DT, DynDims],
    action: List[Float64], mut act: List[Scalar[DT]],
    nv: Int, timestep: Float64,
) raises:
    """One `env.step(action)`: `set_goal` once, then `SUBSTEPS` of control.

    ⚠ `set_goal` ON THE FIRST SUBSTEP ONLY. robosuite's `Robot.control` takes a
    `policy_step` flag and gates exactly that call on it; `run` — the torque —
    is evaluated every substep against the goal set at the top. Calling
    `set_goal` per substep turns a 5 cm position delta into a 5 cm delta
    twenty-five times over, and the arm still looks like it is tracking.
    """
    for s in range(SUBSTEPS):
        osc.update(d, m, scratch)
        if s == 0:
            osc.set_goal(action, d, m)
        var ctrl = osc.run(action, d, m, scratch)
        for i in range(nv):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, ctrl, act, timestep)
        integ.step["cpu"](d, m)
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    detect_contacts["cpu", DT, DynDims, 1](d, m)


def main() raises:
    var args = argv()
    var suite = String("libero_goal")
    var table_path = String("")
    var only_task = String("")
    var csv_path = String("")
    var n_inits = LIBERO_N_EVAL
    var max_steps = LIBERO_MAX_STEPS
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--inits" and i + 1 < len(args):
            n_inits = Int(String(args[i + 1]))
            i += 1
        elif s == "--steps" and i + 1 < len(args):
            max_steps = Int(String(args[i + 1]))
            i += 1
        elif s == "--table" and i + 1 < len(args):
            table_path = String(args[i + 1])
            i += 1
        elif s == "--task" and i + 1 < len(args):
            only_task = String(args[i + 1])
            i += 1
        elif s == "--csv" and i + 1 < len(args):
            csv_path = String(args[i + 1])
            i += 1
        elif not s.startswith("--"):
            suite = s
        else:
            raise Error("libero eval: unknown argument '" + s + "'")
        i += 1
    seed_rng(0)
    if table_path.byte_length() == 0:
        table_path = String("build/init/") + suite + ".init.h5"

    print("=" * 78)
    print("LIBERO's eval protocol on our engine —", suite)
    print("=" * 78)

    var f = load_family(String(FAMILY_DIR) + suite + ".family")
    var path = scene_path(f)
    var fmd = parse_model_runtime(path)
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
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
                fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var scratch = DynamicsScratch[DT, DynDims, 1](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var integ = StudioIntegEll(dims)
    var nb = dims.get_nbody()
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var ns = dims.get_nsite()
    var nact = dims.get_nact()

    var full: InitTable
    try:
        full = load_init_table(table_path, f.name, nq, nv)
    except e:
        # ⚠ SKIPPED, LOUDLY. The table is derived from LIBERO's `.pruned_init`
        # through the recorded demos, both gitignored.
        print("  SKIPPED: no init table at", table_path)
        print("   ", String(e))
        print("  Build it:  pixi run libero-init-dump && pixi run libero-init-freeze")
        print("=== SKIPPED (no frozen inits — this is not a pass) ===")
        return
    var tbl = full.prefix_per_task(n_inits)
    var n_lanes = tbl.n_rows()

    print("  scene :", path, "| nq", nq, "nv", nv, "nbody", nb, "nsite", ns)
    print("  table :", table_path, "|", full.n_rows(), "rows ->", n_lanes,
          "(", n_inits, "per task )")
    print("  policy: ZERO ACTION — `metric.py`'s `dummy = np.zeros((env_num, 7))`")
    print("  horizon:", SETTLE_STEPS, "settle +", max_steps, "steps x",
          SUBSTEPS, "substeps",
          "(LIBERO's own)" if max_steps == LIBERO_MAX_STEPS else "(REDUCED — see the header)")
    print()

    # ── the tables every goal reads ───────────────────────────────────────
    var nqs = List[Int]()
    for k in range(len(fmd.joints)):
        nqs.append(fmd.joints[k].nq)
    var jadr = joint_qpos_addresses(nqs)
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

    # ── the controller ────────────────────────────────────────────────────
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
        var ai = _index(fmd.actuator_names, String("robot_torq_j") + String(j + 1))
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
    var act = List[Scalar[DT]](length=nact if nact > 0 else 1, fill=Scalar[DT](0))

    # ⚠ `zeros(7)`, INCLUDING THE GRIPPER WORD. `osc_gripper_gpu` moves the
    # fingers by `sign(a) * gripper_speed`, and `sign(0)` is 0 — so a zero
    # action holds the hand exactly where the frozen row left it, which is what
    # `metric.py`'s `dummy` does too. The viewer's smoke test passes -1 here
    # (open the hand) and that is a DIFFERENT action.
    var null_action = List[Float64](length=OSC_ACTION_DIM, fill=0.0)

    # ⚠⚠ THE THREE MEASUREMENTS THAT MAKE A ZERO MEAN SOMETHING.
    #
    # "the null policy solves nothing" is ALSO what this file prints when it is
    # evaluating nothing at all — `_a_hit_count_is_not_coverage_of_the_branch`,
    # and vacuity is this tree's default failure. Three ways a green run could
    # be empty, each with its own refusal below:
    #
    #   detect_contacts returns 0     every `On`/`In` in this family is a
    #                                 contact predicate, so a silent zero makes
    #                                 eight of the ten goals unfireable
    #   the physics is not stepping   a frozen scene passes every column
    #   the arm has exploded          a diverged OSC throws the props off the
    #                                 table, after which nothing can be solved
    #                                 and the run still reads 0/20
    var max_ncon = 0
    var eps_with_contact = 0
    var prop_move_sum = 0.0
    var prop_move: List[Float64] = List[Float64]()
    var prop_settle: List[Float64] = List[Float64]()
    var prop_names: List[String] = List[String]()
    var ee_drift_max = 0.0
    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for k in range(len(fmd.joints)):
        jt.append(fmd.joints[k].jnt_type)
        jq.append(fmd.joints[k].nq)
        jv.append(fmd.joints[k].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)
    for sidx in range(len(addrs)):
        if addrs[sidx].qadr < 0:
            continue
        prop_move.append(0.0)
        prop_settle.append(0.0)
        prop_names.append(String(f.slots[sidx].name))

    var rep = SuccessReport(tbl)
    var at_reset = List[Int]()
    var at_settle = List[Int]()
    var per_task = List[Int]()
    var n_tasks = 0
    for k in range(n_lanes):
        if Int(tbl.task_index[k]) + 1 > n_tasks:
            n_tasks = Int(tbl.task_index[k]) + 1
    for _ in range(n_tasks):
        at_reset.append(0)
        at_settle.append(0)
        per_task.append(0)

    # one bound goal per task, resolved once
    #
    # ⚠⚠ THE TABLE'S `task_index` IS AN INDEX INTO THE SORTED `.task` LIST, AND
    # THAT CORRESPONDENCE IS CHECKED, NOT ASSUMED. `libero_init_freeze` walks
    # the dump index (sorted demo stems) and this walks the directory (sorted
    # task names); the two orders agree today and a renamed task would break
    # one of them silently, leaving every episode evaluating the wrong goal on
    # the right init. The manifest carries each row's `language=`, so the
    # comparison below is free and decisive.
    var names_on_disk = task_names(suite)
    var first_row = List[Int]()
    for _ in range(n_tasks):
        first_row.append(-1)
    for k in range(n_lanes):
        var ti = Int(tbl.task_index[k])
        if first_row[ti] < 0:
            first_row[ti] = k
    if len(names_on_disk) < n_tasks:
        raise Error(
            "the table names " + String(n_tasks) + " tasks and "
            + String(len(names_on_disk)) + " `.task` files exist for '" + suite
            + "'"
        )
    var goals = List[BoundGoal]()
    var task_names_used = List[String]()
    for t in range(n_tasks):
        var name = String(names_on_disk[t])
        task_names_used.append(name)
        var ts = load_task(String(TASK_DIR) + name + ".task")
        validate_task_against_family(ts, f)
        var want_label = tbl.task_label(first_row[t])
        if ts.language != want_label:
            raise Error(
                "init table row " + String(first_row[t]) + " has task_index "
                + String(t) + ", whose label is '" + want_label + "', but the "
                + String(t) + "th `.task` on disk is '" + name
                + "' saying '" + ts.language + "'. The table's task order and"
                " the directory's have diverged; re-freeze the table."
            )
        var g = bind_goal(
            parse_goal(ts.goal), f, fmd.body_names, fmd.site_names,
            fmd.joint_names, jadr,
        )
        require_tier_a(g, ts.name)
        goals.append(g^)

    print("  task                                                      reset "
          + "  settle  horizon   steps/s")
    var t0_all = perf_counter_ns()
    for lane in range(n_lanes):
        var ti = Int(tbl.task_index[lane])
        # ⚠ A FILTERED RUN DOES NOT RECORD, AND DOES NOT REPORT. Recording an
        # unvisited lane as False is exactly the "rate over episodes it never
        # ran" that `SuccessReport._require_complete` and
        # `InitTable.prefix_per_task` both exist to prevent; so `--task` skips
        # the lane entirely and the summary below withholds the family verdict.
        if only_task.byte_length() > 0 and task_names_used[ti].find(only_task) < 0:
            continue

        env_reset(d, m, osc, scratch, f, nq, nv)
        set_init_state(d, m, osc, scratch, tbl, lane, nq, nv)

        var st0 = read_state(d, nb, ns, nq, site_body_tab, site_quat_tab, body_parent_tab)
        var solved_reset = eval_goal(goals[ti], f, st0, rsites, rcontact)
        if solved_reset:
            at_reset[ti] += 1
        var p_start = List[Float64]()
        for sidx in range(len(addrs)):
            if addrs[sidx].qadr < 0:
                continue
            for c in range(3):
                p_start.append(Float64(d.qpos.data[addrs[sidx].qadr + c]))
        var ee0 = osc.ee_pos(d)

        for _ in range(SETTLE_STEPS):
            policy_step(d, m, osc, scratch, integ, sf, null_action, act, nv,
                        fmd.timestep)
        var st1 = read_state(d, nb, ns, nq, site_body_tab, site_quat_tab, body_parent_tab)
        if eval_goal(goals[ti], f, st1, rsites, rcontact):
            at_settle[ti] += 1
        # ⚠ THE SETTLE IS MEASURED SEPARATELY FROM THE HORIZON. They are two
        # different quantities: the settle is how far the frozen row had to
        # fall to reach OUR fixtures (the jitter LIBERO never restores), and the
        # horizon is what the null action does afterwards. One number over both
        # cannot tell "the conversion drops props a centimetre" from "the arm
        # is sweeping the table".
        var pj = 0
        for sidx in range(len(addrs)):
            if addrs[sidx].qadr < 0:
                continue
            var dd = 0.0
            for c in range(3):
                var e = Float64(d.qpos.data[addrs[sidx].qadr + c]) - p_start[pj * 3 + c]
                dd += e * e
            prop_settle[pj] += dd ** 0.5
            pj += 1

        # ⚠ `dones[k] = dones[k] or done[k]` — the episode does NOT stop on
        # success. LIBERO breaks only when EVERY lane in the vector env is
        # done, and with one lane that is the same thing; keeping the loop
        # running would change nothing about the rate and would cost the whole
        # horizon on every solved episode.
        var solved = False
        for _ in range(max_steps):
            policy_step(d, m, osc, scratch, integ, sf, null_action, act, nv,
                        fmd.timestep)
            var st = read_state(d, nb, ns, nq, site_body_tab, site_quat_tab,
                                body_parent_tab)
            if st.ncon > max_ncon:
                max_ncon = st.ncon
            if eval_goal(goals[ti], f, st, rsites, rcontact):
                solved = True
                break
        var st_end = read_state(d, nb, ns, nq, site_body_tab, site_quat_tab,
                                body_parent_tab)
        if st_end.ncon > 0:
            eps_with_contact += 1
        if st_end.ncon > max_ncon:
            max_ncon = st_end.ncon
        var pi = 0
        for sidx in range(len(addrs)):
            if addrs[sidx].qadr < 0:
                continue
            var dd = 0.0
            for c in range(3):
                var e = Float64(d.qpos.data[addrs[sidx].qadr + c]) - p_start[pi * 3 + c]
                dd += e * e
            prop_move[pi] += dd ** 0.5
            prop_move_sum += dd ** 0.5
            pi += 1
        var ee1 = osc.ee_pos(d)
        var dr = 0.0
        for c in range(3):
            dr += (ee1[c] - ee0[c]) * (ee1[c] - ee0[c])
        dr = dr ** 0.5
        if dr > ee_drift_max:
            ee_drift_max = dr
        rep.record(lane, solved)
        if solved:
            per_task[ti] += 1

        if lane % n_inits == n_inits - 1:
            var el = Float64(perf_counter_ns() - t0_all) * 1e-9
            var done_eps = lane + 1
            print(
                "  " + _pad(task_names_used[ti], 58)
                + _pad(String(at_reset[ti]) + "/" + String(n_inits), 7)
                + _pad(String(at_settle[ti]) + "/" + String(n_inits), 8)
                + _pad(String(per_task[ti]) + "/" + String(n_inits), 10)
                + String(Int(Float64(done_eps) * Float64(SETTLE_STEPS + max_steps)
                             * Float64(SUBSTEPS) / el))
            )
    var elapsed = Float64(perf_counter_ns() - t0_all) * 1e-9

    print()
    var filtered = only_task.byte_length() > 0
    if filtered:
        print("  FILTERED to '" + only_task + "' — no family verdict; the"
              " per-task columns above are the whole result.")
    else:
        rep.show()
    if not filtered and csv_path.byte_length() > 0:
        var logger = CsvLogger(csv_path)
        rep.log_to(logger, 0)
        logger.close()
        print("  wrote", csv_path)

    # ── the gate ──────────────────────────────────────────────────────────
    print()
    print("  " + String(n_lanes) + " episodes in " + String(Int(elapsed))
          + " s (" + String(Int(elapsed / Float64(n_lanes) * 1000.0))
          + " ms/episode)")
    if filtered:
        print()
        print("=== FILTERED RUN — not the L6 gate ===")
        return
    var n_props = 0
    for sidx in range(len(addrs)):
        if addrs[sidx].qadr >= 0:
            n_props += 1
    var mean_move = prop_move_sum / Float64(n_lanes * n_props)
    print("  contacts: max", max_ncon, "in a state;", eps_with_contact, "of",
          n_lanes, "episodes end in contact")
    print("  props   : per object, mean metres moved — settle / whole episode")
    for k in range(len(prop_names)):
        var padn = String(prop_names[k])
        while padn.byte_length() < 26:
            padn += " "
        print("              " + padn
              + String(prop_settle[k] / Float64(n_lanes)) + "  /  "
              + String(prop_move[k] / Float64(n_lanes)))
    print("  arm     : worst grip-site drift under the null action",
          ee_drift_max, "m")
    if max_ncon == 0:
        raise Error(
            "no contact was detected in any state of any episode. Eight of this"
            " family's ten goals are CONTACT predicates (`On`, `In`), so they"
            " cannot fire and the 0.0 above is vacuous. This is a collision"
            " failure, not a policy result."
        )
    if eps_with_contact < n_lanes:
        raise Error(
            "only " + String(eps_with_contact) + " of " + String(n_lanes)
            + " episodes end with any contact at all. A prop that is airborne"
            " or has left the table cannot satisfy an `On`, and the zero above"
            " would be reporting that rather than the policy."
        )
    if mean_move < 1.0e-4:
        raise Error(
            "the props moved a mean of " + String(mean_move) + " m over "
            + String(SETTLE_STEPS + max_steps) + " policy steps. Nothing is"
            " being simulated; every column above is a constant."
        )
    if ee_drift_max > 0.25:
        raise Error(
            "the grip site drifted " + String(ee_drift_max) + " m under a ZERO"
            " action. OSC_POSE holds the pose it is given, so this is a"
            " diverged controller — and an arm that has swept the table clear"
            " reports 0.0 for every task."
        )
    var bad = String("")
    for t in range(n_tasks):
        if at_reset[t] > 0:
            bad += "\n    " + task_names_used[t] + ": solved AT RESET on " \
                + String(at_reset[t]) + " of " + String(n_inits)
        if at_settle[t] > 0:
            bad += "\n    " + task_names_used[t] + ": solved after the five SETTLE" \
                + " steps on " + String(at_settle[t]) + " of " + String(n_inits)
        if per_task[t] > 0:
            bad += "\n    " + task_names_used[t] + ": solved by the NULL ACTION on " \
                + String(per_task[t]) + " of " + String(n_inits)
    if bad.byte_length() > 0:
        raise Error(
            "the null policy solves tasks in this family:" + bad
            + "\n  A goal a constant action meets is a goal with nothing above"
            " it — the training curve starts at its own ceiling. This is a bug"
            " in the goal (G3-G6), not in the policy; see the header."
        )
    print("  the null policy solves NOTHING: 0 at reset, 0 after the settle,"
          " 0 over the horizon")
    print()
    print("=== " + String(n_lanes) + " episodes, " + String(n_tasks)
          + " tasks, null-action success 0.0 ===")
