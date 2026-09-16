"""LIBERO'S OWN EVAL PROTOCOL, ON THE BATCH — the benchmark number's driver.

    pixi run -e nvidia mojo run -I . examples/tasks/libero_eval_batched.mojo
    pixi run -e nvidia mojo run -I . examples/tasks/libero_eval_batched.mojo --inits 20 --steps 600
    pixi run -e apple  mojo run -I . examples/tasks/libero_eval_batched.mojo --steps 40   # a small family

`examples/tasks/libero_eval.mojo` runs `lifelong/metric.py`'s loop one env at a
time on the CPU. This is the same loop on the batched GPU env, lane per
episode:

    reset_batch                  -> the family's base pose; OSC anchored THERE
    write the frozen row         -> qpos/qvel wholesale, controller NOT re-anchored
    5 settle steps of zeros(7)   -> the props land (LIBERO's own drop)
    up to `--steps` policy steps -> success at ANY step, per lane
    dones[k] |= done[k]

## ⚠⚠ THE ANCHOR, AND WHY THE ORDER IS THE PROTOCOL

robosuite builds the OSC controller in `Robot.reset`, so its nullspace target
`q0` is the configuration AT CONSTRUCTION — the family's rest pose — and
`set_init_state` overwrites `qpos` without rebuilding it. `reset_batch`
anchors after its own reset (`_osc_anchor`, the rest pose), and this driver
then writes the row and does NOT re-anchor. That is deliberate:
`libero_eval.mojo`'s header measured the difference (row 0 of
`open_the_middle_drawer` starts at joint2 = -0.188 against a rest pose of
-0.161), and `libero_demo_batched` takes the OTHER order on purpose because a
demo replay is not an eval.

## ⚠⚠ WHAT MAKES A NUMBER COMPARABLE, AND WHAT DOES NOT

* **The frozen rows.** `build/init/<family>.init.h5`, LIBERO's `.pruned_init`
  through `libero-init-dump` + `libero-init-freeze`. A success rate over states
  the run sampled for itself is not comparable with anything, so the frozen
  table is the only mode that prints a BENCHMARK number. It exists for
  `libero_goal` today.
* ⚠ **`--sampled` IS A SMOKE MODE, NOT A BENCHMARK.** Without a table (22
  families) the lanes are reset by the DEVICE sampler and the header says so
  on every line of the report. It exercises the protocol, the tape, the mask
  and the success word; it cannot be quoted against LIBERO's tables.
* **The mask comes from the TABLE, the goal from the `.task`.** `init_table`'s
  header explains: an eval months later must not change its answer because
  someone edited an `active=` line, while the GOAL is the benchmark's
  definition and belongs under version control.

## ⚠ THE POLICY IS `zeros(7)` UNTIL ONE EXISTS IN THE TREE

That is `metric.py`'s own `dummy`, so this is a real instance of the protocol
rather than a stand-in — and the L6 gate is that the rate is 0: a task solved
by the null action is a goal defect, not a policy result. `_policy_action` is
the one place a policy plugs in; it reads the env's observation rows, which
`task_hooks.write_task_obs` has already written for every lane.

## ⚠ THE SUCCESS WORD IS THE DEVICE'S, CHECKED AGAINST THE HOST ON A SAMPLE

`META_IDX_GOAL_HELD` per lane per step (the config's reward hook wrote it from
`eval_tape_gpu`). `--check-lanes K` re-evaluates K lanes per step with the HOST
evaluator on that lane's downloaded state, the way `libero_demo_batched` does;
a disagreement raises. Checking every lane every step is most of the run's
cost at 200 lanes, which is why it is a sample.

## ⚠ LANES ARE A COMPILE-TIME CONSTANT; THE TABLE IS RUN IN CHUNKS

`LANES` is `N_ENVS` of the env (one kernel instantiation per value, `sed` it —
the `libero_demo_batched` idiom). A table of 200 rows on 20 lanes runs as ten
chunks, and every row is recorded exactly once — `SuccessReport` refuses a
report with an unseen lane rather than counting it as a failure.
"""

from std.os import listdir
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.checkpoint import load_params
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.dynamics.osc_pose import ARM_DOF, OscPoseConfig
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
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, FamilySpec, TaskSpec,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses, BoundGoal,
)
from mojo_rl.tasks.eval import (
    eval_goal, HostState, region_sites, region_contact_bodies,
)
from mojo_rl.tasks.eval_report import SuccessReport
from mojo_rl.tasks.init_table import load_init_table, InitTable
from mojo_rl.tasks.tape import encode_goal, TAPE_WORDS
from mojo_rl.tasks.gpu_eval import region_table_words, require_gpu_regions
from mojo_rl.tasks.active import active_mask, init_region_words
from mojo_rl.tasks.bc_policy import BcNet, BcNorm, load_bc_norm
from mojo_rl.tasks.placement.table import PlacementTable
from mojo_rl.tasks.placement.check import (
    joint_init_words, require_device_placement,
)
from mojo_rl.tasks.libero_osc_config import LiberoOscConfig
from mojo_rl.tasks.placement.libero_goal import LiberoGoalPlacement
from mojo_rl.tasks.libero_goal_xml import LiberoGoalModel
from mojo_rl.tasks.placement.libero_object import LiberoObjectPlacement
from mojo_rl.tasks.libero_object_xml import LiberoObjectModel
from mojo_rl.tasks.placement.libero_spatial import LiberoSpatialPlacement
from mojo_rl.tasks.libero_spatial_xml import LiberoSpatialModel
from mojo_rl.tasks.placement.libero_kitchen_scene3 import LiberoKitchenScene3Placement
from mojo_rl.tasks.libero_envs.libero_kitchen_scene3_xml import LiberoKitchenScene3Model
from mojo_rl.tasks.placement.libero_kitchen_scene5 import LiberoKitchenScene5Placement
from mojo_rl.tasks.libero_envs.libero_kitchen_scene5_xml import LiberoKitchenScene5Model


comptime H = DType.float64
comptime FAMILY = "libero_goal"
"""The family this build evaluates. ⚠ `sed` it — `LANES` too, see the header.

⚠ ONLY THE FAMILIES BELOW ARE IMPORTED. A batched env is a kernel
instantiation per model and this driver is run on one family at a time;
`libero_family_batched.mojo` carries all 23 for the smoke gate. Add a branch
in `main` when a family gains an init table."""
comptime LANES = 20
comptime N_ENVS = LANES
comptime FAMILY_DIR = "mojo_rl/tasks/families/"
comptime TASK_DIR = "mojo_rl/tasks/tasks/"
comptime SETTLE_STEPS = 5
"""`metric.py`: `for _ in range(5): obs, _, _, _ = env.step(dummy)`."""
comptime LIBERO_MAX_STEPS = 600
"""`cfg.eval.max_steps`."""
comptime LIBERO_N_EVAL = 20
"""Frozen inits per task — the number every published LIBERO rate is over."""
comptime SAMPLED_SEED = 11


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("libero eval batched: no '" + want + "' in the scene")


def _task_names(family: String) raises -> List[String]:
    var out = List[String]()
    var want = family + "__"
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


def _clamp(x: Float64) -> Float64:
    """OSC_POSE's action box. ⚠ THE FIT HAS NO OUTPUT SQUASH (`bc_policy`), so
    a regression head can step outside [-1, 1]; robosuite clips there too."""
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def run[T: PlacementTable, M: ModelDefLike](
    n_inits: Int, max_steps: Int, check_lanes: Int, sampled: Bool,
    policy_path: String,
) raises:
    comptime E = Phyics3dBatchedEnv[
        M, LiberoOscConfig[T], LANES, CRBA_TREEWALK=True
    ]
    comptime NQ = M.NQ
    comptime NV = M.NV
    comptime NB = M.NBODY
    comptime NS = M.NSITE
    comptime MC = M.MAX_CONTACTS
    comptime OD = M.OBS_DIM
    var family = String(FAMILY)

    print("=" * 78)
    print("LIBERO's eval protocol on the batch —", family, "|", LANES, "lanes")
    print("=" * 78)

    var f = load_family(String(FAMILY_DIR) + family + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var names = _task_names(family)
    var n_tasks = len(names)

    # ── the tasks: the goal (from the .task) and its words ────────────────
    var nqs = List[Int]()
    var jt = List[Int]()
    var jvn = List[Int]()
    for k in range(len(fmd.joints)):
        nqs.append(fmd.joints[k].nq)
        jt.append(fmd.joints[k].jnt_type)
        jvn.append(fmd.joints[k].nv)
    var jadr = joint_qpos_addresses(nqs)
    var goals = List[BoundGoal]()
    var tapes = List[List[Float64]]()
    var masks = List[Float64]()
    var iwords = List[List[Float64]]()
    var jwords = List[List[Float64]]()
    var languages = List[String]()
    for ti in range(n_tasks):
        var t = load_task(String(TASK_DIR) + names[ti] + ".task")
        validate_task_against_family(t, f)
        var g = bind_goal(
            parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
            fmd.joint_names, jadr,
        )
        require_tier_a(g, t.name)
        require_gpu_regions(g, t.name)
        require_device_placement[T](t, f)
        tapes.append(encode_goal(g))
        masks.append(active_mask(t, f))
        iwords.append(init_region_words(t, f))
        jwords.append(joint_init_words[T](t))
        languages.append(t.language)
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

    # ── the rows: LIBERO's frozen fifty, or the sampler ───────────────────
    var table_path = String("build/init/") + family + ".init.h5"
    var have_table = exists(table_path) and not sampled
    var row_task = List[Int]()
    var row_mask = List[Float64]()
    var row_qpos = List[List[Float64]]()
    var row_qvel = List[List[Float64]]()
    var report_opt = List[SuccessReport]()
    if have_table:
        var full = load_init_table(table_path, f.name, NQ, NV)
        var tbl = full.prefix_per_task(n_inits)
        # ⚠ THE TABLE'S LABEL AGAINST THE `.task`'s `language=`, per row: the
        # table's `task_index` is an ORDER, and `libero_eval.mojo` refuses a
        # table whose order is not this driver's task list.
        for r in range(tbl.n_rows()):
            var ti = Int(tbl.task_index[r])
            if ti < 0 or ti >= n_tasks:
                raise Error(
                    table_path + ": row " + String(r) + " names task index "
                    + String(ti) + ", the family has " + String(n_tasks)
                )
            if tbl.task_label(r) != languages[ti]:
                raise Error(
                    table_path + ": row " + String(r) + " is labelled '"
                    + tbl.task_label(r) + "', task " + String(ti) + " ("
                    + names[ti] + ") says '" + languages[ti] + "'"
                )
            var q = List[Float64](length=NQ, fill=0.0)
            var v = List[Float64](length=NV, fill=0.0)
            tbl.apply(r, q, v)
            row_task.append(ti)
            row_mask.append(tbl.mask[r])
            row_qpos.append(q^)
            row_qvel.append(v^)
        report_opt.append(SuccessReport(tbl))
        print("  inits :", table_path, "|", full.n_rows(), "rows ->",
              tbl.n_rows(), "(", n_inits, "per task ) — LIBERO's OWN")
    else:
        # ⚠ SMOKE MODE. Every line below says so; it is not a benchmark rate.
        for ti in range(n_tasks):
            for _ in range(n_inits):
                row_task.append(ti)
                row_mask.append(masks[ti])
                row_qpos.append(List[Float64]())
                row_qvel.append(List[Float64]())
        if sampled:
            print("  inits : SAMPLED by the device (--sampled) — NOT a"
                  " benchmark number")
        else:
            print("  inits : no table at", table_path,
                  "— SAMPLED by the device, NOT a benchmark number")
            print("          build it: pixi run libero-init-dump && pixi run"
                  " libero-init-freeze")
    var n_rows = len(row_task)
    print("  tasks :", n_tasks, "| rows", n_rows, "| horizon", SETTLE_STEPS,
          "settle +", max_steps, "steps", "(LIBERO's own)" if max_steps
          == LIBERO_MAX_STEPS else "(REDUCED)")
    # ⚠⚠ THE POLICY IS BUILT FROM `tasks/bc_policy.BcNet`, the SAME
    # declaration `libero_bc_train` fitted, and `load_params` validates every
    # layer's name and size — a checkpoint of a different shape raises here
    # rather than loading the layers that happen to match.
    comptime POLICY = BcNet[OD, OSC_ACTION_DIM]
    var have_policy = policy_path != ""
    var net = POLICY.make["cpu", Kaiming](None)
    var norm = BcNorm()
    if have_policy:
        load_params["cpu"](net, policy_path, None)
        norm = load_bc_norm(policy_path + ".norm", OD, OSC_ACTION_DIM)
        print("  policy:", policy_path, "| obs", OD, "-> 7, clamped to [-1, 1]")
    else:
        print("  policy: ZERO ACTION —", "the L6 gate is that the rate is 0")
    if not have_table:
        # ⚠ AND ON THE BOX THE TABLE IS SIMPLY NOT THERE: it is a gitignored
        # build artifact (642 KB), so a machine that pulled the repo has the
        # code and not the inits. Copy it or rebuild it.
        print("          (copy build/init/" + family + ".init.h5 from a"
              " machine that has it, or rebuild it there)")

    # ── the controller record ─────────────────────────────────────────────
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
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))
    var cfg = OscPoseConfig()
    var refs = build_osc_refs(
        dof^, qadr^, jidx^, tmin^, tmax^, act_idx^, site,
        fmd.sites[site].body_id, ga1, ga2,
        ctrl_min[ga1], ctrl_max[ga1], ctrl_min[ga2], ctrl_max[ga2],
        cfg.kp, cfg.damping_ratio, cfg.output_max_pos, cfg.output_max_ori,
        cfg.nullspace_kp, cfg.gripper_speed,
    )

    # ══ THE BATCH ══════════════════════════════════════════════════════════
    var ctx = DeviceContext()
    var env = E(ctx)
    env.set_osc_refs(refs, ctx)
    var cw = region_table_words(f, rsites, rcontact)
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DT](cw[k])
    env.mf.curriculum.upload(ctx)
    var act_h = ctx.enqueue_create_host_buffer[DT](LANES * OSC_ACTION_DIM)
    # ⚠ `env._obs` IS A DEVICE BUFFER, copied into a host one — the same
    # `enqueue_copy` `libero_osc_batched` does, at `M.OBS_DIM` per lane.
    var obs_h = ctx.enqueue_create_host_buffer[DT](LANES * OD)
    var pol_x = Tensor.alloc(LANES * OD)
    var pol_y = Tensor.alloc(LANES * OSC_ACTION_DIM)
    var solved = List[Bool](length=n_rows, fill=False)
    var first_step = List[Int](length=n_rows, fill=-1)
    var at_settle = List[Bool](length=n_rows, fill=False)
    # ⚠ ANTI-VACUITY ON THE POLICY ITSELF: a checkpoint that emits zeros — a
    # dead head, a normalisation that flattened its input, a load that filled
    # nothing — reports exactly like the null run, including its 0 successes.
    var act_abs = 0.0
    var act_words = 0
    var eval_cmp = 0
    var eval_bad = 0
    var saturated = 0
    var nonfinite = 0
    var singular_steps = 0
    var t_start = perf_counter_ns()

    var chunk = 0
    while chunk * LANES < n_rows:
        var base = chunk * LANES
        # ⚠ A SHORT LAST CHUNK RUNS THE SPARE LANES ON ROW 0 AND DISCARDS
        # THEM: the env's lane count is comptime, so the batch is always full.
        # `lane_row[e] < 0` marks a lane whose result is not recorded.
        var lane_row = List[Int]()
        for e in range(LANES):
            var r = base + e
            lane_row.append(r if r < n_rows else -1)

        # the words, then the device reset, then the frozen row
        for e in range(LANES):
            var r = lane_row[e] if lane_row[e] >= 0 else 0
            var ti = row_task[r]
            var mb = e * METADATA_SIZE
            for k in range(METADATA_SIZE):
                env.d.meta.data[mb + k] = Scalar[DT](0)
            for k in range(TAPE_WORDS):
                env.d.meta.data[mb + META_IDX_TASK_PARAM_0 + k] = Scalar[DT](
                    tapes[ti][k]
                )
            # ⚠ THE MASK IS THE TABLE'S ROW, NOT `active_mask(t, f)` — see the
            # header. In sampled mode they are the same value.
            env.d.meta.data[mb + META_IDX_TASK_ACTIVE] = Scalar[DT](row_mask[r])
            for k in range(META_INIT_SLOTS):
                env.d.meta.data[mb + META_IDX_INIT_REGION_0 + k] = Scalar[DT](0)
            for k in range(META_JINIT_SLOTS * META_JINIT_WORDS):
                env.d.meta.data[mb + META_IDX_JINIT_0 + k] = Scalar[DT](0)
            if not have_table:
                # the device sampler places this lane (smoke mode)
                for k in range(len(iwords[ti])):
                    env.d.meta.data[mb + META_IDX_INIT_REGION_0 + k] = Scalar[DT](
                        iwords[ti][k]
                    )
                for k in range(len(jwords[ti])):
                    env.d.meta.data[mb + META_IDX_JINIT_0 + k] = Scalar[DT](
                        jwords[ti][k]
                    )
            env.d.meta.data[mb + META_IDX_SHAPE_W_GOAL] = Scalar[DT](0)
            env.d.meta.data[mb + META_IDX_SHAPE_W_REACH] = Scalar[DT](0)
        env.d.meta.upload(ctx)
        ctx.synchronize()
        # ⚠ THE SEED IS THE CHUNK'S, so two chunks of the SAME sampled task do
        # not run the identical episode (`reset_batch` seeds by lane).
        env.reset_batch[LANES](ctx, UInt64(SAMPLED_SEED + chunk))
        ctx.synchronize()
        if have_table:
            env.d.qpos.download(ctx)
            env.d.qvel.download(ctx)
            ctx.synchronize()
            for e in range(LANES):
                var r = lane_row[e] if lane_row[e] >= 0 else 0
                for k in range(NQ):
                    env.d.qpos.data[e * NQ + k] = Scalar[DT](row_qpos[r][k])
                for k in range(NV):
                    env.d.qvel.data[e * NV + k] = Scalar[DT](row_qvel[r][k])
            env.d.qpos.upload(ctx)
            env.d.qvel.upload(ctx)
            ctx.synchronize()
            # ⚠ NO `_osc_anchor` HERE. See the header: the controller keeps the
            # rest-pose anchor `reset_batch` gave it, which is robosuite's own
            # order and NOT the replay gate's.

        var obs_row = List[Float64]()
        var action = List[Float64](length=OSC_ACTION_DIM, fill=0.0)
        for step in range(SETTLE_STEPS + max_steps):
            var ap = act_h.unsafe_ptr()
            # ⚠ THE SETTLE STEPS ARE ZEROS EVEN WITH A POLICY — `metric.py`
            # steps its `dummy` through them, and the props are still falling.
            if step < SETTLE_STEPS or not have_policy:
                for k in range(LANES * OSC_ACTION_DIM):
                    ap[unsafe_offset=k] = Scalar[DT](0)
            else:
                # the policy, lane by lane, on the env's own observation rows
                ctx.enqueue_copy(obs_h, env._obs)
                ctx.synchronize()
                var op = obs_h.unsafe_ptr()
                # every lane in ONE forward: the net's batch is LANES
                for e in range(LANES):
                    obs_row.clear()
                    for k in range(OD):
                        obs_row.append(Float64(op[unsafe_offset = e * OD + k]))
                    var z = List[Float64]()
                    norm.apply(obs_row, z)
                    for k in range(OD):
                        pol_x.data[e * OD + k] = Scalar[DT](z[k])
                net.forward["cpu", LANES](
                    TensorRefs[1](pol_x), pol_y, None
                )
                for e in range(LANES):
                    for k in range(OSC_ACTION_DIM):
                        var a = _clamp(
                            Float64(pol_y.data[e * OSC_ACTION_DIM + k])
                        )
                        ap[unsafe_offset = e * OSC_ACTION_DIM + k] = Scalar[DT](a)
                        act_abs += abs(a)
                        act_words += 1
            ctx.enqueue_copy(env._action, act_h)
            env.step_batch[LANES](ctx, UInt64(step + 1))
            ctx.synchronize()
            env.d.meta.download(ctx)
            env.d.qpos.download(ctx)
            ctx.synchronize()
            if env.osc_singular_lanes(ctx) > 0:
                singular_steps += 1

            var want_host = check_lanes > 0 and (
                step < SETTLE_STEPS + 2 or step % 25 == 0
            )
            if want_host:
                env.d.xpos.download(ctx)
                env.d.xquat.download(ctx)
                env.d.site_xpos.download(ctx)
                env.d.contacts.download(ctx)
                ctx.synchronize()
            for e in range(LANES):
                var r = lane_row[e]
                if r < 0:
                    continue
                var mb = e * METADATA_SIZE
                var nc = Int(env.d.meta.data[mb + META_IDX_NUM_CONTACTS])
                if nc >= MC:
                    saturated += 1
                for k in range(NQ):
                    var q = Float64(env.d.qpos.data[e * NQ + k])
                    if q != q or q > 1.0e6 or q < -1.0e6:
                        nonfinite += 1
                var dev = Float64(env.d.meta.data[mb + META_IDX_GOAL_HELD]) > 0.5
                # ⚠⚠ SUCCESS AT ANY STEP, AND ONLY AFTER A STEP — `dones[k] =
                # dones[k] or done[k]`. The settle steps count: LIBERO's loop
                # runs them through `env.step` too. A goal met at step 3 and
                # abandoned at step 4 is a SUCCESS.
                if dev:
                    if not solved[r]:
                        first_step[r] = step
                    solved[r] = True
                    if step < SETTLE_STEPS:
                        at_settle[r] = True
                if want_host and (e % max(1, LANES // check_lanes) == 0):
                    var st = HostState(
                        List[Float64](), List[Float64](), List[Float64]()
                    )
                    for k in range(NB * 3):
                        st.xpos.append(Float64(env.d.xpos.data[e * NB * 3 + k]))
                    for k in range(NB * 4):
                        st.xquat.append(Float64(env.d.xquat.data[e * NB * 4 + k]))
                    for k in range(NS * 3):
                        st.site_xpos.append(
                            Float64(env.d.site_xpos.data[e * NS * 3 + k])
                        )
                    for k in range(NQ):
                        st.qpos.append(Float64(env.d.qpos.data[e * NQ + k]))
                    st.site_body = site_body_tab.copy()
                    st.site_quat = site_quat_tab.copy()
                    st.body_parent = body_parent_tab.copy()
                    var ncon = nc if nc < MC else MC
                    st.ncon = ncon
                    for c in range(ncon):
                        var cb = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
                        st.con_a.append(
                            Int(env.d.contacts.data[cb + CONTACT_IDX_BODY_A])
                        )
                        st.con_b.append(
                            Int(env.d.contacts.data[cb + CONTACT_IDX_BODY_B])
                        )
                    var host = eval_goal(
                        goals[row_task[r]], f, st, rsites, rcontact
                    )
                    eval_cmp += 1
                    if host != dev:
                        eval_bad += 1
                        if eval_bad <= 10:
                            print("   EVAL MISMATCH row", r, names[row_task[r]],
                                  "step", step, ": device", dev, "host", host)
        print("  chunk", chunk, "done — rows", base, "..",
              (base + LANES - 1) if base + LANES <= n_rows else n_rows - 1,
              flush=True)
        chunk += 1

    # ── the report ────────────────────────────────────────────────────────
    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
    print()
    if have_table:
        ref report = report_opt[0]
        for r in range(n_rows):
            report.record(r, solved[r])
        report.show(
            String("LIBERO success — ") + family
            + (", policy " + policy_path if have_policy else ", null policy")
        )
    else:
        var per_task = List[Int](length=n_tasks, fill=0)
        for r in range(n_rows):
            if solved[r]:
                per_task[row_task[r]] += 1
        print("  SAMPLED success (NOT a benchmark number):")
        for ti in range(n_tasks):
            print("   ", _pad(names[ti], 58), per_task[ti], "/", n_inits)
    var n_solved = 0
    var n_settle = 0
    for r in range(n_rows):
        if solved[r]:
            n_solved += 1
        if at_settle[r]:
            n_settle += 1
    print()
    print("  rows", n_rows, "| solved", n_solved, "| of those, during the",
          SETTLE_STEPS, "settle steps:", n_settle)
    print("  success word: ", eval_cmp, "host comparisons,", eval_bad,
          "disagreeing")
    if have_policy:
        print("  policy: mean |action|",
              act_abs / Float64(act_words) if act_words > 0 else 0.0, "over",
              act_words, "words")
    print("  contacts saturated lane-steps", saturated, "| non-finite", nonfinite,
          "| singular steps", singular_steps)
    print("  wall", elapsed, "s for", n_rows, "episodes of", SETTLE_STEPS
          + max_steps, "steps")

    # ── the verdict ───────────────────────────────────────────────────────
    var fails = List[String]()
    if n_rows == 0:
        fails.append("no rows evaluated")
    if eval_cmp == 0 and check_lanes > 0:
        fails.append("the host never checked a success word")
    if eval_bad > 0:
        fails.append(String(eval_bad) + " success words disagree with the host")
    if have_policy and (act_words == 0 or act_abs == 0.0):
        fails.append(
            "the policy emitted only zeros — it is the null run wearing a"
            " checkpoint"
        )
    if nonfinite > 0:
        fails.append(String(nonfinite) + " non-finite qpos words")
    if saturated > 0:
        fails.append(String(saturated) + " lane-steps saturated max_contacts")
    # ⚠⚠ THE L6 GATE APPLIES TO THE NULL RUN ONLY: the null action must solve
    # nothing (a task above 0 is a goal defect, not a policy result). With a
    # policy the solved count is THE RESULT, and gating it at 0 would fail the
    # run for succeeding.
    if n_solved > 0 and not have_policy:
        fails.append(
            String(n_solved) + " episodes solved by the NULL action"
        )
    print()
    if len(fails) > 0:
        for i in range(len(fails)):
            print("  FAIL:", fails[i])
        raise Error(family + ": " + String(len(fails)) + " check(s) failed")
    # ⚠ THE MODE IS ON THE VERDICT LINE. A sampled run's PASS, pasted on its
    # own, is indistinguishable from the benchmark's otherwise — and the whole
    # point of the frozen table is that a rate over states the run chose for
    # itself is not comparable with anything.
    print("=== PASS —", family,
          "(LIBERO's frozen inits)" if have_table
          else "(SAMPLED inits — NOT a benchmark number)",
          ", success " + String(n_solved) + " / " + String(n_rows) if have_policy
          else ", null rate 0", "===")


def main() raises:
    var args = argv()
    var n_inits = LIBERO_N_EVAL
    var max_steps = LIBERO_MAX_STEPS
    var check_lanes = 2
    var sampled = False
    var policy_path = String("")
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--inits" and i + 1 < len(args):
            n_inits = Int(String(args[i + 1]))
            i += 1
        elif s == "--steps" and i + 1 < len(args):
            max_steps = Int(String(args[i + 1]))
            i += 1
        elif s == "--check-lanes" and i + 1 < len(args):
            check_lanes = Int(String(args[i + 1]))
            i += 1
        elif s == "--policy" and i + 1 < len(args):
            policy_path = String(args[i + 1])
            i += 1
        elif s == "--sampled":
            sampled = True
        else:
            raise Error(
                "libero eval batched: unknown argument '" + s + "' (--inits N,"
                " --steps N, --check-lanes K, --sampled, --policy PATH)"
            )
        i += 1

    comptime if FAMILY == "libero_goal":
        run[LiberoGoalPlacement, LiberoGoalModel](
            n_inits, max_steps, check_lanes, sampled, policy_path
        )
    elif FAMILY == "libero_object":
        run[LiberoObjectPlacement, LiberoObjectModel](
            n_inits, max_steps, check_lanes, sampled, policy_path
        )
    elif FAMILY == "libero_spatial":
        run[LiberoSpatialPlacement, LiberoSpatialModel](
            n_inits, max_steps, check_lanes, sampled, policy_path
        )
    elif FAMILY == "libero_kitchen_scene3":
        run[LiberoKitchenScene3Placement, LiberoKitchenScene3Model](
            n_inits, max_steps, check_lanes, sampled, policy_path
        )
    elif FAMILY == "libero_kitchen_scene5":
        run[LiberoKitchenScene5Placement, LiberoKitchenScene5Model](
            n_inits, max_steps, check_lanes, sampled, policy_path
        )
    else:
        comptime assert False, (
            "libero_eval_batched: FAMILY is not one of the imported families —"
            " add its branch (see the FAMILY docstring)"
        )
