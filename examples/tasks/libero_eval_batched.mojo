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

## `--act DIR` — THE IMAGE POLICY, WITH THE CAMERAS IN THE LOOP (L7c)

`DIR` is a `libero-act-train` checkpoint directory (`best.ckpt` + `norm.json`).
Every policy step then renders BOTH of LIBERO's cameras for every lane with the
batched tracer — `raytrace/batch.mojo` over `env.d`, 128x128, 4x MSAA, the
visual group, after the step's FK sync — packs the pixels the way the store
holds them (uint8 CHW, top row first) and normalises them with the SAME
`normalize_camera_chw` the trainer's sampler used, reads the nine proprio words
from `qpos` and standardises them with the checkpoint's own `norm.json`, runs
one forward at batch `LANES`, pushes each lane's chunk into its own
`TemporalEnsemble`, and writes the ensembled, denormalised, clamped action.

⚠ THE OBSERVATION IS THE STATE THE LANE IS IN. `SYNC_FK_AFTER_STEP` is on for
the LIBERO config, so `xpos`/`xquat` on the device describe the integrated
`qpos` when the camera reads them; a stale FK here would be a one-step lag no
picture would show. The rendered store pairs frame `r` with `action[r]` for the
same reason (`libero_demo_rerender.mojo`'s header).

⚠⚠ `--act-exec N` — HOW MUCH OF EACH CHUNK IS EXECUTED, AND WHY IT IS A FLAG.
`0` (the default) is the paper's temporal ensemble: query every step, execute
the weighted mean of every chunk that covers it. `N >= 1` is LeRobot's
`n_action_steps`: query, execute the chunk's first N actions open-loop, query
again. The first 5090 run (2026-09-18, ensemble) scored 0/200 with a mean
|action| of 0.107 against the demonstrations' 0.294 — at `m = 0.01` the 40
overlapping predictions are weighted almost uniformly and their mean shrinks
towards the dataset's; `--act-exec 1` isolates that (query every step, take
the newest chunk's first action) and `--act-exec 10` is the half-second
open-loop setting LeRobot's LIBERO configs run. The per-word |action| table
at the end is printed beside the store's own mean and spread for this reason.

⚠⚠ `--check-obs [STORE]` — IS THE OBSERVATION THE ONE THE POLICY TRAINED ON?
Every gate so far compares physics or the store against the recording; none
compares the picture THIS driver renders inside the loop against the picture
the store holds. At the first policy step of the first chunk each lane's two
rendered frames are scored (PSNR, bytes) against the rendered store's frame 0
of the demonstration its init row came from — the frozen inits ARE the demos'
initial states, in order — and against the NEXT demo's frame 0 as a control.
A consistent pipeline reads 30 dB+ on the demo and clearly less on the
control; the five settle steps and the fixture draw (the store carries each
demo's, the env the band centre) cost a few dB, not twenty. Three image
policies at 0-5/200 with no such check is how a wrong picture hides.

⚠ `norm.json` NAMES THE STORE THE CHECKPOINT WAS FITTED ON, and this driver
prints it: a checkpoint from the RECORDED store crosses the pixel-domain gap
here (robosuite's OpenGL -> our tracer), one from the RENDERED store does not,
and the two rates are only comparable when the table says which is which.

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
from mojo_rl.physics3d.raytrace import BatchedCameraRenderer, RGB_CHANNELS
from mojo_rl.physics3d.raytrace.visual import build_visual_model
from mojo_rl.tasks.libero_visual import libero_site_conditions
from mojo_rl.tasks.libero_act import (
    LiberoActTrainer, LIBERO_ACT_QPOS, LIBERO_ACT_PROPRIO, LIBERO_ACT_ADIM,
    LIBERO_ACT_K,
    LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W, LIBERO_ACT_IMG_ELEMS, LIBERO_ACT_N_CAM,
)
from mojo_rl.deep_agents.act.norm_file import ACTNorm
from mojo_rl.deep_agents.act.inference import (
    TemporalEnsemble, normalize_camera_chw, denormalize,
)
from mojo_rl.deep_agents.act.config import ACT_TEMPORAL_ENSEMBLE_M
from mojo_rl.data.store import TrajectoryStore
from mojo_rl.tasks.libero_act import LIBERO_ACT_STORE_RENDERED
from std.math import log10
from std.memory.alloc import unsafe_alloc
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


def _byte(x: Float64) -> Scalar[DType.uint8]:
    """A tracer float in [0, 1] to the byte the store holds — the same
    rounding `libero_demo_rerender.mojo` wrote the training frames with."""
    var v = Int(x * 255.0 + 0.5)
    if v < 0:
        v = 0
    if v > 255:
        v = 255
    return Scalar[DType.uint8](v)


def _psnr_u8(
    a: List[Scalar[DType.uint8]], ao: Int,
    b: Pointer[Scalar[DType.uint8], MutAnyOrigin], bo: Int, n: Int,
) -> Float64:
    var se = 0.0
    for i in range(n):
        var d = Float64(Int(a[ao + i])) - Float64(Int(b[unsafe_offset = bo + i]))
        se += d * d
    var mse = se / Float64(n)
    return 99.0 if mse <= 0.0 else 10.0 * log10(255.0 * 255.0 / mse)


def _check_obs[AIMG: Int, ACAM: Int, ANPIX: Int](
    store_path: String, ref act_u8: List[Scalar[DType.uint8]],
    ref lane_row: List[Int], ref row_task: List[Int], n_inits: Int,
    mut out_own: List[Float64], mut out_ctrl: List[Float64],
) raises:
    """Each lane's packed observation vs the store's frame 0 of its own demo
    (init row i of task t == demo i of task t) and of the next demo."""
    if not exists(store_path):
        print("  ⚠ --check-obs: no store at", store_path, "— not checked")
        return
    var st = TrajectoryStore(store_path)
    var task_col = st.load_column[DType.int32](String("task_index"))
    var spec = st.column(String("images"))
    if spec.row_dim() != AIMG:
        raise Error("--check-obs: the store's images are " + String(spec.row_dim())
                    + " bytes per row, the policy's are " + String(AIMG))
    # episodes per task, in store order
    var per_task = List[List[Int]]()
    var n_tasks = 0
    for e in range(st.n_episodes()):
        var ti = Int(task_col[st.episodes.start_of(e)])
        while n_tasks <= ti:
            per_task.append(List[Int]())
            n_tasks += 1
        per_task[ti].append(e)
    var buf = unsafe_alloc[Scalar[DType.uint8]](AIMG).as_unsafe_any_origin()
    var lanes = len(lane_row)
    for l in range(lanes):
        var r = lane_row[l]
        if r < 0:
            continue
        var ti = row_task[r]
        var di = r % n_inits
        if ti >= n_tasks or di + 1 >= len(per_task[ti]):
            continue
        for which in range(2):
            var e = per_task[ti][di + which]
            var r0 = st.episodes.start_of(e)
            st.read_range[DType.uint8](String("images"), r0, r0 + 1, buf)
            for cam in range(AIMG // ACAM):
                var p = _psnr_u8(act_u8, l * AIMG + cam * ACAM, buf, cam * ACAM, ACAM)
                if which == 0:
                    out_own.append(p)
                else:
                    out_ctrl.append(p)
        if l < 4:
            print("    lane", l, "task", ti, "demo", di, ": agentview",
                  out_own[len(out_own) - 2], "/ ctrl", out_ctrl[len(out_ctrl) - 2],
                  "| eye_in_hand", out_own[len(out_own) - 1], "/ ctrl",
                  out_ctrl[len(out_ctrl) - 1], "dB")
    buf.unsafe_free()


def run[T: PlacementTable, M: ModelDefLike](
    n_inits: Int, max_steps: Int, check_lanes: Int, sampled: Bool,
    policy_path: String, act_dir: String, act_exec: Int, obs_store: String,
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
    # ── the image policy's types: the tracer over THIS env's Data, ACT at
    # a batch of LANES. Both are compiled whether or not `--act` is given;
    # neither is constructed unless it is.
    comptime Renderer = BatchedCameraRenderer[
        DT, E.MD, LANES, LIBERO_ACT_IMG_W, LIBERO_ACT_IMG_H, False, True, 4
    ]
    comptime ACT_T = LiberoActTrainer[LANES, "gpu"]
    comptime AQ = LIBERO_ACT_QPOS
    comptime AQP = LIBERO_ACT_PROPRIO
    comptime AA = LIBERO_ACT_ADIM
    comptime AK = LIBERO_ACT_K
    comptime AIMG = LIBERO_ACT_IMG_ELEMS
    comptime ANPIX = LIBERO_ACT_IMG_H * LIBERO_ACT_IMG_W
    comptime ACAM = 3 * ANPIX
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
    var have_bc = policy_path != ""
    var have_act = act_dir != ""
    if have_bc and have_act:
        raise Error("libero eval batched: --policy and --act are two policies;"
                    " give one")
    var have_policy = have_bc or have_act
    var net = POLICY.make["cpu", Kaiming](None)
    var norm = BcNorm()
    var act_norm = ACTNorm()
    if have_bc:
        load_params["cpu"](net, policy_path, None)
        norm = load_bc_norm(policy_path + ".norm", OD, OSC_ACTION_DIM)
        print("  policy:", policy_path, "| obs", OD, "-> 7, clamped to [-1, 1]")
    elif have_act:
        if not exists(act_dir + "/best.ckpt") or not exists(act_dir + "/norm.json"):
            raise Error("libero eval batched: --act " + act_dir + " has no"
                        " best.ckpt + norm.json (a libero-act-train checkpoint"
                        " directory)")
        act_norm = ACTNorm.load(act_dir + "/norm.json", AQ, AA)
        print("  policy: ACT", act_dir + "/best.ckpt", "| qpos", AQ, "+",
              LIBERO_ACT_N_CAM, "cameras", LIBERO_ACT_IMG_W, "x",
              LIBERO_ACT_IMG_H, "-> chunk", AK, "x", AA,
              ", temporal ensemble m =", ACT_TEMPORAL_ENSEMBLE_M)
        print("          fitted on", act_norm.store)
        if act_exec == 0:
            print("          chunk use: TEMPORAL ENSEMBLE (query every step)")
        else:
            print("          chunk use: execute", act_exec, "of", AK,
                  "open-loop, then re-query (--act-exec)")
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
    # ── the image policy, constructed only for --act ─────────────────────
    var ren_opt = List[Renderer]()
    var act_opt = List[ACT_T]()
    var cam_idx = List[Int]()
    var qadr9 = List[Int]()
    var ens = List[TemporalEnsemble[AA, AK]]()
    var h_rgb = ctx.enqueue_create_host_buffer[DT](1)
    var act_u8 = List[Scalar[DType.uint8]]()
    var act_images = List[Scalar[DT]]()
    var act_qpos = List[Scalar[DT]]()
    var act_dummy = List[Scalar[DT]]()
    var act_valid = List[Scalar[DT]]()
    var act_chunk = List[Scalar[DT]]()
    var act_pred = List[Scalar[DT]](length=AA, fill=Scalar[DT](0))
    var act_out = List[Scalar[DT]](length=AA, fill=Scalar[DT](0))
    var render_ns = 0
    var forward_ns = 0
    var physics_ns = 0
    var obs_checked = False
    var obs_psnr = List[Float64]()
    var obs_psnr_ctrl = List[Float64]()
    var word_abs = List[Float64](length=OSC_ACTION_DIM, fill=0.0)
    var word_n = 0
    if have_act:
        cam_idx.append(_index(fmd.camera_names, String("arena_agentview")))
        cam_idx.append(_index(fmd.camera_names, String("robot_eye_in_hand")))
        for j in range(ARM_DOF):
            qadr9.append(qadr_all[_index(fmd.joint_names, String("robot_joint") + String(j + 1))])
        qadr9.append(qadr_all[_index(fmd.joint_names, String("robot_finger_joint1"))])
        qadr9.append(qadr_all[_index(fmd.joint_names, String("robot_finger_joint2"))])
        if AQ != AQP + n_tasks:
            raise Error("libero eval batched: the ACT declaration carries "
                        + String(AQ - AQP) + " task words, the family has "
                        + String(n_tasks) + " tasks")
        # ⚠ THE SAME VISUAL MODEL THE RENDERED STORE WAS DRAWN WITH: group 1,
        # the stove burner rule, LIBERO's lights and textures.
        ren_opt.append(Renderer(ctx, env.mf, cam_idx[0]))
        ren_opt[0].set_visual(
            ctx,
            build_visual_model[DT, E.MD](
                fmd, env.mf, group_mask=1 << 1,
                conditions=libero_site_conditions(f),
            ),
        )
        print("  camera:", ren_opt[0].vis.describe())
        h_rgb = ctx.enqueue_create_host_buffer[DT](LANES * ANPIX * RGB_CHANNELS)
        act_opt.append(ACT_T.make(ctx=ctx))
        act_opt[0].load(act_dir + "/best.ckpt")
        for _ in range(LANES):
            ens.append(TemporalEnsemble[AA, AK](m=ACT_TEMPORAL_ENSEMBLE_M))
        act_u8 = List[Scalar[DType.uint8]](length=LANES * AIMG, fill=0)
        act_images = List[Scalar[DT]](length=LANES * AIMG, fill=Scalar[DT](0))
        act_qpos = List[Scalar[DT]](length=LANES * AQ, fill=Scalar[DT](0))
        act_dummy = List[Scalar[DT]](length=LANES * AK * AA, fill=Scalar[DT](0))
        act_valid = List[Scalar[DT]](length=LANES * AK, fill=Scalar[DT](1))
        act_chunk = List[Scalar[DT]](length=LANES * AK * AA, fill=Scalar[DT](0))

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
        for e in range(len(ens)):
            ens[e].reset()
        for step in range(SETTLE_STEPS + max_steps):
            var ap = act_h.unsafe_ptr()
            # ⚠ THE SETTLE STEPS ARE ZEROS EVEN WITH A POLICY — `metric.py`
            # steps its `dummy` through them, and the props are still falling.
            if step < SETTLE_STEPS or not have_policy:
                for k in range(LANES * OSC_ACTION_DIM):
                    ap[unsafe_offset=k] = Scalar[DT](0)
            elif have_act:
                var t_pol = step - SETTLE_STEPS
                var query = act_exec == 0 or t_pol % act_exec == 0
                # 1. both cameras, every lane, from the state the lanes are in
                var tr0 = perf_counter_ns()
                ref r = ren_opt[0]
                for cam in range(LIBERO_ACT_N_CAM if query else 0):
                    r.render(ctx, env.d, env.mf, cam_idx[cam])
                    ctx.enqueue_copy(h_rgb, r.rgb)
                    ctx.synchronize()
                    var p = h_rgb.unsafe_ptr()
                    for e in range(LANES):
                        var src = e * ANPIX * RGB_CHANNELS
                        var dst = e * AIMG + cam * ACAM
                        for q in range(ANPIX):
                            for c in range(3):
                                act_u8[dst + c * ANPIX + q] = _byte(
                                    Float64(p[unsafe_offset = src + q * 3 + c])
                                )
                        normalize_camera_chw[LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W](
                            act_u8, dst, act_images, dst
                        )
                if (obs_store.byte_length() > 0 and not obs_checked
                        and query and chunk == 0):
                    obs_checked = True
                    _check_obs[AIMG, ACAM, ANPIX](
                        obs_store, act_u8, lane_row, row_task, n_inits,
                        obs_psnr, obs_psnr_ctrl,
                    )
                # 2. the nine proprio words and the lane's task one-hot,
                #    standardised as the fit was (`env.d.qpos` was downloaded
                #    after the previous step; the task is the row's)
                for e in range(LANES):
                    var r_task = row_task[lane_row[e] if lane_row[e] >= 0 else 0]
                    for k in range(AQ):
                        var raw = (
                            Scalar[DT](env.d.qpos.data[e * NQ + qadr9[k]])
                            if k < AQP else
                            Scalar[DT](1.0 if k - AQP == r_task else 0.0)
                        )
                        act_qpos[e * AQ + k] = (
                            raw - act_norm.qpos_mean[k]
                        ) / act_norm.qpos_std[k]
                render_ns += perf_counter_ns() - tr0
                # 3. one forward at LANES when a query is due, then either
                #    each lane's ensemble or the chunk's next action
                if query:
                    var tf0 = perf_counter_ns()
                    act_opt[0].predict(act_qpos, act_images, act_dummy, act_valid, act_chunk)
                    forward_ns += perf_counter_ns() - tf0
                for e in range(LANES):
                    if act_exec == 0:
                        ens[e].push(t_pol, act_chunk, e * AK * AA)
                        ens[e].action_at(t_pol, act_pred, 0)
                    else:
                        var pos = t_pol % act_exec
                        for k in range(AA):
                            act_pred[k] = act_chunk[e * AK * AA + pos * AA + k]
                    denormalize(act_pred, 0, act_norm.action_mean,
                                act_norm.action_std, act_out, 0, AA)
                    for k in range(OSC_ACTION_DIM):
                        var a = _clamp(Float64(act_out[k]))
                        ap[unsafe_offset = e * OSC_ACTION_DIM + k] = Scalar[DT](a)
                        act_abs += abs(a)
                        act_words += 1
                        word_abs[k] += abs(a)
                    word_n += 1
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
            var tp0 = perf_counter_ns()
            env.step_batch[LANES](ctx, UInt64(step + 1))
            ctx.synchronize()
            physics_ns += perf_counter_ns() - tp0
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
            + (", ACT " + act_dir if have_act else
               (", policy " + policy_path if have_bc else ", null policy"))
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
    if have_act:
        print("  act   : fitted on", act_norm.store, "| cameras + qpos",
              Float64(render_ns) / 1e9, "s | forward", Float64(forward_ns) / 1e9,
              "s | physics (step_batch)", Float64(physics_ns) / 1e9,
              "s over the run")
        if len(obs_psnr) > 0:
            var m1 = 0.0
            var m2 = 0.0
            for k in range(len(obs_psnr)):
                m1 += obs_psnr[k]
                m2 += obs_psnr_ctrl[k]
            print("  obs   : first policy step vs the rendered store's frame 0 —"
                  " own demo", m1 / Float64(len(obs_psnr)), "dB | next demo (control)",
                  m2 / Float64(len(obs_psnr)), "dB over", len(obs_psnr),
                  "lane-cameras (" + obs_store + ")")
        if word_n > 0:
            print("  per word            policy mean|a|   store mean     store std")
            for k in range(OSC_ACTION_DIM):
                print("    " + _pad(String(k), 8) + " " + _pad(String(word_abs[k] / Float64(word_n)), 16)
                      + " " + _pad(String(Float64(act_norm.action_mean[k])), 14)
                      + " " + String(Float64(act_norm.action_std[k])))
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
    var act_dir = String("")
    var act_exec = 0
    var obs_store = String("")
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
        elif s == "--act" and i + 1 < len(args):
            act_dir = String(args[i + 1])
            i += 1
        elif s == "--act-exec" and i + 1 < len(args):
            act_exec = Int(String(args[i + 1]))
            i += 1
        elif s == "--check-obs":
            obs_store = String(LIBERO_ACT_STORE_RENDERED)
            if i + 1 < len(args) and not String(args[i + 1]).startswith("--"):
                obs_store = String(args[i + 1])
                i += 1
        elif s == "--sampled":
            sampled = True
        else:
            raise Error(
                "libero eval batched: unknown argument '" + s + "' (--inits N,"
                " --steps N, --check-lanes K, --sampled, --policy PATH,"
                " --act DIR, --act-exec N, --check-obs [STORE])"
            )
        i += 1

    if act_exec < 0 or act_exec > LIBERO_ACT_K:
        raise Error("--act-exec must be in [0, " + String(LIBERO_ACT_K)
                    + "] (0 = the temporal ensemble)")
    comptime if FAMILY == "libero_goal":
        run[LiberoGoalPlacement, LiberoGoalModel](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store,
        )
    elif FAMILY == "libero_object":
        run[LiberoObjectPlacement, LiberoObjectModel](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store,
        )
    elif FAMILY == "libero_spatial":
        run[LiberoSpatialPlacement, LiberoSpatialModel](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store,
        )
    elif FAMILY == "libero_kitchen_scene3":
        run[LiberoKitchenScene3Placement, LiberoKitchenScene3Model](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store,
        )
    elif FAMILY == "libero_kitchen_scene5":
        run[LiberoKitchenScene5Placement, LiberoKitchenScene5Model](
            n_inits, max_steps, check_lanes, sampled, policy_path, act_dir, act_exec,
            obs_store,
        )
    else:
        comptime assert False, (
            "libero_eval_batched: FAMILY is not one of the imported families —"
            " add its branch (see the FAMILY docstring)"
        )
