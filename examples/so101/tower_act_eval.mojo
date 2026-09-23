"""THE VISION STUDENT, CLOSED LOOP — cube-in-bowl rate with the cameras in the loop.

    pixi run -e nvidia mojo run -I . examples/so101/tower_act_eval.mojo \\
        --ckpt <run_id> --episodes 128
    pixi run -e nvidia mojo run -I . examples/so101/tower_act_eval.mojo \\
        --ckpt runs/<id>/checkpoints --dr full --dr-seed 1000   # held-out looks
    pixi run -e nvidia mojo run -I . examples/so101/tower_act_eval.mojo \\
        --ckpt runs/<id>/checkpoints --record-demo round1.demo   # DAgger states
    pixi run -e apple  mojo run -I . examples/so101/tower_act_eval.mojo --policy hold --steps 60

`LANES` episodes per round on the batched GPU env, lane per episode:

    reset_batch, then each lane's qpos = `posed_qpos(task, seed0 + episode)`
        — the expert recorder's placements (its host sampler), so the student
        and the teacher can be scored on the SAME scenes
    5 settle steps holding the arm where it is (the recorder's settle)
    up to `--steps` policy steps:
        host FK of every lane's qpos -> the rig's tracer, both cameras
        -> uint8 CHW -> `normalize_camera_chw`; the six joints in LeRobot
        units -> the checkpoint's `norm.json`; one ACT forward at batch LANES;
        the chunk (ensembled, or executed open-loop) denormalised -> a joint
        target in LeRobot units -> radians / gripper fraction -> the env's
        action word, clamped to [-1, 1]
    success = the task's goal bit (`META_IDX_GOAL_HELD`) held `HOLD_STEPS`
        consecutive steps — the recorder's rule, so a demo's "success" and
        this one are the same event

## ⚠⚠ SAME PIXELS, SAME UNITS AS THE STORE — BY CONSTRUCTION

The renderer, its visual set and background, the camera slots (overhead 0,
wrist 1), the byte packing and the LeRobot unit map are
`tasks/so101_tower_rig.mojo`, the module `tower_demo_rerender.mojo` renders
the training store with. The frame is host FK of the lane's CURRENT `qpos`
(the env's device poses lag one substep: the tower config does not sync FK
after a step), exactly as the store's frame r is FK of recorded `qpos[r]`.

## Seeds

`--seed0` (default 30000) + episode index. The expert files were recorded at
11000+ and 14000+ (`scripts/so101_tower_vision_box.sh`), 300 episodes each,
so the default block is disjoint from both. A run is a frozen set: the same
flags score the same scenes.

## Where the failures happen

Per failed lane, from the brick's and bowl's body positions after the settle
(`rest` = the brick's height then):

    no grasp        the brick never rose LIFT_DZ above rest
    dropped         lifted, then back to the desk (rest + LAND_DZ) more than
                    NEAR_BOWL from the bowl's centre
    missed bowl     lifted, landed within NEAR_BOWL of the bowl but the goal
                    never held
    goal not held   the goal bit was set at some step but not HOLD_STEPS in a row
    held to end     lifted and still in the air when the steps ran out

## `--joint-zero none|follower` — THE TRAINING STORE'S UNIT MAP

The joint zero the store was rendered with (`tower_demo_rerender.mojo
--joint-zero`, recorded in the store manifest's provenance line; absent =
`none`). The student's degrees mean nothing without it: evaluated under the
other map, every commanded pan is ~10 degrees off and the rate collapses with
nothing raising (`tasks/so101_tower_rig.mojo`).

## `--dr off|light|full` — held-out appearance

`physics3d/raytrace/randomize.mojo` on the rig's tables, ONE draw per ROUND
(draw index = round + `--dr-draw0`): every lane of a round shares the look,
successive rounds get new looks. A HELD-OUT appearance set is a seed the
store was not rendered with (`--dr-seed`), from the same ranges; the plan's
Phase 3 gate compares that rate with the rate under the store's own looks.

## `--record-demo FILE` — DAgger's data collector

Every lane's visited transitions (the env's observation, the action the
student EXECUTED, the env's reward, the next observation), one episode per
lane, success-stamped, written as a `.demo` — the file
`tower_expert_record.mojo`'s labelling pass reads. Rows after a lane's
success are not recorded.

## `-D TOWER_EVAL_NO_ACT` — a build without the network

The env, the tracer and ACT at batch `LANES` in one binary is a large compile
(it was OOM-killed at 18 min on a 16 GB M1 with swap already full). This
define leaves every ACT call out, so `--policy hold` checks the reset, the
settle, the rig, `--dr`, `--record-demo` and the report on a small machine.

## `--policy hold` — the null control

Hold the arm where it is, no network. The rate must be 0: a goal that the
null action meets is a goal defect, not a policy result. It also runs the
whole pipeline (reset, settle, render, report) without a checkpoint.
"""

from std.math import sqrt, log10
from std.memory.alloc import unsafe_alloc
from std.os import makedirs
from std.os.path import exists, isdir
from std.sys import argv, has_accelerator
from std.sys.defines import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.core.ptr import mptr
from noeira.deep_agents.act.config import (
    SO101_ADIM, SO101_IMG_H, SO101_IMG_W, SO101_N_CAM, SO101_QPOS,
    RUN_K, RUN_DIM, RUN_HEADS, RUN_FF, RUN_LATENT, RUN_ENC_LAYERS,
    RUN_DEC_LAYERS, ACT_TEMPORAL_ENSEMBLE_M,
)
from noeira.deep_agents.act.trainer import ACTTrainer
from noeira.deep_agents.act.norm_file import ACTNorm
from noeira.deep_agents.act.inference import (
    TemporalEnsemble, normalize_camera_chw, denormalize,
)
from noeira.deep_agents.demos.file import DemoSet, write_demo_file
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.io.png import save_png
from noeira.physics3d.fields import Data
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_GOAL_HELD, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.physics3d.raytrace.randomize import (
    DomainRandConfig, VisualRandomizer, geom_labels,
    so101_tower_surface_groups,
)
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig, So101TowerTeleopConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, RIG_CAM_W, RIG_CAM_H, RIG_N_CAMS, RIG_NPIX,
    RIG_CAM_ELEMS, RIG_IMG_ELEMS, RIG_ACT, RIG_DR_TARGET, make_tower_model,
    make_tower_renderer, tower_cameras, pack_camera_u8, So101TowerUnits,
    RIG_JOINT_ZERO_NONE,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.core.run import resolve_checkpoint


comptime DT = RIG_DT
comptime LANES = 32
"""Episodes per round. Comptime: the env, the tracer and ACT are instantiated
at it. 32 x 4 rounds = the 128 the other evals report over."""
comptime CFG = So101TowerTeleopConfig
"""The recorder's config (the tower's, with a 1200-step horizon), so the
env never truncates inside `--steps`."""
comptime E = Phyics3dBatchedEnv[So101TowerModel, CFG, LANES]
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime OD = E.OBS_DIM
comptime AD = E.ACT_DIM
comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime DEFAULT_TASK = "so101_tower_cube_in_bowl"

comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime K = RUN_K
comptime T = ACTTrainer[
    QPOS, ADIM, SO101_N_CAM, SO101_IMG_H, SO101_IMG_W, K, RUN_DIM, RUN_HEADS,
    RUN_FF, RUN_LATENT, RUN_ENC_LAYERS, RUN_DEC_LAYERS, LANES, target="gpu",
]

comptime WITH_ACT = not is_defined["TOWER_EVAL_NO_ACT"]()
"""False under `-D TOWER_EVAL_NO_ACT` — see the header."""

comptime HOLD_STEPS = 31
"""`tower_expert_record.mojo`'s success rule: the goal held this many steps."""
comptime SETTLE_STEPS = 5
comptime DEFAULT_SEED0 = 30000
comptime DEFAULT_STEPS = 600
"""19 s at 31.25 Hz. The expert's cube-in-bowl episode is ~236 steps + the
hold; a student is allowed 2.5x that."""
comptime LIFT_DZ = 0.03
comptime LAND_DZ = 0.01
comptime NEAR_BOWL = 0.08


def _usage() -> String:
    return String(
        "usage: tower_act_eval.mojo [--ckpt RUN_ID|DIR|FILE] [--ckpt-name best|last]"
        " [--norm FILE] [--policy act|hold] [--episodes N] [--seed0 S]"
        " [--steps N] [--exec N] [--m M] [--task NAME]"
        " [--dr off|light|full] [--dr-seed N] [--dr-draw0 N]"
        " [--record-demo FILE] [--snap DIR] [--joint-zero none|follower]"
    )


def _clamp1(x: Float64) -> Float64:
    return 1.0 if x > 1.0 else (-1.0 if x < -1.0 else x)


def _pct(n: Int, d: Int) -> String:
    if d == 0:
        return String("-")
    return String(Float64(Int(1000.0 * Float64(n) / Float64(d) + 0.5)) / 10.0) + "%"


def main() raises:
    comptime if not has_accelerator():
        print("  SKIPPED: no accelerator — the env, tracer and ACT are device code")
        print("=== SKIPPED (this is not a pass) ===")
        return
    comptime assert AD == RIG_ACT, "the env's action is the six joint targets"
    comptime assert SO101_N_CAM == RIG_N_CAMS
    comptime assert SO101_IMG_H == RIG_CAM_H and SO101_IMG_W == RIG_CAM_W

    # ── args ──────────────────────────────────────────────────────────────
    var args = argv()
    var ckpt_arg = String("")
    var ckpt_name = String("best")
    var norm_path = String("")
    var policy = String("act")
    var n_episodes = 128
    var seed0 = DEFAULT_SEED0
    var steps = DEFAULT_STEPS
    var exec_n = 0
    var ens_m = ACT_TEMPORAL_ENSEMBLE_M
    var task = String(DEFAULT_TASK)
    var dr_name = String("off")
    var dr_seed = 0
    var dr_draw0 = 0
    var demo_out = String("")
    var snap_dir = String("")
    var joint_zero = String(RIG_JOINT_ZERO_NONE)
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if not a.startswith("--") or i + 1 >= len(args):
            raise Error("bad argument " + a + "\n" + _usage())
        var v = String(args[i + 1])
        if a == "--ckpt":
            ckpt_arg = v
        elif a == "--ckpt-name":
            ckpt_name = v
        elif a == "--norm":
            norm_path = v
        elif a == "--policy":
            policy = v
        elif a == "--episodes":
            n_episodes = Int(v)
        elif a == "--seed0":
            seed0 = Int(v)
        elif a == "--steps":
            steps = Int(v)
        elif a == "--exec":
            exec_n = Int(v)
        elif a == "--m":
            ens_m = Float64(v)
        elif a == "--task":
            task = v
        elif a == "--dr":
            dr_name = v
        elif a == "--dr-seed":
            dr_seed = Int(v)
        elif a == "--dr-draw0":
            dr_draw0 = Int(v)
        elif a == "--record-demo":
            demo_out = v
        elif a == "--snap":
            snap_dir = v
        elif a == "--joint-zero":
            joint_zero = v
        else:
            raise Error("unknown option " + a + "\n" + _usage())
        i += 2
    if policy != "act" and policy != "hold":
        raise Error("--policy is act or hold, not " + policy)
    var use_act = policy == "act"
    comptime if not WITH_ACT:
        if use_act:
            raise Error(
                "this build has no network (-D TOWER_EVAL_NO_ACT): use"
                " --policy hold, or build without the define"
            )
    var ckpt_path = String("")
    if use_act:
        if ckpt_arg.byte_length() == 0:
            raise Error("--ckpt is required with --policy act\n" + _usage())
        if isdir(ckpt_arg):
            ckpt_path = ckpt_arg + "/" + ckpt_name + ".ckpt"
            if norm_path.byte_length() == 0:
                norm_path = ckpt_arg + "/norm.json"
        elif exists(ckpt_arg):
            ckpt_path = ckpt_arg
        else:
            # a RUN ID: its `checkpoints/<ckpt-name>.ckpt`, with the
            # trainer's `norm.json` beside it
            ckpt_path = resolve_checkpoint(ckpt_arg, ckpt_name)
            if norm_path.byte_length() == 0:
                norm_path = (
                    String(ckpt_path[byte=0 : ckpt_path.rfind("/")]) + "/norm.json"
                )
        if norm_path.byte_length() == 0:
            raise Error("--norm is required when --ckpt names a file")
        for pth in [ckpt_path, norm_path]:
            if not exists(pth):
                raise Error("no such file: " + pth)
    var n_rounds = (n_episodes + LANES - 1) // LANES

    print("=" * 78)
    print("so101_tower — vision student, CLOSED LOOP —", task)
    print("=" * 78)
    print("  policy :", policy, (" " + ckpt_path if use_act else String("")),
          "| exec", ("ensemble m=" + String(ens_m)) if exec_n == 0 else String(exec_n))
    print("  lanes  :", LANES, "|", n_rounds, "rounds =", n_rounds * LANES,
          "episodes | seeds", seed0, "..", seed0 + n_rounds * LANES - 1,
          "| steps", steps, "| success = goal held", HOLD_STEPS)

    # ── the env and the task's words ─────────────────────────────────────
    var ctx = DeviceContext()
    var env = E(ctx)
    var f = load_family(String(FAMILY_PATH))
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for k in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[k] = Scalar[DT](cw[k])
    env.mf.curriculum.upload(ctx)
    var mw = task_meta_words(
        task, String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    var brick = -1
    var bowl = -1
    for b in range(len(fmd.body_names)):
        if String(fmd.body_names[b]) == "brick_brick":
            brick = b
        if String(fmd.body_names[b]) == "bowl_bowl":
            bowl = b
    if brick < 0 or bowl < 0:
        raise Error("brick_brick / bowl_bowl not found in the composed scene")
    var units = So101TowerUnits(joint_zero)

    # ── the rig: the store's renderer, its own model and Data ───────────
    var rm = make_tower_model(ctx)
    var rd = Data[DT, TOWER_MD, LANES]()
    rd.upload_all(ctx)
    var r = make_tower_renderer[LANES](ctx, fmd, rm)
    var cams = tower_cameras(fmd)
    var dr_cfg = DomainRandConfig.parse(dr_name, UInt64(dr_seed))
    var dr = VisualRandomizer[DT](
        dr_cfg, so101_tower_surface_groups(), r.vis, rm, geom_labels(fmd),
        cams.copy(), r.background, RIG_DR_TARGET,
    )
    print("  dr     :", String(dr_cfg), "(one draw per round, from", dr_draw0, ")")
    print("  units  :", units.describe(), "— must be the training store's")
    var h_rgb = ctx.enqueue_create_host_buffer[DT](LANES * RIG_NPIX * 3)
    var img_u8 = List[Scalar[DType.uint8]](length=LANES * RIG_IMG_ELEMS, fill=0)

    # ── the policy ───────────────────────────────────────────────────────
    var norm_opt = Optional[ACTNorm](None)
    var tr_opt = Optional[T](None)
    if use_act:
        norm_opt = ACTNorm.load(norm_path, QPOS, ADIM)
        print("  norm   :", norm_path, "|", norm_opt.value().n_rows, "rows from",
              norm_opt.value().store)
        comptime if WITH_ACT:
            tr_opt = T.make(ctx=Optional[DeviceContext](ctx))
            tr_opt.value().load(ckpt_path)
    var qpos_n = List[Scalar[DT]](length=LANES * QPOS, fill=Scalar[DT](0))
    var images_n = List[Scalar[DT]](length=LANES * RIG_IMG_ELEMS, fill=Scalar[DT](0))
    var dummy_a = List[Scalar[DT]](length=LANES * K * ADIM, fill=Scalar[DT](0))
    var dummy_v = List[Scalar[DT]](length=LANES * K, fill=Scalar[DT](1))
    var chunk = List[Scalar[DT]](length=LANES * K * ADIM, fill=Scalar[DT](0))
    var pred_n = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0))
    var pred = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0))
    var ens = List[TemporalEnsemble[ADIM, K]]()
    for _ in range(LANES):
        ens.append(TemporalEnsemble[ADIM, K](m=ens_m))

    # ── host buffers for the env ─────────────────────────────────────────
    var act_h = ctx.enqueue_create_host_buffer[DT](LANES * AD)
    var obs_h = ctx.enqueue_create_host_buffer[DT](LANES * OD)
    var rew_h = ctx.enqueue_create_host_buffer[DT](LANES)
    var recording = demo_out.byte_length() > 0
    var demos = DemoSet(OD, AD)

    # ── the tallies ──────────────────────────────────────────────────────
    var n_ok = 0
    var n_nograsp = 0
    var n_dropped = 0
    var n_missed = 0
    var n_goal_not_held = 0
    var n_held_end = 0
    var steps_to_success = 0.0
    var act_words = 0
    var saturated = 0
    var nonfinite = 0
    var ns_render = 0
    var ns_forward = 0
    var ns_physics = 0
    var t0 = perf_counter_ns()

    for rnd in range(n_rounds):
        # ── reset: the task's words, the device reset, the expert's scenes
        for e in range(LANES):
            var mb = e * METADATA_SIZE
            for k in range(METADATA_SIZE):
                env.d.meta.data[mb + k] = Scalar[DT](0)
            for k in range(len(mw[0])):
                env.d.meta.data[mb + mw[0][k]] = Scalar[DT](mw[1][k])
        env.d.meta.upload(ctx)
        ctx.synchronize()
        env.reset_batch[LANES](ctx, UInt64(seed0 + rnd))
        ctx.synchronize()
        env.d.qpos.download(ctx)
        env.d.qvel.download(ctx)
        ctx.synchronize()
        for e in range(LANES):
            var q0 = posed_qpos[So101TowerPlacement](
                task, String(FAMILY), So101TowerConfig.SLOT_RADIUS,
                UInt64(seed0 + rnd * LANES + e),
            )
            for k in range(NQ):
                env.d.qpos.data[e * NQ + k] = Scalar[DT](q0[k])
            for k in range(NV):
                env.d.qvel.data[e * NV + k] = Scalar[DT](0)
        env.d.qpos.upload(ctx)
        env.d.qvel.upload(ctx)
        if dr_cfg.enabled:
            r.background = dr.apply(dr_draw0 + rnd, r.vis, rm)
            dr.upload(ctx, r.vis, rm)
        # settle: hold the arm where it is (the recorder's settle)
        for _ in range(SETTLE_STEPS):
            for e in range(LANES):
                for k in range(AD):
                    act_h[e * AD + k] = Scalar[DT](
                        _clamp1(units.joint_to_action(k, Float64(env.d.qpos.data[e * NQ + k])))
                    )
            ctx.enqueue_copy(env._action, act_h)
            env.step_batch[LANES](ctx, UInt64(1))
            env.d.qpos.download(ctx)
            ctx.synchronize()

        env.d.xpos.download(ctx)
        ctx.synchronize()
        var rest_z = List[Float64]()
        for e in range(LANES):
            rest_z.append(Float64(env.d.xpos.data[e * So101TowerModel.NBODY * 3 + brick * 3 + 2]))
        var held = List[Int](length=LANES, fill=0)
        var done = List[Bool](length=LANES, fill=False)
        var succ_step = List[Int](length=LANES, fill=-1)
        var lifted = List[Bool](length=LANES, fill=False)
        var land_d = List[Float64](length=LANES, fill=-1.0)
        var goal_ever = List[Bool](length=LANES, fill=False)
        for e in range(LANES):
            ens[e].reset()
        # per-lane transition buffers for --record-demo
        var rows_obs = List[List[Float32]]()
        var rows_act = List[List[Float32]]()
        var rows_rew = List[List[Float32]]()
        var rows_nobs = List[List[Float32]]()
        for _ in range(LANES):
            rows_obs.append(List[Float32]())
            rows_act.append(List[Float32]())
            rows_rew.append(List[Float32]())
            rows_nobs.append(List[Float32]())
        if recording:
            ctx.enqueue_copy(obs_h, env._obs)
            ctx.synchronize()

        var t_query = 0
        for t in range(steps):
            var all_done = True
            for e in range(LANES):
                if not done[e]:
                    all_done = False
            if all_done:
                break
            # ── the policy's action for every lane ──────────────────────
            if use_act:
                var query = exec_n == 0 or t % exec_n == 0
                if query:
                    var tr0 = perf_counter_ns()
                    for e in range(LANES):
                        for k in range(NQ):
                            rd.qpos.data[e * NQ + k] = env.d.qpos.data[e * NQ + k]
                    forward_kinematics["cpu", DT, TOWER_MD, LANES](rd, rm)
                    rd.qpos.upload_resident(ctx)
                    rd.xpos.upload_resident(ctx)
                    rd.xquat.upload_resident(ctx)
                    for slot in range(RIG_N_CAMS):
                        r.render(ctx, rd, rm, cams[slot])
                        ctx.enqueue_copy(h_rgb, r.rgb)
                        ctx.synchronize()
                        for e in range(LANES):
                            _ = pack_camera_u8(
                                mptr(h_rgb.unsafe_ptr()), e, mptr(img_u8),
                                e * RIG_IMG_ELEMS + slot * RIG_CAM_ELEMS,
                            )
                    if snap_dir.byte_length() > 0 and t == 0:
                        makedirs(snap_dir, exist_ok=True)
                        for slot in range(RIG_N_CAMS):
                            var hwc = List[UInt8](length=RIG_CAM_ELEMS, fill=UInt8(0))
                            for q in range(RIG_NPIX):
                                for c in range(3):
                                    hwc[q * 3 + c] = img_u8[slot * RIG_CAM_ELEMS + c * RIG_NPIX + q]
                            save_png(
                                snap_dir + "/round" + String(rnd) + "_lane0_"
                                + ("overhead" if slot == 0 else "wrist") + ".png",
                                hwc, RIG_CAM_W, RIG_CAM_H, 3,
                            )
                    ref nm = norm_opt.value()
                    for e in range(LANES):
                        for c in range(RIG_N_CAMS):
                            var o = e * RIG_IMG_ELEMS + c * RIG_CAM_ELEMS
                            normalize_camera_chw[RIG_CAM_H, RIG_CAM_W](
                                img_u8, o, images_n, o
                            )
                        for k in range(QPOS):
                            var lr = units.joint_to_lerobot(
                                k, Float64(env.d.qpos.data[e * NQ + k])
                            )
                            qpos_n[e * QPOS + k] = (
                                Scalar[DT](lr) - nm.qpos_mean[k]
                            ) / nm.qpos_std[k]
                    ns_render += perf_counter_ns() - tr0
                    var tf0 = perf_counter_ns()
                    comptime if WITH_ACT:
                        tr_opt.value().predict(
                            qpos_n, images_n, dummy_a, dummy_v, chunk
                        )
                    ns_forward += perf_counter_ns() - tf0
                    t_query = t
                ref nm = norm_opt.value()
                for e in range(LANES):
                    if exec_n == 0:
                        ens[e].push(t, chunk, e * K * ADIM)
                        ens[e].action_at(t, pred_n, 0)
                    else:
                        var idx = t - t_query
                        for k in range(ADIM):
                            pred_n[k] = chunk[e * K * ADIM + idx * ADIM + k]
                    denormalize(pred_n, 0, nm.action_mean, nm.action_std, pred, 0, ADIM)
                    for k in range(AD):
                        var lr = Float64(pred[k])
                        var a = units.joint_to_action(k, units.lerobot_to_joint(k, lr))
                        if not (a == a):
                            nonfinite += 1
                            a = 0.0
                        if a > 1.0 or a < -1.0:
                            saturated += 1
                        act_words += 1
                        act_h[e * AD + k] = Scalar[DT](_clamp1(a))
            else:
                for e in range(LANES):
                    for k in range(AD):
                        act_h[e * AD + k] = Scalar[DT](
                            _clamp1(units.joint_to_action(k, Float64(env.d.qpos.data[e * NQ + k])))
                        )
            # ── the step ────────────────────────────────────────────────
            var tp0 = perf_counter_ns()
            ctx.enqueue_copy(env._action, act_h)
            env.step_batch[LANES](ctx, UInt64(t + 2))
            env.d.qpos.download(ctx)
            env.d.meta.download(ctx)
            env.d.xpos.download(ctx)
            if recording:
                ctx.enqueue_copy(rew_h, env._reward)
            ctx.synchronize()
            ns_physics += perf_counter_ns() - tp0
            if recording:
                for e in range(LANES):
                    if done[e]:
                        continue
                    for k in range(OD):
                        rows_obs[e].append(Float32(obs_h[e * OD + k]))
                    for k in range(AD):
                        rows_act[e].append(Float32(act_h[e * AD + k]))
                    rows_rew[e].append(Float32(rew_h[e]))
                ctx.enqueue_copy(obs_h, env._obs)
                ctx.synchronize()
                for e in range(LANES):
                    if done[e]:
                        continue
                    for k in range(OD):
                        rows_nobs[e].append(Float32(obs_h[e * OD + k]))
            # ── the success rule and the failure bookkeeping ────────────
            for e in range(LANES):
                if done[e]:
                    continue
                var goal = Float64(env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD]) > 0.5
                if goal:
                    goal_ever[e] = True
                    held[e] += 1
                else:
                    held[e] = 0
                var bb = e * So101TowerModel.NBODY * 3
                var bz = Float64(env.d.xpos.data[bb + brick * 3 + 2])
                if bz > rest_z[e] + LIFT_DZ:
                    lifted[e] = True
                elif lifted[e] and land_d[e] < 0.0 and bz < rest_z[e] + LAND_DZ:
                    var dx = Float64(env.d.xpos.data[bb + brick * 3]) - Float64(env.d.xpos.data[bb + bowl * 3])
                    var dy = Float64(env.d.xpos.data[bb + brick * 3 + 1]) - Float64(env.d.xpos.data[bb + bowl * 3 + 1])
                    land_d[e] = sqrt(dx * dx + dy * dy)
                if held[e] >= HOLD_STEPS:
                    done[e] = True
                    succ_step[e] = t + 1

        # ── the round's outcome ─────────────────────────────────────────
        var r_ok = 0
        for e in range(LANES):
            var ep = rnd * LANES + e
            if ep >= n_rounds * LANES:
                continue
            if succ_step[e] >= 0:
                n_ok += 1
                r_ok += 1
                steps_to_success += Float64(succ_step[e])
            elif not lifted[e]:
                n_nograsp += 1
            elif goal_ever[e]:
                n_goal_not_held += 1
            elif land_d[e] < 0.0:
                n_held_end += 1
            elif land_d[e] > NEAR_BOWL:
                n_dropped += 1
            else:
                n_missed += 1
            if recording and len(rows_act[e]) > 0:
                demos.begin_episode()
                var n = len(rows_act[e]) // AD
                var o = List[Scalar[DT]](length=OD, fill=Scalar[DT](0))
                var no = List[Scalar[DT]](length=OD, fill=Scalar[DT](0))
                var a = List[Scalar[DT]](length=AD, fill=Scalar[DT](0))
                for rr in range(n):
                    for k in range(OD):
                        o[k] = Scalar[DT](rows_obs[e][rr * OD + k])
                        no[k] = Scalar[DT](rows_nobs[e][rr * OD + k])
                    for k in range(AD):
                        a[k] = Scalar[DT](rows_act[e][rr * AD + k])
                    demos.add(o, a, Float64(rows_rew[e][rr]), no, 0.0)
                demos.end_episode(success=succ_step[e] >= 0)
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        print("  round", rnd, ":", r_ok, "/", LANES, "| running", n_ok, "/",
              (rnd + 1) * LANES, "|", Int(secs), "s")

    # ── the report ──────────────────────────────────────────────────────
    var n = n_rounds * LANES
    print("-" * 78)
    print("  SUCCESS        ", n_ok, "/", n, "=", _pct(n_ok, n),
          ("| mean steps " + String(Int(steps_to_success / Float64(n_ok)))) if n_ok > 0 else String(""))
    print("  no grasp       ", n_nograsp, "(", _pct(n_nograsp, n), ")")
    print("  dropped        ", n_dropped, "(", _pct(n_dropped, n), ") — landed >",
          NEAR_BOWL, "m from the bowl")
    print("  missed bowl    ", n_missed, "(", _pct(n_missed, n), ") — landed near it")
    print("  goal not held  ", n_goal_not_held, "(", _pct(n_goal_not_held, n), ")")
    print("  held to end    ", n_held_end, "(", _pct(n_held_end, n), ")")
    if use_act:
        print("  actions        ", act_words, "words |", saturated, "saturated (",
              _pct(saturated, act_words), ") |", nonfinite, "non-finite")
        print("  time           render", Float64(ns_render) / 1e9, "s | forward",
              Float64(ns_forward) / 1e9, "s | physics", Float64(ns_physics) / 1e9, "s")
    if recording:
        write_demo_file(demo_out, demos)
        print("  wrote", demo_out, "|", demos.summary())
    if n_ok + n_nograsp + n_dropped + n_missed + n_goal_not_held + n_held_end != n:
        raise Error("tower act eval: the outcome buckets do not sum to the episodes")
    if nonfinite > 0:
        raise Error("tower act eval: " + String(nonfinite) + " non-finite action words")
    print("=== DONE ===")
