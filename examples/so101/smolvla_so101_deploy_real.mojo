# +--------------------------------------------------------------------------+ #
# | SmolVLA on the PHYSICAL SO-101 — chunk-at-a-time, from two live cameras
# +--------------------------------------------------------------------------+ #
"""Drive the follower arm from a fine-tuned SmolVLA, on the real rig.

    pixi run build-opencv                     # ONCE
    pixi run build-serial                     # ONCE

    # SAFE BY DEFAULT: reads the arm and the cameras, runs the policy, prints
    # every command it WOULD have sent, and never energises anything.
    pixi run -e jetson smolvla-deploy-jetson -- --project so101-tower \\
        --devices /dev/soarm_cam_top,/dev/soarm_cam_wrist

    # --arm is what actually moves the robot. Be at the desk, hand on the power.
    ... --arm --seconds 30

⚠⚠ **`--arm` MOVES THE FOLLOWER, AND NOTHING ELSE DOES.** The opt-in is a scar
`record.mojo` earned on 2026-08-31; `docs/SO101_SERIAL_LAYER.md` §safety.

⚠⚠ **A `finally` DOES NOT COVER AN ABORT OR A SIGNAL.** If this dies hard the
follower is left holding its pose — recovery is `pixi run soarm-torque-off`
and the power switch.

## Why this loop is shaped differently from ACT's

`act_so101_deploy_real.mojo` queries EVERY 30 Hz step and blends ~58
overlapping chunks with a temporal ensemble, because its forward is 27 ms and
it can. SmolVLA cannot and does not need to:

  * one query is hundreds of milliseconds (`smolvla_so101_latency_probe.mojo`
    is the prerequisite measurement — run it on the board before believing
    any of this);
  * one CHUNK is 50 steps at 30 Hz = **1.667 s of motion**, and
    `n_action_steps` is the whole chunk. SmolVLA is DESIGNED to be executed
    open-loop chunk-to-chunk.

So the shape is: query once, EXECUTE the chunk on the 30 Hz grid, query again
before it runs out. There is no temporal ensemble — the chunks do not overlap.

⚠ THE 30 Hz ACTION GRID IS KEPT, AND IT IS WHAT MAKES THE LATENCY HONEST. A
chunk requested for the observation at grid step `t_obs` describes steps
`t_obs .. t_obs+49`, exactly as in training. The query takes real time, so when
it returns the clock has moved on; the loop then indexes the chunk at
`now - t_obs` and the steps that elapsed during inference are SKIPPED, not
replayed late. The latency is not hidden, it is INDEXED — the same rule as
ACT's `t_cmd`, and the report's `skipped at handover` line is what it costs.

⚠ `--exec-steps` TRADES REACTIVITY AGAINST QUERY RATE, and both ends are
defensible. At 50 the arm runs a full 1.667 s on one observation — maximum
open-loop, minimum queries. Lower re-queries sooner on fresher pixels, at more
inference. It is a knob because the right value depends on a latency this file
refuses to guess.

## What is checked BEFORE anything is armed

The same four as the ACT deployment, for the same reasons:

1. **the arm is somewhere the demonstrations went** — the state box from
   `meta/stats.json`, printed per joint against the follower's present pose;
2. **the commanded action is clamped to the demonstrated action box** — the
   policy has never been asked what to do outside it and its answer there is
   not evidence of anything;
3. **the cameras are the right way round** — slot order is part of the
   checkpoint, and a swap feeds the policy a world it has never seen;
4. **the first query does not happen inside the loop.** Warm-up queries are
   run before arming: the first one compiles kernels, and on the Orin that was
   9.9 s for ACT. Paid inside the loop it is one command arriving seconds late
   with the arm already energised.

⚠ SAFETY IS NOT REIMPLEMENTED HERE. The step clamp and the calibrated range
live in `SO101Arm.write_goals`; the shutdown that ramps home and refuses to
drop the arm is `robot/so101/deploy_shutdown.mojo`, shared with ACT.
"""

from std.math import cos, log, sin, sqrt
from std.os import getenv
from std.os.path import exists
from std.sys import argv
from std.sys.defines import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.deep_agents.act.config import SO101_FPS
from mojo_rl.deep_agents.smolvla.finetune import load_trainables
from mojo_rl.deep_agents.smolvla.heads import (
    SMOLVLA_ACTION_DIM,
    SMOLVLA_EXPERT_W,
    SMOLVLA_STATE_DIM,
)
from mojo_rl.deep_agents.smolvla.expert import EXPERT_FF
from mojo_rl.deep_agents.smolvla.text import (
    SMOLLM_DIM,
    SMOLLM_KV_W,
    SMOLLM_LAYERS,
)
from mojo_rl.deep_agents.smolvla.observation import fill_camera_images
from mojo_rl.deep_agents.smolvla.policy import SmolVLAPolicy
from mojo_rl.deep_agents.smolvla.tasks import TaskTokens
from mojo_rl.io.fileio import StdinReader, stdin_is_tty
from mojo_rl.io.hf import hf_download_file, HF_MODEL
from mojo_rl.io.json import JsonDoc, load_json
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.robot.so101.deploy_shutdown import return_and_release
from mojo_rl.robot.so101.ports import follower_port, port_refusal
from mojo_rl.utils.fmt import col, fixed, pad_left, pad_right
from mojo_rl.vision.camera_thread import CameraReader, parse_camera_specs
from mojo_rl.vision.resize_pad import SIGLIP_INPUT


comptime TARGET: StaticString = "gpu" if is_defined["SMOLVLA_GPU"]() else "cpu"
"""⚠⚠ THE INFERENCE DEVICE, CHOSEN AT BUILD TIME — `-DSMOLVLA_GPU=1`.

⚠ ONE TARGET PER BINARY, AND HERE THAT IS NOT MERELY A COMPILE-TIME
PREFERENCE: every layer is instantiated per target and the weights are 3.2 GB
host+device. Building both would be 6.4 GB — the mistake that took a 16 GiB
laptop down once already (`smolvla_so101_latency_probe.mojo`)."""

comptime BASE_REPO = "lerobot/smolvla_base"
comptime N_CAM = 2
comptime N_LANG = 6
comptime CHUNK = 50
comptime STEPS = 10
"""Euler denoising steps at INFERENCE. Training denoises once; this does not."""
comptime RDIM = 6
"""The robot's real joint count. ⚠ NOT `SMOLVLA_ACTION_DIM` (32), which is the
padding the checkpoint was trained with — `select_action` drops the padding and
returns `CHUNK x stats.action_dim()`."""

comptime Pol = SmolVLAPolicy[N_CAM, N_LANG, CHUNK, STEPS, 1]

comptime CAM_W = 640
comptime CAM_H = 480
"""⚠⚠ THE CAMERA'S NATIVE SIZE, AND IT MUST BE THE ONE THE DEMONSTRATIONS WERE
RECORDED AT. Unlike the ACT deployment nothing is resized before the policy:
`fill_camera_images` does `resize_with_pad` to 512x512 itself, which is why the
recording is imported at 480x640 for SmolVLA and 240x320 for ACT. Feeding it a
pre-shrunk frame would resample twice."""

comptime WARMUP_QUERIES = 2
"""⚠ FEWER THAN ACT'S FIVE, because one query here costs hundreds of ms rather
than 27. Two still separates the first (which compiles kernels) from the
steady state, which is the whole point."""

comptime MAX_STEP_TICKS = 80
comptime TRACK_STEP_TICKS = 512
"""The same two-phase clamp the recorders and the ACT deployment use, so the
arm a policy drives moves like the arm that recorded its demonstrations."""

comptime ACTION_BOX_MARGIN = 0.05
"""5% of each joint's demonstrated span, added around the action box. The box
edges are real poses the arm reached, so clamping exactly at them would fight
the policy at the extremes of a legitimate motion."""

comptime DEFAULT_TASKS = "tools/vla/smolvla_tasks_record-test_20260828_092736.tsv"


def _stats_vec(
    ref doc: JsonDoc, key: String, field: String, mut out: List[Float64]
) raises -> Bool:
    """One vector out of `meta/stats.json`, False when the key is absent.

    ⚠ ABSENT IS NOT AN ERROR HERE, unlike mean/std. `min`/`max` are what the
    safety box is built from and LeRobot writes them, but a stats file without
    them is legible — so this reports the absence and the caller decides,
    rather than refusing to start a DRY RUN over a missing guard.
    """
    var root = doc.root()
    var node = doc.field(root, key)
    if node < 0:
        return False
    var arr = doc.field(node, field)
    if arr < 0:
        return False
    out.clear()
    for i in range(doc.size(arr)):
        out.append(doc.number(doc.at(arr, i)))
    return len(out) > 0


def main() raises:
    var arm_it = False
    var force = False
    var project = String("so101-tower")
    var ckpt = String("")
    var stats_path = String("")
    var tasks_path = String(DEFAULT_TASKS)
    var task_index = 0
    var seconds = 30
    var exec_steps = CHUNK
    var step_ticks = MAX_STEP_TICKS
    var devices = List[String]()
    var cam_fourcc = String("")
    var port_arg = String("")
    var do_return = True

    var args = argv()
    for i in range(len(args)):
        var a = String(args[i])
        if a == "--arm" or a == "--live":
            arm_it = True
        elif a == "--force":
            force = True
        elif a == "--project" and i + 1 < len(args):
            project = String(args[i + 1])
        elif a == "--ckpt" and i + 1 < len(args):
            ckpt = String(args[i + 1])
        elif a == "--stats" and i + 1 < len(args):
            stats_path = String(args[i + 1])
        elif a == "--tasks" and i + 1 < len(args):
            tasks_path = String(args[i + 1])
        elif a == "--task-index" and i + 1 < len(args):
            task_index = Int(String(args[i + 1]))
        elif a == "--seconds" and i + 1 < len(args):
            seconds = Int(String(args[i + 1]))
        elif a == "--exec-steps" and i + 1 < len(args):
            exec_steps = Int(String(args[i + 1]))
        elif a == "--step" and i + 1 < len(args):
            step_ticks = Int(String(args[i + 1]))
        elif a == "--port" and i + 1 < len(args):
            port_arg = String(args[i + 1])
        elif a == "--fourcc" and i + 1 < len(args):
            cam_fourcc = String(args[i + 1])
        elif a == "--no-return":
            do_return = False
        elif a == "--devices" and i + 1 < len(args):
            devices = parse_camera_specs(String(args[i + 1]))

    if len(devices) == 0:
        devices.append(String("0"))
        devices.append(String("1"))
    if len(devices) != N_CAM:
        raise Error(
            "smolvla deploy: the policy takes " + String(N_CAM)
            + " cameras but " + String(len(devices)) + " device(s) were given"
        )
    if exec_steps < 1 or exec_steps > CHUNK:
        raise Error(
            "smolvla deploy: --exec-steps must be 1.." + String(CHUNK)
            + " (the chunk is " + String(CHUNK) + " steps = "
            + fixed(Float64(CHUNK) / Float64(SO101_FPS), 2) + " s of motion)"
        )
    if ckpt == "":
        ckpt = getenv("SMOLVLA_CKPT", String(""))
    if ckpt == "":
        var promoted = "projects/" + project + "/policies/smolvla.ckpt"
        if exists(promoted):
            ckpt = promoted
    if ckpt == "":
        raise Error(
            "smolvla deploy: no fine-tuned weights. Pass --ckpt <file>, set"
            " $SMOLVLA_CKPT, or promote one into projects/" + project
            + "/policies/smolvla.ckpt.\n  ⚠ The BASE checkpoint alone has never"
            " seen this robot; running it would not be a test of anything."
        )
    if not exists(ckpt):
        raise Error("smolvla deploy: no such checkpoint: " + ckpt)
    if stats_path == "":
        stats_path = getenv("SMOLVLA_STATS", String(""))
    if stats_path == "" or not exists(stats_path):
        raise Error(
            "smolvla deploy: --stats <meta/stats.json> is required.\n"
            "  ⚠⚠ IT MUST BE THE SAME FILE THE FINE-TUNE USED. lerobot's"
            " population std and this repo's sample std differ by"
            " sqrt(N/(N-1)); normalizing with one while the weights were fit"
            " with the other is a silent scale error on every joint."
        )
    if not exists(tasks_path):
        raise Error(
            "smolvla deploy: no task table at " + tasks_path + " — it carries"
            " the PRE-TOKENISED instruction (there is no tokenizer at"
            " runtime). Pass --tasks <file>."
        )

    print("=" * 74)
    if arm_it:
        print("SmolVLA / SO-101 — CLOSED LOOP ON THE REAL ARM   [ARMED]")
    else:
        print("SmolVLA / SO-101 — DRY RUN (no torque, no goals written)")
        print("  pass --arm to actually move the follower")
    print("=" * 74)

    # ── the instruction ───────────────────────────────────────────────────
    var tasks = TaskTokens(tasks_path)
    var ids = tasks.for_index(task_index)
    if len(ids) != N_LANG:
        raise Error(
            "smolvla deploy: task " + String(task_index) + " is "
            + String(len(ids)) + " tokens, the policy was built for "
            + String(N_LANG)
        )
    print('instruction  "' + tasks.texts[task_index] + '"')
    print("             " + String(len(ids)) + " tokens, prefix P = "
          + String(Pol.P))

    # ── the policy ────────────────────────────────────────────────────────
    var dev_ctx = Optional[DeviceContext](None)
    comptime if TARGET != "cpu":
        dev_ctx = DeviceContext()
        print("device       " + String(dev_ctx.value().name())
              + "  (-DSMOLVLA_GPU=1)")
    else:
        print("device       CPU")

    print("base         " + String(BASE_REPO) + "  (~907 MB, cached)")
    var base = hf_download_file(
        String(BASE_REPO), String("model.safetensors"), HF_MODEL
    )
    print("             building every layer (3.2 GB host+device) ...")
    var pol = Pol.make[TARGET, Deterministic](dev_ctx)
    pol.load[TARGET](base, dev_ctx)

    # ⚠⚠ THE FINE-TUNE IS NOT SELF-CONTAINED. It holds the trainable set only
    # — the expert and the four action projections — so the base must be
    # loaded FIRST and this applied over it. Loading only this would leave the
    # vision tower and the language model at their initialisation.
    print("fine-tuned   " + ckpt)
    var sp_frozen = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM].make[
        TARGET, Deterministic
    ](dev_ctx)
    load_trainables[
        TARGET, SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
        SMOLLM_KV_W, SMOLVLA_ACTION_DIM,
    ](
        ckpt, pol.expert, pol.action_in, pol.time_mlp_in, pol.time_mlp_out,
        pol.action_out, sp_frozen, dev_ctx,
    )

    pol.load_stats(stats_path)
    if pol.stats.action_dim() != RDIM or pol.stats.state_dim() != RDIM:
        raise Error(
            "smolvla deploy: " + stats_path + " describes a "
            + String(pol.stats.state_dim()) + "-state / "
            + String(pol.stats.action_dim()) + "-action robot, this build is "
            + String(RDIM) + "/" + String(RDIM)
        )
    print("statistics   " + stats_path)

    # ── the safety boxes, from the SAME stats file ────────────────────────
    var doc = load_json(stats_path)
    var a_lo = List[Float64]()
    var a_hi = List[Float64]()
    var q_lo = List[Float64]()
    var q_hi = List[Float64]()
    var have_box = (
        _stats_vec(doc, String("action"), String("min"), a_lo)
        and _stats_vec(doc, String("action"), String("max"), a_hi)
        and _stats_vec(doc, String("observation.state"), String("min"), q_lo)
        and _stats_vec(doc, String("observation.state"), String("max"), q_hi)
    )
    if have_box:
        for j in range(RDIM):
            var pad = ACTION_BOX_MARGIN * (a_hi[j] - a_lo[j])
            a_lo[j] -= pad
            a_hi[j] += pad
    elif arm_it and not force:
        raise Error(
            "smolvla deploy: " + stats_path + " has no min/max, so there is no"
            " action box and nothing would clamp an extrapolated command."
            "\n  Refusing to arm. --force overrides, or use a stats file that"
            " carries min/max."
        )

    # ── warm up BEFORE anything is armed ─────────────────────────────────
    var images = Tensor()
    var scratch = List[Float32]()
    comptime XN = CHUNK * SMOLVLA_ACTION_DIM
    var noise = Tensor.alloc(XN)
    var act = List[Float32]()
    var pose = List[Float32](length=RDIM, fill=Float32(0.0))
    var warm_frames = List[List[UInt8]]()
    for _ in range(N_CAM):
        warm_frames.append(
            List[UInt8](length=CAM_W * CAM_H * 3, fill=UInt8(0))
        )
    var widths = List[Int](length=N_CAM, fill=CAM_W)
    var heights = List[Int](length=N_CAM, fill=CAM_H)
    for j in range(RDIM):
        pose[j] = Float32(pol.stats.state_mean[j])
    fill_camera_images[TARGET, N_CAM, SIGLIP_INPUT](
        warm_frames, widths, heights, True, images, scratch, dev_ctx
    )
    _fill_noise(noise, XN, 12345, dev_ctx)

    var first_ms = 0.0
    var warm_ms = 0.0
    for w in range(WARMUP_QUERIES + 1):
        var t0 = perf_counter_ns()
        pol.select_action[TARGET](images, ids, pose, noise, act, dev_ctx)
        comptime if TARGET != "cpu":
            dev_ctx.value().synchronize()
        var dt = Float64(perf_counter_ns() - t0) / 1e6
        if w == 0:
            first_ms = dt
        else:
            warm_ms += dt
    warm_ms /= Float64(WARMUP_QUERIES)
    var chunk_s = Float64(exec_steps) / Float64(SO101_FPS)
    print(
        "query        " + fixed(warm_ms, 1) + " ms warm (first "
        + fixed(first_ms, 1) + " ms, " + String(TARGET) + ")"
    )
    print(
        "             chunk " + String(CHUNK) + " steps, executing "
        + String(exec_steps) + " = " + fixed(chunk_s, 2) + " s of motion"
        " per query"
    )
    # ⚠⚠ THE COMPARISON THE WHOLE LOOP SHAPE RESTS ON. If a query costs more
    # than the motion it buys, the arm finishes its chunk before the next one
    # exists and STALLS on its last waypoint every cycle. That is not a slow
    # loop, it is a stuttering arm.
    if warm_ms / 1000.0 >= chunk_s:
        print(
            "   ⚠⚠ ONE QUERY (" + fixed(warm_ms / 1000.0, 2) + " s) COSTS MORE"
            " THAN THE " + fixed(chunk_s, 2) + " s IT BUYS. The arm will stall"
            " between chunks.\n      Raise --exec-steps (up to "
            + String(CHUNK) + "), or run the forward off the control thread —"
            " which this program does not do."
        )
    elif warm_ms / 1000.0 > chunk_s * 0.5:
        print(
            "   ⚠ a query costs " + fixed(warm_ms / 1000.0 / chunk_s, 2)
            + "x the motion it buys — over half the chunk is spent thinking"
            " about the next one."
        )

    # ── the cameras ───────────────────────────────────────────────────────
    print("")
    var cams = List[CameraReader]()
    for i in range(N_CAM):
        print("camera slot " + String(i) + " <- " + devices[i])
        # ⚠ NATIVE FRAMES, NOT RESIZED — unlike the ACT deployment, which asks
        # the camera thread for 320x240 CHW. `fill_camera_images` does
        # `resize_with_pad` to 512x512 itself, so a pre-shrunk frame would be
        # resampled twice and the policy would see pixels the fine-tune never
        # produced. rgb=False + swap_rb=True below: one swap, in the pass that
        # already touches every byte.
        var c = CameraReader.from_spec(
            devices[i], CAM_W, CAM_H, Float64(SO101_FPS), rgb=False,
            fourcc=cam_fourcc,
        )
        c.start(wait_ms=8000)
        var where = c.resolved_node()
        var got = c.negotiated_fourcc()
        var nfps = c.negotiated_fps()
        print(
            "            " + (where + "  " if where.byte_length() > 0
                               else String(""))
            + ("format " + got + "  " if got.byte_length() > 0
               else String(""))
            + (fixed(nfps, 1) + " fps" if nfps > 0.0 else String(""))
        )
        cams.append(c^)
    print(
        "            " + String(CAM_W) + "x" + String(CAM_H) + " native ->"
        " 512x512 resize_with_pad (SmolVLA's own, not ours)"
    )

    var frames = List[List[UInt8]]()
    for i in range(N_CAM):
        frames.append(
            List[UInt8](unsafe_uninit_length = cams[i].frame_bytes())
        )

    # ── the arm ───────────────────────────────────────────────────────────
    print("")
    var f_port = follower_port(port_arg)
    print("follower     " + f_port)
    var why_port = port_refusal(f_port, String("follower"))
    if why_port.byte_length() > 0:
        raise Error("smolvla deploy: " + why_port)
    var follower = SO101Arm(
        f_port,
        max_step_ticks=step_ticks,
        track_step_ticks=TRACK_STEP_TICKS,
    )
    follower.bus.timeout_ms = 20

    var raw = Array[Int32, SO101_N](fill=0)
    if follower.read_positions(Span(raw)) != SO101_N:
        raise Error(
            "smolvla deploy: the follower did not report " + String(SO101_N)
            + " positions — not arming"
        )

    print("")
    if have_box:
        print("   joint          present   demonstrated range        action clamp")
        var outside = 0
        for i in range(SO101_N):
            var p = follower.cal.degrees(i, raw[i])
            var note = String("")
            if p < q_lo[i] or p > q_hi[i]:
                outside += 1
                note = " ⚠ OUTSIDE"
            print(
                "   " + pad_right(joint_name(i), 14)
                + col(p, 8, 1) + "   [" + col(q_lo[i], 7, 1) + ","
                + col(q_hi[i], 7, 1) + " ]   [" + col(a_lo[i], 7, 1) + ","
                + col(a_hi[i], 7, 1) + " ]" + note
            )
        if outside > 0:
            print(
                "   ⚠ " + String(outside) + " joint(s) start outside the"
                " demonstrated pose box. The policy has never seen this"
                " observation;\n     its first command is an extrapolation and"
                " the step clamp is what makes that survivable."
            )
    else:
        print("   ⚠ no min/max in the stats file — no box to check against.")

    # ── go ────────────────────────────────────────────────────────────────
    var stdin = StdinReader()
    var interactive = stdin_is_tty()
    print("")
    if arm_it:
        print(
            "⚠⚠ THE FOLLOWER WILL BE ENERGISED AND WILL MOVE FOR "
            + String(seconds) + " s."
        )
    else:
        print("dry run — torque stays OFF and the arm is backdrivable.")
    if interactive:
        print("press Enter to start (q = quit), and Enter again to stop early")
    stdin.discard_pending()
    var answer = stdin.line()
    if answer == "q" or answer == "Q":
        for i in range(N_CAM):
            try:
                cams[i].stop()
            except:
                pass
        print("nothing was armed.")
        return

    var start_pose = List[Int32]()
    for i in range(SO101_N):
        start_pose.append(raw[i])
    var goals = Array[Int32, SO101_N](fill=0)
    for i in range(SO101_N):
        goals[i] = raw[i]

    if arm_it:
        # ⚠ PARK THE GOAL ON THE PRESENT POSE BEFORE ARMING, so torque engages
        # holding where the arm already is instead of snapping to whatever
        # `Goal_Position` happened to hold.
        follower.write_goals(Span(goals))
        follower.set_torque(True)
        print("\nfollower torque ON\n")
    else:
        print("\ndry run — nothing energised\n")

    var queries = 0
    var commands = 0
    var clamped = 0
    var bus_skipped = 0
    var skipped_at_handover = 0
    var sum_q = 0.0
    var worst_q = 0.0
    var sum_cam = 0.0
    var t_obs = -10_000
    var loop_ns = 0
    var cmd = List[Float64](length=RDIM, fill=0.0)
    var prev_cmd = List[Float64](length=RDIM, fill=0.0)
    var have_prev = False
    var sum_step = 0.0
    var max_step = 0.0

    var loop_t0 = perf_counter_ns()
    var deadline = loop_t0 + seconds * 1_000_000_000
    try:
        while perf_counter_ns() < deadline:
            if interactive and stdin.has_input():
                print("stopping early")
                break

            var now_ns = perf_counter_ns()
            var t_now = Int(
                Float64(now_ns - loop_t0) * Float64(SO101_FPS) / 1e9
            )

            # ── a new chunk, when the current one is used up ─────────────
            if t_obs < 0 or t_now - t_obs >= exec_steps:
                var t_c0 = perf_counter_ns()
                for i in range(N_CAM):
                    if cams[i].take_latest(frames[i]) == 0:
                        if not cams[i].take_blocking(frames[i]):
                            raise Error(
                                "smolvla deploy: camera " + devices[i]
                                + " stopped delivering frames"
                            )
                if follower.read_positions(Span(raw)) != SO101_N:
                    bus_skipped += 1
                    continue
                for j in range(RDIM):
                    pose[j] = Float32(follower.cal.degrees(j, raw[j]))
                fill_camera_images[TARGET, N_CAM, SIGLIP_INPUT](
                    frames, widths, heights, True, images, scratch, dev_ctx
                )
                sum_cam += Float64(perf_counter_ns() - t_c0) / 1e6

                # ⚠ FRESH NOISE EVERY QUERY. Flow matching integrates FROM a
                # sample of x_1; reusing one sample would make every chunk a
                # deterministic function of the observation and quietly throw
                # away the policy's action distribution.
                _fill_noise(noise, XN, queries * 7919 + 13, dev_ctx)

                # ⚠ THE OBSERVATION'S GRID STEP IS STAMPED BEFORE THE QUERY,
                # not after. The chunk describes the world as it was when the
                # cameras were read, and indexing it from `t_obs` is what makes
                # the inference latency a SKIP rather than a lag.
                var t_q = perf_counter_ns()
                t_obs = Int(
                    Float64(t_q - loop_t0) * Float64(SO101_FPS) / 1e9
                )
                pol.select_action[TARGET](
                    images, ids, pose, noise, act, dev_ctx
                )
                comptime if TARGET != "cpu":
                    dev_ctx.value().synchronize()
                var q_ms = Float64(perf_counter_ns() - t_q) / 1e6
                sum_q += q_ms
                if q_ms > worst_q:
                    worst_q = q_ms
                queries += 1
                t_now = Int(
                    Float64(perf_counter_ns() - loop_t0) * Float64(SO101_FPS)
                    / 1e9
                )
                skipped_at_handover += t_now - t_obs

            var idx = t_now - t_obs
            if idx < 0:
                idx = 0
            if idx >= CHUNK:
                idx = CHUNK - 1
            for j in range(RDIM):
                var v = Float64(act[idx * RDIM + j])
                if have_box:
                    if v < a_lo[j]:
                        v = a_lo[j]
                        clamped += 1
                    elif v > a_hi[j]:
                        v = a_hi[j]
                        clamped += 1
                cmd[j] = v

            if have_prev:
                var s = 0.0
                for j in range(RDIM):
                    var d = cmd[j] - prev_cmd[j]
                    s += d * d
                var st = sqrt(s)
                sum_step += st
                if st > max_step:
                    max_step = st
            for j in range(RDIM):
                prev_cmd[j] = cmd[j]
            have_prev = True

            if arm_it:
                for j in range(RDIM):
                    goals[j] = follower.cal.raw_from_degrees(j, cmd[j])
                follower.write_goals(Span(goals))
            commands += 1

            if commands % 10 == 0:
                var line = String("  t=") + pad_left(
                    fixed(Float64(t_now) / Float64(SO101_FPS), 1), 5
                ) + "s  i" + pad_left(String(idx), 3) + " "
                for j in range(RDIM):
                    line += " " + col(cmd[j], 7, 1)
                print(line)

            # Pace to the 30 Hz grid: the chunk's entries are 1/30 s apart and
            # commanding them faster would replay the demonstration in fast
            # forward.
            var next_ns = loop_t0 + (t_now + 1) * 1_000_000_000 // SO101_FPS
            while perf_counter_ns() < next_ns:
                pass
    finally:
        # ⚠ SEE THE ACT DEPLOYMENT: Mojo warns that this assignment is never
        # used and the warning is wrong; the value does propagate.
        loop_ns = perf_counter_ns() - loop_t0
        var released = return_and_release(
            follower, start_pose, arm_it, do_return, stdin, interactive
        )
        if not released:
            print(
                "⚠ the follower is STILL ENERGISED — that is deliberate, see"
                " above."
            )
        for i in range(N_CAM):
            try:
                cams[i].stop()
            except:
                pass

    var elapsed = Float64(loop_ns) / 1e9
    print("=" * 74)
    print("SmolVLA closed-loop run")
    print(
        "  commands          = " + String(commands) + " in "
        + fixed(elapsed, 1) + " s = "
        + fixed(Float64(commands) / elapsed if elapsed > 0.0 else 0.0, 1)
        + " Hz  (the 30 Hz action grid)"
    )
    print(
        "  queries           = " + String(queries) + " = one per "
        + fixed(Float64(commands) / Float64(queries) if queries > 0 else 0.0, 1)
        + " commanded steps"
    )
    print(
        "  query cost        = "
        + fixed(sum_q / Float64(queries) if queries > 0 else 0.0, 1)
        + " ms mean, " + fixed(worst_q, 1) + " ms worst"
    )
    # ⚠ THE COST OF NOT HIDING THE LATENCY. Every chunk's first
    # `skipped` steps are thrown away because the world moved on while the
    # policy was thinking. A large number here is the argument for putting the
    # forward on its own thread.
    print(
        "  skipped at handover = " + String(skipped_at_handover)
        + " grid steps total, "
        + fixed(
            Float64(skipped_at_handover) / Float64(queries)
            if queries > 0 else 0.0, 1
        )
        + " per query (the chunk's first steps, already stale on arrival)"
    )
    print("  observation build = "
          + fixed(sum_cam / Float64(queries) if queries > 0 else 0.0, 1)
          + " ms mean (cameras + resize_with_pad + upload)")
    print("  bus-skipped ticks = " + String(bus_skipped))
    print(
        "  action clamped    = " + String(clamped) + " of "
        + String(commands * RDIM) + " joint-commands"
    )
    print(
        "  commanded step    = mean "
        + fixed(sum_step / Float64(commands) if commands > 0 else 0.0, 2)
        + " deg, max " + fixed(max_step, 2) + " deg"
        + ("" if arm_it else "   <- dry run, the arm never moved")
    )
    if not arm_it:
        print("  ⚠ DRY RUN — nothing was written to the arm. Add --arm.")
    print("=" * 74)


def _fill_noise(
    mut noise: Tensor, n: Int, seed: Int, ctx: Optional[DeviceContext]
) raises:
    """x_1 ~ N(0,1) for the flow-matching sampler, freshly drawn.

    ⚠ A BOX-MULLER FROM A LOCAL LCG, not a shared RNG: this runs inside a
    control loop and must not be perturbed by, or perturb, anything else's
    stream. The quality bar is "plausibly Gaussian and different every query",
    which this clears.
    """
    var s = UInt64(seed * 2 + 1)
    for i in range(0, n, 2):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u1 = Float64((s >> 11) & UInt64(0x1FFFFFFFFFFFFF)) / 9.007199254740992e15
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u2 = Float64((s >> 11) & UInt64(0x1FFFFFFFFFFFFF)) / 9.007199254740992e15
        if u1 < 1e-12:
            u1 = 1e-12
        var r = sqrt(-2.0 * log(u1))
        var a = 6.283185307179586 * u2
        noise.data[i] = Scalar[DT](r * cos(a))
        if i + 1 < n:
            noise.data[i + 1] = Scalar[DT](r * sin(a))
    comptime if TARGET != "cpu":
        noise.upload(ctx.value())
