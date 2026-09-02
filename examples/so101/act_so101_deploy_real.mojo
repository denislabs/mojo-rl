# +--------------------------------------------------------------------------+ #
# | ACT on the PHYSICAL SO-101 — closed-loop, from two live cameras
# +--------------------------------------------------------------------------+ #
"""Drive the follower arm from the ACT policy, observing the world through the
same two cameras the demonstrations were recorded with.

    pixi run build-opencv                     # ONCE
    pixi run build-serial                     # ONCE

    # SAFE BY DEFAULT: reads the arm and the cameras, runs the policy, prints
    # every command it WOULD have sent, and never energises anything.
    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/act_deploy \\
        examples/so101/act_so101_deploy_real.mojo
    /tmp/act_deploy --store ~/.cache/mojo_rl/act_so101/<the one you trained on>.h5

    # --arm is what actually moves the robot. Be at the desk, hand on the power.
    /tmp/act_deploy --store <...> --arm --seconds 30

    # and BEFORE the first armed run on a rig whose cameras have been
    # unplugged, moved, or added to: check that device i is really slot i.
    /tmp/act_deploy --store <...> --snap /tmp/snap

This is the closed-loop counterpart of `act_so101_openloop_eval.mojo` and the
ACT counterpart of `deploy_reach_real.mojo`. Where the reach deployment ran a
SIM-trained policy and had to synthesise its observation from a physics model,
this one runs a policy trained on REAL demonstrations and its observation is
the real thing: two camera frames and the follower's own measured pose.

⚠⚠ **`--arm` MOVES THE FOLLOWER, AND NOTHING ELSE DOES.** The opt-in is a scar:
on 2026-08-31 `record.mojo` armed the follower with nobody at the desk because
the dangerous behaviour was the default and the flag was the safe one. Read
`docs/SO101_SERIAL_LAYER.md` §safety before the first armed run.

⚠⚠ **A `finally` DOES NOT COVER AN ABORT OR A SIGNAL.** If this dies hard the
follower is left holding its pose — the recovery is `pixi run soarm-torque-off`
and the power switch, not the `finally`.

## ⚠⚠ The measured fact this program is built around: ONE FORWARD IS ~95 ms

Measured on this M1 Pro, BATCH=1, the checkpoint's own dims, warm:

    CPU   (target="cpu")    95 ms   -> ~10.5 queries/s
    Metal (target="gpu")   155 ms   -> ~6.5  queries/s

**Metal is SLOWER, and that is why this program is CPU-only.** At BATCH=1 the
graph is a few hundred tiny kernels and Metal pays a command-buffer retirement
per launch (~20 us floor, `_the_metal_launch_floor_is_command_buffer_retirement`);
there is not enough work per kernel to hide it. Do not "fix" this by adding a
`--gpu` flag without re-measuring — the number above IS the measurement.

The policy was trained on 30 fps demonstrations, so a chunk's entries are
1/30 s apart. **At ~10 queries per second we cannot query every step, and this
program does not pretend to.** Instead:

  * there is a 30 Hz ACTION GRID, defined by wall clock — `t = round(elapsed *
    30)` — and the chunk index means exactly what it meant in training;
  * an observation captured at grid step `t_obs` is queried, and the chunk it
    returns is pushed into the temporal ensemble AS `t_obs`;
  * by the time the forward returns, the clock has moved on ~3 steps, so what
    is COMMANDED is the ensemble's action for `t_cmd` — the grid step it is
    actually now. The inference latency is not hidden, it is INDEXED.

⚠ THE CONSEQUENCE, STATED PLAINLY: the arm receives a waypoint about every
third demonstrated step, not every step. The trajectory plays at the right
SPEED (the grid is wall clock, so the policy does not run in slow motion), but
between waypoints the servo interpolates instead of the policy. Temporal
ensembling is what makes this tolerable — every commanded action is a weighted
blend of ~15 overlapping chunks, measured — and the `ensemble` figure in the
report is how you check it is actually filling up.

⚠ ONE SEMANTIC CONSEQUENCE OF QUERYING SPARSELY, recorded in
`TemporalEnsemble.action_at`: the ensemble weight `exp(-m*(i - i_min))` is the
paper's rank-based `w_i` only when every step in the window was queried. Here
it becomes a weight on AGE IN GRID STEPS instead — the right generalisation,
and with `m = 0.01` over `K = 60` a few percent either way.

Getting to a true 30 Hz needs the forward OFF the control thread, not a faster
device. That is a real design (the camera and encoder threads already prove the
machinery) and it is deliberately not attempted here: a first closed-loop
bring-up should not also be the debut of a cross-thread inference pipeline.

## What is checked BEFORE anything is armed

1. **the checkpoint and the store agree.** Normalization statistics are part of
   the policy and the checkpoint does not carry them — they are recomputed from
   the store, exactly as training did. Deploying with the WRONG store is
   therefore silent and dangerous: every command comes out shifted and scaled.
   So `--check` replays a held-out episode through the policy and reports the
   error against the same `hold` and `mean` baselines the open-loop evaluator
   uses. Worse than the constant `mean` baseline refuses to arm (`--force`
   overrides).
2. **the arm is somewhere the demonstrations went.** The dataset's own qpos box
   is printed against the follower's present pose, per joint. A joint parked
   outside it means the policy's first command is an extrapolation.
3. **the cameras are the right way round.** Slot 0 and slot 1 are NOT
   interchangeable — the store's camera order is alphabetical by feature key
   (`observation.images.front` then `...side`), and swapping them feeds the
   policy a world it has never seen. The mapping is printed; check it.

## Safety, on top of the two `SO101Arm` already enforces

`SO101Arm.write_goals` clamps to the calibrated `[range_min, range_max]` and to
`present ± max_step_ticks`. This program adds:

  4. the goal is parked on the follower's OWN present pose before torque is
     armed, so arming holds instead of snapping to a stale `Goal_Position`;
  5. a partial `sync_read` skips the tick's WRITE rather than commanding a
     half-updated pose;
  6. **the commanded action is clamped to the dataset's own action box** —
     the per-joint min and max the demonstrations ever reached. The policy has
     never been asked what to do outside it and its answer there is not
     evidence of anything. The report counts how often this fires; a clamp
     that fires constantly is telling you the deployment is off-distribution,
     not that the clamp is doing its job.
  7. **the run does not end by cutting torque where the policy left the arm.**
     The first armed run ended mid-reach and the arm DROPPED under its own
     weight. Releasing torque hands the arm to gravity, so where it is at that
     instant is a safety decision, not a detail. The shutdown now ramps back to
     the pose the run STARTED from — the one pose known to hold unpowered,
     because the arm was already resting there — confirms it arrived, holds,
     and releases on the operator's Enter. **If the ramp does not arrive,
     torque is LEFT ON**; a still-energised arm is recoverable with
     `pixi run soarm-torque-off`, and a fall is not. `--no-return` restores the
     old drop-where-it-stands behaviour and is for a run you are standing over.
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.act.config import (
    ACT_TEMPORAL_ENSEMBLE_M,
    RUN_DEC_LAYERS,
    RUN_DIM,
    RUN_ENC_LAYERS,
    RUN_FF,
    RUN_HEADS,
    RUN_K,
    RUN_LATENT,
    SO101_ADIM,
    SO101_FPS,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from mojo_rl.deep_agents.act.data import ACTDataset
from mojo_rl.deep_agents.act.inference import (
    TemporalEnsemble,
    denormalize,
    normalize_camera_chw,
)
from mojo_rl.deep_agents.act.trainer import ACTTrainer
from mojo_rl.io.fileio import StdinReader, stdin_is_tty
from mojo_rl.io.json import load_json
from mojo_rl.io.png import save_png
from mojo_rl.robot.so101 import SO101Arm, SO101_N, joint_name
from mojo_rl.utils.fmt import col, fixed, pad_left, pad_right
from mojo_rl.vision.camera_thread import CameraReader
from mojo_rl.vision.preprocess import camera_frame_to_chw_rgb


comptime FOLLOWER_PORT = "/dev/cu.usbmodem5B8E1139971"
comptime DEFAULT_CKPT = "act_so101_best_gpu.ckpt"

comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W
# ⚠ FROM `act.config`, never restated. These ARE the checkpoint's parameter
# shapes; the open-loop evaluator once carried its own copy and drifted from
# the trainer, so a checkpoint the trainer told you to evaluate could not be
# loaded by the evaluator.
comptime K = RUN_K
comptime DIM = RUN_DIM
comptime HEADS = RUN_HEADS
comptime FF = RUN_FF
comptime LATENT = RUN_LATENT
comptime N_ENC = RUN_ENC_LAYERS
comptime N_DEC = RUN_DEC_LAYERS
comptime BATCH = 1

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC,
    N_DEC, BATCH,
]
comptime CAM_ELEMS = 3 * IMG_H * IMG_W
comptime IMG_ELEMS = N_CAM * CAM_ELEMS

comptime CAM_W = 640
comptime CAM_H = 480
"""⚠⚠ THE CAMERA'S NATIVE SIZE, AND IT MUST BE THE ONE THE DEMONSTRATIONS WERE
RECORDED AT — 480x640 for `DenisLabs/record-test_*` (`meta/info.json`
`observation.images.front.shape`). It is not a preference: PIL's bilinear
window GROWS with the reduction factor, so 640x480 -> 320x240 and 1280x720 ->
320x240 are different filters and produce different pixels from the same
scene. Feeding the second to a model trained on the first is a silent
train/deploy gap. `--width` / `--height` exist for a different rig, not for
convenience."""

comptime MAX_STEP_TICKS = 80
"""~7 degrees per tick of the CONTROL loop. Same value `teleop.mojo` and
`record.mojo` measured with, and it is a bound on how far ahead of the arm a
goal may sit rather than a speed limit.

⚠ READ THE `throttle` LINE IN THE REPORT BEFORE BLAMING THE POLICY. This clamp
has NO COUNTERPART IN TRAINING — the demonstrations were recorded through it,
but the policy's output is not — so a clamp that binds is a deploy/train gap,
exactly as `deploy_reach_real.mojo` found the hard way (its first live run was
rate-limited by the clamp for its whole duration and looked like a bad policy)."""

comptime ACTION_BOX_MARGIN = 0.05
"""Widen the dataset's action box by 5% of its own span before clamping.

The box is a min/max over ~15,000 demonstrated frames, so its edges are real
poses the arm reached — clamping exactly at them would fight the policy at the
extremes of a legitimate motion (a fully closed gripper is AT the minimum). A
small margin keeps the guard about EXTRAPOLATION, which is what it is for."""


comptime RETURN_STEP_TICKS = 20
"""Per-write slew bound for the RETURN, at 30 Hz: ~52 deg/s.

Deliberately a quarter of `MAX_STEP_TICKS`. The return runs with no inference
in the loop, so it writes ~3.5x more often than the policy did; keeping the
same per-write clamp would make the way home three times faster than anything
the run itself did, which is the wrong direction for a move that happens while
someone is reaching for the arm."""

comptime RETURN_TOLERANCE_TICKS = 25
"""~2 degrees. Close enough to call it home — the servo settles inside its own
deadband and demanding better would spin until the timeout every time."""

comptime RETURN_TIMEOUT_S = 8
"""⚠ AND WHAT HAPPENS AT THE TIMEOUT IS THE POINT: torque is LEFT ON. An arm
that did not reach a pose it is known to rest in is an arm that must not be
released."""


def _spin_until(deadline_ns: Int):
    """Spin. Measured better than `usleep` on this box — see `teleop.mojo`."""
    while perf_counter_ns() < deadline_ns:
        pass


def return_and_release(
    mut arm: SO101Arm,
    ref start: List[Int32],
    armed: Bool,
    do_return: Bool,
    mut stdin: StdinReader,
    interactive: Bool,
) -> Bool:
    """Bring the follower home, hold, and only then release. True if released.

    ⚠⚠ **THIS EXISTS BECAUSE THE ARM FELL.** The first armed run ended by
    cutting torque wherever the policy happened to leave the arm — extended,
    mid-reach — and it dropped under its own weight. Releasing torque is not a
    neutral act: it is the moment gravity takes over, and where the arm IS at
    that moment decides whether that is safe.

    The pose the run STARTED from is the one pose known to be safe, because the
    arm was already resting there, unpowered, before anything was armed. So the
    shutdown goes: ramp back to it under the step clamp, confirm it arrived,
    hold there, and release only on the operator's word.

    ⚠ IF THE RAMP DOES NOT ARRIVE, TORQUE STAYS ON. That is the whole reason
    the arrival is checked rather than assumed. Torque surviving this process
    is recoverable — `pixi run soarm-torque-off` — and a fall is not.
    """
    if not armed:
        # Nothing was energised. The unconditional release still costs one
        # packet and is the net under every path that could have armed.
        try:
            arm.set_torque(False)
        except:
            pass
        return True

    if do_return:
        print("")
        print(
            "returning to the pose the run started from (<= "
            + String(RETURN_TIMEOUT_S) + " s) ..."
        )
        var hold = arm.max_step_ticks
        arm.max_step_ticks = RETURN_STEP_TICKS
        var goals = InlineArray[Int32, SO101_N](fill=0)
        for i in range(SO101_N):
            goals[i] = start[i]
        var present = InlineArray[Int32, SO101_N](fill=0)
        var period = 1_000_000_000 // 30
        var t_end = perf_counter_ns() + RETURN_TIMEOUT_S * 1_000_000_000
        var arrived = False
        var worst = 1 << 30
        while perf_counter_ns() < t_end:
            var t0 = perf_counter_ns()
            try:
                arm.write_goals(Span(goals))
            except:
                # The bus refused. Stop pushing and do NOT release — an arm
                # we can no longer command is the last thing to let go of.
                break
            try:
                if arm.read_positions(Span(present)) == SO101_N:
                    worst = 0
                    for i in range(SO101_N):
                        var d = Int(present[i]) - Int(start[i])
                        if d < 0:
                            d = -d
                        if d > worst:
                            worst = d
                    if worst <= RETURN_TOLERANCE_TICKS:
                        arrived = True
                        break
            except:
                break
            _spin_until(t0 + period)
        arm.max_step_ticks = hold
        if not arrived:
            print(
                "⚠⚠ DID NOT REACH THE START POSE (worst joint still "
                + String(worst) + " ticks away)."
            )
            print(
                "   TORQUE IS LEFT ON deliberately — releasing an arm that is"
                " not where it can rest\n   is how it falls. Support the arm,"
                " then run `pixi run soarm-torque-off`."
            )
            return False
        print("   home, worst joint " + String(worst) + " ticks off")

    if interactive:
        print("")
        print(
            "the follower is HOLDING. Take hold of the arm if you want to"
            " move it,\nthen press Enter to release torque."
        )
        stdin.discard_pending()
        try:
            _ = stdin.line()
        except:
            pass
    try:
        arm.set_torque(False)
        print("follower torque OFF")
    except:
        print(
            "⚠ COULD NOT RELEASE FOLLOWER TORQUE — run"
            " `pixi run soarm-torque-off`"
        )
        return False
    return True


def _split(s: String, sep: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(s.byte_length()):
        var c = chr(Int(s.as_bytes()[i]))
        if c == sep:
            out.append(cur^)
            cur = String("")
        else:
            cur += c
    out.append(cur^)
    return out^


def camera_names(store: String) raises -> List[String]:
    """The store's camera keys, IN SLOT ORDER, from the sidecar if there is one.

    ⚠ SLOT ORDER IS ALPHABETICAL BY FEATURE KEY — that is how the importer
    assigns them (`data/lerobot.mojo`, `cameras` sorted by name), and it is
    the only thing that ties a physical camera to a channel of the tensor.
    The names are cosmetic; printing them is not, because "slot 0" tells an
    operator nothing and "slot 0 = observation.images.front" tells them
    which camera to check.
    """
    var out = List[String]()
    if not store.endswith(".h5"):
        return out^
    var side = store[byte=0 : store.byte_length() - 3] + ".json"
    if not exists(side):
        return out^
    try:
        var doc = load_json(side)
        var cams = doc.field(doc.root(), String("cameras"))
        if cams < 0:
            return out^
        for i in range(doc.size(cams)):
            out.append(doc.string(doc.at(cams, i)))
    except:
        # A sidecar we cannot read is not a reason to refuse to deploy; it is
        # a reason to print slot numbers instead of names.
        out = List[String]()
    return out^


def store_path() raises -> String:
    """`--store` or `$ACT_STORE` — there is deliberately no default.

    ⚠⚠ THE STORE IS PART OF THE POLICY. The normalization statistics are
    recomputed from it, and the checkpoint carries none — so pointing this at
    the wrong recording does not fail, it produces commands that are shifted
    and scaled wrong, on a real arm. Two stores in this cache differ only by a
    date in their name and one has 5 episodes where the other has 50. A
    default here would be a coin flip with the arm as the stake.
    """
    var env = getenv("ACT_STORE")
    if env.byte_length() > 0:
        return env^
    raise Error(
        "act deploy: name the store the checkpoint was TRAINED on, with"
        " --store <path.h5> or ACT_STORE. There is no default — see"
        " `store_path`'s note."
    )


def main() raises:
    var arm_it = False
    var force = False
    var store = String("")
    var ckpt = String("")
    var seconds = 30
    var step_ticks = MAX_STEP_TICKS
    var smooth = 1.0
    var check_steps = 30
    var devices = List[Int]()
    var cam_w = CAM_W
    var cam_h = CAM_H
    var snap = String("")
    var do_return = True

    var args = argv()
    for i in range(len(args)):
        var a = String(args[i])
        if a == "--arm" or a == "--live":
            # `--live` is `deploy_reach_real.mojo`'s spelling and `--arm` is
            # `record.mojo`'s. Both mean "energise the follower"; accepting
            # both is cheaper than an operator discovering the difference by
            # typing the other one and getting a dry run they did not want.
            arm_it = True
        elif a == "--force":
            force = True
        elif a == "--store" and i + 1 < len(args):
            store = String(args[i + 1])
        elif a == "--ckpt" and i + 1 < len(args):
            ckpt = String(args[i + 1])
        elif a == "--seconds" and i + 1 < len(args):
            seconds = Int(String(args[i + 1]))
        elif a == "--step" and i + 1 < len(args):
            step_ticks = Int(String(args[i + 1]))
        elif a == "--smooth" and i + 1 < len(args):
            smooth = Float64(String(args[i + 1]))
        elif a == "--check" and i + 1 < len(args):
            check_steps = Int(String(args[i + 1]))
        elif a == "--width" and i + 1 < len(args):
            cam_w = Int(String(args[i + 1]))
        elif a == "--height" and i + 1 < len(args):
            cam_h = Int(String(args[i + 1]))
        elif a == "--no-return":
            # ⚠ THE ARM IS THEN RELEASED WHEREVER THE POLICY LEFT IT, which is
            # exactly how it fell the first time. For a run you are standing
            # over with a hand under the arm, nothing else.
            do_return = False
        elif a == "--snap" and i + 1 < len(args):
            snap = String(args[i + 1])
        elif a == "--devices" and i + 1 < len(args):
            var parts = _split(String(args[i + 1]), String(","))
            for k in range(len(parts)):
                if parts[k] != "":
                    devices.append(Int(parts[k]))
    if store == "":
        store = store_path()
    if ckpt == "":
        ckpt = String(DEFAULT_CKPT)
    if len(devices) == 0:
        devices.append(0)
        devices.append(1)
    if len(devices) != N_CAM:
        raise Error(
            "act deploy: the policy takes " + String(N_CAM) + " cameras but "
            + String(len(devices)) + " device(s) were given"
        )

    print("=" * 74)
    if arm_it:
        print("ACT / SO-101 — CLOSED LOOP ON THE REAL ARM   [ARMED]")
    else:
        print("ACT / SO-101 — DRY RUN (no torque, no goals written)")
        print("  pass --arm to actually move the follower")
    print("=" * 74)

    if not exists(store):
        raise Error("act deploy: no store at " + store)
    if not exists(ckpt):
        raise Error(
            "act deploy: no checkpoint at " + ckpt + " — train first"
            " (examples/so101/act_so101_train_gpu.mojo) or pass --ckpt"
        )

    # ── the policy's units ────────────────────────────────────────────────
    # ⚠ `max_image_bytes=0` FORCES THE STREAMED PATH. The statistics come from
    # the qpos and action columns, which are a few hundred KB; the image
    # column of the 50-episode store is 7.1 GiB and residency would load all
    # of it to compute nothing. `--check` then streams the handful of rows it
    # actually reads.
    print("store       " + store)
    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](
        store.copy(), seed=7, max_image_bytes=0
    )
    print(
        "            " + String(ds.n_rows()) + " frames, "
        + String(ds.n_episodes()) + " episodes, "
        + String(len(ds.train_eps)) + " train / "
        + String(len(ds.val_eps)) + " held out"
    )

    # The action box: what the demonstrations ever commanded, per joint.
    var a_lo = List[Float64](length=ADIM, fill=1.0e18)
    var a_hi = List[Float64](length=ADIM, fill=-1.0e18)
    var q_lo = List[Float64](length=QPOS, fill=1.0e18)
    var q_hi = List[Float64](length=QPOS, fill=-1.0e18)
    for r in range(ds.n_rows()):
        for j in range(ADIM):
            var v = Float64(ds.action_raw[r * ADIM + j])
            if v < a_lo[j]:
                a_lo[j] = v
            if v > a_hi[j]:
                a_hi[j] = v
        for j in range(QPOS):
            var w = Float64(ds.qpos_raw[r * QPOS + j])
            if w < q_lo[j]:
                q_lo[j] = w
            if w > q_hi[j]:
                q_hi[j] = w
    for j in range(ADIM):
        var pad = ACTION_BOX_MARGIN * (a_hi[j] - a_lo[j])
        a_lo[j] -= pad
        a_hi[j] += pad

    # ── the policy ────────────────────────────────────────────────────────
    print("checkpoint  " + ckpt)
    var tr = T.make()
    tr.load(ckpt)
    print(
        "            K=" + String(K) + " dim=" + String(DIM) + " enc="
        + String(N_ENC) + " dec=" + String(N_DEC) + "  ensemble m="
        + String(ACT_TEMPORAL_ENSEMBLE_M)
    )

    var qpos_n = List[Scalar[DT]](length=BATCH * QPOS, fill=Scalar[DT](0.0))
    var images_n = List[Scalar[DT]](
        length=BATCH * IMG_ELEMS, fill=Scalar[DT](0.0)
    )
    var dummy_actions = List[Scalar[DT]](
        length=BATCH * K * ADIM, fill=Scalar[DT](0.0)
    )
    var dummy_valid = List[Scalar[DT]](
        length=BATCH * K, fill=Scalar[DT](1.0)
    )
    var chunk = List[Scalar[DT]](length=BATCH * K * ADIM, fill=Scalar[DT](0.0))
    var pred_n = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0.0))
    var pred = List[Scalar[DT]](length=ADIM, fill=Scalar[DT](0.0))

    # ── check 1: does THIS checkpoint go with THIS store? ─────────────────
    if check_steps > 0 and snap == "":
        print("")
        print(
            "── replaying " + String(check_steps) + " steps of held-out"
            " episode " + String(ds.val_eps[0]) + " ──"
        )
        var ep = ds.val_eps[0]
        var ep_len = ds.store.episodes.length_of(ep)
        var n = check_steps if check_steps < ep_len else ep_len
        var g0 = ds.store.episodes.start_of(ep)
        var te0 = TemporalEnsemble[ADIM, K](m=ACT_TEMPORAL_ENSEMBLE_M)
        var s_act = 0.0
        var s_hold = 0.0
        var s_mean = 0.0
        var t_q0 = perf_counter_ns()
        for t in range(n):
            ds.fill_at[K](
                0, ep, t, qpos_n, images_n, dummy_actions, dummy_valid
            )
            tr.predict(qpos_n, images_n, dummy_actions, dummy_valid, chunk)
            te0.push(t, chunk, 0)
            te0.action_at(t, pred_n)
            denormalize(
                pred_n, 0, ds.action_mean, ds.action_std, pred, 0, ADIM
            )
            for j in range(ADIM):
                var truth = Float64(ds.action_raw[(g0 + t) * ADIM + j])
                s_act += abs(Float64(pred[j]) - truth)
                s_hold += abs(
                    Float64(ds.qpos_raw[(g0 + t) * QPOS + j]) - truth
                )
                s_mean += abs(Float64(ds.action_mean[j]) - truth)
        var ms_per_query = Float64(perf_counter_ns() - t_q0) / 1e6 / Float64(n)
        var d = Float64(n * ADIM)
        print(
            "   mean |error|   ACT " + fixed(s_act / d, 3)
            + "   hold " + fixed(s_hold / d, 3)
            + "   mean " + fixed(s_mean / d, 3)
            + "   (lerobot units)"
        )
        print("   " + fixed(ms_per_query, 1) + " ms per query")
        # ⚠ THE BAR IS THE CONSTANT BASELINE, NOT `hold`. Failing to beat
        # `hold` is a statement about the POLICY (see the open-loop
        # evaluator's note) and plenty of honest early policies do. Failing to
        # beat `mean` — predicting the dataset average — means the model and
        # these statistics do not belong together at all, which is the
        # wrong-store failure this check exists to catch.
        if s_act >= s_mean and not force:
            raise Error(
                "act deploy: the policy does not beat the constant `mean`"
                " baseline on its own held-out data. That is what deploying a"
                " checkpoint against the WRONG STORE looks like — the"
                " normalization statistics would be someone else's. Check"
                " --store, or pass --force if you know better."
            )
        if s_act >= s_hold:
            print(
                "   ⚠ ACT does not beat `hold` (predicting the current pose)."
                " The policy is weak, not mismatched — arming is allowed, but"
                " expect little."
            )

    # ── the cameras ───────────────────────────────────────────────────────
    print("")
    var names = camera_names(store)
    var cams = List[CameraReader]()
    for i in range(N_CAM):
        var label = names[i] if i < len(names) else String("slot ") + String(i)
        print(
            "camera slot " + String(i) + " = " + pad_right(label, 26)
            + " <- device " + String(devices[i])
        )
        # ⚠ rgb=FALSE, UNLIKE `record.mojo`. `camera_frame_to_chw_rgb` does
        # the BGR->RGB swap during its HWC->CHW transpose, a pass that already
        # touches every byte, so asking the camera thread to swap as well
        # would swap twice and feed the policy inverted colour channels — a
        # failure that looks like a bad policy, not like a bug. The recorder
        # wants rgb=True because its consumer is an encoder, not this.
        var c = CameraReader(
            devices[i], cam_w, cam_h, Float64(SO101_FPS), rgb=False
        )
        # ⚠ 8 s, NOT THE 4 s DEFAULT. Measured on this rig: a camera that has
        # been idle takes longer than 4 s to report ready on its first open,
        # and `CameraReader.start` reports that as "device 0 did not report
        # ready" — which reads exactly like a camera that is not there.
        c.start(wait_ms=8000)
        cams.append(c^)
    print(
        "            " + String(cam_w) + "x" + String(cam_h) + " native ->"
        " " + String(IMG_W) + "x" + String(IMG_H) + " (PIL BILINEAR, the"
        " filter the store was built with)"
    )
    if cam_w != CAM_W or cam_h != CAM_H:
        print(
            "   ⚠⚠ NOT the " + String(CAM_W) + "x" + String(CAM_H)
            + " the demonstrations were recorded at. The resize filter"
            " depends on the reduction factor, so these are different pixels"
            " from the same scene."
        )

    # ── --snap: is physical camera i the camera that filled slot i? ───────
    #
    # ⚠⚠ NOTHING ELSE CAN CATCH A SWAP. Two cameras of the same resolution
    # produce a perfectly well-formed observation either way round; the model
    # simply sees a world it has never been shown, and the failure presents as
    # "the policy does not work" rather than as an error. Device indices are
    # not stable either — they are whatever order the USB stack enumerated in,
    # and this machine currently has THREE cameras attached where the
    # recording had two.
    #
    # So: write what the STORE holds in each slot beside what each device is
    # showing right now, at the same size and through the same resize, and let
    # the operator look. It is the only honest test.
    if snap != "":
        var snap_frames = List[UInt8](
            unsafe_uninit_length = cam_w * cam_h * 3
        )
        var snap_chw = List[UInt8](length=CAM_ELEMS, fill=0)
        var row = List[Scalar[DType.uint8]]()
        ds.image_row_u8(ds.store.episodes.start_of(ds.val_eps[0]), row)
        var hwc = List[UInt8](length=IMG_W * IMG_H * 3, fill=0)
        for i in range(N_CAM):
            var label = (
                names[i] if i < len(names) else String("slot") + String(i)
            )
            # the store's slot i, CHW -> HWC
            for p in range(IMG_W * IMG_H):
                hwc[p * 3] = UInt8(row[i * CAM_ELEMS + p])
                hwc[p * 3 + 1] = UInt8(
                    row[i * CAM_ELEMS + IMG_W * IMG_H + p]
                )
                hwc[p * 3 + 2] = UInt8(
                    row[i * CAM_ELEMS + 2 * IMG_W * IMG_H + p]
                )
            var p_store = snap + "/slot" + String(i) + "_store.png"
            save_png(p_store, hwc, IMG_W, IMG_H, 3)

            # device i, right now, through the SAME resize the policy sees
            if not cams[i].take_blocking(snap_frames):
                raise Error(
                    "act deploy: camera " + String(devices[i])
                    + " delivered no frame for --snap"
                )
            camera_frame_to_chw_rgb(
                snap_frames, cam_w, cam_h, snap_chw, IMG_W, IMG_H
            )
            for p in range(IMG_W * IMG_H):
                hwc[p * 3] = snap_chw[p]
                hwc[p * 3 + 1] = snap_chw[IMG_W * IMG_H + p]
                hwc[p * 3 + 2] = snap_chw[2 * IMG_W * IMG_H + p]
            var p_live = snap + "/slot" + String(i) + "_live.png"
            save_png(p_live, hwc, IMG_W, IMG_H, 3)
            print(
                "  " + pad_right(label, 26) + " store -> " + p_store
                + "   live -> " + p_live
            )
        for i in range(N_CAM):
            try:
                cams[i].stop()
            except:
                pass
        print("")
        print(
            "compare each pair. If slot 0's store frame is the SIDE view and"
            " device 0 is looking\nfrom the front, swap --devices before"
            " arming anything."
        )
        return

    # ── the arm ───────────────────────────────────────────────────────────
    print("")
    print("follower    " + String(FOLLOWER_PORT))
    var follower = SO101Arm(String(FOLLOWER_PORT), max_step_ticks=step_ticks)
    follower.bus.timeout_ms = 20

    var raw = InlineArray[Int32, SO101_N](fill=0)
    if follower.read_positions(Span(raw)) != SO101_N:
        raise Error(
            "act deploy: the follower did not report " + String(SO101_N)
            + " positions — not arming"
        )

    # ── check 2: is the arm anywhere the demonstrations went? ─────────────
    print("")
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
    else:
        print(
            "press Enter to start (q = quit). stdin is not a terminal, so the"
            " run ends on its\n  --seconds and Enter-to-stop is off."
        )
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

    # ⚠ THE POSE TO COME BACK TO, captured BEFORE anything is energised. The
    # arm was resting here under gravity alone, which is what makes it the one
    # pose we know is safe to release torque at. `raw` is overwritten every
    # tick of the loop below, so it has to be copied now.
    var start_pose = List[Int32](length=SO101_N, fill=0)
    for i in range(SO101_N):
        start_pose[i] = raw[i]

    if arm_it:
        # Guard 4: park the goal on the present pose BEFORE torque, so arming
        # holds instead of snapping to a stale `Goal_Position`.
        follower.set_position_mode()
        var hold = follower.max_step_ticks
        follower.max_step_ticks = 0
        follower.write_goals(Span(raw))
        follower.max_step_ticks = hold
        follower.set_torque(True)
        print("follower torque ON\n")
    else:
        print("dry run — nothing energised\n")

    var frames = List[List[UInt8]]()
    var chw = List[List[UInt8]]()
    for i in range(N_CAM):
        frames.append(
            List[UInt8](unsafe_uninit_length = cams[i].frame_bytes())
        )
        chw.append(List[UInt8](length=CAM_ELEMS, fill=0))
    var goals = InlineArray[Int32, SO101_N](fill=0)
    var cmd = List[Float64](length=ADIM, fill=0.0)
    for i in range(ADIM):
        cmd[i] = follower.cal.degrees(i, raw[i])

    var te = TemporalEnsemble[ADIM, K](m=ACT_TEMPORAL_ENSEMBLE_M)

    var queries = 0
    var bus_skipped = 0
    var refused = 0
    var clamped = 0
    var stale_ticks = 0
    var reversals = 0.0
    var n_step = 0.0
    var sum_step = 0.0
    var max_step_seen = 0.0
    var sum_contrib = 0
    var worst_cam = 0.0
    var worst_pre = 0.0
    var worst_fwd = 0.0
    var worst_bus = 0.0
    var sum_cam = 0.0
    var sum_pre = 0.0
    var sum_fwd = 0.0
    var sum_bus = 0.0
    var last_delta = List[Float64](length=SO101_N, fill=0.0)
    var last_goal = List[Float64](length=SO101_N, fill=0.0)
    for i in range(SO101_N):
        last_goal[i] = Float64(raw[i])
    var last_t_cmd = -1

    var loop_t0 = perf_counter_ns()
    var deadline = loop_t0 + seconds * 1_000_000_000
    try:
        while perf_counter_ns() < deadline:
            # ⚠ ONLY ON A TERMINAL. At the end of a pipe `poll` reports
            # POLLHUP as readiness and `line()` hands back the same empty
            # string an Enter does, so a scripted `printf '\\n' | act_deploy`
            # would stop on its very first check. See `stdin_is_tty`.
            if interactive and stdin.has_input():
                _ = stdin.line()
                print("  stopped by the operator")
                break

            # ── observe ──────────────────────────────────────────────────
            var t_c0 = perf_counter_ns()
            for i in range(N_CAM):
                # ⚠ THE NEWEST FRAME, NOT THE OLDEST — see `take_latest`. A
                # forward takes ~3 frame periods, so the ring always holds a
                # queue and taking its head would add 100 ms of staleness to
                # every observation. `take_blocking` only ever runs on the
                # first tick, before the cameras have queued anything.
                if cams[i].take_latest(frames[i]) == 0:
                    stale_ticks += 1
                    if not cams[i].take_blocking(frames[i]):
                        raise Error(
                            "act deploy: camera " + String(devices[i])
                            + " stopped delivering frames"
                        )
            var t_c1 = perf_counter_ns()

            # The grid step this observation BELONGS to, on the 30 Hz clock
            # the demonstrations were recorded on.
            var t_obs = Int(
                Float64(t_c1 - loop_t0) * Float64(SO101_FPS) / 1e9
            )
            # ⚠ STRICTLY INCREASING, and past the last step already COMMANDED.
            # The ensemble's ring is indexed by query step (`slot = t % K`), so
            # a query landing on a step that has already been acted on would
            # be a chunk whose first entries describe the past. Forcing it
            # forward costs at most one grid step of accuracy in `t_obs` and
            # keeps every chunk in the ring speaking about the future.
            if t_obs <= last_t_cmd:
                t_obs = last_t_cmd + 1

            var got = follower.read_positions(Span(raw))
            if got != SO101_N:
                # Guard 5: a partial read is not an observation. Skipping the
                # whole tick holds the last goal, which is safe; a
                # half-updated pose fed to the policy is not.
                bus_skipped += 1
                continue
            for j in range(QPOS):
                qpos_n[j] = (
                    Scalar[DT](follower.cal.degrees(j, raw[j]))
                    - ds.qpos_mean[j]
                ) / ds.qpos_std[j]

            for i in range(N_CAM):
                camera_frame_to_chw_rgb(
                    frames[i], cam_w, cam_h, chw[i], IMG_W, IMG_H
                )
                # ⚠ THE SHARED NORMALIZER, the same call `ACTDataset._fill_one`
                # makes. See its note: this is the one step that must agree
                # exactly between training and deployment.
                normalize_camera_chw[IMG_H, IMG_W](
                    chw[i], 0, images_n, i * CAM_ELEMS
                )
            var t_pre = perf_counter_ns()

            # ── think ────────────────────────────────────────────────────
            tr.predict(qpos_n, images_n, dummy_actions, dummy_valid, chunk)
            te.push(t_obs, chunk, 0)
            queries += 1
            var t_fwd = perf_counter_ns()

            # ── act, at the grid step it is NOW ──────────────────────────
            # ⚠⚠ NOT `action_at(t_obs)`. The forward took ~3 grid steps; the
            # arm is being commanded now, so the action it gets must be the
            # one the trajectory calls for NOW. Commanding `t_obs` would run
            # the whole demonstration permanently one latency behind itself.
            var t_cmd = Int(
                Float64(t_fwd - loop_t0) * Float64(SO101_FPS) / 1e9
            )
            if t_cmd < t_obs:
                t_cmd = t_obs
            if t_cmd > t_obs + K - 1:
                # A stall longer than a whole chunk. The freshest query cannot
                # speak about this step, so ask for the last one it can.
                t_cmd = t_obs + K - 1
            last_t_cmd = t_cmd
            sum_contrib += te.n_contributors(t_cmd)
            te.action_at(t_cmd, pred_n)
            denormalize(
                pred_n, 0, ds.action_mean, ds.action_std, pred, 0, ADIM
            )

            for j in range(ADIM):
                var v = Float64(pred[j])
                # Guard 6: the dataset's own action box.
                if v < a_lo[j]:
                    v = a_lo[j]
                    clamped += 1
                elif v > a_hi[j]:
                    v = a_hi[j]
                    clamped += 1
                # ⚠ OFF BY DEFAULT (`--smooth 1.0`). Temporal ensembling IS
                # the smoother here — every command already blends ~20
                # overlapping chunks — and `deploy_reach_real.mojo`'s EMA
                # exists because a SAC policy trained without an action-rate
                # penalty chattered. Reach for it only if `reversals` in the
                # report says the command is buzzing.
                cmd[j] = (1.0 - smooth) * cmd[j] + smooth * v
                goals[j] = follower.cal.raw_from_degrees(j, cmd[j])

                var delta = Float64(goals[j]) - last_goal[j]
                if delta * last_delta[j] < 0.0:
                    reversals += 1.0
                last_delta[j] = delta
                last_goal[j] = Float64(goals[j])
                var stp = Float64(goals[j] - raw[j])
                if stp < 0.0:
                    stp = -stp
                if stp > max_step_seen:
                    max_step_seen = stp
                sum_step += stp
                n_step += 1.0

            if arm_it:
                try:
                    follower.write_goals(Span(goals))
                except:
                    refused += 1
            var t_bus = perf_counter_ns()

            var d_cam = Float64(t_c1 - t_c0) / 1e6
            var d_pre = Float64(t_pre - t_c1) / 1e6
            var d_fwd = Float64(t_fwd - t_pre) / 1e6
            var d_bus = Float64(t_bus - t_fwd) / 1e6
            if d_cam > worst_cam:
                worst_cam = d_cam
            if d_pre > worst_pre:
                worst_pre = d_pre
            if d_fwd > worst_fwd:
                worst_fwd = d_fwd
            if d_bus > worst_bus:
                worst_bus = d_bus
            sum_cam += d_cam
            sum_pre += d_pre
            sum_fwd += d_fwd
            sum_bus += d_bus

            if queries % 10 == 0:
                var line = String("  t=") + pad_left(
                    fixed(Float64(t_cmd) / Float64(SO101_FPS), 1), 5
                ) + "s  x" + pad_left(
                    String(te.n_contributors(t_cmd)), 3
                ) + " "
                for j in range(ADIM):
                    line += " " + col(cmd[j], 7, 1)
                print(line)
    finally:
        # ⚠ THE SAME SHUTDOWN ON EVERY PATH, including an exception. A camera
        # that stops delivering mid-run leaves the arm extended, and the old
        # code's unconditional `set_torque(False)` would drop it there — the
        # error path is precisely when a fall is least expected. `--no-return`
        # is the only way to get the old behaviour.
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

    var elapsed = Float64(perf_counter_ns() - loop_t0) / 1e9
    print("=" * 74)
    print("ACT closed-loop run")
    print("  queries           = " + String(queries) + " in "
          + fixed(elapsed, 1) + " s = "
          + fixed(Float64(queries) / elapsed, 1) + " Hz")
    print(
        "  action grid       = " + String(SO101_FPS) + " Hz, so the policy"
        " got a waypoint every "
        + (fixed(Float64(SO101_FPS) * elapsed / Float64(queries), 1)
           if queries > 0 else String("n/a"))
        + " demonstrated steps"
    )
    print(
        "  ensemble          = "
        + (fixed(Float64(sum_contrib) / Float64(queries), 1)
           if queries > 0 else String("n/a"))
        + " chunks per command (K=" + String(K) + " is the ceiling)"
    )
    print("  bus-skipped ticks = " + String(bus_skipped))
    print("  write refused     = " + String(refused))
    print("  camera starved    = " + String(stale_ticks))
    # ⚠ THE OFF-DISTRIBUTION SIGNAL. Every one of these is the policy asking
    # for a pose no demonstration ever reached.
    print(
        "  action clamped    = " + String(clamped) + " of "
        + String(Int(n_step)) + " joint-commands"
        + ("   ⚠ the policy is asking for poses the demonstrations never"
           " reached" if clamped * 20 > Int(n_step) else "")
    )
    # ⚠ IN A DRY RUN THIS IS NOT A PER-TICK DEMAND. It is |commanded −
    # present|, and in a dry run the arm never moves, so it settles at the
    # STANDING DISTANCE between where the arm is parked and where the policy
    # wants it. Only an armed run makes it the quantity `max_step_ticks`
    # bounds.
    print(
        "  commanded step    = mean "
        + (fixed(sum_step / n_step, 1) if n_step > 0 else String("n/a"))
        + " ticks, max " + fixed(max_step_seen, 1)
        + " (clamp is " + String(step_ticks) + ")"
        + ("   <- standing distance, the arm never moved"
           if not arm_it else "")
    )
    print(
        "  goal reversals    = "
        + (fixed(100.0 * reversals / n_step, 0) if n_step > 0
           else String("n/a"))
        + " % of writes changed DIRECTION  <- this is the shake"
    )
    var throttle = (sum_step / n_step) / Float64(step_ticks) if (
        n_step > 0 and step_ticks > 0
    ) else 0.0
    if throttle > 2.0 and arm_it:
        print(
            "  ⚠ RATE-LIMITED: the policy asked for " + fixed(throttle, 1)
            + "x the clamp on average.\n     The CLAMP shaped this run, not"
            " the policy — raise --step before reading anything into the"
            " motion."
        )
    # ⚠ MEAN BESIDE WORST, because they say different things and only the
    # mean sets the query rate. A worst-case `forward` of 250 ms against a
    # mean of 100 ms is one scheduling hiccup, not the cost of the model — and
    # optimising the wrong one of those is a wasted afternoon.
    var qd = Float64(queries) if queries > 0 else 1.0
    print("  per query, mean (worst) in ms")
    print(
        "     cameras    " + pad_left(fixed(sum_cam / qd, 1), 6)
        + " (" + fixed(worst_cam, 1) + ")"
        + "      preprocess " + pad_left(fixed(sum_pre / qd, 1), 6)
        + " (" + fixed(worst_pre, 1) + ")"
    )
    print(
        "     forward    " + pad_left(fixed(sum_fwd / qd, 1), 6)
        + " (" + fixed(worst_fwd, 1) + ")"
        + "      bus        " + pad_left(fixed(sum_bus / qd, 1), 6)
        + " (" + fixed(worst_bus, 1) + ")"
    )
    if not arm_it:
        print("  ⚠ DRY RUN — nothing was written to the arm. Add --arm.")
    print("=" * 74)
