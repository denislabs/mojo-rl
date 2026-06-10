"""LeWM2 closed-loop MPC control on the real PushT simulator.

The actual "solve PushT": plan in the world model's latent space and execute
on `BATCH` parallel mojo `PushTEnv`s, receding-horizon. Each control cycle:

  1. render each env's current frame at the WM resolution → frozen encode →
     start latent (one per env).
  2. (once) encode a goal image — block at the goal pose → goal latent.
  3. ContinuousCEM optimizes an action plan minimizing predicted-latent-to-
     goal MSE via the shared LeWM2MPCScorer (latent rollout).
  4. DENORMALIZE + execute the first planned action block on each env:
     actions are per-step DELTAS — `env_target = agent_pos + action · SCALE`
     with SCALE = 100 exactly (ground truth from the stable_worldmodel
     PushT-v1 source: `relative=True`, `action_scale=100`; the earlier
     centroid-regression calibration of ~142/148 was ~1.45× too large) —
     so each of the `frameskip` sub-actions is one env.step target.
  5. record coverage; repeat.

Returns (success_rate, mean_coverage) over the envs. Optionally writes a
horizontal strip PPM of env-0's trajectory (one frame per cycle).

GPU-oriented (the WM is a 224² gpu model); `ctx` is required.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ...nn2.constants import DT
from ...nn2.core.module import Module
from mojo_rl.planners.trajectory import ContinuousCEMOptimizer
from mojo_rl.envs.pusht import PushTEnv, PushTAction
from mojo_rl.envs.pusht.constants import PConstants
from mojo_rl.envs.pusht.render import render_pusht_rgb_at, IMG_C
from mojo_rl.render.image_writer import save_image_row
from .trainer import LeWMTrainer
from .encoder import LeWMEncoder
from .predict_graph import LeWMPredictor
from .mpc import LeWM2MPCScorer
from .pusht_sim_bridge import sim_frame_chw_norm


def run_lewm2_closedloop[
    IN_CH: Int, IMG: Int, PATCH: Int, HIDDEN: Int, ENC_HEADS: Int,
    ENC_LAYERS: Int, EMB: Int, ENC_PROJ_H: Int, ENC_FF_MULT: Int,
    T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, N_PREDS: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
    PRED_PROJ_H: Int, SIG_PROJ: Int, SIG_KNOTS: Int,
    BATCH: Int, MPC_HORIZON: Int, target: StaticString,
    PRED_DIM_HEAD: Int = 0, ACT_DIM: Int = 2, VIZ: Int = 96,
    # Encoder type — trailing, dims-derived default = mean-pooled LeWMEncoder.
    # Pass LeWMEncoderCLS[...same dims...] for the CLS variant; the encode step
    # runs through `wm` (eval_loss + read_node_into["emb"]), so the CLS encoder
    # is used automatically. Existing callers omit it and stay unchanged.
    ENC: Module = LeWMEncoder[
        IN_CH, IMG, PATCH, (IMG // PATCH) * (IMG // PATCH),
        HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H, ENC_FF_MULT,
    ],
](
    mut wm: LeWMTrainer[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, BATCH, target,
        PRED_DIM_HEAD, ENC,
    ],
    n_cycles: Int,
    scale_x: Float64 = 100.0,
    scale_y: Float64 = 100.0,
    # Action z-score stats (the recipe WM trains on z-scored actions; the
    # planner samples in that space and execution de-normalizes:
    # raw = z·std + mean, then env_target = agent + raw·scale). Defaults
    # 0/1 = identity for raw-action WMs.
    act_mean_x: Float64 = 0.0,
    act_mean_y: Float64 = 0.0,
    act_std_x: Float64 = 1.0,
    act_std_y: Float64 = 1.0,
    cem_iters: Int = 10,
    cem_samples: Int = 200,
    cem_topk: Int = 20,
    init_std: Float64 = 0.2,
    goal_agent_x: Float64 = 256.0,
    goal_agent_y: Float64 = 256.0,
    goal_match_agent: Bool = True,
    seed0: Int = 1,
    viz_path: String = "",
    ctx: Optional[DeviceContext] = None,
    verbose: Bool = True,
) raises -> Tuple[Float64, Float64]:
    if not ctx:
        raise Error("run_lewm2_closedloop: ctx (GPU) required")
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime PIX = T * IMG_DIM
    comptime ACTIN = T * ACT
    comptime TE = T * EMB
    comptime BE = BATCH * EMB
    comptime NEEDED = H + MPC_HORIZON - 1
    comptime FRAMESKIP = ACT // ACT_DIM
    comptime VIZN = IN_CH * VIZ * VIZ
    comptime Scorer = LeWM2MPCScorer[
        EMB, T, ACT, SMOOTHED, AE_MLP, H, PRED_HEADS, PRED_FF, DEPTH,
        PRED_PROJ_H, BATCH, MPC_HORIZON, target, PRED_DIM_HEAD,
    ]
    comptime Predictor = LeWMPredictor[
        EMB, T, ACT, SMOOTHED, AE_MLP, H, PRED_HEADS, PRED_FF, DEPTH,
        PRED_PROJ_H, BATCH, target, PRED_DIM_HEAD,
    ]
    var ctx_v = ctx.value()

    # predictor (name-synced, incl. BN running stats) + rollout scorer.
    # Planning runs in EVAL mode (running stats), matching the reference's
    # `model.eval()`: training-mode BN encodes start/goal under different
    # batch statistics and couples CEM candidate scores. The caller must
    # warm the wm's running stats (a few hundred training-mode forwards
    # over dataset windows) BEFORE calling — checkpoints don't carry them.
    var pred_net = Predictor.make(ctx=ctx)
    pred_net.sync_from_named(wm.export_named_params())
    pred_net.set_bn_training(False)
    wm.set_bn_training(False)
    var scorer = Scorer(pred_net^, ctx=ctx)

    # encoding IO
    var pix_host = alloc[Scalar[DT]](BATCH * PIX)
    var emb_host = alloc[Scalar[DT]](BATCH * TE)
    var start_lat = alloc[Scalar[DT]](BE)
    var goal_lat = alloc[Scalar[DT]](BE)
    var pix_dev = ctx_v.enqueue_create_buffer[DT](BATCH * PIX)
    var act_dev = ctx_v.enqueue_create_buffer[DT](BATCH * ACTIN)
    act_dev.enqueue_fill(0.0)   # emb depends only on pixels
    ctx_v.synchronize()
    var pix_d_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        pix_dev.unsafe_ptr()
    )
    var act_d_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        act_dev.unsafe_ptr()
    )

    # Goal latent (block AT the goal pose) is recomputed per cycle inside the
    # loop: with `goal_match_agent` the goal image uses each env's CURRENT
    # agent position, so start vs goal differ ONLY in block pose — the planner
    # then optimizes block pose, not "drift the agent to the goal's agent
    # spot" (which a fixed-agent goal lets it cheat, freezing the block).

    # envs
    var envs = List[PushTEnv[DT]]()
    for b in range(BATCH):
        var e = PushTEnv[DT](seed=UInt64(seed0 + b))
        _ = e.reset()
        envs.append(e^)

    var cem = ContinuousCEMOptimizer[BATCH, ACT](
        horizon=NEEDED, cem_iters=cem_iters, cem_samples=cem_samples,
        cem_topk=cem_topk, init_std=init_std,
    )
    var plan = alloc[Scalar[DT]](BATCH * NEEDED * ACT)

    var do_viz = viz_path.byte_length() > 0
    var viz_buf = alloc[Scalar[DT]](n_cycles * VIZN if do_viz else 1)
    var viz_tmp = alloc[Scalar[DT]](VIZN if do_viz else 1)

    # ── control loop ────────────────────────────────────────────────────
    for cyc in range(n_cycles):
        # render each env's current frame → window (T copies)
        for b in range(BATCH):
            var bp = envs[b].block_pose()
            var ap = envs[b].agent_pos()
            sim_frame_chw_norm[IMG](
                bp[0], bp[1], bp[2], ap[0], ap[1],
                pix_host + (b * T) * IMG_DIM,
            )
            for t in range(1, T):
                for i in range(IMG_DIM):
                    pix_host[(b * T + t) * IMG_DIM + i] = pix_host[
                        (b * T) * IMG_DIM + i
                    ]
        ctx_v.enqueue_copy(pix_dev, pix_host)
        ctx_v.synchronize()
        var pix_t = TileTensor(pix_d_p, row_major[BATCH, PIX]())
        var act_t = TileTensor(act_d_p, row_major[BATCH, ACTIN]())
        _ = wm.eval_loss(pix_t, act_t)
        wm.read_node_into["emb"](emb_host, BATCH * TE)
        for b in range(BATCH):
            for d in range(EMB):
                start_lat[b * EMB + d] = emb_host[b * TE + d]

        # goal window: block @ goal pose + (current agent | fixed) per env,
        # so the start↔goal latent diff is block-pose only.
        for b in range(BATCH):
            var ap = envs[b].agent_pos()
            var gx = Float64(ap[0]) if goal_match_agent else goal_agent_x
            var gy = Float64(ap[1]) if goal_match_agent else goal_agent_y
            sim_frame_chw_norm[IMG](
                Scalar[DT](PConstants.GOAL_X), Scalar[DT](PConstants.GOAL_Y),
                Scalar[DT](PConstants.GOAL_ANGLE),
                Scalar[DT](gx), Scalar[DT](gy),
                pix_host + (b * T) * IMG_DIM,
            )
            for t in range(1, T):
                for i in range(IMG_DIM):
                    pix_host[(b * T + t) * IMG_DIM + i] = pix_host[
                        (b * T) * IMG_DIM + i
                    ]
        ctx_v.enqueue_copy(pix_dev, pix_host)
        ctx_v.synchronize()
        var gpix_t = TileTensor(pix_d_p, row_major[BATCH, PIX]())
        var gact_t = TileTensor(act_d_p, row_major[BATCH, ACTIN]())
        _ = wm.eval_loss(gpix_t, gact_t)
        wm.read_node_into["emb"](emb_host, BATCH * TE)
        for b in range(BATCH):
            for d in range(EMB):
                goal_lat[b * EMB + d] = emb_host[b * TE + d]
        scorer.set_start_goal(start_lat, goal_lat)

        _ = cem.optimize(scorer, plan, verbose=False)

        # Execute the first FUTURE planned block. The scorer's context is
        # the start latent replicated H times, so plan blocks 0..H-2 pair
        # with the frozen context ("imagined past") — the first action
        # whose effect materializes in a NEW imagined latent is block H-1
        # (in training, action[H-1] is the block that takes frame H-1 to
        # frame H). Executing block 0 (pre-fix) executed an imagined PAST
        # action.
        for k in range(FRAMESKIP):
            for b in range(BATCH):
                if envs[b].is_done():
                    continue
                var ap = envs[b].agent_pos()
                var dx = Float64(
                    plan[(b * NEEDED + (H - 1)) * ACT + k * ACT_DIM + 0]
                ) * act_std_x + act_mean_x
                var dy = Float64(
                    plan[(b * NEEDED + (H - 1)) * ACT + k * ACT_DIM + 1]
                ) * act_std_y + act_mean_y
                var tx = Scalar[DT](Float64(ap[0]) + dx * scale_x)
                var ty = Scalar[DT](Float64(ap[1]) + dy * scale_y)
                _ = envs[b].step(PushTAction[DT](tx, ty))

        var mc: Float64 = 0.0
        var ns = 0
        for b in range(BATCH):
            var cov = Float64(envs[b].coverage())
            mc += cov
            if cov > Float64(PConstants.SUCCESS_THRESHOLD):
                ns += 1
        mc /= Float64(BATCH)
        if verbose:
            print("   cycle", cyc, "/", n_cycles, " mean_cov=", mc,
                  " success=", ns, "/", BATCH)

        if do_viz:
            var bp0 = envs[0].block_pose()
            var ap0 = envs[0].agent_pos()
            var vt = LayoutTensor[
                DT, Layout.row_major(VIZ, VIZ, IMG_C), MutAnyOrigin
            ](viz_tmp)
            render_pusht_rgb_at[VIZ](
                bp0[0], bp0[1], bp0[2], ap0[0], ap0[1], vt
            )
            for c in range(IN_CH):
                for y in range(VIZ):
                    for x in range(VIZ):
                        viz_buf[cyc * VIZN + c * VIZ * VIZ + y * VIZ + x] = (
                            viz_tmp[(y * VIZ + x) * IN_CH + c]
                        )

    # final metrics
    var mc: Float64 = 0.0
    var ns = 0
    for b in range(BATCH):
        var cov = Float64(envs[b].coverage())
        mc += cov
        if cov > Float64(PConstants.SUCCESS_THRESHOLD):
            ns += 1
    mc /= Float64(BATCH)
    var success_rate = Float64(ns) / Float64(BATCH)

    if do_viz:
        save_image_row(
            viz_path, viz_buf, n=n_cycles, height=VIZ, width=VIZ,
            channels=IN_CH, vmin=0.0, vmax=255.0,
        )

    pix_host.free(); emb_host.free(); start_lat.free(); goal_lat.free()
    plan.free(); viz_buf.free(); viz_tmp.free()
    _ = scorer^
    return (success_rate, mc)
