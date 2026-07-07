"""LeWM PAPER-PROTOCOL planning eval on the real PushT simulator.

The LeWM paper's actual control benchmark (App F.1 + swm eval pipeline) —
NOT the full-task "push to the canonical goal from random init" solve:

  * the episode STARTS from a DATASET state (on-distribution),
  * the GOAL is the real state 25 env-steps later in the SAME expert
    trajectory (reachable; the goal's agent position sits on the contact
    path, so matching it pulls the agent toward the block),
  * eval budget 50 env steps,
  * success (swm `eval_state`): ‖[agent,block]₄ − goal₄‖₂ < 20 px AND
    block-angle diff < π/9 — NOT coverage > 0.95.

This is the only protocol whose success rate is comparable to the paper's
~90% on PushT. Caller supplies `(start, goal)` state pairs — the NVIDIA
example reads them from the HF dataset window (frame 0 / frame +25); the
Apple toy test fabricates them by rolling the env.

Mechanics shared with `closedloop.mojo`: frozen encode via `wm`
(eval-mode BN, running stats synced into the rollout predictor),
ContinuousCEM over the latent rollout scorer, delta execution
`env_target = agent + action·100`. Differences:

  * `env.set_state(start)` instead of random reset,
  * goal latent encoded ONCE from the goal STATE rendered at WM resolution
    (same renderer domain as the current-frame encodes; the real
    dataset-pixels goal frame is an A/B left to the example),
  * per plan, ALL `MPC_HORIZON` future blocks execute before replanning
    (paper: receding_horizon = horizon — the entire sequence runs), and
    the executed blocks start at plan index H-1: blocks 0..H-2 pair with
    the frozen replicated context ("imagined past"); block H-1 is the
    first action whose effect lands in a NEW imagined latent. CEM still
    co-optimizes the history blocks (they condition the predictor) — a
    known protocol gap vs the reference's variable-length context.

AdaJEPA test-time adaptation (docs/ADAJEPA_LEWM_TTA_PLAN.md, off by
default): with `tta_enabled`, every fully-executed action block pushes a
(frame, z-space block) pair into a rolling per-env window buffer (frames
rendered per block, so multi-block plans still yield training-spaced
windows), and after each plan-execution — once T pairs have accumulated —
the wm takes `tta_steps` masked gradient steps on the pretraining JEPA
loss (BN training mode, subset `tta_keep`, default predictor-side), the
scorer's predictor is re-synced, and the GOAL latent is RE-ENCODED (the
adapt steps move the BN running stats; start and goal latents must share
them). Params + BN state are snapshot/restored at exit. NOTE: AdaJEPA
replans after every chunk — run with MPC_HORIZON=1 for the faithful
plan-execute-adapt-replan loop. This is the benchmark AdaJEPA itself
evaluates PushT on (goals sampled 25 steps ahead).

Returns (success_rate, mean_final_pos_diff_px).
"""

from std.memory import alloc
from std.math import sqrt, pi
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.planners.trajectory import ContinuousCEMOptimizer
from mojo_rl.envs.pusht import PushTEnv, PushTAction
from mojo_rl.envs.pusht.render import render_pusht_rgb_at, IMG_C
from mojo_rl.render.image_writer import save_image_row
from .trainer import LeWMTrainer
from .encoder import LeWMEncoder
from .predict_graph import LeWMPredictor
from .mpc import LeWMMPCScorer
from .pusht_sim_bridge import sim_frame_chw_norm
from .tta_buffer import TTAWindowBuffer


comptime _SUCCESS_POS_PX: Float64 = 20.0  # swm eval_state: ‖Δ[a,b]₄‖ < 20
comptime _SUCCESS_ANG_RAD: Float64 = pi / 9.0  # block angle within 20°


def _state_dist(
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    bang: Float64,
    g: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [ax,ay,bx,by,bang]
) -> Tuple[Float64, Float64]:
    """(4-vec positional distance, wrapped |block-angle diff|) vs goal."""
    var d0 = ax - Float64(g[0])
    var d1 = ay - Float64(g[1])
    var d2 = bx - Float64(g[2])
    var d3 = by - Float64(g[3])
    var pos = sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
    var da = bang - Float64(g[4])
    if da < 0.0:
        da = -da
    while da > 2.0 * pi:
        da -= 2.0 * pi
    if da > pi:
        da = 2.0 * pi - da
    return (pos, da)


def run_lewm_paper_protocol[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    ENC_PROJ_H: Int,
    ENC_FF_MULT: Int,
    T: Int,
    ACT: Int,
    SMOOTHED: Int,
    AE_MLP: Int,
    H: Int,
    N_PREDS: Int,
    PRED_HEADS: Int,
    PRED_FF: Int,
    DEPTH: Int,
    PRED_PROJ_H: Int,
    SIG_PROJ: Int,
    SIG_KNOTS: Int,
    BATCH: Int,
    MPC_HORIZON: Int,
    target: StaticString,
    PRED_DIM_HEAD: Int = 0,
    ACT_DIM: Int = 2,
    VIZ: Int = 96,
    ENC: Module = LeWMEncoder[
        IN_CH,
        IMG,
        PATCH,
        (IMG // PATCH) * (IMG // PATCH),
        HIDDEN,
        ENC_HEADS,
        ENC_LAYERS,
        EMB,
        ENC_PROJ_H,
        ENC_FF_MULT,
    ],
](
    mut wm: LeWMTrainer[
        IN_CH,
        IMG,
        PATCH,
        HIDDEN,
        ENC_HEADS,
        ENC_LAYERS,
        EMB,
        ENC_PROJ_H,
        ENC_FF_MULT,
        T,
        ACT,
        SMOOTHED,
        AE_MLP,
        H,
        N_PREDS,
        PRED_HEADS,
        PRED_FF,
        DEPTH,
        PRED_PROJ_H,
        SIG_PROJ,
        SIG_KNOTS,
        BATCH,
        target,
        PRED_DIM_HEAD,
        ENC,
    ],
    start_states: UnsafePointer[Scalar[DT], MutAnyOrigin],  # (BATCH,5)
    goal_states: UnsafePointer[Scalar[DT], MutAnyOrigin],  # (BATCH,5)
    eval_budget: Int = 50,
    scale_x: Float64 = 100.0,
    scale_y: Float64 = 100.0,
    # Action z-score stats (recipe WM trains on z-scored actions; execution
    # de-normalizes raw = z·std + mean before ·scale). 0/1 = identity.
    act_mean_x: Float64 = 0.0,
    act_mean_y: Float64 = 0.0,
    act_std_x: Float64 = 1.0,
    act_std_y: Float64 = 1.0,
    cem_iters: Int = 30,
    cem_samples: Int = 300,
    cem_topk: Int = 30,
    init_std: Float64 = 0.2,
    seed0: Int = 1,
    viz_path: String = "",
    ctx: Optional[DeviceContext] = None,
    verbose: Bool = True,
    # How many of the planned future blocks to EXECUTE before replanning.
    # 0 (default) ⇒ all MPC_HORIZON blocks = the LeWM paper protocol
    # (receding_horizon = horizon). 1 with MPC_HORIZON > 1 = AdaJEPA's
    # receding-horizon shape: plan with lookahead, execute one chunk,
    # replan (their PushT: plan 25 steps / execute 5).
    execute_blocks: Int = 0,
    # AdaJEPA test-time adaptation (docs/ADAJEPA_LEWM_TTA_PLAN.md §5).
    # tta_keep = kept param-name prefixes; empty ⇒ the v1 predictor-side
    # default. Requires a fresh Adam (zero moments — reset_opt_moments
    # after a moments-carrying v3 load) + wd=0 on `wm`.
    tta_enabled: Bool = False,
    tta_steps: Int = 1,
    tta_keep: List[String] = List[String](),
) raises -> Tuple[Float64, Float64]:
    """State rows are [agent_x, agent_y, block_x, block_y, block_angle] in
    world coords [0,512] (the dataset `state` column order — the example
    cross-checks state[0:2] against the `proprio` agent position)."""
    if not ctx:
        raise Error("run_lewm_paper_protocol: ctx (GPU) required")
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime PIX = T * IMG_DIM
    comptime ACTIN = T * ACT
    comptime TE = T * EMB
    comptime BE = BATCH * EMB
    comptime NEEDED = H + MPC_HORIZON - 1
    comptime FRAMESKIP = ACT // ACT_DIM
    comptime VIZN = IN_CH * VIZ * VIZ
    comptime Scorer = LeWMMPCScorer[
        EMB,
        T,
        ACT,
        SMOOTHED,
        AE_MLP,
        H,
        PRED_HEADS,
        PRED_FF,
        DEPTH,
        PRED_PROJ_H,
        BATCH,
        MPC_HORIZON,
        target,
        PRED_DIM_HEAD,
    ]
    comptime Predictor = LeWMPredictor[
        EMB,
        T,
        ACT,
        SMOOTHED,
        AE_MLP,
        H,
        PRED_HEADS,
        PRED_FF,
        DEPTH,
        PRED_PROJ_H,
        BATCH,
        target,
        PRED_DIM_HEAD,
    ]
    var ctx_v = ctx.value()

    # predictor (name-synced incl. BN running stats) + eval-mode BN.
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
    act_dev.enqueue_fill(0.0)  # emb depends only on pixels
    ctx_v.synchronize()
    var pix_d_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        pix_dev.unsafe_ptr()
    )
    var act_d_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        act_dev.unsafe_ptr()
    )

    # envs at the dataset start states
    var envs = List[PushTEnv[DT]]()
    for b in range(BATCH):
        var e = PushTEnv[DT](seed=UInt64(seed0 + b))
        _ = e.reset()
        var s = start_states + b * 5
        _ = e.set_state(s[0], s[1], s[2], s[3], s[4])
        envs.append(e^)

    # goal latent — encoded ONCE from each episode's goal STATE (real
    # consistent agent+block configuration, rendered in the same sim
    # domain as the per-cycle current-frame encodes).
    for b in range(BATCH):
        var g = goal_states + b * 5
        sim_frame_chw_norm[IMG](
            g[2],
            g[3],
            g[4],
            g[0],
            g[1],
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

    var cem = ContinuousCEMOptimizer[BATCH, ACT](
        horizon=NEEDED,
        cem_iters=cem_iters,
        cem_samples=cem_samples,
        cem_topk=cem_topk,
        init_std=init_std,
    )
    var plan = alloc[Scalar[DT]](BATCH * NEEDED * ACT)

    # ── AdaJEPA TTA state (docs/ADAJEPA_LEWM_TTA_PLAN.md §5) ────────────
    var keep = tta_keep.copy()
    if tta_enabled and len(keep) == 0:
        keep = [  # v1 default: whole predictor side, encoder frozen
            String("pred_raw."),
            String("pred."),
            String("x_pe."),
            String("act_emb."),
        ]
    var tta_buf = TTAWindowBuffer[BATCH, T, IMG_DIM, ACT](enabled=tta_enabled)
    var tta_act_host = alloc[Scalar[DT]](BATCH * ACTIN if tta_enabled else 1)
    var tta_frame = alloc[Scalar[DT]](IMG_DIM if tta_enabled else 1)
    # Persistent copy of the goal pixel windows: the adapt steps move the
    # BN running stats, so the goal latent is re-encoded after each adapt
    # (start and goal latents must share the same statistics).
    var goal_pix_host = alloc[Scalar[DT]](BATCH * PIX if tta_enabled else 1)
    if tta_enabled:
        for i in range(BATCH * PIX):
            goal_pix_host[i] = pix_host[i]  # goal windows just rendered
    var snap = List[Scalar[DT]]()
    if tta_enabled:
        snap = wm.snapshot_all()  # restored at exit (fresh model/episode)

    var n_exec = execute_blocks
    if n_exec <= 0 or n_exec > MPC_HORIZON:
        n_exec = MPC_HORIZON
    var steps_per_plan = n_exec * FRAMESKIP
    var n_plans = (eval_budget + steps_per_plan - 1) // steps_per_plan
    var do_viz = viz_path.byte_length() > 0
    var viz_buf = alloc[Scalar[DT]](n_plans * VIZN if do_viz else 1)
    var viz_tmp = alloc[Scalar[DT]](VIZN if do_viz else 1)

    var succeeded = List[Bool](length=BATCH, fill=False)

    # ── plan / execute, latching success after every env step ──────────
    var steps_done = 0
    for cyc in range(n_plans):
        # encode current frames → start latents
        for b in range(BATCH):
            var bp = envs[b].block_pose()
            var ap = envs[b].agent_pos()
            sim_frame_chw_norm[IMG](
                bp[0],
                bp[1],
                bp[2],
                ap[0],
                ap[1],
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
        scorer.set_start_goal(start_lat, goal_lat)

        _ = cem.optimize(scorer, plan.as_unsafe_any_origin(), verbose=False)

        # execute the first n_exec future blocks (plan indices H-1 ..)
        for j in range(n_exec):
            if steps_done >= eval_budget:
                break
            var blk = H - 1 + j
            # TTA: stage the pre-block frame per live env. Only blocks that
            # will execute ALL FRAMESKIP sub-steps become window pairs — a
            # budget-truncated block is a partial action the WM never saw.
            var tta_block = tta_enabled and (
                steps_done + FRAMESKIP <= eval_budget
            )
            if tta_block:
                for b in range(BATCH):
                    if succeeded[b]:
                        continue
                    if j == 0:
                        # plan-time frame is still staged in pix_host
                        tta_buf.push_frame(b, pix_host + (b * T) * IMG_DIM)
                    else:
                        var bp = envs[b].block_pose()
                        var ap = envs[b].agent_pos()
                        sim_frame_chw_norm[IMG](
                            bp[0], bp[1], bp[2], ap[0], ap[1], tta_frame
                        )
                        tta_buf.push_frame(b, tta_frame)
            for k in range(FRAMESKIP):
                if steps_done >= eval_budget:
                    break
                for b in range(BATCH):
                    if succeeded[b]:
                        continue
                    var ap = envs[b].agent_pos()
                    var dx = (
                        Float64(
                            plan[(b * NEEDED + blk) * ACT + k * ACT_DIM + 0]
                        )
                        * act_std_x
                        + act_mean_x
                    )
                    var dy = (
                        Float64(
                            plan[(b * NEEDED + blk) * ACT + k * ACT_DIM + 1]
                        )
                        * act_std_y
                        + act_mean_y
                    )
                    var tx = Scalar[DT](Float64(ap[0]) + dx * scale_x)
                    var ty = Scalar[DT](Float64(ap[1]) + dy * scale_y)
                    _ = envs[b].step(PushTAction[DT](tx, ty))
                    # success latch (swm eval_state: episode ends on success)
                    var bp = envs[b].block_pose()
                    var ap2 = envs[b].agent_pos()
                    var r = _state_dist(
                        Float64(ap2[0]),
                        Float64(ap2[1]),
                        Float64(bp[0]),
                        Float64(bp[1]),
                        Float64(bp[2]),
                        goal_states + b * 5,
                    )
                    if r[0] < _SUCCESS_POS_PX and r[1] < _SUCCESS_ANG_RAD:
                        succeeded[b] = True
                steps_done += 1
            if tta_block:
                # complete the (frame, action) pairs; an env that succeeded
                # MID-block executed it partially — skip, orphaning the
                # staged frame (the ring overwrites it).
                for b in range(BATCH):
                    if succeeded[b]:
                        continue
                    tta_buf.push_action(b, plan + (b * NEEDED + blk) * ACT)

        # ── AdaJEPA adapt step: masked gradient steps on the fresh windows,
        # re-sync the planner, and RE-ENCODE the goal latent (the training-
        # mode steps move the BN running stats; start/goal must share them).
        if tta_enabled and tta_buf.fill(pix_host, tta_act_host):
            ctx_v.enqueue_copy(pix_dev, pix_host)
            ctx_v.enqueue_copy(act_dev, tta_act_host)
            ctx_v.synchronize()
            var wpix_t = TileTensor(pix_d_p, row_major[BATCH, PIX]())
            var wact_t = TileTensor(act_d_p, row_major[BATCH, ACTIN]())
            var pre = wm.eval_loss(wpix_t, wact_t)
            wm.reset_loss_accum()
            wm.set_bn_training(True)
            for _ in range(tta_steps):
                _ = wm.train_step_masked(wpix_t, wact_t, keep)
            wm.set_bn_training(False)
            # without this re-sync the CEM plans on stale weights and the
            # whole adapt step is a silent no-op
            scorer.pred_net.sync_from_named(wm.export_named_params())
            ctx_v.enqueue_copy(pix_dev, goal_pix_host)
            ctx_v.synchronize()
            var gp_t = TileTensor(pix_d_p, row_major[BATCH, PIX]())
            var ga_t = TileTensor(act_d_p, row_major[BATCH, ACTIN]())
            _ = wm.eval_loss(gp_t, ga_t)
            wm.read_node_into["emb"](emb_host, BATCH * TE)
            for b in range(BATCH):
                for d in range(EMB):
                    goal_lat[b * EMB + d] = emb_host[b * TE + d]
            if verbose:
                # step loss = training-mode loss at the PRE-update params;
                # adaptation's effect shows in the NEXT cycle's pre value.
                print(
                    "   tta: window loss pre=",
                    pre,
                    " step(train-mode)=",
                    wm.read_loss_accum(),
                )

        if verbose:
            var ns = 0
            var mp: Float64 = 0.0
            for b in range(BATCH):
                if succeeded[b]:
                    ns += 1
                var bp = envs[b].block_pose()
                var ap = envs[b].agent_pos()
                var r = _state_dist(
                    Float64(ap[0]),
                    Float64(ap[1]),
                    Float64(bp[0]),
                    Float64(bp[1]),
                    Float64(bp[2]),
                    goal_states + b * 5,
                )
                mp += r[0]
            print(
                "   plan",
                cyc + 1,
                "/",
                n_plans,
                " steps=",
                steps_done,
                " mean_pos_diff=",
                mp / Float64(BATCH),
                " success=",
                ns,
                "/",
                BATCH,
            )

        if do_viz:
            var bp0 = envs[0].block_pose()
            var ap0 = envs[0].agent_pos()
            var vt = LayoutTensor[
                DT, Layout.row_major(VIZ, VIZ, IMG_C), MutAnyOrigin
            ](viz_tmp.as_unsafe_any_origin())
            render_pusht_rgb_at[VIZ](bp0[0], bp0[1], bp0[2], ap0[0], ap0[1], vt)
            for c in range(IN_CH):
                for y in range(VIZ):
                    for x in range(VIZ):
                        viz_buf[
                            cyc * VIZN + c * VIZ * VIZ + y * VIZ + x
                        ] = viz_tmp[(y * VIZ + x) * IN_CH + c]

    # final metrics
    var ns = 0
    var mp: Float64 = 0.0
    for b in range(BATCH):
        if succeeded[b]:
            ns += 1
        var bp = envs[b].block_pose()
        var ap = envs[b].agent_pos()
        var r = _state_dist(
            Float64(ap[0]),
            Float64(ap[1]),
            Float64(bp[0]),
            Float64(bp[1]),
            Float64(bp[2]),
            goal_states + b * 5,
        )
        mp += r[0]
    mp /= Float64(BATCH)
    var success_rate = Float64(ns) / Float64(BATCH)

    if do_viz:
        save_image_row(
            viz_path,
            viz_buf,
            n=n_plans,
            height=VIZ,
            width=VIZ,
            channels=IN_CH,
            vmin=0.0,
            vmax=255.0,
        )

    if tta_enabled:
        wm.restore_all(snap)  # fresh-model-per-episode: undo the TTA steps

    pix_host.free()
    emb_host.free()
    start_lat.free()
    goal_lat.free()
    plan.free()
    viz_buf.free()
    viz_tmp.free()
    tta_act_host.free()
    tta_frame.free()
    goal_pix_host.free()
    _ = scorer^
    return (success_rate, mp)
