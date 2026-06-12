"""EfficientZeroV2 discrete self-play driver (GPU) — the CartPole lighthouse.

The GPU sibling of `selfplay_cpu.mojo`. Same data-collection shape (single CPU
``BoxDiscreteActionEnv``, CPU `MCTSSequenceReplay` as the ground-truth buffer,
visit-policy action sampling, obs-sequence batches for the SimSiam targets) but
two pieces move to the device:

  * **Search** runs on the `GumbelGPUMCTS` orchestrator (Gumbel-Top-k +
    sequential halving) over the *on-device* ``h/g/f`` nets via the MuZero GPU
    adapters (`MZRepGPU`/`MZDynGPU`/`MZPredGPU`) — per decision D1 the GPU path
    gets the Gumbel planner, while the CPU path keeps vanilla PUCT.
  * **Training** runs `ezv2_unroll_train_step_gpu` (MuZero BPTT + consistency)
    on the resident GPU nets — no CPU mirror, the nets never leave the device.

Single-env (``N_ENVS == 1``) keeps the CPU-env ↔ device-obs round-trip trivial;
the per-step obs is the only host↔device traffic in the collection loop. The
projector/predictor carry BatchNorm but are consistency-only (never at MCTS
inference), so no BN train/eval toggle is needed here. Returns the last loss.
"""

from std.math import exp, log
from std.memory import alloc
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.planners.tree_search import GumbelGPUMCTS, SinglePlayer

from .blocks import ezv2_unroll_train_step_gpu
from .unroll_scratch import EZV2UnrollScratch
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.temperature import visit_temperature
from ..zero.mz_diagnostics import append_mz_train_diagnostics


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


def run_ezv2_gumbel_selfplay_gpu[
    ENV: BoxDiscreteActionEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
    L: Logger = NoOpLogger,
](
    ctx: DeviceContext,
    mut env: ENV,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = 1,
    gamma: Scalar[DT] = Scalar[DT](0.997),
    v_min: Scalar[DT] = Scalar[DT](-10.0),
    v_max: Scalar[DT] = Scalar[DT](10.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 500,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    temperature_decay_steps: Int = 0,
    reanalyze_every: Int = 0,
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    diag_every: Int = 0,
    report_every: Int = 0,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    verbose: Bool = False,
) raises -> Float64:
    comptime N_ENVS = 1
    comptime PRED_OUT = ACT + BINS

    # ── GPU Gumbel planner + on-device net adapters ──
    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        # Tree-GLOBAL σ(Q) normalization: per-node rescale is degenerate at
        # small ACT (CartPole ACT=2 → qn ∈ {0,1} exactly → confident-noise
        # one-hot targets, never converges). See GumbelGPUMCTS.qnorm_per_node.
        qnorm_per_node=False,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    # ── device obs buffer (single env) + host mirrors ──
    var d_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_obs = _a(N_ENVS * OBS)
    var h_pol = _a(N_ENVS * ACT)
    var h_val = _a(N_ENVS)

    # ── training batch slabs (time-major), obs is full [K+1, B, OBS] ──
    var t_obs_seq = _a((K + 1) * B * OBS)
    var t_act = _a(K * B)
    var t_pol = _a((K + 1) * B * ACT)
    var t_val = _a((K + 1) * B)
    var t_rew = _a(K * B)
    var t_cmask = _a(K * B)   # consistency episode-boundary mask

    # ── persistent GPU train-step scratch (allocated once, reused per step) ──
    var train_scratch = EZV2UnrollScratch[
        B, K, OBS, ACT, LATENT, BINS, PROJM.OUT_DIM
    ].make(ctx)

    # logger scratch: per-component loss split + root-prediction probe (D2H).
    var l_parts = _a(4)
    var d_diag_obs = ctx.enqueue_create_buffer[DT](B * OBS)
    var d_diag_z = ctx.enqueue_create_buffer[DT](B * LATENT)
    var d_diag_pred = ctx.enqueue_create_buffer[DT](B * PRED_OUT)
    var h_diag_pred = _a(B * PRED_OUT)

    var e_obs = List[Scalar[DT]]()
    var e_act = List[Scalar[DT]]()
    var e_rew = List[Scalar[DT]]()
    var e_pol = List[Scalar[DT]]()
    var e_val = List[Scalar[DT]]()
    var e_tp = List[Scalar[DT]]()
    var e_legal = List[Scalar[DT]]()
    var ep_len = 0

    var rng = seed ^ UInt64(0x123456789)
    var mcts_seed = UInt32(0)
    var last_loss = 0.0
    var ep_returns = List[Float64]()

    var cur = env.reset_obs_list()
    var cur_f = List[Float64]()
    for j in range(OBS):
        cur_f.append(Float64(cur[j]))
    var ep_return = 0.0

    for it in range(iterations):
        # ── GPU Gumbel search over the current obs ──
        for j in range(OBS):
            h_obs[j] = Scalar[DT](cur_f[j])
        ctx.enqueue_copy(d_obs, h_obs)
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
            MutAnyOrigin](mptr(d_obs.unsafe_ptr()))
        planner.search_gpu[
            MZRepGPU[OBS, LATENT, REP],
            MZDynGPU[LATENT, ACT, BINS, DYN],
            MZPredGPU[LATENT, ACT, BINS, PRED],
        ](ctx, rep_a, dyn_a, pred_a, obs_t,
          apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
        mcts_seed += UInt32(1)

        ctx.enqueue_copy(h_pol, planner.policies_view())
        ctx.enqueue_copy(h_val, planner.root_value_view())
        ctx.synchronize()

        var root_v = Float64(h_val[0])

        # ── sample from the improved policy, tempered π^(1/T) ──
        # The *stored* policy target stays the untempered improved policy.
        var temp = visit_temperature(it, temperature_decay_steps)
        var w = InlineArray[Float64, ACT](fill=0.0)
        var wsum = 0.0
        for a in range(ACT):
            var p = Float64(h_pol[a])
            if temp != 1.0 and p > 0.0:
                p = exp(log(p) / temp)
            w[a] = p
            wsum += p
        rng = rng ^ (rng << 13); rng = rng ^ (rng >> 7); rng = rng ^ (rng << 17)
        var r = Float64(rng % UInt64(1_000_000)) / 1_000_000.0 * wsum
        var cum = 0.0
        var action = ACT - 1
        for a in range(ACT):
            cum += w[a]
            if r <= cum:
                action = a
                break

        for j in range(OBS):
            e_obs.append(Scalar[DT](cur_f[j]))
        e_act.append(Scalar[DT](action))
        for a in range(ACT):
            e_pol.append(Scalar[DT](Float64(h_pol[a])))
            e_legal.append(Scalar[DT](1.0))
        e_val.append(Scalar[DT](root_v))
        e_tp.append(Scalar[DT](0.0))

        var stepped = env.step_obs(action)
        var reward = Float64(stepped[1])
        var done = stepped[2]
        e_rew.append(Scalar[DT](reward))
        ep_return += reward
        ep_len += 1

        cur_f = List[Float64]()
        for j in range(OBS):
            cur_f.append(Float64(stepped[0][j]))

        if done or ep_len >= max_ep_steps:
            # Time-limit cut is NOT a terminal — bootstrap past it.
            rb.store_episode(
                mptr(e_obs.unsafe_ptr()),
                mptr(e_act.unsafe_ptr()),
                mptr(e_rew.unsafe_ptr()),
                mptr(e_pol.unsafe_ptr()),
                mptr(e_val.unsafe_ptr()),
                mptr(e_tp.unsafe_ptr()),
                mptr(e_legal.unsafe_ptr()),
                ep_len,
                truncated=not env.was_terminated(),
            )
            ep_returns.append(ep_return)
            e_obs.clear(); e_act.clear(); e_rew.clear()
            e_pol.clear(); e_val.clear(); e_tp.clear(); e_legal.clear()
            ep_len = 0
            ep_return = 0.0
            cur = env.reset_obs_list()
            cur_f = List[Float64]()
            for j in range(OBS):
                cur_f.append(Float64(cur[j]))

        # ── GPU train step ──
        if it >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch_seq[B, K, N](
                    gamma, t_obs_seq, t_act, t_pol, t_val, t_rew,
                    cons_mask=t_cmask,
                )
                last_loss = Float64(
                    ezv2_unroll_train_step_gpu[
                        REP, DYN, PRED, PROJM, PREDH,
                        B, K, OBS, ACT, LATENT, BINS,
                    ](
                        ctx, train_scratch, rep, dyn, pred, proj, predh,
                        orep, odyn, opred, oproj, opredh,
                        t_obs_seq, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef, consistency_coef,
                        cons_mask=t_cmask,
                        loss_parts=l_parts,
                    )
                )

        # ── per-batch diagnostics → logger (root pred re-forwarded on device) ──
        if (
            logger
            and diag_every > 0
            and it >= learning_starts
            and rb.num_episodes() > 0
            and (it + 1) % diag_every == 0
        ):
            # root obs0 = first B*OBS of the last sampled batch (host slab).
            ctx.enqueue_copy(d_diag_obs, t_obs_seq)
            var z_t = TileTensor(
                mptr(d_diag_z.unsafe_ptr()), row_major[B, LATENT]()
            )
            rep.forward["gpu", B](
                TileTensor(
                    mptr(d_diag_obs.unsafe_ptr()), row_major[B, OBS]()
                ),
                output=z_t,
            )
            var pred_t = TileTensor(
                mptr(d_diag_pred.unsafe_ptr()), row_major[B, PRED_OUT]()
            )
            pred.forward["gpu", B](z_t, output=pred_t)
            ctx.enqueue_copy(h_diag_pred, d_diag_pred)
            ctx.synchronize()
            var dn = List[String]()
            var dv = List[Float64]()
            dn.append(String("loss")); dv.append(last_loss)
            dn.append(String("loss_policy")); dv.append(Float64(l_parts[0]))
            dn.append(String("loss_value")); dv.append(Float64(l_parts[1]))
            dn.append(String("loss_reward")); dv.append(Float64(l_parts[2]))
            dn.append(String("loss_consistency"))
            dv.append(Float64(l_parts[3]))
            append_mz_train_diagnostics[ACT, BINS, B](
                h_diag_pred, t_pol, t_val, v_min, v_max, dn, dv
            )
            logger.value()[].log_scalars(dn, dv, it + 1)

        # ── reanalyze: refresh one stored position with a fresh search ──
        # Re-search the stored obs with the current on-device nets and
        # overwrite the stored improved policy + root value (n-step targets
        # bootstrap from those). Reuses the single-env obs/pol/val buffers.
        if (
            reanalyze_every > 0
            and it >= learning_starts
            and (it + 1) % reanalyze_every == 0
            and rb.num_episodes() > 0
        ):
            var rpos = rb.sample_position()
            rb.read_obs(rpos[0], rpos[1], h_obs)
            ctx.enqueue_copy(d_obs, h_obs)
            var robs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
                MutAnyOrigin](mptr(d_obs.unsafe_ptr()))
            planner.search_gpu[
                MZRepGPU[OBS, LATENT, REP],
                MZDynGPU[LATENT, ACT, BINS, DYN],
                MZPredGPU[LATENT, ACT, BINS, PRED],
            ](ctx, rep_a, dyn_a, pred_a, robs_t,
              apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
            mcts_seed += UInt32(1)
            ctx.enqueue_copy(h_pol, planner.policies_view())
            ctx.enqueue_copy(h_val, planner.root_value_view())
            ctx.synchronize()
            rb.update_targets(rpos[0], rpos[1], h_pol, h_val[0])

        # ── greedy eval (argmax of the Gumbel improved policy) ──
        if eval_every > 0 and (it + 1) % eval_every == 0:
            var eval_sum = 0.0
            for _ in range(eval_episodes):
                var eo = env.reset_obs_list()
                var eo_f = List[Float64]()
                for j in range(OBS):
                    eo_f.append(Float64(eo[j]))
                var eret = 0.0
                for _step in range(max_ep_steps):
                    for j in range(OBS):
                        h_obs[j] = Scalar[DT](eo_f[j])
                    ctx.enqueue_copy(d_obs, h_obs)
                    var eobs_t = LayoutTensor[DT,
                        Layout.row_major(N_ENVS, OBS), MutAnyOrigin](
                            mptr(d_obs.unsafe_ptr()))
                    planner.search_gpu[
                        MZRepGPU[OBS, LATENT, REP],
                        MZDynGPU[LATENT, ACT, BINS, DYN],
                        MZPredGPU[LATENT, ACT, BINS, PRED],
                    ](ctx, rep_a, dyn_a, pred_a, eobs_t,
                      apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
                    mcts_seed += UInt32(1)
                    ctx.enqueue_copy(h_pol, planner.policies_view())
                    ctx.synchronize()
                    var best = 0
                    for a in range(1, ACT):
                        if Float64(h_pol[a]) > Float64(h_pol[best]):
                            best = a
                    var es = env.step_obs(best)
                    eret += Float64(es[1])
                    eo_f = List[Float64]()
                    for j in range(OBS):
                        eo_f.append(Float64(es[0][j]))
                    if es[2]:
                        break
                eval_sum += eret
            var eval_avg = eval_sum / Float64(eval_episodes)
            print("  [eval] step", it + 1, "greedy_return", eval_avg)
            if logger:
                logger.value()[].log_scalar(
                    String("eval_return"), eval_avg, it + 1
                )
            e_obs.clear(); e_act.clear(); e_rew.clear()
            e_pol.clear(); e_val.clear(); e_tp.clear(); e_legal.clear()
            ep_len = 0
            ep_return = 0.0
            cur = env.reset_obs_list()
            cur_f = List[Float64]()
            for j in range(OBS):
                cur_f.append(Float64(cur[j]))

        if verbose and (it + 1) % 500 == 0:
            var avg = 0.0
            var cnt = 0
            var lo = len(ep_returns) - 10
            if lo < 0:
                lo = 0
            for e in range(lo, len(ep_returns)):
                avg += ep_returns[e]
                cnt += 1
            if cnt > 0:
                avg /= Float64(cnt)
            print(
                "step", it + 1, "loss", last_loss,
                "eps", rb.num_episodes(), "avg_return(10)", avg,
            )

        # ── report_every: episode-return / replay status to the logger ──
        if (
            logger
            and report_every > 0
            and it >= learning_starts
            and (it + 1) % report_every == 0
        ):
            var ravg = 0.0
            var rcnt = 0
            var rlo = len(ep_returns) - 10
            if rlo < 0:
                rlo = 0
            for e in range(rlo, len(ep_returns)):
                ravg += ep_returns[e]
                rcnt += 1
            if rcnt > 0:
                ravg /= Float64(rcnt)
            var rn = List[String]()
            var rv = List[Float64]()
            rn.append(String("avg_return")); rv.append(ravg)
            rn.append(String("episodes")); rv.append(Float64(rb.num_episodes()))
            rn.append(String("replay_size")); rv.append(Float64(rb.num_steps()))
            logger.value()[].log_scalars(rn, rv, it + 1)

    t_obs_seq.free(); t_act.free(); t_pol.free(); t_val.free(); t_rew.free()
    t_cmask.free()
    h_obs.free(); h_pol.free(); h_val.free()
    l_parts.free(); h_diag_pred.free()
    return last_loss
