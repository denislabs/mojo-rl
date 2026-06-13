"""MuZero batched GPU self-play driver — N parallel envs + batched search.

The throughput-oriented sibling of `selfplay_gpu_device.mojo`'s single-env
`run_muzero_gumbel_selfplay_gpu`. Where that driver steps ONE env per iteration
(fine for CartPole, fatal for pixels), this one steps ``N_ENVS`` envs in lockstep
through a `BatchedGpuDiscreteEnv` and searches all of them in a single
`GumbelGPUMCTS` launch — the configuration Pong needs to converge in reasonable
wall-time (Rainbow only solved Pong by stepping 64 GPU envs in parallel).

Phase 1 of `docs/MUZERO_PIXEL_PONG_PLAN.md`: the batched collection loop wired to
the **host** `MCTSSequenceReplay` (now uint8-capable via ``OBS_STORE_DT``). The
per-iteration shape:

  1. search the env's LIVE device obs (`env.obs_ptr()` wrapped as ``[N_ENVS, OBS]``)
     — no H2D for the search input; the rep CNN runs at batch=``N_ENVS`` at the root.
  2. D2H the improved policy ``[N_ENVS, ACT]``, root value ``[N_ENVS]``, and a
     snapshot of the root obs ``[N_ENVS, OBS]`` (the one per-step D2H Phase 1
     accepts; Phase 2's `GPUMCTSSequenceReplay` removes it).
  3. per env: sample ∝ π^(1/T), append the labelled step to that env's host
     episode buffer, write the action index into `env.action_ptr()`.
  4. `env.step_batch` → reward / done / terminated; `env.selective_reset_batch`.
  5. per env: on done/time-limit, `store_episode` (truncated = not terminated) and
     reset that env's host buffer.
  6. train: `sample_training_batch` → `mz_unroll_train_step_gpu` on the resident
     nets (unchanged from the single-env driver).

Optional batched reanalyze (sample ``N_ENVS`` stored positions → one fresh search
→ overwrite their targets) and a fixed-horizon batched greedy eval mirror the
single-env driver's convergence stack.

Run (GPU env required):
    pixi run -e apple  mojo run -I . tests/deep_agents2/test_mz_selfplay_gpu_batched_smoke.mojo
"""

from std.math import exp, log
from std.memory import alloc
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.planners.tree_search import GumbelGPUMCTS, SinglePlayer

from ..training.batched_env import BatchedEnv
from .blocks import mz_unroll_train_step_gpu, MZScratch
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.mz_diagnostics import append_mz_train_diagnostics
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.gpu_sequence_replay_mcts import GPUMCTSSequenceReplay
from ..zero.temperature import visit_temperature


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


def _avg_last_n(returns: List[Float64], n: Int) -> Float64:
    """Mean of the last ``n`` recorded episode returns (0 if none)."""
    var lo = len(returns) - n
    if lo < 0:
        lo = 0
    var s = 0.0
    var c = 0
    for i in range(lo, len(returns)):
        s += returns[i]
        c += 1
    return s / Float64(c) if c > 0 else 0.0


def _mz_emit_batch_diag[
    REP: Module, PRED: Module,
    B: Int, OBS: Int, ACT: Int, LATENT: Int, BINS: Int, L: Logger,
](
    ctx: DeviceContext,
    mut rep: REP,
    mut pred: PRED,
    d_obs0: DeviceBuffer[DT],   # the last train batch's obs, already on device
    d_z: DeviceBuffer[DT],
    d_pred: DeviceBuffer[DT],
    h_pred: UnsafePointer[Scalar[DT], MutAnyOrigin],
    t_pol: UnsafePointer[Scalar[DT], MutAnyOrigin],
    t_val: UnsafePointer[Scalar[DT], MutAnyOrigin],
    l_parts: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    last_loss: Float64,
    step: Int,
    logger: UnsafePointer[L, MutAnyOrigin],
) raises:
    """Re-forward the root prediction on the last train batch (using the obs
    slab already resident in ``d_obs0`` = ``scratch.d_obs0``) and emit the full
    single-env metric set: loss + policy/value/reward split + the head-fit
    diagnostics (`append_mz_train_diagnostics`). ``t_pol``/``t_val`` start at the
    root (position 0) block, which is what the diagnostics read."""
    comptime PRED_OUT = ACT + BINS
    var z_t = TileTensor(mptr(d_z.unsafe_ptr()), row_major[B, LATENT]())
    rep.forward["gpu", B](
        TileTensor(mptr(d_obs0.unsafe_ptr()), row_major[B, OBS]()), output=z_t
    )
    var pred_t = TileTensor(mptr(d_pred.unsafe_ptr()), row_major[B, PRED_OUT]())
    pred.forward["gpu", B](z_t, output=pred_t)
    ctx.enqueue_copy(h_pred, d_pred)
    ctx.synchronize()
    var dn = List[String]()
    var dv = List[Float64]()
    dn.append(String("loss")); dv.append(last_loss)
    dn.append(String("loss_policy")); dv.append(Float64(l_parts[0]))
    dn.append(String("loss_value")); dv.append(Float64(l_parts[1]))
    dn.append(String("loss_reward")); dv.append(Float64(l_parts[2]))
    append_mz_train_diagnostics[ACT, BINS, B](
        h_pred, t_pol, t_val, v_min, v_max, dn, dv
    )
    logger[].log_scalars(dn, dv, step)


def _sample_action(
    h_pol: UnsafePointer[Scalar[DT], MutAnyOrigin],
    base: Int,
    act: Int,
    temp: Float64,
    mut rng: UInt64,
) -> Int:
    """Sample an action index from the (improved) policy row ``h_pol[base..]``
    with the legacy ∝ π^(1/T) tempering. Mirrors the single-env driver's host
    sampler exactly so a 1-env batched run matches the single-env one."""
    var wsum = 0.0
    var w = List[Float64](capacity=act)
    for a in range(act):
        var p = Float64(h_pol[base + a])
        if temp != 1.0 and p > 0.0:
            p = exp(log(p) / temp)
        w.append(p)
        wsum += p
    rng = rng ^ (rng << 13)
    rng = rng ^ (rng >> 7)
    rng = rng ^ (rng << 17)
    var r = Float64(rng % UInt64(1_000_000)) / 1_000_000.0 * wsum
    var cum = 0.0
    var action = act - 1
    for a in range(act):
        cum += w[a]
        if r <= cum:
            action = a
            break
    return action


def run_muzero_gumbel_selfplay_gpu_batched[
    BENV: BatchedEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_ENVS: Int,
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
    OBS_STORE_DT: DType = DT,
    L: Logger = NoOpLogger,
](
    ctx: DeviceContext,
    mut env: BENV,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = N_ENVS,   # default UTD 1:1 (grad steps == env steps)
    gamma: Scalar[DT] = Scalar[DT](0.997),
    v_min: Scalar[DT] = Scalar[DT](-10.0),
    v_max: Scalar[DT] = Scalar[DT](10.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 27000,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    temperature_decay_steps: Int = 0,
    reanalyze_every: Int = 0,
    reanalyze_batch: Int = N_ENVS,
    target_sync_interval: Int = 0,
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    eval_horizon: Int = 0,   # 0 ⇒ generous step cap; else hard per-eval cap
    eval_env: Optional[UnsafePointer[BENV, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    report_every: Int = 0,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    verbose: Bool = False,
) raises -> Float64:
    """Batched Gumbel MuZero self-play. ``learning_starts`` is in **stored
    steps** (training begins once the host replay holds that many completed-
    episode steps). Each driver iteration advances ``N_ENVS`` env steps, so the
    total environment interaction is ``iterations · N_ENVS``.

    ``reanalyze_batch`` sets how many stored positions are re-targeted with the
    CURRENT net per reanalyze trigger (processed in ``reanalyze_batch // N_ENVS``
    chunks of ``N_ENVS``); fresh root policy + value are written back in place so
    the n-step targets pick them up on the next sample. Default ``N_ENVS`` (one
    chunk, historical low coverage); set ≈ ``B`` for the EfficientZero-style
    high-coverage regime (parity with the devreplay driver)."""
    comptime assert BENV.OBS_DIM == OBS, (
        "batched MuZero: BENV.OBS_DIM must equal OBS"
    )

    # ── on-device Gumbel planner (batched over N_ENVS) + net adapters ──
    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    # Dedicated eval planner — eval must be read-only w.r.t. training. Sharing
    # the training planner's device buffers AND the `mcts_seed` RNG stream (eval
    # advanced it by ~horizon per call) perturbed training: at N_ENVS=1 it tipped
    # the (fragile single-env) run into a greedy collapse that eval-off avoided
    # entirely (see docs/MUZERO_PIXEL_PONG_PLAN.md). The net adapters ARE shared
    # — eval must evaluate the current trained nets (forward is read-only on
    # params; the activation cache is rewritten by each train step's own forward).
    var eval_planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    # ── lagging target nets for reanalyze (gated by `target_sync_interval`) ──
    # Reanalyze ALWAYS searches through these adapters. When
    # `target_sync_interval > 0` they are hard-copied from the live nets every
    # that-many grad steps → a delayed target that decouples target generation
    # from the optimizer step (the standard target-net stabilizer; matches
    # EZv2 / official MuZero's delayed reanalyze model — important now that the
    # coverage lever refreshes ~B targets per trigger). When 0 they are synced
    # to the live weights right before each reanalyze trigger, so reanalyze is
    # bit-identical to the live-net path (params-only copy, like EZv2; the
    # Nature-CNN rep carries no BatchNorm running stats).
    var rep_t = REP.make["gpu", INIT=Kaiming](ctx)
    var dyn_t = DYN.make["gpu", INIT=Kaiming](ctx)
    var pred_t = PRED.make["gpu", INIT=Kaiming](ctx)
    hard_copy_params["gpu", M=REP](rep, rep_t, ctx)
    hard_copy_params["gpu", M=DYN](dyn, dyn_t, ctx)
    hard_copy_params["gpu", M=PRED](pred, pred_t, ctx)
    var rep_ta = MZRepGPU[OBS, LATENT, REP].make(rep_t)
    var dyn_ta = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn_t)
    var pred_ta = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred_t)
    var train_steps = 0

    var rb = MCTSSequenceReplay[OBS, ACT, CAP, OBS_STORE_DT](
        seed=seed ^ UInt64(0xABCDEF)
    )

    # ── host mirrors for the per-step D2H/H2D (N_ENVS-wide) ──
    var h_obs = _a(N_ENVS * OBS)   # root-obs snapshot (prev_obs)
    var h_pol = _a(N_ENVS * ACT)
    var h_val = _a(N_ENVS)
    var h_act = _a(N_ENVS)
    var h_rew = _a(N_ENVS)
    var h_done = _a(N_ENVS)
    var h_term = _a(N_ENVS)

    # ── training batch slabs (time-major), allocated once ──
    var t_obs0 = _a(B * OBS)
    var t_act = _a(K * B)
    var t_pol = _a((K + 1) * B * ACT)
    var t_val = _a((K + 1) * B)
    var t_rew = _a(K * B)
    var l_parts = _a(3)
    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

    # ── reanalyze obs scratch (device) — its own buffer, NOT the live env obs ──
    var d_reana = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_reana = _a(N_ENVS * OBS)

    # ── per-env episode accumulators ──
    var e_obs = List[List[Scalar[DT]]]()
    var e_act = List[List[Scalar[DT]]]()
    var e_rew = List[List[Scalar[DT]]]()
    var e_pol = List[List[Scalar[DT]]]()
    var e_val = List[List[Scalar[DT]]]()
    var e_tp = List[List[Scalar[DT]]]()
    var e_legal = List[List[Scalar[DT]]]()
    var ep_len = List[Int]()
    var ep_return = List[Float64]()
    for _ in range(N_ENVS):
        e_obs.append(List[Scalar[DT]]())
        e_act.append(List[Scalar[DT]]())
        e_rew.append(List[Scalar[DT]]())
        e_pol.append(List[Scalar[DT]]())
        e_val.append(List[Scalar[DT]]())
        e_tp.append(List[Scalar[DT]]())
        e_legal.append(List[Scalar[DT]]())
        ep_len.append(0)
        ep_return.append(0.0)

    var rng = seed ^ UInt64(0x123456789)
    var mcts_seed = UInt32(seed & UInt64(0xFFFF))
    # Separate RNG stream for eval so it never perturbs the training search seeds.
    var eval_seed = UInt32((seed ^ UInt64(0xE7A1B2C3)) & UInt64(0xFFFF))
    var last_loss = 0.0
    var ep_returns = List[Float64]()

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=seed)

    for it in range(iterations):
        # ── 1. batched Gumbel search over the LIVE env obs (no H2D in) ──
        var obs_t = LayoutTensor[
            DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
        ](env.obs_ptr())
        planner.search_gpu[
            MZRepGPU[OBS, LATENT, REP],
            MZDynGPU[LATENT, ACT, BINS, DYN],
            MZPredGPU[LATENT, ACT, BINS, PRED],
        ](
            ctx, rep_a, dyn_a, pred_a, obs_t,
            apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed,
        )
        mcts_seed += UInt32(1)

        # ── 2. D2H policy, value, and the root-obs snapshot ──
        var obs_view = DeviceBuffer[DT](
            ctx, env.obs_ptr(), N_ENVS * OBS, owning=False
        )
        ctx.enqueue_copy(h_obs, obs_view)
        ctx.enqueue_copy(h_pol, planner.policies_view())
        ctx.enqueue_copy(h_val, planner.root_value_view())
        ctx.synchronize()

        # ── 3. per env: sample, record the labelled step, stage the action ──
        var temp = visit_temperature(it, temperature_decay_steps)
        for e in range(N_ENVS):
            var action = _sample_action(h_pol, e * ACT, ACT, temp, rng)
            for j in range(OBS):
                e_obs[e].append(h_obs[e * OBS + j])
            e_act[e].append(Scalar[DT](action))
            for a in range(ACT):
                e_pol[e].append(h_pol[e * ACT + a])
                e_legal[e].append(Scalar[DT](1.0))
            e_val[e].append(h_val[e])
            e_tp[e].append(Scalar[DT](0.0))
            h_act[e] = Scalar[DT](action)

        # ── 4. H2D actions → env, step, D2H reward/done/terminated ──
        var act_view = DeviceBuffer[DT](
            ctx, env.action_ptr(), N_ENVS, owning=False
        )
        ctx.enqueue_copy(act_view, h_act)
        env.step_batch[N_ENVS](ctx=ctx, rng_seed=seed + UInt64(it + 1))
        var rew_view = DeviceBuffer[DT](
            ctx, env.reward_ptr(), N_ENVS, owning=False
        )
        var done_view = DeviceBuffer[DT](
            ctx, env.done_ptr(), N_ENVS, owning=False
        )
        var term_view = DeviceBuffer[DT](
            ctx, env.terminated_ptr(), N_ENVS, owning=False
        )
        ctx.enqueue_copy(h_rew, rew_view)
        ctx.enqueue_copy(h_done, done_view)
        ctx.enqueue_copy(h_term, term_view)
        ctx.synchronize()

        # ── 5. per env: accumulate reward, store + reset finished episodes ──
        for e in range(N_ENVS):
            e_rew[e].append(h_rew[e])
            ep_return[e] += Float64(h_rew[e])
            ep_len[e] += 1
            var done = h_done[e] > Scalar[DT](0.5)
            var terminated = h_term[e] > Scalar[DT](0.5)
            if done or ep_len[e] >= max_ep_steps:
                # Time-limit cut (not terminated) is NOT a terminal → bootstrap.
                rb.store_episode(
                    mptr(e_obs[e].unsafe_ptr()),
                    mptr(e_act[e].unsafe_ptr()),
                    mptr(e_rew[e].unsafe_ptr()),
                    mptr(e_pol[e].unsafe_ptr()),
                    mptr(e_val[e].unsafe_ptr()),
                    mptr(e_tp[e].unsafe_ptr()),
                    mptr(e_legal[e].unsafe_ptr()),
                    ep_len[e],
                    truncated=not terminated,
                )
                ep_returns.append(ep_return[e])
                e_obs[e].clear(); e_act[e].clear(); e_rew[e].clear()
                e_pol[e].clear(); e_val[e].clear(); e_tp[e].clear()
                e_legal[e].clear()
                ep_len[e] = 0
                ep_return[e] = 0.0

        # ── reset the done envs on device (state reset; pixel obs left as-is) ──
        env.selective_reset_batch[N_ENVS](
            ctx=ctx, rng_seed=seed + UInt64(it + 1)
        )

        # ── 6. train (GPU unroll on the resident nets — no mirror sync) ──
        if rb.num_steps() >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch[B, K, N](
                    gamma, t_obs0, t_act, t_pol, t_val, t_rew
                )
                last_loss = Float64(
                    mz_unroll_train_step_gpu[
                        REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS
                    ](
                        ctx, rep, dyn, pred, orep, odyn, opred,
                        scratch,
                        t_obs0, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef,
                        loss_parts=l_parts,
                    )
                )
                train_steps += 1
                if (
                    target_sync_interval > 0
                    and train_steps % target_sync_interval == 0
                ):
                    hard_copy_params["gpu", M=REP](rep, rep_t, ctx)
                    hard_copy_params["gpu", M=DYN](dyn, dyn_t, ctx)
                    hard_copy_params["gpu", M=PRED](pred, pred_t, ctx)

        var trained = rb.num_steps() >= learning_starts and rb.num_episodes() > 0

        # ── per-batch loss diagnostics → logger ──
        if (
            logger
            and diag_every > 0
            and trained
            and (it + 1) % diag_every == 0
        ):
            var dn = List[String]()
            var dv = List[Float64]()
            dn.append(String("loss")); dv.append(last_loss)
            dn.append(String("loss_policy")); dv.append(Float64(l_parts[0]))
            dn.append(String("loss_value")); dv.append(Float64(l_parts[1]))
            dn.append(String("loss_reward")); dv.append(Float64(l_parts[2]))
            logger.value()[].log_scalars(dn, dv, it + 1)

        # ── high-coverage batched reanalyze: re-target `reanalyze_batch` stored
        #    positions with the CURRENT net per trigger, in chunks of N_ENVS (the
        #    planner's root width). Each chunk stages N_ENVS obs host→device, runs
        #    one batched Gumbel search, and writes the fresh root policy + value
        #    back in place — the n-step targets pick them up on the next sample.
        #    Lifting `reanalyze_batch` from N_ENVS toward B is the
        #    EfficientZero-style coverage lever (parity with the devreplay path). ──
        if (
            reanalyze_every > 0
            and trained
            and (it + 1) % reanalyze_every == 0
        ):
            # target_sync_interval == 0 ⇒ live-net reanalyze: refresh the target
            # to the current weights now (search through rep_ta then == rep_a).
            if target_sync_interval == 0:
                hard_copy_params["gpu", M=REP](rep, rep_t, ctx)
                hard_copy_params["gpu", M=DYN](dyn, dyn_t, ctx)
                hard_copy_params["gpu", M=PRED](pred, pred_t, ctx)
            var n_chunks = reanalyze_batch // N_ENVS
            if n_chunks < 1:
                n_chunks = 1
            for _c in range(n_chunks):
                var rpos_e = List[Int]()
                var rpos_o = List[Int]()
                for e in range(N_ENVS):
                    var rpos = rb.sample_position()
                    rpos_e.append(rpos[0])
                    rpos_o.append(rpos[1])
                    var tmp = _a(OBS)
                    rb.read_obs(rpos[0], rpos[1], tmp)
                    for j in range(OBS):
                        h_reana[e * OBS + j] = tmp[j]
                    tmp.free()
                ctx.enqueue_copy(d_reana, h_reana)
                var reana_t = LayoutTensor[
                    DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
                ](mptr(d_reana.unsafe_ptr()))
                planner.search_gpu[
                    MZRepGPU[OBS, LATENT, REP],
                    MZDynGPU[LATENT, ACT, BINS, DYN],
                    MZPredGPU[LATENT, ACT, BINS, PRED],
                ](
                    ctx, rep_ta, dyn_ta, pred_ta, reana_t,
                    apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed,
                )
                mcts_seed += UInt32(1)
                ctx.enqueue_copy(h_pol, planner.policies_view())
                ctx.enqueue_copy(h_val, planner.root_value_view())
                ctx.synchronize()
                for e in range(N_ENVS):
                    rb.update_targets(
                        rpos_e[e], rpos_o[e],
                        h_pol + (e * ACT), h_val[e],
                    )

        # ── batched greedy eval (fixed horizon on a separate eval env) ──
        if eval_every > 0 and eval_env and (it + 1) % eval_every == 0:
            # `eval_horizon` (if set) is the per-eval step CAP; else a generous
            # default that early-exits once `eval_episodes` games complete.
            var cap = (
                eval_horizon if eval_horizon > 0
                else max_ep_steps * (eval_episodes + 1)
            )
            var avg = _eval_greedy_batched[
                BENV, REP, DYN, PRED, N_ENVS, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, eval_env.value(), eval_planner, rep_a, dyn_a, pred_a,
                eval_episodes, cap, eval_seed,
            )
            eval_seed += UInt32(cap + 1)
            print("  [eval] step", it + 1, "greedy_return", avg)
            if logger:
                logger.value()[].log_scalar(
                    String("eval_return"), avg, it + 1
                )

        # ── verbose progress ──
        if verbose and (it + 1) % 500 == 0:
            var avg = 0.0
            var cnt = 0
            var lo = len(ep_returns) - 10
            if lo < 0:
                lo = 0
            for ee in range(lo, len(ep_returns)):
                avg += ep_returns[ee]
                cnt += 1
            if cnt > 0:
                avg /= Float64(cnt)
            print(
                "iter", it + 1, "env_steps", (it + 1) * N_ENVS,
                "loss", last_loss, "eps", rb.num_episodes(),
                "avg_return(10)", avg,
            )

        # ── report_every: episode-return / replay status → logger ──
        if (
            logger
            and report_every > 0
            and trained
            and (it + 1) % report_every == 0
        ):
            var ravg = 0.0
            var rcnt = 0
            var rlo = len(ep_returns) - 10
            if rlo < 0:
                rlo = 0
            for ee in range(rlo, len(ep_returns)):
                ravg += ep_returns[ee]
                rcnt += 1
            if rcnt > 0:
                ravg /= Float64(rcnt)
            var rn = List[String]()
            var rv = List[Float64]()
            rn.append(String("avg_return")); rv.append(ravg)
            rn.append(String("episodes"))
            rv.append(Float64(rb.num_episodes()))
            rn.append(String("replay_size"))
            rv.append(Float64(rb.num_steps()))
            logger.value()[].log_scalars(rn, rv, it + 1)

    t_obs0.free(); t_act.free(); t_pol.free(); t_val.free(); t_rew.free()
    h_obs.free(); h_pol.free(); h_val.free(); h_act.free()
    h_rew.free(); h_done.free(); h_term.free(); h_reana.free(); l_parts.free()
    # keep the target nets (held only via UnsafePointer in the adapters) alive.
    _ = rep_t^
    _ = dyn_t^
    _ = pred_t^
    return last_loss


def run_muzero_gumbel_selfplay_gpu_batched_devreplay[
    BENV: BatchedEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_ENVS: Int,
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
    OBS_STORE_DT: DType = DType.uint8,
    L: Logger = NoOpLogger,
](
    ctx: DeviceContext,
    mut env: BENV,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = N_ENVS,   # default UTD 1:1 (grad steps == env steps)
    gamma: Scalar[DT] = Scalar[DT](0.997),
    v_min: Scalar[DT] = Scalar[DT](-10.0),
    v_max: Scalar[DT] = Scalar[DT](10.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 27000,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    temperature_decay_steps: Int = 0,
    reanalyze_every: Int = 0,
    reanalyze_batch: Int = N_ENVS,
    target_sync_interval: Int = 0,
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    eval_horizon: Int = 0,   # 0 ⇒ generous step cap; else hard per-eval cap
    eval_env: Optional[UnsafePointer[BENV, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    report_every: Int = 0,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    verbose: Bool = False,
) raises -> Float64:
    """Device-obs twin of `run_muzero_gumbel_selfplay_gpu_batched`: identical
    loop, but the obs ring lives on the GPU (`GPUMCTSSequenceReplay`) so the
    collection path never D2H's a full ``[N_ENVS, OBS]`` observation — obs are
    stored device→device from `env.obs_ptr()` and the training obs slab is
    gathered device→device into the train step's own buffer
    (`mz_unroll_train_step_gpu[obs_on_device=True]`). Only the tiny
    reward/done/policy/value scalars cross the bus per step. Requires
    ``CAP >= N_ENVS · max_ep_steps`` (else an in-flight episode self-overwrites;
    see `GPUMCTSSequenceReplay`).

    ``reanalyze_batch`` sets how many stored positions are re-targeted with the
    CURRENT net per reanalyze trigger (processed in ``reanalyze_batch // N_ENVS``
    chunks of ``N_ENVS`` — the planner's root width). The fresh root policy +
    value are written back in place, so the n-step targets pick them up on the
    next sample. Default ``N_ENVS`` (one chunk) keeps the historical low-coverage
    behaviour; set it ≈ ``B`` so a meaningful fraction of each training batch
    carries fresh targets (the EfficientZero-style reanalyze regime that drives
    sample efficiency on hard envs)."""
    comptime assert BENV.OBS_DIM == OBS, (
        "batched MuZero: BENV.OBS_DIM must equal OBS"
    )

    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    # Dedicated eval planner — eval must be read-only w.r.t. training. Sharing
    # the training planner's device buffers AND the `mcts_seed` RNG stream (eval
    # advanced it by ~horizon per call) perturbed training: at N_ENVS=1 it tipped
    # the (fragile single-env) run into a greedy collapse that eval-off avoided
    # entirely (see docs/MUZERO_PIXEL_PONG_PLAN.md). The net adapters ARE shared
    # — eval must evaluate the current trained nets (forward is read-only on
    # params; the activation cache is rewritten by each train step's own forward).
    var eval_planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    # ── lagging target nets for reanalyze (gated by `target_sync_interval`; see
    #    the host-replay driver for the rationale). Reanalyze always searches
    #    through these; synced every `target_sync_interval` grad steps when > 0,
    #    else refreshed to live just before each trigger (bit-identical to the
    #    live-net path). Params-only copy (Nature-CNN rep has no BatchNorm). ──
    var rep_t = REP.make["gpu", INIT=Kaiming](ctx)
    var dyn_t = DYN.make["gpu", INIT=Kaiming](ctx)
    var pred_t = PRED.make["gpu", INIT=Kaiming](ctx)
    hard_copy_params["gpu", M=REP](rep, rep_t, ctx)
    hard_copy_params["gpu", M=DYN](dyn, dyn_t, ctx)
    hard_copy_params["gpu", M=PRED](pred, pred_t, ctx)
    var rep_ta = MZRepGPU[OBS, LATENT, REP].make(rep_t)
    var dyn_ta = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn_t)
    var pred_ta = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred_t)
    var train_steps = 0

    var rb = GPUMCTSSequenceReplay[OBS, ACT, CAP, N_ENVS, OBS_STORE_DT](
        ctx, seed=seed ^ UInt64(0xABCDEF)
    )

    # host mirrors (no full-obs D2H — only N_ENVS-wide scalar/policy traffic).
    var h_pol = _a(N_ENVS * ACT)
    var h_val = _a(N_ENVS)
    var h_act = _a(N_ENVS)
    var h_rew = _a(N_ENVS)
    var h_done = _a(N_ENVS)
    var h_term = _a(N_ENVS)

    # training metadata slabs (obs0 is gathered on-device into scratch.d_obs0).
    var t_act = _a(K * B)
    var t_pol = _a((K + 1) * B * ACT)
    var t_val = _a((K + 1) * B)
    var t_rew = _a(K * B)
    var t_obs0_dummy = _a(1)   # ignored (obs_on_device=True)
    var l_parts = _a(3)
    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

    # diagnostics scratch (root re-forward on the last train batch's obs).
    var d_diag_z = ctx.enqueue_create_buffer[DT](B * LATENT)
    var d_diag_pred = ctx.enqueue_create_buffer[DT](B * (ACT + BINS))
    var h_diag_pred = _a(B * (ACT + BINS))

    # one chunk's worth of reanalyze obs (gathered device→device per chunk).
    var d_reana = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)

    var rng = seed ^ UInt64(0x123456789)
    var mcts_seed = UInt32(seed & UInt64(0xFFFF))
    # Separate RNG stream for eval so it never perturbs the training search seeds.
    var eval_seed = UInt32((seed ^ UInt64(0xE7A1B2C3)) & UInt64(0xFFFF))
    var last_loss = 0.0
    # episode-return tracking (per env running return + closed-episode log) so
    # the batched driver reports avg_return like the single-env driver. The
    # close condition mirrors `record_outcome`'s (done OR len >= max_ep_steps).
    var ep_returns = List[Float64]()
    var per_ret = List[Float64]()
    var ep_steps = List[Int]()
    for _ in range(N_ENVS):
        per_ret.append(0.0)
        ep_steps.append(0)

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=seed)

    for it in range(iterations):
        # ── 1. batched Gumbel search over the LIVE env obs (no H2D in) ──
        var obs_t = LayoutTensor[
            DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
        ](env.obs_ptr())
        planner.search_gpu[
            MZRepGPU[OBS, LATENT, REP],
            MZDynGPU[LATENT, ACT, BINS, DYN],
            MZPredGPU[LATENT, ACT, BINS, PRED],
        ](
            ctx, rep_a, dyn_a, pred_a, obs_t,
            apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed,
        )
        mcts_seed += UInt32(1)
        ctx.enqueue_copy(h_pol, planner.policies_view())
        ctx.enqueue_copy(h_val, planner.root_value_view())
        ctx.synchronize()

        # ── 2. sample actions (host); record obs (device→device) + metadata ──
        var temp = visit_temperature(it, temperature_decay_steps)
        for e in range(N_ENVS):
            h_act[e] = Scalar[DT](_sample_action(h_pol, e * ACT, ACT, temp, rng))
        # record_obs_meta enqueues the obs store kernel reading the ROOT obs
        # (env.obs_ptr()); stream order keeps it before the step kernel below.
        var obs_view = DeviceBuffer[DT](
            ctx, env.obs_ptr(), N_ENVS * OBS, owning=False
        )
        rb.record_obs_meta(obs_view, h_act, h_pol, h_val)

        # ── 3. H2D actions → env, step, D2H reward/done/terminated ──
        var act_view = DeviceBuffer[DT](
            ctx, env.action_ptr(), N_ENVS, owning=False
        )
        ctx.enqueue_copy(act_view, h_act)
        env.step_batch[N_ENVS](ctx=ctx, rng_seed=seed + UInt64(it + 1))
        var rew_view = DeviceBuffer[DT](
            ctx, env.reward_ptr(), N_ENVS, owning=False
        )
        var done_view = DeviceBuffer[DT](
            ctx, env.done_ptr(), N_ENVS, owning=False
        )
        var term_view = DeviceBuffer[DT](
            ctx, env.terminated_ptr(), N_ENVS, owning=False
        )
        ctx.enqueue_copy(h_rew, rew_view)
        ctx.enqueue_copy(h_done, done_view)
        ctx.enqueue_copy(h_term, term_view)
        ctx.synchronize()

        # ── 4. accumulate returns; write outcomes; close finished episodes ──
        for e in range(N_ENVS):
            per_ret[e] += Float64(h_rew[e])
            ep_steps[e] += 1
            if h_done[e] > Scalar[DT](0.5) or ep_steps[e] >= max_ep_steps:
                ep_returns.append(per_ret[e])
                per_ret[e] = 0.0
                ep_steps[e] = 0
        rb.record_outcome(h_rew, h_done, h_term, max_ep_steps)
        env.selective_reset_batch[N_ENVS](
            ctx=ctx, rng_seed=seed + UInt64(it + 1)
        )

        var trained = rb.num_steps() >= learning_starts and rb.num_episodes() > 0

        # ── 5. train: gather obs0 device→device into scratch.d_obs0, unroll ──
        if trained:
            var d_obs0_buf = scratch.d_obs0.value()
            for _ in range(train_per_iter):
                rb.sample_training_batch_dev[B, K, N](
                    gamma, d_obs0_buf, t_act, t_pol, t_val, t_rew
                )
                last_loss = Float64(
                    mz_unroll_train_step_gpu[
                        REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS,
                        obs_on_device=True,
                    ](
                        ctx, rep, dyn, pred, orep, odyn, opred,
                        scratch,
                        t_obs0_dummy, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef,
                        loss_parts=l_parts,
                    )
                )
                train_steps += 1
                if (
                    target_sync_interval > 0
                    and train_steps % target_sync_interval == 0
                ):
                    hard_copy_params["gpu", M=REP](rep, rep_t, ctx)
                    hard_copy_params["gpu", M=DYN](dyn, dyn_t, ctx)
                    hard_copy_params["gpu", M=PRED](pred, pred_t, ctx)

        if (
            logger
            and diag_every > 0
            and trained
            and (it + 1) % diag_every == 0
        ):
            _mz_emit_batch_diag[REP, PRED, B, OBS, ACT, LATENT, BINS, L](
                ctx, rep, pred, scratch.d_obs0.value(), d_diag_z, d_diag_pred,
                h_diag_pred, t_pol, t_val, l_parts,
                v_min, v_max, last_loss, it + 1, logger.value(),
            )

        # ── high-coverage batched reanalyze: re-target `reanalyze_batch` stored
        #    positions with the CURRENT net per trigger, in chunks of N_ENVS (the
        #    planner's root width). Each chunk gathers its obs device→device (one
        #    kernel, no per-position sync), runs one batched Gumbel search, and
        #    writes the fresh root policy + value back in place — the n-step
        #    targets pick them up on the next `sample_training_batch_dev`. Lifting
        #    `reanalyze_batch` from N_ENVS toward B is the EfficientZero-style
        #    coverage lever (a large fraction of each training batch fresh). ──
        if (
            reanalyze_every > 0
            and trained
            and (it + 1) % reanalyze_every == 0
        ):
            # target_sync_interval == 0 ⇒ live-net reanalyze: refresh the target
            # to the current weights now (search through rep_ta then == rep_a).
            if target_sync_interval == 0:
                hard_copy_params["gpu", M=REP](rep, rep_t, ctx)
                hard_copy_params["gpu", M=DYN](dyn, dyn_t, ctx)
                hard_copy_params["gpu", M=PRED](pred, pred_t, ctx)
            var n_chunks = reanalyze_batch // N_ENVS
            if n_chunks < 1:
                n_chunks = 1
            for _c in range(n_chunks):
                var rpos = rb.sample_reanalyze_chunk[N_ENVS](d_reana)
                var rpos_e = rpos[0].copy()
                var rpos_o = rpos[1].copy()
                var reana_t = LayoutTensor[
                    DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
                ](mptr(d_reana.unsafe_ptr()))
                planner.search_gpu[
                    MZRepGPU[OBS, LATENT, REP],
                    MZDynGPU[LATENT, ACT, BINS, DYN],
                    MZPredGPU[LATENT, ACT, BINS, PRED],
                ](
                    ctx, rep_ta, dyn_ta, pred_ta, reana_t,
                    apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed,
                )
                mcts_seed += UInt32(1)
                ctx.enqueue_copy(h_pol, planner.policies_view())
                ctx.enqueue_copy(h_val, planner.root_value_view())
                ctx.synchronize()
                for e in range(N_ENVS):
                    rb.update_targets(
                        rpos_e[e], rpos_o[e],
                        h_pol + (e * ACT), h_val[e],
                    )

        if eval_every > 0 and eval_env and (it + 1) % eval_every == 0:
            # `eval_horizon` (if set) is the per-eval step CAP; else a generous
            # default that early-exits once `eval_episodes` games complete.
            var cap = (
                eval_horizon if eval_horizon > 0
                else max_ep_steps * (eval_episodes + 1)
            )
            var avg = _eval_greedy_batched[
                BENV, REP, DYN, PRED, N_ENVS, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, eval_env.value(), eval_planner, rep_a, dyn_a, pred_a,
                eval_episodes, cap, eval_seed,
            )
            eval_seed += UInt32(cap + 1)
            print("  [eval] step", it + 1, "greedy_return", avg)
            if logger:
                logger.value()[].log_scalar(String("eval_return"), avg, it + 1)

        if verbose and (it + 1) % 500 == 0:
            print(
                "iter", it + 1, "env_steps", (it + 1) * N_ENVS,
                "loss", last_loss, "eps", rb.num_episodes(),
                "completed", len(ep_returns),
                "avg_return(10)", _avg_last_n(ep_returns, 10),
            )

        if (
            logger
            and report_every > 0
            and trained
            and (it + 1) % report_every == 0
        ):
            var rn = List[String]()
            var rv = List[Float64]()
            rn.append(String("avg_return"))
            rv.append(_avg_last_n(ep_returns, 10))
            rn.append(String("episodes")); rv.append(Float64(rb.num_episodes()))
            rn.append(String("replay_size")); rv.append(Float64(rb.num_steps()))
            logger.value()[].log_scalars(rn, rv, it + 1)

    t_act.free(); t_pol.free(); t_val.free(); t_rew.free(); t_obs0_dummy.free()
    h_pol.free(); h_val.free(); h_act.free()
    h_rew.free(); h_done.free(); h_term.free(); l_parts.free()
    h_diag_pred.free()
    # keep the target nets (held only via UnsafePointer in the adapters) alive.
    _ = rep_t^
    _ = dyn_t^
    _ = pred_t^
    return last_loss


def _eval_greedy_batched[
    BENV: BatchedEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_ENVS: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    NUM_SIMS: Int,
](
    ctx: DeviceContext,
    eval_env: UnsafePointer[BENV, MutAnyOrigin],
    mut planner: GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ],
    mut rep_a: MZRepGPU[OBS, LATENT, REP],
    mut dyn_a: MZDynGPU[LATENT, ACT, BINS, DYN],
    mut pred_a: MZPredGPU[LATENT, ACT, BINS, PRED],
    target_episodes: Int,
    max_steps: Int,
    rng_seed: UInt32,
) raises -> Float64:
    """Greedy batched eval: reset the eval env and take the argmax of the Gumbel
    improved policy over all ``N_ENVS`` lanes until ``target_episodes`` episodes
    have **completed** (across lanes), then return their mean return. ``max_steps``
    caps the rollout so a long-rallying policy can't hang it; if the cap is hit
    before the target, the mean of whatever completed is returned (falling back to
    the mean running per-env return only if *nothing* completed). Lanes that finish
    early are `selective_reset` and keep contributing more episodes, so a few lanes
    quickly yield many games. Runs on its own env instance + planner, so it never
    touches the training env, replay, or RNG."""
    var h_pol = _a(N_ENVS * ACT)
    var h_act = _a(N_ENVS)
    var h_rew = _a(N_ENVS)
    var h_done = _a(N_ENVS)
    var run_ret = List[Float64]()
    for _ in range(N_ENVS):
        run_ret.append(0.0)
    var done_sum = 0.0
    var done_cnt = 0
    var seed_i = rng_seed

    eval_env[].reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(rng_seed))
    var step_i = 0
    while done_cnt < target_episodes and step_i < max_steps:
        step_i += 1
        var obs_t = LayoutTensor[
            DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
        ](eval_env[].obs_ptr())
        planner.search_gpu[
            MZRepGPU[OBS, LATENT, REP],
            MZDynGPU[LATENT, ACT, BINS, DYN],
            MZPredGPU[LATENT, ACT, BINS, PRED],
        ](
            ctx, rep_a, dyn_a, pred_a, obs_t,
            apply_legal=False, k_actual=MAX_K, rng_seed=seed_i,
        )
        seed_i += UInt32(1)
        ctx.enqueue_copy(h_pol, planner.policies_view())
        ctx.synchronize()
        for e in range(N_ENVS):
            var best = 0
            for a in range(1, ACT):
                if Float64(h_pol[e * ACT + a]) > Float64(h_pol[e * ACT + best]):
                    best = a
            h_act[e] = Scalar[DT](best)
        var act_view = DeviceBuffer[DT](
            ctx, eval_env[].action_ptr(), N_ENVS, owning=False
        )
        ctx.enqueue_copy(act_view, h_act)
        eval_env[].step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(seed_i))
        var rew_view = DeviceBuffer[DT](
            ctx, eval_env[].reward_ptr(), N_ENVS, owning=False
        )
        var done_view = DeviceBuffer[DT](
            ctx, eval_env[].done_ptr(), N_ENVS, owning=False
        )
        ctx.enqueue_copy(h_rew, rew_view)
        ctx.enqueue_copy(h_done, done_view)
        ctx.synchronize()
        for e in range(N_ENVS):
            run_ret[e] += Float64(h_rew[e])
            if h_done[e] > Scalar[DT](0.5):
                done_sum += run_ret[e]
                done_cnt += 1
                run_ret[e] = 0.0
        eval_env[].selective_reset_batch[N_ENVS](
            ctx=ctx, rng_seed=UInt64(seed_i)
        )

    # Mean over completed episodes; if the cap was hit with none completed, fall
    # back to the mean running per-env return.
    var running = 0.0
    for e in range(N_ENVS):
        running += run_ret[e]
    var out = (
        done_sum / Float64(done_cnt)
        if done_cnt > 0
        else running / Float64(N_ENVS)
    )
    h_pol.free(); h_act.free(); h_rew.free(); h_done.free()
    return out
