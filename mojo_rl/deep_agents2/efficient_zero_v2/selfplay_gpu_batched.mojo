"""EfficientZeroV2 BATCHED discrete self-play driver (GPU nets × N CPU envs).

Stage 6c-3 — the Atari-parity driver. The batched sibling of `selfplay_gpu.mojo`
(single-env), assembling the full EfficientZero-V2 recipe:

  * **N_ENVS CPU-emulated envs** stepped in lockstep through a
    `BatchedCpuDiscreteEnv` (host obs/action/reward/done) — the RGB-96 Atari
    `AtariEnv[2]` is CPU-only, so unlike the GPU MuZero batched driver only the
    search-input obs is H2D'd and the improved policy/value are D2H'd; actions
    and rewards stay host. Random no-op starts (`noop_max`) live in the env.
  * **Batched Gumbel search** over the on-device h/g/f nets (one launch covers
    all N_ENVS roots) via the MuZero GPU adapters.
  * **Prioritized replay** (`PrioritizedMCTSSequenceReplay`): each train step
    samples ∝ priorityᵅ, the unroll is weighted by per-sample IS weights, and
    fresh value-error priorities are written back (α=β=1, atari.yaml).
  * **EZv2 unroll train step** (MuZero BPTT + SimSiam consistency) on the
    resident GPU nets with any `Optimizer` (SGD for Atari) + warmup→const LR.
  * **Reanalyze** through the LIVE nets (like the single-env EZv2 driver — no
    lagging target nets, sidestepping the BatchNorm-running-stat hard-copy bug);
    `reanalyze_batch ≈ B` + `reanalyze_every = 1` gives the EZ ratio-1.0 regime.
    Runs on a SEPARATE wider planner (`REANA_W` roots/search) so a ratio-1.0
    re-target is `ceil(reanalyze_batch/REANA_W)` wide searches, not
    `reanalyze_batch/N_ENVS` narrow N_ENVS-root ones — the bottleneck fix (far
    fewer launches/syncs, real GPU occupancy). `REANA_W` defaults to `N_ENVS`
    (back-compat); set it to e.g. 64 for the Atari run.
  * **UTD 1:1** (`train_per_iter = N_ENVS` default) + batched greedy eval.

DEVIATION (documented, revisit if Pong diverges): the n-step value target
bootstraps from the **search-root value** (what the planner returns), whereas
EZ bootstraps the **value head** at s_{k+5} (`efficient_inference only_value`).
Reanalyze refreshes the stored root value with the current net, which is the
main coverage lever; the head-vs-root bootstrap gap is logged for Stage 8.

Run (GPU env required): see `tests/deep_agents2/test_ezv2_atari_batched_smoke.mojo`.
"""

from std.math import exp, log
from std.memory import alloc
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.core.optimizer import Optimizer
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.planners.tree_search import GumbelGPUMCTS, SinglePlayer

from ..training.batched_env import BatchedEnv
from .blocks import ezv2_unroll_train_step_gpu
from .unroll_scratch import EZV2UnrollScratch
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.prioritized_sequence_replay_mcts import PrioritizedMCTSSequenceReplay
from ..zero.temperature import visit_temperature
from ..muzero.selfplay_gpu_batched import _sample_action


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


def _ai(n: Int) -> UnsafePointer[Int, MutAnyOrigin]:
    return alloc[Int](n)


def run_ezv2_gumbel_selfplay_gpu_batched[
    ENV: BatchedEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
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
    REANA_W: Int = N_ENVS,
    OBS_STORE_DT: DType = DT,
    O: Optimizer = Adam,
    L: Logger = NoOpLogger,
](
    ctx: DeviceContext,
    mut env: ENV,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: O,
    mut odyn: O,
    mut opred: O,
    mut oproj: O,
    mut opredh: O,
    iterations: Int,
    learning_starts: Int = 256,         # in STORED STEPS
    train_per_iter: Int = N_ENVS,       # UTD 1:1 (grad steps == env steps)
    lr: Scalar[DT] = Scalar[DT](0.0),
    lr_warmup_iters: Int = 0,
    gamma: Scalar[DT] = Scalar[DT](0.997),
    v_min: Scalar[DT] = Scalar[DT](-300.0),
    v_max: Scalar[DT] = Scalar[DT](300.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 27000,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    temperature_decay_steps: Int = 0,
    reanalyze_every: Int = 0,
    reanalyze_batch: Int = B,
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    eval_horizon: Int = 0,
    eval_env: Optional[UnsafePointer[ENV, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    report_every: Int = 0,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    verbose: Bool = False,
) raises -> Float64:
    comptime assert ENV.OBS_DIM == OBS, "batched EZv2: ENV.OBS_DIM must equal OBS"

    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    # Reanalyze runs a SEPARATE, WIDER planner (REANA_W roots/search) so the
    # ratio-1.0 re-target covers `reanalyze_batch` positions in
    # ceil(reanalyze_batch/REANA_W) wide searches instead of reanalyze_batch/N_ENVS
    # narrow N_ENVS-root ones — far fewer kernel launches + syncs and much better
    # GPU occupancy. Reuses the SAME nets/adapters (forward["gpu", B] is width-
    # generic; the nets' activation buffers lazy-grow to REANA_W on first use).
    # REANA_W == N_ENVS (default) is the same monomorphization as `planner` (no
    # extra compile), just a second runtime instance.
    var reana_planner = GumbelGPUMCTS[
        REANA_W, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    # Device-resident obs ring (uint8 pixels): the training obs slab is gathered
    # on-device straight into scratch.d_obs, so neither the host dequant-build nor
    # the ~680 MB/step slab H2D happen (the pixel-obs bottleneck). PER + targets
    # stay on host.
    var rb = PrioritizedMCTSSequenceReplay[OBS, ACT, CAP, OBS_STORE_DT](
        ctx, seed=seed ^ UInt64(0xABCDEF),
        alpha=Scalar[DT](1.0), beta=Scalar[DT](1.0),
    )

    # search-input device obs (H2D'd from the CPU env each step) + host mirrors
    var d_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_pol = _a(N_ENVS * ACT)
    var h_val = _a(N_ENVS)

    # training batch slabs (time-major). obs is NOT staged on host any more — the
    # device gather fills scratch.d_obs directly; only the small label slabs are
    # host (H2D'd by the train step). The per-(k,b) ring-slot index array drives
    # the gather.
    var t_act = _a(K * B)
    var t_pol = _a((K + 1) * B * ACT)
    var t_val = _a((K + 1) * B)
    var t_rew = _a(K * B)
    var t_cmask = _a(K * B)
    var t_isw = _a(B)              # PER importance-sampling weights
    var t_prio = _a(B)            # PER value-error priorities (writeback)
    var t_slots = _ai(B)         # PER sampled ring slots
    var l_parts = _a(4)
    var t_obs_dummy = _a(1)      # unused obs_seq arg (obs_on_device=True)
    # gather slot index arrays: training [(K+1)*B] + reanalyze [REANA_W]
    var h_obs_slots = alloc[Int32]((K + 1) * B)
    var d_obs_slots = ctx.enqueue_create_buffer[DType.int32]((K + 1) * B)
    var h_reana_slots = alloc[Int32](REANA_W)
    var d_reana_slots = ctx.enqueue_create_buffer[DType.int32](REANA_W)

    var train_scratch = EZV2UnrollScratch[
        B, K, OBS, ACT, LATENT, BINS, PROJM.OUT_DIM
    ].make(ctx)

    # reanalyze scratch — REANA_W wide. obs gathered on-device into d_reana (no
    # host read_obs / H2D); improved policy/value D2H'd into host mirrors.
    var d_reana = ctx.enqueue_create_buffer[DT](REANA_W * OBS)
    var h_pol_w = _a(REANA_W * ACT)
    var h_val_w = _a(REANA_W)

    # per-env episode accumulators
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
    var eval_seed = UInt32((seed ^ UInt64(0xE7A1B2C3)) & UInt64(0xFFFF))
    var last_loss = 0.0
    var train_steps = 0
    var ep_returns = List[Float64]()

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=seed)

    for it in range(iterations):
        # ── 1. H2D the CPU env's live obs, batched Gumbel search ──
        ctx.enqueue_copy(d_obs, env.obs_ptr())
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
            MutAnyOrigin](mptr(d_obs.unsafe_ptr()))
        planner.search_gpu[
            MZRepGPU[OBS, LATENT, REP],
            MZDynGPU[LATENT, ACT, BINS, DYN],
            MZPredGPU[LATENT, ACT, BINS, PRED],
        ](ctx, rep_a, dyn_a, pred_a, obs_t,
          apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
        mcts_seed += UInt32(1)

        # ── 2. D2H improved policy + root value (obs already host) ──
        ctx.enqueue_copy(h_pol, planner.policies_view())
        ctx.enqueue_copy(h_val, planner.root_value_view())
        ctx.synchronize()

        # ── 3. per env: sample, record the labelled step, stage the action ──
        var temp = visit_temperature(it, temperature_decay_steps)
        var obs_host = env.obs_ptr()
        var act_host = env.action_ptr()
        for e in range(N_ENVS):
            var action = _sample_action(h_pol, e * ACT, ACT, temp, rng)
            for j in range(OBS):
                e_obs[e].append(obs_host[e * OBS + j])
            e_act[e].append(Scalar[DT](action))
            for a in range(ACT):
                e_pol[e].append(h_pol[e * ACT + a])
                e_legal[e].append(Scalar[DT](1.0))
            e_val[e].append(h_val[e])
            e_tp[e].append(Scalar[DT](0.0))
            act_host[e] = Scalar[DT](action)

        # ── 4. step the CPU envs (host action → host reward/done/term) ──
        env.step_batch[N_ENVS](ctx=ctx, rng_seed=seed + UInt64(it + 1))
        var rew_host = env.reward_ptr()
        var done_host = env.done_ptr()
        var term_host = env.terminated_ptr()

        # ── 5. accumulate, store + reset finished episodes ──
        for e in range(N_ENVS):
            e_rew[e].append(rew_host[e])
            ep_return[e] += Float64(rew_host[e])
            ep_len[e] += 1
            var done = done_host[e] > Scalar[DT](0.5)
            var terminated = term_host[e] > Scalar[DT](0.5)
            if done or ep_len[e] >= max_ep_steps:
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

        env.selective_reset_batch[N_ENVS](ctx=ctx, rng_seed=seed + UInt64(it + 1))

        var trained = rb.num_steps() >= learning_starts and rb.num_episodes() > 0

        # ── 6. train (prioritized sample → weighted EZv2 unroll → writeback) ──
        if trained:
            if lr > Scalar[DT](0.0):
                var tstep = train_steps
                var cur_lr = lr
                if lr_warmup_iters > 0 and tstep < lr_warmup_iters:
                    cur_lr = lr * Scalar[DT](
                        Float64(tstep + 1) / Float64(lr_warmup_iters)
                    )
                orep.set_lr(cur_lr); odyn.set_lr(cur_lr); opred.set_lr(cur_lr)
                oproj.set_lr(cur_lr); opredh.set_lr(cur_lr)
            for _ in range(train_per_iter):
                # CPU draws prioritized slots + targets, then gathers the obs
                # slab on-device straight into scratch.d_obs (no host build/H2D).
                rb.sample_training_batch_seq_per_gpu[B, K, N](
                    ctx, gamma, train_scratch.d_obs.value(),
                    d_obs_slots, mptr(h_obs_slots),
                    t_act, t_pol, t_val, t_rew, t_isw, t_slots,
                    cons_mask=t_cmask,
                )
                last_loss = Float64(
                    ezv2_unroll_train_step_gpu[
                        REP, DYN, PRED, PROJM, PREDH,
                        B, K, OBS, ACT, LATENT, BINS,
                    ](
                        ctx, train_scratch, rep, dyn, pred, proj, predh,
                        orep, odyn, opred, oproj, opredh,
                        t_obs_dummy, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef, consistency_coef,
                        cons_mask=t_cmask, loss_parts=l_parts,
                        is_weights=t_isw, out_prio=t_prio,
                        obs_on_device=True,
                    )
                )
                rb.update_priorities(t_slots, t_prio, B)
                train_steps += 1

        # ── per-batch loss diagnostics → logger ──
        if logger and diag_every > 0 and trained and (it + 1) % diag_every == 0:
            var dn = List[String]()
            var dv = List[Float64]()
            dn.append(String("loss")); dv.append(last_loss)
            dn.append(String("loss_policy")); dv.append(Float64(l_parts[0]))
            dn.append(String("loss_value")); dv.append(Float64(l_parts[1]))
            dn.append(String("loss_reward")); dv.append(Float64(l_parts[2]))
            dn.append(String("loss_consistency")); dv.append(Float64(l_parts[3]))
            logger.value()[].log_scalars(dn, dv, it + 1)

        # ── reanalyze through the LIVE nets — WIDE searches (ratio≈1.0 when
        #    reanalyze_batch≈B). Each chunk re-targets REANA_W positions in ONE
        #    search + ONE sync, so the whole reanalyze_batch needs only
        #    ceil(reanalyze_batch/REANA_W) wide searches. ──
        if reanalyze_every > 0 and trained and (it + 1) % reanalyze_every == 0:
            var n_chunks = (reanalyze_batch + REANA_W - 1) // REANA_W
            if n_chunks < 1:
                n_chunks = 1
            for _c in range(n_chunks):
                var rpos_e = List[Int]()
                var rpos_o = List[Int]()
                for _ in range(REANA_W):
                    var rpos = rb.sample_position()
                    rpos_e.append(rpos[0])
                    rpos_o.append(rpos[1])
                # gather the REANA_W positions' obs on-device into d_reana
                rb.gather_obs_for_positions[REANA_W](
                    ctx, d_reana, d_reana_slots, mptr(h_reana_slots),
                    rpos_e, rpos_o,
                )
                var reana_t = LayoutTensor[DT, Layout.row_major(REANA_W, OBS),
                    MutAnyOrigin](mptr(d_reana.unsafe_ptr()))
                reana_planner.search_gpu[
                    MZRepGPU[OBS, LATENT, REP],
                    MZDynGPU[LATENT, ACT, BINS, DYN],
                    MZPredGPU[LATENT, ACT, BINS, PRED],
                ](ctx, rep_a, dyn_a, pred_a, reana_t,
                  apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
                mcts_seed += UInt32(1)
                ctx.enqueue_copy(h_pol_w, reana_planner.policies_view())
                ctx.enqueue_copy(h_val_w, reana_planner.root_value_view())
                ctx.synchronize()
                for e in range(REANA_W):
                    rb.update_targets(
                        rpos_e[e], rpos_o[e], h_pol_w + (e * ACT), h_val_w[e]
                    )

        # ── batched greedy eval (CPU env) ──
        if eval_every > 0 and eval_env and (it + 1) % eval_every == 0:
            var cap = (
                eval_horizon if eval_horizon > 0
                else max_ep_steps * (eval_episodes + 1)
            )
            var avg = _ez_eval_greedy_cpu_batched[
                ENV, REP, DYN, PRED, N_ENVS, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, eval_env.value(), planner, rep_a, dyn_a, pred_a,
                d_obs, h_pol, eval_episodes, cap, eval_seed,
            )
            eval_seed += UInt32(cap + 1)
            print("  [eval] step", it + 1, "greedy_return", avg)
            if logger:
                logger.value()[].log_scalar(String("eval_return"), avg, it + 1)

        if verbose and (it + 1) % 100 == 0:
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
            print("iter", it + 1, "env_steps", (it + 1) * N_ENVS,
                  "loss", last_loss, "eps", rb.num_episodes(),
                  "avg_return(10)", avg)

        if logger and report_every > 0 and trained and (it + 1) % report_every == 0:
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
            rn.append(String("episodes")); rv.append(Float64(rb.num_episodes()))
            rn.append(String("replay_size")); rv.append(Float64(rb.num_steps()))
            logger.value()[].log_scalars(rn, rv, it + 1)

    t_act.free(); t_pol.free(); t_val.free(); t_rew.free()
    t_cmask.free(); t_isw.free(); t_prio.free(); t_slots.free(); l_parts.free()
    t_obs_dummy.free(); h_obs_slots.free(); h_reana_slots.free()
    h_pol.free(); h_val.free()
    h_pol_w.free(); h_val_w.free()
    return last_loss


def _ez_eval_greedy_cpu_batched[
    ENV: BatchedEnv,
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
    eval_env: UnsafePointer[ENV, MutAnyOrigin],
    mut planner: GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ],
    mut rep_a: MZRepGPU[OBS, LATENT, REP],
    mut dyn_a: MZDynGPU[LATENT, ACT, BINS, DYN],
    mut pred_a: MZPredGPU[LATENT, ACT, BINS, PRED],
    d_obs: DeviceBuffer[DT],
    h_pol: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_episodes: Int,
    max_steps: Int,
    rng_seed: UInt32,
) raises -> Float64:
    """Greedy (argmax improved-policy) batched eval over the CPU eval env. Runs
    the N_ENVS eval envs until `target_episodes` complete (or `max_steps`),
    summing the FIRST completed return per env. Read-only w.r.t. the trained
    nets (forward only)."""
    eval_env[].reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(rng_seed))
    var ret_sum = 0.0
    var done_count = 0
    var cur_ret = List[Float64]()
    var counted = List[Bool]()
    for _ in range(N_ENVS):
        cur_ret.append(0.0)
        counted.append(False)
    var es = rng_seed
    for _step in range(max_steps):
        if done_count >= target_episodes:
            break
        ctx.enqueue_copy(d_obs, eval_env[].obs_ptr())
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
            MutAnyOrigin](mptr(d_obs.unsafe_ptr()))
        planner.search_gpu[
            MZRepGPU[OBS, LATENT, REP],
            MZDynGPU[LATENT, ACT, BINS, DYN],
            MZPredGPU[LATENT, ACT, BINS, PRED],
        ](ctx, rep_a, dyn_a, pred_a, obs_t,
          apply_legal=False, k_actual=MAX_K, rng_seed=es)
        es += UInt32(1)
        ctx.enqueue_copy(h_pol, planner.policies_view())
        ctx.synchronize()
        var act_host = eval_env[].action_ptr()
        for e in range(N_ENVS):
            var best = 0
            for a in range(1, ACT):
                if Float64(h_pol[e * ACT + a]) > Float64(h_pol[e * ACT + best]):
                    best = a
            act_host[e] = Scalar[DT](best)
        eval_env[].step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(es))
        var rew_host = eval_env[].reward_ptr()
        var done_host = eval_env[].done_ptr()
        for e in range(N_ENVS):
            if not counted[e]:
                cur_ret[e] += Float64(rew_host[e])
                if done_host[e] > Scalar[DT](0.5):
                    ret_sum += cur_ret[e]
                    counted[e] = True
                    done_count += 1
        eval_env[].selective_reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(es))
    var n = done_count if done_count > 0 else 1
    return ret_sum / Float64(n)
