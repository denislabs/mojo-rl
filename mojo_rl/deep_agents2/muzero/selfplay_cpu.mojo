"""MuZero single-player self-play driver (CPU) — the CartPole lighthouse loop.

Ties every Phase-B piece together on one device (CPU), reusing the validated
CPU MCTS (learned dynamics) and CPU BPTT unroll — no GPU/CPU param sync:

  per env step:  obs → GenericCPUMCTS.search (MZ CPU adapters) → visit policy π_t
                 + root value v_t → sample a_t ∼ π_t → env.step → r_{t+1}, done →
                 append (o_t, a_t, r_{t+1}, π_t, v_t, to_play=0) to the episode
  on done:       store_episode → MCTSSequenceReplay
  every step≥learning_starts: sample_training_batch → mz_unroll_train_step_cpu

Single-player (CartPole): ``to_play`` is always 0, so the n-step sign flips are
no-ops. Actions are sampled proportional to the MCTS visit counts (exploration);
the root value stored is the search value (`mcts.root_value()`), the MuZero
bootstrap target. Returns the last training loss.
"""

from std.memory import alloc

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SinglePlayer,
)

from .blocks import mz_unroll_train_step_cpu
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def run_muzero_selfplay_cpu[
    ENV: BoxDiscreteActionEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
    BATCH_SIMS: Int = 8,
    VIRTUAL_LOSS: Int = 3,
](
    mut env: ENV,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = 1,
    gamma: Scalar[DT] = Scalar[DT](0.997),
    v_min: Scalar[DT] = Scalar[DT](-10.0),
    v_max: Scalar[DT] = Scalar[DT](10.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 500,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    verbose: Bool = False,
) raises -> Float64:
    # PUCT defaults (c_base=19652, c_init=1.25) — matches legacy
    # MuZeroMLPConfig.PUCT. NOTE: ``MuZeroPUCT[1.25]`` would bind c_base=1.25
    # (the *first* positional param), not c_init, badly distorting exploration.
    # BATCH_SIMS=8 + VIRTUAL_LOSS=3 diversify root exploration to counter the
    # spiky DirichletNoise[0.25,0.25] prior (legacy batch_sims=8 / virtual_loss=3).
    var mcts = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
        BATCH_SIMS, VIRTUAL_LOSS,
    ](gamma=Float64(gamma))
    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    var rep_a = MZRepCPU[OBS, LATENT, REP](net=UnsafePointer(to=rep))
    var dyn_a = MZDynCPU[LATENT, ACT, BINS, DYN](
        net=UnsafePointer(to=dyn), v_min=v_min, v_max=v_max
    )
    var pred_a = MZPredCPU[LATENT, ACT, BINS, PRED](
        net=UnsafePointer(to=pred), v_min=v_min, v_max=v_max
    )

    # training batch slabs (time-major), allocated once
    var t_obs0 = _a(B * OBS)
    var t_act = _a(K * B)
    var t_pol = _a((K + 1) * B * ACT)
    var t_val = _a((K + 1) * B)
    var t_rew = _a(K * B)

    # episode accumulation buffers
    var e_obs = List[Scalar[DT]]()
    var e_act = List[Scalar[DT]]()
    var e_rew = List[Scalar[DT]]()
    var e_pol = List[Scalar[DT]]()
    var e_val = List[Scalar[DT]]()
    var e_tp = List[Scalar[DT]]()
    var e_legal = List[Scalar[DT]]()    # all-legal (single-player); reanalyze ch.
    var ep_len = 0

    var rng = seed ^ UInt64(0x123456789)
    var last_loss = 0.0
    var ep_returns = List[Float64]()

    var cur = env.reset_obs_list()
    var cur_f = List[Float64]()
    for j in range(OBS):
        cur_f.append(Float64(cur[j]))
    var ep_return = 0.0

    for it in range(iterations):
        # ── search ──
        var policy = mcts.search[type_of(rep_a), type_of(dyn_a), type_of(pred_a)](
            rep_a, dyn_a, pred_a, cur_f, add_noise=True
        )
        var root_v = mcts.root_value()

        # ── sample action ∝ visit policy ──
        rng = rng ^ (rng << 13); rng = rng ^ (rng >> 7); rng = rng ^ (rng << 17)
        var r = Float64(rng % UInt64(1_000_000)) / 1_000_000.0
        var cum = 0.0
        var action = ACT - 1
        for a in range(ACT):
            cum += policy[a]
            if r <= cum:
                action = a
                break

        # ── record step (o_t, a_t, π_t, v_t, to_play=0) ──
        for j in range(OBS):
            e_obs.append(Scalar[DT](cur_f[j]))
        e_act.append(Scalar[DT](action))
        for a in range(ACT):
            e_pol.append(Scalar[DT](policy[a]))
            e_legal.append(Scalar[DT](1.0))     # CartPole: every action legal
        e_val.append(Scalar[DT](root_v))
        e_tp.append(Scalar[DT](0.0))

        # ── env step → r_{t+1}, done ──
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
            rb.store_episode(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_obs.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_act.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_rew.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_pol.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_val.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_tp.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_legal.unsafe_ptr()
                ),
                ep_len,
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

        # ── train ──
        if it >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch[B, K, N](
                    gamma, t_obs0, t_act, t_pol, t_val, t_rew
                )
                last_loss = Float64(
                    mz_unroll_train_step_cpu[
                        REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS
                    ](
                        rep, dyn, pred, orep, odyn, opred,
                        t_obs0, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef,
                    )
                )

        # ── periodic GREEDY eval (noise off, argmax visits) ──
        # Training return above is exploratory (∝-visit sampling + root
        # Dirichlet noise) and understates the policy. This measures the
        # deterministic policy. Interrupts the in-progress training episode,
        # so keep ``eval_every`` large.
        if eval_every > 0 and (it + 1) % eval_every == 0:
            var eval_sum = 0.0
            for _ in range(eval_episodes):
                var eo = env.reset_obs_list()
                var eo_f = List[Float64]()
                for j in range(OBS):
                    eo_f.append(Float64(eo[j]))
                var eret = 0.0
                for _step in range(max_ep_steps):
                    var ep = mcts.search[
                        type_of(rep_a), type_of(dyn_a), type_of(pred_a)
                    ](rep_a, dyn_a, pred_a, eo_f, add_noise=False)
                    var best = 0
                    for a in range(1, ACT):
                        if ep[a] > ep[best]:
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
            # restart a clean training episode (eval clobbered ``env``)
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

    t_obs0.free(); t_act.free(); t_pol.free(); t_val.free(); t_rew.free()
    return last_loss
