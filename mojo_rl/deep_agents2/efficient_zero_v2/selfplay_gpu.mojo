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

from std.memory import alloc
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import dtype
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.planners.tree_search import GumbelGPUMCTS, SinglePlayer

from .blocks import ezv2_unroll_train_step_gpu
from .unroll_scratch import EZV2UnrollScratch
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


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
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    verbose: Bool = False,
) raises -> Float64:
    comptime N_ENVS = 1

    # ── GPU Gumbel planner + on-device net adapters ──
    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max)
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

    # ── persistent GPU train-step scratch (allocated once, reused per step) ──
    var train_scratch = EZV2UnrollScratch[
        B, K, OBS, ACT, LATENT, BINS, PROJM.OUT_DIM
    ].make(ctx)

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
        var obs_t = LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS),
            MutAnyOrigin](rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                d_obs.unsafe_ptr()))
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

        # ── sample an action from the improved policy ──
        rng = rng ^ (rng << 13); rng = rng ^ (rng >> 7); rng = rng ^ (rng << 17)
        var r = Float64(rng % UInt64(1_000_000)) / 1_000_000.0
        var cum = 0.0
        var action = ACT - 1
        for a in range(ACT):
            cum += Float64(h_pol[a])
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
            rb.store_episode(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_obs.unsafe_ptr()),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_act.unsafe_ptr()),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_rew.unsafe_ptr()),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_pol.unsafe_ptr()),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_val.unsafe_ptr()),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_tp.unsafe_ptr()),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_legal.unsafe_ptr()),
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

        # ── GPU train step ──
        if it >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch_seq[B, K, N](
                    gamma, t_obs_seq, t_act, t_pol, t_val, t_rew
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
                    )
                )

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
                    var eobs_t = LayoutTensor[dtype,
                        Layout.row_major(N_ENVS, OBS), MutAnyOrigin](
                            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                                d_obs.unsafe_ptr()))
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

    t_obs_seq.free(); t_act.free(); t_pol.free(); t_val.free(); t_rew.free()
    h_obs.free(); h_pol.free(); h_val.free()
    return last_loss
