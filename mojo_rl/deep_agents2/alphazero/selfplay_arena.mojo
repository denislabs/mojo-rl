"""Arena-gated AlphaZero self-play with symmetry augmentation — full AlphaZero.

Extends `run_alphazero_selfplay` with the two production pieces:

  * **Best/learner split + Arena gating.** A frozen *best* net generates all
    self-play (MCTS expansion + the policy/value priors); a *learner* net trains
    on that data. Every `arena_every` iterations the learner plays the best in
    the Arena (`candidate_winrate_mcts` — both nets at full MCTS strength, both
    colors, random openings); if it wins a clear majority of decisive games
    (`should_promote`) it is copied over the best (`hard_copy_params`) and
    becomes the new generator. The gate runs MCTS (not argmax) because that is
    the condition the winner is used in — self-play generation is always net+MCTS
    — and because it exercises the value head, which argmax ignores. This
    accept/reject loop stops a transiently-worse learner from poisoning the
    self-play distribution.

  * **Symmetry augmentation.** Each recorded `(obs, π)` sample is replicated
    under the board's symmetry group (`AUG`, e.g. TicTacToe's 8 D4 symmetries);
    the scalar value target `z` is symmetry-invariant and copied. Multiplies
    effective data and bakes in the game's invariances.

The passed-in `net` is the *best* and holds the final (best) weights on return.
`AUG = IdentityAugmenter` recovers the un-augmented behaviour.

**Telemetry.** Two `GPUEvaluator` opponents (`OPP1`, `OPP2`, e.g. random +
minimax) and a `Logger` (`L`) can be plugged in. Every `report_every`
iterations the best net is evaluated (both colors) against each enabled
opponent, a one-line progress summary is printed, and the metrics (loss, replay
size, promotions, per-opponent win/draw/loss + win-rate) are flushed to the
logger. With the defaults (`OPP*=RandomOpponent`, `L=NoOpLogger`,
`report_every=0`) this is a no-op and matches the original silent behaviour.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import dtype
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.initializer import Zero, Kaiming
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node, ExternalNode
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.planners.tree_search import (
    GenericGPUMCTS, AlphaGoPUCT, DirichletNoise, SelfPlay,
)

from .loss_ops import AZLossOp
from .arena import candidate_winrate_mcts, should_promote
from .eval import eval_mcts_vs_opponent, EvalResult
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.example_replay import MCTSExampleReplay
from ..zero.symmetries import BoardAugmenter
from ..zero.evaluators import GPUEvaluator, RandomOpponent


@fieldwise_init
struct ArenaRunResult(Copyable, Movable):
    """Summary of an arena-gated run: last mean train loss + how many times the
    learner was accepted over the best."""
    var last_loss: Float64
    var promotions: Int


def _eval_both_colors[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    OPP: GPUEvaluator,
    N_GAMES: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](ctx: DeviceContext, mut net: NET, seed: UInt64) raises -> EvalResult:
    """Aggregate the net's **MCTS** record vs `OPP` over `N_GAMES` games as
    *each* color (P0 then P1), so first-move advantage cancels — the legacy eval
    convention. The agent plays at full search strength (`eval_mcts_vs_opponent`)
    so the numbers reflect the deployed agent, not the bare policy head."""
    var p0 = eval_mcts_vs_opponent[
        ENV, NET, OPP, N_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](ctx, net, agent_player=0, seed=seed)
    var p1 = eval_mcts_vs_opponent[
        ENV, NET, OPP, N_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](ctx, net, agent_player=1, seed=seed + 33333)
    return EvalResult(
        wins=p0.wins + p1.wins,
        draws=p0.draws + p1.draws,
        losses=p0.losses + p1.losses,
    )


def run_alphazero_selfplay_arena[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    AUG: BoardAugmenter,
    N_ENVS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
    ARENA_GAMES: Int = 32,        # arena games per color (comptime: sizes buffers)
    RESULT_IDX: Int = 10,         # vestigial (MCTS arena/eval attribute by
    #                               reward+turn); kept for caller API stability
    MAX_PLIES: Int = 9,
    OPP1: GPUEvaluator = RandomOpponent,   # primary eval opponent (do_eval)
    OPP2: GPUEvaluator = RandomOpponent,   # secondary eval opponent (do_eval2)
    L: Logger = NoOpLogger,                # metrics sink (NoOp = silent)
    EVAL_GAMES: Int = 64,                  # games per color in each periodic eval
](
    ctx: DeviceContext,
    mut net: NET,                 # the BEST net — holds final weights on return
    iterations: Int,
    learning_starts: Int = 0,
    train_per_iter: Int = 1,
    lr: Scalar[DT] = Scalar[DT](0.01),
    seed: UInt64 = 0,
    arena_every: Int = 100,
    arena_open_plies: Int = 2,
    promote_threshold: Float64 = 0.55,
    report_every: Int = 0,        # 0 → fall back to arena_every (0 = no reports)
    do_eval: Bool = True,         # MCTS-eval the best vs OPP1 each report
    do_eval2: Bool = False,       # also MCTS-eval vs OPP2 each report
    verbose: Bool = True,         # print a per-report progress line
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
) raises -> ArenaRunResult:
    # NOTE on units: one loop pass advances every one of `N_ENVS` games by a
    # single self-play *move* (not a full game). `iterations` is therefore the
    # number of self-play moves; `report_every` / `arena_every` are in moves. A
    # report prints the moves done, the cumulative finished-game count, the last
    # train loss, promotions, and the MCTS eval vs each opponent. This differs
    # from the legacy batch-then-train driver, whose "iter" was a whole
    # collect+train+eval round — hence the deliberate `move` / `games` labels.
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM          # ACT + 1
    comptime STATE = ENV.STATE_SIZE
    comptime NSYM = AUG.NUM_SYMMETRIES
    comptime MCTS = GenericGPUMCTS[
        N_ENVS, ACT, OBS, 1, MAX_NODES, NUM_SIMS, 1,
        AlphaGoPUCT[2.5], DirichletNoise[0.25, 0.25], SelfPlay,
        STATE_SIZE=STATE,
    ]
    comptime Graph = ComputeGraph[
        1,
        InputSlot["obs", OBS],
        ExternalNode["pred", NET, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    # Learner net (trains) initialised to the best net's weights.
    var learner = NET.make["gpu", INIT=Kaiming](ctx=ctx)
    hard_copy_params["gpu", M=NET](net, learner, ctx)

    var opt = Adam.make["gpu", M=NET](learner, ctx)
    opt.lr = lr
    var graph = Graph.make["gpu", INIT=Zero](ctx=ctx)
    var mcts = MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)
    var replay = MCTSExampleReplay[OBS, W, CAP]()

    # ── Device buffers ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done = ctx.enqueue_create_buffer[DT](N_ENVS)
    var term = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_next = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var tb_obs = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var tb_tgt = ctx.enqueue_create_buffer[DT](BATCH * W)
    var tb_loss = ctx.enqueue_create_buffer[DT](BATCH)
    var tb_grad = ctx.enqueue_create_buffer[DT](BATCH)

    # ── Host buffers ──
    var obs_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
    var pol_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var done_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var tb_obs_h = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var tb_tgt_h = ctx.enqueue_create_host_buffer[DT](BATCH * W)
    var loss_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    var grad_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()

    # ── Host trajectory storage (per-env in-progress game) ──
    # Augmenter signatures pin MutAnyOrigin — rebind the owned slabs to match.
    var traj_obs = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](N_ENVS * MAX_TRAJ * OBS)
    )
    var traj_pol = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](N_ENVS * MAX_TRAJ * ACT)
    )
    var traj_len = alloc[Int](N_ENVS)
    var tmp_tgt = alloc[Scalar[DT]](W)
    var aug_obs = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](OBS)
    )
    var aug_pol = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](ACT)
    )
    for e in range(N_ENVS):
        traj_len[e] = 0

    # Constant 1/BATCH grad seed (uploaded once).
    for i in range(BATCH):
        grad_h.unsafe_ptr()[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    ctx.enqueue_copy(tb_grad, grad_h)
    ctx.synchronize()

    var tbo_t = TileTensor(tb_obs, row_major[BATCH, OBS]())
    var tbt_t = TileTensor(tb_tgt, row_major[BATCH, W]())
    var loss_t = TileTensor(tb_loss, row_major[BATCH, 1]())
    var grad_t = TileTensor(tb_grad, row_major[BATCH, 1]())

    # ── Initialize all games ──
    ENV.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](ctx, states, obs_dev, legal_dev)
    ctx.synchronize()

    var last_loss: Float64 = 0.0
    var promotions = 0
    var total_games = 0          # cumulative finished self-play games

    # Effective reporting cadence: explicit `report_every`, else piggy-back on
    # the arena cadence (0 ⇒ no periodic eval/print/log at all).
    var rep = report_every if report_every > 0 else arena_every
    if verbose:
        print(
            "AlphaZero self-play:", iterations, "moves,",
            N_ENVS, "envs,", NUM_SIMS, "sims/move | eval(MCTS)1=", OPP1.NAME,
            "eval2=", OPP2.NAME if do_eval2 else String("off"),
            "| report_every=", rep, "moves",
        )

    for it in range(iterations):
        # 1. MCTS search with the BEST net (eval mode for any BatchNorm).
        net.set_attr["training"](Scalar[DT](0.0))
        var pred = AZPredGPU[OBS, ACT, NET].make(net)
        var env_ad = AZEnvGPU[ENV, STATE, OBS, ACT]()
        var root_obs = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
        ](obs_dev.unsafe_ptr())
        var root_legal = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
        ](legal_dev.unsafe_ptr())
        mcts.search_gpu_alphazero[type_of(pred), type_of(env_ad)](
            ctx, pred, env_ad, root_obs, states, root_legal,
            rng_seed=seed + UInt64(it),
        )

        # 2. Pull root obs + visit-count policy, record into trajectory.
        ctx.enqueue_copy(obs_h, obs_dev)
        ctx.enqueue_copy(pol_h, mcts.policies_out)
        ctx.synchronize()
        for e in range(N_ENVS):
            var k = traj_len[e]
            if k < MAX_TRAJ:
                var ob = (e * MAX_TRAJ + k) * OBS
                for j in range(OBS):
                    traj_obs[ob + j] = obs_h.unsafe_ptr()[e * OBS + j]
                var pb = (e * MAX_TRAJ + k) * ACT
                for a in range(ACT):
                    traj_pol[pb + a] = pol_h.unsafe_ptr()[e * ACT + a]
                traj_len[e] = k + 1

        # 3. Step every game by its chosen action.
        ENV.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, mcts.actions_out, rew, done, term, obs_next, legal_next,
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        # 4. Flush finished games with value target z, augmented by symmetry.
        for e in range(N_ENVS):
            if done_h.unsafe_ptr()[e] > 0.5:
                total_games += 1
                var L = traj_len[e]
                var win = Float64(rew_h.unsafe_ptr()[e]) > 0.5
                for k in range(L):
                    var z: Float64 = 0.0
                    if win:
                        z = 1.0 if ((L - 1 - k) % 2 == 0) else -1.0
                    var ob = (e * MAX_TRAJ + k) * OBS
                    var pb = (e * MAX_TRAJ + k) * ACT
                    for s in range(NSYM):
                        AUG.augment_obs[OBS](traj_obs + ob, s, aug_obs)
                        AUG.augment_policy[ACT](traj_pol + pb, s, aug_pol)
                        for a in range(ACT):
                            tmp_tgt[a] = aug_pol[a]
                        tmp_tgt[ACT] = Scalar[DT](z)
                        replay.record(aug_obs, tmp_tgt)
                traj_len[e] = 0

        # 5. Reset finished games, refresh obs/legal.
        ENV.selective_reset_kernel_gpu[N_ENVS, STATE](
            ctx, states, done, rng_seed=seed + UInt64(it)
        )
        ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](ctx, states, obs_dev, legal_dev)
        ctx.synchronize()

        # 6. Train the LEARNER (train mode for BatchNorm).
        if len(replay) >= BATCH and it >= learning_starts:
            learner.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                replay.sample_batch[BATCH](
                    tb_obs_h.unsafe_ptr(), tb_tgt_h.unsafe_ptr()
                )
                ctx.enqueue_copy(tb_obs, tb_obs_h)
                ctx.enqueue_copy(tb_tgt, tb_tgt_h)
                ctx.synchronize()
                opt.zero_grad["gpu", M=NET](learner)
                graph.set_external["pred", NET](learner)
                graph.set_input["obs", BATCH](tbo_t)
                graph.set_input["tgt", BATCH](tbt_t)
                graph.forward["gpu", BATCH](loss_t)
                graph.vjp["gpu", BATCH](grad_t)
                opt.step["gpu", M=NET](learner)
            ctx.enqueue_copy(loss_h, tb_loss)
            ctx.synchronize()
            var ml: Float64 = 0.0
            for b in range(BATCH):
                ml += Float64(loss_h.unsafe_ptr()[b])
            last_loss = ml / Float64(BATCH)

        # 7. Arena gating: periodically challenge the best with the learner.
        if (
            it >= learning_starts
            and arena_every > 0
            and (it + 1) % arena_every == 0
            and len(replay) >= BATCH
        ):
            var rec = candidate_winrate_mcts[
                ENV, NET, NET, ARENA_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES,
            ](ctx, learner, net, seed=seed + UInt64(it) * 7 + 1,
              open_plies=arena_open_plies)
            var accepted = should_promote(
                rec, promote_threshold, min_decisive=ARENA_GAMES // 2
            )
            if accepted:
                hard_copy_params["gpu", M=NET](learner, net, ctx)
                promotions += 1
            if verbose:
                print(
                    "  arena @ move", it + 1,
                    "| learner vs best (MCTS)  W", rec.wins, "D", rec.draws,
                    "L", rec.losses,
                    "→ ACCEPTED" if accepted else "→ rejected",
                    "(promotions", promotions, ")",
                )

        # 8. Periodic report: MCTS-eval the LEARNER (the net actively training,
        #    matching legacy's "eval after train") vs the plugged opponents
        #    (both colors), print a progress line, and flush metrics to logger.
        if rep > 0 and (it + 1) % rep == 0 and it >= learning_starts:
            learner.set_attr["training"](Scalar[DT](0.0))
            var names = List[String]()
            var values = List[Float64]()
            names.append(String("loss"))
            values.append(last_loss)
            names.append(String("games"))
            values.append(Float64(total_games))
            names.append(String("replay_size"))
            values.append(Float64(len(replay)))
            names.append(String("promotions"))
            values.append(Float64(promotions))

            var line = String("  move ") + String(it + 1)
            line += " | games " + String(total_games)
            line += " | loss " + String(last_loss)
            line += " | replay " + String(len(replay))
            line += " | promo " + String(promotions)

            if do_eval:
                var e1 = _eval_both_colors[
                    ENV, NET, OPP1, EVAL_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
                ](ctx, learner, seed=seed + UInt64(it) * 13 + 5)
                var tot1 = e1.wins + e1.draws + e1.losses
                var wr1 = (
                    Float64(e1.wins) / Float64(tot1) if tot1 > 0 else 0.0
                )
                names.append(String("eval1_win"))
                values.append(Float64(e1.wins))
                names.append(String("eval1_draw"))
                values.append(Float64(e1.draws))
                names.append(String("eval1_loss"))
                values.append(Float64(e1.losses))
                names.append(String("eval1_winrate"))
                values.append(wr1)
                line += (
                    " | vs " + OPP1.NAME + " W" + String(e1.wins)
                    + " D" + String(e1.draws) + " L" + String(e1.losses)
                    + " (wr " + String(Int(wr1 * 100.0)) + "%)"
                )

            if do_eval2:
                var e2 = _eval_both_colors[
                    ENV, NET, OPP2, EVAL_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
                ](ctx, learner, seed=seed + UInt64(it) * 17 + 9)
                var tot2 = e2.wins + e2.draws + e2.losses
                var wr2 = (
                    Float64(e2.wins) / Float64(tot2) if tot2 > 0 else 0.0
                )
                names.append(String("eval2_win"))
                values.append(Float64(e2.wins))
                names.append(String("eval2_draw"))
                values.append(Float64(e2.draws))
                names.append(String("eval2_loss"))
                values.append(Float64(e2.losses))
                names.append(String("eval2_winrate"))
                values.append(wr2)
                line += (
                    " | vs " + OPP2.NAME + " W" + String(e2.wins)
                    + " D" + String(e2.draws) + " L" + String(e2.losses)
                    + " (wr " + String(Int(wr2 * 100.0)) + "%)"
                )

            if verbose:
                print(line)
            if logger:
                logger.value()[].log_scalars(names, values, it + 1)

    # Final flush: ensure the returned best is at least as new as the learner if
    # the learner ended clearly ahead (covers runs shorter than arena_every).
    var final_rec = candidate_winrate_mcts[
        ENV, NET, NET, ARENA_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES,
    ](ctx, learner, net, seed=seed + 9991, open_plies=arena_open_plies)
    if should_promote(final_rec, promote_threshold, min_decisive=ARENA_GAMES // 2):
        hard_copy_params["gpu", M=NET](learner, net, ctx)
        promotions += 1

    traj_obs.free()
    traj_pol.free()
    traj_len.free()
    tmp_tgt.free()
    aug_obs.free()
    aug_pol.free()
    return ArenaRunResult(last_loss=last_loss, promotions=promotions)
