"""Arena-gated two-player GUMBEL MuZero self-play (GPU) — the board-game driver.

The MuZero counterpart of `alphazero/selfplay_arena_gumbel.mojo`. Where the
AlphaZero arena searches the TRUE game rules with a single net, MuZero plans in
latent space over a *learned model* (three nets h/g/f = representation /
dynamics / prediction) and trains them with a K-step BPTT unroll. This driver
keeps the AlphaZero arena's production structure — best/learner split, Arena
gating, symmetry augmentation, two pluggable eval opponents — but swaps every
algorithmic core to MuZero:

  * **Search.** `GumbelGPUMCTS[..., SelfPlay]` over the on-device h/g/f adapters
    (`MZRepGPU`/`MZDynGPU`/`MZPredGPU`), with the env's legal mask applied at the
    root (`apply_legal=True`) and the two-player (`SelfPlay`) backup that negates
    value each ply. The improved policy is the stored target.

  * **Targets.** Per-step the driver records `(obs, action, π, root_value,
    to_play, legal)` into `MCTSSequenceReplay`; the n-step value targets carry
    the two-player sign flips (`zero/nstep_targets.mojo`). `to_play` is the ply
    parity (ConnectFour's canonical obs is always the mover's frame, P0 first).

  * **Best/learner gating.** A frozen *best* h/g/f trio generates all self-play;
    a *learner* trio trains. Every `arena_every` moves the learner plays the best
    at full Gumbel strength from both colors (`mz_candidate_winrate`); on a clear
    majority of decisive games it is promoted (params-only `hard_copy_params` ×3
    — the MLP h/g/f carry no BatchNorm, so there are no running stats to copy,
    sidestepping the C4-gumbel promotion pitfall seen with BN ResNets).

  * **Augmentation.** Each finished game is replicated under the board's symmetry
    group (`AUG`): obs / policy / legal are permuted and the stored *action*
    index is re-derived through the same permutation (one-hot → augment → argmax),
    so the whole `(obs, a, π)` sequence stays a valid trajectory.

Telemetry mirrors the AlphaZero arena: a periodic full-strength MuZero-MCTS eval
vs `OPP1`/`OPP2` (both colors), per-batch loss split to the logger, and a
progress line. `report_every` / `arena_every` are in self-play *moves* (one loop
pass advances all `N_ENVS` games one move).

Single-file, GPU-only (the Gumbel planner is device-side). Exposed as a free
function (like `run_muzero_selfplay_2p_cpu`) rather than on the single-player
`MuZeroAgent` facade — the example wires the three nets + optimizers directly.
"""

from std.math import exp, log
from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.planners.tree_search import GumbelGPUMCTS, SelfPlay

from .blocks import mz_unroll_train_step_gpu, MZScratch
from .selfplay_gpu_device import _mz_emit_train_diag
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.symmetries import BoardAugmenter, IdentityAugmenter
from ..zero.evaluators import GPUEvaluator, RandomOpponent
from ..zero.temperature import visit_temperature


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


@always_inline
def _xs(s: UInt64) -> UInt64:
    var x = s
    x = x ^ (x << 13)
    x = x ^ (x >> 7)
    x = x ^ (x << 17)
    return x


@fieldwise_init
struct MZArenaResult(Copyable, Movable):
    """Aggregate match record from the candidate's perspective."""

    var wins: Int
    var draws: Int
    var losses: Int


@fieldwise_init
struct MZArenaRunResult(Copyable, Movable):
    var last_loss: Float64
    var promotions: Int


def _mz_should_promote(
    rec: MZArenaResult, threshold: Float64 = 0.55, min_decisive: Int = 1
) -> Bool:
    """Standard AlphaZero accept rule: among decisive games (draws ignored) the
    candidate must win at least `threshold` of them, and there must be at least
    `min_decisive` decisive games (else a 0-0 tie spuriously promotes)."""
    var decisive = rec.wins + rec.losses
    if decisive < min_decisive:
        return False
    return Float64(rec.wins) >= threshold * Float64(decisive)


# ══════════════════════════════════════════════════════════════════════════
# Batched MuZero-Gumbel search → greedy (argmax-over-legal) actions.
# Shared by the arena (two net trios) and the eval-vs-opponent helper.
# ══════════════════════════════════════════════════════════════════════════


def _mz_search_argmax[
    NREP: Module,
    NDYN: Module,
    NPRED: Module,
    N: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    NUM_SIMS: Int,
](
    ctx: DeviceContext,
    mut planner: GumbelGPUMCTS[
        N, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ],
    mut rep_a: MZRepGPU[OBS, LATENT, NREP],
    mut dyn_a: MZDynGPU[LATENT, ACT, BINS, NDYN],
    mut pred_a: MZPredGPU[LATENT, ACT, BINS, NPRED],
    obs_dev: DeviceBuffer[DT],
    legal_dev: DeviceBuffer[DT],
    mut h_pol: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mut legal_h: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mut actions_h: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rng_seed: UInt32,
) raises:
    """Run one batched Gumbel search over the learned model on `N` root obs and
    write the greedy (argmax of the improved policy over legal moves) action per
    env into `actions_h`. The planner's gumbel_scale governs determinism (set 0
    at the call site for eval/arena)."""
    ctx.enqueue_copy(planner.legal_mask_view(), legal_dev)
    var obs_t = LayoutTensor[DT, Layout.row_major(N, OBS), MutAnyOrigin](
        mptr(obs_dev.unsafe_ptr())
    )
    planner.search_gpu[
        MZRepGPU[OBS, LATENT, NREP],
        MZDynGPU[LATENT, ACT, BINS, NDYN],
        MZPredGPU[LATENT, ACT, BINS, NPRED],
    ](
        ctx, rep_a, dyn_a, pred_a, obs_t,
        apply_legal=True, k_actual=MAX_K, rng_seed=rng_seed,
    )
    ctx.enqueue_copy(h_pol, planner.policies_view())
    ctx.enqueue_copy(legal_h, legal_dev)
    ctx.synchronize()
    for e in range(N):
        var best = -1
        var bestv = Float64(-1e30)
        for a in range(ACT):
            if Float64(legal_h[e * ACT + a]) > 0.5:
                var v = Float64(h_pol[e * ACT + a])
                if best < 0 or v > bestv:
                    bestv = v
                    best = a
        actions_h[e] = Scalar[DT](best if best >= 0 else 0)


# ══════════════════════════════════════════════════════════════════════════
# Arena: learner net trio vs best net trio, both at full Gumbel strength.
# ══════════════════════════════════════════════════════════════════════════


def mz_arena_match[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_GAMES: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut a_rep: REP, mut a_dyn: DYN, mut a_pred: PRED,
    mut b_rep: REP, mut b_dyn: DYN, mut b_pred: PRED,
    gamma: Float64,
    a_player: Int = 0,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> MZArenaResult:
    """Net trio A vs net trio B, both planning over their own learned model at
    full Gumbel strength (deterministic, gumbel_scale=0). A plays color
    `a_player`. Per-env done tracking + selective reset (early finishers stop
    interfering); attribution by reward + whose turn it was. Returns A's record.
    `open_plies` random opening plies (both sides) diversify the lockstep batch.
    """
    comptime STATE = ENV.STATE_SIZE

    a_rep.set_attr["training"](Scalar[DT](0.0))
    a_dyn.set_attr["training"](Scalar[DT](0.0))
    a_pred.set_attr["training"](Scalar[DT](0.0))
    b_rep.set_attr["training"](Scalar[DT](0.0))
    b_dyn.set_attr["training"](Scalar[DT](0.0))
    b_pred.set_attr["training"](Scalar[DT](0.0))

    var planner = GumbelGPUMCTS[
        N_GAMES, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ](ctx, gamma=gamma, v_min=-1.0, v_max=1.0, gumbel_scale=0.0)
    var ar = MZRepGPU[OBS, LATENT, REP].make(a_rep)
    var ad = MZDynGPU[LATENT, ACT, BINS, DYN].make(a_dyn)
    var ap = MZPredGPU[LATENT, ACT, BINS, PRED].make(a_pred)
    var br = MZRepGPU[OBS, LATENT, REP].make(b_rep)
    var bd = MZDynGPU[LATENT, ACT, BINS, DYN].make(b_dyn)
    var bp = MZPredGPU[LATENT, ACT, BINS, PRED].make(b_pred)

    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var act_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var h_pol = _a(N_GAMES * ACT)
    var legal_h = _a(N_GAMES * ACT)
    var act_h = _a(N_GAMES)
    var done_h = _a(N_GAMES)
    var rew_h = _a(N_GAMES)
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_GAMES, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var eval_done = InlineArray[Bool, N_GAMES](fill=False)
    var eval_result = InlineArray[Int, N_GAMES](fill=0)  # 1=A win 2=A loss 3=draw
    var all_done = False
    var move = 0
    var rng = seed | 1

    comptime MAX_MOVES = MAX_PLIES + ACT
    while not all_done and move < MAX_MOVES:
        var a_turn = (move % 2) == a_player
        if move < open_plies:
            rng = _xs(rng)
            RandomOpponent.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, act_dev, legal_dev, states, rng
            )
        elif a_turn:
            _mz_search_argmax[
                REP, DYN, PRED, N_GAMES, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, planner, ar, ad, ap, obs_dev, legal_dev,
                h_pol, legal_h, act_h, rng_seed=UInt32(move + 1),
            )
            ctx.enqueue_copy(act_dev, act_h)
        else:
            _mz_search_argmax[
                REP, DYN, PRED, N_GAMES, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, planner, br, bd, bp, obs_dev, legal_dev,
                h_pol, legal_h, act_h, rng_seed=UInt32(move * 3 + 7),
            )
            ctx.enqueue_copy(act_dev, act_h)
        ctx.synchronize()

        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, act_dev, rew, done, term, obs_next, legal_next
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        all_done = True
        for e in range(N_GAMES):
            if not eval_done[e] and Float64(done_h[e]) > 0.5:
                eval_done[e] = True
                var r = Float64(rew_h[e])
                if r > 0.5:
                    eval_result[e] = 1 if a_turn else 2
                elif r < -0.5:
                    eval_result[e] = 2 if a_turn else 1
                else:
                    eval_result[e] = 3
            if not eval_done[e]:
                all_done = False

        ENV.selective_reset_kernel_gpu[N_GAMES, STATE](
            ctx, states, done, rng_seed=seed + UInt64(move) * 7 + 3
        )
        ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()
        move += 1

    var wins = 0
    var draws = 0
    var losses = 0
    for e in range(N_GAMES):
        if eval_result[e] == 1:
            wins += 1
        elif eval_result[e] == 2:
            losses += 1
        else:
            draws += 1

    h_pol.free(); legal_h.free(); act_h.free(); done_h.free(); rew_h.free()
    return MZArenaResult(wins=wins, draws=draws, losses=losses)


def mz_candidate_winrate[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_PER_COLOR: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut c_rep: REP, mut c_dyn: DYN, mut c_pred: PRED,
    mut b_rep: REP, mut b_dyn: DYN, mut b_pred: PRED,
    gamma: Float64,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> MZArenaResult:
    """Candidate (learner) vs best, both at full Gumbel strength, from BOTH
    colors, aggregated from the candidate's perspective."""
    var p0 = mz_arena_match[
        ENV, REP, DYN, PRED, N_PER_COLOR, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](
        ctx, c_rep, c_dyn, c_pred, b_rep, b_dyn, b_pred, gamma,
        a_player=0, seed=seed, open_plies=open_plies,
    )
    var p1 = mz_arena_match[
        ENV, REP, DYN, PRED, N_PER_COLOR, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](
        ctx, c_rep, c_dyn, c_pred, b_rep, b_dyn, b_pred, gamma,
        a_player=1, seed=seed + 1, open_plies=open_plies,
    )
    return MZArenaResult(
        wins=p0.wins + p1.wins,
        draws=p0.draws + p1.draws,
        losses=p0.losses + p1.losses,
    )


# ══════════════════════════════════════════════════════════════════════════
# Eval: MuZero-Gumbel agent vs a pluggable GPUEvaluator opponent (both colors).
# ══════════════════════════════════════════════════════════════════════════


def _mz_eval_one_color[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    OPP: GPUEvaluator,
    N_GAMES: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut rep: REP, mut dyn: DYN, mut pred: PRED,
    gamma: Float64,
    agent_player: Int,
    seed: UInt64,
    open_plies: Int,
) raises -> MZArenaResult:
    comptime STATE = ENV.STATE_SIZE

    rep.set_attr["training"](Scalar[DT](0.0))
    dyn.set_attr["training"](Scalar[DT](0.0))
    pred.set_attr["training"](Scalar[DT](0.0))

    var planner = GumbelGPUMCTS[
        N_GAMES, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ](ctx, gamma=gamma, v_min=-1.0, v_max=1.0, gumbel_scale=0.0)
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var act_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var h_pol = _a(N_GAMES * ACT)
    var legal_h = _a(N_GAMES * ACT)
    var act_h = _a(N_GAMES)
    var done_h = _a(N_GAMES)
    var rew_h = _a(N_GAMES)
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_GAMES, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var eval_done = InlineArray[Bool, N_GAMES](fill=False)
    var eval_result = InlineArray[Int, N_GAMES](fill=0)
    var all_done = False
    var move = 0
    var rng = seed | 1

    comptime MAX_MOVES = MAX_PLIES + ACT
    while not all_done and move < MAX_MOVES:
        var agent_turn = (move % 2) == agent_player
        if move < open_plies:
            rng = _xs(rng)
            RandomOpponent.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, act_dev, legal_dev, states, rng
            )
        elif agent_turn:
            _mz_search_argmax[
                REP, DYN, PRED, N_GAMES, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, planner, rep_a, dyn_a, pred_a, obs_dev, legal_dev,
                h_pol, legal_h, act_h, rng_seed=UInt32(move + 1),
            )
            ctx.enqueue_copy(act_dev, act_h)
        else:
            rng = _xs(rng)
            OPP.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, act_dev, legal_dev, states, rng
            )
        ctx.synchronize()

        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, act_dev, rew, done, term, obs_next, legal_next
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        all_done = True
        for e in range(N_GAMES):
            if not eval_done[e] and Float64(done_h[e]) > 0.5:
                eval_done[e] = True
                var r = Float64(rew_h[e])
                if r > 0.5:
                    eval_result[e] = 1 if agent_turn else 2
                elif r < -0.5:
                    eval_result[e] = 2 if agent_turn else 1
                else:
                    eval_result[e] = 3
            if not eval_done[e]:
                all_done = False

        ENV.selective_reset_kernel_gpu[N_GAMES, STATE](
            ctx, states, done, rng_seed=seed + UInt64(move) * 7 + 3
        )
        ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()
        move += 1

    var wins = 0
    var draws = 0
    var losses = 0
    for e in range(N_GAMES):
        if eval_result[e] == 1:
            wins += 1
        elif eval_result[e] == 2:
            losses += 1
        else:
            draws += 1

    h_pol.free(); legal_h.free(); act_h.free(); done_h.free(); rew_h.free()
    return MZArenaResult(wins=wins, draws=draws, losses=losses)


def mz_eval_both_colors[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    OPP: GPUEvaluator,
    N_GAMES: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut rep: REP, mut dyn: DYN, mut pred: PRED,
    gamma: Float64,
    seed: UInt64 = 1,
    open_plies: Int = 0,
) raises -> MZArenaResult:
    """Aggregate the MuZero agent's full-strength (Gumbel) record vs `OPP` over
    `N_GAMES` games as each color, so first-move advantage cancels."""
    var p0 = _mz_eval_one_color[
        ENV, REP, DYN, PRED, OPP, N_GAMES, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](ctx, rep, dyn, pred, gamma, agent_player=0, seed=seed, open_plies=open_plies)
    var p1 = _mz_eval_one_color[
        ENV, REP, DYN, PRED, OPP, N_GAMES, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](
        ctx, rep, dyn, pred, gamma,
        agent_player=1, seed=seed + 33333, open_plies=open_plies,
    )
    return MZArenaResult(
        wins=p0.wins + p1.wins,
        draws=p0.draws + p1.draws,
        losses=p0.losses + p1.losses,
    )


# ══════════════════════════════════════════════════════════════════════════
# Main driver.
# ══════════════════════════════════════════════════════════════════════════


def run_muzero_selfplay_arena_gumbel_2p[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    AUG: BoardAugmenter,
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
    MAX_PLIES: Int,
    OPP1: GPUEvaluator = RandomOpponent,
    OPP2: GPUEvaluator = RandomOpponent,
    L: Logger = NoOpLogger,
    ARENA_GAMES: Int = 32,
    EVAL_GAMES: Int = 64,
    TEMP_MOVES: Int = 8,
](
    ctx: DeviceContext,
    mut rep: REP,           # the BEST representation net (final weights on return)
    mut dyn: DYN,           # the BEST dynamics net
    mut pred: PRED,         # the BEST prediction net
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = 1,
    lr: Scalar[DT] = Scalar[DT](2e-3),
    gamma: Scalar[DT] = Scalar[DT](1.0),
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    max_grad_norm: Scalar[DT] = Scalar[DT](1.0),
    seed: UInt64 = 0,
    arena_every: Int = 2000,
    arena_open_plies: Int = 2,
    promote_threshold: Float64 = 0.55,
    report_every: Int = 1000,
    diag_every: Int = 0,
    do_eval: Bool = True,
    do_eval2: Bool = False,
    verbose: Bool = True,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    selfplay_open_plies: Int = 2,
    eval_open_plies: Int = 4,
    temperature_decay_steps: Int = 0,
    qnorm_per_node: Bool = True,
) raises -> MZArenaRunResult:
    """Two-player Gumbel MuZero self-play with Arena gating. `rep`/`dyn`/`pred`
    are the BEST nets and hold the final (best) weights on return. One loop pass
    advances all `N_ENVS` games one move; `iterations`/`report_every`/
    `arena_every` are in moves. v_min/v_max are pinned to -1/+1 (board outcome)."""
    comptime STATE = ENV.STATE_SIZE
    comptime NSYM = AUG.NUM_SYMMETRIES
    comptime VMIN = Scalar[DT](-1.0)
    comptime VMAX = Scalar[DT](1.0)

    # ── Learner net trio (trains), initialised to the best weights ──
    var l_rep = REP.make["gpu", INIT=Kaiming](ctx=ctx)
    var l_dyn = DYN.make["gpu", INIT=Kaiming](ctx=ctx)
    var l_pred = PRED.make["gpu", INIT=Kaiming](ctx=ctx)
    hard_copy_params["gpu", M=REP](rep, l_rep, ctx)
    hard_copy_params["gpu", M=DYN](dyn, l_dyn, ctx)
    hard_copy_params["gpu", M=PRED](pred, l_pred, ctx)

    var orep = Adam.make["gpu", M=REP](l_rep, ctx)
    var odyn = Adam.make["gpu", M=DYN](l_dyn, ctx)
    var opred = Adam.make["gpu", M=PRED](l_pred, ctx)
    orep.lr = lr; odyn.lr = lr; opred.lr = lr
    orep.max_grad_norm = max_grad_norm
    odyn.max_grad_norm = max_grad_norm
    opred.max_grad_norm = max_grad_norm

    # ── Self-play planner: best nets generate the data (Gumbel exploration on) ──
    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ](
        ctx, gamma=Float64(gamma), v_min=-1.0, v_max=1.0,
        gumbel_scale=1.0, qnorm_per_node=qnorm_per_node,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))
    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

    # ── diag scratch: re-forward the root prediction on the last train batch,
    #    D2H, and emit the full MuZero head-fit metric set (target entropy /
    #    max-prob / value head) — same set as the other MuZero drivers. ──
    var d_diag_obs = ctx.enqueue_create_buffer[DT](B * OBS)
    var d_diag_z = ctx.enqueue_create_buffer[DT](B * LATENT)
    var d_diag_pred = ctx.enqueue_create_buffer[DT](B * (ACT + BINS))
    var h_diag_pred = _a(B * (ACT + BINS))

    # ── Device buffers ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done = ctx.enqueue_create_buffer[DT](N_ENVS)
    var term = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_next = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var act_dev = ctx.enqueue_create_buffer[DT](N_ENVS)

    # ── Host mirrors ──
    var obs_h = _a(N_ENVS * OBS)
    var pol_h = _a(N_ENVS * ACT)
    var val_h = _a(N_ENVS)
    var legal_h = _a(N_ENVS * ACT)
    var done_h = _a(N_ENVS)
    var rew_h = _a(N_ENVS)
    var act_h = _a(N_ENVS)

    # ── Training slabs (host, time-major) ──
    var t_obs0 = _a(B * OBS)
    var t_act = _a(K * B)
    var t_pol = _a((K + 1) * B * ACT)
    var t_val = _a((K + 1) * B)
    var t_rew = _a(K * B)
    var l_parts = _a(3)

    # ── Augmentation scratch (one game's worth, reused) ──
    var aug_obs = _a(MAX_PLIES * OBS)
    var aug_pol = _a(MAX_PLIES * ACT)
    var aug_legal = _a(MAX_PLIES * ACT)
    var aug_act = _a(MAX_PLIES)
    var onehot = _a(ACT)
    var onehot_out = _a(ACT)

    # ── Per-env in-progress episode accumulators ──
    var e_obs = List[List[Scalar[DT]]]()
    var e_act = List[List[Scalar[DT]]]()
    var e_rew = List[List[Scalar[DT]]]()
    var e_pol = List[List[Scalar[DT]]]()
    var e_val = List[List[Scalar[DT]]]()
    var e_tp = List[List[Scalar[DT]]]()
    var e_legal = List[List[Scalar[DT]]]()
    for _ in range(N_ENVS):
        e_obs.append(List[Scalar[DT]]())
        e_act.append(List[Scalar[DT]]())
        e_rew.append(List[Scalar[DT]]())
        e_pol.append(List[Scalar[DT]]())
        e_val.append(List[Scalar[DT]]())
        e_tp.append(List[Scalar[DT]]())
        e_legal.append(List[Scalar[DT]]())
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var rng = seed ^ UInt64(0x9E3779B97F4A7C15)
    var mcts_seed = UInt32(seed & UInt64(0xFFFF))
    var last_loss = 0.0
    var promotions = 0
    var total_games = 0
    var games_prev = 0
    var rep_cadence = report_every if report_every > 0 else arena_every

    if verbose:
        print(
            "MuZero(2p Gumbel) self-play:", iterations, "moves,",
            N_ENVS, "envs,", NUM_SIMS, "sims/move | eval1=", OPP1.NAME,
            "eval2=", OPP2.NAME if do_eval2 else String("off"),
            "| report_every=", rep_cadence, "moves",
        )

    for it in range(iterations):
        # ── 1. Self-play search with the BEST nets (eval mode) ──
        rep.set_attr["training"](Scalar[DT](0.0))
        dyn.set_attr["training"](Scalar[DT](0.0))
        pred.set_attr["training"](Scalar[DT](0.0))
        ctx.enqueue_copy(planner.legal_mask_view(), legal_dev)
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin](
            mptr(obs_dev.unsafe_ptr())
        )
        planner.search_gpu[
            MZRepGPU[OBS, LATENT, REP],
            MZDynGPU[LATENT, ACT, BINS, DYN],
            MZPredGPU[LATENT, ACT, BINS, PRED],
        ](
            ctx, rep_a, dyn_a, pred_a, obs_t,
            apply_legal=True, k_actual=MAX_K, rng_seed=mcts_seed,
        )
        mcts_seed += UInt32(1)

        ctx.enqueue_copy(obs_h, obs_dev)
        ctx.enqueue_copy(pol_h, planner.policies_view())
        ctx.enqueue_copy(val_h, planner.root_value_view())
        ctx.enqueue_copy(legal_h, legal_dev)
        ctx.synchronize()

        # ── 2. Per env: record the labelled step, choose the action played ──
        var temp = visit_temperature(it, temperature_decay_steps)
        for e in range(N_ENVS):
            var ply = len(e_act[e])
            var tp = Scalar[DT](ply % 2)
            for j in range(OBS):
                e_obs[e].append(obs_h[e * OBS + j])
            for a in range(ACT):
                e_pol[e].append(pol_h[e * ACT + a])
                e_legal[e].append(legal_h[e * ACT + a])
            e_val[e].append(val_h[e])
            e_tp[e].append(tp)

            # Action selection: opening diversity → temp sampling → argmax,
            # with a final legality guard (improved policy is legal-masked, but
            # guard against a degenerate -inf row picking an illegal column).
            var a_sel = -1
            if ply < selfplay_open_plies:
                var n_legal = 0
                for a in range(ACT):
                    if Float64(legal_h[e * ACT + a]) > 0.5:
                        n_legal += 1
                if n_legal > 0:
                    rng = _xs(rng)
                    var pick = Int(rng % UInt64(n_legal))
                    var seen = 0
                    for a in range(ACT):
                        if Float64(legal_h[e * ACT + a]) > 0.5:
                            if seen == pick:
                                a_sel = a
                                break
                            seen += 1
            elif ply < TEMP_MOVES:
                var wsum = 0.0
                var w = List[Float64](capacity=ACT)
                for a in range(ACT):
                    var p = Float64(pol_h[e * ACT + a])
                    if temp != 1.0 and p > 0.0:
                        p = exp(log(p) / temp)
                    w.append(p)
                    wsum += p
                rng = _xs(rng)
                var r = Float64(rng % UInt64(1_000_000)) / 1_000_000.0 * wsum
                var cum = 0.0
                for a in range(ACT):
                    cum += w[a]
                    if r <= cum and w[a] > 0.0:
                        a_sel = a
                        break
            if a_sel < 0:
                var bv = -1.0
                for a in range(ACT):
                    var p = Float64(pol_h[e * ACT + a])
                    if p > bv:
                        bv = p
                        a_sel = a
            if a_sel < 0 or Float64(legal_h[e * ACT + a_sel]) <= 0.5:
                var bestl = -1
                var bvl = -1.0e30
                for a in range(ACT):
                    if Float64(legal_h[e * ACT + a]) > 0.5:
                        var p = Float64(pol_h[e * ACT + a])
                        if bestl < 0 or p > bvl:
                            bvl = p
                            bestl = a
                if bestl >= 0:
                    a_sel = bestl
            e_act[e].append(Scalar[DT](a_sel))
            act_h[e] = Scalar[DT](a_sel)
        ctx.enqueue_copy(act_dev, act_h)

        # ── 3. Step every game ──
        ENV.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, act_dev, rew, done, term, obs_next, legal_next
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        # ── 4. Flush finished games into the replay (reward recorded per step;
        #      symmetry-augmented). Board games always terminate → truncated
        #      = False (every `done` is a real terminal). ──
        for e in range(N_ENVS):
            e_rew[e].append(rew_h[e])
            if Float64(done_h[e]) > 0.5:
                total_games += 1
                var Lg = len(e_act[e])
                for s in range(NSYM):
                    if s == 0:
                        rb.store_episode(
                            mptr(e_obs[e].unsafe_ptr()),
                            mptr(e_act[e].unsafe_ptr()),
                            mptr(e_rew[e].unsafe_ptr()),
                            mptr(e_pol[e].unsafe_ptr()),
                            mptr(e_val[e].unsafe_ptr()),
                            mptr(e_tp[e].unsafe_ptr()),
                            mptr(e_legal[e].unsafe_ptr()),
                            Lg,
                            truncated=False,
                        )
                    else:
                        for k in range(Lg):
                            # `mut out` needs a mutable lvalue — bind the offset
                            # destinations to locals.
                            var src_obs = mptr(e_obs[e].unsafe_ptr()) + k * OBS
                            var dst_obs = aug_obs + k * OBS
                            AUG.augment_obs[OBS](src_obs, s, dst_obs)
                            var src_pol = mptr(e_pol[e].unsafe_ptr()) + k * ACT
                            var dst_pol = aug_pol + k * ACT
                            AUG.augment_policy[ACT](src_pol, s, dst_pol)
                            var src_leg = mptr(e_legal[e].unsafe_ptr()) + k * ACT
                            var dst_leg = aug_legal + k * ACT
                            AUG.augment_policy[ACT](src_leg, s, dst_leg)
                            # Permute the stored action through the same symmetry
                            # (one-hot → augment_policy → argmax).
                            for a in range(ACT):
                                onehot[a] = Scalar[DT](0.0)
                            onehot[Int(e_act[e][k])] = Scalar[DT](1.0)
                            AUG.augment_policy[ACT](onehot, s, onehot_out)
                            var ba = 0
                            for a in range(1, ACT):
                                if Float64(onehot_out[a]) > Float64(onehot_out[ba]):
                                    ba = a
                            aug_act[k] = Scalar[DT](ba)
                        rb.store_episode(
                            aug_obs,
                            aug_act,
                            mptr(e_rew[e].unsafe_ptr()),
                            aug_pol,
                            mptr(e_val[e].unsafe_ptr()),
                            mptr(e_tp[e].unsafe_ptr()),
                            aug_legal,
                            Lg,
                            truncated=False,
                        )
                e_obs[e].clear(); e_act[e].clear(); e_rew[e].clear()
                e_pol[e].clear(); e_val[e].clear(); e_tp[e].clear()
                e_legal[e].clear()

        # ── 5. Reset finished games, refresh obs/legal ──
        ENV.selective_reset_kernel_gpu[N_ENVS, STATE](
            ctx, states, done, rng_seed=seed + UInt64(it)
        )
        ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()

        # ── 6. Train the LEARNER nets (K-step BPTT unroll) ──
        var trained = (
            rb.num_steps() >= learning_starts and rb.num_episodes() > 0
        )
        if trained:
            l_rep.set_attr["training"](Scalar[DT](1.0))
            l_dyn.set_attr["training"](Scalar[DT](1.0))
            l_pred.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                rb.sample_training_batch[B, K, N](
                    gamma, t_obs0, t_act, t_pol, t_val, t_rew
                )
                last_loss = Float64(
                    mz_unroll_train_step_gpu[
                        REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS
                    ](
                        ctx, l_rep, l_dyn, l_pred, orep, odyn, opred,
                        scratch,
                        t_obs0, t_act, t_pol, t_val, t_rew,
                        VMIN, VMAX, value_coef,
                        loss_parts=l_parts,
                    )
                )

        # ── 6b. Per-batch diagnostics → logger. Re-forwards the LEARNER root
        #      prediction on the last train batch (`t_obs0`) and emits loss +
        #      loss_policy/value/reward split + the head-fit metrics
        #      (policy_ce/entropy, target_entropy/max_prob, value_mse/mean,
        #      value_target_mean) — the same set the other MuZero drivers log. ──
        if (
            Bool(logger)
            and diag_every > 0
            and trained
            and (it + 1) % diag_every == 0
        ):
            _mz_emit_train_diag[REP, PRED, B, OBS, ACT, BINS, L](
                ctx, l_rep, l_pred, d_diag_obs, d_diag_z, d_diag_pred,
                h_diag_pred, t_obs0, t_pol, t_val, l_parts,
                VMIN, VMAX, last_loss, it + 1, logger.value(),
            )

        # ── 7. Arena gating: learner challenges best ──
        if (
            trained
            and arena_every > 0
            and (it + 1) % arena_every == 0
        ):
            var rec = mz_candidate_winrate[
                ENV, REP, DYN, PRED, ARENA_GAMES, OBS, ACT, LATENT, BINS,
                NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
            ](
                ctx, l_rep, l_dyn, l_pred, rep, dyn, pred, Float64(gamma),
                seed=seed + UInt64(it) * 7 + 1, open_plies=arena_open_plies,
            )
            var accepted = _mz_should_promote(
                rec, promote_threshold, min_decisive=ARENA_GAMES // 2
            )
            if accepted:
                hard_copy_params["gpu", M=REP](l_rep, rep, ctx)
                hard_copy_params["gpu", M=DYN](l_dyn, dyn, ctx)
                hard_copy_params["gpu", M=PRED](l_pred, pred, ctx)
                promotions += 1
            if verbose:
                print(
                    "  arena @ move", it + 1,
                    "| learner vs best  W", rec.wins, "D", rec.draws,
                    "L", rec.losses,
                    "→ ACCEPTED" if accepted else "→ rejected",
                    "(promotions", promotions, ")",
                )

        # ── 8. Periodic report: MCTS-eval the LEARNER vs the opponents ──
        if (
            rep_cadence > 0
            and (it + 1) % rep_cadence == 0
            and trained
        ):
            var games_delta = total_games - games_prev
            games_prev = total_games
            var names = List[String]()
            var values = List[Float64]()
            names.append(String("loss")); values.append(last_loss)
            names.append(String("games")); values.append(Float64(total_games))
            names.append(String("replay_size"))
            values.append(Float64(rb.num_steps()))
            names.append(String("promotions"))
            values.append(Float64(promotions))

            var line = String("  move ") + String(it + 1)
            line += " | games " + String(total_games)
            line += " (+" + String(games_delta) + ")"
            line += " | loss " + String(last_loss)
            line += " | replay " + String(rb.num_steps())
            line += " | promo " + String(promotions)

            if do_eval:
                var e1 = mz_eval_both_colors[
                    ENV, REP, DYN, PRED, OPP1, EVAL_GAMES, OBS, ACT, LATENT,
                    BINS, NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
                ](
                    ctx, l_rep, l_dyn, l_pred, Float64(gamma),
                    seed=seed + UInt64(it) * 13 + 5, open_plies=eval_open_plies,
                )
                var tot1 = e1.wins + e1.draws + e1.losses
                var wr1 = Float64(e1.wins) / Float64(tot1) if tot1 > 0 else 0.0
                names.append(String("eval1_win")); values.append(Float64(e1.wins))
                names.append(String("eval1_draw")); values.append(Float64(e1.draws))
                names.append(String("eval1_loss")); values.append(Float64(e1.losses))
                names.append(String("eval1_winrate")); values.append(wr1)
                line += (
                    " | vs " + OPP1.NAME + " W" + String(e1.wins)
                    + " D" + String(e1.draws) + " L" + String(e1.losses)
                    + " (wr " + String(Int(wr1 * 100.0)) + "%)"
                )

            if do_eval2:
                var e2 = mz_eval_both_colors[
                    ENV, REP, DYN, PRED, OPP2, EVAL_GAMES, OBS, ACT, LATENT,
                    BINS, NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
                ](
                    ctx, l_rep, l_dyn, l_pred, Float64(gamma),
                    seed=seed + UInt64(it) * 17 + 9, open_plies=eval_open_plies,
                )
                var tot2 = e2.wins + e2.draws + e2.losses
                var wr2 = Float64(e2.wins) / Float64(tot2) if tot2 > 0 else 0.0
                names.append(String("eval2_win")); values.append(Float64(e2.wins))
                names.append(String("eval2_draw")); values.append(Float64(e2.draws))
                names.append(String("eval2_loss")); values.append(Float64(e2.losses))
                names.append(String("eval2_winrate")); values.append(wr2)
                line += (
                    " | vs " + OPP2.NAME + " W" + String(e2.wins)
                    + " D" + String(e2.draws) + " L" + String(e2.losses)
                    + " (wr " + String(Int(wr2 * 100.0)) + "%)"
                )

            if verbose:
                print(line)
            if logger:
                logger.value()[].log_scalars(names, values, it + 1)

    # ── Final flush: promote the learner if it ended clearly ahead ──
    var final_rec = mz_candidate_winrate[
        ENV, REP, DYN, PRED, ARENA_GAMES, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](
        ctx, l_rep, l_dyn, l_pred, rep, dyn, pred, Float64(gamma),
        seed=seed + 9991, open_plies=arena_open_plies,
    )
    if _mz_should_promote(
        final_rec, promote_threshold, min_decisive=ARENA_GAMES // 2
    ):
        hard_copy_params["gpu", M=REP](l_rep, rep, ctx)
        hard_copy_params["gpu", M=DYN](l_dyn, dyn, ctx)
        hard_copy_params["gpu", M=PRED](l_pred, pred, ctx)
        promotions += 1

    t_obs0.free(); t_act.free(); t_pol.free(); t_val.free(); t_rew.free()
    l_parts.free(); h_diag_pred.free()
    obs_h.free(); pol_h.free(); val_h.free(); legal_h.free()
    done_h.free(); rew_h.free(); act_h.free()
    aug_obs.free(); aug_pol.free(); aug_legal.free(); aug_act.free()
    onehot.free(); onehot_out.free()
    # keep the learner nets alive past the adapters' borrowed pointers.
    _ = l_rep^
    _ = l_dyn^
    _ = l_pred^
    return MZArenaRunResult(last_loss=last_loss, promotions=promotions)
