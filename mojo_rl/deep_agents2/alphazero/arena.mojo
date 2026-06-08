"""Arena — net-vs-net match harness for AlphaZero candidate/best gating.

Two flavours of head-to-head match, both in lockstep batched games:

  * **argmax** (`arena_match` / `candidate_winrate`) — each net picks the argmax
    of its policy head over legal moves. Cheap; measures the *raw network*. Good
    for "did it learn?" smoke checks.
  * **MCTS** (`arena_match_mcts` / `candidate_winrate_mcts`) — each net plays at
    full search strength (`NUM_SIMS`, temp=0, NoNoise). This is the **correct
    gate** for AlphaZero promotion: self-play generation always runs MCTS on top
    of the net, so the arena should compare under that same condition. Crucially
    MCTS exercises the *value head* (which evaluates the tree), whereas argmax is
    blind to it — an argmax gate can wrongly reject a learner whose value head
    improved but whose raw policy didn't. The driver gates on this variant.

Both deterministic-MCTS players from a fixed start would play one identical
line, so `open_plies` random opening moves (both sides) spread the batch over
diverse positions; counts then reflect relative strength, not one game.

`arena_match*` returns net A's (win, draw, loss) with A at `a_player`.
`candidate_winrate*` plays a candidate vs a reference from *both* colors
(removing first-move bias) and aggregates from the candidate's perspective;
`should_promote` applies the standard accept rule (win a clear majority of
decisive games).
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.core import TwoPlayerDiscreteEnv, Saveable
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv
from mojo_rl.planners.tree_search import (
    GenericGPUMCTS, GenericCPUMCTS, AlphaGoPUCT, NoNoise, SelfPlay,
)
from ..zero.evaluators import RandomOpponent
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.mcts_adapters_cpu import AZRepCPU, AZDynCPU, AZPredCPU
from .eval import EvalResult


@always_inline
def _xs(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


@always_inline
def _mptr(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(b.unsafe_ptr())


@always_inline
def _argmax_legal[
    ACT: Int
](pol: InlineArray[Float64, ACT], legal: List[Bool]) -> Int:
    """argmax of a visit-count policy over legal actions (0 fallback)."""
    var best = -1
    var bestv = Float64(-1.0)
    for a in range(ACT):
        if a < len(legal) and legal[a] and pol[a] > bestv:
            bestv = pol[a]
            best = a
    return best if best >= 0 else 0


def arena_match[
    ENV: GPUTwoPlayerDiscreteEnv,
    NETA: Module,
    NETB: Module,
    N_GAMES: Int,
    RESULT_IDX: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut a: NETA,
    mut b: NETB,
    a_player: Int = 0,
    seed: UInt64 = 1,
    open_plies: Int = 0,
) raises -> EvalResult:
    """Greedy net A vs greedy net B (A at `a_player`). Returns A's record."""
    comptime OBS = NETA.IN_DIMS[0]
    comptime ACT = NETA.OUT_DIM - 1
    comptime W = NETA.OUT_DIM
    comptime STATE = ENV.STATE_SIZE

    a.set_attr["training"](Scalar[DT](0.0))
    b.set_attr["training"](Scalar[DT](0.0))

    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var pred_dev = ctx.enqueue_create_buffer[DT](N_GAMES * W)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var actions_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var legal_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * ACT)
    var pred_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * W)
    var actions_h = ctx.enqueue_create_host_buffer[DT](N_GAMES)
    var states_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * STATE)
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_GAMES, STATE](ctx, states)
    ctx.synchronize()

    var obs_t = TileTensor(_mptr(obs_dev), row_major[N_GAMES, OBS]())
    var pred_t = TileTensor(_mptr(pred_dev), row_major[N_GAMES, W]())
    var rng = seed | 1

    for ply in range(MAX_PLIES):
        ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()

        if ply < open_plies:
            rng = _xs(rng)
            RandomOpponent.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, actions_dev, legal_dev, states, rng
            )
            ctx.synchronize()
        else:
            # Whose greedy turn is it? A moves when (ply % 2) == a_player.
            ctx.enqueue_copy(legal_h, legal_dev)
            ctx.synchronize()
            var a_turn = (ply % 2) == a_player
            if a_turn:
                a.forward["gpu", N_GAMES](obs_t, output=pred_t)
            else:
                b.forward["gpu", N_GAMES](obs_t, output=pred_t)
            ctx.enqueue_copy(pred_h, pred_dev)
            ctx.synchronize()
            for e in range(N_GAMES):
                var best = -1
                var bestv = Float64(-1e30)
                for act in range(ACT):
                    if Float64(legal_h.unsafe_ptr()[e * ACT + act]) > 0.5:
                        var v = Float64(pred_h.unsafe_ptr()[e * W + act])
                        if v > bestv:
                            bestv = v
                            best = act
                actions_h.unsafe_ptr()[e] = Scalar[DT](best if best >= 0 else 0)
            ctx.enqueue_copy(actions_dev, actions_h)
            ctx.synchronize()

        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, actions_dev, rew, done, term, obs_next, legal_next
        )
        ctx.synchronize()

    ctx.enqueue_copy(states_h, states)
    ctx.synchronize()

    var a_win = a_player + 1
    var a_loss = (1 - a_player) + 1
    var wins = 0
    var draws = 0
    var losses = 0
    for e in range(N_GAMES):
        var r = Int(Float64(states_h.unsafe_ptr()[e * STATE + RESULT_IDX]))
        if r == a_win:
            wins += 1
        elif r == a_loss:
            losses += 1
        else:
            draws += 1
    return EvalResult(wins=wins, draws=draws, losses=losses)


def arena_match_mcts[
    ENV: GPUTwoPlayerDiscreteEnv,
    NETA: Module,
    NETB: Module,
    N_GAMES: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut a: NETA,
    mut b: NETB,
    a_player: Int = 0,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> EvalResult:
    """Net A vs net B, BOTH at full MCTS strength (temp=0, NoNoise). A plays
    color `a_player`. Mirrors the legacy `arena_compare_gpu` / eval MCTS pattern:
    per-env done-tracking + selective reset (so early-finishing games stop
    interfering), attribution by reward + whose-turn (color-agnostic). Returns
    A's (win, draw, loss). `open_plies` randomises the opening for diversity."""
    comptime OBS = NETA.IN_DIMS[0]
    comptime ACT = NETA.OUT_DIM - 1
    comptime STATE = ENV.STATE_SIZE
    comptime ARENA_MCTS = GenericGPUMCTS[
        N_GAMES, ACT, OBS, 1, MAX_NODES, NUM_SIMS, 1,
        AlphaGoPUCT[2.5], NoNoise, SelfPlay, STATE_SIZE=STATE,
    ]
    comptime MAX_ARENA_MOVES = MAX_PLIES + ACT

    a.set_attr["training"](Scalar[DT](0.0))
    b.set_attr["training"](Scalar[DT](0.0))

    var mcts = ARENA_MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)
    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var actions_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var done_h = ctx.enqueue_create_host_buffer[DT](N_GAMES)
    var rew_h = ctx.enqueue_create_host_buffer[DT](N_GAMES)
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

    while not all_done and move < MAX_ARENA_MOVES:
        var a_turn = (move % 2) == a_player
        if move < open_plies:
            # Random opening for both sides → diverse positions across the batch.
            rng = _xs(rng)
            RandomOpponent.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, actions_dev, legal_dev, states, rng
            )
        elif a_turn:
            var pred = AZPredGPU[OBS, ACT, NETA].make(a)
            var env_ad = AZEnvGPU[ENV, STATE, OBS, ACT]()
            var root_obs = LayoutTensor[
                DT, Layout.row_major(N_GAMES, OBS), MutAnyOrigin
            ](obs_dev.unsafe_ptr())
            var root_legal = LayoutTensor[
                DT, Layout.row_major(N_GAMES * ACT), MutAnyOrigin
            ](legal_dev.unsafe_ptr())
            mcts.search_gpu_alphazero[type_of(pred), type_of(env_ad)](
                ctx, pred, env_ad, root_obs, states, root_legal,
                rng_seed=seed + UInt64(move),
            )
            ctx.enqueue_copy(actions_dev, mcts.actions_out)
        else:
            var pred = AZPredGPU[OBS, ACT, NETB].make(b)
            var env_ad = AZEnvGPU[ENV, STATE, OBS, ACT]()
            var root_obs = LayoutTensor[
                DT, Layout.row_major(N_GAMES, OBS), MutAnyOrigin
            ](obs_dev.unsafe_ptr())
            var root_legal = LayoutTensor[
                DT, Layout.row_major(N_GAMES * ACT), MutAnyOrigin
            ](legal_dev.unsafe_ptr())
            mcts.search_gpu_alphazero[type_of(pred), type_of(env_ad)](
                ctx, pred, env_ad, root_obs, states, root_legal,
                rng_seed=seed + UInt64(move) * 3 + 1,
            )
            ctx.enqueue_copy(actions_dev, mcts.actions_out)
        ctx.synchronize()

        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, actions_dev, rew, done, term, obs_next, legal_next
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        all_done = True
        for e in range(N_GAMES):
            if not eval_done[e] and Float64(done_h.unsafe_ptr()[e]) > 0.5:
                eval_done[e] = True
                var r = Float64(rew_h.unsafe_ptr()[e])
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
            draws += 1  # 3 (draw) or 0 (never finished) → draw
    return EvalResult(wins=wins, draws=draws, losses=losses)


def candidate_winrate[
    ENV: GPUTwoPlayerDiscreteEnv,
    CAND: Module,
    BEST: Module,
    N_PER_COLOR: Int,
    RESULT_IDX: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut cand: CAND,
    mut best: BEST,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> EvalResult:
    """Candidate vs best from BOTH colors (`N_PER_COLOR` games each), aggregated
    from the candidate's perspective. Random openings remove the single-line
    degeneracy and the color swap removes first-move bias."""
    var as_p0 = arena_match[
        ENV, CAND, BEST, N_PER_COLOR, RESULT_IDX, MAX_PLIES
    ](ctx, cand, best, a_player=0, seed=seed, open_plies=open_plies)
    var as_p1 = arena_match[
        ENV, CAND, BEST, N_PER_COLOR, RESULT_IDX, MAX_PLIES
    ](ctx, cand, best, a_player=1, seed=seed + 1, open_plies=open_plies)
    return EvalResult(
        wins=as_p0.wins + as_p1.wins,
        draws=as_p0.draws + as_p1.draws,
        losses=as_p0.losses + as_p1.losses,
    )


def candidate_winrate_mcts[
    ENV: GPUTwoPlayerDiscreteEnv,
    CAND: Module,
    BEST: Module,
    N_PER_COLOR: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut cand: CAND,
    mut best: BEST,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> EvalResult:
    """Candidate vs best, both at full **MCTS** strength, from BOTH colors
    (`N_PER_COLOR` games each), aggregated from the candidate's perspective. This
    is the principled AlphaZero promotion gate — it compares the nets under the
    self-play generation condition (net+MCTS) and is sensitive to value-head
    quality, unlike the argmax `candidate_winrate`."""
    var as_p0 = arena_match_mcts[
        ENV, CAND, BEST, N_PER_COLOR, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](ctx, cand, best, a_player=0, seed=seed, open_plies=open_plies)
    var as_p1 = arena_match_mcts[
        ENV, CAND, BEST, N_PER_COLOR, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](ctx, cand, best, a_player=1, seed=seed + 1, open_plies=open_plies)
    return EvalResult(
        wins=as_p0.wins + as_p1.wins,
        draws=as_p0.draws + as_p1.draws,
        losses=as_p0.losses + as_p1.losses,
    )


def arena_match_cpu[
    ENV: TwoPlayerDiscreteEnv & Saveable & Defaultable & ImplicitlyDestructible,
    NETA: Module,
    NETB: Module,
    N_GAMES: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](
    ctx_unused: Int,
    mut a: NETA,
    mut b: NETB,
    a_player: Int = 0,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> EvalResult:
    """CPU twin of `arena_match_mcts`: net A vs net B, both at full
    `GenericCPUMCTS` strength (temp=0, NoNoise), single env, `N_GAMES` games. A
    plays color `a_player`; `open_plies` random opening moves diversify. Returns
    A's record. `ctx_unused` keeps the call shape parallel to the GPU arena —
    pass 0."""
    comptime OBS = NETA.IN_DIMS[0]
    comptime ACT = NETA.OUT_DIM - 1
    comptime LATENT = ENV.SAVE_SIZE
    comptime AMCTS = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES, AlphaGoPUCT[2.5], NoNoise, SelfPlay,
    ]
    _ = ctx_unused
    a.set_attr["training"](Scalar[DT](0.0))
    b.set_attr["training"](Scalar[DT](0.0))

    var env = ENV()
    var root_save = alloc[Scalar[DT]](LATENT)
    var env_ptr = UnsafePointer(to=env)
    var rep = AZRepCPU[ENV, OBS](env=env_ptr)
    var dyn = AZDynCPU[ENV, ACT](env=env_ptr)
    var pred_a = AZPredCPU[ENV, OBS, ACT, NETA](
        env=env_ptr, net=UnsafePointer(to=a)
    )
    var pred_b = AZPredCPU[ENV, OBS, ACT, NETB](
        env=env_ptr, net=UnsafePointer(to=b)
    )
    var mcts = AMCTS(gamma=1.0)

    var wins = 0
    var draws = 0
    var losses = 0
    var rng = seed | 1

    for _g in range(N_GAMES):
        _ = env.reset()
        var ply = 0
        while env.game_result() == 0 and ply < MAX_PLIES:
            var act = 0
            var legal = env.legal_action_mask()
            if ply < open_plies:
                rng = _xs(rng)
                act = RandomOpponent.select_action_cpu[ENV](env, rng)
            elif env.current_player() == a_player:
                env.save_env_state(root_save)
                var root_obs = List[Float64](length=OBS, fill=Float64(0.0))
                var pol = mcts.search[
                    AZRepCPU[ENV, OBS],
                    AZDynCPU[ENV, ACT],
                    AZPredCPU[ENV, OBS, ACT, NETA],
                ](rep, dyn, pred_a, root_obs, add_noise=False, legal_mask=legal)
                env.load_env_state(root_save)
                act = _argmax_legal[ACT](pol, legal)
            else:
                env.save_env_state(root_save)
                var root_obs = List[Float64](length=OBS, fill=Float64(0.0))
                var pol = mcts.search[
                    AZRepCPU[ENV, OBS],
                    AZDynCPU[ENV, ACT],
                    AZPredCPU[ENV, OBS, ACT, NETB],
                ](rep, dyn, pred_b, root_obs, add_noise=False, legal_mask=legal)
                env.load_env_state(root_save)
                act = _argmax_legal[ACT](pol, legal)
            _ = env.step(env.action_from_index(act))
            ply += 1

        var gr = env.game_result()
        var a_win = a_player + 1
        var a_loss = (1 - a_player) + 1
        if gr == a_win:
            wins += 1
        elif gr == a_loss:
            losses += 1
        else:
            draws += 1

    root_save.free()
    return EvalResult(wins=wins, draws=draws, losses=losses)


def candidate_winrate_cpu[
    ENV: TwoPlayerDiscreteEnv & Saveable & Defaultable & ImplicitlyDestructible,
    CAND: Module,
    BEST: Module,
    N_PER_COLOR: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](
    mut cand: CAND,
    mut best: BEST,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> EvalResult:
    """CPU twin of `candidate_winrate_mcts`: candidate vs best, both at full CPU
    MCTS strength, both colors, aggregated from the candidate's perspective."""
    var as_p0 = arena_match_cpu[
        ENV, CAND, BEST, N_PER_COLOR, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](0, cand, best, a_player=0, seed=seed, open_plies=open_plies)
    var as_p1 = arena_match_cpu[
        ENV, CAND, BEST, N_PER_COLOR, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](0, cand, best, a_player=1, seed=seed + 1, open_plies=open_plies)
    return EvalResult(
        wins=as_p0.wins + as_p1.wins,
        draws=as_p0.draws + as_p1.draws,
        losses=as_p0.losses + as_p1.losses,
    )


def should_promote(
    rec: EvalResult, threshold: Float64 = 0.55, min_decisive: Int = 1
) -> Bool:
    """Standard AlphaZero accept rule: among decisive games (draws ignored),
    the candidate must win at least `threshold` fraction — and there must be at
    least `min_decisive` decisive games (else a 0–0 tie spuriously promotes)."""
    var decisive = rec.wins + rec.losses
    if decisive < min_decisive:
        return False
    return Float64(rec.wins) >= threshold * Float64(decisive)
