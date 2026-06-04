"""Arena — net-vs-net match harness for AlphaZero candidate/best gating.

Plays two nets head-to-head (both greedy argmax over legal moves) in lockstep
batched games. Greedy-vs-greedy from the fixed start is a single deterministic
line, so `open_plies` randomises the opening moves (both sides) to spread the
batch over diverse positions — the win/draw/loss counts then reflect relative
strength rather than one game.

`arena_match` returns net A's (win, draw, loss) with A at `a_player`.
`candidate_winrate` plays a candidate against a reference from *both* colors
(removing first-move bias) and reports the candidate's aggregate record, plus
`should_promote` applying the standard AlphaZero accept rule (win a clear
majority of decisive games).
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv
from ..zero.evaluators import RandomOpponent
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
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


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
