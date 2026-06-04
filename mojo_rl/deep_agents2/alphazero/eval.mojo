"""AlphaZero evaluation — greedy net-policy vs a random opponent (batched GPU).

`eval_policy_vs_random` plays `N_GAMES` complete games in parallel: on the
agent's plies it picks ``argmax`` of the net's policy head over legal moves
(pure policy, no MCTS — tests what the network learned); on the opponent's plies
it picks a uniform random legal move. Games run in lockstep for ``MAX_PLIES``
(no resets, so `current_player` stays synchronized across the batch); finished
games no-op on further steps. Returns the agent's (wins, draws, losses).

In TicTacToe, optimal play as P0 never loses, so a learning agent's loss-rate
vs random should fall toward zero — the convergence signal.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.core.env_traits import GPUTwoPlayerDiscreteEnv


@always_inline
def _xs(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


@always_inline
def _mptr(
    b: DeviceBuffer[DT]
) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    """Origin-erase a device buffer pointer so `Module.forward` (which pins
    `origin=MutAnyOrigin` on its output) accepts the tile."""
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


@fieldwise_init
struct EvalResult(Copyable, Movable):
    var wins: Int
    var draws: Int
    var losses: Int


def eval_policy_vs_random[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    N_GAMES: Int,
    RESULT_IDX: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut net: NET,
    agent_player: Int = 0,
    seed: UInt64 = 1,
) raises -> EvalResult:
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM
    comptime STATE = ENV.STATE_SIZE

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
        ctx.enqueue_copy(legal_h, legal_dev)
        ctx.synchronize()

        var agent_turn = (ply % 2) == agent_player
        if agent_turn:
            net.forward["gpu", N_GAMES](obs_t, output=pred_t)
            ctx.enqueue_copy(pred_h, pred_dev)
            ctx.synchronize()
            for e in range(N_GAMES):
                var best = -1
                var bestv = Float64(-1e30)
                for a in range(ACT):
                    if Float64(legal_h.unsafe_ptr()[e * ACT + a]) > 0.5:
                        var v = Float64(pred_h.unsafe_ptr()[e * W + a])
                        if v > bestv:
                            bestv = v
                            best = a
                actions_h.unsafe_ptr()[e] = Scalar[DT](best if best >= 0 else 0)
        else:
            for e in range(N_GAMES):
                var cnt = 0
                for a in range(ACT):
                    if Float64(legal_h.unsafe_ptr()[e * ACT + a]) > 0.5:
                        cnt += 1
                if cnt == 0:
                    actions_h.unsafe_ptr()[e] = Scalar[DT](0)
                else:
                    rng = _xs(rng)
                    var pick = Int(rng % UInt64(cnt))
                    var chosen = 0
                    var seen = 0
                    for a in range(ACT):
                        if Float64(legal_h.unsafe_ptr()[e * ACT + a]) > 0.5:
                            if seen == pick:
                                chosen = a
                                break
                            seen += 1
                    actions_h.unsafe_ptr()[e] = Scalar[DT](chosen)

        ctx.enqueue_copy(actions_dev, actions_h)
        ctx.synchronize()
        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, actions_dev, rew, done, term, obs_next, legal_next
        )
        ctx.synchronize()

    ctx.enqueue_copy(states_h, states)
    ctx.synchronize()

    var agent_win = agent_player + 1
    var agent_loss = (1 - agent_player) + 1
    var wins = 0
    var draws = 0
    var losses = 0
    for e in range(N_GAMES):
        var r = Int(Float64(states_h.unsafe_ptr()[e * STATE + RESULT_IDX]))
        if r == agent_win:
            wins += 1
        elif r == agent_loss:
            losses += 1
        else:
            draws += 1
    return EvalResult(wins=wins, draws=draws, losses=losses)
