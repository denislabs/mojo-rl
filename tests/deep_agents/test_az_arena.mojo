"""Arena net-vs-net harness: a trained net beats a fresh net.

Validates `arena_match` / `candidate_winrate` / `should_promote`: a net trained
by self-play should clearly out-score a random-init net across diverse openings
and from both colors, so the accept rule fires.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_arena.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay import run_alphazero_selfplay
from mojo_rl.deep_agents.alphazero.arena import (
    arena_match, candidate_winrate, should_promote,
)
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime RESULT_IDX = 10
    comptime MAX_PLIES = 9
    comptime NG = 64

    var ctx = DeviceContext()
    var trained = Net.make["gpu", Kaiming](Optional(ctx))
    var fresh = Net.make["gpu", Kaiming](Optional(ctx))

    _ = run_alphazero_selfplay[
        Env, Net, N_ENVS=16, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=8192, MAX_TRAJ=16,
    ](ctx, trained, iterations=1000, learning_starts=20, train_per_iter=2,
      lr=0.01, seed=7)

    # Single-color arena (trained=P0 vs fresh=P1), diverse openings.
    var p0 = arena_match[Env, Net, Net, NG, RESULT_IDX, MAX_PLIES](
        ctx, trained, fresh, a_player=0, seed=123, open_plies=2
    )
    print("trained(P0) vs fresh  win=", p0.wins, " draw=", p0.draws,
          " loss=", p0.losses)

    # Both-color aggregate + accept rule.
    var rec = candidate_winrate[
        Env, Net, Net, NG, RESULT_IDX, MAX_PLIES
    ](ctx, trained, fresh, seed=123, open_plies=2)
    print("trained vs fresh (both colors)  win=", rec.wins,
          " draw=", rec.draws, " loss=", rec.losses)

    # A clearly-stronger candidate must win more than it loses and pass accept.
    assert_true(
        rec.wins > rec.losses,
        "trained net did not out-score fresh net in the arena",
    )
    assert_true(
        should_promote(rec, threshold=0.55, min_decisive=NG),
        "accept rule rejected a clearly-stronger candidate",
    )
    # Symmetric sanity: fresh should NOT be promoted over trained.
    var rev = candidate_winrate[
        Env, Net, Net, NG, RESULT_IDX, MAX_PLIES
    ](ctx, fresh, trained, seed=123, open_plies=2)
    assert_true(
        not should_promote(rev, threshold=0.55, min_decisive=NG),
        "accept rule wrongly promoted the weaker net",
    )
    print("AZ arena harness: OK")
