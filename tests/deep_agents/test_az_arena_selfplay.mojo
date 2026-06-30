"""Arena-gated AlphaZero self-play + D4 symmetry augmentation on TicTacToe.

End-to-end test of the "full AlphaZero" driver: best/learner split, periodic
Arena accept/reject promotion, and 8× D4 symmetry augmentation of the recorded
targets. Asserts the gated run (a) accepts the learner at least once and (b)
produces a best net that clearly beats random.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_arena_selfplay.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay_arena import (
    run_alphazero_selfplay_arena,
)
from mojo_rl.deep_agents.alphazero.eval import eval_policy_vs_random
from mojo_rl.deep_agents.zero.symmetries import D4SquareAugmenter
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Aug = D4SquareAugmenter[3, 3]   # TicTacToe: 8 symmetries
    comptime N_EVAL = 200
    comptime RESULT_IDX = 10
    comptime MAX_PLIES = 9

    var ctx = DeviceContext()
    var best = Net.make["gpu", Kaiming](Optional(ctx))

    var before = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, best, agent_player=0, seed=12345
    )
    print(
        "BEFORE  win=", before.wins, " draw=", before.draws,
        " loss=", before.losses, " (/", N_EVAL, ")",
    )

    var res = run_alphazero_selfplay_arena[
        Env, Net, Aug, N_ENVS=16, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=16384, MAX_TRAJ=16,
    ](
        ctx, best, iterations=1000, learning_starts=20, train_per_iter=2,
        lr=0.01, seed=7, arena_every=250, arena_open_plies=2,
        promote_threshold=0.55,
    )
    print("arena run  last_loss=", res.last_loss, " promotions=", res.promotions)

    # `best` now holds the promoted weights — evaluate it vs random.
    var after = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, best, agent_player=0, seed=12345
    )
    print(
        "AFTER   win=", after.wins, " draw=", after.draws,
        " loss=", after.losses, " (/", N_EVAL, ")",
    )

    # The learner must have been accepted at least once (it starts equal to the
    # best, then trains while the best is frozen → must overtake it).
    assert_true(res.promotions >= 1, "learner was never promoted over best")
    # The promoted best must clearly beat random.
    assert_true(
        after.losses < before.losses // 2,
        "arena-gated best did not clearly improve vs random",
    )
    assert_true(
        after.losses < N_EVAL // 4,
        "arena-gated best loss-rate vs random still too high (< 25%)",
    )
    print("AZ arena-gated self-play + D4 augmentation: OK")
