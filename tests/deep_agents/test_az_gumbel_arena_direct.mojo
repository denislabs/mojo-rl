"""Direct gate for the GPU Gumbel arena driver on the storage surface
(`run_alphazero_selfplay_arena_gumbel`, bypassing the agent facade): best/learner
split + arena promotion (hard_copy) + D4 symmetry augmentation, Gumbel-Top-k roots
+ Sequential Halving + improved-policy targets. Asserts the gated run promotes the
learner at least once and the best clearly beats random.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_gumbel_arena_direct.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay_arena_gumbel import (
    run_alphazero_selfplay_arena_gumbel,
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

    var res = run_alphazero_selfplay_arena_gumbel[
        Env, Net, Aug, N_ENVS=16, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=16384, MAX_TRAJ=16, MAX_K=4,
    ](
        ctx, best, iterations=1000, learning_starts=20, train_per_iter=2,
        lr=0.01, seed=7, arena_every=250, arena_open_plies=2,
        promote_threshold=0.55,
    )
    print("gumbel arena run  last_loss=", res.last_loss, " promotions=", res.promotions)

    var after = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, best, agent_player=0, seed=12345
    )
    print(
        "AFTER   win=", after.wins, " draw=", after.draws,
        " loss=", after.losses, " (/", N_EVAL, ")",
    )

    assert_true(res.promotions >= 1, "learner was never promoted over best")
    assert_true(
        after.losses < before.losses // 2,
        "gumbel arena best did not clearly improve vs random",
    )
    assert_true(
        after.losses < N_EVAL // 4,
        "gumbel arena best loss-rate vs random still too high (< 25%)",
    )
    print("AZ Gumbel arena-gated self-play + D4 augmentation: OK")
