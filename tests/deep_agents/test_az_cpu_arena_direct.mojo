"""Direct gate for the CPU arena driver on the storage surface
(`run_alphazero_selfplay_arena_cpu`, bypassing the agent facade): best/learner
split + arena promotion (hard_copy) + D4 augmentation, single-env CPU MCTS.
Asserts a finite training loss, ≥1 promotion, and the best clearly beats random.

Run: pixi run mojo run -I . tests/deep_agents/test_az_cpu_arena_direct.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay_arena_cpu import (
    run_alphazero_selfplay_arena_cpu,
)
from mojo_rl.deep_agents.alphazero.eval import eval_policy_vs_random_cpu
from mojo_rl.deep_agents.zero.symmetries import D4SquareAugmenter
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Aug = D4SquareAugmenter[3, 3]
    comptime N_EVAL = 200
    comptime MAX_PLIES = 9

    var best = Net.make["cpu", Kaiming]()

    var before = eval_policy_vs_random_cpu[Env, Net, N_EVAL, MAX_PLIES](
        best, agent_player=0, seed=12345
    )
    print("BEFORE  win=", before.wins, " draw=", before.draws,
          " loss=", before.losses, " (/", N_EVAL, ")")

    var res = run_alphazero_selfplay_arena_cpu[
        Env, Net, Aug, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=16384, MAX_TRAJ=16,
    ](
        best, iterations=1200, learning_starts=20, train_per_iter=2,
        lr=0.01, seed=7, arena_every=300, arena_open_plies=2,
        promote_threshold=0.55,
    )
    print("cpu arena run  last_loss=", res.last_loss, " promotions=", res.promotions)

    var after = eval_policy_vs_random_cpu[Env, Net, N_EVAL, MAX_PLIES](
        best, agent_player=0, seed=12345
    )
    print("AFTER   win=", after.wins, " draw=", after.draws,
          " loss=", after.losses, " (/", N_EVAL, ")")

    assert_true(
        res.last_loss == res.last_loss and res.last_loss < 1e30,
        "CPU arena produced non-finite loss",
    )
    assert_true(res.promotions >= 1, "learner was never promoted over best")
    assert_true(
        after.losses < before.losses * 2 // 3,
        "CPU arena best did not clearly improve vs random",
    )
    print("AZ CPU arena-gated self-play + D4 augmentation: OK")
