"""AlphaZero CPU arena parity — full train_arena on the CPU path, no GPU.

Validates `TARGET="cpu"` `train_arena`: best/learner split + CPU-MCTS Arena
gating (`candidate_winrate_cpu`) + D4 symmetry augmentation + two pluggable
`CPUEvaluator` opponents (minimax + random, the SAME structs used on GPU, now
dual-conforming) + a `CsvLogger`. Asserts the learner gets promoted, the metric
series is flushed, and the trained net beats random on a CPU greedy eval — the
CPU twin of `test_az_telemetry`. No `DeviceContext` is ever created.

Run (no GPU needed):
    pixi run mojo run -I . tests/deep_agents2/test_az_cpu_arena.mojo
"""

from std.memory import UnsafePointer
from std.testing import assert_true

from mojo_rl.core.logger import CsvLogger
from mojo_rl.deep_agents2.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents2.alphazero.agent import AlphaZeroAgent
from mojo_rl.deep_agents2.zero.symmetries import D4SquareAugmenter
from mojo_rl.deep_agents2.zero.evaluators import (
    RandomOpponent, GPUMinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime Net = AZMLPNet[27, 9, 64]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Aug = D4SquareAugmenter[3, 3]

    var agent = AlphaZeroAgent[
        "cpu", Env, Net, 1, 24, 64, 64, 16384, 16
    ](None, lr=0.01)

    var before = agent.eval_vs_random_cpu[200, 9](agent_player=0, seed=12345)
    print(
        "BEFORE  win=", before.wins, " draw=", before.draws,
        " loss=", before.losses,
    )

    var csv = CsvLogger(String("logs/az_cpu_arena.csv"), buffer_size=4)
    var res = agent.train_arena[
        AUG=Aug,
        OPP1=GPUMinimaxTicTacToe,
        OPP2=RandomOpponent,
        L=CsvLogger,
        ARENA_GAMES=16,
        RESULT_IDX=10,
        MAX_PLIES=9,
        EVAL_GAMES=16,
    ](
        iterations=900,
        learning_starts=20,
        train_per_iter=2,
        seed=7,
        arena_every=300,
        arena_open_plies=2,
        promote_threshold=0.55,
        report_every=300,
        do_eval=True,
        do_eval2=True,
        verbose=True,
        logger=UnsafePointer(to=csv),
    )
    csv.close()

    var after = agent.eval_vs_random_cpu[200, 9](agent_player=0, seed=12345)
    print(
        "AFTER   win=", after.wins, " draw=", after.draws,
        " loss=", after.losses,
    )
    print(
        "train_arena (cpu)  last_loss=", res.last_loss,
        " promotions=", res.promotions, " csv_rows=", csv.total_logged(),
    )

    # The learner must overtake the frozen best at least once.
    assert_true(res.promotions >= 1, "CPU learner was never promoted")
    # Finite loss + a flushed metric series (≥1 report × 12 scalars).
    assert_true(
        res.last_loss == res.last_loss and res.last_loss < 1e30,
        "CPU arena produced non-finite loss",
    )
    assert_true(
        csv.total_logged() >= 12,
        "CPU arena logger did not receive the per-report metric series",
    )
    # The CPU-trained net clearly improved vs random.
    assert_true(
        after.losses * 4 < before.losses * 3,
        "CPU arena-trained agent did not clearly improve vs random",
    )
    print("AZ CPU arena + D4 augmentation + minimax/random eval: OK")
