"""AlphaZero telemetry: 2 pluggable evaluators + per-report printing + Logger.

Exercises the production "facade" features ported from the legacy
`train_selfplay_gpu`: `AlphaZeroAgent.train_arena` with two `GPUEvaluator`
opponents (minimax as the primary signal, random as the secondary), a periodic
progress print, and a `CsvLogger` sink that captures the per-report metrics.

Asserts the run completes, the logger received the expected scalar series
(loss / replay_size / promotions / eval{1,2}_*), and the printed/flushed numbers
are well-formed. Convergence itself is covered elsewhere — this is the wiring.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents2/test_az_telemetry.mojo
"""

from std.memory import UnsafePointer
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.core.logger import CsvLogger
from mojo_rl.deep_agents2.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents2.alphazero.agent import AlphaZeroAgent
from mojo_rl.deep_agents2.zero.symmetries import D4SquareAugmenter
from mojo_rl.deep_agents2.zero.evaluators import (
    RandomOpponent, GPUMinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Aug = D4SquareAugmenter[3, 3]
    comptime RESULT_IDX = 10
    comptime MAX_PLIES = 9

    var ctx = DeviceContext()
    var agent = AlphaZeroAgent[
        Env, Net, N_ENVS=16, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=16384, MAX_TRAJ=16,
    ].make(ctx, lr=0.01)

    var csv_path = String("logs/az_telemetry_test.csv")
    var logger = CsvLogger(csv_path, buffer_size=4)

    # Minimax as the primary eval (the "never lose vs perfect play" signal),
    # random as the secondary. Report every 100 iters with diverse openings so
    # the deterministic minimax line is not the only game played.
    var res = agent.train_arena[
        AUG=Aug,
        OPP1=GPUMinimaxTicTacToe,
        OPP2=RandomOpponent,
        L=CsvLogger,
        ARENA_GAMES=32,
        RESULT_IDX=RESULT_IDX,
        MAX_PLIES=MAX_PLIES,
        EVAL_GAMES=32,
    ](
        iterations=400,
        learning_starts=20,
        train_per_iter=2,
        seed=7,
        arena_every=200,
        arena_open_plies=2,
        promote_threshold=0.55,
        report_every=100,
        do_eval=True,
        do_eval2=True,
        verbose=True,
        logger=UnsafePointer(to=logger),
    )
    logger.close()

    print("train_arena  last_loss=", res.last_loss, " promotions=", res.promotions)
    print("csv logged rows:", logger.total_logged())

    # The run must have produced finite loss and reached the report cadence at
    # least 3× (iters 100/200/300/400 with learning_starts=20 → ≥3 reports).
    assert_true(
        res.last_loss == res.last_loss and res.last_loss < 1e30,
        "telemetry run produced non-finite loss",
    )
    # Each report flushes: loss, games, replay_size, promotions (4) + eval1 (4)
    # + eval2 (4) = 12 scalars. With ≥3 reports that is ≥36 logged rows.
    assert_true(
        logger.total_logged() >= 36,
        "logger did not receive the expected per-report metric series",
    )
    print("AZ telemetry (2 evaluators + logger + per-report print): OK")
