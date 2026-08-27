"""Convergence gate: Gumbel AlphaZero LEARNS TicTacToe (GPU).

The Gumbel-planner mirror of `test_az_tictactoe_convergence`: same net, env,
budget and eval, but self-play runs `run_alphazero_gumbel_selfplay`
(Gumbel-Top-k roots + Sequential Halving + improved-policy targets) instead
of the PUCT/Dirichlet driver. The published Gumbel AlphaZero result is policy
improvement at LOW sim budgets, so the bar matches the PUCT baseline: losses
vs random at least halve from the untrained net and end below 25%.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_gumbel_tictactoe_convergence.mojo
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay_gumbel import (
    run_alphazero_gumbel_selfplay,
)
from mojo_rl.deep_agents.alphazero.eval import eval_policy_vs_random
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime Net = AZMLPNet[27, 9, 64]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime N_EVAL = 200
    comptime RESULT_IDX = 10   # TicTacToe state[10] = game_result
    comptime MAX_PLIES = 9

    var ctx = DeviceContext()
    var net = Net.make["gpu", Kaiming](Optional(ctx))

    var before = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, net, agent_player=0, seed=3
    )
    print(
        "BEFORE  win=", before.wins, " draw=", before.draws,
        " loss=", before.losses, " (/", N_EVAL, ")",
    )

    var last_loss = run_alphazero_gumbel_selfplay[
        Env, Net,
        N_ENVS=16, NUM_SIMS=24, MAX_NODES=64, MAX_K=4,
        BATCH=64, CAP=8192, MAX_TRAJ=16,
    ](ctx, net, iterations=1200, learning_starts=20, train_per_iter=2,
      lr=0.01, seed=7)
    print("train loss:", last_loss)

    var after = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, net, agent_player=0, seed=3
    )
    print(
        "AFTER   win=", after.wins, " draw=", after.draws,
        " loss=", after.losses, " (/", N_EVAL, ")",
    )

    assert_true(last_loss > 0.0, "training never ran")
    assert_true(
        after.losses < before.losses // 2,
        "losses vs random did not at least halve",
    )
    assert_true(after.losses < N_EVAL // 4, "loss rate vs random >= 25%")
    print("Gumbel AlphaZero TicTacToe convergence: OK")
