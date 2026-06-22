"""Convergence: AlphaZero self-play learns to play TicTacToe.

Measures the greedy net-policy's win/draw/loss vs a random opponent (agent as
P0) BEFORE and AFTER a self-play training run. As P0, optimal TicTacToe never
loses, so a learning agent's loss-rate vs random must fall.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_tictactoe_convergence.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay import run_alphazero_selfplay
from mojo_rl.deep_agents.alphazero.eval import eval_policy_vs_random
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime N_EVAL = 200
    comptime RESULT_IDX = 10   # TicTacToe state[10] = game_result
    comptime MAX_PLIES = 9

    var ctx = DeviceContext()
    var net = Net.make["gpu", Kaiming](Optional(ctx))

    var before = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, net, agent_player=0, seed=12345
    )
    print(
        "BEFORE  win=", before.wins, " draw=", before.draws,
        " loss=", before.losses, " (/", N_EVAL, ")",
    )

    var last_loss = run_alphazero_selfplay[
        Env, Net,
        N_ENVS=16, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=8192, MAX_TRAJ=16,
    ](ctx, net, iterations=1200, learning_starts=20, train_per_iter=2,
      lr=0.01, seed=7)
    print("train last_loss =", last_loss)

    var after = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, net, agent_player=0, seed=12345
    )
    print(
        "AFTER   win=", after.wins, " draw=", after.draws,
        " loss=", after.losses, " (/", N_EVAL, ")",
    )

    # Observed (seed-fixed, ~deterministic): 93 → 24 losses / 200. Guard with
    # margin for minor GPU reduction-ordering nondeterminism: the agent must
    # both clearly improve and reach a decent absolute loss-rate (< 25%).
    assert_true(
        after.losses < before.losses // 2,
        "agent did not clearly improve (loss-rate vs random did not halve)",
    )
    assert_true(
        after.losses < N_EVAL // 4,
        "agent loss-rate vs random still too high (< 25% expected)",
    )
    print("AZ TicTacToe convergence: OK")
