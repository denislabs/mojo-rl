"""Convergence: AlphaZero self-play learns TicTacToe with a CNN torso.

Exercises the full BatchNorm path end-to-end — the self-play driver toggles
`set_attr["training"]` (eval during MCTS inference, train during the update),
so BN running stats stabilise the conv backbone. As P0, optimal TicTacToe never
loses, so the greedy CNN policy's loss-rate vs random must fall.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_cnn_convergence.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZTicTacToeCNN
from mojo_rl.deep_agents.alphazero.selfplay import run_alphazero_selfplay
from mojo_rl.deep_agents.alphazero.eval import eval_policy_vs_random
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime Net = AZTicTacToeCNN[16, 32]   # 16 conv filters, 32-wide head
    comptime Env = TicTacToeEnv[DType.float64]
    comptime N_EVAL = 200
    comptime RESULT_IDX = 10
    comptime MAX_PLIES = 9

    var ctx = DeviceContext()
    var net = Net.make["gpu", INIT=Kaiming](ctx=ctx)

    var before = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, net, agent_player=0, seed=12345
    )
    print(
        "BEFORE  win=", before.wins, " draw=", before.draws,
        " loss=", before.losses, " (/", N_EVAL, ")",
    )

    var last_loss = run_alphazero_selfplay[
        Env, Net, N_ENVS=16, NUM_SIMS=24, MAX_NODES=64,
        BATCH=64, CAP=8192, MAX_TRAJ=16,
    ](ctx, net, iterations=1200, learning_starts=20, train_per_iter=2,
      lr=2e-3, seed=7)   # CNN: lower LR than the MLP path (deeper backbone)
    print("train last_loss =", last_loss)

    var after = eval_policy_vs_random[Env, Net, N_EVAL, RESULT_IDX, MAX_PLIES](
        ctx, net, agent_player=0, seed=12345
    )
    print(
        "AFTER   win=", after.wins, " draw=", after.draws,
        " loss=", after.losses, " (/", N_EVAL, ")",
    )

    # The CNN agent must clearly improve and reach a decent absolute loss-rate.
    assert_true(
        after.losses < before.losses // 2,
        "CNN agent did not clearly improve (loss-rate vs random did not halve)",
    )
    assert_true(
        after.losses < N_EVAL // 3,
        "CNN agent loss-rate vs random still too high (< 33% expected)",
    )
    print("AZ CNN TicTacToe convergence: OK")
