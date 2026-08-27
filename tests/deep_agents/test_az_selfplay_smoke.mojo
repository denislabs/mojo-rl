"""Smoke: AlphaZero self-play training loop runs end-to-end on TicTacToe.

Exercises the full pipeline — GPU MCTS self-play → trajectory recording →
outcome-z assignment → replay → GPU graph training — for a short run, and
asserts training actually triggered (finite, positive mean loss).

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_selfplay_smoke.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay import run_alphazero_selfplay
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]

    var ctx = DeviceContext()
    var net = Net.make["gpu", Kaiming](Optional(ctx))

    var last_loss = run_alphazero_selfplay[
        Env, Net,
        N_ENVS=16, NUM_SIMS=16, MAX_NODES=64,
        BATCH=32, CAP=4096, MAX_TRAJ=16,
    ](ctx, net, iterations=120, learning_starts=10, train_per_iter=1,
      lr=0.01, seed=1)

    print("AZ self-play smoke: last_loss =", last_loss)
    assert_true(last_loss == last_loss, "last_loss is NaN")
    assert_true(last_loss > 0.0, "training never triggered (replay never filled)")
    _ = net^
    print("AZ self-play smoke: OK")
