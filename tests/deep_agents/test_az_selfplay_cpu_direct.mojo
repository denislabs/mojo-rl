"""Direct gate for the CPU AlphaZero self-play driver on the storage surface
(bypasses the AlphaZeroAgent facade, which pulls in the not-yet-migrated GPU
drivers). Drives `run_alphazero_selfplay_cpu` on TicTacToe end-to-end: CPU MCTS
via the AZ*CPU adapters (List-based env save/load), the example replay (List ring
buffer, RAII), and the storage AZ loss graph (`forward/vjp["cpu"]` + Adam).

Asserts the returned mean train loss is finite and has clearly dropped from the
~2.9 AZ-loss floor — i.e. the whole CPU path (search → record → train) is wired.

Run: pixi run mojo run -I . tests/deep_agents/test_az_selfplay_cpu_direct.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.selfplay_cpu import run_alphazero_selfplay_cpu
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime H = 64
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]

    var net = Net.make["cpu", Kaiming]()
    var loss = run_alphazero_selfplay_cpu[
        Env, Net, 24, 64, 64, 16384, 16
    ](net, iterations=700, learning_starts=20, train_per_iter=2, seed=7)

    print("CPU direct self-play  last_loss=", loss)
    assert_true(
        loss == loss and loss < 1e30, "CPU self-play produced non-finite loss"
    )
    assert_true(
        loss < 2.5,
        "CPU self-play loss did not drop from the ~2.9 floor (last=" + String(loss) + ")",
    )
    _ = net^
    print("AZ selfplay_cpu direct: OK")
