"""AlphaZero training on Connect Four — fully GPU.

CNN architecture with 128 filters, batch-then-train, GPU eval vs Random.
Based on alpha-zero-general parameters:
  SIMS=25, cpuct=1.0, LR=0.001/Adam, 2x symmetries (horizontal flip)

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero.mojo
    pixi run -e apple mojo run -I . examples/board_games/connect_four_alphazero.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourCNNConfig,
    AlphaZeroConnectFourConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print("=== AlphaZero on Connect Four ===")
    print("CNN 128 filters | SIMS=25 | cpuct=1.0 | 2x symmetries")
    print()

    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    # CNN config (matches alpha-zero-general style)
    comptime Config = AlphaZeroConnectFourCNNConfig[]
    # MLP alternative:
    # comptime Config = AlphaZeroConnectFourConfig[]

    var agent = GenericAlphaZeroAgent[Config, 64]()

    _ = agent.train_selfplay_gpu[C4, RandomOpponent](
        ctx,
        num_iters=100,
        steps_per_iter=2000,     # ~100+ games per iter (longer games than TTT)
        train_epochs=10,
        warmup_iters=1,
        arena_threshold=0.5,
        do_eval=True,
        do_arena=True,
        checkpoint_every=10,
        checkpoint_path="connect_four_alphazero.ckpt",
    )

    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")
