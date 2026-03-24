"""AlphaZero training on Connect Four — fully GPU.

ResNet architecture with 4 residual blocks, 128 filters, 100 MCTS sims.
Closer to original AlphaZero paper than alpha-zero-general's shallow search.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero.mojo
    pixi run -e apple mojo run -I . examples/board_games/connect_four_alphazero.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourResNetConfig,
    AlphaZeroConnectFourCNNConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print("=== AlphaZero on Connect Four ===")
    print("ResNet 4 blocks × 128 filters | SIMS=100 | cpuct=1.0")
    print()

    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    # ResNet config (closer to original AlphaZero)
    comptime Config = AlphaZeroConnectFourResNetConfig[]
    # Lighter CNN alternative:
    # comptime Config = AlphaZeroConnectFourCNNConfig[]

    var agent = GenericAlphaZeroAgent[Config, 64]()

    _ = agent.train_selfplay_gpu[C4, RandomOpponent](
        ctx,
        num_iters=200,
        steps_per_iter=2000,     # ~100+ games per iter
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
