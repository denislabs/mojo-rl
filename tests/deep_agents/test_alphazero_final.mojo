"""AlphaZero — fully GPU training, arena, and evaluation.

Everything runs on GPU:
- Self-play with GPU MCTS
- Training (forward/backward/optimizer)
- Arena comparison (GPU MCTS temp=0, new vs old)
- Evaluation vs Random (GPU MCTS vs random legal actions)
"""

from std.gpu.host import DeviceContext
from mojo_rl.nn.model import (
    Linear, LinearReLU, Sequential, Parallel,
    Conv2DReLU, FlattenLayer,
)
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
    AlphaZeroTicTacToeCNNConfig,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero — Fully GPU ===")
    print("Self-play + training + GPU arena + GPU eval vs Random")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    # Switch between MLP and CNN configs here
    comptime Config = AlphaZeroTicTacToeCNNConfig[]
    # comptime Config = AlphaZeroTicTacToeConfig[]

    var agent = GenericAlphaZeroAgent[Config, 64]()

    # Single call — everything on GPU, no CPU round-trips
    _ = agent.train_selfplay_gpu[TTT](
        ctx,
        num_steps=250000,
        warmup_steps=500,
        gradient_steps=4,
        print_every=25000,
        arena_every=25000,
        arena_games=64,
        arena_threshold=0.6,
        eval_every=25000,        # GPU eval vs Random every 25K steps
    )

    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")
