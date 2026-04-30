"""MuZero training on Connect Four — fully GPU self-play.

MuZero learns a dynamics model g(s,a) and plans in latent space.
Uses ResNet-style networks for the larger board (6x7) and AdamW
with weight decay for regularization.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_muzero.mojo
    pixi run -e apple mojo run -I . examples/board_games/connect_four_muzero.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroConnectFourConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxConnectFour,
    RandomOpponent,
)
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print("=== MuZero on Connect Four ===")
    print()

    comptime Config = MuZeroConnectFourConfig[]

    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    var agent = GenericMuZeroAgent[Config, 64]()

    _ = agent.train_selfplay_gpu[
        C4, GPUMinimaxConnectFour[5], RandomOpponent, 20
    ](
        ctx,
        num_iters=50,
        steps_per_iter=5000,
        train_epochs=5,
        warmup_iters=1,
        arena_threshold=0.52,
        do_eval=True,
        do_arena=True,
        checkpoint_every=10,
        checkpoint_path="connect_four_muzero.ckpt",
    )

    print()
    print("=== Done ===")
