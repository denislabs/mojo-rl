"""AlphaZero Connect Four — ResNet profiling script for nsys.

Short run (1 iter, 500 steps) to capture GPU kernel patterns without
blowing up disk space. Extrapolate from the per-kernel timings.

Usage:
    nsys profile -o c4_resnet_profile \
        pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero_profile.mojo

    # Then view:
    nsys stats c4_resnet_profile.nsys-rep
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourResNetConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print("=== AlphaZero Connect Four — ResNet nsys profile ===")
    print()

    comptime Config = AlphaZeroConnectFourResNetConfig[]

    print("Network:", Config.NAME)
    print("Filters:", Config.FILTERS)
    print("Sims:", Config.num_simulations)
    print("Batch size:", Config.batch_size)
    print("Max nodes:", Config.max_nodes)
    print()

    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    # Default NoOpLogger — no remote logging overhead in profile.
    var agent = GenericAlphaZeroAgent[Config, 64, USE_STREAM=True]()

    # Short run: 1 iter, 500 steps — enough to capture all kernel patterns.
    # No eval/arena/checkpoint to keep it focused on training kernels.
    _ = agent.train_selfplay_gpu[C4, RandomOpponent](
        ctx,
        num_iters=1,
        steps_per_iter=500,
        train_epochs=1,
        warmup_iters=0,
        arena_threshold=0.52,
        do_eval=False,
        do_arena=False,
        checkpoint_every=0,
        checkpoint_path="",
        diag_every=100,
    )

    print()
    print("Train steps:", agent.train_step_count)
    print("=== Profile run done ===")
