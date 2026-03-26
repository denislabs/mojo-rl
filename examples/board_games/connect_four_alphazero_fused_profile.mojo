"""AlphaZero Connect Four — Fused ResNet profiling script for nsys.

Same as connect_four_alphazero_profile.mojo but uses
AlphaZeroConnectFourFusedResNetConfig with ResBlockConv2DBN
(fused BN2+skip+ReLU kernels).

Compare kernel launch counts against the non-fused version:
    nsys profile -o c4_fused_profile \
        pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero_fused_profile.mojo

    nsys stats c4_fused_profile.nsys-rep
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourFusedResNetConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print("=== AlphaZero Connect Four — Fused ResNet nsys profile ===")
    print()

    comptime Config = AlphaZeroConnectFourFusedResNetConfig[]

    print("Network:", Config.NAME)
    print("Filters:", Config.FILTERS)
    print("Sims:", Config.num_simulations)
    print("Batch size:", Config.batch_size)
    print("Max nodes:", Config.max_nodes)
    print()

    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    var agent = GenericAlphaZeroAgent[Config, 64]()

    # Short run: 1 iter, 500 steps — enough to capture all kernel patterns.
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
    print("=== Fused profile run done ===")
