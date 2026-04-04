"""AlphaZero Connect Four — Fused ResNet + CUDA Graph profiling script for nsys.

Same as connect_four_alphazero_fused_profile.mojo but with CUDA graph
capture enabled for both training and MCTS rounds.

Compare against the non-graph version to measure launch overhead savings:
    # Without CUDA graphs:
    nsys profile -o c4_fused_profile \
        pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero_fused_profile.mojo

    # With CUDA graphs:
    nsys profile -o c4_fused_cuda_graph_profile \
        pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero_fused_cuda_graph_profile.mojo

    nsys stats c4_fused_profile.nsys-rep
    nsys stats c4_fused_cuda_graph_profile.nsys-rep
"""

from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourFusedResNetConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    print(
        "=== AlphaZero Connect Four — Fused ResNet + CUDA Graph nsys"
        " profile ==="
    )
    print()

    comptime Config = AlphaZeroConnectFourFusedResNetConfig[]

    print("Network:", Config.NAME)
    print("Filters:", Config.FILTERS)
    print("Sims:", Config.num_simulations)
    print("Batch size:", Config.batch_size)
    print("Max nodes:", Config.max_nodes)
    print("CUDA Graph: enabled")
    print()

    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    var agent = GenericAlphaZeroAgent[Config, 512]()

    # 2 iters, 3000 steps each — enough to complete games and trigger training.
    _ = agent.train_selfplay_gpu[C4, RandomOpponent, USE_CUDA_GRAPH=True](
        ctx,
        num_iters=2,
        steps_per_iter=64_000,
        train_epochs=2,
        warmup_iters=0,
        arena_threshold=0.52,
        do_eval=True,
        do_arena=True,
        eval_games=64,
        arena_games=100,
    )

    print()
    print("Train steps:", agent.train_step_count)
    print("=== Fused + CUDA Graph profile run done ===")
