"""Phase 3 agent rewiring smoke test: AlphaZero with USE_NEW_MCTS=True.

Sibling config of ``AlphaZeroTicTacToeConfig`` with the flag flipped, so
``train_selfplay_gpu``'s self-play MCTS routes through
``GenericGPUMCTS.search_gpu_alphazero`` + ``extract_actions_temp``
instead of the inline kernel block.

Asserts: compiles + runs one self-play iteration to completion.

Usage:
    pixi run -e apple mojo run -I . tests/deep_agents/test_alphazero_new_mcts.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.alphazero import GenericAlphaZeroAgent
from mojo_rl.deep_agents.alphazero.configs import (
    AlphaZeroConfig,
    DirichletNoise,
    AlphaGoPUCT,
    SelfPlay,
    MonteCarloReturn,
    D4SquareAugmenter,
)
from mojo_rl.nn.model import (
    Linear,
    LinearReLU,
    Sequential,
    Parallel,
)
from mojo_rl.nn.optimizer import Adam
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


# Clone of AlphaZeroTicTacToeConfig with USE_NEW_MCTS=True.
struct DevAZNewMCTSConfig[
    HIDDEN: Int = 128,
    LR: Float64 = 0.01,
    BS: Int = 16,
    CAP: Int = 120000,
    SIMS: Int = 100,
    NODES: Int = 128,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """Dev clone of AlphaZeroTicTacToeConfig with the planner rewiring on."""

    comptime NAME: String = "AlphaZero-TicTacToe-NewMCTS"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9

    comptime PredModel = Sequential[
        LinearReLU[27, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 9],
            Linear[Self.HIDDEN, 1],
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime history_window: Int = 20
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 4
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 9
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = D4SquareAugmenter[3, 3]

    comptime USE_NEW_MCTS: Bool = True   # ← the actual change


def main() raises:
    print("=== AlphaZero TicTacToe — USE_NEW_MCTS=True ===")
    var ctx = DeviceContext()

    comptime TTT = TicTacToeEnv[DType.float32]
    comptime Config = DevAZNewMCTSConfig[
        HIDDEN=64, LR=0.01, BS=16, SIMS=8, NODES=32
    ]

    # CUDA Graph disabled — the new path doesn't participate in graph
    # capture yet. Apple has no graph support so this is also implicit
    # on Apple.
    var agent = GenericAlphaZeroAgent[Config, 16]()
    print("Agent created:", Config.NAME, "USE_NEW_MCTS=", Config.USE_NEW_MCTS)

    print("Training one self-play iteration via GenericGPUMCTS...")
    var metrics = agent.train_selfplay_gpu[TTT, USE_CUDA_GRAPH=False](
        ctx,
        num_iters=1,
        steps_per_iter=200,
        train_epochs=1,
        warmup_iters=0,
        do_eval=False,
        do_arena=False,
        verbose=False,
    )

    print("\n=== Results ===")
    print("Train steps:", agent.train_step_count)

    if agent.train_step_count >= 0:
        print("PASS: AZ training completed via GenericGPUMCTS")
    _ = metrics
    print("=== Done ===")
