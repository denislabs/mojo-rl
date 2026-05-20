"""AlphaZero CPU training on TicTacToe (MLP config).

CPU-only — no DeviceContext required. Mirrors the GPU recipe but at a much
smaller scale: a few iterations of MCTS self-play interleaved with CPU SGD
on the (obs, MCTS-policy, game-outcome) replay buffer.

Usage:
    pixi run mojo run -I . examples/board_games/tictactoe_alphazero_cpu.mojo
"""

from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero CPU on TicTacToe (MLP) ===")
    print()

    # Small, CPU-friendly config:
    #   HIDDEN=64 (vs default 128) keeps the MLP tiny.
    #   SIMS=25  (vs default 100) caps per-move MCTS at ~25 sims.
    #   BS=32    matches our SGD step size on the CPU.
    comptime Config = AlphaZeroTicTacToeConfig[
        HIDDEN=64, LR=0.01, BS=32, CAP=20000, SIMS=25, NODES=64
    ]

    var env = TicTacToeEnv[DType.float32]()
    var agent = GenericAlphaZeroAgent[Config]()
    var opp = RandomOpponent()

    _ = agent.train_selfplay_cpu[
        TicTacToeEnv[DType.float32], RandomOpponent
    ](
        env,
        opp,
        num_iters=10,
        games_per_iter=20,
        train_epochs=5,
        warmup_iters=1,
        eval_games=20,
        verbose=True,
    )

    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")
