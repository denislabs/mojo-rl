"""Diagnostic: Can MCTS beat Random on ConnectFour with a fresh network?

If this fails, the issue is in env/MCTS, not training.
If this passes, training is corrupting the network.
"""

from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourCNNConfig,
    AlphaZeroConnectFourConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def main() raises:
    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    # Test 1: MLP with fresh (untrained) network
    print("=" * 60)
    print("TEST 1: Fresh MLP network + MCTS(25) vs Random")
    print("=" * 60)
    comptime MLPConfig = AlphaZeroConnectFourConfig[]
    var mlp_agent = GenericAlphaZeroAgent[MLPConfig, 64]()
    var mlp_gpu = mlp_agent.GPUStateType(ctx)
    mlp_gpu.upload_from(mlp_agent.state, ctx)
    var r1 = mlp_agent.gpu_eval[C4, RandomOpponent](ctx, mlp_gpu, rng_offset=42)
    print("  W", r1[0], "D", r1[1], "L", r1[2])
    if r1[0] > r1[2]:
        print("  PASS: Fresh MLP+MCTS beats Random")
    elif r1[0] == r1[2]:
        print("  NEUTRAL: Fresh MLP+MCTS ties Random")
    else:
        print("  FAIL: Fresh MLP+MCTS LOSES to Random!")

    # Run again with different seed to check consistency
    var r1b = mlp_agent.gpu_eval[C4, RandomOpponent](
        ctx, mlp_gpu, rng_offset=9999
    )
    print("  (seed 2) W", r1b[0], "D", r1b[1], "L", r1b[2])

    print()

    # Test 2: CNN with fresh network
    print("=" * 60)
    print("TEST 2: Fresh CNN network + MCTS(25) vs Random")
    print("=" * 60)
    comptime CNNConfig = AlphaZeroConnectFourCNNConfig[]
    var cnn_agent = GenericAlphaZeroAgent[CNNConfig, 64]()
    var cnn_gpu = cnn_agent.GPUStateType(ctx)
    cnn_gpu.upload_from(cnn_agent.state, ctx)
    print(" Test beginning")
    var r2 = cnn_agent.gpu_eval[C4, RandomOpponent](ctx, cnn_gpu, rng_offset=42)
    print(" Test ending")
    print("  W", r2[0], "D", r2[1], "L", r2[2])
    if r2[0] > r2[2]:
        print("  PASS: Fresh CNN+MCTS beats Random")
    elif r2[0] == r2[2]:
        print("  NEUTRAL: Fresh CNN+MCTS ties Random")
    else:
        print("  FAIL: Fresh CNN+MCTS LOSES to Random!")

    var r2b = cnn_agent.gpu_eval[C4, RandomOpponent](
        ctx, cnn_gpu, rng_offset=9999
    )
    print("  (seed 2) W", r2b[0], "D", r2b[1], "L", r2b[2])

    print()
    print("If FRESH network + MCTS can't beat Random → env/MCTS bug")
    print("If FRESH network + MCTS beats Random → training is corrupting")
