"""AlphaZero matching alpha-zero-general:
- Batch-then-train (collect → train epochs → arena → repeat)
- Temperature annealing (temp=1 first 15 moves, then temp=0)
- Arena: draws excluded from denominator, threshold=0.6
- Draw value = 1e-4 (tiny positive)
- Sliding window: last 20 iterations
- 8x symmetry augmentation
- CNN variant available (AlphaZeroTicTacToeCNNConfig)
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
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
from mojo_rl.deep_agents.alphazero.configs import AlphaZeroConfig
from mojo_rl.deep_agents.muzero.strategies import (
    DirichletNoise, AlphaGoPUCT, SelfPlay,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent, MinimaxTicTacToe
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero (alpha-zero-general style) ===")
    print("SIMS=25, cpuct=1.0, LR=0.001/Adam, temp_threshold=15")
    print("Arena: 40 games, 60% threshold (draws excluded)")
    print("Sliding window: last 20 iterations | 8x symmetries")
    print("Draw value: 1e-4 | Batch-then-train")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    # Use default config (MLP variant, matches alpha-zero-general hyperparams)
    comptime Config = AlphaZeroTicTacToeConfig[]

    var agent = GenericAlphaZeroAgent[Config, 64]()
    var random_eval = RandomOpponent()
    var minimax_eval = MinimaxTicTacToe()
    var eval_env = TTTCPU()
    var arena_env = TTTCPU()

    print("Before:")
    var r0 = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
    print("vs Random: W", r0[0], "D", r0[1], "L", r0[2])
    var m0 = agent.evaluate_against[TTTCPU](eval_env, minimax_eval, 100)
    print("vs Minimax: W", m0[0], "D", m0[1], "L", m0[2])

    # Save best params for arena comparison
    comptime PS = Config.PredModel.PARAM_SIZE
    var best_params = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        best_params[i] = agent.state.prediction.params[i]
    var arena_accepts = 0
    var arena_rejects = 0

    for iter in range(25):
        # 1. New iteration — evict old data
        agent.start_new_iteration()

        # 2. Collect self-play games (frozen network, no training)
        _ = agent.train_selfplay_gpu[TTT, TTTCPU](
            ctx,
            arena_env,
            num_steps=1000,
            warmup_steps=0,
            gradient_steps=0,       # No training during collection
            print_every=100000,
        )

        # 3. Train for multiple epochs on collected data
        # 10 epochs like alpha-zero-general
        var num_train_steps = 10 * agent.state.buf_size // Config.batch_size
        if num_train_steps < 10:
            num_train_steps = 10
        if num_train_steps > 5000:
            num_train_steps = 5000

        var gpu = agent.GPUStateType(ctx)
        gpu.upload_from(agent.state, ctx)
        for _ in range(num_train_steps):
            agent.train_step_gpu(ctx, gpu)
        gpu.download_to(agent.state, ctx)

        # 4. Arena comparison (skip first 5 iters to let model learn first)
        # Like alpha-zero-general: threshold=0.6, draws excluded
        if iter >= 5:
            var accepted = agent.arena_compare[TTTCPU](
                arena_env, best_params,
                num_games=40, threshold=0.6,
            )
            if accepted:
                for i in range(PS):
                    best_params[i] = agent.state.prediction.params[i]
                arena_accepts += 1
            else:
                arena_rejects += 1
        else:
            # Always accept during warmup
            for i in range(PS):
                best_params[i] = agent.state.prediction.params[i]

        # 5. Evaluate
        print("Iter", iter + 1,
              "(replay:", agent.state.buf_size,
              "train:", num_train_steps,
              "arena:", arena_accepts, "/", arena_rejects, ")")
        var r = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
        print("  vs Random: W", r[0], "D", r[1], "L", r[2])
        var m = agent.evaluate_against[TTTCPU](eval_env, minimax_eval, 100)
        print("  vs Minimax: W", m[0], "D", m[1], "L", m[2])

    best_params.free()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")
