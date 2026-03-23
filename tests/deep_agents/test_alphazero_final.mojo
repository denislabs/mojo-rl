"""AlphaZero batch-then-train with sliding window + arena + draws=0.5."""

from std.memory import alloc
from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearReLU, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.alphazero import GenericAlphaZeroAgent
from mojo_rl.deep_agents.alphazero.configs import AlphaZeroConfig
from mojo_rl.deep_agents.muzero.strategies import (
    DirichletNoise, AlphaGoPUCT, SelfPlay,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent, MinimaxTicTacToe
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


struct TTTConfig(AlphaZeroConfig):
    comptime NAME: String = "AlphaZero-TTT-Final"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9
    comptime PredModel = Sequential[
        LinearReLU[27, 128],
        LinearReLU[128, 128],
        Parallel[Linear[128, 9], Linear[128, 1]],
    ]
    comptime OptType = Adam[LR=0.001]
    comptime batch_size: Int = 64
    comptime buffer_capacity: Int = 50000
    comptime history_window: Int = 3     # Keep last 3 iterations
    comptime num_simulations: Int = 25
    comptime max_nodes: Int = 64
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[1.0]
    comptime Players = SelfPlay


fn main() raises:
    print("=== AlphaZero: Batch-Then-Train + Arena ===")
    print("SIMS=25, cpuct=1.0, LR=0.001/Adam")
    print("Arena: 40 games, 55% threshold (draws=0.5)")
    print("Sliding window: last 3 iterations | 8x symmetries")
    print()

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    var agent = GenericAlphaZeroAgent[TTTConfig, 64]()
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
    comptime PS = TTTConfig.PredModel.PARAM_SIZE
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
        var num_train_steps = 10 * agent.state.buf_size // 64
        if num_train_steps < 10:
            num_train_steps = 10
        if num_train_steps > 3000:
            num_train_steps = 3000

        var gpu = agent.GPUStateType(ctx)
        gpu.upload_from(agent.state, ctx)
        for _ in range(num_train_steps):
            agent.train_step_gpu(ctx, gpu)
        gpu.download_to(agent.state, ctx)

        # 4. Arena comparison (skip first 5 iters to let model learn first)
        if iter >= 5:
            var accepted = agent.arena_compare[TTTCPU](
                arena_env, best_params,
                num_games=40, threshold=0.55,
            )
            if accepted:
                for i in range(PS):
                    best_params[i] = agent.state.prediction.params[i]
                arena_accepts += 1
            else:
                arena_rejects += 1
        else:
            # Always accept during warmup — save current as best
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
