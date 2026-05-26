"""MuZero TTT — GPU self-play, aligned to the working CPU config.

Direct GPU counterpart to ``tictactoe_muzero_cpu.mojo``. Uses the
SAME network (``MuZeroTicTacToeCNNConfig`` — CNN representation +
MLP dynamics/prediction) and the same training hyperparameters as
the CPU run that's confirmed stable (Parameter Norm 83 → 95 over
53k steps, loss 2.1 → 1.2, vs Minimax reaches 0/20/0 by iter 10).

Goal: pin down whether the pred-weight runaway observed on GPU is:
  (a) MLP-specific (CNN+BN would normalize feature scales) — if this
      run trains stably, the bug is in how the MLP backbone interacts
      with the GPU training pipeline.
  (b) GPU-specific (independent of network architecture) — if this
      run ALSO explodes, the bug is in shared GPU self-play code
      (``search_gpu_selfplay`` sign convention, episode flush,
      TwoHot value encoding, etc.) and not in MLP-vs-CNN.

Config differences vs ``tictactoe_muzero_puct.mojo`` (the broken
MLP-on-GPU baseline):
  * Network: ``MuZeroTicTacToeCNNConfig`` (CNN rep, MLP dyn/pred)
    instead of all-MLP.
  * K=3, N=5, SIMS=50 (was K=5, N=10, SIMS=100).
  * steps_per_iter=500 (was 1000).
  * train_epochs=2, num_iters=40 (was 100).
  * arena_threshold=0.55 (was 0.5).
  * do_arena=True with the higher threshold.
  * temp_threshold=5 (same as the other GPU TTT example).

API gap from CPU: ``train_selfplay_gpu`` doesn't expose
``arena_games``, ``eval_games``, ``reanalyze_warmup``, or
``reanalyze_interval`` — GPU uses ``reanalyze_per_iter`` as a
single knob. Left at default (64).

Usage:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_muzero_gpu_aligned.mojo
    pixi run -e nvidia mojo run -I . examples/board_games/tictactoe_muzero_gpu_aligned.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroTicTacToeCNNConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxTicTacToe,
    RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== MuZero on TicTacToe (GPU, aligned to CPU CNN config) ===")
    print()

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="MuZero TicTacToe CNN GPU (aligned)",
        buffer_size=13,
        api_key=api_key,
    )

    # Same parameters as ``tictactoe_muzero_cpu.mojo``.
    comptime Config = MuZeroTicTacToeCNNConfig[
        FILTERS=64,
        LATENT=128,
        HIDDEN=128,
        BINS=51,
        LR=1e-3,
        BS=64,
        K=3,
        N=5,
        SIMS=50,
        NODES=128,
        C_PUCT=1.0,
    ]

    logger.set_config("agent", "MuZero")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("unroll_steps", String(Config.unroll_steps))
    logger.set_config("td_steps", String(Config.td_steps))
    logger.set_config("device", "gpu")

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericMuZeroAgent[Config, 64, RemoteLogger]()

    var t0 = perf_counter_ns()

    # Match the CPU example's training schedule wherever the GPU API
    # exposes the same knob.
    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe, 5
    ](
        ctx,
        num_iters=40,
        steps_per_iter=500,
        train_epochs=2,
        warmup_iters=1,
        arena_threshold=0.55,
        do_eval=True,
        do_eval2=True,
        do_arena=True,
        checkpoint_every=10,
        checkpoint_path="tictactoe_muzero_gpu_aligned.ckpt",
        use_reanalyze=True,
        logger=UnsafePointer(to=logger),
        diag_every=500,
        use_one_cycle=True,
    )

    var dt_s = Float64(perf_counter_ns() - t0) / 1e9

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("Elapsed:    ", dt_s, "s")
    print("=== Done ===")
