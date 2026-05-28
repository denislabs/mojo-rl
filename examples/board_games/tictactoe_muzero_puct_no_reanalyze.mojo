"""MuZero TTT — PUCT, NO reanalyze (Experiment A, 2026-05-25).

Differential ablation against ``tictactoe_muzero_puct.mojo``. The
only difference is ``use_reanalyze=False`` (vs ``True`` in the
original). Arena is also disabled here to match the original
no-arena 60-iter PUCT run that showed:
  * ``|W|`` pred runaway: 19.7 → 679 in 60 iters
  * Bistable Minimax eval (D 64 ↔ L 64)

Hypothesis: reanalyze is the GPU-only code path causing the
runaway. It refreshes value targets each train step by running the
prediction head on stored observations against a Polyak target
network — if it writes corrupted value targets (wrong scale, wrong
sign, wrong layout), the value head receives an unbounded CE
gradient → pred weights grow without bound. Symptoms match.

What to look for in the run:
  1. **``|W|`` pred at iter ~30**: if it's ≤ ~50 (CPU-like), the
     hypothesis is confirmed — reanalyze is the bug. If it's still
     climbing to ~260 like the original run, the bug is elsewhere
     (next candidates: augmentation, ``search_gpu_selfplay``).
  2. **vs Minimax**: if it consistently reaches D 64 without
     oscillating into L 64, the bistable behavior is downstream of
     reanalyze too.
  3. **vs Random**: should improve over iters as a sanity check
     that training is actually happening.

If this experiment confirms reanalyze, the next step is to bisect
inside the reanalyze pipeline: target-net Polyak rate, value-target
computation kernel, value-target write into the replay buffer.

Usage:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_muzero_puct_no_reanalyze.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxTicTacToe,
    RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== MuZero on TicTacToe (PUCT, NO reanalyze — Experiment A) ===")
    print()

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="MuZero TicTacToe (PUCT, no reanalyze)",
        buffer_size=13,
        api_key=api_key,
    )

    comptime Config = MuZeroTicTacToeConfig[]

    logger.set_config("agent", "MuZero-PUCT-NoReanalyze")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("unroll_steps", String(Config.unroll_steps))
    logger.set_config("td_steps", String(Config.td_steps))
    logger.set_config("reanalyze", "False")

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericMuZeroAgent[Config, 64, RemoteLogger]()

    # Same training hyperparameters as ``tictactoe_muzero_puct.mojo``
    # EXCEPT use_reanalyze=False. Arena also disabled to match the
    # baseline no-arena run that originally surfaced the runaway —
    # so the ONLY difference between this run and the broken baseline
    # is reanalyze.
    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe, 5
    ](
        ctx,
        num_iters=100,
        steps_per_iter=1000,
        train_epochs=2,
        warmup_iters=1,
        arena_threshold=0.5,
        do_eval=True,
        do_eval2=True,
        do_arena=False,
        checkpoint_every=10,
        checkpoint_path="tictactoe_muzero_puct_no_reanalyze.ckpt",
        use_reanalyze=False,
        logger=UnsafePointer(to=logger),
        diag_every=500,
        use_one_cycle=True,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")
