"""MuZero TTT — PUCT, NO D4 augmentation (Experiment B, 2026-05-25).

Differential ablation against ``tictactoe_muzero_puct_no_reanalyze.mojo``.
Both reanalyze AND augmentation off — testing the augmentation
hypothesis after Experiment A falsified reanalyze.

Hypothesis: ``D4SquareAugmenter[3, 3]`` replicates each stored
episode 8× under rotations and reflections. For the replicated
boards to be correct training data, the policy-target action labels
must be permuted to match the rotated board. If that permutation
kernel is wrong on GPU, 7/8 of the stored replay is (board, wrong
policy) pairs — policy CE chases impossible targets → unbounded
logit growth → pred-weight runaway. The symptom (pred head grows
17–35× while rep/dyn lag, eval bistable vs Minimax) matches.

What to look for:
  1. **``|W|`` pred at iter 10**: baseline was 58.5. If this run is
     ≤ ~25, augmentation confirmed.
  2. **``|W|`` pred at iter 30**: baseline was ≈260. If < 100, almost
     certainly augmentation.
  3. **vs Minimax**: if it consistently reaches D 64 without bistable
     oscillation, augmentation confirmed.
  4. **vs Random**: trades off — without augmentation, sample
     efficiency drops 8×. Slower learning is expected, but it
     should still improve monotonically.

Implementation note: ``AUG`` template parameter was added to
``MuZeroTicTacToeConfig`` in this session so the augmenter can be
swapped from the example. Defaults to ``D4SquareAugmenter[3, 3]``
(unchanged production behavior); this example overrides to
``IdentityAugmenter``.

Usage:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_muzero_puct_no_aug.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.alphazero.strategies import IdentityAugmenter
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxTicTacToe,
    RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== MuZero on TicTacToe (PUCT, NO aug, NO reanalyze — Experiment B) ===")
    print()

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="MuZero TicTacToe (PUCT, no aug)",
        buffer_size=13,
        api_key=api_key,
    )

    # Same as Experiment A's config except AUG=IdentityAugmenter.
    comptime Config = MuZeroTicTacToeConfig[AUG=IdentityAugmenter]

    logger.set_config("agent", "MuZero-PUCT-NoAug")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("unroll_steps", String(Config.unroll_steps))
    logger.set_config("td_steps", String(Config.td_steps))
    logger.set_config("reanalyze", "False")
    logger.set_config("aug", "Identity")

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericMuZeroAgent[Config, 64, RemoteLogger]()

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
        checkpoint_path="tictactoe_muzero_puct_no_aug.ckpt",
        use_reanalyze=False,
        logger=UnsafePointer(to=logger),
        diag_every=500,
        use_one_cycle=True,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")
