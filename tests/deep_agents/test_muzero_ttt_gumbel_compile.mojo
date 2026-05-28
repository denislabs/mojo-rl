"""Smoke test: MuZero TicTacToe self-play with GumbelMuZeroPolicy.

Validates Phase 5's selfplay + eval Gumbel dispatch — flips
``Config.PolicyMode`` to ``GumbelMuZeroPolicy[4]`` on TTT and runs a
tiny self-play iteration so the compiler instantiates the Gumbel
branches in ``train_selfplay_gpu`` AND ``_gpu_eval_muzero``.

Usage:
    pixi run -e apple mojo run -I . tests/deep_agents/test_muzero_ttt_gumbel_compile.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.muzero import (
    GenericMuZeroAgent,
    MuZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.muzero.policy_mode import GumbelMuZeroPolicy
from mojo_rl.deep_agents.muzero.evaluators import (
    GPUMinimaxTicTacToe,
    RandomOpponent,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== MuZero TTT — GumbelMuZeroPolicy smoke ===")
    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    # MAX_K=4 fits TTT's 9 actions; SIMS=16 → 2 halving phases × 4 cands × 2 sims.
    # That's plenty of search to exercise both kernels and the eval path.
    comptime Config = MuZeroTicTacToeConfig[
        LATENT=64, HIDDEN=64, BINS=21, BS=16,
        SIMS=16,
        POLICY=GumbelMuZeroPolicy[4],
    ]
    print("Config.PolicyMode.IS_GUMBEL =", Config.PolicyMode.IS_GUMBEL)
    print("Config.PolicyMode.MAX_K     =", Config.PolicyMode.MAX_K)

    var agent = GenericMuZeroAgent[Config, 4]()
    print("Gumbel-mode TTT agent constructed; running train_selfplay_gpu...")
    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe, 5
    ](
        ctx,
        num_iters=1,
        steps_per_iter=64,
        train_epochs=1,
        warmup_iters=0,
        do_eval=True,
        do_eval2=False,
        do_arena=False,
        checkpoint_every=0,
        use_reanalyze=False,
    )
    print("Train steps:", agent.train_step_count)
    print("=== Gumbel selfplay smoke OK ===")
