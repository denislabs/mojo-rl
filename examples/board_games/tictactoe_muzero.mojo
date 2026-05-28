"""MuZero training on TicTacToe — fully GPU self-play with remote logging.

MuZero learns a dynamics model g(s,a) and plans in latent space,
unlike AlphaZero which uses true game rules. This tests whether
the learned model is accurate enough for a simple board game.

Uses Gumbel-MuZero (mctx-style) for action selection: Gumbel-Top-k
root sampling + Sequential Halving + deterministic σ(Q)-N/(1+ΣN)
interior selection + improved-policy training target. Replaces
the PUCT + Dirichlet-noise + visit-count target of vanilla MuZero.
See ``docs/mctx-main/`` for the reference.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/tictactoe_muzero.mojo
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_muzero.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
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
    print("=== MuZero on TicTacToe (Gumbel-MuZero) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="MuZero TicTacToe (Gumbel)",
        buffer_size=13,
        api_key=api_key,
    )

    # MAX_K=8 = largest power of 2 ≤ ACT (TTT has 9 actions). Sequential
    # Halving runs log2(8)=3 phases over the SIMS=100 budget — phase 1
    # tries 8 candidates × 4 sims each, phase 2 halves to 4 candidates ×
    # 8 sims, phase 3 picks between 2 × 16 sims. 4 leftover sims spent
    # on the survivor's slot 0.
    comptime Config = MuZeroTicTacToeConfig[
        POLICY=GumbelMuZeroPolicy[8],
    ]

    logger.set_config("agent", "MuZero-Gumbel")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("unroll_steps", String(Config.unroll_steps))
    logger.set_config("td_steps", String(Config.td_steps))
    logger.set_config("max_k", String(Config.PolicyMode.MAX_K))

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericMuZeroAgent[Config, 64, RemoteLogger]()

    # temp_threshold=5: temp=1 only for the first 5 moves; greedy from
    # move 6+ during self-play. AlphaZero TTT uses 4. Previously 15 (>9
    # max moves) which made every move exploratory and weakened the
    # endgame policy signal.
    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe, 5
    ](
        ctx,
        num_iters=100,
        steps_per_iter=1000,
        # train_epochs=2 (was 10): with 10 epochs, late iters did ~7800
        # grad steps over the same replay each iter. Network overfit to
        # current MCTS-target distribution which softens as Dirichlet
        # root noise propagates → policy collapses to uniform. AZ-general
        # uses 1-3 epochs for the same reason.
        train_epochs=2,
        warmup_iters=1,
        arena_threshold=0.5,
        do_eval=True,
        do_eval2=True,
        do_arena=False,
        checkpoint_every=10,
        checkpoint_path="tictactoe_muzero.ckpt",
        # Enable GPU reanalyze + Polyak target net (E2 / E4): bootstrap
        # values are refreshed each train step from a slowly-tracking
        # copy of the online networks, mirroring muzero-general's
        # use_last_model_value=True.
        use_reanalyze=True,
        logger=UnsafePointer(to=logger),
        # Log loss diagnostics every 500 grad steps. TTT does ~3K grad
        # steps/iter so this gives ~6 loss samples per iter — enough to
        # spot trends without drowning the logger.
        diag_every=500,
        # Per-iter 1cycle LR schedule (Smith): ramp up to base LR over
        # the first 30% of grad steps, cosine-anneal to 1% by end.
        # Mirrors AlphaZero. Helps prevent the late-iter weight-collapse
        # we saw at iter ~63 with constant LR=1e-3.
        use_one_cycle=True,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")
