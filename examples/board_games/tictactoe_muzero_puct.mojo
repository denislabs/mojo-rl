"""MuZero training on TicTacToe — fully GPU self-play with PUCT policy.

Companion to ``tictactoe_muzero.mojo`` (Gumbel-MuZero). Same config,
same training hyperparameters, same self-play harness — only the MCTS
policy mode differs:

  * Gumbel-MuZero (``tictactoe_muzero.mojo``): Gumbel-Top-k root
    sampling + Sequential Halving + deterministic σ(Q)-N/(1+ΣN)
    interior + improved-policy target.
  * PUCT (this file): vanilla MuZero PUCT selection + Dirichlet root
    noise + visit-count training target (the production / mctx
    ``muzero_policy`` equivalent).

Purpose: diagnostic A/B for the 2026-05-23 Gumbel-MuZero TTT run
showing prediction-net weight runaway (|W| pred: 19.7 → 347 in 38
iters while rep/dyn stayed flat) and bistable eval vs Minimax
oscillating between D 64 / L 0 and L 64 / D 0.

Update 2026-05-24: A longer 60-iter PUCT run reproduces both
issues — pred-weight runaway is shared across policies (root cause
is in training infrastructure, not MCTS) and Minimax-eval is also
bistable in PUCT once iter > ~13. Arena gating is now enabled
(``do_arena=True``) as a probe: if it catches the catastrophic
iters and stabilizes the eval curve, that confirms the bistable
behavior is downstream of network checkpoint quality. The
underlying pred-weight runaway is a separate problem and arena
won't fix it.

Note: the 2026-05-23 root-hidden-scatter fix was also applied to
``mcts_gpu_orchestrator.mojo::search_gpu_selfplay`` in the same
session so PUCT self-play at N_ENVS > 1 is no longer poisoning
envs 1..N-1 with zero hidden state.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/tictactoe_muzero_puct.mojo
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_muzero_puct.mojo
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
    print("=== MuZero on TicTacToe (PUCT) ===")
    print()

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="MuZero TicTacToe (PUCT)",
        buffer_size=13,
        api_key=api_key,
    )

    # MuZeroTicTacToeConfig defaults to MuZeroPUCTPolicy, so no POLICY
    # override is needed. PUCT consumes Dirichlet root noise (configured
    # via ``Noise = DirichletNoise[0.25, 0.25]`` in the config) and
    # trains the policy toward the MCTS visit-count distribution.
    comptime Config = MuZeroTicTacToeConfig[]

    logger.set_config("agent", "MuZero-PUCT")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", Config.NAME)
    logger.set_config("sims", String(Config.num_simulations))
    logger.set_config("batch_size", String(Config.batch_size))
    logger.set_config("unroll_steps", String(Config.unroll_steps))
    logger.set_config("td_steps", String(Config.td_steps))

    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericMuZeroAgent[Config, 64, RemoteLogger]()

    # All training hyperparameters mirror ``tictactoe_muzero.mojo``
    # (Gumbel) exactly so the only difference between the two runs is
    # the MCTS policy mode.
    _ = agent.train_selfplay_gpu[
        TTT, RandomOpponent, GPUMinimaxTicTacToe, 5
    ](
        ctx,
        num_iters=100,
        steps_per_iter=1000,
        train_epochs=2,
        warmup_iters=1,
        # Arena gating ON (2026-05-24): candidate vs previous champion
        # each iter, accept only if win-rate ≥ threshold. Probes whether
        # gating out catastrophic iters prevents the bistable Minimax-eval
        # pattern observed in the no-arena run. TTT optimal play = always
        # draw, so 0.5 (draws count 0.5) is the right threshold — a
        # candidate that matches the champion's strength sits exactly on
        # the gate; only candidates that genuinely play better get
        # accepted. Higher thresholds (e.g. 0.52 used for Connect Four)
        # are too strict for TTT since optimal play can never *win*
        # against an equally optimal champion.
        arena_threshold=0.5,
        do_eval=True,
        do_eval2=True,
        do_arena=True,
        checkpoint_every=10,
        checkpoint_path="tictactoe_muzero_puct.ckpt",
        use_reanalyze=True,
        logger=UnsafePointer(to=logger),
        diag_every=500,
        use_one_cycle=True,
    )

    logger.close()
    print()
    print("Train steps:", agent.train_step_count)
    print("=== Done ===")
