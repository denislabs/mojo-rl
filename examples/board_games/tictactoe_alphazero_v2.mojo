"""AlphaZero on TicTacToe (deep_agents / nn) — full GPU with remote logging.

Second-generation port of `tictactoe_alphazero.mojo`. Uses the config-free
nn net torsos + the `AlphaZeroAgent` facade, and exercises the production
telemetry: two pluggable `GPUEvaluator` opponents (minimax + random), a
per-report progress print, and a `RemoteLogger` metrics sink. The periodic eval
plays the agent at full **MCTS** strength (temp=0), so the numbers reflect the
deployed agent; over the run it learns strong play (drawing minimax, beating
random) — the textbook "optimal never loses" result.

Note `iterations` / `report_every` are in self-play *moves* (one loop pass
advances all N_ENVS games by one move), not legacy-style collect+train rounds.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/tictactoe_alphazero_v2.mojo
    pixi run -e apple  mojo run -I . examples/board_games/tictactoe_alphazero_v2.mojo

With no `RL_MONITOR_URL` in the environment the RemoteLogger is a silent no-op,
so this runs anywhere; the per-report lines still print to stdout.
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents.alphazero.agent import AlphaZeroAgent
from mojo_rl.deep_agents.zero.symmetries import D4SquareAugmenter
from mojo_rl.deep_agents.zero.evaluators import (
    RandomOpponent, GPUMinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero on TicTacToe (deep_agents / nn) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero TicTacToe (nn)",
        buffer_size=22,
        api_key=api_key,
    )
    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", "AZMLPNet[27,9,128]")
    logger.set_config("framework", "deep_agents/nn")

    comptime OBS = 27
    comptime ACT = 9
    comptime H = 128
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Aug = D4SquareAugmenter[3, 3]    # 8 D4 board symmetries

    var ctx = DeviceContext()
    var agent = AlphaZeroAgent[
        "gpu", Env, Net, N_ENVS=32, NUM_SIMS=50, MAX_NODES=128,
        BATCH=64, CAP=80000, MAX_TRAJ=16,
    ](ctx, lr=0.005)

    # Full AlphaZero: best/learner Arena gating + D4 augmentation, evaluated
    # every 200 self-play iterations vs minimax (primary) and random
    # (secondary). Metrics flush to the logger; progress prints to stdout.
    var res = agent.train_arena[
        AUG=Aug,
        OPP1=GPUMinimaxTicTacToe,
        OPP2=RandomOpponent,
        L=RemoteLogger,
        ARENA_GAMES=40,
        RESULT_IDX=10,
        MAX_PLIES=9,
        EVAL_GAMES=64,
    ](
        iterations=4000,
        learning_starts=20,
        train_per_iter=4,
        seed=42,
        arena_every=400,
        arena_open_plies=2,
        promote_threshold=0.55,
        report_every=200,
        diag_every=20,
        do_eval=True,
        do_eval2=True,
        verbose=True,
        logger=UnsafePointer(to=logger).as_unsafe_any_origin(),
    )

    logger.close()
    agent.save("tictactoe_alphazero_v2.ckpt")

    print()
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("saved → tictactoe_alphazero_v2.ckpt")
    print("=== Done ===")
