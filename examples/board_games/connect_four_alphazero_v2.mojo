"""AlphaZero on Connect Four (deep_agents2 / nn2) — full GPU with remote logging.

Second-generation port of `connect_four_alphazero.mojo`. Uses the config-free
nn2 net torsos (`AZConnectFourResNet` — conv stem → 5 identity-skip ResBlocks →
FC policy/value heads, 128 filters, the closest match to the original AlphaZero
backbone) + the `AlphaZeroAgent` facade, and exercises the production telemetry:
two pluggable `GPUEvaluator` opponents (5-ply minimax + random), a per-report
progress print, and a `RemoteLogger` metrics sink. The periodic eval plays the
agent at full **MCTS** strength (temp=0), so the numbers reflect the deployed
agent, not the bare policy head.

Connect Four is heavier than TicTacToe (126D obs = 3 planes × 6×7, 7 actions,
games up to 42 plies, a 5-block ResNet) — this needs an NVIDIA GPU to train at a
useful pace. The ResNet torso carries BatchNorm, which the self-play / eval
harness toggles (`set_attr["training"]`) automatically.

Note `iterations` / `report_every` are in self-play *moves* (one loop pass
advances all N_ENVS games by one move), not legacy-style collect+train rounds.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/connect_four_alphazero_v2.mojo

With no `RL_MONITOR_URL` in the environment the RemoteLogger is a silent no-op;
the per-report lines still print to stdout.
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents2.alphazero.nets import AZConnectFourResNet
from mojo_rl.deep_agents2.alphazero.agent import AlphaZeroAgent
from mojo_rl.deep_agents2.zero.symmetries import HFlipColumnAugmenter
from mojo_rl.deep_agents2.zero.evaluators import (
    RandomOpponent, GPUMinimaxConnectFour,
)
from mojo_rl.envs.board_games.connect_four.connect_four import ConnectFourEnv


def main() raises:
    print("=== AlphaZero on Connect Four (deep_agents2 / nn2) ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero Connect Four (nn2)",
        buffer_size=22,
        api_key=api_key,
    )
    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "ConnectFour")
    logger.set_config("network", "AZConnectFourResNet[F=128,NB=5,FC=128]")
    logger.set_config("framework", "deep_agents2/nn2")

    comptime OBS = 126
    comptime ACT = 7
    # ResNet torso: conv stem → 5 ResBlocks → FC heads (128 filters), the
    # closest match to the legacy `AlphaZeroConnectFourFusedResNetConfig`.
    comptime Net = AZConnectFourResNet[F=128, NB=5, FC=128]
    comptime Env = ConnectFourEnv[DType.float64]
    # Connect Four's only board symmetry is the left↔right column flip; the
    # board is not square, so the D4 group does NOT apply (no rotations).
    comptime Aug = HFlipColumnAugmenter[ROWS=6, COLS=7, PLANES=3]

    var ctx = DeviceContext()
    var agent = AlphaZeroAgent[
        "gpu", Env, Net, N_ENVS=64, NUM_SIMS=100, MAX_NODES=256,
        BATCH=128, CAP=1_000_000, MAX_TRAJ=42,
    ](ctx, lr=0.002)

    # Full AlphaZero: best/learner Arena gating + horizontal-flip augmentation,
    # evaluated periodically vs 5-ply minimax (primary) and random (secondary).
    # Metrics flush to the logger; progress prints to stdout. `RESULT_IDX=43` is
    # the `S_GAME_RESULT` slot in the Connect Four state; `MAX_PLIES=42` is a
    # full board.
    var res = agent.train_arena[
        AUG=Aug,
        OPP1=GPUMinimaxConnectFour[5],
        OPP2=RandomOpponent,
        L=RemoteLogger,
        ARENA_GAMES=128,
        RESULT_IDX=43,
        MAX_PLIES=42,
        EVAL_GAMES=64,
    ](
        iterations=40_000,
        learning_starts=200,
        train_per_iter=4,
        seed=42,
        arena_every=2_000,
        arena_open_plies=4,
        promote_threshold=0.55,
        report_every=1_000,
        do_eval=True,
        do_eval2=True,
        verbose=True,
        logger=UnsafePointer(to=logger),
    )

    logger.close()
    agent.save("connect_four_alphazero_v2.ckpt")

    print()
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("saved → connect_four_alphazero_v2.ckpt")
    print("=== Done ===")
