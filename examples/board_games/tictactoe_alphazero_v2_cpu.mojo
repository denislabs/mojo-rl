"""AlphaZero on TicTacToe (deep_agents2 / nn2) — CPU path, no GPU.

CPU twin of `tictactoe_alphazero_v2.mojo`. Same `AlphaZeroAgent` facade and
full-AlphaZero recipe (best/learner Arena gating + D4 symmetry augmentation +
two pluggable eval opponents + telemetry), but `TARGET="cpu"`: the net is
host-resident, MCTS runs through `GenericCPUMCTS` + the true-rules adapters, and
the agent plays a single game at a time (`N_ENVS` is ignored on this path). No
`DeviceContext` is ever created, so this runs anywhere — useful for debugging
the algorithm without a GPU, at the cost of wall-clock (single-env self-play).

The same `GPUMinimaxTicTacToe` / `RandomOpponent` evaluators are used — they are
dual-conforming (`GPUEvaluator & CPUEvaluator`), so the CPU path drives them
through their CPU surface. The periodic eval plays the agent at full **MCTS**
strength (temp=0); a before/after greedy-policy eval vs random brackets the run
to show the net actually learning.

Note `iterations` / `report_every` are in self-play *moves* (one move per loop
pass), not legacy-style collect+train rounds.

Usage:
    pixi run mojo run -I . examples/board_games/tictactoe_alphazero_v2_cpu.mojo

With no `RL_MONITOR_URL` in the environment the RemoteLogger is a silent no-op,
so this runs anywhere; the per-report lines still print to stdout.
"""

from std.memory import UnsafePointer

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn2.core.checkpoint import save_state_v2
from mojo_rl.deep_agents2.alphazero.nets import AZMLPNet
from mojo_rl.deep_agents2.alphazero.agent import AlphaZeroAgent
from mojo_rl.deep_agents2.zero.symmetries import D4SquareAugmenter
from mojo_rl.deep_agents2.zero.evaluators import (
    RandomOpponent,
    GPUMinimaxTicTacToe,
)
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    print("=== AlphaZero on TicTacToe (deep_agents2 / nn2) — CPU ===")
    print()

    # ── Logger setup ────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero TicTacToe (nn2, CPU)",
        buffer_size=22,
        api_key=api_key,
    )
    logger.set_config("agent", "AlphaZero")
    logger.set_config("env", "TicTacToe")
    logger.set_config("network", "AZMLPNet[27,9,128]")
    logger.set_config("framework", "deep_agents2/nn2")
    logger.set_config("target", "cpu")

    comptime OBS = 27
    comptime ACT = 9
    comptime H = 128
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime Env = TicTacToeEnv[DType.float64]
    comptime Aug = D4SquareAugmenter[3, 3]  # 8 D4 board symmetries

    # CPU path: no DeviceContext (ctx=None), host-resident net. `N_ENVS` is a
    # required facade param but unused on CPU (single-env self-play), so it is
    # set to 1.
    var agent = AlphaZeroAgent[
        "cpu",
        Env,
        Net,
        N_ENVS=1,
        NUM_SIMS=50,
        MAX_NODES=128,
        BATCH=64,
        CAP=80000,
        MAX_TRAJ=16,
    ](None, lr=0.005)

    # Baseline: greedy policy head (search-free) vs random before training.
    # NB: this is the bare policy net; the *deployed* agent adds MCTS on top
    # (the "vs Random/Minimax" lines in the report below are full-strength).
    var before = agent.eval_vs_random_cpu[200, 9](agent_player=0, seed=12345)
    print(
        "BEFORE (greedy policy vs random)  win=",
        before.wins,
        " draw=",
        before.draws,
        " loss=",
        before.losses,
    )
    print()

    # Full AlphaZero: best/learner Arena gating + D4 augmentation, evaluated
    # periodically vs minimax (primary) and random (secondary). Metrics flush
    # to the logger; progress prints to stdout. Iteration count is trimmed vs
    # the GPU example since CPU self-play is single-env (slower wall-clock).
    var res = agent.train_arena[
        AUG=Aug,
        OPP1=GPUMinimaxTicTacToe,
        OPP2=RandomOpponent,
        L=RemoteLogger,
        ARENA_GAMES=20,
        RESULT_IDX=10,
        MAX_PLIES=9,
        EVAL_GAMES=32,
    ](
        iterations=10_000,
        learning_starts=20,
        train_per_iter=4,
        seed=42,
        # arena_every = moves the *best* (self-play generator) stays frozen
        # between promotion checks, so #generations ≈ iterations / arena_every.
        # AlphaZero bootstraps via search amplification across MANY generations,
        # so this must stay SMALL: 400 here gives only ~25 generations and the
        # agent barely learns; 100 gives ~100 generations and it learns well.
        arena_every=100,
        arena_open_plies=2,
        promote_threshold=0.55,
        report_every=200,
        diag_every=20,
        do_eval=True,
        do_eval2=True,
        verbose=True,
        logger=UnsafePointer(to=logger),
    )

    logger.close()

    # Endline: greedy net-policy vs random after training (should clearly beat
    # the baseline). The facade `agent.save` is GPU-only (needs a ctx), so the
    # host-resident net is checkpointed directly via the CPU `save_state_v2`.
    var after = agent.eval_vs_random_cpu[200, 9](agent_player=0, seed=12345)
    save_state_v2(agent.net, String("tictactoe_alphazero_v2_cpu.ckpt"))

    print()
    print(
        "AFTER  (greedy policy vs random)  win=",
        after.wins,
        " draw=",
        after.draws,
        " loss=",
        after.losses,
    )
    print("last_loss:", res.last_loss, "| promotions:", res.promotions)
    print("saved → tictactoe_alphazero_v2_cpu.ckpt")
    print("=== Done ===")
