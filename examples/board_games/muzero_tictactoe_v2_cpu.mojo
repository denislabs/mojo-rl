"""MuZero TicTacToe convergence run (v2, CPU) — the two-player lighthouse.

Drives `run_muzero_selfplay_2p_cpu` on TicTacToe with the SelfPlay backup + the
two-player sign-flipped n-step targets, and evaluates the agent at greedy MCTS
strength vs a **perfect minimax** opponent. TicTacToe is a forced draw under
optimal play, so the convergence target is **0 losses vs minimax** (an agent that
never blunders draws every game from either side). Watch the `L` column fall to 0.

Run (no GPU):
    pixi run mojo run -I . examples/board_games/muzero_tictactoe_v2_cpu.mojo
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents.muzero.selfplay_2p_cpu import run_muzero_selfplay_2p_cpu
from mojo_rl.deep_agents.zero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime LATENT = 64
    comptime BINS = 51
    comptime H = 128
    comptime NUM_SIMS = 25
    comptime MAX_NODES = 96
    comptime CAP = 4000     # small window → recent (non-stale) policy targets
    comptime B = 64
    comptime K = 5
    comptime N = 9      # full-game horizon → Monte-Carlo-ish value targets

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var env = TicTacToeEnv[DType.float64]()
    var rep = Rep.make["cpu", Kaiming]()
    var dyn = Dyn.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()
    var orep = Adam(lr=Scalar[DT](2e-3))
    var odyn = Adam(lr=Scalar[DT](2e-3))
    var opred = Adam(lr=Scalar[DT](2e-3))

    print("MuZero TicTacToe convergence (v2, CPU) — reanalyze on, eval vs random")
    print("  LATENT", LATENT, "H", H, "sims", NUM_SIMS, "K", K, "N", N, "B", B)

    var loss = run_muzero_selfplay_2p_cpu[
        TicTacToeEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
        RandomOpponent,
    ](
        env, rep, dyn, pred, orep, odyn, opred,
        iterations=20000,
        learning_starts=500,
        gamma=Scalar[DT](1.0),
        v_min=Scalar[DT](-1.0),
        v_max=Scalar[DT](1.0),
        value_coef=Scalar[DT](0.25),
        seed=42,
        eval_every=2000,
        eval_games=40,
        reanalyze_every=1,      # refresh stale targets every step
        reanalyze_batch=2,      # 2 old positions replanned per step (~3x MCTS)
        verbose=True,
    )

    print("final loss:", loss)
