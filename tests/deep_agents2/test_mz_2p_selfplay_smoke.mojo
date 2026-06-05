"""MuZero two-player self-play smoke — TicTacToe pipeline connects (CPU, no GPU).

Runs `run_muzero_selfplay_2p_cpu` on TicTacToe for a short horizon and asserts the
zero-sum pipeline connects: alternating-`to_play` episodes get stored, the SelfPlay
MCTS + learned-model BPTT train fires after warmup, the loss stays finite, and a
greedy eval vs a random opponent runs (reports W/D/L). This is the Phase-B #29
integration check (env → SelfPlay MCTS over h/g/f → replay w/ sign-flipped n-step
targets → BPTT). Strength convergence is a separate, longer run.

Run (no GPU):
    pixi run mojo run -I . tests/deep_agents2/test_mz_2p_selfplay_smoke.mojo
"""

from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents2.muzero.selfplay_2p_cpu import run_muzero_selfplay_2p_cpu
from mojo_rl.deep_agents2.zero.evaluators import RandomOpponent
from mojo_rl.envs.board_games.tic_tac_toe.tic_tac_toe import TicTacToeEnv


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime LATENT = 32
    comptime BINS = 51
    comptime H = 48
    comptime NUM_SIMS = 16
    comptime MAX_NODES = 64
    comptime CAP = 20000
    comptime B = 16
    comptime K = 3
    comptime N = 9

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var env = TicTacToeEnv[DType.float64]()
    var rep = Rep.make["cpu", INIT=Kaiming]()
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    var pred = Pred.make["cpu", INIT=Kaiming]()
    var orep = Adam.make["cpu", M=Rep](rep)
    var odyn = Adam.make["cpu", M=Dyn](dyn)
    var opred = Adam.make["cpu", M=Pred](pred)
    orep.lr = Scalar[DT](0.002)
    odyn.lr = Scalar[DT](0.002)
    opred.lr = Scalar[DT](0.002)

    var loss = run_muzero_selfplay_2p_cpu[
        TicTacToeEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
        RandomOpponent,
    ](
        env, rep, dyn, pred, orep, odyn, opred,
        iterations=1200,
        learning_starts=300,
        gamma=Scalar[DT](1.0),
        v_min=Scalar[DT](-1.0),
        v_max=Scalar[DT](1.0),
        seed=7,
        eval_every=600,
        eval_games=20,
        verbose=True,
    )

    print("final loss:", loss)
    assert_true(loss == loss and loss < 1e30 and loss > 0.0,
        "2p self-play training loss not finite/positive")
    print("MuZero 2-player self-play smoke: OK")
