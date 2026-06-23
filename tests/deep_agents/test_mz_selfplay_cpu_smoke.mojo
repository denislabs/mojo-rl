"""MuZero CPU self-play driver smoke — full pipeline on CartPole, no GPU.

Runs `run_muzero_selfplay_cpu` for a short horizon and asserts the whole loop
connects: episodes get stored, training fires after warmup, and the loss stays
finite. This is the end-to-end Phase-B integration check (env → CPU MCTS →
replay → CPU BPTT unroll). Convergence tuning is a separate, longer run.

Run (no GPU):
    pixi run mojo run -I . tests/deep_agents/test_mz_selfplay_cpu_smoke.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents.muzero.selfplay_cpu import run_muzero_selfplay_cpu
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 16
    comptime BINS = 51
    comptime H = 32
    comptime NUM_SIMS = 12
    comptime MAX_NODES = 48
    comptime CAP = 20000
    comptime B = 8
    comptime K = 3
    comptime N = 3

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["cpu", Kaiming]()
    var dyn = Dyn.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()
    var orep = Adam(lr=Scalar[DT](0.01))
    var odyn = Adam(lr=Scalar[DT](0.01))
    var opred = Adam(lr=Scalar[DT](0.01))

    var loss = run_muzero_selfplay_cpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        env, rep, dyn, pred, orep, odyn, opred,
        iterations=900,
        learning_starts=200,
        train_per_iter=1,
        seed=7,
        verbose=True,
    )

    print("final loss:", loss)
    assert_true(loss == loss and loss < 1e30 and loss > 0.0,
        "self-play training loss not finite/positive")
    print("MuZero CPU self-play driver smoke: OK")
