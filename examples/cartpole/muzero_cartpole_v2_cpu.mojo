"""MuZero CartPole convergence run (v2, CPU) — the single-player lighthouse.

Drives `run_muzero_selfplay_cpu` with hyperparameters mirroring the legacy
`MuZeroMLPConfig` CartPole setup (LATENT=64/HIDDEN=128, BINS=51, K=5, N=10,
25 sims, lr 3e-4, gamma 0.997, value support [-10,10] h-space, value_coef 0.25,
batched MCTS 8/virtual-loss 3 to counter the spiky Dirichlet root prior).

This is the convergence/tuning harness for Phase B #28 — NOT a smoke. Random
CartPole returns ~22; "solving" is ~195+. Watch `avg_return(10)` climb.

Run (no GPU):
    pixi run mojo run -I . examples/cartpole/muzero_cartpole_v2_cpu.mojo
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents2.muzero.selfplay_cpu import run_muzero_selfplay_cpu
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 64
    comptime BINS = 51
    comptime H = 128
    comptime NUM_SIMS = 25
    comptime MAX_NODES = 128
    comptime CAP = 50000
    comptime B = 64
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["cpu", INIT=Kaiming]()
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    var pred = Pred.make["cpu", INIT=Kaiming]()
    var orep = Adam.make["cpu", M=Rep](rep)
    var odyn = Adam.make["cpu", M=Dyn](dyn)
    var opred = Adam.make["cpu", M=Pred](pred)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)

    print("MuZero CartPole convergence (v2, CPU)")
    print("  LATENT", LATENT, "H", H, "BINS", BINS, "sims", NUM_SIMS,
          "K", K, "N", N, "B", B, "lr 3e-4")

    var loss = run_muzero_selfplay_cpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        env, rep, dyn, pred, orep, odyn, opred,
        iterations=30000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-10.0),
        v_max=Scalar[DT](10.0),
        value_coef=Scalar[DT](0.25),
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
