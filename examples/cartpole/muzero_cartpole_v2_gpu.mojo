"""MuZero CartPole convergence run (v2, GPU) — the single-player hybrid lighthouse.

The GPU twin of `muzero_cartpole_v2_cpu`: identical hyperparameters, but driven
through `run_muzero_selfplay_gpu` — the **CPU-search / GPU-train hybrid**. The
MCTS search plans on a CPU mirror of the three nets; the K-step BPTT unroll runs
on the device (`mz_unroll_train_step_gpu`); the mirror is re-synced from the
device after each train step. Same convergence target as the CPU run (random
CartPole ~22, "solving" ~195+) — watch `avg_return(10)` / the greedy `[eval]`.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/muzero_cartpole_v2_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents2.muzero.selfplay_gpu import run_muzero_selfplay_gpu
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

    var ctx = DeviceContext()
    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)

    print("MuZero CartPole convergence (v2, GPU — CPU-search / GPU-train hybrid)")
    print("  LATENT", LATENT, "H", H, "BINS", BINS, "sims", NUM_SIMS,
          "K", K, "N", N, "B", B, "lr 3e-4")

    var loss = run_muzero_selfplay_gpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, orep, odyn, opred,
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
