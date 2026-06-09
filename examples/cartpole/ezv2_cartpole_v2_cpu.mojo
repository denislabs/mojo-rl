"""EfficientZeroV2 discrete CartPole convergence run (v2, CPU) — the lighthouse.

The EZv2 discrete twin of `muzero_cartpole_v2_cpu`: identical learned-model
hyperparameters and self-play loop, but the update is MuZero BPTT **plus** the
SimSiam temporal-consistency objective (the genuinely-new EZv2 piece). Watch the
greedy ``[eval]`` return — random CartPole ~22, "solving" ~195+. The training
``avg_return(10)`` understates the policy (∝-visit sampling + root Dirichlet
noise), so judge by the greedy eval, exactly as for MuZero.

Run (no GPU):
    pixi run mojo run -I . examples/cartpole/ezv2_cartpole_v2_cpu.mojo
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents2.efficient_zero_v2.selfplay_cpu import (
    run_ezv2_selfplay_cpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 64
    comptime BINS = 51
    comptime H = 128
    comptime PROJ = 128
    comptime PROJ_HID = 128
    comptime BOTTLENECK = 64
    comptime NUM_SIMS = 25
    comptime MAX_NODES = 128
    comptime CAP = 50000
    comptime B = 64
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["cpu", INIT=Kaiming]()
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    var pred = Pred.make["cpu", INIT=Kaiming]()
    var proj = Proj.make["cpu", INIT=Kaiming]()
    var predh = Predh.make["cpu", INIT=Kaiming]()
    var orep = Adam.make["cpu", M=Rep](rep)
    var odyn = Adam.make["cpu", M=Dyn](dyn)
    var opred = Adam.make["cpu", M=Pred](pred)
    var oproj = Adam.make["cpu", M=Proj](proj)
    var opredh = Adam.make["cpu", M=Predh](predh)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)
    oproj.lr = Scalar[DT](3e-4)
    opredh.lr = Scalar[DT](3e-4)

    print("EZv2 CartPole convergence (v2, CPU — MuZero BPTT + SimSiam)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K", K, "N", N, "B", B, "lr 3e-4 cons 2.0")

    var loss = run_ezv2_selfplay_cpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
    ](
        env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=30000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-10.0),
        v_max=Scalar[DT](10.0),
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
