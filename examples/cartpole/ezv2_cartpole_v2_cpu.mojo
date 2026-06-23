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

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_cpu import (
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
    var rep = Rep.make["cpu", Kaiming]()
    var dyn = Dyn.make["cpu", Kaiming]()
    var pred = Pred.make["cpu", Kaiming]()
    var proj = Proj.make["cpu", Kaiming]()
    var predh = Predh.make["cpu", Kaiming]()
    var orep = Adam(lr=Scalar[DT](3e-4))
    var odyn = Adam(lr=Scalar[DT](3e-4))
    var opred = Adam(lr=Scalar[DT](3e-4))
    var oproj = Adam(lr=Scalar[DT](3e-4))
    var opredh = Adam(lr=Scalar[DT](3e-4))

    print("EZv2 CartPole convergence (v2, CPU — MuZero BPTT + SimSiam)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K", K, "N", N, "B", B,
          "lr 3e-4 cons 2.0 v±20 vcoef 0.5 clip 5 reanalyze temp")

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
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](0.5),
        consistency_coef=Scalar[DT](2.0),
        temperature_decay_steps=30000,
        reanalyze_every=1,
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
