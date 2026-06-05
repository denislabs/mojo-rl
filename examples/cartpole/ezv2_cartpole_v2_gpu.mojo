"""EfficientZeroV2 discrete CartPole convergence run (v2, GPU Gumbel) — lighthouse.

The GPU twin of `ezv2_cartpole_v2_cpu`: identical learned-model hyperparameters,
but **search runs on the `GumbelGPUMCTS` planner** (Gumbel-Top-k + sequential
halving over the on-device h/g/f nets) and the update is `ezv2_unroll_train_step_gpu`
(MuZero BPTT + SimSiam consistency) on the resident GPU nets. Watch the greedy
``[eval]`` return — random CartPole ~22, "solving" ~195+. The training
``avg_return(10)`` understates the policy (∝-improved-policy sampling); judge by
the greedy eval.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/ezv2_cartpole_v2_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents2.efficient_zero_v2.selfplay_gpu import (
    run_ezv2_gumbel_selfplay_gpu,
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
    comptime MAX_K = 2
    comptime CAP = 50000
    comptime B = 64
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var ctx = DeviceContext()
    var env = CartPoleEnv[DType.float32]()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx)
    var proj = Proj.make["gpu", INIT=Kaiming](ctx)
    var predh = Predh.make["gpu", INIT=Kaiming](ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    var oproj = Adam.make["gpu", M=Proj](proj, ctx)
    var opredh = Adam.make["gpu", M=Predh](predh, ctx)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)
    oproj.lr = Scalar[DT](3e-4)
    opredh.lr = Scalar[DT](3e-4)

    print("EZv2 CartPole convergence (v2, GPU Gumbel — MuZero BPTT + SimSiam)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K_gumbel", MAX_K, "K", K, "N", N, "B", B)

    var loss = run_ezv2_gumbel_selfplay_gpu[
        CartPoleEnv[DType.float32], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
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
