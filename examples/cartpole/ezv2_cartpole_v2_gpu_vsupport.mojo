"""EfficientZeroV2 discrete CartPole — value-support ABLATION (v2, GPU Gumbel).

Isolates the single high-confidence fix from `ezv2_cartpole_v2_gpu`: this file is
the **original** EZv2 config (value_coef 0.25, no reanalyze, no temperature
schedule, no gradient clipping) with the **only** change being the categorical
value/reward support widened from ±10 to **±20** (h-space).

The original ±10 support hard-clamps any return whose transformed value exceeds
10 — i.e. raw value above ~117 (`h(x)=sign(x)(√(|x|+1)−1)+0.001x`), which a
CartPole policy surviving past ~150 steps already exceeds. The value head then
saturates and MCTS can no longer rank long-horizon states, capping the policy.
±20 covers raw ~420, well past CartPole's max discounted return (~259 → h≈15.4).

Run this against the original-plateau baseline to attribute the EZv2 gap to the
value support alone; run `ezv2_cartpole_v2_gpu` (the full bundle: +value_coef 0.5,
+clip 5, +reanalyze, +temperature) to see the combined effect.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/ezv2_cartpole_v2_gpu_vsupport.mojo
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

    print("EZv2 CartPole value-support ablation (v2, GPU Gumbel — v±20 only)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K_gumbel", MAX_K, "K", K, "N", N, "B", B,
          "v±20 vcoef 0.25 (no clip/reanalyze/temp)")

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
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
