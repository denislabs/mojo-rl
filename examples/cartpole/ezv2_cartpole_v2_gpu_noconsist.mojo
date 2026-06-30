"""EfficientZeroV2 discrete CartPole — consistency-OFF ablation (v2, GPU Gumbel).

Identical to `ezv2_cartpole_v2_gpu` (the full bundle: v±20, value_coef 0.5,
grad clip 5, reanalyze, temperature) in EVERY hyperparameter EXCEPT the SimSiam
temporal-consistency weight, which is set to **0**. With ``consistency_coef=0``
the projector/predictor receive no gradient and the update reduces to plain
**MuZero BPTT** over the shared h/g/f model — the same model and Gumbel planner
MuZero uses.

Purpose: isolate the SimSiam consistency loss as the cause of the residual
EZv2-vs-MuZero gap. The full bundle (consistency_coef=2.0) plateaus ~230 greedy
on this config while MuZero (consistency-free) solves to a stable 500.
  * If this run solves like MuZero → the consistency loss is the culprit.
  * If this run still plateaus → the gap is capacity/budget (see
    `ezv2_cartpole_v2_gpu_matched`, which bumps LATENT/B/iters to MuZero's).

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/ezv2_cartpole_v2_gpu_noconsist.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu import (
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
    var rep = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Pred.make["gpu", Kaiming](Optional(ctx))
    var proj = Proj.make["gpu", Kaiming](Optional(ctx))
    var predh = Predh.make["gpu", Kaiming](Optional(ctx))
    var orep = Adam(lr=Scalar[DT](3e-4))
    var odyn = Adam(lr=Scalar[DT](3e-4))
    var opred = Adam(lr=Scalar[DT](3e-4))
    var oproj = Adam(lr=Scalar[DT](3e-4))
    var opredh = Adam(lr=Scalar[DT](3e-4))

    print("EZv2 CartPole consistency-OFF ablation (v2, GPU Gumbel — pure MuZero BPTT)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K_gumbel", MAX_K, "K", K, "N", N, "B", B,
          "v±20 vcoef 0.5 clip 5 reanalyze temp CONS=0")

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
        value_coef=Scalar[DT](0.5),
        consistency_coef=Scalar[DT](0.0),
        temperature_decay_steps=30000,
        reanalyze_every=1,
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
