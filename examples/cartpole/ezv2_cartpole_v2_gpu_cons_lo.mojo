"""EfficientZeroV2 discrete CartPole — consistency dose-response (v2, GPU Gumbel).

Sweep point between the full bundle (`ezv2_cartpole_v2_gpu`, consistency 2.0,
plateaus ~230) and the consistency-OFF ablation (`ezv2_cartpole_v2_gpu_noconsist`,
solves 500). Everything else matches the 64/64/30k bundle; only the SimSiam
consistency weight changes.

Edit ``CONS`` and re-run to trace the dose-response (suggested: 0.25, 0.5, 1.0):
  * If a low coef SOLVES *and* learns faster early than consistency-off → SimSiam
    helps at the right strength and 2.0 was simply too hot for state-based obs.
  * If every nonzero coef plateaus (monotonic harm) → the consistency objective
    fights this task; the default should drop toward 0 for low-dim state envs.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/ezv2_cartpole_v2_gpu_cons_lo.mojo
"""

from max.gpu.host import DeviceContext

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

    # ── sweep knob: edit this and re-run (try 0.25, 0.5, 1.0) ──
    comptime CONS = 0.5

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

    print("EZv2 CartPole consistency dose-response (v2, GPU Gumbel)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K_gumbel", MAX_K, "K", K, "N", N, "B", B,
          "v±20 vcoef 0.5 clip 5 reanalyze temp CONS", CONS)

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
        consistency_coef=Scalar[DT](CONS),
        temperature_decay_steps=30000,
        reanalyze_every=1,
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
