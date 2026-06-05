"""EfficientZeroV2 continuous Pendulum convergence run (v2, GPU sampled-Gumbel).

The continuous lighthouse for the deep_agents2 EZv2 port — the continuous twin of
`examples/cartpole/ezv2_cartpole_v2_gpu.mojo`. Search runs on the
`SampledGumbelGPUMCTS` planner (Gumbel-Top-k + sequential halving over *sampled*
continuous action vectors via the squashed-Gaussian head `MZContPredGPU`), and
the update is `ezv2_unroll_train_step_continuous_gpu` (MuZero BPTT + SimSiam
consistency + squashed-Gaussian policy NLL) on the resident GPU nets. Watch the
greedy ``[eval]`` return — random Pendulum ≈ −1200..−1600, a good policy
≈ −150..−250 (swung up and held).

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/pendulum/ezv2_pendulum_v2_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents2.efficient_zero_v2.nets_continuous import EZContPredNet
from mojo_rl.deep_agents2.efficient_zero_v2.selfplay_gpu_continuous import (
    run_ezv2_sampled_selfplay_gpu,
)
from mojo_rl.envs.pendulum import PendulumEnv


def main() raises:
    comptime OBS = 3
    comptime ACT_DIM = 1
    comptime LATENT = 64
    comptime BINS = 51
    comptime H = 64
    comptime PROJ = 128
    comptime PROJ_HID = 128
    comptime BOTTLENECK = 64
    comptime NUM_SIMS = 64        # 64 sims over 8 root candidates ≈ 8 visits each
    comptime MAX_NODES = 128
    comptime K_ROOT = 8
    comptime K_NON_ROOT = 4
    comptime CAP = 50000
    comptime B = 128
    comptime K = 5
    comptime N = 5

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT_DIM, BINS, H]
    comptime Pred = EZContPredNet[LATENT, ACT_DIM, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var ctx = DeviceContext()
    var env = PendulumEnv[DType.float32]()
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

    print("EZv2 Pendulum convergence (v2, GPU sampled-Gumbel — MuZero BPTT + SimSiam)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K_root", K_ROOT, "K", K, "N", N, "B", B)

    var loss = run_ezv2_sampled_selfplay_gpu[
        PendulumEnv[DType.float32], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT_DIM, LATENT, BINS, NUM_SIMS, MAX_NODES, K_ROOT, K_NON_ROOT,
        CAP, B, K, N,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=30000,
        learning_starts=2000,
        train_per_iter=1,
        gamma=Scalar[DT](0.99),
        # value/reward two-hot support in h-space. Pendulum n-step targets (N=5)
        # live in ~h[-17, 0]; the old [-50, 2] wasted >80% of the 51 bins, so
        # tighten to [-20, 1] for ~2.5x finer value resolution (margin avoids
        # clipping the most-negative bootstrap tail ≈ h(-313) ≈ -16.7).
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](1.0),
        max_action=Scalar[DT](2.0),
        min_std=Scalar[DT](0.5),
        std_magnification=Scalar[DT](3.0),
        ent_scale=Scalar[DT](0.05),
        max_ep_steps=200,
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        # stale-target fix: lagging target net + reanalyze refresh (~2 stored
        # positions re-searched per train step once warmed up).
        target_sync_interval=200,
        reanalyze_interval=1,
        reanalyze_warmup=500,
        reanalyze_batch=2,
        eval_every=2000,
        eval_episodes=5,
        seed=42,
        verbose=True,
    )

    print("final loss:", loss)
