"""EZv2 CartPole with metrics logging (GPU Gumbel smoke) — CsvLogger.

GPU sibling of `ezv2_cartpole_logged_cpu`: wires a `CsvLogger` into the on-device
Gumbel self-play driver. Emits the per-batch diagnostics (loss + loss split +
policy_ce/value_mse/target stats) on ``diag_every``, ``eval_return`` on eval, and
episode/replay status on ``report_every``. Swap `CsvLogger` for `RemoteLogger`
to ship to the dashboard.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/ezv2_cartpole_logged_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.core.logger import CsvLogger
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

    var logger = CsvLogger("/tmp/ezv2_cartpole_metrics_gpu.csv", buffer_size=64)
    logger.set_config("algo", "ezv2")
    logger.set_config("env", "cartpole")

    print("EZv2 CartPole logged smoke GPU (CSV) — diag/report 200")
    _ = run_ezv2_gumbel_selfplay_gpu[
        CartPoleEnv[DType.float32], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        L=CsvLogger,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=2000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](0.5),
        consistency_coef=Scalar[DT](2.0),
        temperature_decay_steps=2000,
        reanalyze_every=1,
        eval_every=1000,
        eval_episodes=3,
        diag_every=200,
        report_every=200,
        logger=UnsafePointer(to=logger),
        verbose=True,
    )
    logger.close()
    print("metrics written to /tmp/ezv2_cartpole_metrics_gpu.csv")
