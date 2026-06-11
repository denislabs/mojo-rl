"""MuZero CartPole with metrics logging (GPU Gumbel smoke) — CsvLogger.

GPU sibling of `muzero_cartpole_logged_cpu`: wires a `CsvLogger` into the
fully-on-device Gumbel self-play driver. Emits the per-batch diagnostics on
``diag_every``, ``eval_return`` on eval, and episode/replay status on
``report_every``. Swap `CsvLogger` for `RemoteLogger` to ship to the dashboard.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/muzero_cartpole_logged_gpu.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core.logger import CsvLogger
from mojo_rl.deep_agents2.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents2.muzero.selfplay_gpu_device import (
    run_muzero_gumbel_selfplay_gpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 128
    comptime BINS = 51
    comptime H = 128
    comptime NUM_SIMS = 24
    comptime MAX_NODES = 128
    comptime MAX_K = 2
    comptime CAP = 50000
    comptime B = 128
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
    orep.max_grad_norm = Scalar[DT](10.0)
    odyn.max_grad_norm = Scalar[DT](10.0)
    opred.max_grad_norm = Scalar[DT](10.0)

    var logger = CsvLogger("/tmp/muzero_cartpole_metrics_gpu.csv", buffer_size=64)
    logger.set_config("algo", "muzero")
    logger.set_config("env", "cartpole")

    print("MuZero CartPole logged smoke GPU (CSV) — diag/report 200")
    _ = run_muzero_gumbel_selfplay_gpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        L=CsvLogger,
    ](
        ctx, env, rep, dyn, pred, orep, odyn, opred,
        iterations=2000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
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
    print("metrics written to /tmp/muzero_cartpole_metrics_gpu.csv")
