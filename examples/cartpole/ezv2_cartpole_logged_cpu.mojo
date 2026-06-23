"""EZv2 CartPole with metrics logging (CPU smoke) — CsvLogger.

Exercises the MuZero/EZv2 logger parity path: runs the EZv2 CPU self-play loop
with a `CsvLogger` wired in, emitting per-batch diagnostics on ``diag_every``
(loss + loss_policy/value/reward/consistency split, latent_std/proj_norm_std,
policy_ce/entropy/target stats/value_mse via `append_mz_train_diagnostics`) and
episode/replay status on ``report_every``, plus ``eval_return`` on eval. Swap
`CsvLogger` for `RemoteLogger(server_url=..., run_name=...)` to ship to the
dashboard instead.

Run (no GPU):
    pixi run mojo run -I . examples/cartpole/ezv2_cartpole_logged_cpu.mojo
    # metrics land in /tmp/ezv2_cartpole_metrics.csv
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.core.logger import CsvLogger
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

    var logger = CsvLogger("/tmp/ezv2_cartpole_metrics.csv", buffer_size=64)
    logger.set_config("algo", "ezv2")
    logger.set_config("env", "cartpole")

    print("EZv2 CartPole logged smoke (CSV) — diag_every 200 report_every 200")
    _ = run_ezv2_selfplay_cpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
        L=CsvLogger,
    ](
        env, rep, dyn, pred, proj, predh,
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
    print("metrics written to /tmp/ezv2_cartpole_metrics.csv")
