"""MuZero CartPole with metrics logging (CPU smoke) — CsvLogger.

Exercises the MuZero remote-logger parity path: per-batch diagnostics on
``diag_every`` (loss + loss_policy/value/reward split, policy_ce/entropy/target
stats/value_mse via `append_mz_train_diagnostics`), ``eval_return`` on eval, and
episode/replay status on ``report_every``. Swap `CsvLogger` for `RemoteLogger`
to ship to the dashboard.

Run (no GPU):
    pixi run mojo run -I . examples/cartpole/muzero_cartpole_logged_cpu.mojo
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core.logger import CsvLogger
from mojo_rl.deep_agents2.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents2.muzero.selfplay_cpu import run_muzero_selfplay_cpu
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 128
    comptime BINS = 51
    comptime H = 128
    comptime NUM_SIMS = 25
    comptime MAX_NODES = 128
    comptime CAP = 50000
    comptime B = 128
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["cpu", INIT=Kaiming]()
    var dyn = Dyn.make["cpu", INIT=Kaiming]()
    var pred = Pred.make["cpu", INIT=Kaiming]()
    var orep = Adam.make["cpu", M=Rep](rep)
    var odyn = Adam.make["cpu", M=Dyn](dyn)
    var opred = Adam.make["cpu", M=Pred](pred)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)

    var logger = CsvLogger("/tmp/muzero_cartpole_metrics.csv", buffer_size=64)
    logger.set_config("algo", "muzero")
    logger.set_config("env", "cartpole")

    print("MuZero CartPole logged smoke (CSV) — diag/report 200")
    _ = run_muzero_selfplay_cpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, CAP, B, K, N,
        L=CsvLogger,
    ](
        env, rep, dyn, pred, orep, odyn, opred,
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
    print("metrics written to /tmp/muzero_cartpole_metrics.csv")
