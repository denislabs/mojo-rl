"""MuZero CartPole with metrics logging (CPU smoke) — CsvLogger.

Exercises the MuZero remote-logger parity path through the `MuZeroAgent` facade
(``"cpu"`` target → `run_muzero_selfplay_cpu`): per-batch diagnostics on
``diag_every`` (loss + loss_policy/value/reward split, policy_ce/entropy/target
stats/value_mse via `append_mz_train_diagnostics`), ``eval_return`` on eval, and
episode/replay status on ``report_every``. Swap `CsvLogger` for `RemoteLogger`
to ship to the dashboard.

Run (no GPU):
    pixi run mojo run -I . examples/cartpole/muzero_cartpole_logged_cpu.mojo
"""

from std.memory import UnsafePointer

from mojo_rl.nn.constants import DT
from mojo_rl.core.logger import CsvLogger
from mojo_rl.deep_agents.muzero import MuZeroMLPConfig, MuZeroAgent
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime Env = CartPoleEnv[DType.float64]
    comptime Cfg = MuZeroMLPConfig[OBS=4, ACT=2, LATENT=128, HIDDEN=128, BINS=51]
    comptime Agent = MuZeroAgent[
        "cpu", Env,
        Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=25, MAX_NODES=128, CAP=50000, B=128, K=5, N=10,
    ]

    var env = Env()
    var agent = Agent(
        ctx=None,
        lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
    )

    var logger = CsvLogger("/tmp/muzero_cartpole_metrics.csv", buffer_size=64)
    logger.set_config("algo", "muzero")
    logger.set_config("env", "cartpole")

    print("MuZero CartPole logged smoke (CSV) — diag/report 200")
    _ = agent.train[L=CsvLogger](
        env,
        iterations=2000,
        learning_starts=500,
        train_per_iter=1,
        temperature_decay_steps=2000,
        reanalyze_every=1,
        eval_every=1000,
        eval_episodes=3,
        diag_every=200,
        report_every=200,
        logger=UnsafePointer(to=logger).as_unsafe_any_origin(),
        verbose=True,
    )
    logger.close()
    print("metrics written to /tmp/muzero_cartpole_metrics.csv")
