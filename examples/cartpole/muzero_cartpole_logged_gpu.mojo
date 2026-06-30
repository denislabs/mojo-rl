"""MuZero CartPole with metrics logging (GPU Gumbel smoke) — CsvLogger.

GPU sibling of `muzero_cartpole_logged_cpu`: drives the `MuZeroAgent` facade on
the ``"gpu"`` target (whose `train` wires the fully-on-device Gumbel self-play
driver) with a `CsvLogger` attached. Emits the per-batch diagnostics on
``diag_every``, ``eval_return`` on eval, and episode/replay status on
``report_every``. Swap `CsvLogger` for `RemoteLogger` to ship to the dashboard.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/muzero_cartpole_logged_gpu.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.logger import CsvLogger
from mojo_rl.deep_agents.muzero import MuZeroMLPConfig, MuZeroAgent
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime Env = CartPoleEnv[DType.float64]
    comptime Cfg = MuZeroMLPConfig[OBS=4, ACT=2, LATENT=128, HIDDEN=128, BINS=51]
    comptime Agent = MuZeroAgent[
        "gpu", Env,
        Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=24, MAX_NODES=128, CAP=50000, B=128, K=5, N=10, MAX_K=2,
    ]

    var ctx = DeviceContext()
    var env = Env()
    var agent = Agent(
        ctx=ctx,
        lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
        max_grad_norm=Scalar[DT](10.0),
    )

    var logger = CsvLogger("/tmp/muzero_cartpole_metrics_gpu.csv", buffer_size=64)
    logger.set_config("algo", "muzero")
    logger.set_config("env", "cartpole")

    print("MuZero CartPole logged smoke GPU (CSV) — diag/report 200")
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
    print("metrics written to /tmp/muzero_cartpole_metrics_gpu.csv")
