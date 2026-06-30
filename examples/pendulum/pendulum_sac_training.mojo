"""SAC training on Pendulum V1 via the storage `SAC[...]` preset facade.

Uses the storage `deep_agents/` surface:

  * `SAC[target, OBS, ACT, BATCH, CAP, HIDDEN]` — preset facade that builds the
    canonical fused-`LinearReLU` SAC actor/twin-critic nets with SAC's tuned
    defaults (lr=3e-4, gamma=0.99, tau=0.005, init_alpha=0.2,
    target_entropy=-ACT, …) over the single-env off-policy driver.

Pendulum V1 (CPU single-env):
  * 3D observation, 1D continuous action (torque).

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_training.mojo
"""

from std.random import seed

from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC

from mojo_rl.envs.pendulum import PendulumEnv


comptime EnvT = PendulumEnv[DT]
comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime NUM_STEPS = 30_000
comptime PRINT_EVERY = 1_000


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn SAC (SAC facade) — Pendulum V1 (CPU)")
    print("=" * 70)

    var logger = RemoteLogger(
        server_url="",
        run_name="SAC Pendulum NN (CPU)",
        buffer_size=200,
    )
    logger.set_config("algorithm", "SAC")
    logger.set_config("env", "Pendulum-v1")
    logger.set_config("seed", "42")

    var logger_ptr = UnsafePointer(to=logger).as_unsafe_any_origin()

    var agent = SAC[
        "cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
    ](
        window_size=10,
        initial_episode_fill=-1250.0,
    )
    var env = EnvT()

    _ = agent.train_single[
        EnvT,
        L=RemoteLogger,
    ](
        env,
        NUM_STEPS,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
    )
    logger.close()
    _ = logger  # lifetime extender for logger_ptr

    print("=" * 70)
    var final_mean = agent.mean_return()
    print("Final mean ep return (last 10): ", final_mean)
    print("Episodes completed:             ", agent.ep_count())
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)
