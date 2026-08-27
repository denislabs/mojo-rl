"""SAC training on Pendulum V1 via the storage `SAC[...]` facade + RemoteLogger.

Demonstrates the Track-1 monitoring path: pass a `logger=` kwarg to
`agent.train_single()` and the off-policy driver emits `avg_reward` +
`episodes` at the `print_every` cadence automatically. After training, the
agent is round-tripped through `save()` / `load()`.

The dashboard endpoint defaults to `http://localhost:3000/api`. `RemoteLogger`
silently swallows HTTP errors, so this example runs end-to-end even without a
server listening.

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_training_remote.mojo
"""

from std.random import seed

from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from mojo_rl.deep_agents.training.blocks import ReplaySampleStep
from mojo_rl.deep_agents.data.any_replay import AnyReplay

from mojo_rl.envs.pendulum import PendulumEnv


comptime EnvT = PendulumEnv[DT]
comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime NUM_STEPS = 3_000
comptime PRINT_EVERY = 500
comptime CHECKPOINT_PATH = "/tmp/sac_pendulum_remote.ckpt"


comptime SAC_T = SACAgent[
    "cpu",
    ReplaySampleStep[
        AnyReplay["cpu", OBS_DIM, ACT_DIM, REPLAY_CAPACITY], BATCH
    ],
    SACActorNet[OBS_DIM, ACT_DIM, HIDDEN],
    SACCriticNet[OBS_DIM, ACT_DIM, HIDDEN],
]


def _make_agent() raises -> SAC_T:
    return SAC[
        "cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
    ](
        window_size=10,
        initial_episode_fill=-1250.0,
    )


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC + RemoteLogger demo — Pendulum V1 (CPU)")
    print("=" * 70)

    # 1. Build a RemoteLogger. If the dashboard server isn't running, all
    # flush() / log_scalar() calls silently no-op (errors swallowed by the
    # helper in mojo_rl/core/logger.mojo).
    var logger = RemoteLogger(
        server_url="http://localhost:3000/api",
        run_name="sac_pendulum_remote_demo",
        buffer_size=50,
    )
    logger.set_config("algorithm", "SAC")
    logger.set_config("env", "Pendulum-v1")
    logger.set_config("seed", "42")

    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    # 2. Build the agent + env.
    var agent = _make_agent()
    var env = EnvT()

    # 3. Single train() call — the driver flushes `avg_reward` and `episodes`
    # through the logger at `print_every` cadence automatically.
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
    print("Final mean ep return (last 10): ", agent.mean_return())
    print("Total logged points:            ", logger.total_logged())

    # 4. Save the agent (single-file `.ckpt`).
    agent.save(CHECKPOINT_PATH)
    print("Saved agent state to:           ", CHECKPOINT_PATH)

    # 5. Probe greedy action, reload into a fresh agent, confirm it matches.
    var probe_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    probe_obs[0] = Scalar[DT](0.5)
    probe_obs[1] = Scalar[DT](0.8)
    probe_obs[2] = Scalar[DT](-1.2)
    var act_before = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_before)

    var fresh = _make_agent()
    fresh.load(CHECKPOINT_PATH)
    var act_after = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    fresh.select_greedy_action(probe_obs, act_after)

    print(
        "Greedy action before save:      ", act_before[0],
        " after load:", act_after[0],
    )
    print("=" * 70)
