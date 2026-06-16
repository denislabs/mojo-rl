"""SAC training on Pendulum V1 via SACAgent facade + RemoteLogger.

Demonstrates the Track-1 monitoring path with the post-#23 driver-level
logger threading: pass a single `logger=` kwarg to `agent.train()` and
the off-policy driver emits `avg_reward` + `episodes` at the
`print_every` cadence automatically. After training, drain the
trainer-side `SACMetrics` bundle via `agent.flush_metrics(logger)`
and round-trip the agent through `save()` / `load()`.

The dashboard endpoint defaults to `http://localhost:3000/api`.
`RemoteLogger` silently swallows HTTP errors, so this example runs
end-to-end even without a server listening.

Bit-identity: when `logger=None` (default) the entire emit path is
comptime-elided via `comptime if L.ENABLED` — `pendulum_sac_nn_driver.mojo`
still produces `mean_ret(10) = -169.04118` at 30k steps seed=42.

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_sac_nn_agent_remote.mojo
"""

from std.random import seed

from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents.training.batched_env import BatchedCpuEnv

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime TOTAL_TIMESTEPS = 3_000
comptime PRINT_EVERY = 500
comptime CHECKPOINT_DIR = "/tmp/sac_pendulum_ckpt_demo"

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


comptime SAC_T = SACAgent[
    "cpu",
    UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
    ActorNet,
    CriticNet,
]


def _make_agent() raises -> SAC_T:
    return SAC_T(
        actor_lr=3e-4,
        critic_lr=1e-3,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        init_alpha=0.2,
        target_entropy=-1.0,
        learning_starts=1_000,
        window_size=10,
        initial_episode_fill=-1250.0,
    )


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC + RemoteLogger demo — Pendulum V1 (CPU)")
    print("=" * 70)

    # 1. Build a RemoteLogger. If the dashboard server isn't running, all
    # flush() / log_scalar() calls silently no-op (errors swallowed by
    # the helper in mojo_rl/core/logger.mojo).
    var remote_logger = RemoteLogger(
        server_url="http://localhost:3000/api",
        run_name="sac_pendulum_remote_demo",
        buffer_size=50,
    )
    remote_logger.set_config("algorithm", "SAC")
    remote_logger.set_config("env", "Pendulum-v1")
    remote_logger.set_config("seed", "42")

    # 2. Build the agent + env.
    var agent = _make_agent()
    var template = PendulumEnv[DT]()
    var env = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](template)

    # 3. Single train() call — the driver flushes `avg_reward` and
    # `episodes` through the logger at `print_every` cadence
    # automatically (#23). Drain the SAC-specific metric bundle once
    # at the end via `agent.flush_metrics(logger)`.
    var logger_ptr = Optional[UnsafePointer[RemoteLogger, MutAnyOrigin]](
        UnsafePointer(to=remote_logger),
    )
    _ = agent.train[BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
                    N_ENVS=1, NS=1, L=RemoteLogger](
        env,
        TOTAL_TIMESTEPS,
        rng_seed=42,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
    )
    _ = remote_logger  # lifetime extender for logger_ptr

    var bundle = agent.flush_metrics[RemoteLogger](
        logger_ptr, TOTAL_TIMESTEPS,
    )
    _ = bundle^

    remote_logger.close()
    print("=" * 70)
    print("Final mean ep return (last 10): ", agent.mean_return())
    print("Total logged points:            ", remote_logger.total_logged())

    # 4. Save the agent. The directory must already exist — Mojo nightly's
    # `open(...)` doesn't auto-create parent dirs.
    try:
        with open(CHECKPOINT_DIR + "/.touch", "w") as f:
            f.write("")
    except _e:
        print(
            "WARNING: could not touch checkpoint dir `" + CHECKPOINT_DIR
            + "`. Create it manually (mkdir -p) and re-run for the save"
            " round-trip to exercise."
        )
        print("=" * 70)
        return
    agent.save(CHECKPOINT_DIR)
    print("Saved agent state to:           ", CHECKPOINT_DIR)

    # 5. Reload into a fresh agent and confirm greedy actions match.
    var probe_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    probe_obs[0] = Scalar[DT](0.5)
    probe_obs[1] = Scalar[DT](0.8)
    probe_obs[2] = Scalar[DT](-1.2)
    var act_before = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_before)

    var fresh = _make_agent()
    fresh.load(CHECKPOINT_DIR)
    var act_after = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    fresh.select_greedy_action(probe_obs, act_after)

    print(
        "Greedy action before save:      ", act_before[0],
        " after load:", act_after[0],
    )
    print("=" * 70)
