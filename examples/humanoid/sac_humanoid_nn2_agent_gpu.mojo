"""SAC training on Humanoid (GPU, multi-env) via the new `SACAgent` facade.

GPU successor of `sac_humanoid_nn2_agent.mojo` and counterpart of the legacy
`sac_humanoid_training_gpu.mojo`. Mirrors
`examples/half_cheetah/sac_half_cheetah_nn2_agent_gpu.mojo`:

  * `SACAgent["gpu", ...]` — facade over the GPU `SACTrainer` + the batched
    off-policy driver. All optimizers, the replay buffer, and the SAC
    train-step pipeline run on-device.
  * `BatchedGpuEnv[Humanoid[DT], N_ENVS, OBS, ACT]` — wraps the physics3d env
    (`GPUContinuousEnv`) into a `BatchedEnv`.
  * `RemoteLogger` — streams `env/mean_ret` and `env/ep_count`.

`updates_per_step=N_ENVS` keeps the effective UTD = 1 per collected transition.

NOTE on checkpointing: the facade's `save`/`load` are CPU-only and the batched
`train` entry point has no inline checkpoint/diag cadence (those live on
`train_single`). This GPU example trains + summarizes only.

Humanoid (Phyics3dEnv, MuJoCo-style):
  * 45D observation (qpos[2:24] + qvel[0:23])
  * 17D continuous action (joint torques); `action_scale=0.4` to match the
    legacy Humanoid SAC runs.
  * Reward ≈ forward velocity + healthy bonus − control/contact costs;
    episode ends when the torso leaves a healthy z-range
    (`TERMINATE_ON_UNHEALTHY=True`).

Run:
    pixi run -e apple  mojo run -I . examples/humanoid/sac_humanoid_nn2_agent_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/humanoid/sac_humanoid_nn2_agent_gpu.mojo  # NVIDIA GPU
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.sac import SACAgent
from mojo_rl.deep_agents2.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents2.training.batched_env import BatchedGpuEnv
from mojo_rl.envs.humanoid import Humanoid


# =============================================================================
# Architecture
# =============================================================================

comptime EnvT = Humanoid[DT, TERMINATE_ON_UNHEALTHY=True]
comptime OBS_DIM = EnvT.OBS_DIM  # 45
comptime ACT_DIM = EnvT.ACTION_DIM  # 17
comptime HIDDEN = 256

# Off-policy GPU training parameters (mirror the legacy GPU script).
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
# Humanoid physics (NV=23) is the heaviest model here — its per-env RK4
# workspace (mass matrix ∝ NV² + contacts) is replicated across all N_ENVS.
# Kept small to avoid OOM (Ant at NV=14 already OOMs at 32). Bump up if you
# have GPU headroom.
comptime N_ENVS = 4

# Training duration. Drop NUM_STEPS to ~50_000 for a smoke run.
comptime NUM_STEPS = 3_000_000
comptime WARMUP_STEPS = 25_000
comptime PRINT_EVERY = 50_000


comptime BatchedEnvT = BatchedGpuEnv[EnvT, N_ENVS, OBS_DIM, ACT_DIM]

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


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents2) — Humanoid GPU (multi-env) + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  N_ENVS             =", N_ENVS)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  WARMUP_STEPS       =", WARMUP_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Logger (remote) ─────────────────────────────────────────────
        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="SAC Humanoid NN2 (GPU)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "Humanoid")
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))

        var logger_ptr = UnsafePointer(to=logger)

        # ─── Agent + batched GPU env ─────────────────────────────────────
        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=0.4,  # match legacy Humanoid SAC runs
            init_alpha=0.2,
            target_entropy=-Scalar[DT](ACT_DIM),  # SAC default heuristic
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        # ─── Single train() call — batched GPU off-policy driver ─────────
        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            L=RemoteLogger,
        ](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
        )
        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        logger.close()
        _ = logger  # lifetime extender for logger_ptr

        # ─── Summary ─────────────────────────────────────────────────────
        print("-" * 70)
        print("=" * 70)
        print("Training complete")
        print("  total env_steps           =", NUM_STEPS)
        print("  elapsed                   =", elapsed_s, "s")
        print("  mean ep return (last 100) =", agent.mean_return())
        print("  episodes completed        =", agent.ep_count())
        print("  remote points sent        =", logger.total_logged())
        print("=" * 70)

        var final_avg = Float64(agent.mean_return())
        if final_avg > 5000.0:
            print("EXCELLENT — walking (mean > 5000).")
        elif final_avg > 2000.0:
            print("STRONG — sustained upright locomotion (mean > 2000).")
        elif final_avg > 1000.0:
            print("PROGRESS — staying upright (mean > 1000).")
        elif final_avg > 0.0:
            print("LEARNING — positive return (mean > 0).")
        else:
            print("EARLY — still exploring (mean < 0).")
        print("=" * 70)
