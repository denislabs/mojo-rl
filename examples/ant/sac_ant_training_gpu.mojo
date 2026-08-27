"""SAC training on Ant (GPU, multi-env) via the `SAC[...]` storage facade.

GPU successor of `sac_ant_training.mojo`. Mirrors
`examples/walker2d/sac_walker2d_training_gpu.mojo`:

  * `SAC[target, OBS, ACT, BATCH, CAP, HIDDEN]` — preset facade over the GPU
    `SACTrainer` + the batched off-policy driver. All optimizers, the replay
    buffer, and the SAC train-step pipeline run on-device.
  * `Phyics3dBatchedEnv[AntModel, AntConfig, N_ENVS]` — the physics3d env
    (`GPUContinuousEnv`) into a `BatchedEnv`.
  * `RemoteLogger` — streams `env/mean_ret` and `env/ep_count`.

`updates_per_step=N_ENVS` keeps the effective UTD = 1 per collected transition.

Ant (Phyics3dEnv, MuJoCo-style):
  * 27D observation (qpos[2:] + qvel)
  * 8D continuous action (joint torques)
  * Reward ≈ forward velocity + healthy bonus − control/contact costs;
    episode ends when the torso leaves a healthy z-range
    (`TERMINATE_ON_UNHEALTHY=True`).

Run:
    pixi run -e apple  mojo run -I . examples/ant/sac_ant_training_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/ant/sac_ant_training_gpu.mojo  # NVIDIA GPU
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.ant import AntModel, AntConfig


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = AntModel.OBS_DIM  # 27
comptime ACT_DIM = AntModel.ACTION_DIM  # 8
comptime HIDDEN = 256

# Off-policy GPU training parameters (mirror the legacy GPU script).
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
# Ant physics (NV=14) allocates a large per-env RK4 workspace (mass matrix
# ∝ NV² + contacts), replicated across all N_ENVS. 32 envs OOM the GPU; the
# legacy `sac_ant_training_gpu.mojo` used 4. Bump this up if you have headroom.
comptime N_ENVS = 16

# Training duration. Drop NUM_STEPS to ~50_000 for a smoke run.
comptime NUM_STEPS = 3_000_000
comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = 50_000


comptime BatchedEnvT = Phyics3dBatchedEnv[
    AntModel, AntConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=True
]

# Actor + twin critics come from the `SAC[...]` preset (deep_agents.sac),
# which bundles the canonical fused-`LinearReLU` `SACActorNet` /
# `SACCriticNet` (matmul+bias+ReLU in one kernel — halves the per-hidden-
# layer launch count on the eager GPU path) plus SAC's tuned defaults.


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — Ant GPU (multi-env) + logger")
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
            run_name="SAC Ant NN (GPU)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "Ant")
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))

        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        # ─── Agent + batched GPU env ─────────────────────────────────────
        # `SAC[target, OBS, ACT, BATCH, CAP, HIDDEN]` reads like a
        # constructor: it builds the SACAgent with the fused default nets
        # and SAC's tuned scalar defaults (lr=3e-4, gamma=0.99, tau=0.005,
        # init_alpha=0.2, target_entropy=-ACT, …). We override only the
        # example-specific knobs below; everything else comes from the preset.
        var agent = SAC[
            "gpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
        ](
            ctx=ctx,
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
        if final_avg > 4000.0:
            print("EXCELLENT — running fast (mean > 4000).")
        elif final_avg > 2000.0:
            print("STRONG — learned locomotion (mean > 2000).")
        elif final_avg > 1000.0:
            print("PROGRESS — staying healthy + moving (mean > 1000).")
        elif final_avg > 0.0:
            print("LEARNING — positive return (mean > 0).")
        else:
            print("EARLY — still exploring (mean < 0).")
        print("=" * 70)
