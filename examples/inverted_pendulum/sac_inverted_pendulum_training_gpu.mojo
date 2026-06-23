"""SAC training on InvertedPendulum (GPU, multi-env) via the new `SACAgent` facade.

GPU successor of `sac_inverted_pendulum_training.mojo` and counterpart of the
legacy `sac_inverted_pendulum_training_gpu.mojo`. Mirrors
`examples/half_cheetah/sac_half_cheetah_training_gpu.mojo`:

  * `SAC["gpu", ...]` preset — builds the `SACAgent` facade over the GPU
    `SACTrainer` + the batched off-policy driver. All optimizers, the replay
    buffer, and the SAC train-step pipeline run on-device.
  * `BatchedGpuEnv[InvertedPendulum[DT], N_ENVS, OBS, ACT]` — wraps the
    physics3d env (`GPUContinuousEnv`) into a `BatchedEnv`.
  * `RemoteLogger` — streams `env/mean_ret` and `env/ep_count`.

`updates_per_step=N_ENVS` keeps the effective UTD = 1 per collected transition.

InvertedPendulum (Phyics3dEnv, MuJoCo-style):
  * 4D observation (qpos[0:2] + qvel[0:2])
  * 1D continuous action (cart slider force)
  * Reward = +1 per step while upright; episode ends when the pole falls
    (`TERMINATE_ON_UNHEALTHY=True`). Max return ≈ 1000.

Run:
    pixi run -e apple  mojo run -I . examples/inverted_pendulum/sac_inverted_pendulum_training_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/inverted_pendulum/sac_inverted_pendulum_training_gpu.mojo  # NVIDIA GPU
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv
from mojo_rl.envs.inverted_pendulum import InvertedPendulum


# =============================================================================
# Architecture
# =============================================================================

comptime EnvT = InvertedPendulum[DT, TERMINATE_ON_UNHEALTHY=True]
comptime OBS_DIM = EnvT.OBS_DIM  # 4
comptime ACT_DIM = EnvT.ACTION_DIM  # 1
comptime HIDDEN = 256

# Off-policy GPU training parameters (mirror the legacy GPU script).
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
# Sized to the legacy `sac_inverted_pendulum_training_gpu.mojo` env count.
comptime N_ENVS = 8

# Training duration. Drop NUM_STEPS to ~50_000 for a smoke run.
comptime NUM_STEPS = 300_000
comptime WARMUP_STEPS = 5_000
comptime PRINT_EVERY = 25_000


comptime BatchedEnvT = BatchedGpuEnv[EnvT, N_ENVS, OBS_DIM, ACT_DIM]

# Actor + twin critics come from the `SAC[...]` preset (deep_agents.sac),
# which bundles the canonical fused-`LinearReLU` `SACActorNet` /
# `SACCriticNet` (matmul+bias+ReLU in one kernel — halves the per-hidden-
# layer launch count on the eager GPU path) plus SAC's tuned defaults.


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — InvertedPendulum GPU (multi-env) + logger")
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
            run_name="SAC InvertedPendulum NN (GPU)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "InvertedPendulum")
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))

        var logger_ptr = UnsafePointer(to=logger)

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
        if final_avg > 950.0:
            print("EXCELLENT — balancing reliably (mean > 950).")
        elif final_avg > 500.0:
            print("STRONG — mostly upright (mean > 500).")
        elif final_avg > 100.0:
            print("PROGRESS — learning to balance (mean > 100).")
        elif final_avg > 10.0:
            print("LEARNING — some control (mean > 10).")
        else:
            print("EARLY — still falling fast (mean < 10).")
        print("=" * 70)
