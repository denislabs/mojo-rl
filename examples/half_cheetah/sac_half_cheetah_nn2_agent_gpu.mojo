"""SAC training on HalfCheetah (GPU, multi-env) via the new `SACAgent` facade.

GPU successor of `sac_half_cheetah_nn2_agent.mojo` (the CPU nn2 example) and
direct counterpart of the legacy `sac_half_cheetah_training_gpu.mojo` (which
uses `deep_agents.core.agents.DeepSACAgent.train_gpu`). Uses the new
`deep_agents2/` surface end-to-end:

  * `SACAgent["gpu", ...]` — facade over the GPU `SACTrainer` + the batched
    off-policy driver (`run_offpolicy_train_batched`). All optimizers, the
    replay buffer, and the SAC train-step pipeline run on-device.
  * `BatchedGpuEnv[HalfCheetah[DT], N_ENVS, OBS, ACT]` — wraps the
    HalfCheetah physics3d env (which conforms to `GPUContinuousEnv`) into a
    `BatchedEnv`. Steps/resets/obs-extraction all dispatch HalfCheetah's
    native GPU physics kernels — the exact same kernels the legacy
    `train_gpu[HalfCheetah[...]]` path drives, just behind the deep_agents2
    wrapper.
  * `RemoteLogger` — streams `env/mean_ret` and `env/ep_count` at the
    driver's `print_every` cadence. Config (server URL + API key) read from
    a `.env` via `mojo_rl.core.dotenv`.

`updates_per_step=N_ENVS` keeps the effective UTD = 1 per collected
transition: each driver iteration steps all `N_ENVS` envs once and runs
`N_ENVS` gradient updates.

NOTE on checkpointing: the facade's `save`/`load` are CPU-only (they write a
`nn2-ckpt v2` envelope from host-resident params), and the batched `train`
entry point has no inline checkpoint/diag cadence (those live on
`train_single`). This GPU example therefore trains + summarizes only; for
mid-run checkpointing use the CPU example or the single-env cross-target path.

HalfCheetah (Phyics3dEnv, MuJoCo-style):
  * 17D observation (qpos + qvel excluding rootx and head)
  * 6D continuous action (joint torques)
  * Reward ≈ forward velocity - 0.1·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

Run:
    pixi run -e apple  mojo run -I . examples/half_cheetah/sac_half_cheetah_nn2_agent_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/half_cheetah/sac_half_cheetah_nn2_agent_gpu.mojo  # NVIDIA GPU
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
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture (matches the CPU nn2 / legacy DeepSACAgent half_cheetah runs)
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256

# Off-policy GPU training parameters (mirror the legacy GPU script).
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
comptime N_ENVS = 32

# Training duration (off-policy uses env steps, not episodes). Drop NUM_STEPS
# to ~50_000 for a smoke run.
comptime NUM_STEPS = 600_000
comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = 50_000  # driver-cadence verbose + env/mean_ret emit
comptime DIAG_EVERY = 10_000  # full metric-bundle flush cadence (mean_q, …)


comptime EnvT = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]
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
    print("SAC (deep_agents2) — HalfCheetah GPU (multi-env) + logger")
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
            run_name="SAC HalfCheetah NN2 (GPU)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))

        var logger_ptr = UnsafePointer(to=logger)

        # ─── Agent + batched GPU env ─────────────────────────────────────
        # GPU training: the DeviceContext MUST be threaded through the agent
        # (the trainer keeps it for H2D/D2H staging + all on-device kernels).
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
            action_scale=1.0,
            init_alpha=0.2,
            target_entropy=-Scalar[DT](ACT_DIM),  # SAC default heuristic
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        # ─── Single train() call — batched GPU off-policy driver ─────────
        # Every PRINT_EVERY env-steps the driver emits `env/mean_ret` +
        # `env/ep_count` through the logger and prints a progress line.
        # `updates_per_step=N_ENVS` ⇒ effective UTD = 1 per transition.
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
            diag_every=DIAG_EVERY,
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
        elif final_avg > 1000.0:
            print("STRONG — learned locomotion (mean > 1000).")
        elif final_avg > 100.0:
            print("PROGRESS — early locomotion (mean > 100).")
        elif final_avg > 0.0:
            print("LEARNING — positive return (mean > 0).")
        else:
            print("EARLY — still exploring (mean < 0).")
        print("=" * 70)
