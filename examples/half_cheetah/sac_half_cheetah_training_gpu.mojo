"""SAC training on HalfCheetah (GPU, multi-env) via the new `SACAgent` facade.

GPU successor of `sac_half_cheetah_training.mojo` (the CPU nn example) and
direct counterpart of the legacy `sac_half_cheetah_training_gpu.mojo` (which
uses `deep_agents.core.agents.DeepSACAgent.train_gpu`). Uses the new
`deep_agents/` surface end-to-end:

  * `SACAgent["gpu", ...]` — facade over the GPU `SACTrainer` + the batched
    off-policy driver (`run_offpolicy_train_batched`). All optimizers, the
    replay buffer, and the SAC train-step pipeline run on-device.
  * `BatchedGpuEnv[HalfCheetah[DT], N_ENVS, OBS, ACT]` — wraps the
    HalfCheetah physics3d env (which conforms to `GPUContinuousEnv`) into a
    `BatchedEnv`. Steps/resets/obs-extraction all dispatch HalfCheetah's
    native GPU physics kernels — the exact same kernels the legacy
    `train_gpu[HalfCheetah[...]]` path drives, just behind the deep_agents
    wrapper.
  * `RemoteLogger` — streams `avg_reward` + `episodes` at the driver's
    `print_every` cadence, AND (via `diag_every`) the full SAC metric bundle
    — `actor_loss`, `critic_loss`, `alpha`, `mean_q`, `mean_reward`,
    `train_steps`, … — so the dashboard shows the same panels as the
    single-env path. Config (server URL + API key) read from a `.env` via
    `mojo_rl.core.dotenv`.

`updates_per_step=N_ENVS` keeps the effective UTD = 1 per collected
transition: each driver iteration steps all `N_ENVS` envs once and runs
`N_ENVS` gradient updates.

Checkpointing: the batched `train` entry point auto-saves the SAC
weights+optimizers (one-file `nn-ckpt v2`) every `CHECKPOINT_EVERY` env-steps
and once at the end (a host-side D2H between iterations, safe with the CUDA-
graph capture). It writes `CHECKPOINT_PATH` (`sac_half_cheetah_nn.ckpt`) —
render it with `sac_half_cheetah_nn_eval_cpu.mojo`, which rebuilds the same
fused-`LinearReLU` architecture via the `SAC[...]` preset.

HalfCheetah (Phyics3dEnv, MuJoCo-style):
  * 17D observation (qpos + qvel excluding rootx and head)
  * 6D continuous action (joint torques)
  * Reward ≈ forward velocity - 0.1·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

Run:
    pixi run -e apple  mojo run -I . examples/half_cheetah/sac_half_cheetah_training_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/half_cheetah/sac_half_cheetah_training_gpu.mojo  # NVIDIA GPU
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.envs.phyics3d_batched_env_fields import Phyics3dBatchedEnvFields
from mojo_rl.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig


# =============================================================================
# Architecture (matches the CPU nn / legacy DeepSACAgent half_cheetah runs)
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
comptime DIAG_EVERY = 1_000  # full metric-bundle flush cadence (mean_q, …)
comptime CHECKPOINT_EVERY = 50_000  # auto-save cadence (env steps)
# Written by the batched trainer; loaded by `sac_half_cheetah_nn_eval_cpu.mojo`
# (same fused-`LinearReLU` architecture, so the param layout matches).
comptime CHECKPOINT_PATH = "sac_half_cheetah_nn.ckpt"


# Per-field tensor physics path (migration P5+): the batched fields facade is
# a `BatchedEnv` that runs the LEGACY PRODUCTION physics bundle by default
# (RK4 + Newton, parallel _mt schedules, treewalk CRBA, auto broadphase).
comptime BatchedEnvT = Phyics3dBatchedEnvFields[
    HalfCheetahModel, HalfCheetahConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — HalfCheetah GPU (multi-env) + logger")
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
            run_name="SAC HalfCheetah NN (GPU)",
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

        var logger_ptr = UnsafePointer(to=logger).as_unsafe_any_origin()

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
            # use_bf16=True,
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
            USE_TRAIN_CUDA_GRAPH=True,
            USE_ENV_CUDA_GRAPH=True,
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
            episode_sync_every=32,
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_path=CHECKPOINT_PATH,
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
