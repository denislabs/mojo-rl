"""SAC training on Walker2d (GPU, multi-env) via the new `SACAgent` facade.

GPU successor of `sac_walker2d_nn2_agent.mojo` and counterpart of the legacy
`sac_walker2d_training_gpu.mojo`. Mirrors
`examples/half_cheetah/sac_half_cheetah_nn2_agent_gpu.mojo`:

  * `SACAgent["gpu", ...]` — facade over the GPU `SACTrainer` + the batched
    off-policy driver. All optimizers, the replay buffer, and the SAC
    train-step pipeline run on-device.
  * `BatchedGpuEnv[Walker2d[DT], N_ENVS, OBS, ACT]` — wraps the physics3d env
    (`GPUContinuousEnv`) into a `BatchedEnv`.
  * `RemoteLogger` — streams `avg_reward` + `episodes` at `print_every`, AND
    (via `diag_every`) the full SAC metric bundle (`actor_loss`,
    `critic_loss`, `alpha`, `mean_q`, `mean_reward`, `train_steps`, …).

`updates_per_step=N_ENVS` keeps the effective UTD = 1 per collected transition.

NOTE on checkpointing: the batched `train` entry point now supports an inline
checkpoint cadence (`checkpoint_every` + `checkpoint_path`) — it auto-saves the
trainer's one-file `nn2-ckpt v2` envelope (actor + twin critics + optimizers +
alpha optimizer) every `CHECKPOINT_EVERY` env-steps and one final time at the
end. The save runs between iterations (a D2H of the live GPU params) so it is
safe to combine with the CUDA-graph capture below. The replay buffer / episode
tracker are NOT persisted, so a resumed run starts with a fresh replay. Load a
saved checkpoint back into a fresh agent with `agent.load(CHECKPOINT_PATH)`.

Walker2d (Phyics3dEnv, MuJoCo-style):
  * 17D observation (qpos[1:9] + qvel[0:9])
  * 6D continuous action (thigh/leg/foot torques × 2 legs)
  * Reward ≈ forward velocity + healthy bonus − control cost; episode ends
    when the torso leaves a healthy height/angle range
    (`TERMINATE_ON_UNHEALTHY=True`).

Run:
    pixi run -e apple  mojo run -I . examples/walker2d/sac_walker2d_nn2_agent_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/walker2d/sac_walker2d_nn2_agent_gpu.mojo  # NVIDIA GPU
"""

from std.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.sac import SAC
from mojo_rl.deep_agents2.training.batched_env import BatchedGpuEnv
from mojo_rl.envs.walker2d import Walker2d


# =============================================================================
# Architecture
# =============================================================================

comptime EnvT = Walker2d[DT, TERMINATE_ON_UNHEALTHY=True]
comptime OBS_DIM = EnvT.OBS_DIM  # 17
comptime ACT_DIM = EnvT.ACTION_DIM  # 6
comptime HIDDEN = 256

# Off-policy GPU training parameters (mirror the legacy GPU script).
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
# Walker2d physics (NV=9, articulated chain) allocates a sizeable per-env RK4
# workspace (mass matrix ∝ NV² + contacts), replicated across all N_ENVS. The
# legacy `sac_walker2d_training_gpu.mojo` used 4; bump up if you have headroom.
comptime N_ENVS = 4

# Training duration. Drop NUM_STEPS to ~50_000 for a smoke run.
comptime NUM_STEPS = 1_000_000
comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = 50_000
comptime DIAG_EVERY = 1_000  # full metric-bundle flush cadence (mean_q, …)
comptime CHECKPOINT_EVERY = 50_000  # auto-save cadence (env steps)
comptime CHECKPOINT_PATH = "sac_walker2d_nn2.ckpt"


comptime BatchedEnvT = BatchedGpuEnv[EnvT, N_ENVS, OBS_DIM, ACT_DIM]

# Actor + twin critics come from the `SAC[...]` preset (deep_agents2.sac),
# which bundles the canonical fused-`LinearReLU` `SACActorNet` /
# `SACCriticNet` (matmul+bias+ReLU in one kernel — halves the per-hidden-
# layer launch count on the eager GPU path) plus SAC's tuned defaults.


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents2) — Walker2d GPU (multi-env) + logger")
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
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)
    print("  CHECKPOINT_PATH    =", CHECKPOINT_PATH)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Logger (remote) ─────────────────────────────────────────────
        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="SAC Walker2d NN2 (GPU)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "Walker2d")
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
            # CUDA-graph capture of the train step. The earlier capture
            # divergence was a replay-buffer bug — the uniform sample kernel
            # took the buffer fill count as a HOST scalar, which capture baked
            # at capture time, freezing sampling to the warmup-era transitions.
            # Fixed in gpu_replay.mojo (device-resident `size`); the sample
            # range now tracks the live count on every replay. Uniform replay
            # only — do NOT combine with ERE (still host-scalar / not capture
            # safe). NVIDIA only; no-op on Apple/Metal.
            USE_TRAIN_CUDA_GRAPH=True,
            # Capture the deterministic physics step too — collapses the env's
            # per-step eager kernel launches (newton/integrators/collision) into
            # one graph replay/iteration. The decisive lever at N_ENVS=4 / 250k
            # iters, where per-iteration launch+dispatch (not GPU compute)
            # dominates wall-clock. Safe: physics3d's GPU step is RNG-free
            # (RNG only in reset, which stays eager). NVIDIA only.
            USE_ENV_CUDA_GRAPH=True,
        ](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            # Defer the per-iteration episode-tracking D2H+synchronize: batch
            # the reward/done readback over this many iterations so the host
            # only stalls the GPU pipeline ~1/32 as often (returns are drained
            # exactly at every print/diag boundary, so logged values are fresh).
            episode_sync_every=32,
            # Auto-save the SAC weights+optimizers every CHECKPOINT_EVERY
            # env-steps (and once more at the end). Safe alongside the
            # CUDA-graph capture above — the save is host-side D2H between
            # iterations. Resume/eval later via `agent.load(CHECKPOINT_PATH)`.
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
        print("  checkpoint saved to       =", CHECKPOINT_PATH)
        print("=" * 70)

        var final_avg = Float64(agent.mean_return())
        if final_avg > 4000.0:
            print("EXCELLENT — walking fast (mean > 4000).")
        elif final_avg > 2000.0:
            print("STRONG — sustained walking (mean > 2000).")
        elif final_avg > 1000.0:
            print("PROGRESS — staying upright + moving (mean > 1000).")
        elif final_avg > 0.0:
            print("LEARNING — positive return (mean > 0).")
        else:
            print("EARLY — still exploring (mean < 0).")
        print("=" * 70)
