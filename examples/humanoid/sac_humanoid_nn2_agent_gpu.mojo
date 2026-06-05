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

SAMPLING: this variant enables **ERE** (Emphasizing Recent Experience,
Wang & Ross 2019) via `use_ere=True` — each successive gradient step in an
update burst samples from a progressively shrinking window of the most recent
transitions (window decays by `ere_eta` per update toward `ere_c_min`), so
learning emphasizes fresh on-policy data without abandoning the full buffer.
It is a SAMPLING change only — the nets are the preset's plain fused
`LinearReLU` critic (same architecture as the 6006-reward baseline), so the
checkpoint stays loadable by the preset-based eval/render script.

EVAL: a periodic GPU-parallel DETERMINISTIC eval logs `eval/mean_return` (the
deployable-policy signal) — the always-on stochastic `avg_reward` under-reports
SAC by the entropy term.

NOTE on checkpointing: the batched `train` entry point auto-saves the SAC
weights+optimizers (one-file `nn2-ckpt v2`) every `CHECKPOINT_EVERY` env-steps
and once at the end (a host-side D2H between iterations, safe with the CUDA-
graph capture). The agent is the shared `SAC[...]` preset, so this checkpoint
(`sac_humanoid_nn2_ere.ckpt`) loads directly into `sac_humanoid_nn2_eval_cpu.mojo`
for rendering.

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
from mojo_rl.deep_agents2.sac import SAC
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
comptime BATCH = 512
comptime REPLAY_CAPACITY = 1_000_000
# Humanoid physics (NV=23) is the heaviest model here — its per-env RK4
# workspace (mass matrix ∝ NV² + contacts) is replicated across all N_ENVS.
# Kept small to avoid OOM (Ant at NV=14 already OOMs at 32). Bump up if you
# have GPU headroom.
comptime N_ENVS = 32

# Training duration. Drop NUM_STEPS to ~50_000 for a smoke run.
comptime NUM_STEPS = 3_000_000
comptime WARMUP_STEPS = 25_000
comptime PRINT_EVERY = 50_000
comptime DIAG_EVERY = 1_000  # full metric-bundle flush cadence (mean_q, …)
comptime CHECKPOINT_EVERY = 50_000  # auto-save cadence (env steps)
# Plain-critic preset (same arch as the 6006 baseline `sac_humanoid_nn2.ckpt`),
# so this checkpoint IS loadable by the preset-based eval script. Distinct path
# to preserve the baseline ckpt while testing the ERE variant.
comptime CHECKPOINT_PATH = "sac_humanoid_nn2_ere.ckpt"

# Periodic DETERMINISTIC eval (greedy, no exploration noise) on an isolated set
# of `EVAL_ENVS` parallel envs — the deployable-policy signal. The always-on
# `avg_reward` is a STOCHASTIC rollout that under-reports SAC by the entropy
# term (training showed ~2655 stochastic vs ~5700 greedy), so `eval/mean_return`
# is the curve to trust. Eval runs GPU-parallel and touches no replay/optimizer
# state. VRAM: a 2nd BatchedGpuEnv adds EVAL_ENVS more per-env RK4 workspaces
# (NV=23) — kept at 16 (< N_ENVS) to stay within headroom; drop if OOM.
comptime EVAL_ENVS = 16
comptime EVAL_EVERY = 250_000  # env-steps between eval passes (~40 over 10M)
comptime EVAL_EPISODES = 16  # <= EVAL_ENVS → completes in one eval window


comptime BatchedEnvT = BatchedGpuEnv[EnvT, N_ENVS, OBS_DIM, ACT_DIM]
comptime EvalEnvT = BatchedGpuEnv[EnvT, EVAL_ENVS, OBS_DIM, ACT_DIM]

# Actor + twin critics come from the `SAC[...]` preset (canonical fused
# `LinearReLU` nets at HIDDEN=256) — the same architecture as the 6006-reward
# baseline. The experiment here is ERE (Emphasizing Recent Experience), not an
# architecture change, so the checkpoint stays portable to the preset-based
# eval/render script.


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
            run_name="SAC Humanoid NN2 (GPU, ERE, H256, 3M)",
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
        # `SAC[target, OBS, ACT, BATCH, CAP, HIDDEN]` builds the SACAgent with
        # the fused default nets + SAC's tuned defaults (lr=3e-4, gamma=0.99,
        # tau=0.005, init_alpha=0.2, target_entropy=-ACT). Humanoid overrides
        # action_scale=0.4 + warmup/window knobs. THIS RUN: `use_ere=True`
        # enables ERE (Emphasizing Recent Experience) — each gradient step
        # samples from a shrinking recent-history window (size decays by
        # `ere_eta` per update down to `ere_c_min`), biasing learning toward
        # fresh transitions. It's a sampling change only (no architecture /
        # checkpoint change vs the plain-critic baseline).
        var agent = SAC[
            "gpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
        ](
            ctx=ctx,
            action_scale=0.4,  # match legacy Humanoid SAC runs
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
            use_ere=True,
            ere_eta=0.996,  # standard ERE recency-decay (Wang & Ross 2019)
            ere_c_min=1,
            ere_k_max=1000,
        )
        var env = BatchedEnvT(ctx)
        # Isolated eval env (greedy deterministic rollouts; never touches the
        # training env's state or the replay buffer).
        var eval_env = EvalEnvT(ctx)
        var eval_env_ptr = UnsafePointer(to=eval_env)

        # ─── Single train() call — batched GPU off-policy driver ─────────
        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            L=RemoteLogger,
            USE_TRAIN_CUDA_GRAPH=True,
            USE_ENV_CUDA_GRAPH=True,
            EE=EvalEnvT,
            EVAL_ENVS=EVAL_ENVS,
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
            eval_env=eval_env_ptr,
            eval_every=EVAL_EVERY,
            eval_episodes=EVAL_EPISODES,
            eval_max_steps=1000,
        )
        _ = eval_env  # lifetime extender for eval_env_ptr
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
