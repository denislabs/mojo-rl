"""SAC training on Hopper (CPU) via the new `SACAgent` facade.

CPU nn2 counterpart of the legacy `sac_hopper_training_gpu.mojo` (which uses
`deep_agents.core.agents.DeepSACAgent`). Mirrors the half_cheetah nn2 CPU
example (`sac_half_cheetah_nn2_agent.mojo`) with Hopper's dimensions and the
ERE/early-termination settings the legacy Hopper script uses.

  * `SACAgent["cpu", ...]` — facade over `SACTrainer` + the single-env
    off-policy driver. ERE (Emphasizing Recent Experience) on, matching the
    legacy Hopper recipe.
  * `RemoteLogger` — streams metrics to a dashboard at the driver's
    `print_every`/`diag_every` cadence. Config (server URL + API key) read
    from a `.env` via `mojo_rl.core.dotenv`.
  * Single-file checkpointing — `agent.save(CHECKPOINT_PATH)` writes ONE
    `.ckpt` file (overwritten each cadence) under a single `nn2-ckpt v2`
    envelope containing actor + twin critics + their Adam states +
    `alpha_opt` ScalarAdam.

After training, the final checkpoint is reloaded into the same agent and a
greedy probe confirms the action vector reproduces dimension-by-dimension to
`|diff| < 1e-5`.

Hopper (Phyics3dEnv, MuJoCo-style):
  * 11D observation (qpos + qvel excluding rootx)
  * 3D continuous action (joint torques)
  * Reward ≈ forward velocity + alive bonus - 1e-3·||action||²
  * Early termination on unhealthy state (`TERMINATE_ON_UNHEALTHY=True`).

Run:
    pixi run mojo run -I . examples/hopper/sac_hopper_nn2_agent.mojo
"""

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
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep
from mojo_rl.envs.hopper import Hopper, HopperConfig


# =============================================================================
# Architecture (matches the legacy DeepSACAgent hopper training)
# =============================================================================

comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACT_DIM = HopperConfig.ACTION_DIM  #  3
comptime HIDDEN = 256
comptime BATCH = 64
comptime REPLAY_CAPACITY = 100_000

# Training duration. Drop NUM_STEPS to ~20_000 for a smoke run.
comptime NUM_STEPS = 200_000
comptime PRINT_EVERY = 5_000  # driver-cadence verbose + `avg_reward`/`episodes` emit
comptime DIAG_EVERY = 5_000  # `flush_metrics` cadence — full SACMetrics bundle
comptime CHECKPOINT_EVERY = 50_000  # auto-save cadence (env steps)

comptime CHECKPOINT_PATH = "sac_hopper_nn2.ckpt"


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
    print("SAC (deep_agents2) — Hopper CPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("  DIAG_EVERY         =", DIAG_EVERY)
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)
    print("  Checkpoint path    =", CHECKPOINT_PATH)
    print("=" * 70)

    # ─── Logger (remote) ───────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="SAC Hopper NN2 (CPU)",
        buffer_size=200,
        api_key=api_key,
    )
    logger.set_config("algorithm", "SAC")
    logger.set_config("env", "Hopper")
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("batch", String(BATCH))
    logger.set_config("ere", "0.996")

    var logger_ptr = UnsafePointer(to=logger)

    # ─── Agent + env ─────────────────────────────────────────────────────
    var agent = SACAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet,
        CriticNet,
    ](
        actor_lr=3e-4,
        critic_lr=1e-3,  # CleanRL default: q_lr higher than policy_lr
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        init_alpha=0.2,
        target_entropy=-Scalar[DT](ACT_DIM),  # SAC default heuristic (-3)
        learning_starts=1_000,
        window_size=100,
        initial_episode_fill=0.0,
        # ERE — same shape the legacy Hopper recipe uses. Down-weights
        # ancient transitions; helps on long horizons.
        use_ere=True,
        ere_eta=0.996,
    )
    var env = Hopper[DT, TERMINATE_ON_UNHEALTHY=True]()

    # ─── Single train() call — auto-flush + auto-checkpoint ──────────────
    var t_start = perf_counter_ns()
    _ = agent.train_single[
        Hopper[DT, TERMINATE_ON_UNHEALTHY=True],
        L=RemoteLogger,
    ](
        env,
        NUM_STEPS,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
    var total = NUM_STEPS
    logger.close()
    _ = logger  # lifetime extender for logger_ptr

    # ─── Summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training complete")
    print("  total env_steps           =", total)
    print("  elapsed                   =", elapsed_s, "s")
    print("  mean ep return (last 100) =", agent.mean_return())
    print("  episodes completed        =", agent.ep_count())
    print("  remote points sent        =", logger.total_logged())
    print("=" * 70)

    var final_avg = Float64(agent.mean_return())
    if final_avg > 3000.0:
        print("EXCELLENT — hopping fast (mean > 3000).")
    elif final_avg > 1500.0:
        print("STRONG — learned to hop (mean > 1500).")
    elif final_avg > 500.0:
        print("PROGRESS — early locomotion (mean > 500).")
    elif final_avg > 0.0:
        print("LEARNING — positive return (mean > 0).")
    else:
        print("EARLY — still exploring (mean < 0).")
    print("=" * 70)

    # ─── Save/load round-trip smoke test ─────────────────────────────────
    var probe_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    for d in range(OBS_DIM):
        probe_obs[d] = Scalar[DT](0.1 * Float64(d - OBS_DIM // 2))
    var act_before = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_before)

    agent.load(CHECKPOINT_PATH)
    var act_after = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_after)

    print("Save/load round-trip on probe obs:")
    var ok = True
    for j in range(ACT_DIM):
        var diff = Float64(act_after[j] - act_before[j])
        if diff < 0:
            diff = -diff
        print(
            "  dim",
            j,
            " before =",
            act_before[j],
            " after =",
            act_after[j],
            " |diff| =",
            diff,
        )
        if diff > 1e-5:
            ok = False
    if ok:
        print("Round-trip OK (max |diff| < 1e-5 on every action dim).")
    else:
        print("Round-trip MISMATCH — investigate save/load semantics.")
    print("=" * 70)
