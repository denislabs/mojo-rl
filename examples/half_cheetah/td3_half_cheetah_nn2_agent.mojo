"""TD3 training on HalfCheetah (CPU) via the new `TD3Agent` facade.

Sibling of `sac_half_cheetah_nn2_agent.mojo` and
`ddpg_half_cheetah_nn2_agent.mojo`. TD3 adds three tricks on top of
DDPG:

  * Twin critics (the trainer instantiates two copies of `CriticNet`
    internally and takes the min for the TD target).
  * Target-policy smoothing (Gaussian noise on the target action,
    `target_policy_noise` / `target_noise_clip`).
  * Delayed actor updates (`policy_delay`).

Uses the new `deep_agents2/` surface:

  * `TD3Agent[...]` — facade over `TD3Trainer` + the single-env
    off-policy driver.
  * `RemoteLogger` — driver `print_every` cadence + chunk cadence.
  * One-file `.ckpt` envelope holding actor + both critics + Adam
    states.

Metric names (driver cadence): `avg_reward`, `episodes`.
Chunk cadence (`TD3Metrics` fields): `actor_loss`, `critic_loss`,
  `train_steps`, `n_actor_updates`, `n_critic_updates` (actor delayed).

HalfCheetah (Physics3dEnv, MuJoCo-style):
  * 17D observation (qpos + qvel excluding rootx and head)
  * 6D continuous action (joint torques)
  * Reward ≈ forward velocity - 0.1·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

Run:
    pixi run mojo run -I . examples/half_cheetah/td3_half_cheetah_nn2_agent.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.deep_agents2.td3 import TD3Agent
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 64
comptime REPLAY_CAPACITY = 100_000

comptime NUM_STEPS = 100_000
comptime PRINT_EVERY = 5_000
comptime DIAG_EVERY = 5_000
comptime CHECKPOINT_EVERY = 50_000

comptime CHECKPOINT_PATH = "td3_half_cheetah_nn2.ckpt"


# TD3 actor is deterministic (Tanh-bounded). Twin critics live inside
# `TD3Trainer` — pass a single `CriticNet`, the trainer makes two.
comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, ACT_DIM],
    Tanh[ACT_DIM],
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
    print("TD3 (deep_agents2) — HalfCheetah CPU + checkpoints + logger")
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
        run_name="TD3 HalfCheetah NN2 (CPU)",
        buffer_size=200,
        api_key=api_key,
    )
    logger.set_config("algorithm", "TD3")
    logger.set_config("env", "HalfCheetah")
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("batch", String(BATCH))
    logger.set_config("policy_delay", "2")

    var logger_ptr = UnsafePointer(to=logger)

    # ─── Agent + env ─────────────────────────────────────────────────────
    var agent = TD3Agent[
        ActorNet,
        CriticNet,
        OBS_DIM,
        ACT_DIM,
        BATCH,
        REPLAY_CAPACITY,
    ](
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        exploration_noise=0.1,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_delay=2,
        learning_starts=1_000,
        window_size=100,
        initial_episode_fill=0.0,
    )
    var env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()

    # ─── Single train() call — auto-flush + auto-checkpoint ──────────────
    var t_start = perf_counter_ns()
    _ = agent.train_single[
        HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False],
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
    print("  total env_steps        =", total)
    print("  elapsed                =", elapsed_s, "s")
    print("  mean ep return (last 100) =", agent.mean_return())
    print("  episodes completed     =", agent.ep_count())
    print("  remote points sent     =", logger.total_logged())
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
