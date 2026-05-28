"""PPO training on HalfCheetah (CPU) via the new `PPOAgent` facade.

Sibling of `sac_half_cheetah_nn2_agent.mojo` for the on-policy
Gaussian-policy PPO flavour. Uses the new `deep_agents2/` surface:

  * `PPOAgent[...]` — facade over `PPOTrainer` + the single-env
    on-policy driver. PPO is on-policy → no replay buffer, no
    `SAMPLE` block, no `learning_starts`. Every `ROLLOUT_LEN` env
    steps the driver computes GAE and runs `N_EPOCHS` passes of
    minibatch SGD (`MINIBATCH` rows each).
  * `RemoteLogger` — driver `print_every` cadence (`avg_reward` /
    `episodes`) + chunk cadence (`PPOMetrics`).
  * One-file `.ckpt` envelope holding actor (incl. Gaussian
    log-std parameter) + critic + Adam states.

Metric names (driver cadence): `avg_reward`, `episodes`.
Chunk cadence (`PPOMetrics` fields): `actor_loss`, `critic_loss`,
  `train_steps`, `n_updates`.

HalfCheetah (Physics3dEnv, MuJoCo-style):
  * 17D observation (qpos + qvel excluding rootx and head)
  * 6D continuous action (joint torques)
  * Reward ≈ forward velocity - 0.1·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

Run:
    pixi run mojo run -I . examples/half_cheetah/ppo_half_cheetah_nn2_agent.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.deep_agents2.primitives.gaussian_head import GaussianHead
from mojo_rl.deep_agents2.ppo import PPOAgent
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture (CleanRL-style continuous PPO: Tanh MLP + GaussianHead)
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime ROLLOUT_LEN = 2_048
comptime MINIBATCH = 64
comptime N_EPOCHS = 10

# Training duration. Match the legacy-script scale; drop NUM_STEPS to
# ~20k for a smoke run. NOTE: PPO env-step count is independent of
# ROLLOUT_LEN; the driver groups them into chunks internally.
comptime NUM_STEPS = 1_000_000
comptime PRINT_EVERY = 50_000
comptime DIAG_EVERY = 50_000
comptime CHECKPOINT_EVERY = 50_000

comptime CHECKPOINT_PATH = "ppo_half_cheetah_nn2.ckpt"

comptime LOG_STD_INIT: Scalar[DT] = -0.5
comptime MAX_TORQUE: Scalar[DT] = 1.0  # HalfCheetah torque range

comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT_DIM],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO (deep_agents2) — HalfCheetah CPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  ROLLOUT_LEN        =", ROLLOUT_LEN)
    print("  MINIBATCH          =", MINIBATCH)
    print("  N_EPOCHS           =", N_EPOCHS)
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
        run_name="PPO HalfCheetah NN2 (CPU)",
        buffer_size=200,
        api_key=api_key,
    )
    logger.set_config("algorithm", "PPO")
    logger.set_config("env", "HalfCheetah")
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("rollout_len", String(ROLLOUT_LEN))
    logger.set_config("minibatch", String(MINIBATCH))
    logger.set_config("n_epochs", String(N_EPOCHS))

    var logger_ptr = UnsafePointer(to=logger)

    # ─── Agent + env ─────────────────────────────────────────────────────
    var agent = PPOAgent[
        "cpu",
        ActorNet,
        CriticNet,
        OBS_DIM,
        ACT_DIM,
        ROLLOUT_LEN,
        MINIBATCH,
        N_EPOCHS,
    ](
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.0,
        action_scale=MAX_TORQUE,
        log_std_init=LOG_STD_INIT,
        # Canonical PPO grad-norm clip (Schulman 2017). Closed the
        # nn2-vs-bespoke gap on Pendulum 200k (-230 → -135 at seed=42).
        max_grad_norm=0.5,
    )

    # CleanRL-style log_std init — the trainer leaves this to the caller
    # because Mojo nightly can't reflect into Sequential's variadic
    # children generically. Index 4 = `GaussianHead` (children: Linear,
    # Tanh, Linear, Tanh, GaussianHead).
    var ls_ptr = agent.trainer.actor.children[4].log_std.value_unsafe_ptr_cpu()
    for k in range(ACT_DIM):
        ls_ptr[k] = LOG_STD_INIT

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
