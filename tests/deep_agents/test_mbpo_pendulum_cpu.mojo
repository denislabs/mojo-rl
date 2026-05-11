"""Vanilla MBPO CPU on Pendulum — Phase-3 control (Phase A0).

All 12 existing MBPO examples in `examples/` are GPU. This is the first
test of the CPU `train()` path on a small env (Pendulum). It also serves
as the **vanilla-MBPO control** for the upcoming PCN-MBPO comparison —
same SAC hyperparameters, same env, NUM_ENSEMBLE=3 (smaller than the
default 7) for laptop tractability.

Goal: solve Pendulum in <30 min wall, beat the Phase-2 SAC baseline
(`test_pendulum_sac_baseline.mojo` last-20 avg ≈ −181) on sample
efficiency. If MBPO CPU can't reach that bar, fix the CPU path before
touching PCN.

Run:
    pixi run mojo run -I . tests/deep_agents/test_mbpo_pendulum_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.envs import PendulumEnv
from mojo_rl.deep_agents import MBPOAgent
from mojo_rl.deep_agents.core.configs.mbpo_config import DefaultMBPOConfig
from mojo_rl.deep_agents.core.strategies.termination import NeverTerminate


# Pendulum: 3D obs, 1D continuous action ∈ [-2, 2].
comptime OBS_DIM = 3
comptime ACTION_DIM = 1

# SAC architecture — match Phase-2 baseline (`test_pendulum_sac_baseline.mojo`).
comptime HIDDEN_DIM = 64
comptime BATCH_SIZE = 64

# Buffers.
comptime BUFFER_CAPACITY = 50_000
comptime SYNTH_CAPACITY = 200_000  # MBPO synthetic buffer

# Dynamics ensemble — 3 instead of 7 for CPU tractability.
comptime NUM_ENSEMBLE = 3
comptime NUM_ELITES = 2
comptime DYN_HIDDEN = 200

# Training duration. Pendulum SAC reaches ≈ -181 in 200 episodes × 200 steps =
# 40K env steps. MBPO claims ~10× sample efficiency, so 4-5K env steps target.
comptime NUM_EPOCHS = 5
comptime STEPS_PER_EPOCH = 1000
comptime WARMUP_STEPS = 1000
comptime MAX_STEPS_PER_EPISODE = 200

# Pendulum needs no early termination.
comptime TermFn = NeverTerminate

comptime PendulumMBPOConfig = DefaultMBPOConfig[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    SYNTH_CAPACITY,
    BATCH_SIZE,
    NUM_ENSEMBLE,
    NUM_ELITES,
    DYN_HIDDEN,
    0.0003,  # actor_lr
    0.0003,  # critic_lr
    0.001,   # model_lr
    TermFn,
    2.0,     # action_scale (Pendulum torque ∈ [-2, 2])
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("Vanilla MBPO CPU on Pendulum — Phase-3 control")
    print("=" * 70)
    print("  Env        : PendulumEnv (3D obs, 1D action ∈ [-2, 2])")
    print("  SAC arch   : hidden=", HIDDEN_DIM, " batch=", BATCH_SIZE)
    print("  Ensemble   :", NUM_ENSEMBLE, " nets,", NUM_ELITES, " elites,")
    print("                hidden=", DYN_HIDDEN)
    print("  Buffers    : real=", BUFFER_CAPACITY, " synth=", SYNTH_CAPACITY)
    print("  Training   :", NUM_EPOCHS, " epochs ×", STEPS_PER_EPOCH, " steps")
    print("  Warmup     :", WARMUP_STEPS, " env steps")
    print()

    var env = PendulumEnv[DType.float64]()

    # SAC hyperparameters — same as `test_pendulum_sac_baseline.mojo`.
    var agent = MBPOAgent[PendulumMBPOConfig](
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        alpha=0.1,
        auto_alpha=True,
        alpha_lr=0.0001,
        target_entropy=-Float64(ACTION_DIM),  # = -1.0
        # MBPO-specific:
        model_train_freq=250,
        rollout_min_length=1,
        rollout_max_length=1,  # Pendulum: 1-step rollouts (paper default for non-locomotion)
        rollout_min_epoch=0,
        rollout_max_epoch=NUM_EPOCHS,
        num_rollouts_per_step=1000,
        real_ratio=0.05,
        sac_updates_per_step=20,
    )

    print("Starting CPU training...")
    print("-" * 70)
    var t0 = perf_counter_ns()

    var metrics = agent.train(
        env,
        num_epochs=NUM_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        warmup_steps=WARMUP_STEPS,
        eval_episodes=5,
        eval_every=1,
        verbose=True,
        print_every=1,
        environment_name="Pendulum (vanilla-MBPO CPU)",
    )

    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    print("-" * 70)
    print()

    # Per-episode CSV for comparison vs Phase-2 SAC baseline.
    print("=== per-episode returns (CSV: ep,return,steps) ===")
    var rewards = metrics.get_rewards()
    var steps = metrics.get_steps()
    for i in range(len(rewards)):
        print("  CSV:", i, ",", rewards[i], ",", steps[i])

    print()
    print("=== Vanilla MBPO Pendulum CPU summary ===")
    print("  Total env steps    :", NUM_EPOCHS * STEPS_PER_EPOCH)
    print("  Wall time          :", elapsed, "s")
    print("  Final α            :", String(agent.alpha)[byte=:6])
    print("  Last-20 avg return :", metrics.mean_reward_last_n(20))
    print("  Mean reward (all)  :", metrics.mean_reward())
    print("  Max episode reward :", metrics.max_reward())
    print("=== Done ===")
