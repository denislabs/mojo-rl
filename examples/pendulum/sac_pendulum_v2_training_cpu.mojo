"""SAC Agent CPU Training on Pendulum V2.

This trains the SAC (Soft Actor-Critic) agent on the GPU-aware Pendulum V2
environment, but driven through its CPU single-env path. Hyperparameters
match sac_pendulum_v1_training_cpu.mojo and sac_pendulum_v2_training_gpu.mojo
so the three runs can be compared apples-to-apples.

Pendulum:
- 3D observation: [cos(theta), sin(theta), theta_dot]
- 1D continuous action: torque in [-2, 2]
- Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
- Episode length: 200 (no early termination)

V1 vs V2 physics use the same equations and target gymnasium-v1; tiny numerical
differences are expected (clamp ordering, post-step vs pre-step reward).

Run with:
    pixi run mojo run -I . examples/pendulum/sac_pendulum_v2_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.pendulum import PendulumV2


# =============================================================================
# Constants (shared across V1/V2 CPU/GPU SAC pendulum scripts)
# =============================================================================

comptime OBS_DIM = 3
comptime ACTION_DIM = 1

comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 32

comptime NUM_STEPS = 100_000
comptime MAX_STEPS_PER_EPISODE = 200
comptime WARMUP_STEPS = 2_000

comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent CPU Training on Pendulum V2")
    print("=" * 70)
    print()

    var env = PendulumV2[dtype]()

    var agent = DeepSACAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        actor_lr=0.0003,
        critic_lr=0.0003,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        alpha=0.1,
        auto_alpha=True,
        alpha_lr=0.0003,
        target_entropy=-1.0,
        use_ere=True,
        ere_eta=0.996,
    )

    print("Environment: Pendulum V2 (CPU single-env path)")
    print("Agent: SAC (Soft Actor-Critic)")
    print("  Observation dim: " + String(OBS_DIM))
    print("  Action dim: " + String(ACTION_DIM))
    print("  Hidden dim: " + String(HIDDEN_DIM))
    print("  Buffer capacity: " + String(BUFFER_CAPACITY))
    print("  Batch size: " + String(BATCH_SIZE))
    print("  Total steps: " + String(NUM_STEPS))
    print("  Max steps/episode: " + String(MAX_STEPS_PER_EPISODE))
    print("  Warmup steps: " + String(WARMUP_STEPS))
    print()

    print("Starting CPU training...")
    print("-" * 70)

    var start_time = perf_counter_ns()

    var metrics = agent.train(
        env,
        num_steps=NUM_STEPS,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        warmup_steps=WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=4_000,
        environment_name="PendulumV2",
    )

    var end_time = perf_counter_ns()
    var elapsed_s = Float64(end_time - start_time) / 1e9

    print("-" * 70)
    print()
    print("=" * 70)
    print("Pendulum V2 CPU Training Complete")
    print("=" * 70)
    print()
    print("Total steps: " + String(NUM_STEPS))
    print("Training time: " + String(elapsed_s)[byte=:6] + " seconds")
    print()

    print(
        "Final average reward (last 100 episodes): "
        + String(metrics.mean_reward_last_n(100))[byte=:8]
    )
    print("Best episode reward: " + String(metrics.max_reward())[byte=:8])
    print()

    var final_avg = metrics.mean_reward_last_n(100)
    if final_avg > -200.0:
        print("EXCELLENT: Agent solved swing-up! (avg reward > -200)")
    elif final_avg > -500.0:
        print("SUCCESS: Agent largely swings up (avg reward > -500)")
    elif final_avg > -1000.0:
        print("GOOD PROGRESS: Agent is learning (avg reward > -1000)")
    else:
        print("EARLY STAGE: Agent still exploring (avg reward < -1000)")

    print()
    print("=" * 70)
