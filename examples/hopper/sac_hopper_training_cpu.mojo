"""SAC Agent CPU Training on Hopper.

This trains the SAC (Soft Actor-Critic) agent on the Hopper environment
using single-env CPU training with:
- Generalized Coordinates (GC) physics engine (MuJoCo-style)
- 3D continuous action space (joint torques)
- 11D observation (qpos + qvel excluding rootx)

SAC key features:
- Maximum entropy RL (reward + alpha * entropy)
- Stochastic Gaussian policy (reparameterization trick)
- Twin Q-networks (min of Q1, Q2 reduces overestimation)
- Automatic entropy temperature (alpha) tuning
- No target actor (only critic targets)

Run with:
    pixi run mojo run -I . examples/hopper/sac_hopper_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.hopper import Hopper, HopperConfig


# =============================================================================
# Constants
# =============================================================================

# Hopper: 11D observation, 3D continuous action
comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3

# Network architecture
comptime HIDDEN_DIM = 256

# Off-policy CPU training parameters
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 64

# Training duration
comptime NUM_STEPS = 1_000_000
comptime MAX_STEPS_PER_EPISODE = 1000
comptime WARMUP_STEPS = 25_000

comptime dtype = DType.float64


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent CPU Training on Hopper")
    print("=" * 70)
    print()

    # =========================================================================
    # Create environment and agent
    # =========================================================================

    var env = Hopper[dtype, TERMINATE_ON_UNHEALTHY=True]()

    var agent = DeepSACAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        actor_lr=0.0003,
        critic_lr=0.001,
        L=RemoteLogger,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=0.0003,
        target_entropy=-3.0,  # -ACTION_DIM
        checkpoint_every=100_000,
        checkpoint_path="sac_hopper_cpu.ckpt",
        use_ere=True,
        ere_eta=0.996,
    )

    print("Environment: Hopper Continuous (CPU)")
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

    # =========================================================================
    # Setup logger
    # =========================================================================

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="SAC Hopper CPU",
        buffer_size=64,
        api_key=api_key,
    )
    logger.set_config("agent", "SAC")
    logger.set_config("env", "Hopper")
    logger.set_config("device", "CPU")
    logger.set_config("hidden_dim", String(HIDDEN_DIM))
    logger.set_config("actor_lr", "3e-4")
    logger.set_config("critic_lr", "1e-3")
    logger.set_config("alpha_lr", "3e-4")
    logger.set_config("batch_size", String(BATCH_SIZE))
    logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))

    # =========================================================================
    # Train
    # =========================================================================

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
        print_every=50_000,
        environment_name="Hopper",
        logger=UnsafePointer(to=logger),
        diag_every=5_000,
    )

    var end_time = perf_counter_ns()
    var elapsed_s = Float64(end_time - start_time) / 1e9

    logger.close()

    # =========================================================================
    # Summary
    # =========================================================================

    print("-" * 70)
    print()
    print("=" * 70)
    print("CPU Training Complete")
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
    if final_avg > 3000.0:
        print("EXCELLENT: Agent is hopping fast! (avg reward > 3000)")
    elif final_avg > 1500.0:
        print("SUCCESS: Agent learned to hop! (avg reward > 1500)")
    elif final_avg > 500.0:
        print("GOOD PROGRESS: Agent is learning locomotion (avg reward > 500)")
    elif final_avg > 0.0:
        print(
            "LEARNING: Agent improving but needs more training (avg reward > 0)"
        )
    else:
        print("EARLY STAGE: Agent still exploring (avg reward < 0)")

    print()
    print("=" * 70)
