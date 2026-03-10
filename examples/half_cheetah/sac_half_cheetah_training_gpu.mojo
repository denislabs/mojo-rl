"""SAC Agent GPU Training on HalfCheetah.

This trains the SAC (Soft Actor-Critic) agent on the HalfCheetah environment
using GPU-accelerated off-policy training with:
- Parallel environments on GPU
- Generalized Coordinates (GC) physics engine (MuJoCo-style)
- 6D continuous action space (joint torques)
- 17D observation (qpos + qvel excluding rootx and head)

SAC key features:
- Maximum entropy RL (reward + alpha * entropy)
- Stochastic Gaussian policy (reparameterization trick)
- Twin Q-networks (min of Q1, Q2 reduces overestimation)
- Automatic entropy temperature (alpha) tuning
- No target actor (only critic targets)

Run with:
    pixi run -e apple mojo run examples/half_cheetah/sac_half_cheetah_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run examples/half_cheetah/sac_half_cheetah_training_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from deep_agents.sac import DeepSACAgent
from envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


# =============================================================================
# Constants
# =============================================================================

# HalfCheetah: 17D observation, 6D continuous action
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# Network architecture
comptime HIDDEN_DIM = 256

# Off-policy GPU training parameters
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 1024
comptime MAX_N_ENVS = 256

# Training duration (off-policy uses steps, not episodes)
comptime NUM_STEPS = 5_000_000
comptime WARMUP_STEPS = 10_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent GPU Training on HalfCheetah")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DeepSACAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.001,  # CleanRL default: q_lr=1e-3 (higher than actor)
            max_n_envs=MAX_N_ENVS,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.001,  # CleanRL uses q_lr for alpha too
            target_entropy=-Float64(ACTION_DIM),
            checkpoint_every=100_000,
            checkpoint_path="sac_half_cheetah.ckpt",
        )

        # agent.load_checkpoint("sac_half_cheetah.ckpt")

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: SAC (Soft Actor-Critic)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Max parallel envs: " + String(MAX_N_ENVS))
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 1e-3 (CleanRL default)")
        print("    - Alpha LR: 1e-3 (CleanRL default)")
        print("    - Tau (soft update): 0.005")
        print("    - Initial alpha: 0.2 (auto-tuned)")
        print("    - Target entropy: -" + String(ACTION_DIM))
        print("    - Warmup steps: " + String(WARMUP_STEPS))
        print()
        print("HalfCheetah specifics:")
        print("  - Generalized Coordinates (GC) physics engine")
        print("  - MuJoCo-style joint-space dynamics")
        print("  - Semi-implicit Euler integration (symplectic)")
        print("  - 8 bodies: torso, 2 legs (thigh+shin+foot), head")
        print("  - 6D continuous actions (joint torques with gear ratios)")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
            ](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                gradient_steps=64,  # Decouple from n_envs (256 would be too slow)
                sync_every=5_000,
                verbose=True,
                print_every=50_000,
                environment_name="HalfCheetah",
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            print("-" * 70)
            print()
            print(">>> train_gpu returned successfully! <<<")

            # =================================================================
            # Summary
            # =================================================================

            print("=" * 70)
            print("GPU Training Complete")
            print("=" * 70)
            print()
            print("Total steps: " + String(NUM_STEPS))
            print("Training time: " + String(elapsed_s)[:6] + " seconds")
            print()

            # Print metrics summary
            print(
                "Final average reward (last 100 episodes): "
                + String(metrics.mean_reward_last_n(100))[:8]
            )
            print("Best episode reward: " + String(metrics.max_reward())[:8])
            print()

            # Check for successful training
            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > 1000.0:
                print("EXCELLENT: Agent is running fast! (avg reward > 1000)")
            elif final_avg > 500.0:
                print("SUCCESS: Agent learned to run! (avg reward > 500)")
            elif final_avg > 100.0:
                print(
                    "GOOD PROGRESS: Agent is learning locomotion"
                    " (avg reward > 100)"
                )
            elif final_avg > 0.0:
                print(
                    "LEARNING: Agent improving but needs more training"
                    " (avg reward > 0)"
                )
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < 0)")

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
