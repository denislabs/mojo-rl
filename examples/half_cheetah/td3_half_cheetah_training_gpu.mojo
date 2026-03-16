"""TD3 Agent GPU Training on HalfCheetah.

This trains the TD3 (Twin Delayed DDPG) agent on the HalfCheetah environment
using GPU-accelerated off-policy training with:
- Parallel environments on GPU
- Generalized Coordinates (GC) physics engine (MuJoCo-style)
- 6D continuous action space (joint torques)
- 17D observation (qpos + qvel excluding rootx and head)

TD3 key features:
- Twin Q-networks (min of Q1, Q2 reduces overestimation)
- Delayed policy updates (actor updated every 2 critic updates)
- Target policy smoothing (clipped noise on target actions)

Run with:
    pixi run -e apple mojo run -I . examples/half_cheetah/td3_half_cheetah_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/half_cheetah/td3_half_cheetah_training_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.generic import DeepTD3Agent
from mojo_rl.envs.half_cheetah import (
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
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 64

# Training duration (total env transitions across all parallel envs)
comptime NUM_STEPS = 3_000_000
comptime WARMUP_STEPS = 25_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("TD3 Agent GPU Training on HalfCheetah")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DeepTD3Agent[
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
            action_scale=1.0,
            noise_std=0.1,
            noise_std_min=0.01,
            noise_decay=0.995,
            policy_delay=2,
            target_noise_std=0.2,
            target_noise_clip=0.5,
            checkpoint_every=500_000,
            checkpoint_path="td3_half_cheetah.ckpt",
        )

        # agent.load_checkpoint("td3_half_cheetah.ckpt")

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: TD3 (Twin Delayed DDPG)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Max parallel envs: " + String(MAX_N_ENVS))
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 3e-4")
        print("    - Tau (soft update): 0.005")
        print("    - Exploration noise: 0.1 (decaying)")
        print("    - Policy delay: 2")
        print("    - Target noise: 0.2 (clip 0.5)")
        print("    - Warmup transitions: " + String(WARMUP_STEPS))
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
                # gradient_steps=0 uses n_envs (1:1 replay ratio)
                sync_every=5_000,
                verbose=True,
                print_every=50_000,
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
