"""Test PPO Agent GPU Training on CarRacing.

This tests the GPU implementation of PPO using the GPU-compatible
CarRacing environment with:
- Parallel environments on GPU
- 2D car physics simulation
- Track generation and collision detection
- Continuous action space (steering, gas, brake)

Run with:
    pixi run -e apple mojo run -I . tests/test_ppo_car_racing_gpu_v2.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . tests/test_ppo_car_racing_gpu_v2.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.generic import DeepPPOContinuousAgent
from mojo_rl.envs.car_racing import CarRacing


# =============================================================================
# Constants
# =============================================================================

# CarRacing: 13D observation, 3 continuous actions
comptime OBS_DIM = 13
comptime NUM_ACTIONS = 3

# Network architecture
comptime HIDDEN_DIM = 300

# GPU training parameters
comptime ROLLOUT_LEN = 512  # Steps per rollout per environment
comptime N_ENVS = 512  # Parallel environments
comptime GPU_MINIBATCH_SIZE = 512  # Minibatch size for PPO updates

# Training duration
comptime NUM_EPISODES = 10_000  # Many episodes for CarRacing (complex environment)

comptime dtype = DType.float32

# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Agent GPU Test on CarRacing")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DeepPPOContinuousAgent[
            obs_dim=OBS_DIM,
            action_dim=NUM_ACTIONS,
            hidden_dim=HIDDEN_DIM,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=GPU_MINIBATCH_SIZE,
            actor_lr=0.00005,  # Moderate LR
            critic_lr=0.0005,  # Moderate for value learning
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,  # Standard clipping
            entropy_coef=0.01,  # Standard entropy
            value_loss_coef=0.5,
            num_epochs=4,  # Fewer epochs to prevent overfitting on rollout
            # Advanced hyperparameters
            target_kl=0.0,  # Disabled - mean clamping prevents worst saturation
            max_grad_norm=0.5,
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_car_racing_gpu_v2.ckpt",
        )

        # agent.load_checkpoint("ppo_car_racing_gpu_v2.ckpt")

        print("Environment: CarRacing (GPU)")
        print("Agent: PPO (GPU)")
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print("  Advanced features:")
        print("    - LR annealing: enabled")
        print("    - KL early stopping: DISABLED (debugging)")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print()
        print("CarRacing specifics:")
        print(
            "  - 13D observations: [x, y, angle, vx, vy, ang_vel,"
            " wheel_angles(2), wheel_omegas(4), speed]"
        )
        print("  - 3 actions: steering, gas, brake")
        print("  - Action biases: gas=0.5, brake=-0.5 (moderate forward bias)")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            # Use specialized CarRacing training with full track support
            var metrics = agent.train_gpu[CarRacing[dtype]](
                ctx,
                num_updates=NUM_EPISODES,
                verbose=True,
                print_every=1,
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
            print("Total episodes: " + String(NUM_EPISODES))
            print("Training time: " + String(elapsed_s)[:6] + " seconds")
            print(
                "Episodes/second: "
                + String(Float64(NUM_EPISODES) / elapsed_s)[:7]
            )
            print()

            # Print metrics summary
            print(
                "Final average reward (last 100 episodes): "
                + String(metrics.mean_reward_last_n(100))[:7]
            )
            print("Best episode reward: " + String(metrics.max_reward())[:7])
            print()

            # Check for successful training
            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > 500.0:
                print(
                    "SUCCESS: Agent learned to drive well! (avg reward > 500)"
                )
            elif final_avg > 200.0:
                print(
                    "PARTIAL: Agent is learning but needs more training"
                    " (avg reward > 200)"
                )
            else:
                print(
                    "NEEDS MORE TRAINING: Agent still struggling (avg"
                    " reward < 200)"
                )

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
