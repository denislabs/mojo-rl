"""Test PPO Agent GPU Training on Pendulum.

This tests the GPU implementation of PPO using the PendulumV2 environment with:
- Parallel environments on GPU
- Simple pendulum physics (swing-up task)
- Continuous action space (torque)

Pendulum is a great test case because:
- Simple physics (2 state variables, 1 action)
- Fixed episode length (200 steps, no early termination)
- Well-defined reward function: -(θ² + 0.1*θ_dot² + 0.001*u²)
- Optimal policy is to swing up and balance at θ=0

Run with:
    pixi run -e apple mojo run tests/test_ppo_pendulum_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_pendulum_gpu.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo_continuous import DeepPPOContinuousAgent
from envs.pendulum import PendulumV2, PConstants


# =============================================================================
# Constants
# =============================================================================

# PendulumV2: 3D observation, 1 continuous action
comptime OBS_DIM = PConstants.OBS_DIM  # 3: [cos(θ), sin(θ), θ_dot]
comptime NUM_ACTIONS = PConstants.ACTION_DIM  # 1: torque in [-2, 2]

# Network architecture (smaller since Pendulum is simpler)
comptime HIDDEN_DIM = 64

# GPU training parameters
comptime ROLLOUT_LEN = 200  # One full episode per rollout
comptime N_ENVS = 512  # Parallel environments
comptime GPU_MINIBATCH_SIZE = 256  # Minibatch size for PPO updates

# Training duration
comptime NUM_EPISODES = 5_000  # More episodes for stable training

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Agent GPU Test on Pendulum")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DeepPPOContinuousAgent[
            OBS_DIM,
            NUM_ACTIONS,
            HIDDEN_DIM,
            ROLLOUT_LEN,
            N_ENVS,
            GPU_MINIBATCH_SIZE,
        ](
            gamma=0.99,  # Standard discount
            gae_lambda=0.95,  # Standard GAE lambda
            clip_epsilon=0.2,  # Standard clipping
            actor_lr=0.0003,  # Standard learning rate
            critic_lr=0.001,  # Higher critic LR
            entropy_coef=0.01,  # Small entropy for exploration
            value_loss_coef=0.5,
            num_epochs=4,  # Standard epochs
            # Advanced hyperparameters
            target_kl=0.0,  # Disable KL early stopping (let all epochs run)
            max_grad_norm=0.5,
            anneal_lr=True,  # Enable LR annealing
            anneal_entropy=False,
            target_total_steps=0,
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_pendulum_gpu.ckpt",
            # Note: action_scale not needed - PendulumV2 handles scaling internally
            # PPO outputs [-1, 1], environment scales to [-2, 2]
        )

        print("Environment: PendulumV2 (GPU)")
        print("Agent: PPO Continuous (GPU)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(NUM_ACTIONS))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print("  Advanced features:")
        print("    - LR annealing: enabled")
        print("    - KL early stopping: target_kl=0.05")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print()
        print("Pendulum specifics:")
        print("  - 3D observations: [cos(θ), sin(θ), θ_dot]")
        print("  - 1D action: torque in [-2, 2]")
        print("  - Reward: -(θ² + 0.1*θ_dot² + 0.001*u²)")
        print("  - Episode length: 200 steps (fixed)")
        print("  - Goal: Swing up and balance at θ=0 (pointing up)")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-1200 to -1600")
        print("  - Good policy: > -200")
        print("  - Optimal: ~0 (balanced at top)")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[PendulumV2[dtype]](
                ctx,
                num_episodes=NUM_EPISODES,
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
                + String(metrics.mean_reward_last_n(100))[:8]
            )
            print("Best episode reward: " + String(metrics.max_reward())[:8])
            print()

            # Check for successful training
            # Pendulum rewards are negative, closer to 0 is better
            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > -200.0:
                print(
                    "SUCCESS: Agent learned to swing up and balance!"
                    " (avg reward > -200)"
                )
            elif final_avg > -500.0:
                print(
                    "GOOD PROGRESS: Agent is learning swing-up"
                    " (avg reward > -500)"
                )
            elif final_avg > -1000.0:
                print(
                    "LEARNING: Agent is improving but needs more training"
                    " (avg reward > -1000)"
                )
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < -1000)")

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
