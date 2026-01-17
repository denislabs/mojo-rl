"""Test PPO Continuous Agent Hybrid GPU+CPU Training on CarRacing.

This tests the hybrid GPU+CPU implementation of PPO for continuous actions
using the native Mojo CarRacing environment:
- Neural network computations on GPU (forward/backward passes)
- Environment physics on CPU (accurate 2D simulation)
- Gaussian policy for continuous action spaces

Run with:
    pixi run -e apple mojo run tests/test_ppo_car_racing_hybrid.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_car_racing_hybrid.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo_continuous import DeepPPOContinuousAgent
from envs.car_racing import CarRacingEnv


# =============================================================================
# Constants
# =============================================================================

# CarRacing: 13D observation, 3 continuous actions (steering, gas, brake)
comptime OBS_DIM = 13
comptime ACTION_DIM = 3

# Network architecture
comptime HIDDEN_DIM = 256

# Hybrid training parameters
comptime ROLLOUT_LEN = 512  # Longer rollouts for better advantage estimates
comptime N_ENVS = 32  # Fewer envs to fit in memory with longer rollouts
comptime GPU_MINIBATCH_SIZE = 256  # Minibatch size for PPO updates

# Training duration
comptime NUM_EPISODES = 5000  # Reduced for faster iteration


# =============================================================================
# Main
# =============================================================================

comptime dtype = DType.float32


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent Hybrid (GPU+CPU) Test on CarRacing")
    print("=" * 70)
    print()

    # =========================================================================
    # Create CPU environments
    # =========================================================================

    print("Creating " + String(N_ENVS) + " CPU environments...")
    var envs = List[CarRacingEnv[dtype]]()
    for i in range(N_ENVS):
        var env = CarRacingEnv[dtype](
            continuous=True,
            lap_complete_percent=0.95,
            domain_randomize=False,
            max_steps=100,  # Short episodes for faster learning feedback
        )
        envs.append(env^)
    print("Done!")
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        # Initialize action mean biases for CarRacing:
        # - steering: 0 (centered)
        # - gas: 2.0 (tanh(2) ≈ 0.96, maps to gas ≈ 0.98 after remapping)
        # - brake: -2.0 (tanh(-2) ≈ -0.96, maps to brake ≈ 0.02 after remapping)
        # This makes the default policy drive forward with full gas, no brake
        var action_mean_biases = List[Float64]()
        action_mean_biases.append(0.0)   # steering: centered
        action_mean_biases.append(2.0)   # gas: bias toward high
        action_mean_biases.append(-2.0)  # brake: bias toward low

        var agent = DeepPPOContinuousAgent[
            OBS_DIM,
            ACTION_DIM,
            HIDDEN_DIM,
            ROLLOUT_LEN,
            N_ENVS,
            GPU_MINIBATCH_SIZE,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            actor_lr=0.0003,  # Higher LR for faster learning
            critic_lr=0.001,  # Higher LR for value function
            entropy_coef=0.01,  # Higher entropy for exploration
            value_loss_coef=0.5,
            num_epochs=10,  # More epochs per rollout
            # Action scaling for CarRacing
            # Actions are in [-1, 1] after tanh, scale to environment range
            action_scale=1.0,  # steering, gas, brake all in [-1, 1]
            action_bias=0.0,
            # Advanced hyperparameters
            target_kl=0.02,  # Allow some KL divergence for learning
            max_grad_norm=0.5,
            anneal_lr=False,  # Keep LR constant
            anneal_entropy=False,  # Keep entropy constant
            target_total_steps=0,  # Auto-calculate
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=500,
            checkpoint_path="ppo_car_racing_hybrid.ckpt",
            # Per-action mean biases for initialization
            action_mean_biases=action_mean_biases^,
        )

        # Try to load existing checkpoint
        # agent.load_checkpoint("ppo_car_racing_hybrid.ckpt")

        print("Environment: CarRacing (CPU - native Mojo physics)")
        print("Training: Hybrid GPU+CPU")
        print("  Neural network: GPU")
        print("  Physics: CPU (native Mojo 2D physics)")
        print()
        print("Agent: PPO Continuous")
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel CPU): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print()
        print("CarRacing specifics:")
        print("  - 13D observations (car state, sensor readings)")
        print("  - 3D continuous actions: [steering, gas, brake]")
        print("  - Gaussian policy with tanh squashing")
        print("  - Physics: Native Mojo 2D car simulation")
        print()

        # =====================================================================
        # Train using the train_gpu_cpu_env() method (hybrid)
        # =====================================================================

        print("Starting Hybrid GPU+CPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu_cpu_env(
                ctx,
                envs,
                num_episodes=NUM_EPISODES,
                verbose=True,
                print_every=1,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            print("-" * 70)
            print()
            print(">>> train_gpu_cpu_env returned successfully! <<<")

            # =================================================================
            # Summary
            # =================================================================

            print("=" * 70)
            print("Hybrid GPU+CPU Training Complete")
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
            print("Key benefit: Trained policy will transfer correctly to")
            print("evaluation since the same physics are used throughout!")
            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
