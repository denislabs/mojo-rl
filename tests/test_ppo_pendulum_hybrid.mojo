"""Test PPO Continuous Agent Hybrid GPU+CPU Training on Pendulum.

This tests the hybrid GPU+CPU implementation of PPO for continuous actions
using the native Mojo Pendulum environment:
- Neural network computations on GPU (forward/backward passes)
- Environment physics on CPU (accurate simulation)
- Gaussian policy for continuous action spaces

Pendulum is a simpler continuous control task than CarRacing:
- 3D observations: [cos(θ), sin(θ), θ_dot]
- 1D continuous action: torque in [-2, 2]
- Max 200 steps per episode

Run with:
    pixi run -e apple mojo run tests/test_ppo_pendulum_hybrid.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_pendulum_hybrid.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo_continuous import DeepPPOContinuousAgent
from envs.pendulum import PendulumEnv


# =============================================================================
# Constants
# =============================================================================

# Pendulum: 3D observation, 1 continuous action (torque)
comptime OBS_DIM = 3
comptime ACTION_DIM = 1

# Network architecture
comptime HIDDEN_DIM = 64  # Smaller network for simpler task

# Hybrid training parameters
comptime ROLLOUT_LEN = 200  # Steps per rollout per environment (one episode)
comptime N_ENVS = 32  # Parallel CPU environments
comptime GPU_MINIBATCH_SIZE = 128  # Minibatch size for PPO updates

# Training duration
comptime NUM_EPISODES = 2000


# =============================================================================
# Main
# =============================================================================

comptime dtype = DType.float32


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent Hybrid (GPU+CPU) Test on Pendulum")
    print("=" * 70)
    print()

    # =========================================================================
    # Create CPU environments
    # =========================================================================

    print("Creating " + String(N_ENVS) + " CPU environments...")
    var envs = List[PendulumEnv[dtype]]()
    for i in range(N_ENVS):
        var env = PendulumEnv[dtype]()
        envs.append(env^)
    print("Done!")
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
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
            actor_lr=0.0001,  # Lower LR for stability
            critic_lr=0.0003,  # Lower LR for stability
            entropy_coef=0.005,  # Lower entropy (was too high causing instability)
            value_loss_coef=0.5,
            num_epochs=4,  # Fewer epochs but more stable
            # Action scaling for Pendulum
            # Actions are in [-1, 1] after tanh, scale to [-2, 2]
            action_scale=2.0,  # Torque range
            action_bias=0.0,
            # Advanced hyperparameters
            target_kl=0.015,  # Stricter KL threshold like reference implementations
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,  # Auto-calculate
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=100,
            checkpoint_path="ppo_pendulum_hybrid.ckpt",
        )

        print("Environment: Pendulum (CPU - native Mojo physics)")
        print("Training: Hybrid GPU+CPU")
        print("  Neural network: GPU")
        print("  Physics: CPU (native Mojo pendulum simulation)")
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
        print("Pendulum specifics:")
        print("  - 3D observations: [cos(θ), sin(θ), θ_dot]")
        print("  - 1D continuous action: torque in [-2, 2]")
        print("  - Gaussian policy with tanh squashing")
        print("  - Goal: Swing up and balance")
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
            # Pendulum rewards are always negative (optimal is ~-0.1 per step)
            # Total reward for 200 steps: optimal ~-200, random ~-1600
            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > -200.0:
                print(
                    "SUCCESS: Agent learned to balance! (avg reward > -200)"
                )
            elif final_avg > -500.0:
                print(
                    "GOOD: Agent is learning well (avg reward > -500)"
                )
            elif final_avg > -1000.0:
                print(
                    "PARTIAL: Agent is learning but needs more training"
                    " (avg reward > -1000)"
                )
            else:
                print(
                    "NEEDS MORE TRAINING: Agent still struggling (avg"
                    " reward < -1000)"
                )

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
