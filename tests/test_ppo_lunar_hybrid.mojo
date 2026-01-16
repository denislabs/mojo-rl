"""Test PPO Agent Hybrid GPU+CPU Training on LunarLander.

This tests the hybrid GPU+CPU implementation of PPO using the native Mojo
LunarLander environment with accurate 2D physics:
- Neural network computations on GPU (forward/backward passes)
- Environment physics on CPU (accurate Box2D-style simulation)
- No physics mismatch between training and evaluation

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_hybrid.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_lunar_hybrid.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOAgent
from envs.lunar_lander import LunarLanderEnv


# =============================================================================
# Constants
# =============================================================================

# LunarLander: 8D observation, 4 discrete actions
comptime OBS_DIM = 8
comptime NUM_ACTIONS = 4

# Network architecture
comptime HIDDEN_DIM = 300

# Hybrid training parameters
comptime ROLLOUT_LEN = 128  # Steps per rollout per environment
comptime N_ENVS = 256  # Parallel CPU environments (smaller than GPU version)
comptime GPU_MINIBATCH_SIZE = 512  # Minibatch size for PPO updates

# Training duration (fewer episodes since CPU physics is slower)
comptime NUM_EPISODES = 50_000


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Agent Hybrid (GPU+CPU) Test on LunarLander")
    print("=" * 70)
    print()

    # =========================================================================
    # Create CPU environments
    # =========================================================================

    print("Creating " + String(N_ENVS) + " CPU environments...")
    var envs = List[LunarLanderEnv]()
    for i in range(N_ENVS):
        var env = LunarLanderEnv(
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=15.0,
            turbulence_power=1.5,
        )
        envs.append(env^)
    print("Done!")
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DeepPPOAgent[
            OBS_DIM,
            NUM_ACTIONS,
            HIDDEN_DIM,
            ROLLOUT_LEN,
            N_ENVS,
            GPU_MINIBATCH_SIZE,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            actor_lr=0.0003,
            critic_lr=0.0003,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=10,
            # Advanced hyperparameters
            target_kl=0.02,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,  # Auto-calculate
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=500,
            checkpoint_path="ppo_lunar_hybrid.ckpt",
        )

        agent.load_checkpoint("ppo_lunar_hybrid.ckpt")

        print("Environment: LunarLander (CPU - accurate physics)")
        print("Training: Hybrid GPU+CPU")
        print("  Neural network: GPU")
        print("  Physics: CPU (native Mojo 2D physics)")
        print()
        print("Agent: PPO")
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel CPU): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print()
        print("LunarLander specifics:")
        print(
            "  - 8D observations: [x, y, vx, vy, angle, ang_vel, left_leg,"
            " right_leg]"
        )
        print("  - 4 actions: nop, left, main, right")
        print("  - Wind effects: enabled")
        print("  - Physics: Accurate 2D rigid body simulation")
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
            if final_avg > 0.0:
                print("SUCCESS: Agent learned to land! (avg reward > 0)")
            elif final_avg > -100.0:
                print(
                    "PARTIAL: Agent is learning but not consistently landing"
                    " (avg reward > -100)"
                )
            else:
                print(
                    "NEEDS MORE TRAINING: Agent still crashing frequently (avg"
                    " reward < -100)"
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
