"""Test PPO Continuous Agent Hybrid GPU+CPU Training on LunarLander.

This tests the hybrid GPU+CPU implementation of PPO with continuous actions using the
LunarLanderV2 environment with:
- Neural network computations on GPU (forward/backward passes)
- Environment physics on CPU (accurate Box2D-style simulation)
- 2D continuous action space (main throttle + side control)
- No physics mismatch between training and evaluation

Action space (matching Gymnasium LunarLander-v3 continuous):
- action[0]: main engine throttle (0.0 to 1.0)
- action[1]: side engine control (-1.0 to 1.0)
            negative = left engine, positive = right engine
            magnitude > 0.5 activates the engine

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_continuous_hybrid.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_lunar_continuous_hybrid.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.lunar_lander import LunarLanderV2, LLConstants


# =============================================================================
# Constants
# =============================================================================

# LunarLander: 8D observation, 2D continuous action
comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8: [x, y, vx, vy, angle, ang_vel, left_leg, right_leg]
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2: [main_throttle, side_control]

# Network architecture (larger for LunarLander - more complex than Pendulum)
comptime HIDDEN_DIM = 256

# Hybrid training parameters
comptime ROLLOUT_LEN = 128  # Steps per rollout per environment
comptime N_ENVS = 256  # Parallel CPU environments (smaller than pure GPU version)
comptime GPU_MINIBATCH_SIZE = 512  # Minibatch size for PPO updates

# Training duration
comptime NUM_EPISODES = 50_000  # LunarLander needs many episodes

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent Hybrid (GPU+CPU) Test on LunarLander")
    print("=" * 70)
    print()

    # =========================================================================
    # Create CPU environments
    # =========================================================================

    print("Creating " + String(N_ENVS) + " CPU environments...")
    var envs = List[LunarLanderV2[dtype]]()
    for i in range(N_ENVS):
        var env = LunarLanderV2[dtype](
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
        var agent = DeepPPOContinuousAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=GPU_MINIBATCH_SIZE,
            clip_value=True,
        ](
            gamma=0.99,  # Standard discount
            gae_lambda=0.95,  # Standard GAE lambda
            clip_epsilon=0.2,  # Standard clipping
            actor_lr=0.0003,  # Standard learning rate
            critic_lr=0.001,  # Higher critic LR for faster value learning
            entropy_coef=0.05,  # Higher entropy to prevent mean collapse
            value_loss_coef=0.5,
            num_epochs=10,  # More epochs for LunarLander
            # Advanced hyperparameters
            target_kl=0.1,  # KL early stopping
            max_grad_norm=0.5,
            anneal_lr=False,  # Disabled - causes late-training collapse
            anneal_entropy=False,
            target_total_steps=0,  # Auto-calculate
            norm_adv_per_minibatch=True,
            checkpoint_every=500,
            checkpoint_path="ppo_lunar_continuous_hybrid.ckpt",
            # Reward normalization (CleanRL-style) - prevents fuel penalties from dominating
            normalize_rewards=True,
            # Action scaling: PPO outputs [-1, 1], we need [0, 1] for main and [-1, 1] for side
            # The environment handles this internally via step_continuous_vec
        )

        print("Environment: LunarLander Continuous (CPU - accurate physics)")
        print("Training: Hybrid GPU+CPU")
        print("  Neural network: GPU")
        print("  Physics: CPU (native Mojo 2D physics)")
        print()
        print("Agent: PPO Continuous")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel CPU): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print()
        print("  Advanced features:")
        print("    - LR annealing: disabled (prevents late collapse)")
        print("    - KL early stopping: target_kl=0.1")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print("    - Reward normalization: enabled (CleanRL-style)")
        print("    - Higher entropy: 0.05 (prevents mean collapse)")
        print()
        print("LunarLander Continuous specifics:")
        print(
            "  - 8D observations: [x, y, vx, vy, angle, ang_vel, left_leg,"
            " right_leg]"
        )
        print("  - 2D continuous actions:")
        print("      action[0]: main engine throttle (0.0 to 1.0)")
        print("      action[1]: side engine control (-1.0 to 1.0)")
        print("                 negative = left, positive = right")
        print("  - Reward shaping: distance + velocity + angle penalties")
        print("  - Landing bonus: +100, Crash penalty: -100")
        print("  - Fuel costs: proportional to throttle")
        print("  - Wind effects: enabled")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-200 to -400")
        print("  - Learning policy: > -100")
        print("  - Good policy: > 0")
        print("  - Successful landing: > 100")
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
                + String(metrics.mean_reward_last_n(100))[:8]
            )
            print("Best episode reward: " + String(metrics.max_reward())[:8])
            print()

            # Check for successful training
            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > 100.0:
                print(
                    "EXCELLENT: Agent consistently lands successfully!"
                    " (avg reward > 100)"
                )
            elif final_avg > 0.0:
                print("SUCCESS: Agent learned to land! (avg reward > 0)")
            elif final_avg > -100.0:
                print(
                    "GOOD PROGRESS: Agent is learning to control"
                    " (avg reward > -100)"
                )
            elif final_avg > -200.0:
                print(
                    "LEARNING: Agent improving but needs more training"
                    " (avg reward > -200)"
                )
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < -200)")

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
