"""CPU evaluation with Gymnasium MuJoCo wrapper for continuous PPO on HalfCheetah.

This tests the trained continuous PPO model using the Gymnasium HalfCheetah-v5
wrapper instead of the native Mojo physics environment.

Requires:
    pip install "gymnasium[mujoco]" mujoco

Run with:
    pixi run mojo run tests/test_ppo_half_cheetah_gymnasium_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.gymnasium.gymnasium_mujoco import GymMuJoCoEnv, make_half_cheetah

# =============================================================================
# Constants (must match training configuration)
# =============================================================================

# HalfCheetah-v5: obs_dim=17, action_dim=6
comptime OBS_DIM = 17
comptime ACTION_DIM = 6
# Must match training configuration!
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000  # HalfCheetah episodes run for 1000 steps
comptime RENDER = True  # Set to False for headless evaluation


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent CPU Evaluation - Gymnasium HalfCheetah-v5")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent (must match training architecture)
    # =========================================================================

    var agent = DeepPPOContinuousAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        rollout_len=ROLLOUT_LEN,
        n_envs=N_ENVS,
        gpu_minibatch_size=GPU_MINIBATCH_SIZE,
        clip_value=True,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        actor_lr=0.0003,
        critic_lr=0.0003,
        entropy_coef=0.0,
        value_loss_coef=0.5,
        num_epochs=10,
        target_kl=0.0,
        max_grad_norm=0.5,
        anneal_lr=False,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("ppo_half_cheetah.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run"
            " tests/test_ppo_half_cheetah_continuous_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create Gymnasium environment
    # =========================================================================

    comptime if RENDER:
        var env = make_half_cheetah(render_mode="human")
        print("Environment: HalfCheetah-v5 (Gymnasium, render_mode=human)")
        print("  Obs dim:", env.obs_dim())
        print("  Action dim:", env.action_dim())
        print(
            "  Action range: [", env.action_low(), ",", env.action_high(), "]"
        )
        print()

        print("Running CPU evaluation...")
        print("  Episodes:", NUM_EPISODES)
        print("  Max steps per episode:", MAX_STEPS)
        print("  Rendering: True (Gymnasium window)")
        print("  Controls: Close window to exit")
        print()
        print("-" * 70)

        var start_time = perf_counter_ns()

        var avg_reward = agent.evaluate(
            env,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            verbose=True,
            debug=False,
            stochastic=False,
            render=RENDER,
            frame_delay_ms=0,  # Gymnasium controls its own frame rate
        )

        var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

        print()
        print("-" * 70)
        print("CPU EVALUATION SUMMARY - HalfCheetah-v5 (Gymnasium)")
        print("-" * 70)
        print("Episodes:", NUM_EPISODES)
        print("Average reward:", avg_reward)
        print("Evaluation time:", elapsed_ms / 1000, "seconds")
        print()

        if avg_reward > 1000:
            print("Result: EXCELLENT - Agent is running fast!")
        elif avg_reward > 500:
            print("Result: GOOD - Agent learned to run!")
        elif avg_reward > 0:
            print("Result: OKAY - Model is learning but not optimal")
        else:
            print("Result: POOR - Model needs more training")

        env.close()
    else:
        var env = make_half_cheetah()
        print("Environment: HalfCheetah-v5 (Gymnasium, headless)")
        print("  Obs dim:", env.obs_dim())
        print("  Action dim:", env.action_dim())
        print(
            "  Action range: [", env.action_low(), ",", env.action_high(), "]"
        )
        print()

        print("Running CPU evaluation...")
        print("  Episodes:", NUM_EPISODES)
        print("  Max steps per episode:", MAX_STEPS)
        print("  Rendering: False")
        print()
        print("-" * 70)

        var start_time = perf_counter_ns()

        var avg_reward = agent.evaluate(
            env,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            verbose=True,
            debug=False,
            stochastic=False,
            render=False,
            frame_delay_ms=0,
        )

        var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

        print()
        print("-" * 70)
        print("CPU EVALUATION SUMMARY - HalfCheetah-v5 (Gymnasium)")
        print("-" * 70)
        print("Episodes:", NUM_EPISODES)
        print("Average reward:", avg_reward)
        print("Evaluation time:", elapsed_ms / 1000, "seconds")
        print()

        if avg_reward > 1000:
            print("Result: EXCELLENT - Agent is running fast!")
        elif avg_reward > 500:
            print("Result: GOOD - Agent learned to run!")
        elif avg_reward > 0:
            print("Result: OKAY - Model is learning but not optimal")
        else:
            print("Result: POOR - Model needs more training")

        env.close()

    print()
    print(">>> Gymnasium CPU Evaluation completed <<<")
