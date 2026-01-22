"""CPU evaluation with rendering for continuous PPO on LunarLander.

This tests the trained continuous PPO model using the CPU evaluate method
with optional rendering to visualize the agent's behavior.

Run with:
    pixi run mojo run tests/test_ppo_lunar_continuous_eval_cpu.mojo
"""

from random import seed
from time import perf_counter_ns
from memory import UnsafePointer

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.lunar_lander import LunarLanderV2, LLConstants
from render import RendererBase
from deep_rl import dtype


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000
comptime RENDER = True  # Set to False for headless evaluation


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent CPU Evaluation with Rendering")
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
        critic_lr=0.001,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        num_epochs=10,
        target_kl=0.02,
        max_grad_norm=0.5,
        anneal_lr=True,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("ppo_lunar_continuous_gpu.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print("  pixi run -e apple mojo run tests/test_ppo_lunar_continuous_gpu.mojo")
        return

    print()

    # =========================================================================
    # Create environment and renderer
    # =========================================================================

    # Create CPU environment (LunarLanderV2 supports both discrete and continuous)
    var env = LunarLanderV2[dtype]()

    # Create renderer if enabled
    @parameter
    if RENDER:
        var renderer = RendererBase(
            width=600, height=400, title="PPO LunarLander Continuous - CPU Eval"
        )

        print("Running CPU evaluation with rendering...")
        print("  Episodes:", NUM_EPISODES)
        print("  Max steps per episode:", MAX_STEPS)
        print()
        print("----------------------------------------------------------------------")

        var start_time = perf_counter_ns()

        # Run evaluation with rendering
        var avg_reward = agent.evaluate[LunarLanderV2[dtype]](
            env,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            verbose=True,
            renderer=UnsafePointer(to=renderer),
        )

        var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

        print()
        print("----------------------------------------------------------------------")
        print("CPU EVALUATION SUMMARY (with rendering)")
        print("----------------------------------------------------------------------")
        print("Episodes completed:", NUM_EPISODES)
        print("Average reward:", avg_reward)
        print("Evaluation time:", elapsed_ms / 1000, "seconds")
        print()

        if avg_reward > 0:
            print("Result: GOOD - Positive average reward!")
        elif avg_reward > -100:
            print("Result: OKAY - Model is learning but not optimal")
        else:
            print("Result: POOR - Model needs more training")

        renderer.close()

    else:
        print("Running CPU evaluation (headless)...")
        print("  Episodes:", NUM_EPISODES)
        print("  Max steps per episode:", MAX_STEPS)
        print()
        print("----------------------------------------------------------------------")

        var start_time = perf_counter_ns()

        # Run evaluation without rendering
        var avg_reward = agent.evaluate[LunarLanderV2[dtype]](
            env,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            verbose=True,
        )

        var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

        print()
        print("----------------------------------------------------------------------")
        print("CPU EVALUATION SUMMARY (headless)")
        print("----------------------------------------------------------------------")
        print("Episodes completed:", NUM_EPISODES)
        print("Average reward:", avg_reward)
        print("Evaluation time:", elapsed_ms / 1000, "seconds")
        print()

        if avg_reward > 0:
            print("Result: GOOD - Positive average reward!")
        elif avg_reward > -100:
            print("Result: OKAY - Model is learning but not optimal")
        else:
            print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
