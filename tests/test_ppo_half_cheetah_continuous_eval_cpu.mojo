"""CPU evaluation with rendering for continuous PPO on HalfCheetah V2.

This tests the trained continuous PPO model using the CPU evaluate method
with optional rendering to visualize the agent's behavior.

Run with:
    pixi run mojo run tests/test_ppo_half_cheetah_continuous_eval_cpu.mojo
"""

from random import seed
from time import perf_counter_ns
from memory import UnsafePointer

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.half_cheetah import HalfCheetahPlanarV2, HCConstants
from render import RendererBase
from deep_rl import dtype


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = HCConstants.OBS_DIM_VAL  # 17
comptime ACTION_DIM = HCConstants.ACTION_DIM_VAL  # 6
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
        target_kl=0.0,
        max_grad_norm=0.5,
        anneal_lr=False,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("ppo_half_cheetah_cleanrl.ckpt")
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
    # Create environment and renderer
    # =========================================================================

    # Create CPU environment
    var env = HalfCheetahPlanarV2[dtype]()

    # Create renderer if enabled
    @parameter
    if RENDER:
        var renderer = RendererBase(
            width=800,
            height=400,
            title="PPO HalfCheetah V2 Continuous - CPU Eval",
        )

        print("Running CPU evaluation with rendering...")
        print("  Episodes:", NUM_EPISODES)
        print("  Max steps per episode:", MAX_STEPS)
        print()
        print(
            "----------------------------------------------------------------------"
        )

        var start_time = perf_counter_ns()

        # Run evaluation with rendering
        var avg_reward = agent.evaluate[HalfCheetahPlanarV2[dtype]](
            env,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            verbose=True,
            stochastic=True,  # Use sampling like training (set False for deterministic)
            renderer=UnsafePointer(to=renderer),
        )

        var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

        print()
        print(
            "----------------------------------------------------------------------"
        )
        print("CPU EVALUATION SUMMARY (with rendering)")
        print(
            "----------------------------------------------------------------------"
        )
        print("Episodes completed:", NUM_EPISODES)
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

        renderer.close()

    else:
        print("Running CPU evaluation (headless)...")
        print("  Episodes:", NUM_EPISODES)
        print("  Max steps per episode:", MAX_STEPS)
        print()
        print(
            "----------------------------------------------------------------------"
        )

        var start_time = perf_counter_ns()

        # Run evaluation without rendering (stochastic to match training)
        var avg_reward = agent.evaluate[HalfCheetahPlanarV2[dtype]](
            env,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS,
            verbose=True,
            stochastic=True,  # Use sampling like training
        )

        var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

        print()
        print(
            "----------------------------------------------------------------------"
        )
        print("CPU EVALUATION SUMMARY (headless)")
        print(
            "----------------------------------------------------------------------"
        )
        print("Episodes completed:", NUM_EPISODES)
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

    print()
    print(">>> CPU Evaluation completed <<<")
