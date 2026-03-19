"""CPU evaluation with 3D rendering for continuous PPO on Hopper.

This tests the trained continuous PPO model using CPU evaluation
with optional 3D visualization using the RenderableEnv trait.

Run with:
    pixi run mojo run -I . tests/test_ppo_hopper_continuous_eval_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepPPOContinuousAgent
from mojo_rl.envs.hopper import Hopper, HopperConfig


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3
# Must match training configuration!
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000  # Hopper episodes run for max 1000 steps
comptime RENDER = True  # Set to False for headless evaluation


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent CPU Evaluation with 3D Rendering")
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
        actor_lr=0.0003,
        critic_lr=0.0003,
    ](
        clip_value=True,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.0,
        value_loss_coef=0.5,
        num_epochs=10,
        target_kl=0.0,
        max_grad_norm=0.5,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("ppo_hopper.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run"
            " tests/test_ppo_hopper_continuous_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate using RenderableEnv trait
    # =========================================================================

    var env = Hopper[DType.float64, TERMINATE_ON_UNHEALTHY=False]()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print("  Rendering:", RENDER)

    comptime if RENDER:
        print("  Controls: Close window to exit")
    print()
    print("-" * 70)

    var start_time = perf_counter_ns()

    # Use the evaluate method that leverages RenderableEnv trait
    # The environment handles its own 3D renderer internally
    var avg_reward = agent.evaluate(
        env,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        verbose=True,
        stochastic=False,  # Use deterministic policy for evaluation
        render=RENDER,
        frame_delay_ms=16,  # ~60 FPS
    )

    var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

    print()
    print("-" * 70)
    print("CPU EVALUATION SUMMARY")
    print("-" * 70)
    print("Episodes:", NUM_EPISODES)
    print("Average reward:", avg_reward)
    print("Evaluation time:", elapsed_ms / 1000, "seconds")
    print()

    if avg_reward > 1000:
        print("Result: EXCELLENT - Agent is hopping well!")
    elif avg_reward > 500:
        print("Result: GOOD - Agent learned to hop!")
    elif avg_reward > 0:
        print("Result: OKAY - Model is learning but not optimal")
    else:
        print("Result: POOR - Model needs more training")

    print()
    print(">>> CPU Evaluation completed <<<")
