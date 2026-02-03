"""CPU evaluation with 3D rendering for continuous PPO on Hopper3D.

This tests the trained continuous PPO model using CPU evaluation
with optional 3D visualization using the RenderableEnv trait.

Run with:
    pixi run mojo run tests/test_ppo_hopper_3d_continuous_eval_cpu.mojo
"""

from random import seed
from time import perf_counter_ns

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.hopper_3d import Hopper3D
from envs.hopper_3d.constants3d import Hopper3DConstantsGPU


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = Hopper3DConstantsGPU.OBS_DIM  # 11
comptime ACTION_DIM = Hopper3DConstantsGPU.ACTION_DIM  # 3
# Must match training configuration!
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

# Evaluation settings
comptime NUM_EPISODES = 10
comptime MAX_STEPS = 1000  # Hopper3D episodes run for max 1000 steps
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
        agent.load_checkpoint("ppo_hopper_3d.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first:")
        print(
            "  pixi run -e apple mojo run"
            " tests/test_ppo_hopper_3d_continuous_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Create environment and evaluate using RenderableEnv trait
    # =========================================================================

    var env = Hopper3D()

    print("Running CPU evaluation...")
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print("  Rendering:", RENDER)

    @parameter
    if RENDER:
        print("  Controls: Close window to exit")
    print()
    print("-" * 70)

    var start_time = perf_counter_ns()

    # Use the evaluate_renderable method that leverages RenderableEnv trait
    # The environment handles its own 3D renderer internally
    var avg_reward = agent.evaluate_renderable(
        env,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
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
