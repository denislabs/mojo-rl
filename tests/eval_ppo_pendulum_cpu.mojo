"""Evaluate PPO Agent on Pendulum using CPU with Rendering.

This script loads a trained PPO checkpoint and evaluates it using
the CPU evaluation method with optional SDL2 rendering.

This helps verify if the train-eval gap issue is environment-specific
by testing on Pendulum instead of LunarLander.

Run with:
    pixi run mojo run tests/eval_ppo_pendulum_cpu.mojo

Requirements:
    - SDL2 installed (brew install sdl2 sdl2_ttf on macOS)
    - Trained checkpoint file (ppo_pendulum_gpu.ckpt)
"""

from random import seed
from time import perf_counter_ns, sleep

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.pendulum import PendulumV2, PConstants
from render import RendererBase


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

# PendulumV2: 3D observation, 1 continuous action
comptime OBS_DIM = PConstants.OBS_DIM  # 3: [cos(θ), sin(θ), θ_dot]
comptime NUM_ACTIONS = PConstants.ACTION_DIM  # 1: torque in [-2, 2]

# Network architecture (must match training)
comptime HIDDEN_DIM = 64

# GPU training parameters (must match training for checkpoint compatibility)
comptime ROLLOUT_LEN = 200
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 256

comptime dtype = DType.float32

# Evaluation parameters
comptime EVAL_EPISODES = 10  # Fewer episodes since we're rendering
comptime MAX_STEPS_PER_EPISODE = 200

# Rendering parameters
comptime RENDER = True
comptime RENDER_DELAY_MS = 50  # Delay between frames (20 FPS)
comptime WINDOW_WIDTH = 500
comptime WINDOW_HEIGHT = 500


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Pendulum CPU Evaluation (with Rendering)")
    print("=" * 70)
    print()

    # Create agent with same architecture as training
    var agent = DeepPPOContinuousAgent[
        obs_dim=OBS_DIM,
        action_dim=NUM_ACTIONS,
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
        num_epochs=4,
    )

    # Load checkpoint
    var checkpoint_path = "ppo_pendulum_gpu.ckpt"
    print("Loading checkpoint: " + checkpoint_path)

    try:
        agent.load_checkpoint(checkpoint_path)
        print("Checkpoint loaded successfully!")
    except e:
        print("ERROR: Could not load checkpoint: " + String(e))
        print(
            "Make sure you have trained the agent first with"
            " test_ppo_pendulum_gpu.mojo"
        )
        return

    print()

    # Print weight diagnostics
    print("Weight diagnostics after loading:")
    print(
        "  Actor params (first 5):",
        agent.actor.params[0],
        agent.actor.params[1],
        agent.actor.params[2],
        agent.actor.params[3],
        agent.actor.params[4],
    )

    var actor_l1: Float64 = 0.0
    for i in range(len(agent.actor.params)):
        actor_l1 += Float64(abs(agent.actor.params[i]))
    print("  Actor weight L1 norm:", actor_l1)

    # Log_std params are the last ACTION_DIM params
    var log_std_offset = len(agent.actor.params) - NUM_ACTIONS
    print("  log_std params:", agent.actor.params[log_std_offset])
    print()

    # Create environment
    var env = PendulumV2[dtype]()

    @parameter
    if RENDER:
        print("Initializing SDL2 renderer...")
        var renderer = RendererBase(
            width=600, height=400, title="PPO Pendulum Continuous - CPU Eval"
        )

        print("Renderer initialized!")
        print()

        # =========================================================================
        # CPU Evaluation with Rendering
        # =========================================================================

        print("Running CPU evaluation...")
        print("  Episodes: " + String(EVAL_EPISODES))
        print("  Max steps per episode: " + String(MAX_STEPS_PER_EPISODE))
        print("  Rendering: " + String(RENDER))
        print()

        var start_time = perf_counter_ns()

        # Evaluate with stochastic policy
        var stochastic_reward = agent.evaluate(
            env,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EPISODE,
            verbose=True,
            stochastic=True,
            renderer=UnsafePointer(to=renderer),
        )

        var stochastic_time = perf_counter_ns() - start_time

        print()
        print("-" * 70)

        # Evaluate with deterministic policy (no rendering for speed)
        print("Running deterministic evaluation (no rendering)...")
        start_time = perf_counter_ns()

        var deterministic_reward = agent.evaluate(
            env,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EPISODE,
            verbose=True,
            stochastic=False,
            renderer=UnsafePointer[
                RendererBase, MutAnyOrigin
            ](),  # No rendering
        )

        var deterministic_time = perf_counter_ns() - start_time

        # =========================================================================
        # Results
        # =========================================================================

        print()
        print("=" * 70)
        print("CPU Evaluation Results")
        print("=" * 70)
        print()
        print("Stochastic policy (sampling from distribution):")
        print("  Average reward: " + String(stochastic_reward)[:10])
        print("  Time: " + String(Float64(stochastic_time) / 1e9)[:6] + "s")
        print()
        print("Deterministic policy (using mean actions):")
        print("  Average reward: " + String(deterministic_reward)[:10])
        print("  Time: " + String(Float64(deterministic_time) / 1e9)[:6] + "s")
        print()

        # Interpret results
        print("Expected Pendulum rewards:")
        print("  Random policy: ~-1200 to -1600")
        print("  Good policy: > -200")
        print("  Optimal: ~0 (balanced at top)")
        print()

        if stochastic_reward > -200.0:
            print("SUCCESS: Agent learned to swing up and balance!")
        elif stochastic_reward > -500.0:
            print("GOOD: Agent shows learning progress")
        elif stochastic_reward > -1000.0:
            print("LEARNING: Agent improving but needs more training")
        else:
            print("POOR: Agent performance similar to random policy")
            print("  This may indicate a train-eval gap issue!")

        print()

        # Compare to expected training performance
        print("Diagnostic comparison:")
        print("  If training showed good rewards but evaluation is poor,")
        print("  there may be a train-eval gap similar to LunarLander.")
        print()
        print("  Gap indicator:")
        if stochastic_reward < -1000.0:
            print(
                "    LARGE GAP DETECTED - evaluation much worse than typical"
                " training"
            )
        elif stochastic_reward < -500.0:
            print(
                "    MODERATE GAP - some discrepancy between training and eval"
            )
        else:
            print(
                "    SMALL/NO GAP - evaluation consistent with training"
                " performance"
            )

        print()
        print("=" * 70)

        # Cleanup renderer
        renderer.close()
    else:
        # =========================================================================
        # CPU Evaluation with Rendering
        # =========================================================================

        print("Running CPU evaluation...")
        print("  Episodes: " + String(EVAL_EPISODES))
        print("  Max steps per episode: " + String(MAX_STEPS_PER_EPISODE))
        print("  Rendering: " + String(RENDER))
        print()

        var start_time = perf_counter_ns()

        # Evaluate with stochastic policy
        var stochastic_reward = agent.evaluate(
            env,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EPISODE,
            verbose=True,
            stochastic=True,
        )

        var stochastic_time = perf_counter_ns() - start_time

        print()
        print("-" * 70)

        # Evaluate with deterministic policy (no rendering for speed)
        print("Running deterministic evaluation (no rendering)...")
        start_time = perf_counter_ns()

        var deterministic_reward = agent.evaluate(
            env,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EPISODE,
            verbose=True,
            stochastic=False,
            renderer=UnsafePointer[
                RendererBase, MutAnyOrigin
            ](),  # No rendering
        )

        var deterministic_time = perf_counter_ns() - start_time

        # =========================================================================
        # Results
        # =========================================================================

        print()
        print("=" * 70)
        print("CPU Evaluation Results")
        print("=" * 70)
        print()
        print("Stochastic policy (sampling from distribution):")
        print("  Average reward: " + String(stochastic_reward)[:10])
        print("  Time: " + String(Float64(stochastic_time) / 1e9)[:6] + "s")
        print()
        print("Deterministic policy (using mean actions):")
        print("  Average reward: " + String(deterministic_reward)[:10])
        print("  Time: " + String(Float64(deterministic_time) / 1e9)[:6] + "s")
        print()

        # Interpret results
        print("Expected Pendulum rewards:")
        print("  Random policy: ~-1200 to -1600")
        print("  Good policy: > -200")
        print("  Optimal: ~0 (balanced at top)")
        print()

        if stochastic_reward > -200.0:
            print("SUCCESS: Agent learned to swing up and balance!")
        elif stochastic_reward > -500.0:
            print("GOOD: Agent shows learning progress")
        elif stochastic_reward > -1000.0:
            print("LEARNING: Agent improving but needs more training")
        else:
            print("POOR: Agent performance similar to random policy")
            print("  This may indicate a train-eval gap issue!")

        print()

        # Compare to expected training performance
        print("Diagnostic comparison:")
        print("  If training showed good rewards but evaluation is poor,")
        print("  there may be a train-eval gap similar to LunarLander.")
        print()
        print("  Gap indicator:")
        if stochastic_reward < -1000.0:
            print(
                "    LARGE GAP DETECTED - evaluation much worse than typical"
                " training"
            )
        elif stochastic_reward < -500.0:
            print(
                "    MODERATE GAP - some discrepancy between training and eval"
            )
        else:
            print(
                "    SMALL/NO GAP - evaluation consistent with training"
                " performance"
            )

        print()
        print("=" * 70)

    print(">>> CPU evaluation completed <<<")
