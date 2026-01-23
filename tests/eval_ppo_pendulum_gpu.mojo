"""Evaluate PPO Agent on Pendulum using GPU.

This script loads a trained PPO checkpoint and evaluates it using
the GPU evaluation method (parallel environments).

This helps verify if the train-eval gap issue is environment-specific
by testing on Pendulum instead of LunarLander.

Run with:
    pixi run -e apple mojo run tests/eval_ppo_pendulum_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/eval_ppo_pendulum_gpu.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.pendulum import PendulumV2, PConstants


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

# PendulumV2: 3D observation, 1 continuous action
comptime OBS_DIM = PConstants.OBS_DIM  # 3: [cos(θ), sin(θ), θ_dot]
comptime NUM_ACTIONS = PConstants.ACTION_DIM  # 1: torque in [-2, 2]

# Network architecture (must match training)
comptime HIDDEN_DIM = 64

# GPU training parameters (must match training)
comptime ROLLOUT_LEN = 200
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 256

comptime dtype = DType.float32

# Evaluation parameters
comptime EVAL_EPISODES = 100
comptime MAX_STEPS_PER_EPISODE = 200


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Pendulum GPU Evaluation")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
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

        # =====================================================================
        # GPU Evaluation
        # =====================================================================

        print("Running GPU evaluation...")
        print("  Episodes: " + String(EVAL_EPISODES))
        print("  Max steps per episode: " + String(MAX_STEPS_PER_EPISODE))
        print()

        var start_time = perf_counter_ns()

        # Evaluate with stochastic policy (same as training)
        var stochastic_reward = agent.evaluate_gpu[PendulumV2[dtype]](
            ctx,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EPISODE,
            verbose=True,
            stochastic=True,
        )

        var stochastic_time = perf_counter_ns() - start_time

        print()
        print("-" * 70)

        # Evaluate with deterministic policy (mean actions)
        print("Running deterministic evaluation...")
        start_time = perf_counter_ns()

        var deterministic_reward = agent.evaluate_gpu[PendulumV2[dtype]](
            ctx,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS_PER_EPISODE,
            verbose=False,
            stochastic=False,
        )

        var deterministic_time = perf_counter_ns() - start_time

        # =====================================================================
        # Results
        # =====================================================================

        print()
        print("=" * 70)
        print("GPU Evaluation Results")
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
        print("=" * 70)

    print(">>> GPU evaluation completed <<<")
