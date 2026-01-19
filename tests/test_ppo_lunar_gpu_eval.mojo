"""Test PPO Agent Evaluation on LunarLander (CPU).

This evaluates a PPO agent trained on GPU using the CPU LunarLander environment.
Includes debug output to diagnose any issues with the transfer.

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_gpu_eval.mojo
"""

from math import exp

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOAgent
from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_v2_gpu import LunarLanderV2GPU
from render import RendererBase

# =============================================================================
# Constants
# =============================================================================

# LunarLander: 8D observation, 4 discrete actions
comptime OBS_DIM = 8
comptime NUM_ACTIONS = 4

# Network architecture
comptime HIDDEN_DIM = 300

# GPU training parameters (must match training)
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 512

# Evaluation parameters
comptime MAX_STEPS = 1000
comptime DEBUG_PRINT_EVERY = 50  # Print debug info every N steps


# =============================================================================
# Helper function to format observation
# =============================================================================


fn format_obs(obs: List[Float32]) -> String:
    """Format observation for debug printing."""
    return (
        "x="
        + String(obs[0])[:6]
        + " y="
        + String(obs[1])[:6]
        + " vx="
        + String(obs[2])[:6]
        + " vy="
        + String(obs[3])[:6]
        + " ang="
        + String(obs[4])[:6]
        + " angv="
        + String(obs[5])[:6]
        + " L="
        + String(Int(obs[6]))
        + " R="
        + String(Int(obs[7]))
    )


fn action_name(action: Int) -> String:
    """Get human-readable action name."""
    if action == 0:
        return "nop"
    elif action == 1:
        return "left"
    elif action == 2:
        return "main"
    elif action == 3:
        return "right"
    return "???"


fn format_probs(probs: InlineArray[Scalar[DType.float32], 4]) -> String:
    """Format action probabilities for debug printing."""
    return (
        "nop="
        + String(probs[0])[:5]
        + " left="
        + String(probs[1])[:5]
        + " main="
        + String(probs[2])[:5]
        + " right="
        + String(probs[3])[:5]
    )


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Agent Evaluation on LunarLander (CPU)")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    var renderer = RendererBase(width=600, height=400, title="PPO LunarLander")

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
            target_kl=0.02,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_lunar_gpu.ckpt",
        )

        print("Loading checkpoint...")
        agent.load_checkpoint("ppo_lunar_hybrid_v2.ckpt")
        print("Checkpoint loaded successfully!")
        print()

        # =====================================================================
        # Manual evaluation loop with debug output
        # =====================================================================

        print("Starting evaluation with debug output...")
        print("-" * 70)
        print()

        var env = LunarLanderV2GPU[DType.float32]()

        try:
            # Reset environment
            var obs_list = env.reset_obs_list()
            var obs = agent._list_to_inline(obs_list)

            print("Initial observation:")
            print("  " + format_obs(obs_list))
            print()

            var total_reward: Float32 = 0.0
            var step = 0

            for i in range(MAX_STEPS):
                step = i

                # Get action from agent (greedy, no exploration)
                var action_result = agent.select_action(obs, training=False)
                var action = action_result[0]

                # Get action probabilities for debugging
                var logits = InlineArray[Scalar[DType.float32], 4](
                    uninitialized=True
                )
                agent.actor.forward[1](obs, logits)

                # Compute softmax manually for debugging
                var max_logit = logits[0]
                for j in range(1, 4):
                    if logits[j] > max_logit:
                        max_logit = logits[j]
                var sum_exp = Scalar[DType.float32](0.0)
                var probs = InlineArray[Scalar[DType.float32], 4](
                    uninitialized=True
                )
                for j in range(4):
                    probs[j] = exp(logits[j] - max_logit)
                    sum_exp += probs[j]
                for j in range(4):
                    probs[j] /= sum_exp

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                total_reward += reward

                # Debug print every N steps or on significant events
                var should_print = (
                    (i % DEBUG_PRINT_EVERY == 0) or done or (i < 10)
                )
                if should_print:
                    print(
                        "Step "
                        + String(i)
                        + ": action="
                        + action_name(action)
                        + " reward="
                        + String(reward)[:7]
                        + " total="
                        + String(total_reward)[:7]
                    )
                    print("  obs: " + format_obs(next_obs_list))
                    print("  probs: " + format_probs(probs))
                    print(
                        "  logits: ["
                        + String(logits[0])[:7]
                        + ", "
                        + String(logits[1])[:7]
                        + ", "
                        + String(logits[2])[:7]
                        + ", "
                        + String(logits[3])[:7]
                        + "]"
                    )

                # Render
                env.render(renderer)

                # Update observation
                obs = agent._list_to_inline(next_obs_list)

                if done:
                    print()
                    print("Episode terminated at step " + String(i))
                    break

            print()
            print("-" * 70)
            print("EVALUATION SUMMARY")
            print("-" * 70)
            print("Total steps: " + String(step + 1))
            print("Total reward: " + String(total_reward))
            print()

            # Analyze the result
            if total_reward > 200:
                print("Result: EXCELLENT - Successful landing!")
            elif total_reward > 0:
                print("Result: GOOD - Positive reward, learning works")
            elif total_reward > -100:
                print("Result: PARTIAL - Some progress but not landing")
            elif total_reward > -300:
                print("Result: POOR - Struggling to control")
            else:
                print(
                    "Result: FAILING - Agent not transferring well to CPU"
                    " physics"
                )

            env.close()

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print()
    print(">>> Evaluation completed <<<")
