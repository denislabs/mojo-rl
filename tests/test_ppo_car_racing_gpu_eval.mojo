"""Test PPO Agent Evaluation on CarRacing (CPU).

This evaluates a PPO agent trained on GPU using the CPU CarRacingV2 environment.
Includes debug output to diagnose any issues with the transfer.

Run with:
    pixi run -e apple mojo run tests/test_ppo_car_racing_gpu_eval.mojo
"""

from math import exp, tanh

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.car_racing import CarRacingV2, CarRacingV2Action
from render import RendererBase

# =============================================================================
# Constants
# =============================================================================

# CarRacingV2: 13D observation, 3 continuous actions
comptime OBS_DIM = 13
comptime NUM_ACTIONS = 3

# Network architecture
comptime HIDDEN_DIM = 300

# GPU training parameters (must match training)
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512

# Evaluation parameters
comptime MAX_STEPS = 500
comptime DEBUG_PRINT_EVERY = 50  # Print debug info every N steps

comptime dtype = DType.float32


# =============================================================================
# Helper function to format observation
# =============================================================================


fn format_obs(obs: List[Scalar[dtype]]) -> String:
    """Format observation for debug printing."""
    return (
        "x="
        + String(obs[0])[:6]
        + " y="
        + String(obs[1])[:6]
        + " ang="
        + String(obs[2])[:5]
        + " vx="
        + String(obs[3])[:5]
        + " vy="
        + String(obs[4])[:5]
        + " spd="
        + String(obs[12])[:5]
    )


fn format_action(steering: Float32, gas: Float32, brake: Float32) -> String:
    """Format action for debug printing."""
    return (
        "steer="
        + String(steering)[:6]
        + " gas="
        + String(gas)[:5]
        + " brake="
        + String(brake)[:5]
    )


fn format_actor_output(
    means: InlineArray[Scalar[dtype], NUM_ACTIONS],
    log_stds: InlineArray[Scalar[dtype], NUM_ACTIONS],
) -> String:
    """Format actor output for debug printing."""
    return (
        "means=["
        + String(means[0])[:6]
        + ", "
        + String(means[1])[:6]
        + ", "
        + String(means[2])[:6]
        + "] log_stds=["
        + String(log_stds[0])[:5]
        + ", "
        + String(log_stds[1])[:5]
        + ", "
        + String(log_stds[2])[:5]
        + "]"
    )


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Agent Evaluation on CarRacing (CPU)")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    var renderer = RendererBase(width=800, height=600, title="PPO CarRacing")

    with DeviceContext() as ctx:
        # Initialize with same action biases as training
        var action_mean_biases = List[Float64]()
        action_mean_biases.append(0.0)   # steering: centered
        action_mean_biases.append(0.5)   # gas: slight forward bias
        action_mean_biases.append(-0.5)  # brake: slight no-brake bias

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
            num_epochs=10,
            target_kl=0.0,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_car_racing_gpu_v2.ckpt",
            action_mean_biases=action_mean_biases^,
        )

        print("Loading checkpoint...")
        agent.load_checkpoint("ppo_car_racing_gpu_v2.ckpt")
        print("Checkpoint loaded successfully!")
        print()

        # =====================================================================
        # Manual evaluation loop with debug output
        # =====================================================================

        print("Starting evaluation with debug output...")
        print("-" * 70)
        print()

        var env = CarRacingV2[dtype]()

        try:
            # Reset environment
            var obs_list = env.reset_obs_list()

            # Convert to InlineArray for forward pass
            var obs = InlineArray[Scalar[dtype], OBS_DIM](uninitialized=True)
            for i in range(OBS_DIM):
                obs[i] = obs_list[i]

            print("Initial observation:")
            print("  " + format_obs(obs_list))
            print()

            var total_reward: Float32 = 0.0
            var step = 0

            for i in range(MAX_STEPS):
                step = i

                # Forward pass through actor to get mean and log_std
                # StochasticActor outputs: [mean_0, ..., mean_n, log_std_0, ..., log_std_n]
                var actor_output = InlineArray[Scalar[dtype], 2 * NUM_ACTIONS](
                    uninitialized=True
                )
                agent.actor.forward[1](obs, actor_output)

                # Extract means and log_stds
                var means = InlineArray[Scalar[dtype], NUM_ACTIONS](
                    uninitialized=True
                )
                var log_stds = InlineArray[Scalar[dtype], NUM_ACTIONS](
                    uninitialized=True
                )
                for j in range(NUM_ACTIONS):
                    means[j] = actor_output[j]
                    log_stds[j] = actor_output[NUM_ACTIONS + j]

                # For evaluation, use mean action (deterministic)
                # Apply tanh squashing to bound actions to [-1, 1]
                var steering = Scalar[dtype](tanh(Float64(means[0])))
                var gas = Scalar[dtype](tanh(Float64(means[1])))
                var brake = Scalar[dtype](tanh(Float64(means[2])))

                # Create action
                var action = CarRacingV2Action[dtype](steering, gas, brake)

                # Step environment
                var result = env.step(action)
                var next_state = result[0]
                var reward = result[1]
                var done = result[2]

                total_reward += Float32(reward)

                # Get next observation
                var next_obs_list = next_state.to_list_typed[dtype]()

                # Debug print every N steps or on significant events
                var should_print = (
                    (i % DEBUG_PRINT_EVERY == 0) or done or (i < 10)
                )
                if should_print:
                    print(
                        "Step "
                        + String(i)
                        + ": "
                        + format_action(Float32(steering), Float32(gas), Float32(brake))
                        + " reward="
                        + String(reward)[:7]
                        + " total="
                        + String(total_reward)[:7]
                    )
                    print("  obs: " + format_obs(next_obs_list))
                    print("  " + format_actor_output(means, log_stds))

                # Render
                env.render(renderer)

                # Update observation
                for j in range(OBS_DIM):
                    obs[j] = next_obs_list[j]

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
            if total_reward > 1000:
                print("Result: EXCELLENT - Great driving!")
            elif total_reward > 500:
                print("Result: GOOD - Solid forward motion")
            elif total_reward > 100:
                print("Result: PARTIAL - Some progress but can improve")
            elif total_reward > 0:
                print("Result: LEARNING - Positive reward, moving forward")
            else:
                print("Result: NEEDS TRAINING - Agent not performing well")

            env.close()

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print()
    print(">>> Evaluation completed <<<")
