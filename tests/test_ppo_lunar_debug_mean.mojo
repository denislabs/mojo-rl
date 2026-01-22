"""Debug training to track mean action values over time.

This tests the reward normalization fix that prevents dense fuel penalties
from dominating over sparse landing bonuses.

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_debug_mean.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.lunar_lander import LunarLanderV2, LLConstants
from deep_rl import dtype


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = LLConstants.OBS_DIM_VAL
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512

# Shorter training for debugging
comptime NUM_EPISODES = 5000


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Debug: Tracking Mean Action Values During Training")
    print("=" * 70)
    print()

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
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            actor_lr=0.0003,
            critic_lr=0.001,
            entropy_coef=0.05,  # Higher entropy to prevent mean collapse
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.02,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=0,  # Disable auto-checkpoint
            checkpoint_path="",
            normalize_rewards=True,  # CleanRL-style reward normalization (NEW FIX)
        )

        print("Reward normalization: ENABLED (CleanRL-style)")
        print("Entropy coefficient: 0.05 (higher to prevent mean collapse)")
        print("This prevents fuel penalties from dominating landing bonuses.")

        # Helper function to compute mean action for a test observation
        fn get_mean_action(
            agent: DeepPPOContinuousAgent[
                OBS_DIM, ACTION_DIM, HIDDEN_DIM, ROLLOUT_LEN, N_ENVS, GPU_MINIBATCH_SIZE, True
            ]
        ) -> Tuple[Scalar[dtype], Scalar[dtype]]:
            """Get mean action for a typical initial observation."""
            # Typical initial obs: lander at top center
            var test_obs = InlineArray[Scalar[dtype], OBS_DIM](fill=Scalar[dtype](0))
            test_obs[1] = Scalar[dtype](1.4)  # y position (high)

            var result = agent.select_action(test_obs, training=False)
            var actions = result[0]
            return (actions[0], actions[1])

        # Print initial mean action
        var initial_action = get_mean_action(agent)
        print("Initial mean action (before training):")
        print("  action[0] (throttle):", initial_action[0], "-> throttle =", (Float64(initial_action[0]) + 1.0) * 0.5)
        print("  action[1] (side):", initial_action[1])
        print()

        print("Starting training with mean action tracking...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        # Train with periodic mean action checks
        var metrics = agent.train_gpu[LunarLanderV2[dtype]](
            ctx,
            num_episodes=NUM_EPISODES,
            verbose=True,
            print_every=5,  # Print every 5 rollouts
        )

        var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9

        print("-" * 70)
        print()

        # Print final mean action
        # Need to copy params from GPU first
        var final_action = get_mean_action(agent)
        print("Final mean action (after training):")
        print("  action[0] (throttle):", final_action[0], "-> throttle =", (Float64(final_action[0]) + 1.0) * 0.5)
        print("  action[1] (side):", final_action[1])
        print()

        # Run quick evaluation
        print("Quick GPU evaluation with deterministic mean:")
        var eval_reward = agent.evaluate_gpu[LunarLanderV2[dtype]](
            ctx, num_episodes=50, max_steps=1000, verbose=False
        )
        print("  Deterministic eval reward:", eval_reward)
        print()

        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("Training episodes:", NUM_EPISODES)
        print("Training time:", elapsed_s, "seconds")
        print("Final training avg (last 100):", metrics.mean_reward_last_n(100))
        print("Deterministic eval avg:", eval_reward)
        print()

        var gap = metrics.mean_reward_last_n(100) - eval_reward
        print("Gap (training - eval):", gap)

        if gap > 200:
            print()
            print("LARGE GAP DETECTED!")
            print("The mean action is not learning properly.")
            print("Training relies on exploration noise for good rewards.")

    print()
    print(">>> Debug complete <<<")
