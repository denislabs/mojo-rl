"""Debug script to track train-eval gap growth over time.

This script trains with periodic evaluation to see when/how the gap appears.

Run with:
    pixi run -e apple mojo run tests/debug_ppo_gap_growth.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.pendulum import PendulumV2, PConstants


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = PConstants.OBS_DIM  # 3
comptime NUM_ACTIONS = PConstants.ACTION_DIM  # 1
comptime HIDDEN_DIM = 64
comptime ROLLOUT_LEN = 200
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 256

comptime dtype = DType.float32

# Training configuration
comptime TOTAL_EPISODES = 20000
comptime EVAL_EVERY = 2000  # Evaluate every N episodes
comptime EVAL_EPISODES = 50


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DEBUG: Tracking Train-Eval Gap Growth Over Time")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Total episodes:", TOTAL_EPISODES)
    print("  Evaluate every:", EVAL_EVERY, "episodes")
    print("  Eval episodes:", EVAL_EPISODES)
    print()

    with DeviceContext() as ctx:
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
            target_kl=0.02,  # Enable KL early stopping
            max_grad_norm=0.5,
            anneal_lr=True,
            target_total_steps=TOTAL_EPISODES * ROLLOUT_LEN,
        )

        var env = PendulumV2[dtype]()

        # Track gap over time
        var train_rewards = List[Float64]()
        var eval_rewards = List[Float64]()
        var checkpoints = List[Int]()

        var episodes_trained = 0
        var checkpoint_num = 0

        print("=" * 70)
        print("Episode | Training Avg | Eval Avg | GAP")
        print("=" * 70)

        while episodes_trained < TOTAL_EPISODES:
            # Train for EVAL_EVERY episodes
            var episodes_this_round = min(EVAL_EVERY, TOTAL_EPISODES - episodes_trained)

            var metrics = agent.train_gpu[PendulumV2[dtype]](
                ctx,
                num_episodes=episodes_this_round,
                verbose=False,
                print_every=0,  # Suppress normal output
            )

            episodes_trained += episodes_this_round
            var train_avg = metrics.mean_reward_last_n(min(100, episodes_this_round))

            # Evaluate on CPU
            var eval_avg = agent.evaluate(
                env,
                num_episodes=EVAL_EPISODES,
                max_steps=200,
                verbose=False,
                stochastic=True,
            )

            var gap = train_avg - eval_avg

            train_rewards.append(train_avg)
            eval_rewards.append(eval_avg)
            checkpoints.append(episodes_trained)

            # Print results
            print(
                String(episodes_trained).rjust(7),
                "|",
                String(train_avg)[:10].ljust(12),
                "|",
                String(eval_avg)[:10].ljust(10),
                "|",
                String(gap)[:8],
            )

            checkpoint_num += 1

        # Final summary
        print()
        print("=" * 70)
        print("GAP GROWTH ANALYSIS")
        print("=" * 70)
        print()

        print("Checkpoint | Episodes | Training | Eval | Gap")
        print("-" * 60)
        for i in range(len(checkpoints)):
            var gap = train_rewards[i] - eval_rewards[i]
            print(
                String(i + 1).rjust(10),
                "|",
                String(checkpoints[i]).rjust(8),
                "|",
                String(train_rewards[i])[:8].rjust(8),
                "|",
                String(eval_rewards[i])[:8].rjust(8),
                "|",
                String(gap)[:8].rjust(8),
            )

        print()

        # Analyze gap growth
        if len(train_rewards) > 1:
            var first_gap = train_rewards[0] - eval_rewards[0]
            var last_gap = train_rewards[len(train_rewards) - 1] - eval_rewards[
                len(eval_rewards) - 1
            ]
            var gap_growth = last_gap - first_gap

            print("Gap at start:", first_gap)
            print("Gap at end:", last_gap)
            print("Gap growth:", gap_growth)
            print()

            if gap_growth > 500.0:
                print("SIGNIFICANT GAP GROWTH DETECTED!")
                print("The train-eval gap grows during training.")
                print(
                    "This suggests the reported training rewards become"
                    " increasingly unreliable."
                )
            elif gap_growth > 100.0:
                print("MODERATE GAP GROWTH - some divergence over time")
            else:
                print("STABLE GAP - train and eval remain consistent")

        print()
        print("=" * 70)

    print(">>> Gap growth analysis completed <<<")
