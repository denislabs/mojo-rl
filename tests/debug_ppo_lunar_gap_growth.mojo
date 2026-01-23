"""Debug script to track train-eval gap growth over time for LunarLander.

This script trains with periodic evaluation to see when/how the gap appears.
It also tracks log_std parameters to detect drift.

Run with:
    pixi run -e apple mojo run tests/debug_ppo_lunar_gap_growth.mojo
    pixi run -e nvidia mojo run tests/debug_ppo_lunar_gap_growth.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.lunar_lander import LunarLanderV2, LLConstants
from deep_rl import dtype as gpu_dtype


# =============================================================================
# Constants (matching test_ppo_lunar_continuous_gpu.mojo)
# =============================================================================

comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512

comptime dtype = DType.float32

# Training configuration
comptime TOTAL_EPISODES = 10000
comptime EVAL_EVERY = 1000  # Evaluate every N episodes
comptime EVAL_EPISODES = 50
comptime MAX_STEPS = 1000


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DEBUG: Tracking Train-Eval Gap Growth Over Time (LunarLander)")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Total episodes:", TOTAL_EPISODES)
    print("  Evaluate every:", EVAL_EVERY, "episodes")
    print("  Eval episodes:", EVAL_EPISODES)
    print("  Max steps per episode:", MAX_STEPS)
    print()
    print("Agent configuration (matching training script):")
    print("  OBS_DIM:", OBS_DIM)
    print("  ACTION_DIM:", ACTION_DIM)
    print("  HIDDEN_DIM:", HIDDEN_DIM)
    print("  entropy_coef: 0.05")
    print("  target_kl: 0.1")
    print("  anneal_lr: False")
    print("  normalize_rewards: True")
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
            entropy_coef=0.05,  # Higher entropy for LunarLander
            value_loss_coef=0.5,
            num_epochs=10,  # More epochs for LunarLander
            target_kl=0.1,  # KL early stopping
            max_grad_norm=0.5,
            anneal_lr=False,  # Disabled for LunarLander
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=0,  # Disable auto checkpoints
            checkpoint_path="",
            normalize_rewards=True,  # CleanRL-style
        )

        # Get log_std offset in actor params
        var num_actor_params = len(agent.actor.params)
        var log_std_offset = num_actor_params - ACTION_DIM

        # Track gap over time
        var train_rewards = List[Float64]()
        var eval_cpu_rewards = List[Float64]()
        var eval_gpu_rewards = List[Float64]()
        var checkpoints = List[Int]()
        var log_std_0_values = List[Float64]()
        var log_std_1_values = List[Float64]()

        var episodes_trained = 0
        var checkpoint_num = 0

        # Initial log_std values
        print("Initial log_std params:")
        print("  log_std[0]:", agent.actor.params[log_std_offset])
        print("  log_std[1]:", agent.actor.params[log_std_offset + 1])
        print()

        print("=" * 90)
        print(
            "Episode  | Train Avg | CPU Eval | GPU Eval | Gap(CPU) |"
            " Gap(GPU) | log_std[0] | log_std[1]"
        )
        print("=" * 90)

        while episodes_trained < TOTAL_EPISODES:
            # Train for EVAL_EVERY episodes
            var episodes_this_round = min(
                EVAL_EVERY, TOTAL_EPISODES - episodes_trained
            )

            var metrics = agent.train_gpu[LunarLanderV2[gpu_dtype]](
                ctx,
                num_episodes=episodes_this_round,
                verbose=False,
                print_every=0,  # Suppress normal output
            )

            episodes_trained += episodes_this_round
            var train_avg = metrics.mean_reward_last_n(
                min(100, episodes_this_round)
            )

            # Get current log_std values
            var log_std_0 = Float64(agent.actor.params[log_std_offset])
            var log_std_1 = Float64(agent.actor.params[log_std_offset + 1])

            # Evaluate on GPU (stochastic)
            var eval_gpu_avg = agent.evaluate_gpu[LunarLanderV2[gpu_dtype]](
                ctx,
                num_episodes=EVAL_EPISODES,
                max_steps=MAX_STEPS,
                verbose=False,
                stochastic=True,
            )

            # Evaluate on CPU (stochastic)
            var env = LunarLanderV2[dtype]()
            var eval_cpu_avg = agent.evaluate(
                env,
                num_episodes=EVAL_EPISODES,
                max_steps=MAX_STEPS,
                verbose=False,
                stochastic=True,
            )

            var gap_cpu = train_avg - eval_cpu_avg
            var gap_gpu = train_avg - eval_gpu_avg

            train_rewards.append(train_avg)
            eval_cpu_rewards.append(eval_cpu_avg)
            eval_gpu_rewards.append(eval_gpu_avg)
            checkpoints.append(episodes_trained)
            log_std_0_values.append(log_std_0)
            log_std_1_values.append(log_std_1)

            # Print results
            print(
                String(episodes_trained).rjust(8),
                "|",
                String(train_avg)[:9].ljust(9),
                "|",
                String(eval_cpu_avg)[:8].ljust(8),
                "|",
                String(eval_gpu_avg)[:8].ljust(8),
                "|",
                String(gap_cpu)[:8].ljust(8),
                "|",
                String(gap_gpu)[:8].ljust(8),
                "|",
                String(log_std_0)[:10].ljust(10),
                "|",
                String(log_std_1)[:10],
            )

            checkpoint_num += 1

        # Final summary
        print()
        print("=" * 90)
        print("GAP GROWTH ANALYSIS")
        print("=" * 90)
        print()

        print(
            "Chkpt | Episodes | Training | CPU Eval | GPU Eval |"
            " Gap(CPU) | Gap(GPU) | log_std[0] | log_std[1]"
        )
        print("-" * 110)
        for i in range(len(checkpoints)):
            var gap_cpu = train_rewards[i] - eval_cpu_rewards[i]
            var gap_gpu = train_rewards[i] - eval_gpu_rewards[i]
            print(
                String(i + 1).rjust(5),
                "|",
                String(checkpoints[i]).rjust(8),
                "|",
                String(train_rewards[i])[:8].rjust(8),
                "|",
                String(eval_cpu_rewards[i])[:8].rjust(8),
                "|",
                String(eval_gpu_rewards[i])[:8].rjust(8),
                "|",
                String(gap_cpu)[:8].rjust(8),
                "|",
                String(gap_gpu)[:8].rjust(8),
                "|",
                String(log_std_0_values[i])[:10].rjust(10),
                "|",
                String(log_std_1_values[i])[:10].rjust(10),
            )

        print()

        # Analyze gap growth
        if len(train_rewards) > 1:
            var first_gap_cpu = train_rewards[0] - eval_cpu_rewards[0]
            var last_gap_cpu = (
                train_rewards[len(train_rewards) - 1]
                - eval_cpu_rewards[len(eval_cpu_rewards) - 1]
            )
            var gap_growth_cpu = last_gap_cpu - first_gap_cpu

            var first_gap_gpu = train_rewards[0] - eval_gpu_rewards[0]
            var last_gap_gpu = (
                train_rewards[len(train_rewards) - 1]
                - eval_gpu_rewards[len(eval_gpu_rewards) - 1]
            )
            var gap_growth_gpu = last_gap_gpu - first_gap_gpu

            print("CPU Evaluation Gap Analysis:")
            print("  Gap at start:", first_gap_cpu)
            print("  Gap at end:", last_gap_cpu)
            print("  Gap growth:", gap_growth_cpu)
            print()

            print("GPU Evaluation Gap Analysis:")
            print("  Gap at start:", first_gap_gpu)
            print("  Gap at end:", last_gap_gpu)
            print("  Gap growth:", gap_growth_gpu)
            print()

            # Log_std analysis
            var log_std_0_start = log_std_0_values[0]
            var log_std_0_end = log_std_0_values[len(log_std_0_values) - 1]
            var log_std_1_start = log_std_1_values[0]
            var log_std_1_end = log_std_1_values[len(log_std_1_values) - 1]

            print("log_std Parameter Analysis:")
            print(
                "  log_std[0]: ",
                log_std_0_start,
                " -> ",
                log_std_0_end,
                " (change: ",
                log_std_0_end - log_std_0_start,
                ")",
            )
            print(
                "  log_std[1]: ",
                log_std_1_start,
                " -> ",
                log_std_1_end,
                " (change: ",
                log_std_1_end - log_std_1_start,
                ")",
            )
            print()

            # Check for log_std drift
            if log_std_0_end > 2.0 or log_std_1_end > 2.0:
                print("WARNING: log_std exceeded LOG_STD_MAX=2.0!")
                print(
                    "  This causes very high variance actions during training"
                )
                print("  but deterministic eval uses clamped values.")
            elif log_std_0_end < -5.0 or log_std_1_end < -5.0:
                print("WARNING: log_std went below LOG_STD_MIN=-5.0!")
                print("  This causes near-deterministic training behavior.")

            # Gap analysis
            var max_gap_growth = max(abs(gap_growth_cpu), abs(gap_growth_gpu))
            if max_gap_growth > 100.0:
                print("SIGNIFICANT GAP GROWTH DETECTED!")
                print("The train-eval gap grows during training.")
                print(
                    "This suggests the reported training rewards become"
                    " increasingly unreliable."
                )
            elif max_gap_growth > 50.0:
                print("MODERATE GAP GROWTH - some divergence over time")
            else:
                print(
                    "STABLE GAP - train and eval remain relatively consistent"
                )

            # CPU vs GPU eval comparison
            var cpu_gpu_diff = (
                eval_cpu_rewards[len(eval_cpu_rewards) - 1]
                - eval_gpu_rewards[len(eval_gpu_rewards) - 1]
            )
            print()
            print("CPU vs GPU Evaluation difference (final):", cpu_gpu_diff)
            if abs(cpu_gpu_diff) > 50.0:
                print("  WARNING: Large CPU/GPU eval difference detected!")
                print("  This may indicate environment physics differences.")

        print()
        print("=" * 90)

    print(">>> Gap growth analysis completed <<<")
