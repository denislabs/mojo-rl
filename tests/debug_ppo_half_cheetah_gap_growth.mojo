"""Debug script to track train-eval gap growth over time for HalfCheetah.

This script trains with periodic evaluation to see when/how the gap appears.
It also tracks log_std parameters to detect drift.

Run with:
    pixi run -e apple mojo run tests/debug_ppo_half_cheetah_gap_growth.mojo
    pixi run -e nvidia mojo run tests/debug_ppo_half_cheetah_gap_growth.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.half_cheetah import HalfCheetahPlanarV2, HCConstants
from deep_rl import dtype as gpu_dtype


# =============================================================================
# Constants (matching test_ppo_half_cheetah_continuous_gpu.mojo)
# =============================================================================

comptime OBS_DIM = HCConstants.OBS_DIM_VAL  # 17
comptime ACTION_DIM = HCConstants.ACTION_DIM_VAL  # 6
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

comptime dtype = DType.float32

# Training configuration
comptime TOTAL_EPISODES = 20_000
comptime EVAL_EVERY = 500  # Evaluate every N episodes
comptime EVAL_EPISODES = 50
comptime MAX_STEPS = 1000  # HalfCheetah episodes


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DEBUG: Tracking Train-Eval Gap Growth Over Time (HalfCheetah)")
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
    print("  entropy_coef: 0.0 (CleanRL MuJoCo setting)")
    print("  target_kl: 0.0 (disabled)")
    print("  anneal_lr: True")
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
            critic_lr=0.0003,  # Same as actor (CleanRL style)
            entropy_coef=0.0,  # CleanRL: 0 for MuJoCo
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.0,  # KL early stopping disabled
            max_grad_norm=0.5,
            anneal_lr=True,  # CleanRL uses LR annealing
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=0,  # Disable auto checkpoints
            checkpoint_path="",
            normalize_rewards=True,  # CleanRL-style
            obs_noise_std=0.0,  # No observation noise
        )

        # Get log_std offset in actor params
        var num_actor_params = len(agent.actor.params)
        var log_std_offset = num_actor_params - ACTION_DIM

        # Track gap over time
        var train_rewards = List[Float64]()
        var eval_cpu_rewards = List[Float64]()
        var eval_gpu_rewards = List[Float64]()
        var checkpoints = List[Int]()
        var log_std_values = List[List[Float64]]()

        var episodes_trained = 0
        var checkpoint_num = 0

        # Initial log_std values
        print("Initial log_std params:")
        for i in range(ACTION_DIM):
            print(
                "  log_std[" + String(i) + "]:",
                agent.actor.params[log_std_offset + i],
            )
        print()

        print("=" * 130)
        print(
            "Episode  | Train Avg | CPU Eval | GPU Eval | Gap(CPU) |"
            " Gap(GPU) | log_std[0] | log_std[1] | log_std[2] |"
            " log_std[3] | log_std[4] | log_std[5]"
        )
        print("=" * 130)

        while episodes_trained < TOTAL_EPISODES:
            # Train for EVAL_EVERY episodes
            var episodes_this_round = min(
                EVAL_EVERY, TOTAL_EPISODES - episodes_trained
            )

            var metrics = agent.train_gpu[HalfCheetahPlanarV2[gpu_dtype]](
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
            var current_log_std = List[Float64]()
            for i in range(ACTION_DIM):
                current_log_std.append(
                    Float64(agent.actor.params[log_std_offset + i])
                )

            # Evaluate on GPU (stochastic)
            var eval_gpu_avg = agent.evaluate_gpu[
                HalfCheetahPlanarV2[gpu_dtype]
            ](
                ctx,
                num_episodes=EVAL_EPISODES,
                max_steps=MAX_STEPS,
                verbose=False,
                stochastic=True,
            )

            # Evaluate on CPU (stochastic)
            var env = HalfCheetahPlanarV2[dtype]()
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
            log_std_values.append(current_log_std.copy())

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
                String(current_log_std[0])[:10].ljust(10),
                "|",
                String(current_log_std[1])[:10].ljust(10),
                "|",
                String(current_log_std[2])[:10].ljust(10),
                "|",
                String(current_log_std[3])[:10].ljust(10),
                "|",
                String(current_log_std[4])[:10].ljust(10),
                "|",
                String(current_log_std[5])[:10],
            )

            checkpoint_num += 1

        # Final summary
        print()
        print("=" * 130)
        print("GAP GROWTH ANALYSIS")
        print("=" * 130)
        print()

        print(
            "Chkpt | Episodes | Training | CPU Eval | GPU Eval | Gap(CPU) |"
            " Gap(GPU) | log_std[0] | log_std[1] | log_std[2] |"
            " log_std[3] | log_std[4] | log_std[5]"
        )
        print("-" * 150)
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
                String(log_std_values[i][0])[:10].rjust(10),
                "|",
                String(log_std_values[i][1])[:10].rjust(10),
                "|",
                String(log_std_values[i][2])[:10].rjust(10),
                "|",
                String(log_std_values[i][3])[:10].rjust(10),
                "|",
                String(log_std_values[i][4])[:10].rjust(10),
                "|",
                String(log_std_values[i][5])[:10].rjust(10),
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
            print("log_std Parameter Analysis:")
            for i in range(ACTION_DIM):
                var log_std_start = log_std_values[0][i]
                var log_std_end = log_std_values[len(log_std_values) - 1][i]
                print(
                    "  log_std[" + String(i) + "]: ",
                    log_std_start,
                    " -> ",
                    log_std_end,
                    " (change: ",
                    log_std_end - log_std_start,
                    ")",
                )
            print()

            # Check for log_std drift
            var any_high = False
            var any_low = False
            for i in range(ACTION_DIM):
                var log_std_end = log_std_values[len(log_std_values) - 1][i]
                if log_std_end > 2.0:
                    any_high = True
                if log_std_end < -5.0:
                    any_low = True

            if any_high:
                print("WARNING: Some log_std exceeded LOG_STD_MAX=2.0!")
                print(
                    "  This causes very high variance actions during training"
                )
                print("  but deterministic eval uses clamped values.")
            elif any_low:
                print("WARNING: Some log_std went below LOG_STD_MIN=-5.0!")
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
        print("=" * 130)

    print(">>> Gap growth analysis completed <<<")
