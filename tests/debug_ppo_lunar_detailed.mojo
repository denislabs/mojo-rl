"""Detailed debug script to compare training vs evaluation episode distributions.

This script does a single training batch and compares the episode reward
distributions between training and evaluation to identify the source of the gap.

Run with:
    pixi run -e apple mojo run tests/debug_ppo_lunar_detailed.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.lunar_lander import LunarLanderV2, LLConstants
from deep_rl import dtype as gpu_dtype


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512

comptime dtype = DType.float32

# Quick training for diagnosis
comptime TRAIN_EPISODES = 2000
comptime EVAL_EPISODES = 100
comptime MAX_STEPS = 1000


fn categorize_reward(reward: Float64) -> String:
    """Categorize episode based on final reward."""
    if reward > 100:
        return "GREAT_LAND"  # Landed with bonus
    elif reward > 0:
        return "OK_LAND"  # Positive but not great
    elif reward > -100:
        return "HOVER"  # Didn't crash, didn't land
    elif reward > -200:
        return "CRASH"  # Crashed
    else:
        return "BAD_CRASH"  # Very bad crash


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DETAILED DEBUG: Train vs Eval Episode Distribution")
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
            entropy_coef=0.05,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.1,
            max_grad_norm=0.5,
            anneal_lr=False,
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=0,
            checkpoint_path="",
            normalize_rewards=True,
        )

        # Get log_std offset
        var num_actor_params = len(agent.actor.params)
        var log_std_offset = num_actor_params - ACTION_DIM

        print("Initial log_std params:")
        print("  log_std[0]:", agent.actor.params[log_std_offset])
        print("  log_std[1]:", agent.actor.params[log_std_offset + 1])
        print()

        # =====================================================================
        # Train
        # =====================================================================
        print("-" * 70)
        print("TRAINING for", TRAIN_EPISODES, "episodes...")
        print("-" * 70)

        var metrics = agent.train_gpu[LunarLanderV2[gpu_dtype]](
            ctx,
            num_episodes=TRAIN_EPISODES,
            verbose=True,
            print_every=10,  # Print every 10 rollouts
        )

        print()
        print("Training log_std params after training:")
        print("  log_std[0]:", agent.actor.params[log_std_offset])
        print("  log_std[1]:", agent.actor.params[log_std_offset + 1])
        print()

        # =====================================================================
        # Analyze training episode distribution
        # =====================================================================
        print("-" * 70)
        print("TRAINING EPISODE ANALYSIS (last 100 episodes)")
        print("-" * 70)

        var train_rewards = metrics.get_rewards()
        var train_n = min(100, len(train_rewards))
        var train_start = len(train_rewards) - train_n

        var train_great = 0
        var train_ok = 0
        var train_hover = 0
        var train_crash = 0
        var train_bad = 0
        var train_sum: Float64 = 0.0
        var train_min: Float64 = 9999.0
        var train_max: Float64 = -9999.0

        print("Sample of last 20 training episodes:")
        for i in range(train_start, len(train_rewards)):
            var r = train_rewards[i]
            train_sum += r
            if r < train_min:
                train_min = r
            if r > train_max:
                train_max = r

            var cat = categorize_reward(r)
            if cat == "GREAT_LAND":
                train_great += 1
            elif cat == "OK_LAND":
                train_ok += 1
            elif cat == "HOVER":
                train_hover += 1
            elif cat == "CRASH":
                train_crash += 1
            else:
                train_bad += 1

            # Print last 20
            if i >= len(train_rewards) - 20:
                print(
                    "  Episode",
                    i + 1,
                    ":",
                    String(r)[:10].ljust(12),
                    cat,
                )

        var train_avg = train_sum / Float64(train_n)
        print()
        print("Training statistics (last", train_n, "episodes):")
        print("  Average:", train_avg)
        print("  Min:", train_min)
        print("  Max:", train_max)
        print("  Distribution:")
        print("    GREAT_LAND (>100):", train_great)
        print("    OK_LAND (0-100):", train_ok)
        print("    HOVER (-100 to 0):", train_hover)
        print("    CRASH (-200 to -100):", train_crash)
        print("    BAD_CRASH (<-200):", train_bad)
        print()

        # =====================================================================
        # GPU Evaluation
        # =====================================================================
        print("-" * 70)
        print("GPU EVALUATION for", EVAL_EPISODES, "episodes (stochastic)...")
        print("-" * 70)

        # We need to track individual episode rewards, so let's use a custom eval loop
        # that mirrors what evaluate_gpu does but with more logging.

        # For simplicity, use the built-in method first
        var gpu_eval_avg = agent.evaluate_gpu[LunarLanderV2[gpu_dtype]](
            ctx,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS,
            verbose=False,
            stochastic=True,
        )

        print("GPU eval average (stochastic):", gpu_eval_avg)
        print()

        # =====================================================================
        # CPU Evaluation with detailed logging
        # =====================================================================
        print("-" * 70)
        print("CPU EVALUATION for", EVAL_EPISODES, "episodes...")
        print("-" * 70)

        var env = LunarLanderV2[dtype]()

        var cpu_great = 0
        var cpu_ok = 0
        var cpu_hover = 0
        var cpu_crash = 0
        var cpu_bad = 0
        var cpu_sum: Float64 = 0.0
        var cpu_min: Float64 = 9999.0
        var cpu_max: Float64 = -9999.0

        print("Sample of CPU evaluation episodes:")
        for episode in range(EVAL_EPISODES):
            var obs_list = env.reset_obs_list()
            var obs = InlineArray[Scalar[dtype], OBS_DIM](uninitialized=True)
            for i in range(OBS_DIM):
                obs[i] = Scalar[dtype](obs_list[i])

            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(MAX_STEPS):
                var action_result = agent.select_action(obs, training=True)
                var actions = action_result[0].copy()

                var action_list = List[Scalar[dtype]]()
                for j in range(ACTION_DIM):
                    var action_val = Float64(actions[j])
                    if action_val > 1.0:
                        action_val = 1.0
                    elif action_val < -1.0:
                        action_val = -1.0
                    action_list.append(Scalar[dtype](action_val))

                var result = env.step_continuous_vec[dtype](action_list)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                episode_reward += Float64(reward)
                episode_steps += 1

                for i in range(OBS_DIM):
                    obs[i] = next_obs_list[i]

                if done:
                    break

            cpu_sum += episode_reward
            if episode_reward < cpu_min:
                cpu_min = episode_reward
            if episode_reward > cpu_max:
                cpu_max = episode_reward

            var cat = categorize_reward(episode_reward)
            if cat == "GREAT_LAND":
                cpu_great += 1
            elif cat == "OK_LAND":
                cpu_ok += 1
            elif cat == "HOVER":
                cpu_hover += 1
            elif cat == "CRASH":
                cpu_crash += 1
            else:
                cpu_bad += 1

            # Print first 20 and every 10th after that
            if episode < 20 or episode % 10 == 0:
                print(
                    "  Episode",
                    episode + 1,
                    ":",
                    String(episode_reward)[:10].ljust(12),
                    "steps:",
                    episode_steps,
                    cat,
                )

        var cpu_avg = cpu_sum / Float64(EVAL_EPISODES)
        print()
        print("CPU evaluation statistics:")
        print("  Average:", cpu_avg)
        print("  Min:", cpu_min)
        print("  Max:", cpu_max)
        print("  Distribution:")
        print("    GREAT_LAND (>100):", cpu_great)
        print("    OK_LAND (0-100):", cpu_ok)
        print("    HOVER (-100 to 0):", cpu_hover)
        print("    CRASH (-200 to -100):", cpu_crash)
        print("    BAD_CRASH (<-200):", cpu_bad)
        print()

        # =====================================================================
        # Summary
        # =====================================================================
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print("Training (last 100):", String(train_avg)[:10])
        print("GPU Eval (stochastic):", String(gpu_eval_avg)[:10])
        print("CPU Eval (stochastic):", String(cpu_avg)[:10])
        print()
        print("Gap (Train - CPU):", String(train_avg - cpu_avg)[:10])
        print("Gap (Train - GPU):", String(train_avg - gpu_eval_avg)[:10])
        print()

        print("Distribution comparison:")
        print("                 Train   CPU-Eval")
        print("  GREAT_LAND    ", String(train_great).rjust(5), " ", String(cpu_great).rjust(5))
        print("  OK_LAND       ", String(train_ok).rjust(5), " ", String(cpu_ok).rjust(5))
        print("  HOVER         ", String(train_hover).rjust(5), " ", String(cpu_hover).rjust(5))
        print("  CRASH         ", String(train_crash).rjust(5), " ", String(cpu_crash).rjust(5))
        print("  BAD_CRASH     ", String(train_bad).rjust(5), " ", String(cpu_bad).rjust(5))
        print()

        if train_great > cpu_great * 2:
            print("OBSERVATION: Training reports many more GREAT_LAND episodes than eval!")
            print("This suggests training rewards may be inflated.")
        elif cpu_crash > train_crash * 2:
            print("OBSERVATION: CPU eval has many more CRASH episodes than training!")
            print("This suggests training may not be logging crashes properly.")
        else:
            print("OBSERVATION: Distributions are similar, gap source unclear.")

        print()
        print("=" * 70)

    print(">>> Detailed debug completed <<<")
