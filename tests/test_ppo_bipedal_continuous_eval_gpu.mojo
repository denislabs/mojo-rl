"""Quick GPU-only evaluation to verify continuous PPO checkpoint works on GPU environment.

This tests that the trained continuous PPO model performs well on the GPU environment
it was trained on.

Run with:
    pixi run -e apple mojo run tests/test_ppo_bipedal_continuous_eval_gpu.mojo
    pixi run -e nvidia mojo run tests/test_ppo_bipedal_continuous_eval_gpu.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.bipedal_walker import BipedalWalkerV2, BWConstants
from deep_rl import dtype as gpu_dtype


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = BWConstants.OBS_DIM_VAL  # 24
comptime ACTION_DIM = BWConstants.ACTION_DIM_VAL  # 4
comptime HIDDEN_DIM = 512
comptime ROLLOUT_LEN = 256
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512

# Evaluation settings
comptime EVAL_EPISODES = 100
comptime MAX_STEPS = 1600


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Evaluation on BipedalWalker")
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
            entropy_coef=0.02,  # Match training
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.1,  # Match training
            max_grad_norm=0.5,
            anneal_lr=False,  # Match training
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_bipedal_continuous_gpu.ckpt",
            normalize_rewards=True,  # Match training
        )

        print("Loading checkpoint...")
        try:
            agent.load_checkpoint("ppo_bipedal_continuous_gpu.ckpt")
            print("Checkpoint loaded successfully!")
        except:
            print("Error loading checkpoint!")
            print("Make sure you have trained the agent first:")
            print(
                "  pixi run -e apple mojo run"
                " tests/test_ppo_bipedal_continuous_gpu.mojo"
            )
            return

        # Show first and last actor parameters to verify checkpoint loaded
        print()
        print("Actor param diagnostics:")
        print("  Total params:", len(agent.actor.params))
        print("  First 5:", agent.actor.params[0], agent.actor.params[1],
              agent.actor.params[2], agent.actor.params[3], agent.actor.params[4])

        # log_std params are the last ACTION_DIM params
        var log_std_offset = len(agent.actor.params) - ACTION_DIM
        print("  log_std params:", agent.actor.params[log_std_offset],
              agent.actor.params[log_std_offset + 1],
              agent.actor.params[log_std_offset + 2],
              agent.actor.params[log_std_offset + 3])
        print()

        # =====================================================================
        # GPU Evaluation using built-in method
        # =====================================================================

        print("-" * 70)
        print("Running GPU evaluation (stochastic policy)...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        var stochastic_reward = agent.evaluate_gpu[BipedalWalkerV2[gpu_dtype]](
            ctx,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS,
            verbose=True,
            stochastic=True,
        )

        var stochastic_time = perf_counter_ns() - start_time

        print()
        print("-" * 70)
        print("Running GPU evaluation (deterministic policy)...")
        print("-" * 70)

        start_time = perf_counter_ns()

        var deterministic_reward = agent.evaluate_gpu[BipedalWalkerV2[gpu_dtype]](
            ctx,
            num_episodes=EVAL_EPISODES,
            max_steps=MAX_STEPS,
            verbose=False,
            stochastic=False,
        )

        var deterministic_time = perf_counter_ns() - start_time

        # =====================================================================
        # Results
        # =====================================================================

        print()
        print("=" * 70)
        print("GPU EVALUATION SUMMARY")
        print("=" * 70)
        print()
        print("Stochastic policy (sampling from distribution):")
        print("  Average reward:", String(stochastic_reward)[:10])
        print("  Time:", String(Float64(stochastic_time) / 1e9)[:6] + "s")
        print()
        print("Deterministic policy (using mean actions):")
        print("  Average reward:", String(deterministic_reward)[:10])
        print("  Time:", String(Float64(deterministic_time) / 1e9)[:6] + "s")
        print()

        print("BipedalWalker expected rewards:")
        print("  Random policy: ~-100 to -300")
        print("  Good policy: > 100")
        print("  Walking well: > 200")
        print("  Solved: > 300")
        print()

        if stochastic_reward > 300:
            print("EXCELLENT: Agent solved BipedalWalker!")
        elif stochastic_reward > 200:
            print("VERY GOOD: Agent is walking consistently!")
        elif stochastic_reward > 100:
            print("GOOD: Agent learned to walk!")
        elif stochastic_reward > 0:
            print("LEARNING: Agent shows progress")
        else:
            print("POOR: Needs more training or checkpoint is corrupted")

        print()
        print("=" * 70)

    print(">>> GPU Evaluation completed <<<")
