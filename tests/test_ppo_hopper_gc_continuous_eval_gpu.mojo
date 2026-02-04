"""Quick GPU-only evaluation to verify continuous PPO checkpoint works on HopperGC.

This tests that the trained continuous PPO model performs well on the GPU environment
it was trained on.

Run with:
    pixi run -e apple mojo run tests/test_ppo_hopper_gc_continuous_eval_gpu.mojo
    pixi run -e nvidia mojo run tests/test_ppo_hopper_gc_continuous_eval_gpu.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.hopper_gc import HopperGC
from envs.hopper_gc.constants_gc import HopperGCConstantsGPU
from deep_rl import dtype as gpu_dtype


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = HopperGCConstantsGPU.OBS_DIM  # 11
comptime ACTION_DIM = HopperGCConstantsGPU.ACTION_DIM  # 3
# Must match training configuration!
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

# Evaluation settings
comptime EVAL_EPISODES = 100
comptime MAX_STEPS = 1000  # HopperGC MAX_STEPS


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Evaluation on HopperGC")
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
            critic_lr=0.0003,
            entropy_coef=0.0,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.0,
            max_grad_norm=0.5,
            anneal_lr=False,
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_hopper_gc.ckpt",
            normalize_rewards=True,
        )

        print("Loading checkpoint...")
        try:
            agent.load_checkpoint("ppo_hopper_gc.ckpt")
            print("Checkpoint loaded successfully!")
        except:
            print("Error loading checkpoint!")
            print("Make sure you have trained the agent first:")
            print(
                "  pixi run -e apple mojo run"
                " tests/test_ppo_hopper_gc_continuous_gpu.mojo"
            )
            return

        # Show first and last actor parameters to verify checkpoint loaded
        print()
        print("Actor param diagnostics:")
        print("  Total params:", len(agent.actor.params))
        print(
            "  First 5:",
            agent.actor.params[0],
            agent.actor.params[1],
            agent.actor.params[2],
            agent.actor.params[3],
            agent.actor.params[4],
        )

        # log_std params are the last ACTION_DIM params
        var log_std_offset = len(agent.actor.params) - ACTION_DIM
        print(
            "  log_std params:",
            agent.actor.params[log_std_offset],
            agent.actor.params[log_std_offset + 1],
            agent.actor.params[log_std_offset + 2],
        )
        print()

        # =====================================================================
        # GPU Evaluation using built-in method
        # =====================================================================

        print("-" * 70)
        print("Running GPU evaluation (stochastic policy)...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        var stochastic_reward = agent.evaluate_gpu[HopperGC[gpu_dtype]](
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

        var deterministic_reward = agent.evaluate_gpu[HopperGC[gpu_dtype]](
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
        print("GPU EVALUATION SUMMARY - HopperGC")
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

        print("HopperGC expected rewards:")
        print("  Random policy: ~-500 to -100")
        print("  Learning policy: > 0")
        print("  Good policy: > 500")
        print("  Hopping well: > 1000")
        print("  Excellent: > 2000")
        print()

        if stochastic_reward > 2000:
            print("EXCELLENT: Agent is hopping very well!")
        elif stochastic_reward > 1000:
            print("VERY GOOD: Agent is hopping well!")
        elif stochastic_reward > 500:
            print("GOOD: Agent learned to hop!")
        elif stochastic_reward > 0:
            print("LEARNING: Agent shows progress")
        else:
            print("POOR: Needs more training or checkpoint is corrupted")

        print()
        print("=" * 70)

    print(">>> GPU Evaluation completed <<<")
