"""Quick GPU-only evaluation to verify continuous PPO checkpoint works on Ant.

This tests that the trained continuous PPO model performs well on the GPU environment
it was trained on using the Generalized Coordinates (GC) physics engine.

Run with:
    pixi run -e apple mojo run -I . tests/test_ppo_ant_continuous_eval_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/test_ppo_ant_continuous_eval_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepPPOContinuousAgent
from mojo_rl.envs.ant import Ant, AntConfig
from mojo_rl.nn import dtype as gpu_dtype


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = AntConfig.OBS_DIM  # 27
comptime ACTION_DIM = AntConfig.ACTION_DIM  # 8
# Must match training configuration!
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

# Evaluation settings
comptime EVAL_EPISODES = 100
comptime MAX_STEPS = 1000  # Ant MAX_STEPS


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Evaluation on Ant")
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
            actor_lr=0.0003,
            critic_lr=0.0003,
        ](
            clip_value=True,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.0,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.0,
            max_grad_norm=0.5,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_ant.ckpt",
        )

        print("Loading checkpoint...")
        try:
            agent.load_checkpoint("ppo_ant.ckpt")
            print("Checkpoint loaded successfully!")
        except:
            print("Error loading checkpoint!")
            print("Make sure you have trained the agent first:")
            print(
                "  pixi run -e apple mojo run"
                " tests/test_ppo_ant_continuous_gpu.mojo"
            )
            return

        # =====================================================================
        # GPU Evaluation using built-in method
        # =====================================================================

        print("-" * 70)
        print("Running GPU evaluation (stochastic policy)...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        var stochastic_reward = agent.evaluate_gpu[Ant[gpu_dtype]](
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

        var deterministic_reward = agent.evaluate_gpu[Ant[gpu_dtype]](
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
        print("GPU EVALUATION SUMMARY - Ant")
        print("=" * 70)
        print()
        print("Stochastic policy (sampling from distribution):")
        print("  Average reward:", String(stochastic_reward)[byte=:10])
        print("  Time:", String(Float64(stochastic_time) / 1e9)[byte=:6] + "s")
        print()
        print("Deterministic policy (using mean actions):")
        print("  Average reward:", String(deterministic_reward)[byte=:10])
        print("  Time:", String(Float64(deterministic_time) / 1e9)[byte=:6] + "s")
        print()

        print("Ant expected rewards:")
        print("  Random policy: ~-100 to -200")
        print("  Learning policy: > 0")
        print("  Good policy: > 500")
        print("  Running well: > 1000")
        print("  Excellent: > 2000")
        print()

        if stochastic_reward > 2000:
            print("EXCELLENT: Agent is running very fast!")
        elif stochastic_reward > 1000:
            print("VERY GOOD: Agent is running well!")
        elif stochastic_reward > 500:
            print("GOOD: Agent learned to run!")
        elif stochastic_reward > 0:
            print("LEARNING: Agent shows progress")
        else:
            print("POOR: Needs more training or checkpoint is corrupted")

        print()
        print("=" * 70)

    print(">>> GPU Evaluation completed <<<")
