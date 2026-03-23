"""PPO CNN GPU Training on Pong with Pixel Observations.

Trains a PPO CNN agent on the native Pong environment using
pixel observations (4×84×84 stacked grayscale frames).

Run with:
    pixi run -e apple mojo run -I . examples/arcade_games/ppo_pong_pixel_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/arcade_games/ppo_pong_pixel_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepPPOCNNAgent
from mojo_rl.envs.arcade_games.pong import PongPixelEnv

# Pong: 3 discrete actions, pixel observations (4×84×84)
comptime NUM_ACTIONS = 3  # NOOP, UP, DOWN

# PPO hyperparameters
comptime N_ENVS = 64
comptime ROLLOUT_LEN = 128  # Steps per rollout per env
comptime MINIBATCH_SIZE = 256

# Training: num_updates × rollout_len × n_envs = total transitions
# 500 updates × 128 × 64 = 4,096,000 transitions
comptime NUM_UPDATES = 500

comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO CNN GPU Training on Pong — Pixel Observations")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = DeepPPOCNNAgent[
            num_actions=NUM_ACTIONS,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=MINIBATCH_SIZE,
            actor_lr=2.5e-4,
            critic_lr=2.5e-4,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=4,
            normalize_advantages=True,
            target_kl=0.015,
            max_grad_norm=0.5,
            clip_value=True,
            norm_adv_per_minibatch=True,
        )

        print("Environment: Pong (GPU-batched, Pixel)")
        print("Agent: PPO CNN (GPU)")
        print("  Observation: 4 × 84 × 84 = 28224 (pixel frames)")
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Network: Nature DQN CNN (actor + critic)")
        print("  N envs (parallel):", N_ENVS)
        print("  Rollout length:", ROLLOUT_LEN)
        print("  Minibatch size:", MINIBATCH_SIZE)
        print("  Num updates:", NUM_UPDATES)
        print(
            "  Total transitions:",
            NUM_UPDATES * ROLLOUT_LEN * N_ENVS,
        )
        print("  Learning rate: 2.5e-4 (actor + critic)")
        print("  PPO epochs: 4, clip: 0.2, GAE λ: 0.95")
        print()

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[PongPixelEnv[dtype]](
                ctx,
                num_updates=NUM_UPDATES,
                verbose=True,
                print_every=10,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9
            var total_transitions = NUM_UPDATES * ROLLOUT_LEN * N_ENVS

            print("-" * 70)
            print()
            print("=" * 70)
            print("GPU Training Complete")
            print("=" * 70)
            print()
            print("Total transitions:", total_transitions)
            print("Training time:", String(elapsed_s)[:6], "seconds")
            print(
                "Transitions/second:",
                String(Float64(total_transitions) / elapsed_s)[:9],
            )
            print()

            var final_avg = metrics.mean_reward_last_n(100)
            print(
                "Final average reward (last 100 episodes):",
                String(final_avg)[:8],
            )
            print("Best episode reward:", String(metrics.max_reward())[:8])
            print()

            if final_avg > 10.0:
                print("EXCELLENT: Agent dominates CPU! (avg reward > 10)")
            elif final_avg > 0.0:
                print("SUCCESS: Agent beats CPU! (avg reward > 0)")
            elif final_avg > -10.0:
                print("GOOD PROGRESS: Agent is competitive (avg reward > -10)")
            elif final_avg > -15.0:
                print("LEARNING: Agent improving (avg reward > -15)")
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < -15)")

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
