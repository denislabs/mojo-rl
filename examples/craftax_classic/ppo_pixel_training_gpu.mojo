"""PPO-CNN GPU training on Craftax-Classic (pixel observations).

Same agent as `ppo_pong_pixel_training_gpu.mojo` but pointed at
`CraftaxClassicPixelEnv`. Obs = 4 × 84 × 84 grayscale frame stack,
17 discrete actions, 22 sparse achievements.

Reference (Craftax-Full, not Classic):
  - PPO     11.9 % of max=226   (~2.6 per-episode return at 1B steps)
  - PPO-RNN 15.3 %
  - Random   ~ 0

On Classic (max return ≈ 22) we expect modest progress within 16M steps;
this is primarily an env / pipeline smoke test for the pixel path.

Run with:
    pixi run -e nvidia mojo run -I . examples/craftax_classic/ppo_pixel_training_gpu.mojo
    pixi run -e apple  mojo run -I . examples/craftax_classic/ppo_pixel_training_gpu.mojo   # slow
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepPPOCNNAgent
from mojo_rl.envs.craftax_classic import CraftaxClassicPixelEnv

# Craftax-Classic: 17 discrete actions, 4×84×84 pixel obs
comptime NUM_ACTIONS = 17

# PPO hyperparameters — same shape as Pong pixel for consistency.
comptime N_ENVS = 64
comptime ROLLOUT_LEN = 128
comptime MINIBATCH_SIZE = 256

# 500 updates × 128 × 64 = ~4M transitions (smoke). Bump higher for serious runs.
comptime NUM_UPDATES = 500

comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO-CNN GPU training on Craftax-Classic — pixel obs")
    print("=" * 70)

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

        print("Environment: Craftax-Classic (GPU-batched, pixel)")
        print("Agent: PPO-CNN (Nature DQN trunk)")
        print("  Observation: 4 × 84 × 84 = 28224 (pixel frames)")
        print("  Actions:", NUM_ACTIONS)
        print("  N envs (parallel):", N_ENVS)
        print("  Rollout length:", ROLLOUT_LEN)
        print("  Minibatch size:", MINIBATCH_SIZE)
        print("  Num updates:", NUM_UPDATES)
        print(
            "  Total transitions:",
            NUM_UPDATES * ROLLOUT_LEN * N_ENVS,
        )
        print()

        var start = perf_counter_ns()
        try:
            var metrics = agent.train_gpu[CraftaxClassicPixelEnv[dtype]](
                ctx,
                num_updates=NUM_UPDATES,
                verbose=True,
                print_every=10,
            )

            var elapsed_s = Float64(perf_counter_ns() - start) / 1e9
            var total = NUM_UPDATES * ROLLOUT_LEN * N_ENVS
            print("-" * 70)
            print("Training time:", String(elapsed_s)[byte=:6], "s")
            print(
                "Transitions/sec:",
                String(Float64(total) / elapsed_s)[byte=:9],
            )

            var final_avg = metrics.mean_reward_last_n(100)
            print(
                "Final avg reward (last 100):",
                String(final_avg)[byte=:8],
            )
            print("Best episode:", String(metrics.max_reward())[byte=:8])

            if final_avg > 2.0:
                print("GREAT: multi-achievement runs")
            elif final_avg > 0.5:
                print("LEARNING: >=1 achievement / episode on average")
            elif final_avg > 0.05:
                print("EARLY SIGNAL: achievement triggered, scaling reward")
            elif final_avg > 0.0:
                print("MINIMAL: small reward, needs more updates")
            else:
                print("NO SIGNAL: agent has not found achievements yet")

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
    print(">>> main() completed <<<")
