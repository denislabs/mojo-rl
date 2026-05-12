"""PPO GPU Training on Craftax-Classic.

End-to-end PPO on the Mojo port of Craftax-Classic. Feedforward MLP
(no recurrence) — same baseline used by Craftax_Baselines `ppo.py`.

Reference (paper / leaderboard, all on Craftax-Full not Classic):
  - PPO     11.9%  of max=226  (≈ 2.6 per-episode return at 1B steps)
  - PPO-RNN 15.3%
  - Random   ~ 0

On Classic (max return ≈ 22), our 20-update Apple smoke already hit
AvgR(100) = 3.46 / Best = 9.1, so the env, reward, obs, and policy
gradient are wired correctly. This config is sized for a serious GPU
run (~16M env steps) — expect convergence well above smoke values.

Run with:
    pixi run -e nvidia mojo run -I . examples/craftax_classic/ppo_training_gpu.mojo
    pixi run -e apple  mojo run -I . examples/craftax_classic/ppo_training_gpu.mojo   # slow
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepPPOAgent
from mojo_rl.envs.craftax_classic import CraftaxClassicEnv


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = CraftaxClassicEnv[DType.float32].OBS_DIM  # 1345
comptime NUM_ACTIONS = CraftaxClassicEnv[DType.float32].NUM_ACTIONS  # 17

# Network: wider than Pong because obs is mostly sparse one-hot.
comptime HIDDEN_DIM = 256

# PPO rollout shape — matches Craftax_Baselines defaults closely.
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 256              # parallel envs on GPU
comptime GPU_MINIBATCH_SIZE = 2048

# Run length. Each update = ROLLOUT_LEN * N_ENVS = 32,768 transitions.
# 500 updates ≈ 16M transitions — comfortably above the 10M Phase-5
# gate. Bump higher (e.g. 5000 → 160M) for a publication-grade run.
comptime NUM_UPDATES = 500

comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO GPU Training on Craftax-Classic (smoke)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = DeepPPOAgent[
            obs_dim=OBS_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dim=HIDDEN_DIM,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=GPU_MINIBATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.001,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=4,
            target_kl=0.015,
            max_grad_norm=0.5,
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=50,
            checkpoint_path="ppo_craftax_classic.ckpt",
        )

        var transitions_per_update = ROLLOUT_LEN * N_ENVS
        var total_transitions = transitions_per_update * NUM_UPDATES

        print("Environment: Craftax-Classic (GPU-batched)")
        print("Agent: PPO (feedforward MLP)")
        print("  Observation dim:", OBS_DIM)
        print("  Actions:", NUM_ACTIONS)
        print("  Hidden dim:", HIDDEN_DIM)
        print("  Rollout length:", ROLLOUT_LEN)
        print("  N envs (parallel):", N_ENVS)
        print("  Minibatch size:", GPU_MINIBATCH_SIZE)
        print("  Transitions per update:", transitions_per_update)
        print("  Total updates:", NUM_UPDATES)
        print("  Total transitions:", total_transitions)
        print()
        print("Reward shape: Σ Δachievements + 0.1 × Δhealth")
        print("  Random policy: typically 0 reward (no achievements reached)")
        print("  Smoke target:  > 0 (something gets discovered)")
        print("  Paper PPO:     ~2.6 over 1B steps (11.9% of max=22)")
        print()

        print("Starting GPU training...")
        print("-" * 70)
        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[CraftaxClassicEnv[dtype]](
                ctx,
                num_updates=NUM_UPDATES,
                verbose=True,
                print_every=10,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            print("-" * 70)
            print()
            print(">>> train_gpu returned successfully! <<<")

            print("=" * 70)
            print("GPU Training Complete")
            print("=" * 70)
            print("Total updates:", NUM_UPDATES)
            print("Total transitions:", total_transitions)
            print("Training time:", String(elapsed_s)[byte=:6], "seconds")
            print(
                "Transitions/second:",
                String(Float64(total_transitions) / elapsed_s)[byte=:9],
            )
            print()

            var final_avg = metrics.mean_reward_last_n(100)
            print(
                "Final average reward (last 100 episodes):",
                String(final_avg)[byte=:8],
            )
            print("Best episode reward:", String(metrics.max_reward())[byte=:8])
            print()

            if final_avg > 2.0:
                print("GREAT: agent discovering multiple achievements")
            elif final_avg > 0.5:
                print("LEARNING: at least one achievement per episode on average")
            elif final_avg > 0.05:
                print("EARLY SIGNAL: achievements triggered, scaling reward")
            elif final_avg > 0.0:
                print("MINIMAL: tiny reward — needs more updates")
            else:
                print("NO SIGNAL: agent hasn't found any achievements yet")
            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)

    print(">>> main() completed normally <<<")
