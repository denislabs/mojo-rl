"""PPO CPU Training on Craftax-Full.

End-to-end PPO smoke run on the Mojo port of Craftax-Full. Feedforward MLP
(no recurrence) over the 8268-D symbolic observation. CPU-only — Full
Craftax does not yet expose GPU step/reset kernels (the path lifts to
`agent.train_gpu` once `GPUDiscreteEnv` conformance is added). Use the
Classic GPU script for fast experiments while that lands.

Reference (paper / leaderboard, all on Craftax-Full):
  - PPO     11.9%  of max=226  (≈ 27 per-episode return at 1B steps)
  - PPO-RNN 15.3%
  - Random  ~ 0    (achievement-shaped reward; random rarely scores)

This config is sized for a CPU smoke (~50 updates of length 128 = ~6 400
transitions per env, single env). Expect a small positive signal from a few
of the easy achievements (collect wood / drink / sapling). Bump `NUM_UPDATES`
or wire GPU kernels for a serious run.

Run:
    pixi run mojo run -I . examples/craftax_full/ppo_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepPPOAgent
from mojo_rl.envs.craftax_full import CraftaxFullEnv, OBS_DIM, NUM_ACTIONS

# Network: wider than Classic because the obs is much higher-D (8268 vs 1345).
comptime HIDDEN_DIM = 256

# PPO rollout shape — sized for CPU. Each update collects ROLLOUT_LEN steps
# from a single env. 50 updates × 128 steps = 6 400 env steps total.
comptime ROLLOUT_LEN = 128
comptime MINIBATCH_SIZE = 64

# Smoke budget. Each update ≈ ROLLOUT_LEN steps; on CPU each step does
# physics + symbolic obs encode (~0.5 ms), so 50 updates ≈ a few minutes.
comptime NUM_UPDATES = 50

comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO CPU Training on Craftax-Full (smoke)")
    print("=" * 70)
    print()

    var env = CraftaxFullEnv[dtype]()

    var agent = DeepPPOAgent[
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        rollout_len=ROLLOUT_LEN,
        actor_lr=0.0003,
        critic_lr=0.001,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        num_epochs=4,
        minibatch_size=MINIBATCH_SIZE,
        checkpoint_every=25,
        checkpoint_path="ppo_craftax_full.ckpt",
    )

    var total_transitions = ROLLOUT_LEN * NUM_UPDATES

    print("Environment: Craftax-Full (CPU)")
    print("Agent: PPO (feedforward MLP)")
    print("  Observation dim:", OBS_DIM)
    print("  Actions:", NUM_ACTIONS)
    print("  Hidden dim:", HIDDEN_DIM)
    print("  Rollout length:", ROLLOUT_LEN)
    print("  Minibatch size:", MINIBATCH_SIZE)
    print("  Total updates:", NUM_UPDATES)
    print("  Total transitions:", total_transitions)
    print()
    print("Reward shape: Σ tier-weighted Δachievements + 0.1 × Δhealth")
    print("  Random policy: typically 0 reward")
    print("  Smoke target:  > 0 (an easy achievement gets hit)")
    print("  Paper PPO:     ~27 over 1B steps (11.9% of max=226)")
    print()

    print("Starting CPU training...")
    print("-" * 70)
    var start_time = perf_counter_ns()

    try:
        var metrics = agent.train[CraftaxFullEnv[dtype]](
            env,
            num_updates=NUM_UPDATES,
            verbose=True,
            print_every=5,
            environment_name="Craftax-Full",
        )

        var end_time = perf_counter_ns()
        var elapsed_s = Float64(end_time - start_time) / 1e9

        print("-" * 70)
        print()
        print(">>> agent.train returned successfully! <<<")

        print("=" * 70)
        print("CPU Training Complete")
        print("=" * 70)
        print("Total updates:", NUM_UPDATES)
        print("Total transitions:", total_transitions)
        print("Training time:", String(elapsed_s)[byte=:6], "seconds")
        if elapsed_s > 0.0:
            print(
                "Transitions/second:",
                String(Float64(total_transitions) / elapsed_s)[byte=:9],
            )
        print()

        var final_avg = metrics.mean_reward_last_n(20)
        print(
            "Final average reward (last 20 episodes):",
            String(final_avg)[byte=:8],
        )
        print("Best episode reward:",
              String(metrics.max_reward())[byte=:8])
        print()

        if final_avg > 5.0:
            print("STRONG: agent stringing multiple achievements together")
        elif final_avg > 1.0:
            print("LEARNING: easy achievements consistent")
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
