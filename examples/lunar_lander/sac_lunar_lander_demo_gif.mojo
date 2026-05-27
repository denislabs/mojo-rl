"""Train SAC on LunarLander Continuous (CPU), then record a GIF.

Run with:
    pixi run mojo run -I . examples/lunar_lander/sac_lunar_lander_demo_gif.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.lunar_lander import LunarLander


comptime OBS_DIM = 8
comptime ACTION_DIM = 2
comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 64
comptime NUM_STEPS = 500_000
comptime MAX_STEPS_PER_EPISODE = 1000
comptime WARMUP_STEPS = 5_000
comptime dtype = DType.float32


def main() raises:
    seed(42)

    # ==========================================================================
    # Phase 1: Train
    # ==========================================================================
    print("=" * 60)
    print("  LunarLander SAC — Train + GIF Export")
    print("=" * 60)
    print()

    var env = LunarLander[dtype]()

    var agent = DeepSACAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        actor_lr=0.0003,
        critic_lr=0.001,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=0.0003,
        target_entropy=-2.0,
        use_ere=True,
        ere_eta=0.996,
    )

    print("Training SAC for", NUM_STEPS, "steps...")
    var start_ns = perf_counter_ns()

    var metrics = agent.train(
        env,
        num_steps=NUM_STEPS,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        warmup_steps=WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=10_000,
        environment_name="LunarLander",
    )

    var elapsed_s = Float64(perf_counter_ns() - start_ns) / 1e9
    var final_avg = metrics.mean_reward_last_n(100)
    print()
    print("Training done in", String(elapsed_s)[byte=:6], "seconds")
    print("Final avg reward (last 100):", String(final_avg)[byte=:10])
    print()

    # ==========================================================================
    # Phase 2: Record GIF of trained agent
    # ==========================================================================
    print("-" * 60)
    print("Recording trained agent to GIF...")
    print("-" * 60)

    _ = env.init_renderer()
    env.start_recording("gifs/lunar_lander_sac_trained.gif", fps=30, skip=3)

    for episode in range(3):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64](capacity=OBS_DIM)
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            var action = agent.select_greedy_action_obs(obs)
            var result = env.step_continuous_vec(action)

            episode_reward += Float64(result[1])

            env.render_frame()
            env.renderer_delay(16)

            if env.check_renderer_quit():
                break

            if result[2]:
                break

            obs.clear()
            for i in range(len(result[0])):
                obs.append(Float64(result[0][i]))

        print(
            "  Episode",
            episode + 1,
            "| Reward:",
            String(episode_reward)[byte=:10],
        )

    env.stop_recording()
    env.close_renderer()

    print()
    print("Saved: gifs/lunar_lander_sac_trained.gif")
    print("Done!")
