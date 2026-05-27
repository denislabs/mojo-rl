"""Train Tiled Q-Learning on MountainCar, then record a GIF of the trained agent.

Trains for 3000 episodes with tile coding, then records 3 episodes
showing the car swinging back and forth to reach the goal.

Run with:
    pixi run mojo run -I . examples/mountain_car_demo_gif.mojo
"""

from mojo_rl.envs import MountainCarEnv
from mojo_rl.agents.tiled_qlearning import TiledQLearningAgent


def main() raises:
    print("=" * 60)
    print("  MountainCar Q-Learning — Train + GIF Export")
    print("=" * 60)
    print()

    # ==========================================================================
    # Phase 1: Train
    # ==========================================================================
    var tc = MountainCarEnv[DType.float64].make_tile_coding(
        num_tilings=8, tiles_per_dim=8
    )
    var env = MountainCarEnv[DType.float64]()
    var max_steps = 200

    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=env.num_actions(),
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    print("Training Tiled Q-Learning for 3000 episodes...")
    _ = agent.train(
        env,
        tc,
        num_episodes=3000,
        max_steps_per_episode=max_steps,
        verbose=True,
    )

    var eval_reward = agent.evaluate(env, tc, num_episodes=20, render=False)
    print()
    print("Eval avg steps:", Int(-eval_reward))
    print()

    # ==========================================================================
    # Phase 2: Record GIF of trained agent
    # ==========================================================================
    print("-" * 60)
    print("Recording trained agent to GIF...")
    print("-" * 60)

    _ = env.init_renderer()
    env.start_recording("gifs/mountain_car_trained.gif", fps=30)

    for episode in range(3):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64](capacity=len(obs_raw))
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0

        for _ in range(max_steps):
            var tiles = tc.get_tiles(obs)
            var action = agent.get_best_action(tiles)
            var result = env.step_obs(action)

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
            "| Steps:",
            Int(-episode_reward),
            "| Reward:",
            String(episode_reward)[byte=:10],
        )

    env.stop_recording()
    env.close_renderer()

    print()
    print("Saved: gifs/mountain_car_trained.gif")
    print("Done!")
