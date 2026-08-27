"""Train Q-Learning on Acrobot, then record a GIF of the trained agent.

Trains tabular Q-Learning with 6-bin discretization for 500 episodes,
then records 3 episodes of the trained agent swinging up.

Run with:
    pixi run mojo run -I . examples/acrobot/acrobot_demo_gif.mojo
"""

from mojo_rl.envs import AcrobotEnv
from mojo_rl.agents import QLearningAgent
from mojo_rl.core.fmt import fit


def main() raises:
    print("=" * 60)
    print("  Acrobot Q-Learning — Train + GIF Export")
    print("=" * 60)
    print()

    # ==========================================================================
    # Phase 1: Train
    # ==========================================================================
    var num_bins = 6
    var num_states = AcrobotEnv[DType.float64].get_num_states(num_bins)
    var max_steps = 500

    var env = AcrobotEnv[DType.float64](num_bins=num_bins)
    var agent = QLearningAgent(
        num_states=num_states,
        num_actions=env.num_actions(),
        learning_rate=0.1,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    print("Training Q-Learning for 2000 episodes...")
    var metrics = agent.train(
        env,
        num_episodes=2000,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="Acrobot",
    )

    var eval_reward = agent.evaluate(env, num_episodes=10)
    print()
    print("Eval avg reward:", fit(String(eval_reward), 8))
    print()

    # ==========================================================================
    # Phase 2: Record GIF of trained agent
    # ==========================================================================
    print("-" * 60)
    print("Recording trained agent to GIF...")
    print("-" * 60)

    _ = env.init_renderer()
    env.start_recording("gifs/acrobot_trained.gif", fps=15)

    for episode in range(3):
        _ = env.reset()
        var episode_reward: Float64 = 0.0
        var steps = 0

        for _ in range(max_steps):
            var state = env.get_state()
            var state_idx = env.state_to_index(state)
            var action_idx = agent.get_best_action(state_idx)
            var action = env.action_from_index(action_idx)

            var result = env.step(action)
            episode_reward += result[1]
            steps += 1

            env.render_frame()
            env.renderer_delay(16)

            if env.check_renderer_quit():
                break

            if result[2]:
                break

        print(
            "  Episode",
            episode + 1,
            "| Steps:",
            steps,
            "| Reward:",
            fit(String(episode_reward), 10),
        )

    env.stop_recording()
    env.close_renderer()

    print()
    print("Saved: gifs/acrobot_trained.gif")
    print("Done!")
