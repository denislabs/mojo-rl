"""Test native Pong environment — CPU path."""

from mojo_rl.envs.arcade_games.pong import PongEnv


fn main() raises:
    print("=== Testing PongEnv (CPU) ===")

    var env = PongEnv[DType.float64]()

    # Test reset
    var obs = env.reset_obs_list()
    print("Reset obs (", len(obs), "dims):")
    for i in range(len(obs)):
        print("  [", i, "]", obs[i])

    # Test stepping with random actions
    var total_reward: Float64 = 0.0
    var steps = 0

    for episode in range(3):
        _ = env.reset_obs_list()
        var ep_reward: Float64 = 0.0
        var ep_steps = 0

        while True:
            # Random action: 0=NOOP, 1=UP, 2=DOWN
            var action = Int(steps % 3)
            var result = env.step_obs(action)
            var obs_list = result[0].copy()
            var reward = result[1]
            var done = result[2]
            ep_reward += Float64(reward)
            ep_steps += 1
            steps += 1

            if done:
                print(
                    "Episode",
                    episode,
                    ": steps=",
                    ep_steps,
                    ", reward=",
                    ep_reward,
                )
                total_reward += ep_reward
                break

            # Safety: max steps per episode
            if ep_steps >= 15000:
                print("Episode", episode, ": hit max steps limit")
                total_reward += ep_reward
                break

    print("\nTotal steps:", steps, ", Total reward:", total_reward)
    print("Obs dim:", env.obs_dim())
    print("Num actions:", env.num_actions())
    print("=== DONE ===")
