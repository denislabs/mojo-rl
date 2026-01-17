"""Debug script to check reward variation in CarRacing."""

from random import seed, random_float64
from envs.car_racing import CarRacingEnv


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Reward Variation Debug Test")
    print("=" * 70)

    var env = CarRacingEnv[DType.float32](
        continuous=True,
        lap_complete_percent=0.95,
        domain_randomize=False,
        max_steps=100,
    )

    # Collect rewards for 100 episodes
    var rewards = List[Float64]()
    var steps_list = List[Int]()

    for ep in range(100):
        _ = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0
        var done = False

        while not done:
            # Fixed "go straight" action
            var action = List[Float64]()
            action.append(0.0)   # steering
            action.append(1.0)   # gas
            action.append(-1.0)  # brake

            var result = env.step_continuous_vec_f64(action^)
            total_reward += result[1]
            done = result[2]
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    # Compute statistics
    var min_reward = rewards[0]
    var max_reward = rewards[0]
    var sum_reward: Float64 = 0.0

    for i in range(len(rewards)):
        sum_reward += rewards[i]
        if rewards[i] < min_reward:
            min_reward = rewards[i]
        if rewards[i] > max_reward:
            max_reward = rewards[i]

    var mean_reward = sum_reward / Float64(len(rewards))

    print("Statistics over 100 episodes:")
    print("  Mean reward: " + String(mean_reward))
    print("  Min reward:  " + String(min_reward))
    print("  Max reward:  " + String(max_reward))
    print("  Range:       " + String(max_reward - min_reward))
    print()

    # Print first 10 individual rewards
    print("First 10 episode rewards:")
    for i in range(min(10, len(rewards))):
        print(
            "  Episode "
            + String(i + 1)
            + ": "
            + String(rewards[i])
            + " (steps: "
            + String(steps_list[i])
            + ")"
        )
