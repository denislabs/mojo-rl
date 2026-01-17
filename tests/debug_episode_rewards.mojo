"""Debug script to check episode rewards in CarRacing."""

from random import seed, random_float64
from envs.car_racing import CarRacingEnv


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Episode Reward Debug Test")
    print("=" * 70)

    # Create environment with max_steps=500 like the training test
    var env = CarRacingEnv[DType.float32](
        continuous=True,
        lap_complete_percent=0.95,
        domain_randomize=False,
        max_steps=500,  # Same as training test
    )

    # Run 10 episodes with random actions (simulating untrained policy)
    for ep in range(10):
        _ = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0
        var done = False

        while not done:
            # Random actions in [-1, 1] like the policy would output
            var action_list = List[Float64]()
            action_list.append(random_float64() * 2.0 - 1.0)  # steering
            action_list.append(random_float64() * 2.0 - 1.0)  # gas
            action_list.append(random_float64() * 2.0 - 1.0)  # brake

            var result = env.step_continuous_vec_f64(action_list^)
            var reward = result[1]
            done = result[2]
            total_reward += reward
            steps += 1

        print(
            "Episode "
            + String(ep + 1)
            + ": steps="
            + String(steps)
            + ", reward="
            + String(total_reward)
        )

    print()
    print("=" * 70)
    print("Now testing with 'do nothing' action (like untrained policy mean=0):")
    print("=" * 70)

    # Run 10 episodes with action=[0, 0, 0] (do nothing)
    # This simulates what an untrained policy with mean=0 would do
    for ep in range(10):
        _ = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0
        var done = False

        while not done:
            # Action [0, 0, 0] -> gas=0.5, brake=0.5 after remapping
            var action_list = List[Float64]()
            action_list.append(0.0)  # steering = 0
            action_list.append(0.0)  # gas = 0 -> 0.5 actual gas
            action_list.append(0.0)  # brake = 0 -> 0.5 actual brake

            var result = env.step_continuous_vec_f64(action_list^)
            var reward = result[1]
            done = result[2]
            total_reward += reward
            steps += 1

        print(
            "Episode "
            + String(ep + 1)
            + ": steps="
            + String(steps)
            + ", reward="
            + String(total_reward)
        )

    print()
    print("Debug complete!")
