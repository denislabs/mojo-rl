"""Debug script to check internal reward values."""

from random import seed
from envs.car_racing import CarRacingEnv


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Internal Reward Debug Test")
    print("=" * 70)

    var env = CarRacingEnv[DType.float32](
        continuous=True,
        lap_complete_percent=0.95,
        domain_randomize=False,
        max_steps=20,
    )

    _ = env.reset()

    print("\nStep-by-step reward tracking:")
    print("-" * 50)

    var total_reward: Float64 = 0.0

    for i in range(20):
        var action = List[Float64]()
        action.append(0.0)   # steering
        action.append(1.0)   # gas
        action.append(-1.0)  # brake

        var result = env.step_continuous_vec_f64(action^)
        var step_reward = result[1]
        total_reward += step_reward

        print(
            "Step "
            + String(i + 1)
            + ": step_reward="
            + String(step_reward)
            + ", total="
            + String(total_reward)
        )

    print()
    print("Final total reward: " + String(total_reward))
