"""Debug script to verify actions are being applied correctly."""

from random import seed
from envs.car_racing import CarRacingEnv


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Action Application Debug Test")
    print("=" * 70)

    var env = CarRacingEnv[DType.float32](
        continuous=True,
        lap_complete_percent=0.95,
        domain_randomize=False,
        max_steps=50,  # Short for debugging
    )

    # Test 1: Action [0, 0, 0] (do nothing - gas=0.5, brake=0.5)
    print("\nTest 1: Action [0, 0, 0] (gas=0.5, brake=0.5)")
    print("-" * 50)
    _ = env.reset()
    for i in range(10):
        var action = List[Float64]()
        action.append(0.0)  # steering
        action.append(0.0)  # gas (-> 0.5)
        action.append(0.0)  # brake (-> 0.5)
        var result = env.step_continuous_vec_f64(action^)
        var obs = result[0].copy()
        print(
            "Step "
            + String(i + 1)
            + ": vx="
            + String(obs[2])[:8]
            + ", vy="
            + String(obs[3])[:8]
            + ", reward="
            + String(result[1])[:8]
        )

    # Test 2: Action [0, 1, -1] (full gas, no brake)
    print("\nTest 2: Action [0, 1, -1] (full gas, no brake)")
    print("-" * 50)
    _ = env.reset()
    for i in range(10):
        var action = List[Float64]()
        action.append(0.0)   # steering
        action.append(1.0)   # gas (-> 1.0)
        action.append(-1.0)  # brake (-> 0.0)
        var result = env.step_continuous_vec_f64(action^)
        var obs = result[0].copy()
        print(
            "Step "
            + String(i + 1)
            + ": vx="
            + String(obs[2])[:8]
            + ", vy="
            + String(obs[3])[:8]
            + ", reward="
            + String(result[1])[:8]
        )

    # Test 3: Action [0, -1, 1] (no gas, full brake)
    print("\nTest 3: Action [0, -1, 1] (no gas, full brake)")
    print("-" * 50)
    _ = env.reset()
    for i in range(10):
        var action = List[Float64]()
        action.append(0.0)   # steering
        action.append(-1.0)  # gas (-> 0.0)
        action.append(1.0)   # brake (-> 1.0)
        var result = env.step_continuous_vec_f64(action^)
        var obs = result[0].copy()
        print(
            "Step "
            + String(i + 1)
            + ": vx="
            + String(obs[2])[:8]
            + ", vy="
            + String(obs[3])[:8]
            + ", reward="
            + String(result[1])[:8]
        )

    # Test 4: Action with steering
    print("\nTest 4: Action [1, 0.5, -1] (steer right, half gas)")
    print("-" * 50)
    _ = env.reset()
    for i in range(10):
        var action = List[Float64]()
        action.append(1.0)   # steering right
        action.append(0.5)   # gas (-> 0.75)
        action.append(-1.0)  # brake (-> 0.0)
        var result = env.step_continuous_vec_f64(action^)
        var obs = result[0].copy()
        print(
            "Step "
            + String(i + 1)
            + ": angle="
            + String(obs[4])[:8]
            + ", vx="
            + String(obs[2])[:8]
            + ", vy="
            + String(obs[3])[:8]
        )

    print()
    print("Debug complete!")
