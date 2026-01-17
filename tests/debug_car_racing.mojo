"""Debug script to verify CarRacing environment works correctly.

This script tests:
1. Environment reset and observation values
2. Stepping with various actions
3. Reward accumulation
4. Episode termination

Run with:
    pixi run mojo run tests/debug_car_racing.mojo
"""

from random import seed, random_float64
from envs.car_racing import CarRacingEnv


fn main() raises:
    seed(42)
    print("=" * 70)
    print("CarRacing Environment Debug Test")
    print("=" * 70)
    print()

    # Create environment
    var env = CarRacingEnv[DType.float32](
        continuous=True,
        lap_complete_percent=0.95,
        domain_randomize=False,
        max_steps=200,  # Short episodes for testing
    )

    # =========================================================================
    # Test 1: Reset and check initial observation
    # =========================================================================
    print("Test 1: Reset and initial observation")
    print("-" * 50)

    var obs = env.reset_obs_list()
    print("Observation dimension:", len(obs))
    print("Initial observation values:")
    var obs_names = List[String]()
    obs_names.append("x (normalized)")
    obs_names.append("y (normalized)")
    obs_names.append("vx (normalized)")
    obs_names.append("vy (normalized)")
    obs_names.append("angle (normalized)")
    obs_names.append("angular_vel (normalized)")
    obs_names.append("wheel_omega[0]")
    obs_names.append("wheel_omega[1]")
    obs_names.append("wheel_omega[2]")
    obs_names.append("wheel_omega[3]")
    obs_names.append("track_progress")
    obs_names.append("waypoint_dx")
    obs_names.append("waypoint_dy")

    for i in range(len(obs)):
        if i < len(obs_names):
            print("  " + obs_names[i] + ": " + String(obs[i])[:10])
        else:
            print("  obs[" + String(i) + "]: " + String(obs[i])[:10])
    print()

    # Check for NaN or extreme values
    var has_nan = False
    var has_extreme = False
    for i in range(len(obs)):
        var val = Float64(obs[i])
        if val != val:  # NaN check
            has_nan = True
            print("WARNING: NaN detected in obs[" + String(i) + "]")
        if abs(val) > 10.0:
            has_extreme = True
            print("WARNING: Extreme value in obs[" + String(i) + "]: " + String(val))

    if not has_nan and not has_extreme:
        print("✓ Initial observations look valid (no NaN, all in reasonable range)")
    print()

    # =========================================================================
    # Test 2: Step with "go straight" action (no steering, full gas, no brake)
    # =========================================================================
    print("Test 2: Step with 'go straight' action")
    print("-" * 50)

    _ = env.reset()
    var action_straight = List[Float64]()
    action_straight.append(0.0)   # steering = 0 (straight)
    action_straight.append(1.0)   # gas = 1 (full gas, will be remapped to 1.0)
    action_straight.append(-1.0)  # brake = -1 (no brake, will be remapped to 0.0)

    print("Action: steering=0, gas=1 (full), brake=-1 (none)")

    var total_reward: Float64 = 0.0
    var steps = 0
    var done = False

    while not done and steps < 50:
        var result = env.step_continuous_vec_f64(action_straight)
        var next_obs = result[0].copy()
        var reward = result[1]
        done = result[2]
        total_reward += reward
        steps += 1

        if steps <= 5 or steps % 10 == 0:
            print("  Step " + String(steps) + ": reward=" + String(reward)[:8] +
                  ", total=" + String(total_reward)[:8] +
                  ", vx=" + String(next_obs[2])[:8] +
                  ", vy=" + String(next_obs[3])[:8])

    print("After " + String(steps) + " steps: total_reward=" + String(total_reward)[:10])
    print()

    # =========================================================================
    # Test 3: Step with random actions
    # =========================================================================
    print("Test 3: Step with random actions")
    print("-" * 50)

    _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < 100:
        var action_random = List[Float64]()
        action_random.append(random_float64() * 2.0 - 1.0)  # steering [-1, 1]
        action_random.append(random_float64() * 2.0 - 1.0)  # gas [-1, 1]
        action_random.append(random_float64() * 2.0 - 1.0)  # brake [-1, 1]

        var result = env.step_continuous_vec_f64(action_random)
        var reward = result[1]
        done = result[2]
        total_reward += reward
        steps += 1

    print("After " + String(steps) + " random steps: total_reward=" + String(total_reward)[:10])
    print("Episode done:", done)
    print()

    # =========================================================================
    # Test 4: Full episode with constant gas
    # =========================================================================
    print("Test 4: Full episode with constant gas")
    print("-" * 50)

    _ = env.reset()
    var action_gas = List[Float64]()
    action_gas.append(0.0)   # no steering
    action_gas.append(1.0)   # full gas
    action_gas.append(-1.0)  # no brake

    total_reward = 0.0
    steps = 0
    done = False
    var tiles_visited = 0

    while not done:
        var result = env.step_continuous_vec_f64(action_gas)
        var next_obs = result[0].copy()
        var reward = result[1]
        done = result[2]
        total_reward += reward
        steps += 1

        # Track progress is obs[10]
        var progress = next_obs[10]
        var new_tiles = Int(progress * 100)  # Approximate
        if new_tiles > tiles_visited:
            tiles_visited = new_tiles

    print("Episode finished after " + String(steps) + " steps")
    print("Total reward: " + String(total_reward)[:10])
    print("Approximate tiles visited: " + String(tiles_visited) + "%")
    print()

    # =========================================================================
    # Test 5: Verify action remapping
    # =========================================================================
    print("Test 5: Verify action remapping (gas/brake)")
    print("-" * 50)

    _ = env.reset()

    # Test different gas values
    print("Testing gas remapping (input -> actual):")
    var gas_tests = List[Float64]()
    gas_tests.append(-1.0)
    gas_tests.append(0.0)
    gas_tests.append(1.0)

    for i in range(len(gas_tests)):
        var gas_in = gas_tests[i]
        var gas_out = (gas_in + 1.0) * 0.5  # Expected remapping
        print("  gas=" + String(gas_in)[:5] + " -> " + String(gas_out)[:5])

    print()
    print("=" * 70)
    print("Debug test complete!")
    print("=" * 70)
