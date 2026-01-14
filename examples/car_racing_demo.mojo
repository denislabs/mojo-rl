"""
CarRacing Demo: Native Mojo Implementation

Demonstrates the CarRacing environment with a simple control policy.

The car will drive forward and steer automatically based on its
position relative to the track center. This demo shows the
environment rendering and physics working correctly.

Run with:
    pixi run mojo run examples/car_racing_demo.mojo
"""

from random import random_float64
from envs.car_racing import CarRacingEnv, CarRacingAction


fn main() raises:
    print("CarRacing Demo - Native Mojo Implementation")
    print("============================================")
    print("")
    print("Running autonomous demo with simple control policy...")
    print("The car will steer based on the waypoint direction.")
    print("")

    var env = CarRacingEnv(continuous=True)
    var state = env.reset()

    var total_reward: Float64 = 0.0
    var episode: Int = 1
    var step_count: Int = 0
    var max_episodes: Int = 3
    var max_steps_per_episode: Int = 1000

    print("Episode", episode, "started")
    print("Track has", env.track_length, "tiles")

    while episode <= max_episodes:
        # Simple control policy:
        # - Steer towards waypoint direction
        # - Apply gas to maintain speed
        # - Brake on sharp turns

        # Get state observation
        var obs = state.to_list()

        # Waypoint direction from state (indices 11, 12)
        var waypoint_dx: Float64 = 0.0
        if len(obs) > 11:
            waypoint_dx = obs[11]
            # Note: obs[12] is waypoint_dy, unused in simple steering

        # Simple steering: steer in waypoint direction
        # Use waypoint_dx as steering input (negative = turn left)
        var steering = -waypoint_dx * 2.0  # Amplify steering
        steering = clamp(steering, -1.0, 1.0)

        # Gas: accelerate more when heading straight
        var gas: Float64 = 0.5 + 0.3 * (1.0 - abs_f64(steering))

        # Brake: light braking on sharp turns
        var brake: Float64 = 0.0
        if abs_f64(steering) > 0.6:
            brake = 0.3

        # Create and apply action
        var action = CarRacingAction(steering, gas, brake)
        var result = env.step(action)
        state = result[0]
        var reward = result[1]
        var done = result[2]

        total_reward += reward
        step_count += 1

        # Render
        env.render()

        # Print progress periodically
        if step_count % 100 == 0:
            var progress = env.tile_visited_count * 100 / max(env.track_length, 1)
            print(
                "Step", step_count,
                "| Reward:", Int(total_reward),
                "| Progress:", progress, "%",
                "| Speed:", Int(sqrt(state.vx * state.vx + state.vy * state.vy) * 100.0)
            )

        if done or step_count >= max_steps_per_episode:
            print("\n=== Episode", episode, "finished ===")
            print("Total Reward:", Int(total_reward))
            print("Tiles visited:", env.tile_visited_count, "/", env.track_length)
            print("Steps:", step_count)

            # Reset for next episode
            state = env.reset()
            total_reward = 0.0
            step_count = 0
            episode += 1

            if episode <= max_episodes:
                print("\nEpisode", episode, "started")
                print("Track has", env.track_length, "tiles")

    env.close()
    print("\nDemo finished!")


fn clamp(x: Float64, low: Float64, high: Float64) -> Float64:
    if x < low:
        return low
    if x > high:
        return high
    return x


fn abs_f64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


fn max(a: Int, b: Int) -> Int:
    return a if a > b else b


fn sqrt(x: Float64) -> Float64:
    from math import sqrt as math_sqrt
    return math_sqrt(x)
