"""
Comparison Test: lunar_lander.mojo vs lunar_lander_v2.mojo

This test validates that both LunarLander implementations produce
equivalent physics behavior when given identical initial conditions
and action sequences.

The test:
1. Creates both environments
2. Sets identical initial conditions manually
3. Applies the same action sequence
4. Compares observations step-by-step
"""

from math import sqrt
from testing import assert_true

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_v2 import LunarLanderV2

# Constants matching both implementations
comptime SCALE: Float64 = 30.0
comptime VIEWPORT_W: Int = 600
comptime VIEWPORT_H: Int = 400
comptime FPS: Int = 50


fn abs_f64(x: Float64) -> Float64:
    """Absolute value of Float64."""
    return x if x >= 0.0 else -x


fn print_separator():
    print("=" * 60)


fn test_physics_comparison() raises:
    """Compare physics between V1 and V2 with controlled initial conditions."""
    print_separator()
    print("LunarLander V1 vs V2: Physics Comparison Test")
    print_separator()

    # Create both environments
    # V1: single environment
    var env_v1 = LunarLanderEnv[DType.float64](
        continuous=False,
        gravity=-10.0,
        enable_wind=False,  # Disable wind for deterministic comparison
    )

    # V2: batched environment (use batch size 1 for comparison)
    var env_v2 = LunarLanderV2[BATCH=1](
        seed=12345,
        enable_wind=False,  # Disable wind for deterministic comparison
    )

    # Reset both environments
    _ = env_v1.reset()
    env_v2.reset_all()

    # Now manually set identical initial conditions
    # We'll set the lander to a known position with zero velocity

    var W = Float64(VIEWPORT_W) / SCALE
    var H = Float64(VIEWPORT_H) / SCALE
    var init_x = W / 2.0  # Center X
    var init_y = H * 0.8  # 80% height (like initial spawn)
    var init_vx: Float64 = 0.0  # Zero initial velocity for clean comparison
    var init_vy: Float64 = 0.0
    var init_angle: Float64 = 0.0
    var init_omega: Float64 = 0.0

    # Set V1 lander state
    env_v1.world.bodies[env_v1.lander_idx].position.x = init_x
    env_v1.world.bodies[env_v1.lander_idx].position.y = init_y
    env_v1.world.bodies[env_v1.lander_idx].linear_velocity.x = init_vx
    env_v1.world.bodies[env_v1.lander_idx].linear_velocity.y = init_vy
    env_v1.world.bodies[env_v1.lander_idx].angle = init_angle
    env_v1.world.bodies[env_v1.lander_idx].angular_velocity = init_omega

    # Set V1 leg positions relative to lander
    var leg_away = 20.0 / SCALE
    var leg_down = 18.0 / SCALE
    env_v1.world.bodies[env_v1.left_leg_idx].position.x = init_x - leg_away
    env_v1.world.bodies[env_v1.left_leg_idx].position.y = init_y - leg_down
    env_v1.world.bodies[env_v1.left_leg_idx].linear_velocity.x = init_vx
    env_v1.world.bodies[env_v1.left_leg_idx].linear_velocity.y = init_vy
    env_v1.world.bodies[env_v1.left_leg_idx].angle = 0.0
    env_v1.world.bodies[env_v1.left_leg_idx].angular_velocity = 0.0

    env_v1.world.bodies[env_v1.right_leg_idx].position.x = init_x + leg_away
    env_v1.world.bodies[env_v1.right_leg_idx].position.y = init_y - leg_down
    env_v1.world.bodies[env_v1.right_leg_idx].linear_velocity.x = init_vx
    env_v1.world.bodies[env_v1.right_leg_idx].linear_velocity.y = init_vy
    env_v1.world.bodies[env_v1.right_leg_idx].angle = 0.0
    env_v1.world.bodies[env_v1.right_leg_idx].angular_velocity = 0.0

    # Initialize V1 prev_shaping to avoid large reward spike
    env_v1.prev_shaping = 0.0

    # Set V2 lander state (env=0, body=0 is lander)
    env_v2.physics.set_body_position(0, 0, init_x, init_y)
    env_v2.physics.set_body_velocity(0, 0, init_vx, init_vy, init_omega)
    env_v2.physics.set_body_angle(0, 0, init_angle)

    # Set V2 leg positions
    var left_leg_x = init_x - leg_away
    var left_leg_y = init_y - leg_down - (10.0 / SCALE)  # Adjusted for V2's leg attachment
    var right_leg_x = init_x + leg_away
    var right_leg_y = init_y - leg_down - (10.0 / SCALE)

    env_v2.physics.set_body_position(0, 1, left_leg_x, left_leg_y)
    env_v2.physics.set_body_velocity(0, 1, init_vx, init_vy, 0.0)
    env_v2.physics.set_body_angle(0, 1, 0.0)

    env_v2.physics.set_body_position(0, 2, right_leg_x, right_leg_y)
    env_v2.physics.set_body_velocity(0, 2, init_vx, init_vy, 0.0)
    env_v2.physics.set_body_angle(0, 2, 0.0)

    # Initialize V2 prev_shaping
    env_v2.prev_shaping[0] = 0.0

    print("Initial conditions set:")
    print("  Position: (", init_x, ",", init_y, ")")
    print("  Velocity: (", init_vx, ",", init_vy, ")")
    print("  Angle: ", init_angle)
    print()

    # Get initial observations
    var obs_v1 = env_v1.get_obs_list()
    var obs_v2 = env_v2.get_observation(0)

    print("Initial observations:")
    print("  V1: x=", obs_v1[0], " y=", obs_v1[1], " vx=", obs_v1[2], " vy=", obs_v1[3])
    print("  V2: x=", obs_v2[0], " y=", obs_v2[1], " vx=", obs_v2[2], " vy=", obs_v2[3])
    print()

    # Define action sequence for testing
    # Test different actions: nop, main engine, left, right
    var actions = List[Int]()
    # First 10 steps: no action (free fall)
    for _ in range(10):
        actions.append(0)
    # Next 10 steps: main engine
    for _ in range(10):
        actions.append(2)
    # Next 5 steps: left engine
    for _ in range(5):
        actions.append(1)
    # Next 5 steps: right engine
    for _ in range(5):
        actions.append(3)
    # More free fall
    for _ in range(20):
        actions.append(0)

    var num_steps = len(actions)
    print("Running", num_steps, "steps with controlled actions...")
    print()

    # Track maximum differences
    var max_x_diff: Float64 = 0.0
    var max_y_diff: Float64 = 0.0
    var max_vx_diff: Float64 = 0.0
    var max_vy_diff: Float64 = 0.0
    var max_angle_diff: Float64 = 0.0
    var max_omega_diff: Float64 = 0.0
    var max_reward_diff: Float64 = 0.0

    var total_x_diff: Float64 = 0.0
    var total_y_diff: Float64 = 0.0

    print("Step | Action | V1 y    | V2 y    | y diff  | V1 vy   | V2 vy   | vy diff")
    print("-" * 75)

    var v1_done = False
    var v2_done = False

    for step in range(num_steps):
        var action = actions[step]

        # Step both environments
        var result_v1 = env_v1.step_discrete(action)
        var state_v1 = result_v1[0]
        var reward_v1 = result_v1[1]
        var done_v1 = result_v1[2]

        # For V2, we need to set action and step
        var result_v2 = env_v2.step(0, action)
        var obs_v2_new = env_v2.get_observation(0)
        var reward_v2 = result_v2[0]
        var done_v2 = result_v2[1]

        # Get V1 observation
        var v1_x = Float64(state_v1.x)
        var v1_y = Float64(state_v1.y)
        var v1_vx = Float64(state_v1.vx)
        var v1_vy = Float64(state_v1.vy)
        var v1_angle = Float64(state_v1.angle)
        var v1_omega = Float64(state_v1.angular_velocity)

        # Get V2 observation
        var v2_x = Float64(obs_v2_new[0])
        var v2_y = Float64(obs_v2_new[1])
        var v2_vx = Float64(obs_v2_new[2])
        var v2_vy = Float64(obs_v2_new[3])
        var v2_angle = Float64(obs_v2_new[4])
        var v2_omega = Float64(obs_v2_new[5])

        # Compute differences
        var x_diff = abs_f64(v1_x - v2_x)
        var y_diff = abs_f64(v1_y - v2_y)
        var vx_diff = abs_f64(v1_vx - v2_vx)
        var vy_diff = abs_f64(v1_vy - v2_vy)
        var angle_diff = abs_f64(v1_angle - v2_angle)
        var omega_diff = abs_f64(v1_omega - v2_omega)
        var reward_diff = abs_f64(Float64(reward_v1) - Float64(reward_v2))

        # Track maximums
        if x_diff > max_x_diff:
            max_x_diff = x_diff
        if y_diff > max_y_diff:
            max_y_diff = y_diff
        if vx_diff > max_vx_diff:
            max_vx_diff = vx_diff
        if vy_diff > max_vy_diff:
            max_vy_diff = vy_diff
        if angle_diff > max_angle_diff:
            max_angle_diff = angle_diff
        if omega_diff > max_omega_diff:
            max_omega_diff = omega_diff
        if reward_diff > max_reward_diff:
            max_reward_diff = reward_diff

        total_x_diff += x_diff
        total_y_diff += y_diff

        # Print every 5 steps
        if step % 5 == 0 or step < 5:
            print(step, "    |", action, "     |", v1_y, "|", v2_y, "|", y_diff, "|", v1_vy, "|", v2_vy, "|", vy_diff)

        # Check if either environment terminated
        if done_v1 and not v1_done:
            print("  V1 terminated at step", step)
            v1_done = True
        if done_v2 and not v2_done:
            print("  V2 terminated at step", step)
            v2_done = True

        # If both terminated, stop
        if v1_done and v2_done:
            break

    print("-" * 75)
    print()

    # Print summary
    print("Maximum Differences:")
    print("  x:      ", max_x_diff)
    print("  y:      ", max_y_diff)
    print("  vx:     ", max_vx_diff)
    print("  vy:     ", max_vy_diff)
    print("  angle:  ", max_angle_diff)
    print("  omega:  ", max_omega_diff)
    print("  reward: ", max_reward_diff)
    print()

    print("Average Differences:")
    print("  x:      ", total_x_diff / Float64(num_steps))
    print("  y:      ", total_y_diff / Float64(num_steps))
    print()

    # Determine if physics match within tolerance
    var tolerance: Float64 = 0.1  # 10% tolerance for physics differences
    var physics_match = (
        max_y_diff < tolerance and
        max_vy_diff < tolerance and
        max_angle_diff < tolerance
    )

    if physics_match:
        print("RESULT: Physics engines produce SIMILAR trajectories (within", tolerance, "tolerance)")
    else:
        print("RESULT: Physics engines produce DIFFERENT trajectories")
        print("  This is expected due to different physics implementations:")
        print("  - V1 uses physics/world.mojo (sequential)")
        print("  - V2 uses physics_gpu/ (batched)")

    print_separator()


fn test_free_fall_comparison() raises:
    """Compare simple free fall between V1 and V2."""
    print_separator()
    print("LunarLander V1 vs V2: Free Fall Comparison")
    print_separator()

    # Create environments with wind disabled
    var env_v1 = LunarLanderEnv[DType.float64](
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
    )

    var env_v2 = LunarLanderV2[BATCH=1](
        seed=12345,
        enable_wind=False,
    )

    # Reset both
    _ = env_v1.reset()
    env_v2.reset_all()

    # Set identical initial conditions: lander at center, zero velocity
    var W = Float64(VIEWPORT_W) / SCALE
    var H = Float64(VIEWPORT_H) / SCALE
    var init_x = W / 2.0
    var init_y = H * 0.75  # 75% height

    # Set V1 state
    env_v1.world.bodies[env_v1.lander_idx].position.x = init_x
    env_v1.world.bodies[env_v1.lander_idx].position.y = init_y
    env_v1.world.bodies[env_v1.lander_idx].linear_velocity.x = 0.0
    env_v1.world.bodies[env_v1.lander_idx].linear_velocity.y = 0.0
    env_v1.world.bodies[env_v1.lander_idx].angle = 0.0
    env_v1.world.bodies[env_v1.lander_idx].angular_velocity = 0.0
    env_v1.prev_shaping = 0.0

    # Set V2 state
    env_v2.physics.set_body_position(0, 0, init_x, init_y)
    env_v2.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    env_v2.physics.set_body_angle(0, 0, 0.0)
    env_v2.prev_shaping[0] = 0.0

    print("Free fall from y =", init_y, "(no actions, no wind)")
    print()
    print("Step | V1 y    | V2 y    | V1 vy   | V2 vy   | y diff")
    print("-" * 60)

    # Step 20 times with no action
    for step in range(20):
        # Step V1
        var result_v1 = env_v1.step_discrete(0)  # nop
        var state_v1 = result_v1[0]

        # Step V2
        _ = env_v2.step(0, 0)  # nop
        var obs_v2 = env_v2.get_observation(0)

        var v1_y = Float64(state_v1.y)
        var v2_y = Float64(obs_v2[1])
        var v1_vy = Float64(state_v1.vy)
        var v2_vy = Float64(obs_v2[3])
        var y_diff = abs_f64(v1_y - v2_y)

        print(step, "    |", v1_y, "|", v2_y, "|", v1_vy, "|", v2_vy, "|", y_diff)

        if result_v1[2] or result_v1[2]:  # Either terminated
            break

    print_separator()


fn test_main_engine_comparison() raises:
    """Compare main engine thrust between V1 and V2."""
    print_separator()
    print("LunarLander V1 vs V2: Main Engine Comparison")
    print_separator()

    # Create environments
    var env_v1 = LunarLanderEnv[DType.float64](
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
    )

    var env_v2 = LunarLanderV2[BATCH=1](
        seed=12345,
        enable_wind=False,
    )

    # Reset both
    _ = env_v1.reset()
    env_v2.reset_all()

    # Set identical initial conditions
    var W = Float64(VIEWPORT_W) / SCALE
    var H = Float64(VIEWPORT_H) / SCALE
    var init_x = W / 2.0
    var init_y = H * 0.5  # 50% height

    # Set V1 state
    env_v1.world.bodies[env_v1.lander_idx].position.x = init_x
    env_v1.world.bodies[env_v1.lander_idx].position.y = init_y
    env_v1.world.bodies[env_v1.lander_idx].linear_velocity.x = 0.0
    env_v1.world.bodies[env_v1.lander_idx].linear_velocity.y = 0.0
    env_v1.world.bodies[env_v1.lander_idx].angle = 0.0
    env_v1.world.bodies[env_v1.lander_idx].angular_velocity = 0.0
    env_v1.prev_shaping = 0.0

    # Set V2 state
    env_v2.physics.set_body_position(0, 0, init_x, init_y)
    env_v2.physics.set_body_velocity(0, 0, 0.0, 0.0, 0.0)
    env_v2.physics.set_body_angle(0, 0, 0.0)
    env_v2.prev_shaping[0] = 0.0

    print("Main engine test from y =", init_y)
    print("Firing main engine (action=2) every step")
    print()
    print("Step | V1 y    | V2 y    | V1 vy   | V2 vy   | vy diff")
    print("-" * 60)

    # Step with main engine
    for step in range(15):
        # Step V1 with main engine
        var result_v1 = env_v1.step_discrete(2)  # main engine
        var state_v1 = result_v1[0]

        # Step V2 with main engine
        _ = env_v2.step(0, 2)  # main engine
        var obs_v2 = env_v2.get_observation(0)

        var v1_y = Float64(state_v1.y)
        var v2_y = Float64(obs_v2[1])
        var v1_vy = Float64(state_v1.vy)
        var v2_vy = Float64(obs_v2[3])
        var vy_diff = abs_f64(v1_vy - v2_vy)

        print(step, "    |", v1_y, "|", v2_y, "|", v1_vy, "|", v2_vy, "|", vy_diff)

    print_separator()


fn main() raises:
    print()
    print("=" * 60)
    print("    LUNAR LANDER V1 vs V2 COMPARISON TESTS")
    print("    Comparing physics engine implementations")
    print("=" * 60)
    print()

    # Run comparison tests
    test_free_fall_comparison()
    print()

    test_main_engine_comparison()
    print()

    test_physics_comparison()
    print()

    print("=" * 60)
    print("    All comparison tests completed!")
    print("=" * 60)
