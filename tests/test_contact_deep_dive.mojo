"""Deep dive into contact physics differences between CPU and GPU.

This test specifically examines:
1. Terrain height values
2. Leg position calculations
3. Contact detection thresholds
4. Contact response mechanics

Run with:
    pixi run -e apple mojo run tests/test_contact_deep_dive.mojo
"""

from math import cos, sin, sqrt, pi
from random import seed, random_float64

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_gpu import (
    LunarLanderGPU,
    gpu_dtype,
    HELIPAD_Y as GPU_HELIPAD_Y,
    LEG_AWAY as GPU_LEG_AWAY,
    LEG_DOWN as GPU_LEG_DOWN,
    W_UNITS,
    H_UNITS,
    HELIPAD_X,
)


fn abs_f32(x: Float32) -> Float32:
    return x if x >= 0 else -x


fn format_float(val: Float32, width: Int = 10) -> String:
    var s = String(val)
    if len(s) > width:
        return String(s[:width])
    return s


fn main() raises:
    print("=" * 70)
    print("CONTACT PHYSICS DEEP DIVE")
    print("=" * 70)
    print("")

    # =========================================================================
    # 1. Compare terrain/helipad heights
    # =========================================================================
    print("1. TERRAIN HEIGHT COMPARISON")
    print("-" * 50)

    seed(42)
    var cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    print("CPU Environment:")
    print("  helipad_y = " + String(cpu_env.helipad_y))
    print("  helipad_x1 = " + String(cpu_env.helipad_x1))
    print("  helipad_x2 = " + String(cpu_env.helipad_x2))

    print("")
    print("GPU Environment (constants):")
    print("  HELIPAD_Y = " + String(GPU_HELIPAD_Y) + " (H_UNITS/4 = " + String(H_UNITS/4.0) + ")")
    print("  HELIPAD_X = " + String(HELIPAD_X) + " (W_UNITS/2 = " + String(W_UNITS/2.0) + ")")
    print("  LEG_AWAY = " + String(GPU_LEG_AWAY) + " (20/30)")
    print("  LEG_DOWN = " + String(GPU_LEG_DOWN) + " (18/30)")

    # =========================================================================
    # 2. Compare leg position calculations
    # =========================================================================
    print("")
    print("2. LEG POSITION CALCULATION COMPARISON")
    print("-" * 50)

    # Get CPU lander and leg positions
    var cpu_obs = cpu_env.get_obs_list()

    # Denormalize CPU position
    var cpu_x_norm = cpu_obs[0]
    var cpu_y_norm = cpu_obs[1]
    var cpu_angle = cpu_obs[4]

    # CPU normalization formula (from lunar_lander.mojo:914-919):
    # x_norm = (pos.x - W/2) / (W/2)
    # y_norm = (pos.y - (helipad_y + LEG_DOWN/SCALE)) / (H/2)
    var SCALE: Float32 = 30.0
    var W = Float32(600.0) / SCALE  # ~20 units
    var H = Float32(400.0) / SCALE  # ~13.33 units

    var cpu_x_world = cpu_x_norm * (W / 2.0) + W / 2.0
    var cpu_y_world = cpu_y_norm * (H / 2.0) + (Float32(cpu_env.helipad_y) + 18.0/SCALE)

    print("CPU lander (denormalized):")
    print("  x_world = " + format_float(cpu_x_world))
    print("  y_world = " + format_float(cpu_y_world))
    print("  angle = " + format_float(cpu_angle))

    # CPU leg positions from physics bodies
    var left_leg_body = cpu_env.world.bodies[cpu_env.left_leg_idx].copy()
    var right_leg_body = cpu_env.world.bodies[cpu_env.right_leg_idx].copy()

    print("  CPU left_leg_body.position = (" + format_float(Float32(left_leg_body.position.x))
          + ", " + format_float(Float32(left_leg_body.position.y)) + ")")
    print("  CPU right_leg_body.position = (" + format_float(Float32(right_leg_body.position.x))
          + ", " + format_float(Float32(right_leg_body.position.y)) + ")")

    # GPU leg position calculation (geometric)
    var cos_angle = cos(cpu_angle)
    var sin_angle = sin(cpu_angle)

    # From GPU code (lunar_lander_gpu.mojo:507-528):
    # left_leg_x = x - LEG_AWAY * cos_angle + LEG_DOWN * sin_angle
    # left_leg_y = y - LEG_AWAY * sin_angle - LEG_DOWN * cos_angle
    var gpu_left_leg_x = cpu_x_world - Float32(GPU_LEG_AWAY) * cos_angle + Float32(GPU_LEG_DOWN) * sin_angle
    var gpu_left_leg_y = cpu_x_world - Float32(GPU_LEG_AWAY) * sin_angle - Float32(GPU_LEG_DOWN) * cos_angle  # BUG: should use y_world

    # Correct calculation
    var gpu_left_leg_y_correct = cpu_y_world - Float32(GPU_LEG_AWAY) * sin_angle - Float32(GPU_LEG_DOWN) * cos_angle
    var gpu_right_leg_y_correct = cpu_y_world + Float32(GPU_LEG_AWAY) * sin_angle - Float32(GPU_LEG_DOWN) * cos_angle

    print("")
    print("GPU geometric leg positions (using CPU body position):")
    print("  GPU left_leg_y (geometric) = " + format_float(gpu_left_leg_y_correct))
    print("  GPU right_leg_y (geometric) = " + format_float(gpu_right_leg_y_correct))

    # =========================================================================
    # 3. Contact detection threshold analysis
    # =========================================================================
    print("")
    print("3. CONTACT DETECTION THRESHOLDS")
    print("-" * 50)

    # CPU contact: uses fixture collision detection
    # GPU contact: uses leg_y <= terrain_y

    print("CPU: Uses Box2D collision detection with fixtures")
    print("  - Left leg fixture collides with terrain edge fixtures")
    print("  - Contact detected when shapes overlap")
    print("")
    print("GPU: Uses geometric comparison")
    print("  - Contact when leg_y <= HELIPAD_Y (" + String(GPU_HELIPAD_Y) + ")")
    print("  - No actual collision detection, just height check")

    # =========================================================================
    # 4. Simulate a vertical drop and compare contact timing
    # =========================================================================
    print("")
    print("4. VERTICAL DROP CONTACT TIMING TEST")
    print("-" * 50)
    print("")

    # Reset environment
    seed(123)
    cpu_env = LunarLanderEnv[DType.float32](enable_wind=False)
    _ = cpu_env.reset()

    var ctx = DeviceContext()

    print("Dropping lander with no-op actions until contact...")
    print("Comparing when each system detects first contact")
    print("")

    print("Step | CPU y_obs  | CPU leg_L | GPU_y (geom) | Would GPU contact?")
    print("-" * 70)

    var cpu_contact_step = -1
    var gpu_would_contact_step = -1

    for step in range(150):
        var obs = cpu_env.get_obs_list()
        var cpu_left_contact = obs[6]

        # Denormalize y to world coords
        var y_norm = obs[1]
        var angle = obs[4]
        var y_world = y_norm * (H / 2.0) + (Float32(cpu_env.helipad_y) + 18.0/SCALE)

        # Compute GPU-style leg position
        var cos_a = cos(angle)
        var sin_a = sin(angle)
        var gpu_style_leg_y = y_world - Float32(GPU_LEG_AWAY) * sin_a - Float32(GPU_LEG_DOWN) * cos_a

        # Would GPU detect contact?
        var gpu_would_contact = gpu_style_leg_y <= Float32(GPU_HELIPAD_Y)

        if cpu_contact_step < 0 and cpu_left_contact > 0.5:
            cpu_contact_step = step
        if gpu_would_contact_step < 0 and gpu_would_contact:
            gpu_would_contact_step = step

        if step < 5 or step % 10 == 0 or cpu_contact_step == step or gpu_would_contact_step == step:
            var gpu_contact_str = "YES" if gpu_would_contact else "no"
            print(
                String(step) + "    | "
                + format_float(y_norm, 10) + " | "
                + format_float(cpu_left_contact, 9) + " | "
                + format_float(gpu_style_leg_y, 12) + " | "
                + gpu_contact_str
            )

        if cpu_left_contact > 0.5:
            print("\n*** CPU contact at step " + String(step) + " ***")
            break

        # Step with no-op
        var result = cpu_env.step_discrete(0)
        if result[2]:
            print("\nCPU terminated at step " + String(step))
            break

    print("")
    print("CONTACT TIMING RESULTS:")
    print("  CPU first contact: step " + String(cpu_contact_step))
    print("  GPU would contact: step " + String(gpu_would_contact_step))

    if cpu_contact_step != gpu_would_contact_step:
        print("  MISMATCH: " + String(abs_f32(Float32(cpu_contact_step - gpu_would_contact_step))) + " steps difference!")

    # =========================================================================
    # 5. Analysis of CPU terrain
    # =========================================================================
    print("")
    print("5. CPU TERRAIN ANALYSIS")
    print("-" * 50)

    print("CPU terrain_x values: ", end="")
    for i in range(len(cpu_env.terrain_x)):
        print(String(cpu_env.terrain_x[i])[:5] + " ", end="")
    print("")

    print("CPU terrain_y values: ", end="")
    for i in range(len(cpu_env.terrain_y)):
        print(String(cpu_env.terrain_y[i])[:5] + " ", end="")
    print("")

    print("")
    print("Helipad region terrain height: " + String(cpu_env.helipad_y))
    print("GPU flat terrain height: " + String(GPU_HELIPAD_Y))

    var terrain_diff = abs_f32(Float32(cpu_env.helipad_y) - Float32(GPU_HELIPAD_Y))
    print("Difference: " + format_float(terrain_diff))

    # =========================================================================
    # 6. Key findings
    # =========================================================================
    print("")
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("")
    print("1. TERRAIN HEIGHT:")
    print("   - CPU helipad_y varies per reset (random terrain generation)")
    print("   - GPU uses fixed HELIPAD_Y = H_UNITS/4 = " + String(GPU_HELIPAD_Y))
    print("   - This causes contact timing mismatch!")
    print("")
    print("2. LEG POSITION:")
    print("   - CPU: Actual physics bodies with revolute joints")
    print("   - GPU: Geometric calculation from body center")
    print("   - Joints allow legs to flex, GPU legs are rigidly attached")
    print("")
    print("3. CONTACT DETECTION:")
    print("   - CPU: Box2D fixture collision (considers leg shape)")
    print("   - GPU: Simple height comparison (point vs line)")
    print("")
    print("4. CONTACT RESPONSE:")
    print("   - CPU: Impulse-based solver with friction (0.1)")
    print("   - GPU: Velocity damping (50% per iteration)")
    print("   - GPU position correction is different from Box2D")
    print("")
    print("RECOMMENDATIONS:")
    print("-" * 50)
    print("1. CRITICAL: Match terrain height exactly")
    print("   - GPU should use cpu_env.helipad_y (need to pass as param)")
    print("   - Or: CPU should use fixed H/4 helipad like GPU")
    print("")
    print("2. Improve contact detection:")
    print("   - Account for leg joint angles in GPU")
    print("   - Or: Use same geometric calculation in CPU for comparison")
    print("")
    print("3. Match contact response:")
    print("   - Use impulse-based response: j = -m*v for restitution=0")
    print("   - Apply friction as: j_friction = -friction * j_normal")
    print("=" * 70)
