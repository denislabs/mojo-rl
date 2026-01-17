"""Test physics with deterministic (no random) engine dispersion.

This test isolates how much drift comes from random number differences
vs actual physics implementation differences.

Run with:
    pixi run -e apple mojo run tests/test_deterministic_physics.mojo
"""

from math import cos, sin, sqrt, pi
from random import seed

from gpu.host import DeviceContext

from envs.lunar_lander import LunarLanderEnv, LunarLanderAction
from envs.lunar_lander_gpu_v3 import gpu_dtype

# Constants matching GPU
comptime SCALE: Float64 = 30.0
comptime FPS: Int = 50
comptime TAU: Float64 = 0.02
comptime GRAVITY: Float64 = -10.0
comptime MAIN_ENGINE_POWER: Float64 = 13.0
comptime SIDE_ENGINE_POWER: Float64 = 0.6
comptime LANDER_MASS: Float64 = 5.0
comptime LANDER_INERTIA: Float64 = 2.0
comptime MAIN_ENGINE_Y_OFFSET: Float64 = 4.0 / 30.0
comptime SIDE_ENGINE_AWAY: Float64 = 12.0 / 30.0
comptime SIDE_ENGINE_HEIGHT: Float64 = 14.0 / 30.0
comptime LANDER_HALF_HEIGHT: Float64 = 17.0 / 30.0
comptime W_UNITS: Float64 = 20.0
comptime H_UNITS: Float64 = 13.333
comptime HELIPAD_Y: Float64 = H_UNITS / 4.0
comptime HELIPAD_X: Float64 = W_UNITS / 2.0
comptime LEG_DOWN: Float64 = 18.0 / 30.0


fn abs_f64(x: Float64) -> Float64:
    return x if x >= 0 else -x


fn format_float(val: Float64, width: Int = 10) -> String:
    var s = String(val)
    if len(s) > width:
        return String(s[:width])
    return s


struct SimplifiedPhysics:
    """Minimal physics matching GPU implementation exactly (no random)."""
    var x: Float64
    var y: Float64
    var vx: Float64
    var vy: Float64
    var angle: Float64
    var angular_vel: Float64

    fn __init__(out self, x_obs: Float64, y_obs: Float64, vx_obs: Float64,
                vy_obs: Float64, angle: Float64, angular_vel_obs: Float64):
        # Denormalize from observation to world coords
        self.x = x_obs * (W_UNITS / 2.0) + HELIPAD_X
        self.y = y_obs * (H_UNITS / 2.0) + (HELIPAD_Y + LEG_DOWN)
        self.vx = vx_obs / (W_UNITS / 2.0 / Float64(FPS))
        self.vy = vy_obs / (H_UNITS / 2.0 / Float64(FPS))
        self.angle = angle
        self.angular_vel = angular_vel_obs / (20.0 / Float64(FPS))

    fn step(mut self, action: Int):
        """Step physics with NO random dispersion."""
        var tip_x = sin(self.angle)
        var tip_y = cos(self.angle)
        var side_x = -tip_y
        var side_y = tip_x

        # Zero dispersion - deterministic
        var dispersion_x: Float64 = 0.0
        var dispersion_y: Float64 = 0.0

        if action == 2:
            # Main engine (no dispersion)
            var ox = tip_x * MAIN_ENGINE_Y_OFFSET
            var oy = -tip_y * MAIN_ENGINE_Y_OFFSET
            var impulse_x = -ox * MAIN_ENGINE_POWER
            var impulse_y = -oy * MAIN_ENGINE_POWER
            self.vx += impulse_x / LANDER_MASS
            self.vy += impulse_y / LANDER_MASS
            var torque = ox * impulse_y - oy * impulse_x
            self.angular_vel += torque / LANDER_INERTIA

        elif action == 1:
            # Left engine
            var direction: Float64 = -1.0
            var ox = side_x * direction * SIDE_ENGINE_AWAY
            var oy = -side_y * direction * SIDE_ENGINE_AWAY
            var impulse_x = -ox * SIDE_ENGINE_POWER
            var impulse_y = -oy * SIDE_ENGINE_POWER
            self.vx += impulse_x / LANDER_MASS
            self.vy += impulse_y / LANDER_MASS
            var r_x = ox - tip_x * LANDER_HALF_HEIGHT
            var r_y = oy + tip_y * SIDE_ENGINE_HEIGHT
            var torque = r_x * impulse_y - r_y * impulse_x
            self.angular_vel += torque / LANDER_INERTIA

        elif action == 3:
            # Right engine
            var direction: Float64 = 1.0
            var ox = side_x * direction * SIDE_ENGINE_AWAY
            var oy = -side_y * direction * SIDE_ENGINE_AWAY
            var impulse_x = -ox * SIDE_ENGINE_POWER
            var impulse_y = -oy * SIDE_ENGINE_POWER
            self.vx += impulse_x / LANDER_MASS
            self.vy += impulse_y / LANDER_MASS
            var r_x = ox - tip_x * LANDER_HALF_HEIGHT
            var r_y = oy + tip_y * SIDE_ENGINE_HEIGHT
            var torque = r_x * impulse_y - r_y * impulse_x
            self.angular_vel += torque / LANDER_INERTIA

        # Apply gravity
        self.vy += GRAVITY * TAU

        # Integrate position
        self.x += self.vx * TAU
        self.y += self.vy * TAU
        self.angle += self.angular_vel * TAU

        # Normalize angle
        while self.angle > pi:
            self.angle -= 2.0 * pi
        while self.angle < -pi:
            self.angle += 2.0 * pi

    fn get_obs(self) -> List[Float64]:
        """Return normalized observation."""
        var obs = List[Float64]()
        obs.append((self.x - HELIPAD_X) / (W_UNITS / 2.0))
        obs.append((self.y - (HELIPAD_Y + LEG_DOWN)) / (H_UNITS / 2.0))
        obs.append(self.vx * (W_UNITS / 2.0 / Float64(FPS)))
        obs.append(self.vy * (H_UNITS / 2.0 / Float64(FPS)))
        obs.append(self.angle)
        obs.append(self.angular_vel * (20.0 / Float64(FPS)))
        obs.append(0.0)  # left contact
        obs.append(0.0)  # right contact
        return obs^


fn main() raises:
    print("=" * 70)
    print("DETERMINISTIC PHYSICS TEST")
    print("=" * 70)
    print("")
    print("Comparing CPU Box2D vs simplified deterministic physics")
    print("(both with ZERO random engine dispersion)")
    print("")

    seed(42)
    var cpu_env = LunarLanderEnv[DType.float64](enable_wind=False)
    _ = cpu_env.reset()

    # Get initial state from CPU
    var cpu_obs = cpu_env.get_obs_list()

    # Create simplified physics from same initial state
    var simple_phys = SimplifiedPhysics(
        Float64(cpu_obs[0]), Float64(cpu_obs[1]),
        Float64(cpu_obs[2]), Float64(cpu_obs[3]),
        Float64(cpu_obs[4]), Float64(cpu_obs[5])
    )

    print("Initial state:")
    print("  x_obs=" + format_float(Float64(cpu_obs[0])))
    print("  y_obs=" + format_float(Float64(cpu_obs[1])))
    print("  vx_obs=" + format_float(Float64(cpu_obs[2])))
    print("  vy_obs=" + format_float(Float64(cpu_obs[3])))
    print("")

    # Run both physics for several steps
    print("Running 50 steps with action sequence: [nop, main, main, left, right] x 10")
    print("")

    var actions = List[Int]()
    for _ in range(10):
        actions.append(0)
        actions.append(2)
        actions.append(2)
        actions.append(1)
        actions.append(3)

    print("Step | CPU vy    | Simple vy | vy diff   | CPU angle | Simple ang | angle diff")
    print("-" * 80)

    var max_vy_diff: Float64 = 0.0
    var max_angle_diff: Float64 = 0.0

    for step in range(50):
        var action = actions[step]

        # Step CPU
        var cpu_result = cpu_env.step_discrete(action)
        var cpu_next = cpu_result[0].to_list()
        var cpu_done = cpu_result[2]

        # Step simplified physics
        simple_phys.step(action)
        var simple_obs = simple_phys.get_obs()

        # Compare key values
        var vy_diff = abs_f64(Float64(cpu_next[3]) - simple_obs[3])
        var angle_diff = abs_f64(Float64(cpu_next[4]) - simple_obs[4])

        if vy_diff > max_vy_diff:
            max_vy_diff = vy_diff
        if angle_diff > max_angle_diff:
            max_angle_diff = angle_diff

        if step < 10 or step % 10 == 0:
            var action_str = "nop  "
            if action == 1:
                action_str = "left "
            elif action == 2:
                action_str = "main "
            elif action == 3:
                action_str = "right"

            print(
                String(step) + "    | "
                + format_float(Float64(cpu_next[3]), 9) + " | "
                + format_float(simple_obs[3], 9) + " | "
                + format_float(vy_diff, 9) + " | "
                + format_float(Float64(cpu_next[4]), 9) + " | "
                + format_float(simple_obs[4], 9) + " | "
                + format_float(angle_diff, 9)
            )

        if cpu_done:
            print("\nCPU terminated at step " + String(step))
            break

    print("")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("")
    print("Max vy difference: " + format_float(max_vy_diff))
    print("Max angle difference: " + format_float(max_angle_diff))
    print("")

    if max_vy_diff < 0.01 and max_angle_diff < 0.01:
        print("CONCLUSION: Physics implementation is correct!")
        print("The drift in GPU tests comes from RANDOM DISPERSION differences.")
    elif max_vy_diff < 0.1 and max_angle_diff < 0.1:
        print("CONCLUSION: Physics mostly correct, small numerical differences.")
        print("Random dispersion contributes significantly to overall drift.")
    else:
        print("CONCLUSION: Significant physics implementation differences found.")
        print("Need to investigate engine impulse or integration code.")

    print("=" * 70)
