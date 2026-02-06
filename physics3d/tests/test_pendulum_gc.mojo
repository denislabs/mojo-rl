"""Tests for pendulum dynamics using the Generalized Coordinates engine.

Tests:
1. Pendulum period matches analytical T = 2*pi*sqrt(L/g) (within 2%)
2. Energy conservation (drift < 5% over 5 periods)
3. Basic dynamics (gravity causes swinging)

Run with:
    pixi run mojo run physics3d/generalized/tests/test_pendulum_gc.mojo
"""

from math import sqrt, pi, sin, cos
from builtin.math import abs
from physics3d.types import ModelGC, DataGC
from physics3d.integrator.semi_implicit_euler_integrator import (
    SemiImplicitEulerIntegrator,
)
from physics3d.kinematics.forward_kinematics import forward_kinematics


fn test_pendulum_period() -> Bool:
    """Test that pendulum period matches analytical value.

    For a physical pendulum with moment of inertia I about the pivot:
    T = 2*pi*sqrt(I/(m*g*L))

    With I = I_cm + m*L^2 = 0.01 + 1.0*1.0^2 = 1.01, m=1, L=1, g=9.81:
    T ≈ 2.016 seconds
    """
    print("Test pendulum period...")

    var L = Float64(1.0)  # Pendulum length
    var g = Float64(9.81)  # Gravity
    var m = Float64(1.0)  # Mass
    var I_cm = Float64(0.01)  # Body inertia at CoM
    var I_pivot = I_cm + m * L * L  # Total inertia about pivot
    var expected_period = Float64(2.0) * pi * sqrt(I_pivot / (m * g * L))
    print("  Expected period:", expected_period, "s")

    # Create simple pendulum
    # Mass at end of massless rod (approximated by body at distance L)
    var model = ModelGC[DType.float64, 1, 1, 1, 1, 5](
        gravity_z=-g,
        timestep=0.001,  # Small timestep for accuracy
        ground_z=-10.0,  # Below pendulum to avoid collision
    )

    model.set_body(0, mass=1.0, inertia=(0.01, 0.01, 0.01), radius=0.1)
    model.set_body_parent(0, -1)

    # Body at (0, 0, -L) relative to pivot at origin
    model.set_body_local_frame(0, pos=(0.0, 0.0, -L))

    # Hinge at origin, Y axis
    _ = model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data = DataGC[DType.float64, 1, 1, 1, 1, 5]()

    # Small initial angle (for small-angle approximation to be valid)
    var initial_angle = Float64(0.1)  # ~6 degrees
    data.qpos[0] = initial_angle
    data.qvel[0] = Float64(0.0)

    # Run simulation and measure period
    var dt = model.timestep
    var max_time = Float64(10.0)  # 10 seconds
    var steps = Int(max_time / Float64(dt))

    var prev_sign_positive = data.qpos[0] > Float64(0)
    var zero_crossings = 0
    var first_crossing_time = Float64(0.0)
    var last_crossing_time = Float64(0.0)
    var time = Float64(0.0)

    for i in range(steps):
        SemiImplicitEulerIntegrator.step(model, data)
        time = time + Float64(dt)

        var current_sign_positive = data.qpos[0] > Float64(0)

        # Detect zero crossing (positive to negative = half period)
        if prev_sign_positive and not current_sign_positive:
            zero_crossings += 1
            if zero_crossings == 1:
                first_crossing_time = time
            last_crossing_time = time

        prev_sign_positive = current_sign_positive

        # Stop after we have enough crossings
        if zero_crossings >= 10:
            break

    if zero_crossings < 2:
        print("  FAIL: Not enough zero crossings detected")
        return False

    # Period = time between consecutive positive-to-negative crossings
    # Each such crossing occurs once per full period
    var num_periods = zero_crossings - 1
    var measured_period = (last_crossing_time - first_crossing_time) / Float64(
        num_periods
    )
    print("  Measured period:", measured_period, "s")

    var error_pct = (
        abs(measured_period - expected_period)
        / expected_period
        * Float64(100.0)
    )
    print("  Error:", error_pct, "%")

    if error_pct < Float64(2.0):
        print("  PASS: Period within 2% of analytical value")
        return True
    else:
        print("  FAIL: Period error exceeds 2%")
        return False


fn test_energy_conservation() -> Bool:
    """Test that total energy is conserved over multiple periods."""
    print("Test energy conservation...")

    var L = Float64(1.0)
    var g = Float64(9.81)
    var m = Float64(1.0)

    var model = ModelGC[DType.float64, 1, 1, 1, 1, 5](
        gravity_z=-g,
        timestep=0.001,
        ground_z=-10.0,  # Below pendulum to avoid collision
    )

    model.set_body(0, mass=m, inertia=(0.01, 0.01, 0.01), radius=0.1)
    model.set_body_parent(0, -1)
    model.set_body_local_frame(0, pos=(0.0, 0.0, -L))

    _ = model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data = DataGC[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = Float64(0.5)  # ~30 degrees
    data.qvel[0] = Float64(0.0)

    # Initial forward kinematics to get xpos
    forward_kinematics(model, data)

    # Compute initial energy
    # PE = m*g*h where h is height of mass relative to lowest point
    # KE = 0.5 * I * omega^2 where I = m*L^2
    fn compute_energy(
        data: DataGC[DType.float64, 1, 1, 1, 1, 5],
        m: Float64,
        g: Float64,
        L: Float64,
    ) -> Float64:
        var z = data.xpos[2]
        var h = z + L  # Height relative to lowest point (z = -L)
        var PE = m * g * h
        var omega = data.qvel[0]
        var I = m * L * L
        var KE = Float64(0.5) * I * omega * omega
        return PE + KE

    var initial_energy = compute_energy(data, m, g, L)
    print("  Initial energy:", initial_energy, "J")

    # Run for 5 periods
    var expected_period = Float64(2.0) * pi * sqrt(L / g)
    var sim_time = Float64(5.0) * expected_period
    var steps = Int(sim_time / Float64(model.timestep))

    var max_energy_deviation = Float64(0.0)

    for i in range(steps):
        SemiImplicitEulerIntegrator.step(model, data)
        forward_kinematics(model, data)

        var current_energy = compute_energy(data, m, g, L)
        var deviation = abs(current_energy - initial_energy)
        if deviation > max_energy_deviation:
            max_energy_deviation = deviation

    var drift_pct = max_energy_deviation / initial_energy * Float64(100.0)
    print("  Max energy drift:", drift_pct, "%")

    if drift_pct < Float64(5.0):
        print("  PASS: Energy drift < 5%")
        return True
    else:
        print("  FAIL: Energy drift exceeds 5%")
        return False


fn test_gravity_swinging() -> Bool:
    """Test that gravity causes the pendulum to swing."""
    print("Test gravity causes swinging...")

    var model = ModelGC[DType.float64, 1, 1, 1, 1, 5](
        gravity_z=-9.81,
        timestep=0.01,
        ground_z=-10.0,  # Below pendulum to avoid collision
    )

    model.set_body(0, mass=1.0, inertia=(0.1, 0.1, 0.1), radius=0.1)
    model.set_body_parent(0, -1)
    model.set_body_local_frame(0, pos=(0.0, 0.0, -1.0))

    _ = model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data = DataGC[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = Float64(0.5)  # Initial angle
    data.qvel[0] = Float64(0.0)

    var initial_angle = data.qpos[0]

    # Run for 0.5 seconds
    for _ in range(50):
        SemiImplicitEulerIntegrator.step(model, data)

    # Angle should have changed (pendulum swinging)
    var angle_change = abs(data.qpos[0] - initial_angle)
    print("  Initial angle:", initial_angle)
    print("  Final angle:", data.qpos[0])
    print("  Angular velocity:", data.qvel[0])

    if angle_change > Float64(0.1) and abs(data.qvel[0]) > Float64(0.1):
        print("  PASS: Pendulum is swinging")
        return True
    else:
        print("  FAIL: Pendulum did not swing")
        return False


fn main():
    print("=== Pendulum GC Tests ===\n")

    var all_pass = True

    if not test_gravity_swinging():
        all_pass = False

    if not test_pendulum_period():
        all_pass = False

    if not test_energy_conservation():
        all_pass = False

    print("")
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
