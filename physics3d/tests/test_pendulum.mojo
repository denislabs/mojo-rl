"""Tests for pendulum dynamics using the Generalized Coordinates engine.

Tests:
1. Pendulum period matches analytical T = 2*pi*sqrt(L/g) (within 2%)
2. Energy conservation (drift < 5% over 5 periods)
3. Basic dynamics (gravity causes swinging)

Run with:
    pixi run mojo run physics3d/tests/test_pendulum.mojo
"""

from math import sqrt, pi, sin, cos
from builtin.math import abs
from physics3d.types import Model, Data
from physics3d.integrator import DefaultIntegrator
from physics3d.kinematics.forward_kinematics import forward_kinematics
from testing import assert_true, TestSuite


fn setup_pendulum(
    mut model: Model[DType.float64, 1, 1, 2, 1, 5],
    L: Float64,
    m: Float64,
    I_cm: Float64,
):
    """Set up a simple pendulum: pivot at body origin, CoM at (0,0,-L) via ipos.

    MuJoCo-style convention: jnt_pos is in body frame.
    With jnt_pos=(0,0,0) and body_pos=(0,0,0), the pivot is at the world origin.
    ipos=(0,0,-L) places the CoM L metres below the pivot.
    """
    model.set_body(1, name="bob", mass=m, inertia=(I_cm, I_cm, I_cm))
    model.set_body_parent(1, 0)
    model.set_body_local_frame(1, pos=(0.0, 0.0, 0.0))  # body at pivot (origin)
    model.set_body_ipos_iquat(1, ipos=(0.0, 0.0, -L))   # CoM at -L below pivot
    _ = model.add_hinge_joint(
        body_id=1,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )


fn test_pendulum_period() raises:
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

    var model = Model[DType.float64, 1, 1, 2, 1, 5]()
    model.gravity = SIMD[DType.float64, 4](0, 0, -g, 0)
    model.timestep = 0.001  # Small timestep for accuracy

    setup_pendulum(model, L, m, I_cm)

    var data = Data[DType.float64, 1, 1, 2, 1, 5]()

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
        DefaultIntegrator.step(model, data)
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
        assert_true(False, "Pendulum period test failed: not enough zero crossings detected")

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
    else:
        print("  FAIL: Period error exceeds 2%")
        assert_true(error_pct < Float64(2.0), "Pendulum period test failed: period error exceeds 2%")


fn test_energy_conservation() raises:
    """Test that total energy is conserved over multiple periods."""
    print("Test energy conservation...")

    var L = Float64(1.0)
    var g = Float64(9.81)
    var m = Float64(1.0)
    var I_cm = Float64(0.01)

    var model = Model[DType.float64, 1, 1, 2, 1, 5]()
    model.gravity = SIMD[DType.float64, 4](0, 0, -g, 0)
    model.timestep = 0.001

    setup_pendulum(model, L, m, I_cm)

    var data = Data[DType.float64, 1, 1, 2, 1, 5]()
    data.qpos[0] = Float64(0.5)  # ~30 degrees
    data.qvel[0] = Float64(0.0)

    # Initial forward kinematics to get xipos (CoM position)
    forward_kinematics(model, data)

    # Compute initial energy
    # PE = m*g*h where h is CoM height relative to lowest point
    # KE = 0.5 * I_pivot * omega^2 where I_pivot = I_cm + m*L^2
    fn compute_energy(
        data: Data[DType.float64, 1, 1, 2, 1, 5],
        m: Float64,
        g: Float64,
        L: Float64,
        I_cm: Float64,
    ) -> Float64:
        var z_com = data.xipos[5]  # body 1 CoM, z component
        var h = z_com + L  # Height relative to lowest point (CoM z = -L at theta=0)
        var PE = m * g * h
        var omega = data.qvel[0]
        var I_pivot = I_cm + m * L * L
        var KE = Float64(0.5) * I_pivot * omega * omega
        return PE + KE

    var initial_energy = compute_energy(data, m, g, L, I_cm)
    print("  Initial energy:", initial_energy, "J")

    # Run for 5 periods
    var expected_period = Float64(2.0) * pi * sqrt((I_cm + m*L*L) / (m*g*L))
    var sim_time = Float64(5.0) * expected_period
    var steps = Int(sim_time / Float64(model.timestep))

    var max_energy_deviation = Float64(0.0)

    for i in range(steps):
        DefaultIntegrator.step(model, data)
        forward_kinematics(model, data)

        var current_energy = compute_energy(data, m, g, L, I_cm)
        var deviation = abs(current_energy - initial_energy)
        if deviation > max_energy_deviation:
            max_energy_deviation = deviation

    var drift_pct = max_energy_deviation / abs(initial_energy) * Float64(100.0)
    print("  Max energy drift:", drift_pct, "%")

    if drift_pct < Float64(5.0):
        print("  PASS: Energy drift < 5%")
    else:
        print("  FAIL: Energy drift exceeds 5%")
        assert_true(drift_pct < Float64(5.0), "Energy conservation test failed: energy drift exceeds 5%")


fn test_gravity_swinging() raises:
    """Test that gravity causes the pendulum to swing."""
    print("Test gravity causes swinging...")

    var L = Float64(1.0)
    var model = Model[DType.float64, 1, 1, 2, 1, 5]()

    setup_pendulum(model, L, Float64(1.0), Float64(0.1))

    var data = Data[DType.float64, 1, 1, 2, 1, 5]()
    data.qpos[0] = Float64(0.5)  # Initial angle
    data.qvel[0] = Float64(0.0)

    var initial_angle = data.qpos[0]

    # Run for 0.5 seconds (default dt=0.01)
    for _ in range(50):
        DefaultIntegrator.step(model, data)

    # Angle should have changed (pendulum swinging)
    var angle_change = abs(data.qpos[0] - initial_angle)
    print("  Initial angle:", initial_angle)
    print("  Final angle:", data.qpos[0])
    print("  Angular velocity:", data.qvel[0])

    if angle_change > Float64(0.1) and abs(data.qvel[0]) > Float64(0.1):
        print("  PASS: Pendulum is swinging")
    else:
        print("  FAIL: Pendulum did not swing")
        assert_true(angle_change > Float64(0.1) and abs(data.qvel[0]) > Float64(0.1), "Gravity swinging test failed: pendulum did not swing, angle_change=" + String(angle_change))


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
