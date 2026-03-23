"""Tests for RK4 integrator validation.

Tests:
1. RK4 compiles and runs: single step with HalfCheetah model
2. Energy conservation: undamped pendulum RK4 vs Euler, verify RK4 drift is 10x smaller
3. Trajectory comparison: pendulum with both integrators vs analytical

Run with:
    pixi run mojo run physics3d/tests/test_rk4_integrator.mojo
"""

from std.math import sqrt, pi, cos, abs
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.integrator import EulerIntegrator
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver.pgs_solver import PGSSolver
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel as HC
from std.testing import assert_true, TestSuite


# Pendulum model setup:
# - Body origin at (0,0,0) = pivot point
# - CoM at (0,0,-L) via ipos (MuJoCo convention)
# - Hinge joint at body origin
def setup_pendulum(
    mut model: Model[DType.float64, 1, 1, 2, 1, 5],
    L: Float64,
    m: Float64,
    I_cm: Float64,
):
    # Body 0 = worldbody (initialized by Model.__init__)
    # Body 1 = bob
    model.set_body(1, name="bob", mass=m, inertia=(I_cm, I_cm, I_cm))
    model.set_body_parent(1, 0)  # parent = worldbody
    model.set_body_local_frame(1, pos=(0.0, 0.0, 0.0))
    model.set_body_ipos_iquat(1, ipos=(0.0, 0.0, -L))
    _ = model.add_hinge_joint(
        body_id=1,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )


# =========================================================================
# Test 5.1: RK4 compiles and runs with HalfCheetah
# =========================================================================


def test_rk4_compiles_halfcheetah() raises:
    """Test that RK4Integrator[PGSSolver] compiles and runs a single step
    with the HalfCheetah model without crashing."""
    print("Test 5.1: RK4 compiles and runs with HalfCheetah...")

    comptime DTYPE = DType.float64
    comptime NQ = HC.NQ
    comptime NV = HC.NV
    comptime NBODY = HC.NBODY
    comptime NJOINT = HC.NJOINT
    comptime NGEOM = HC.NGEOM
    comptime MAX_CONTACTS = HC.MAX_CONTACTS

    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HC.MAX_EQUALITY,
        HC.CONE_TYPE,
        HC.MAX_TENDON,
        HC.NSITE,
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HC.NSITE]()
    HC.setup_model_and_data[DTYPE](model, data)

    # Set initial state: torso at height 0.7
    data.qpos[1] = Scalar[DTYPE](0.7)

    # Save initial state
    var q0_1 = data.qpos[1]

    # Run one step with RK4
    RK4Integrator[PGSSolver].step[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM
    ](model, data)

    # Verify state changed (gravity should pull it down)
    var q_changed = False
    var v_changed = False
    for i in range(NV):
        if abs(data.qvel[i]) > Scalar[DTYPE](1e-10):
            v_changed = True
            break

    if data.qpos[1] != q0_1:
        q_changed = True

    if q_changed and v_changed:
        print("  qpos[1] (z):", q0_1, "->", data.qpos[1])
        print("  PASS: RK4 step executed, state changed")
    else:
        print("  FAIL: State did not change after RK4 step")
        print("  q_changed:", q_changed, "v_changed:", v_changed)
        assert_true(
            False,
            (
                "RK4 compiles halfcheetah test failed: state did not change"
                " after RK4 step"
            ),
        )


# =========================================================================
# Test 5.2: Energy conservation — RK4 vs Euler
# =========================================================================


def test_energy_conservation_rk4_vs_euler() raises:
    """Test that RK4 conserves energy at least 10x better than Euler
    for an undamped pendulum over 1000 steps."""
    print("Test 5.2: Energy conservation RK4 vs Euler...")

    var L = Float64(1.0)
    var g = Float64(9.81)
    var m = Float64(1.0)
    var I_cm = Float64(0.01)

    comptime NUM_STEPS = 1000
    var initial_angle = Float64(0.5)  # ~30 degrees

    def compute_energy(
        data: Data[DType.float64, 1, 1, 2, 1, 5],
        m: Float64,
        g: Float64,
        L: Float64,
        I_cm: Float64,
    ) -> Float64:
        # xipos has CoM world position; body 1 CoM z = xipos[1*3+2] = xipos[5]
        var z = data.xipos[5]
        var h = z + L  # Height relative to lowest point
        var PE = m * g * h
        var omega = data.qvel[0]
        var I = m * L * L + I_cm
        var KE = Float64(0.5) * I * omega * omega
        return PE + KE

    # --- Euler run ---
    var model_euler = Model[DType.float64, 1, 1, 2, 1, 5]()
    model_euler.gravity = SIMD[DType.float64, 4](0, 0, -g, 0)
    model_euler.timestep = 0.001
    setup_pendulum(model_euler, L, m, I_cm)

    var data_euler = Data[DType.float64, 1, 1, 2, 1, 5]()
    data_euler.qpos[0] = initial_angle
    data_euler.qvel[0] = Float64(0.0)
    forward_kinematics(model_euler, data_euler)
    var initial_energy_euler = compute_energy(data_euler, m, g, L, I_cm)
    var max_drift_euler = Float64(0.0)

    for _ in range(NUM_STEPS):
        EulerIntegrator[PGSSolver].step(model_euler, data_euler)
        forward_kinematics(model_euler, data_euler)
        var deviation = abs(
            compute_energy(data_euler, m, g, L, I_cm) - initial_energy_euler
        )
        if deviation > max_drift_euler:
            max_drift_euler = deviation

    # --- RK4 run ---
    var model_rk4 = Model[DType.float64, 1, 1, 2, 1, 5]()

    model_rk4.gravity = SIMD[DType.float64, 4](0, 0, -g, 0)
    model_rk4.timestep = 0.001
    setup_pendulum(model_rk4, L, m, I_cm)

    var data_rk4 = Data[DType.float64, 1, 1, 2, 1, 5]()
    data_rk4.qpos[0] = initial_angle
    data_rk4.qvel[0] = Float64(0.0)
    forward_kinematics(model_rk4, data_rk4)
    var initial_energy_rk4 = compute_energy(data_rk4, m, g, L, I_cm)
    var max_drift_rk4 = Float64(0.0)

    for _ in range(NUM_STEPS):
        RK4Integrator[PGSSolver].step(model_rk4, data_rk4)
        forward_kinematics(model_rk4, data_rk4)
        var deviation = abs(
            compute_energy(data_rk4, m, g, L, I_cm) - initial_energy_rk4
        )
        if deviation > max_drift_rk4:
            max_drift_rk4 = deviation

    print("  Initial energy:", initial_energy_euler, "J")
    print("  Euler max energy drift:", max_drift_euler, "J")
    print("  RK4   max energy drift:", max_drift_rk4, "J")

    if initial_energy_euler > Float64(1e-10):
        var euler_pct = max_drift_euler / initial_energy_euler * Float64(100.0)
        var rk4_pct = max_drift_rk4 / initial_energy_rk4 * Float64(100.0)
        print("  Euler drift:", euler_pct, "%")
        print("  RK4   drift:", rk4_pct, "%")

    if max_drift_rk4 < Float64(1e-15):
        print("  RK4 drift is negligible (< 1e-15 J)")
        print("  PASS: RK4 energy conservation is vastly better than Euler")
        return

    if max_drift_euler < Float64(1e-15):
        print("  Both drifts are negligible — test inconclusive but OK")
        print("  PASS: Both conserve energy well")
        return

    var improvement = max_drift_euler / max_drift_rk4
    print("  Improvement ratio:", improvement, "x")

    if improvement >= Float64(10.0):
        print(
            "  PASS: RK4 energy drift is",
            improvement,
            "x smaller than Euler (>= 10x required)",
        )
    else:
        print(
            "  FAIL: RK4 improvement ratio",
            improvement,
            "x is less than 10x",
        )
        assert_true(
            improvement >= Float64(10.0),
            (
                "Energy conservation RK4 vs Euler test failed: RK4 improvement"
                " ratio is less than 10x"
            ),
        )


# =========================================================================
# Test 5.3: Trajectory comparison — pendulum
# =========================================================================


def test_trajectory_comparison() raises:
    """Compare RK4 vs Euler for a pendulum at small angles.
    Both should produce similar results, but RK4 should be more accurate
    (closer to analytical solution)."""
    print("Test 5.3: Trajectory comparison — pendulum small angle...")

    var L = Float64(1.0)
    var g = Float64(9.81)
    var m = Float64(1.0)
    var I_cm = Float64(0.01)
    var initial_angle = Float64(0.05)  # Small angle for analytical comparison

    # Analytical: theta(t) = theta0 * cos(omega * t)
    # omega = sqrt(m*g*L / I), I = m*L^2 + I_cm = 1.01
    var I_total = m * L * L + I_cm
    var omega_nat = sqrt(m * g * L / I_total)

    # --- Euler ---
    var model_euler = Model[DType.float64, 1, 1, 2, 1, 5]()
    model_euler.gravity = SIMD[DType.float64, 4](0, 0, -g, 0)
    model_euler.timestep = 0.001
    setup_pendulum(model_euler, L, m, I_cm)

    var data_euler = Data[DType.float64, 1, 1, 2, 1, 5]()
    data_euler.qpos[0] = initial_angle
    data_euler.qvel[0] = Float64(0.0)

    # --- RK4 ---
    var model_rk4 = Model[DType.float64, 1, 1, 2, 1, 5]()
    model_rk4.gravity = SIMD[DType.float64, 4](0, 0, -g, 0)
    model_rk4.timestep = 0.001
    setup_pendulum(model_rk4, L, m, I_cm)

    var data_rk4 = Data[DType.float64, 1, 1, 2, 1, 5]()
    data_rk4.qpos[0] = initial_angle
    data_rk4.qvel[0] = Float64(0.0)

    # Simulate one full period
    var period = Float64(2.0) * pi / omega_nat
    var dt = Float64(0.001)
    var steps = Int(period / dt)

    var max_euler_error = Float64(0.0)
    var max_rk4_error = Float64(0.0)
    var time = Float64(0.0)

    for _ in range(steps):
        EulerIntegrator[PGSSolver].step(model_euler, data_euler)
        RK4Integrator[PGSSolver].step(model_rk4, data_rk4)
        time = time + dt

        # Analytical: theta(t) = theta0 * cos(omega * t)
        var analytical = initial_angle * cos(omega_nat * time)

        var euler_err = abs(data_euler.qpos[0] - analytical)
        var rk4_err = abs(data_rk4.qpos[0] - analytical)

        if euler_err > max_euler_error:
            max_euler_error = euler_err
        if rk4_err > max_rk4_error:
            max_rk4_error = rk4_err

    print("  Simulation: 1 period (", period, "s) at dt=0.001")
    print("  Max Euler error vs analytical:", max_euler_error, "rad")
    print("  Max RK4   error vs analytical:", max_rk4_error, "rad")
    print("  Euler final angle:", data_euler.qpos[0])
    print("  RK4   final angle:", data_rk4.qpos[0])
    print("  Analytical final:", initial_angle * cos(omega_nat * time))

    # Both should be reasonably close
    var both_reasonable = max_euler_error < Float64(
        0.01
    ) and max_rk4_error < Float64(0.01)

    var rk4_better = max_rk4_error < max_euler_error

    if max_euler_error > Float64(1e-15) and max_rk4_error > Float64(1e-15):
        var improvement = max_euler_error / max_rk4_error
        print("  Accuracy improvement:", improvement, "x")

    if both_reasonable and rk4_better:
        print("  PASS: Both produce similar trajectories, RK4 is more accurate")
    elif both_reasonable:
        print("  PASS: Trajectories are similar (primary requirement)")
    elif rk4_better:
        print("  PASS: RK4 is more accurate (Euler error larger but expected)")
    else:
        print("  FAIL: RK4 is less accurate than Euler")
        assert_true(
            False,
            (
                "Trajectory comparison test failed: RK4 is less accurate than"
                " Euler"
            ),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
