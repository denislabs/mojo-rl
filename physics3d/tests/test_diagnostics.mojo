"""Physics3D Diagnostic Tests — Isolate penetration & bouncing issues.

Tests:
1. Single sphere drop: logs penetration, contact force, velocity per step
2. Solver convergence: same drop with varying PGS iterations
3. Energy audit: free double pendulum (no contacts), check energy drift
4. Contact parameter sweep: vary solref/solimp, observe penetration vs bouncing

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_diagnostics.mojo
"""

from math import sqrt, pi, sin, cos
from builtin.math import abs
from physics3d.types import Model, Data, compute_capsule_inertia
from physics3d.integrator import DefaultIntegrator
from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver.pgs_solver import PGSSolver
from physics3d.solver.cg_solver import CGSolver
from physics3d.solver.newton_solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.constants import GEOM_SPHERE, GEOM_CAPSULE
from physics3d.traits.solver import ConstraintSolver


# ============================================================================
# Test 1: Single Sphere Drop
# ============================================================================

fn test_sphere_drop():
    """Drop a sphere onto ground from 1m. Log penetration, force, velocity."""
    print("=" * 70)
    print("TEST 1: Single Sphere Drop (1m height)")
    print("=" * 70)

    # Single body with slide joint (vertical only) — simplest possible setup
    # NQ=1, NV=1, NBODY=1, NJOINT=1, MAX_CONTACTS=5
    var mass = Float64(1.0)
    var radius = Float64(0.1)
    var dt = Float64(0.002)  # 500 Hz like HalfCheetah

    var model = Model[DType.float64, 1, 1, 1, 1, 5](
        gravity_z=-9.81,
        timestep=dt,
        ground_z=0.0,
        friction=0.5,
    )

    # Sphere body
    model.set_body(0, mass=mass, inertia=(0.004, 0.004, 0.004), radius=radius)
    model.set_body_parent(0, -1)
    model.set_body_local_frame(0, pos=(0.0, 0.0, 0.0))
    model.body_geom_type[0] = GEOM_SPHERE

    # Vertical slide joint (Z axis)
    _ = model.add_slide_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        force_limit=Float64(0.0),  # No actuation
    )

    var data = Data[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = Float64(1.0)  # Start at z=1m (center of sphere)
    data.qvel[0] = Float64(0.0)

    # Expected: hits ground when center z = radius = 0.1m
    # Free-fall time: t = sqrt(2h/g) = sqrt(2*0.9/9.81) ≈ 0.428s
    var total_time = Float64(2.0)
    var steps = Int(total_time / dt)

    print("  dt =", dt, "s, total =", total_time, "s, steps =", steps)
    print("  Expected impact at z_center ≈ 0.1m (radius)")
    print("")
    print("  step | time(s)  | z_center  | velocity  | contacts | penetration | impulse_n")
    print("  " + "-" * 85)

    var max_penetration = Float64(0.0)
    var max_bounce_vel = Float64(0.0)
    var settled = False

    for i in range(steps):
        DefaultIntegrator.step(model, data)
        var time = Float64(i + 1) * dt
        var z = data.qpos[0]
        var vz = data.qvel[0]
        var nc = data.num_contacts
        var pen = Float64(0.0)
        var imp_n = Float64(0.0)

        if nc > 0:
            pen = -Float64(data.contacts[0].dist)  # positive = penetration
            imp_n = Float64(data.contacts[0].impulse_n)

        if pen > max_penetration:
            max_penetration = pen

        # Track bounce velocity (positive = upward after contact)
        if nc > 0 and vz > Float64(0.0) and vz > max_bounce_vel:
            max_bounce_vel = vz

        # Print at key moments: before impact, during contact, settling
        var print_step = False
        if i < 5:
            print_step = True  # First few steps
        elif i % 50 == 0:
            print_step = True  # Every 50 steps
        elif nc > 0 and i % 5 == 0:
            print_step = True  # During contact (every 5 steps)
        elif z < Float64(0.2) and z > Float64(-0.1) and i % 10 == 0:
            print_step = True  # Near ground

        if print_step:
            print(
                "  ",
                i + 1,
                " | ",
                time,
                " | ",
                z,
                " | ",
                vz,
                " | ",
                nc,
                " | ",
                pen,
                " | ",
                imp_n,
            )

        # Check if settled (near ground, low velocity)
        if not settled and nc > 0 and abs(vz) < Float64(0.01) and abs(z - Float64(radius)) < Float64(0.01):
            settled = True
            print("  >>> SETTLED at step", i + 1, "time =", time, "s")

    print("")
    print("  SUMMARY:")
    print("    Max penetration:", max_penetration, "m")
    print("    Max bounce velocity:", max_bounce_vel, "m/s")
    print("    Final z:", data.qpos[0], "m (expected:", radius, ")")
    print("    Final vz:", data.qvel[0], "m/s (expected: ~0)")

    if max_penetration > Float64(0.01):
        print("    ⚠ PENETRATION exceeds 1cm!")
    if max_bounce_vel > Float64(1.0):
        print("    ⚠ EXCESSIVE BOUNCE detected!")
    if abs(Float64(data.qpos[0]) - radius) > Float64(0.05):
        print("    ⚠ DID NOT SETTLE to correct height!")
    print("")


# ============================================================================
# Test 2: Solver Convergence — vary PGS iterations
# ============================================================================

fn _run_drop_with_solver[SOLVER: ConstraintSolver](
    label: String,
    dt: Float64,
    radius: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """Run sphere drop with given solver, return (max_pen, max_bounce, final_z)."""

    var model = Model[DType.float64, 1, 1, 1, 1, 5](
        gravity_z=-9.81,
        timestep=dt,
        ground_z=0.0,
        friction=0.5,
    )

    model.set_body(0, mass=1.0, inertia=(0.004, 0.004, 0.004), radius=radius)
    model.set_body_parent(0, -1)
    model.set_body_local_frame(0, pos=(0.0, 0.0, 0.0))
    model.body_geom_type[0] = GEOM_SPHERE

    _ = model.add_slide_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        force_limit=Float64(0.0),
    )

    var data = Data[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = Float64(1.0)
    data.qvel[0] = Float64(0.0)

    var steps = Int(Float64(2.0) / dt)
    var max_pen = Float64(0.0)
    var max_bounce = Float64(0.0)

    for i in range(steps):
        EulerIntegrator[SOLVER].step(model, data)
        if data.num_contacts > 0:
            var pen = -Float64(data.contacts[0].dist)
            if pen > max_pen:
                max_pen = pen
            if Float64(data.qvel[0]) > Float64(0.0) and Float64(data.qvel[0]) > max_bounce:
                max_bounce = Float64(data.qvel[0])

    return (max_pen, max_bounce, Float64(data.qpos[0]))


fn test_solver_convergence():
    """Compare all three solvers on the same drop test."""
    print("=" * 70)
    print("TEST 2: Solver Convergence Comparison")
    print("=" * 70)
    print("")

    var dt = Float64(0.002)
    var radius = Float64(0.1)

    print("  Solver      | max_penetration | max_bounce_vel | final_z (expect 0.1)")
    print("  " + "-" * 70)

    var pgs = _run_drop_with_solver[PGSSolver]("PGS", dt, radius)
    print("  PGS         |", pgs[0], "|", pgs[1], "|", pgs[2])

    var cg = _run_drop_with_solver[CGSolver]("CG", dt, radius)
    print("  CG          |", cg[0], "|", cg[1], "|", cg[2])

    var newton = _run_drop_with_solver[NewtonSolver]("Newton", dt, radius)
    print("  Newton      |", newton[0], "|", newton[1], "|", newton[2])

    print("")
    if pgs[0] > Float64(0.01) and cg[0] > Float64(0.01) and newton[0] > Float64(0.01):
        print("  → All solvers show penetration: problem is likely in bias/impedance formulation")
    elif pgs[0] > Float64(0.01) and newton[0] < Float64(0.005):
        print("  → PGS worse than Newton: needs more iterations or warmstart")
    else:
        print("  → Results vary by solver — check per-solver details above")
    print("")


# ============================================================================
# Test 3: Energy Audit — free double pendulum (no contacts)
# ============================================================================

fn test_energy_conservation():
    """Free double pendulum, no contacts. Track total energy drift."""
    print("=" * 70)
    print("TEST 3: Energy Conservation — Double Pendulum (no contacts)")
    print("=" * 70)

    # Double pendulum: 2 bodies, 2 hinge joints
    # NQ=2, NV=2, NBODY=2, NJOINT=2, MAX_CONTACTS=5
    var L1 = Float64(1.0)
    var L2 = Float64(1.0)
    var m1 = Float64(1.0)
    var m2 = Float64(1.0)
    var g = Float64(9.81)
    var dt = Float64(0.001)  # Small dt for energy test

    var model = Model[DType.float64, 2, 2, 2, 2, 5](
        gravity_z=-g,
        timestep=dt,
        ground_z=-10.0,  # Far below — no contacts
    )

    # Body 0: upper pendulum
    model.set_body(0, mass=m1, inertia=(0.01, 0.01, 0.01), radius=0.05)
    model.set_body_parent(0, -1)
    model.set_body_local_frame(0, pos=(0.0, 0.0, -L1))

    _ = model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )

    # Body 1: lower pendulum (child of body 0)
    model.set_body(1, mass=m2, inertia=(0.01, 0.01, 0.01), radius=0.05)
    model.set_body_parent(1, 0)
    model.set_body_local_frame(1, pos=(0.0, 0.0, -L2))

    _ = model.add_hinge_joint(
        body_id=1,
        pos=(0.0, 0.0, -L1),  # Joint at end of body 0
        axis=(0.0, 1.0, 0.0),
    )

    var data = Data[DType.float64, 2, 2, 2, 2, 5]()

    # Start with moderate angles (not small-angle regime)
    data.qpos[0] = Float64(0.5)  # ~29 degrees
    data.qpos[1] = Float64(0.3)  # ~17 degrees
    data.qvel[0] = Float64(0.0)
    data.qvel[1] = Float64(0.0)

    # Compute initial energy
    forward_kinematics(model, data)
    var z0_b0 = Float64(data.xpos[2])
    var z0_b1 = Float64(data.xpos[5])
    var PE0 = m1 * g * z0_b0 + m2 * g * z0_b1

    # KE = 0.5 * qvel^T * M * qvel = 0 initially
    var E0 = PE0

    var total_time = Float64(5.0)
    var steps = Int(total_time / dt)

    print("  dt =", dt, "s, total =", total_time, "s, steps =", steps)
    print("  Initial angles: q0 =", data.qpos[0], "q1 =", data.qpos[1])
    print("  Initial energy E0 =", E0, "J")
    print("")
    print("  time(s) | q0       | q1       | vq0      | vq1      | E_approx  | drift(%)")
    print("  " + "-" * 85)

    var max_drift_pct = Float64(0.0)

    for i in range(steps):
        DefaultIntegrator.step(model, data)

        if (i + 1) % 500 == 0 or i < 5:
            var time = Float64(i + 1) * dt
            forward_kinematics(model, data)

            var z_b0 = Float64(data.xpos[2])
            var z_b1 = Float64(data.xpos[5])
            var PE = m1 * g * z_b0 + m2 * g * z_b1

            # Approximate KE using qvel (not exact without M, but useful)
            var I1 = Float64(0.01) + m1 * L1 * L1
            var I2 = Float64(0.01) + m2 * L2 * L2
            var KE_approx = Float64(0.5) * I1 * Float64(data.qvel[0]) * Float64(
                data.qvel[0]
            ) + Float64(0.5) * I2 * Float64(data.qvel[1]) * Float64(data.qvel[1])

            var E = PE + KE_approx
            var drift_pct = (E - E0) / abs(E0) * Float64(100.0)

            if abs(drift_pct) > abs(max_drift_pct):
                max_drift_pct = drift_pct

            print(
                "  ",
                time,
                " | ",
                data.qpos[0],
                " | ",
                data.qpos[1],
                " | ",
                data.qvel[0],
                " | ",
                data.qvel[1],
                " | ",
                E,
                " | ",
                drift_pct,
            )

    print("")
    print("  SUMMARY:")
    print("    Max energy drift:", max_drift_pct, "%")
    if abs(max_drift_pct) < Float64(1.0):
        print("    Energy conservation: GOOD (< 1%)")
    elif abs(max_drift_pct) < Float64(5.0):
        print("    Energy conservation: ACCEPTABLE (< 5%)")
    elif abs(max_drift_pct) < Float64(20.0):
        print("    Energy conservation: POOR (< 20%) — consider RK4 or smaller dt")
    else:
        print("    Energy conservation: BAD (> 20%) — integrator may inject energy!")
    print("")


# ============================================================================
# Test 4: Contact Parameter Sweep — vary solref/solimp
# ============================================================================

fn _run_drop_with_params(
    timeconst: Float64,
    dampratio: Float64,
    dmin: Float64,
    dmax: Float64,
    width: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """Run sphere drop with custom solref/solimp. Return (max_pen, max_bounce, final_z)."""

    var dt = Float64(0.002)
    var radius = Float64(0.1)

    var model = Model[DType.float64, 1, 1, 1, 1, 5](
        gravity_z=-9.81,
        timestep=dt,
        ground_z=0.0,
        friction=0.5,
    )

    # Override solref/solimp
    model.solref_contact[0] = timeconst
    model.solref_contact[1] = dampratio
    model.solimp_contact[0] = dmin
    model.solimp_contact[1] = dmax
    model.solimp_contact[2] = width

    model.set_body(0, mass=1.0, inertia=(0.004, 0.004, 0.004), radius=radius)
    model.set_body_parent(0, -1)
    model.set_body_local_frame(0, pos=(0.0, 0.0, 0.0))
    model.body_geom_type[0] = GEOM_SPHERE

    _ = model.add_slide_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        force_limit=Float64(0.0),
    )

    var data = Data[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = Float64(1.0)
    data.qvel[0] = Float64(0.0)

    var steps = Int(Float64(2.0) / dt)
    var max_pen = Float64(0.0)
    var max_bounce = Float64(0.0)

    for i in range(steps):
        DefaultIntegrator.step(model, data)
        if data.num_contacts > 0:
            var pen = -Float64(data.contacts[0].dist)
            if pen > max_pen:
                max_pen = pen
            if Float64(data.qvel[0]) > Float64(0.0) and Float64(data.qvel[0]) > max_bounce:
                max_bounce = Float64(data.qvel[0])

    return (max_pen, max_bounce, Float64(data.qpos[0]))


fn test_contact_parameter_sweep():
    """Vary solref/solimp parameters and observe effect on penetration/bouncing."""
    print("=" * 70)
    print("TEST 4: Contact Parameter Sweep (solref/solimp)")
    print("=" * 70)
    print("")
    print("  Default: solref=[0.02, 1.0], solimp=[0.9, 0.95, 0.001]")
    print("")

    # --- Vary timeconst ---
    print("  --- Varying timeconst (dampratio=1.0, solimp=default) ---")
    print("  timeconst | max_pen    | max_bounce | final_z (expect 0.1)")
    print("  " + "-" * 65)

    var tc_values = List[Float64]()
    tc_values.append(0.005)
    tc_values.append(0.01)
    tc_values.append(0.02)
    tc_values.append(0.05)
    tc_values.append(0.1)

    for idx in range(len(tc_values)):
        var tc = tc_values[idx]
        var r = _run_drop_with_params(tc, 1.0, 0.9, 0.95, 0.001)
        print("  ", tc, "  |", r[0], "|", r[1], "|", r[2])

    # --- Vary dampratio ---
    print("")
    print("  --- Varying dampratio (timeconst=0.02, solimp=default) ---")
    print("  dampratio | max_pen    | max_bounce | final_z (expect 0.1)")
    print("  " + "-" * 65)

    var dr_values = List[Float64]()
    dr_values.append(0.5)
    dr_values.append(0.8)
    dr_values.append(1.0)
    dr_values.append(1.5)
    dr_values.append(2.0)

    for idx in range(len(dr_values)):
        var dr = dr_values[idx]
        var r = _run_drop_with_params(0.02, dr, 0.9, 0.95, 0.001)
        print("  ", dr, "  |", r[0], "|", r[1], "|", r[2])

    # --- Vary solimp width ---
    print("")
    print("  --- Varying solimp width (solref=default, dmin=0.9, dmax=0.95) ---")
    print("  width     | max_pen    | max_bounce | final_z (expect 0.1)")
    print("  " + "-" * 65)

    var w_values = List[Float64]()
    w_values.append(0.0005)
    w_values.append(0.001)
    w_values.append(0.005)
    w_values.append(0.01)
    w_values.append(0.05)

    for idx in range(len(w_values)):
        var w = w_values[idx]
        var r = _run_drop_with_params(0.02, 1.0, 0.9, 0.95, w)
        print("  ", w, "     |", r[0], "|", r[1], "|", r[2])

    # --- Vary dmin (impedance floor) ---
    print("")
    print("  --- Varying dmin (solref=default, dmax=0.95, width=0.001) ---")
    print("  dmin      | max_pen    | max_bounce | final_z (expect 0.1)")
    print("  " + "-" * 65)

    var dmin_values = List[Float64]()
    dmin_values.append(0.0)
    dmin_values.append(0.2)
    dmin_values.append(0.5)
    dmin_values.append(0.8)
    dmin_values.append(0.9)

    for idx in range(len(dmin_values)):
        var dm = dmin_values[idx]
        var r = _run_drop_with_params(0.02, 1.0, dm, 0.95, 0.001)
        print("  ", dm, "      |", r[0], "|", r[1], "|", r[2])

    print("")


# ============================================================================
# Test 5: Capsule Drop — more realistic (HalfCheetah-like body)
# ============================================================================

fn test_capsule_drop():
    """Drop a capsule (torso-sized) onto ground. More realistic than sphere."""
    print("=" * 70)
    print("TEST 5: Capsule Drop (HalfCheetah torso-like)")
    print("=" * 70)

    # Capsule with slide-z + hinge-y (can tip over)
    # NQ=2, NV=2, NBODY=1, NJOINT=2, MAX_CONTACTS=5
    var mass = Float64(6.25)  # HalfCheetah torso mass
    var radius = Float64(0.046)
    var half_length = Float64(0.5)
    var dt = Float64(0.002)

    var model = Model[DType.float64, 2, 2, 1, 2, 5](
        gravity_z=-9.81,
        timestep=dt,
        ground_z=0.0,
        friction=0.5,
    )

    var inertia = compute_capsule_inertia(mass, radius, half_length)

    model.set_body(0, mass=mass, inertia=inertia, radius=radius)
    model.set_body_parent(0, -1)
    model.set_body_local_frame(0, pos=(0.0, 0.0, 0.0))
    model.body_geom_type[0] = GEOM_CAPSULE
    model.body_half_length[0] = half_length

    # Root slide Z (vertical)
    _ = model.add_slide_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        force_limit=Float64(0.0),
    )

    # Root hinge Y (tipping)
    _ = model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        tau_limit=Float64(0.0),
    )

    var data = Data[DType.float64, 2, 2, 1, 2, 5]()
    data.qpos[0] = Float64(0.5)  # Start at 0.5m height
    data.qpos[1] = Float64(0.0)  # No initial tilt
    data.qvel[0] = Float64(0.0)
    data.qvel[1] = Float64(0.0)

    var total_time = Float64(2.0)
    var steps = Int(total_time / dt)

    print("  Capsule: mass =", mass, "kg, radius =", radius, "m, half_length =", half_length, "m")
    print("  dt =", dt, "s")
    print("")
    print("  step | time(s)  | z_center | angle    | vz       | contacts | max_pen")
    print("  " + "-" * 80)

    var max_pen = Float64(0.0)
    var max_bounce = Float64(0.0)

    for i in range(steps):
        DefaultIntegrator.step(model, data)

        var nc = data.num_contacts
        var pen = Float64(0.0)
        for c in range(nc):
            var p = -Float64(data.contacts[c].dist)
            if p > pen:
                pen = p
        if pen > max_pen:
            max_pen = pen
        if nc > 0 and Float64(data.qvel[0]) > Float64(0.0) and Float64(data.qvel[0]) > max_bounce:
            max_bounce = Float64(data.qvel[0])

        if i < 5 or (i + 1) % 100 == 0 or (nc > 0 and (i + 1) % 20 == 0):
            print(
                "  ",
                i + 1,
                " | ",
                Float64(i + 1) * dt,
                " | ",
                data.qpos[0],
                " | ",
                data.qpos[1],
                " | ",
                data.qvel[0],
                " | ",
                nc,
                " | ",
                pen,
            )

    print("")
    print("  SUMMARY:")
    print("    Max penetration:", max_pen, "m")
    print("    Max bounce vel:", max_bounce, "m/s")
    print("    Final z:", data.qpos[0], "m")
    print("    Final angle:", data.qpos[1], "rad")

    # Expected rest height: radius ≈ 0.046m (capsule resting on its side)
    # or half_length + radius ≈ 0.546 if standing upright
    print("    Expected rest z ≈ radius =", radius, "(on side) or half_len+radius =", half_length + radius, "(upright)")
    print("")


# ============================================================================
# Main
# ============================================================================

fn main():
    print("")
    print("Physics3D Diagnostic Tests")
    print("==========================")
    print("")

    test_sphere_drop()
    test_solver_convergence()
    test_energy_conservation()
    test_contact_parameter_sweep()
    test_capsule_drop()

    print("=" * 70)
    print("ALL DIAGNOSTIC TESTS COMPLETE")
    print("=" * 70)
