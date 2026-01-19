"""
Test velocity integration only (no joint solver).

This isolates the force application + velocity integration to verify
the engine force is correctly applied.

Run with:
    pixi run -e apple mojo run tests/test_velocity_integration_only.mojo
"""

from gpu.host import DeviceContext
from layout import LayoutTensor, Layout

from physics_gpu.constants import (
    dtype,
    BODY_STATE_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
)
from physics_gpu.integrators.euler import SemiImplicitEuler

# Constants
comptime BATCH_SIZE = 1
comptime NUM_BODIES = 1  # Single body for simplicity
comptime BODIES_OFFSET = 0
comptime FORCES_OFFSET = BODY_STATE_SIZE  # After body state
comptime STATE_SIZE = BODY_STATE_SIZE + 3  # body + force

comptime GRAVITY_X: Float64 = 0.0
comptime GRAVITY_Y: Float64 = -10.0
comptime DT: Float64 = 0.02
comptime LANDER_MASS: Float64 = 5.0
comptime MAIN_ENGINE_POWER: Float64 = 13.0
comptime SCALE: Float64 = 30.0


fn print_separator():
    print("=" * 70)


fn test_direct_integration() raises:
    """Test direct velocity integration with force."""
    print_separator()
    print("Direct Velocity Integration Test")
    print_separator()

    with DeviceContext() as ctx:
        var states_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * STATE_SIZE
        )
        var host_states = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)

        # Initialize state
        # Body: x, y, angle, vx, vy, omega, fx, fy, tau, mass, inv_mass, inv_inertia, shape_idx
        host_states[IDX_X] = Scalar[dtype](10.0)
        host_states[IDX_Y] = Scalar[dtype](8.0)
        host_states[IDX_ANGLE] = Scalar[dtype](0.0)
        host_states[IDX_VX] = Scalar[dtype](0.0)
        host_states[IDX_VY] = Scalar[dtype](0.0)
        host_states[IDX_OMEGA] = Scalar[dtype](0.0)
        host_states[IDX_INV_MASS] = Scalar[dtype](1.0 / LANDER_MASS)  # 0.2
        host_states[IDX_INV_INERTIA] = Scalar[dtype](1.0 / 2.0)  # 0.5

        # Force (fx, fy, tau)
        var main_offset = 4.0 / SCALE  # 0.133
        var main_impulse = main_offset * MAIN_ENGINE_POWER  # 1.73
        var main_force = main_impulse / DT  # 86.67

        host_states[FORCES_OFFSET + 0] = Scalar[dtype](0.0)  # fx
        host_states[FORCES_OFFSET + 1] = Scalar[dtype](main_force)  # fy = 86.67
        host_states[FORCES_OFFSET + 2] = Scalar[dtype](0.0)  # tau

        print("Initial state:")
        print("  vy =", host_states[IDX_VY])
        print("  fy =", host_states[FORCES_OFFSET + 1])
        print("  inv_mass =", host_states[IDX_INV_MASS])
        print()

        # Copy to GPU
        ctx.enqueue_copy(states_buf, host_states)
        ctx.synchronize()

        # Run velocity integration
        SemiImplicitEuler.integrate_velocities_gpu[
            BATCH_SIZE, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](
            ctx,
            states_buf,
            Scalar[dtype](GRAVITY_X),
            Scalar[dtype](GRAVITY_Y),
            Scalar[dtype](DT),
        )
        ctx.synchronize()

        # Read back
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var vy_after = Float64(host_states[IDX_VY])
        print("After integration:")
        print("  vy =", vy_after)
        print()

        # Expected: dvy = (86.67 * 0.2 + (-10)) * 0.02 = (17.33 - 10) * 0.02 = 0.147
        var expected_dvy = (
            Float64(main_force) * (1.0 / LANDER_MASS) + GRAVITY_Y
        ) * DT
        print("Expected dvy =", expected_dvy)
        print("Actual dvy =", vy_after)
        print("Ratio =", vy_after / expected_dvy if expected_dvy != 0 else 0.0)

    print_separator()


fn test_gravity_only() raises:
    """Test velocity integration with gravity only (no force)."""
    print_separator()
    print("Gravity Only Integration Test")
    print_separator()

    with DeviceContext() as ctx:
        var states_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * STATE_SIZE
        )
        var host_states = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)

        # Initialize state
        host_states[IDX_X] = Scalar[dtype](10.0)
        host_states[IDX_Y] = Scalar[dtype](8.0)
        host_states[IDX_ANGLE] = Scalar[dtype](0.0)
        host_states[IDX_VX] = Scalar[dtype](0.0)
        host_states[IDX_VY] = Scalar[dtype](0.0)
        host_states[IDX_OMEGA] = Scalar[dtype](0.0)
        host_states[IDX_INV_MASS] = Scalar[dtype](1.0 / LANDER_MASS)  # 0.2
        host_states[IDX_INV_INERTIA] = Scalar[dtype](1.0 / 2.0)  # 0.5

        # Force = 0
        host_states[FORCES_OFFSET + 0] = Scalar[dtype](0.0)
        host_states[FORCES_OFFSET + 1] = Scalar[dtype](0.0)
        host_states[FORCES_OFFSET + 2] = Scalar[dtype](0.0)

        print("Initial state:")
        print("  vy =", host_states[IDX_VY])
        print("  fy =", host_states[FORCES_OFFSET + 1])
        print()

        # Copy to GPU
        ctx.enqueue_copy(states_buf, host_states)
        ctx.synchronize()

        # Run velocity integration
        SemiImplicitEuler.integrate_velocities_gpu[
            BATCH_SIZE, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](
            ctx,
            states_buf,
            Scalar[dtype](GRAVITY_X),
            Scalar[dtype](GRAVITY_Y),
            Scalar[dtype](DT),
        )
        ctx.synchronize()

        # Read back
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var vy_after = Float64(host_states[IDX_VY])
        print("After integration:")
        print("  vy =", vy_after)
        print()

        # Expected: dvy = gravity * dt = -10 * 0.02 = -0.2
        var expected_dvy = GRAVITY_Y * DT
        print("Expected dvy =", expected_dvy)
        print("Actual dvy =", vy_after)
        print("Ratio =", vy_after / expected_dvy if expected_dvy != 0 else 0.0)

    print_separator()


fn main() raises:
    print()
    print("=" * 70)
    print("    VELOCITY INTEGRATION ISOLATION TEST")
    print("=" * 70)
    print()

    test_gravity_only()
    print()

    test_direct_integration()
    print()

    print("=" * 70)
    print("    Tests completed!")
    print("=" * 70)
