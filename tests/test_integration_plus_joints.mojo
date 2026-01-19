"""
Test velocity integration with joint solving.

This isolates the force application + velocity integration + joint solver
to verify if the joint solver is causing the velocity reduction.

Run with:
    pixi run -e apple mojo run tests/test_integration_plus_joints.mojo
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
    JOINT_DATA_SIZE,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REVOLUTE,
)
from physics_gpu.integrators.euler import SemiImplicitEuler
from physics_gpu.joints.revolute import RevoluteJointSolver

from envs.lunar_lander_v2_gpu import (
    STATE_SIZE_VAL,
    BODIES_OFFSET,
    FORCES_OFFSET,
    JOINTS_OFFSET,
    JOINT_COUNT_OFFSET,
    NUM_BODIES,
    MAX_JOINTS,
    VELOCITY_ITERATIONS,
    DT,
    GRAVITY_X,
    GRAVITY_Y,
    LANDER_MASS,
    LANDER_INERTIA,
    LEG_MASS,
    LEG_INERTIA,
    LEG_AWAY,
    LEG_H,
    SCALE,
    MAIN_ENGINE_POWER,
)

comptime gpu_dtype = DType.float32
comptime BATCH_SIZE = 1


fn print_separator():
    print("=" * 70)


fn test_with_joints() raises:
    """Test velocity integration followed by joint solving."""
    print_separator()
    print("Velocity Integration + Joint Solver Test")
    print_separator()

    with DeviceContext() as ctx:
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE_VAL)
        var joint_counts_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE_VAL)
        var host_joint_counts = ctx.enqueue_create_host_buffer[gpu_dtype](1)

        # Initialize lander body at index BODIES_OFFSET
        var lander_off = BODIES_OFFSET
        host_states[lander_off + IDX_X] = Scalar[gpu_dtype](10.0)
        host_states[lander_off + IDX_Y] = Scalar[gpu_dtype](8.0)
        host_states[lander_off + IDX_ANGLE] = Scalar[gpu_dtype](0.0)
        host_states[lander_off + IDX_VX] = Scalar[gpu_dtype](0.0)
        host_states[lander_off + IDX_VY] = Scalar[gpu_dtype](0.0)
        host_states[lander_off + IDX_OMEGA] = Scalar[gpu_dtype](0.0)
        host_states[lander_off + IDX_INV_MASS] = Scalar[gpu_dtype](1.0 / LANDER_MASS)
        host_states[lander_off + IDX_INV_INERTIA] = Scalar[gpu_dtype](1.0 / LANDER_INERTIA)

        # Initialize left leg body
        var left_leg_off = BODIES_OFFSET + 1 * BODY_STATE_SIZE
        host_states[left_leg_off + IDX_X] = Scalar[gpu_dtype](10.0 - LEG_AWAY)
        host_states[left_leg_off + IDX_Y] = Scalar[gpu_dtype](8.0 - 10.0/SCALE - LEG_H)
        host_states[left_leg_off + IDX_ANGLE] = Scalar[gpu_dtype](0.0)
        host_states[left_leg_off + IDX_VX] = Scalar[gpu_dtype](0.0)
        host_states[left_leg_off + IDX_VY] = Scalar[gpu_dtype](0.0)
        host_states[left_leg_off + IDX_OMEGA] = Scalar[gpu_dtype](0.0)
        host_states[left_leg_off + IDX_INV_MASS] = Scalar[gpu_dtype](1.0 / LEG_MASS)
        host_states[left_leg_off + IDX_INV_INERTIA] = Scalar[gpu_dtype](1.0 / LEG_INERTIA)

        # Initialize right leg body
        var right_leg_off = BODIES_OFFSET + 2 * BODY_STATE_SIZE
        host_states[right_leg_off + IDX_X] = Scalar[gpu_dtype](10.0 + LEG_AWAY)
        host_states[right_leg_off + IDX_Y] = Scalar[gpu_dtype](8.0 - 10.0/SCALE - LEG_H)
        host_states[right_leg_off + IDX_ANGLE] = Scalar[gpu_dtype](0.0)
        host_states[right_leg_off + IDX_VX] = Scalar[gpu_dtype](0.0)
        host_states[right_leg_off + IDX_VY] = Scalar[gpu_dtype](0.0)
        host_states[right_leg_off + IDX_OMEGA] = Scalar[gpu_dtype](0.0)
        host_states[right_leg_off + IDX_INV_MASS] = Scalar[gpu_dtype](1.0 / LEG_MASS)
        host_states[right_leg_off + IDX_INV_INERTIA] = Scalar[gpu_dtype](1.0 / LEG_INERTIA)

        # Initialize forces - main engine force on lander only
        var main_offset = 4.0 / SCALE
        var main_impulse = main_offset * MAIN_ENGINE_POWER
        var main_force = main_impulse / DT

        for body in range(NUM_BODIES):
            var force_off = FORCES_OFFSET + body * 3
            host_states[force_off + 0] = Scalar[gpu_dtype](0.0)
            host_states[force_off + 1] = Scalar[gpu_dtype](0.0)
            host_states[force_off + 2] = Scalar[gpu_dtype](0.0)

        # Apply main engine force to lander only
        host_states[FORCES_OFFSET + 1] = Scalar[gpu_dtype](main_force)

        # Initialize joints
        host_states[JOINT_COUNT_OFFSET] = Scalar[gpu_dtype](2)

        # Left leg joint
        var joint0_off = JOINTS_OFFSET + 0 * JOINT_DATA_SIZE
        host_states[joint0_off + JOINT_TYPE] = Scalar[gpu_dtype](JOINT_REVOLUTE)
        host_states[joint0_off + JOINT_BODY_A] = Scalar[gpu_dtype](0)  # lander
        host_states[joint0_off + JOINT_BODY_B] = Scalar[gpu_dtype](1)  # left leg
        host_states[joint0_off + JOINT_ANCHOR_AX] = Scalar[gpu_dtype](-LEG_AWAY)
        host_states[joint0_off + JOINT_ANCHOR_AY] = Scalar[gpu_dtype](-10.0 / SCALE)
        host_states[joint0_off + JOINT_ANCHOR_BX] = Scalar[gpu_dtype](0.0)
        host_states[joint0_off + JOINT_ANCHOR_BY] = Scalar[gpu_dtype](LEG_H)

        # Right leg joint
        var joint1_off = JOINTS_OFFSET + 1 * JOINT_DATA_SIZE
        host_states[joint1_off + JOINT_TYPE] = Scalar[gpu_dtype](JOINT_REVOLUTE)
        host_states[joint1_off + JOINT_BODY_A] = Scalar[gpu_dtype](0)  # lander
        host_states[joint1_off + JOINT_BODY_B] = Scalar[gpu_dtype](2)  # right leg
        host_states[joint1_off + JOINT_ANCHOR_AX] = Scalar[gpu_dtype](LEG_AWAY)
        host_states[joint1_off + JOINT_ANCHOR_AY] = Scalar[gpu_dtype](-10.0 / SCALE)
        host_states[joint1_off + JOINT_ANCHOR_BX] = Scalar[gpu_dtype](0.0)
        host_states[joint1_off + JOINT_ANCHOR_BY] = Scalar[gpu_dtype](LEG_H)

        host_joint_counts[0] = Scalar[gpu_dtype](2)

        print("Initial state (before integration):")
        print("  Lander vy =", host_states[lander_off + IDX_VY])
        print("  Left leg vy =", host_states[left_leg_off + IDX_VY])
        print("  Right leg vy =", host_states[right_leg_off + IDX_VY])
        print("  Lander force fy =", host_states[FORCES_OFFSET + 1])
        print()

        # Copy to GPU
        ctx.enqueue_copy(states_buf, host_states)
        ctx.enqueue_copy(joint_counts_buf, host_joint_counts)
        ctx.synchronize()

        # Step 1: Integrate velocities
        SemiImplicitEuler.integrate_velocities_gpu_strided[
            BATCH_SIZE, NUM_BODIES, STATE_SIZE_VAL, BODIES_OFFSET, FORCES_OFFSET
        ](
            ctx,
            states_buf,
            Scalar[gpu_dtype](GRAVITY_X),
            Scalar[gpu_dtype](GRAVITY_Y),
            Scalar[gpu_dtype](DT),
        )
        ctx.synchronize()

        # Read intermediate state
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        print("After velocity integration (before joint solver):")
        print("  Lander vy =", host_states[lander_off + IDX_VY])
        print("  Left leg vy =", host_states[left_leg_off + IDX_VY])
        print("  Right leg vy =", host_states[right_leg_off + IDX_VY])
        print()

        # Step 2: Solve joint velocity constraints (multiple iterations)
        for _ in range(VELOCITY_ITERATIONS):
            RevoluteJointSolver.solve_velocity_gpu_strided[
                BATCH_SIZE, NUM_BODIES, MAX_JOINTS, STATE_SIZE_VAL, BODIES_OFFSET, JOINTS_OFFSET
            ](ctx, states_buf, joint_counts_buf, Scalar[gpu_dtype](DT))
        ctx.synchronize()

        # Read final state
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var lander_vy_final = Float64(host_states[lander_off + IDX_VY])
        var left_leg_vy_final = Float64(host_states[left_leg_off + IDX_VY])
        var right_leg_vy_final = Float64(host_states[right_leg_off + IDX_VY])

        print("After joint solver (" + String(VELOCITY_ITERATIONS) + " iterations):")
        print("  Lander vy =", lander_vy_final)
        print("  Left leg vy =", left_leg_vy_final)
        print("  Right leg vy =", right_leg_vy_final)
        print()

        print("Summary:")
        print("  Expected lander dvy (before joints) = 0.147")
        print("  Actual lander dvy (after joints) =", lander_vy_final)
        print("  Joint solver reduced lander vy by:", 0.147 - lander_vy_final)

    print_separator()


fn test_without_joints() raises:
    """Test velocity integration without joint solving (for comparison)."""
    print_separator()
    print("Velocity Integration Only (No Joints) Test")
    print_separator()

    with DeviceContext() as ctx:
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * STATE_SIZE_VAL)
        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](STATE_SIZE_VAL)

        # Initialize lander body only
        var lander_off = BODIES_OFFSET
        host_states[lander_off + IDX_X] = Scalar[gpu_dtype](10.0)
        host_states[lander_off + IDX_Y] = Scalar[gpu_dtype](8.0)
        host_states[lander_off + IDX_ANGLE] = Scalar[gpu_dtype](0.0)
        host_states[lander_off + IDX_VX] = Scalar[gpu_dtype](0.0)
        host_states[lander_off + IDX_VY] = Scalar[gpu_dtype](0.0)
        host_states[lander_off + IDX_OMEGA] = Scalar[gpu_dtype](0.0)
        host_states[lander_off + IDX_INV_MASS] = Scalar[gpu_dtype](1.0 / LANDER_MASS)
        host_states[lander_off + IDX_INV_INERTIA] = Scalar[gpu_dtype](1.0 / LANDER_INERTIA)

        # Force on lander
        var main_offset = 4.0 / SCALE
        var main_impulse = main_offset * MAIN_ENGINE_POWER
        var main_force = main_impulse / DT

        host_states[FORCES_OFFSET + 1] = Scalar[gpu_dtype](main_force)

        print("Initial state:")
        print("  Lander vy =", host_states[lander_off + IDX_VY])
        print()

        # Copy to GPU
        ctx.enqueue_copy(states_buf, host_states)
        ctx.synchronize()

        # Integrate velocities (no joint solving)
        SemiImplicitEuler.integrate_velocities_gpu_strided[
            BATCH_SIZE, NUM_BODIES, STATE_SIZE_VAL, BODIES_OFFSET, FORCES_OFFSET
        ](
            ctx,
            states_buf,
            Scalar[gpu_dtype](GRAVITY_X),
            Scalar[gpu_dtype](GRAVITY_Y),
            Scalar[gpu_dtype](DT),
        )
        ctx.synchronize()

        # Read final state
        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()

        var lander_vy_final = Float64(host_states[lander_off + IDX_VY])

        print("After velocity integration (no joints):")
        print("  Lander vy =", lander_vy_final)
        print("  Expected =", 0.147)

    print_separator()


fn main() raises:
    print()
    print("=" * 70)
    print("    VELOCITY + JOINT SOLVER TEST")
    print("=" * 70)
    print()

    test_without_joints()
    print()

    test_with_joints()
    print()

    print("=" * 70)
    print("    Tests completed!")
    print("=" * 70)
