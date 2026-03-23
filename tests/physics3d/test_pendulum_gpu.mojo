"""GPU Pendulum Test for Generalized Coordinates (GC) Engine.

Tests the GPU implementation of the GC physics engine by simulating a simple pendulum.

Verifies:
1. Period matches analytical T = 2*pi*sqrt(I/(m*g*L)) (within 5%)
2. Energy conservation (drift < 10% over 5 periods)
3. InlineArray works correctly in GPU kernels

Run with:
    cd mojo-rl
    pixi run -e apple mojo run physics3d/tests/test_pendulum_gpu.mojo
"""

from std.testing import assert_true
from std.math import sqrt, sin, cos, pi
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.integrator import DefaultIntegrator
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size,
    qpos_offset,
    qvel_offset,
    xpos_offset,
    integrator_workspace_size,
)
from mojo_rl.physics3d.solver import PGSSolver
from mojo_rl.physics3d.gpu.buffer_utils import (
    create_state_buffer,
    create_model_buffer,
    copy_model_to_buffer,
    copy_data_to_buffer,
)


# Configuration
comptime DTYPE = DType.float32
comptime NQ: Int = 1  # Single hinge joint = 1 qpos
comptime NV: Int = 1  # Single hinge joint = 1 qvel
comptime NBODY: Int = 2  # worldbody + 1 real body
comptime NJOINT: Int = 1
comptime MAX_CONTACTS: Int = 5
comptime BATCH: Int = 1


def test_pendulum_gpu() raises:
    print("=" * 60)
    print("    GPU Pendulum Test")
    print("=" * 60)
    print()

    # Physics parameters
    var L: Float32 = 1.0  # Pendulum length
    var mass: Float32 = 1.0
    var g: Float32 = 9.81
    var I_cm: Float32 = 0.01  # Body inertia at CoM
    var I_pivot = I_cm + mass * L * L  # Total inertia about pivot
    var expected_period = (
        Float32(2.0) * Float32(pi) * sqrt(I_pivot / (mass * g * L))
    )

    var initial_angle: Float32 = 0.3  # ~17 degrees
    var dt: Float32 = 0.001
    var ground_z: Float32 = -10.0  # Below pendulum

    print("Physics setup:")
    print("  Pendulum length:", L, "m")
    print("  Mass:", mass, "kg")
    print("  Gravity:", g, "m/s^2")
    print(
        "  Initial angle:",
        initial_angle,
        "rad (~",
        initial_angle * 180.0 / Float32(pi),
        "deg)",
    )
    print("  Expected period:", expected_period, "s")
    print()

    # Compute buffer sizes
    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
    comptime MODEL_SIZE = model_size[NBODY, NJOINT]()
    print("Buffer sizes:")
    print("  State size:", STATE_SIZE, "floats")
    print("  Model size:", MODEL_SIZE, "floats")
    print()

    # Create CPU model and data first
    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, 0]()
    model.gravity = SIMD[DTYPE, 4](0, 0, Scalar[DTYPE](-g), 0)
    model.timestep = Scalar[DTYPE](dt)

    # Set body properties
    model.set_body(
        1,
        name="bob",
        mass=Scalar[DTYPE](mass),
        inertia=(Scalar[DTYPE](I_cm), Scalar[DTYPE](I_cm), Scalar[DTYPE](I_cm)),
    )
    model.set_body_parent(1, 0)  # Parent is worldbody
    model.set_body_local_frame(
        1,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
    )
    model.set_body_ipos_iquat(
        1,
        ipos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](-L)),
    )

    # Add hinge joint at origin with Y axis
    _ = model.add_hinge_joint(
        body_id=1,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
    )

    # Initialize data
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    data.qpos[0] = Scalar[DTYPE](initial_angle)
    data.qvel[0] = Scalar[DTYPE](0.0)

    # Create GPU context
    var ctx = DeviceContext()
    print("GPU device initialized")

    # Create host buffers
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, 0, BATCH
    ](ctx)
    var model_host = create_model_buffer[DTYPE, NBODY, NJOINT](ctx)

    # Copy model and data to host buffers
    copy_model_to_buffer[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        model, model_host
    )
    copy_data_to_buffer[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        data, state_host, 0
    )

    # Create GPU buffers
    comptime WS_SIZE = integrator_workspace_size[
        NV, NBODY
    ]() + NV * NV + PGSSolver.solver_workspace_size[NV, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)

    # Copy to GPU
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    print("Initial state copied to GPU")
    print()

    # Run simulation
    var num_periods = 5
    var sim_time = Float32(num_periods) * expected_period
    var num_steps = Int(sim_time / dt)

    print("Running", num_steps, "steps on GPU (~", sim_time, "s)...")

    # Track zero crossings to measure period
    var prev_qpos: Float32 = initial_angle
    var zero_crossings = 0
    var first_crossing_time: Float32 = 0.0
    var last_crossing_time: Float32 = 0.0
    var current_time: Float32 = 0.0

    # Track energy
    var initial_energy: Float32 = 0.0
    var max_energy_deviation: Float32 = 0.0
    var energy_computed = False

    for step in range(num_steps):
        # Run one step on GPU
        DefaultIntegrator.step_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH
        ](
            ctx,
            state_buf,
            model_buf,
            workspace_buf,
        )
        current_time = current_time + dt

        # Read state periodically
        if (step + 1) % 100 == 0 or step == 0:
            ctx.synchronize()
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()

            var qpos_off = qpos_offset[NQ, NV]()
            var qvel_off = qvel_offset[NQ, NV]()
            var xpos_off = xpos_offset[NQ, NV, NBODY]()

            var qpos = Float32(state_host[qpos_off])
            var qvel = Float32(state_host[qvel_off])
            var body_z = Float32(state_host[xpos_off + 1 * 3 + 2])  # body 1, z

            # Compute energy: PE + KE
            var h = body_z + L  # Height relative to lowest point
            var PE = mass * g * h
            var I = mass * L * L
            var KE = Float32(0.5) * I * qvel * qvel
            var current_energy = PE + KE

            if not energy_computed:
                initial_energy = current_energy
                energy_computed = True
                print("  Initial energy:", initial_energy, "J")

            var deviation = abs_f32(current_energy - initial_energy)
            if deviation > max_energy_deviation:
                max_energy_deviation = deviation

            # Detect zero crossing (positive to negative)
            if prev_qpos > 0.0 and qpos <= 0.0:
                zero_crossings += 1
                if zero_crossings == 1:
                    first_crossing_time = current_time
                last_crossing_time = current_time

            prev_qpos = qpos

        # Print progress
        if (step + 1) % 1000 == 0:
            ctx.synchronize()
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()

            var qpos_off = qpos_offset[NQ, NV]()
            var xpos_off = xpos_offset[NQ, NV, NBODY]()

            var qpos = Float32(state_host[qpos_off])
            var body_x = Float32(state_host[xpos_off + 1 * 3 + 0])  # body 1, x
            var body_z = Float32(state_host[xpos_off + 1 * 3 + 2])  # body 1, z

            print(
                "  Step",
                step + 1,
                ": t =",
                current_time,
                "s",
                ", qpos =",
                qpos,
                "rad",
                ", pos = (",
                body_x,
                ",",
                body_z,
                ")",
            )

    ctx.synchronize()

    # Final results
    print()
    print("=" * 60)
    print("Results:")
    print()

    # Period measurement
    if zero_crossings >= 2:
        var num_measured_periods = zero_crossings - 1
        var measured_period = (
            last_crossing_time - first_crossing_time
        ) / Float32(num_measured_periods)
        var period_error_pct = (
            abs_f32(measured_period - expected_period) / expected_period * 100.0
        )

        print("  Expected period:", expected_period, "s")
        print("  Measured period:", measured_period, "s")
        print("  Period error:", period_error_pct, "%")

        if period_error_pct < 5.0:
            print("  PERIOD TEST: PASSED (error < 5%)")
        else:
            print("  PERIOD TEST: FAILED (error >= 5%)")
        assert_true(
            period_error_pct < 5.0,
            "Period error too large: " + String(period_error_pct) + "% >= 5%",
        )
    else:
        print(
            "  PERIOD TEST: FAILED (not enough zero crossings:",
            zero_crossings,
            ")",
        )
        assert_true(
            False,
            "Not enough zero crossings for period measurement: "
            + String(zero_crossings),
        )

    print()

    # Energy conservation
    var energy_drift_pct = max_energy_deviation / initial_energy * 100.0
    print("  Initial energy:", initial_energy, "J")
    print("  Max energy deviation:", max_energy_deviation, "J")
    print("  Energy drift:", energy_drift_pct, "%")

    if energy_drift_pct < 10.0:
        print("  ENERGY TEST: PASSED (drift < 10%)")
    else:
        print("  ENERGY TEST: FAILED (drift >= 10%)")
    assert_true(
        energy_drift_pct < 10.0,
        "Energy drift too large: " + String(energy_drift_pct) + "% >= 10%",
    )

    print()
    print("=" * 60)


def main() raises:
    test_pendulum_gpu()


def abs_f32(x: Float32) -> Float32:
    if x < 0:
        return -x
    return x
