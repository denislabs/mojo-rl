"""Deterministic smoke + regression gate for PhysicsStepKernel.step_gpu (GPU).

Drives the fused physics2d GPU step directly with a hand-built, self-consistent
minimal scene (no terrain edges, no joints -> the collision/joint code paths are
skipped, so the step is pure integration and stays finite). Runs a few steps and
prints a deterministic checksum of the state/contacts/contact-count buffers.

This is the gate for the UnsafePointer->MutAnyOrigin migration of the physics2d
GPU kernels: `mojo precompile` does not instantiate generics, so only an actual
GPU run like this can catch a mutability mismatch. The checksum must be
bit-identical before and after the migration.
"""

from std.gpu.host import DeviceContext
from mojo_rl.physics2d import (
    PhysicsStepKernel,
    PhysicsStepKernelParallel,
    dtype,
)


def main() raises:
    # --- Self-consistent minimal layout (independent of any env) -----------
    comptime BATCH = 8
    comptime NUM_BODIES = 2
    comptime BODY_STATE_SIZE = 13  # physics2d constant
    comptime NUM_SHAPES = 2
    comptime SHAPE_MAX_SIZE = 20
    comptime MAX_CONTACTS = 4
    comptime CONTACT_DATA_SIZE = 9
    comptime MAX_JOINTS = 2
    comptime MAX_EDGES = 4

    comptime BODIES_OFFSET = 8
    comptime FORCES_OFFSET = BODIES_OFFSET + NUM_BODIES * BODY_STATE_SIZE  # 34
    comptime JOINTS_OFFSET = FORCES_OFFSET + NUM_BODIES * 3  # 40 (unused; n_joints=0)
    comptime EDGES_OFFSET = 100  # unused; n_edges=0
    comptime STATE_SIZE = 256

    comptime VEL_ITERATIONS = 4
    comptime POS_ITERATIONS = 2

    var ctx = DeviceContext()

    var state_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
    var shapes_buf = ctx.enqueue_create_buffer[dtype](NUM_SHAPES * SHAPE_MAX_SIZE)
    var edge_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var joint_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var contacts_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE
    )
    var contact_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH)

    # Deterministic ramp in [0, 1) for the state; zero everything else.
    var state_h = ctx.enqueue_create_host_buffer[dtype](BATCH * STATE_SIZE)
    for i in range(BATCH * STATE_SIZE):
        state_h[i] = Scalar[dtype](Float64((i * 7 + 3) % 97) / 97.0)
    ctx.enqueue_copy(state_buf, state_h)

    var shapes_h = ctx.enqueue_create_host_buffer[dtype](NUM_SHAPES * SHAPE_MAX_SIZE)
    for i in range(NUM_SHAPES * SHAPE_MAX_SIZE):
        shapes_h[i] = Scalar[dtype](0.0)
    ctx.enqueue_copy(shapes_buf, shapes_h)

    var zeros_b = ctx.enqueue_create_host_buffer[dtype](BATCH)
    for i in range(BATCH):
        zeros_b[i] = Scalar[dtype](0.0)
    ctx.enqueue_copy(edge_counts_buf, zeros_b)
    ctx.enqueue_copy(joint_counts_buf, zeros_b)
    ctx.enqueue_copy(contact_counts_buf, zeros_b)

    var contacts_h = ctx.enqueue_create_host_buffer[dtype](
        BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE
    )
    for i in range(BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE):
        contacts_h[i] = Scalar[dtype](0.0)
    ctx.enqueue_copy(contacts_buf, contacts_h)
    ctx.synchronize()

    # --- Run a few steps ---------------------------------------------------
    comptime N_STEPS = 5
    for _ in range(N_STEPS):
        PhysicsStepKernel.step_gpu[
            BATCH,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS,
            MAX_JOINTS,
            MAX_EDGES,
            STATE_SIZE,
            BODIES_OFFSET,
            FORCES_OFFSET,
            JOINTS_OFFSET,
            EDGES_OFFSET,
            VEL_ITERATIONS,
            POS_ITERATIONS,
        ](
            ctx,
            state_buf,
            shapes_buf,
            edge_counts_buf,
            joint_counts_buf,
            contacts_buf,
            contact_counts_buf,
            Scalar[dtype](0.0),    # gravity_x
            Scalar[dtype](-10.0),  # gravity_y
            Scalar[dtype](0.02),   # dt
            Scalar[dtype](0.5),    # friction
            Scalar[dtype](0.0),    # restitution
            Scalar[dtype](0.2),    # baumgarte
            Scalar[dtype](0.01),   # slop
        )
    ctx.synchronize()

    # --- Deterministic checksum -------------------------------------------
    ctx.enqueue_copy(state_h, state_buf)
    ctx.enqueue_copy(contacts_h, contacts_buf)
    var cc_h = ctx.enqueue_create_host_buffer[dtype](BATCH)
    ctx.enqueue_copy(cc_h, contact_counts_buf)
    ctx.synchronize()

    var checksum = Float64(0.0)
    var n_nonfinite = 0
    for i in range(BATCH * STATE_SIZE):
        var v = Float64(state_h[i])
        if v == v and v - v == 0.0:
            checksum += v * Float64((i % 13) + 1)
        else:
            n_nonfinite += 1
    for i in range(BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE):
        var v = Float64(contacts_h[i])
        if v == v and v - v == 0.0:
            checksum += v
        else:
            n_nonfinite += 1
    var cc_sum = Float64(0.0)
    for i in range(BATCH):
        cc_sum += Float64(cc_h[i])

    print("state_checksum =", checksum)
    print("contact_count_sum =", cc_sum)
    print("n_nonfinite =", n_nonfinite)

    # --- Parallel variant (PhysicsStepKernelParallel.step_parallel_gpu) -----
    # Collision is external for this kernel; with empty contact_flags the
    # sparse solvers are no-ops, so this stays finite too. Same fresh ramp.
    comptime MAX_CONTACTS_PER_BODY_EDGE = 2
    comptime TOTAL_SLOTS = NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE

    var pstate_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
    ctx.enqueue_copy(pstate_buf, state_h)  # reuse the original ramp (pre-step)
    var pcontacts_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * TOTAL_SLOTS * CONTACT_DATA_SIZE
    )
    var pflags_buf = ctx.enqueue_create_buffer[dtype](BATCH * TOTAL_SLOTS)
    ctx.enqueue_memset(pcontacts_buf, 0)
    ctx.enqueue_memset(pflags_buf, 0)
    # re-load the original ramp into pstate (state_h was overwritten above)
    for i in range(BATCH * STATE_SIZE):
        state_h[i] = Scalar[dtype](Float64((i * 7 + 3) % 97) / 97.0)
    ctx.enqueue_copy(pstate_buf, state_h)
    ctx.synchronize()

    for _ in range(N_STEPS):
        PhysicsStepKernelParallel.step_parallel_gpu[
            BATCH,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS_PER_BODY_EDGE,
            MAX_JOINTS,
            MAX_EDGES,
            STATE_SIZE,
            BODIES_OFFSET,
            FORCES_OFFSET,
            JOINTS_OFFSET,
            EDGES_OFFSET,
            VEL_ITERATIONS,
            POS_ITERATIONS,
        ](
            ctx,
            pstate_buf,
            shapes_buf,
            edge_counts_buf,
            joint_counts_buf,
            pcontacts_buf,
            pflags_buf,
            Scalar[dtype](0.0),
            Scalar[dtype](-10.0),
            Scalar[dtype](0.02),
            Scalar[dtype](0.5),
            Scalar[dtype](0.0),
            Scalar[dtype](0.2),
            Scalar[dtype](0.01),
        )
    ctx.synchronize()

    var pstate_h = ctx.enqueue_create_host_buffer[dtype](BATCH * STATE_SIZE)
    ctx.enqueue_copy(pstate_h, pstate_buf)
    ctx.synchronize()
    var pchecksum = Float64(0.0)
    for i in range(BATCH * STATE_SIZE):
        var v = Float64(pstate_h[i])
        if v == v and v - v == 0.0:
            pchecksum += v * Float64((i % 13) + 1)
    print("parallel_state_checksum =", pchecksum)

    print("PhysicsStepKernel.step_gpu smoke: OK")
