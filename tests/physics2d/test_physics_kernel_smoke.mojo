"""Deterministic smoke + regression gate for PhysicsKernel GPU step variants.

Drives PhysicsKernel.step_gpu_edge_terrain / step_gpu_flat_terrain /
step_gpu_with_joints with a hand-built minimal scene (no terrain edges, no
joints -> collision/joint solves are no-ops, integration stays finite). Each
variant instantiates its batch-kernel call tree:

  edge_terrain: euler.integrate_*_gpu + EdgeTerrainCollision.detect_gpu
                + ImpulseSolver.solve_*_gpu
  flat_terrain: euler + FlatTerrainCollision.detect_gpu + ImpulseSolver
  with_joints : euler + EdgeTerrainCollision.detect_gpu + ImpulseSolver
                + RevoluteJointSolver.solve_*_gpu

This is the gate for migrating those kernels off the UnsafeAnyOrigin hatch:
`mojo precompile` doesn't instantiate generics, so only a GPU run instantiates
the whole tree. Checksums must be bit-identical before and after the migration.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.physics2d import PhysicsKernel, PhysicsConfig, dtype

comptime BATCH = 8
comptime NUM_BODIES = 2
comptime BODY_STATE_SIZE = 13
comptime NUM_SHAPES = 2
comptime SHAPE_MAX_SIZE = 20
comptime MAX_CONTACTS = 4
comptime CONTACT_DATA_SIZE = 9
comptime MAX_JOINTS = 2
comptime MAX_EDGES = 4

comptime BODIES_OFFSET = 8
comptime FORCES_OFFSET = BODIES_OFFSET + NUM_BODIES * BODY_STATE_SIZE
comptime JOINTS_OFFSET = FORCES_OFFSET + NUM_BODIES * 3
comptime EDGES_OFFSET = 100
comptime STATE_SIZE = 256


def fresh_state(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
    """A state buffer filled with a fixed [0,1) ramp."""
    var buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
    var host = ctx.enqueue_create_host_buffer[dtype](BATCH * STATE_SIZE)
    for i in range(BATCH * STATE_SIZE):
        host[i] = Scalar[dtype](Float64((i * 7 + 3) % 97) / 97.0)
    ctx.enqueue_copy(buf, host)
    ctx.synchronize()
    return buf


def zeros(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dtype]:
    var buf = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_memset(buf, 0)
    ctx.synchronize()
    return buf


def checksum(ctx: DeviceContext, buf: DeviceBuffer[dtype]) raises -> Float64:
    var host = ctx.enqueue_create_host_buffer[dtype](BATCH * STATE_SIZE)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var acc = Float64(0.0)
    for i in range(BATCH * STATE_SIZE):
        var v = Float64(host[i])
        if v == v and v - v == 0.0:
            acc += v * Float64((i % 13) + 1)
    return acc


def main() raises:
    var ctx = DeviceContext()
    var config = PhysicsConfig()
    comptime N_STEPS = 5

    # --- edge terrain ------------------------------------------------------
    var s1 = fresh_state(ctx)
    var shapes = zeros(ctx, NUM_SHAPES * SHAPE_MAX_SIZE)
    var edge_counts = zeros(ctx, BATCH)
    var contacts = zeros(ctx, BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE)
    var contact_counts = zeros(ctx, BATCH)
    for _ in range(N_STEPS):
        PhysicsKernel.step_gpu_edge_terrain[
            BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS, MAX_EDGES,
            STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, EDGES_OFFSET,
        ](ctx, s1, shapes, edge_counts, contacts, contact_counts, config)
    print("edge_terrain_checksum =", checksum(ctx, s1))

    # --- flat terrain ------------------------------------------------------
    var s2 = fresh_state(ctx)
    var contacts2 = zeros(ctx, BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE)
    var contact_counts2 = zeros(ctx, BATCH)
    for _ in range(N_STEPS):
        PhysicsKernel.step_gpu_flat_terrain[
            BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS,
            STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET,
        ](ctx, s2, shapes, contacts2, contact_counts2, config, -1.0)
    print("flat_terrain_checksum =", checksum(ctx, s2))

    # --- with joints -------------------------------------------------------
    var s3 = fresh_state(ctx)
    var edge_counts3 = zeros(ctx, BATCH)
    var contacts3 = zeros(ctx, BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE)
    var contact_counts3 = zeros(ctx, BATCH)
    var joint_counts3 = zeros(ctx, BATCH)
    for _ in range(N_STEPS):
        PhysicsKernel.step_gpu_with_joints[
            BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS, MAX_JOINTS, MAX_EDGES,
            STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, JOINTS_OFFSET, EDGES_OFFSET,
        ](
            ctx, s3, shapes, edge_counts3, contacts3, contact_counts3,
            joint_counts3, config,
        )
    print("with_joints_checksum =", checksum(ctx, s3))

    print("PhysicsKernel GPU smoke: OK")
