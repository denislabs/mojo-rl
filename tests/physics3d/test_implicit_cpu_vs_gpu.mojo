"""Test Implicit (full) integrator: CPU vs GPU.

Compares qpos/qvel after running physics steps using ImplicitIntegrator[PGSSolver]
on CPU vs GPU for the HalfCheetah model.

The full Implicit integrator includes the RNE velocity derivative (qDeriv),
so this test validates the GPU compute_rne_vel_derivative_gpu function.

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_implicit_cpu_vs_gpu.mojo
"""

from std.testing import assert_true, TestSuite
from std.math import abs
from std.collections import InlineArray
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.implicit_integrator import ImplicitIntegrator
from mojo_rl.physics3d.solver.pgs_solver import PGSSolver
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    integrator_workspace_size,
    implicit_extra_workspace_size,
)
from mojo_rl.physics3d.gpu.buffer_utils import (
    create_state_buffer,
)
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM
comptime BATCH = 1

comptime NSITE = HalfCheetahModel.NSITE
comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime WS_SIZE = (
    integrator_workspace_size[NV, NBODY]()
    + NV * NV
    + PGSSolver.solver_workspace_size[NV, MAX_CONTACTS]()
    + implicit_extra_workspace_size[NV, NBODY]()
)

# Tolerances — at zero velocity, GPU and CPU should match closely since
# the qDeriv is zero. At nonzero velocity, PGS iteration differences + float32
# can cause divergence.
comptime QPOS_ABS_TOL: Float64 = 5e-2
comptime QPOS_REL_TOL: Float64 = 3e-1
comptime QVEL_ABS_TOL: Float64 = 1.0
comptime QVEL_REL_TOL: Float64 = 5e-1

comptime Integrator = ImplicitIntegrator[SOLVER=PGSSolver]


# =============================================================================
# Compare helper
# =============================================================================


def compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int,
    ctx: DeviceContext,
    mut model_buf: DeviceBuffer[DTYPE],
    mut state_host: HostBuffer[DTYPE],
    mut state_buf: DeviceBuffer[DTYPE],
    mut workspace_buf: DeviceBuffer[DTYPE],
    mut ws_host: HostBuffer[DTYPE],
) raises:
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === CPU pipeline ===
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var data_cpu = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data(model_cpu, data_cpu)

    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data_cpu.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    for _ in range(num_steps):
        for i in range(NV):
            data_cpu.qfrc[i] = Scalar[DTYPE](0)
        HalfCheetahModel.apply_actions(data_cpu, action_list)
        Integrator.step[NGEOM=NGEOM](model_cpu, data_cpu)

    # === GPU pipeline ===
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](qvel_init[i])

    var data_temp = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.apply_actions(data_temp, action_list)
    for i in range(NV):
        state_host[qfrc_offset[NQ, NV]() + i] = data_temp.qfrc[i]

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    for step in range(num_steps):
        if step > 0:
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()
            for i in range(NV):
                state_host[qfrc_offset[NQ, NV]() + i] = data_temp.qfrc[i]
            ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
            for i in range(BATCH * WS_SIZE):
                ws_host[i] = Scalar[DTYPE](0)
            ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
            ctx.synchronize()

        Integrator.step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM=NGEOM,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
        ](
            ctx,
            state_buf,
            model_buf,
            workspace_buf,
        )
        ctx.synchronize()

    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # === Compare ===
    var qpos_pass = True
    var qpos_max_abs: Float64 = 0.0
    var qvel_pass = True
    var qvel_max_abs: Float64 = 0.0
    var qpos_fails = 0
    var qvel_fails = 0

    for i in range(NQ):
        var cpu_val = Float64(data_cpu.qpos[i])
        var gpu_val = Float64(state_host[qpos_offset[NQ, NV]() + i])
        var abs_err = abs(cpu_val - gpu_val)
        var ref_mag = abs(cpu_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag
        if abs_err > qpos_max_abs:
            qpos_max_abs = abs_err
        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            if qpos_fails < 5:
                print(
                    "  FAIL qpos[",
                    i,
                    "]",
                    " cpu=",
                    cpu_val,
                    " gpu=",
                    gpu_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qpos_fails += 1
            qpos_pass = False

    for i in range(NV):
        var cpu_val = Float64(data_cpu.qvel[i])
        var gpu_val = Float64(state_host[qvel_offset[NQ, NV]() + i])
        var abs_err = abs(cpu_val - gpu_val)
        var ref_mag = abs(cpu_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag
        if abs_err > qvel_max_abs:
            qvel_max_abs = abs_err
        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            if qvel_fails < 5:
                print(
                    "  FAIL qvel[",
                    i,
                    "]",
                    " cpu=",
                    cpu_val,
                    " gpu=",
                    gpu_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass
    if all_pass:
        print(
            "  ALL OK  qpos_max_abs=",
            qpos_max_abs,
            " qvel_max_abs=",
            qvel_max_abs,
        )
    else:
        print(
            "  FAILED  qpos:",
            qpos_fails,
            "fails (max_abs=",
            qpos_max_abs,
            ")",
            " qvel:",
            qvel_fails,
            "fails (max_abs=",
            qvel_max_abs,
            ")",
        )
    print("  CPU contacts:", Int(data_cpu.num_contacts))

    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


def test_free_fall_zero_vel_1_step() raises:
    print("=" * 60)
    print("Implicit (full) + PGS: CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NQ=9, NV=9, NGEOM=", NGEOM, ")")
    print("Integrator: ImplicitIntegrator + PGSSolver (elliptic)")
    print("Precision: float32")
    print()

    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Free fall, zero vel (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


def test_free_fall_with_actions_zero_vel_1_step() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act[0] = 0.5
    act[1] = -0.3
    act[2] = 0.2
    act[3] = 0.5
    act[4] = -0.3
    act[5] = 0.1
    compare_step(
        "Free fall + actions, zero vel (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


def test_free_fall_nonzero_vel_1_step() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0  # root x
    qvel[2] = 0.5  # root y
    qvel[3] = -0.3  # bthigh
    qvel[5] = 0.7  # fthigh
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Free fall, nonzero vel (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


def test_free_fall_nonzero_vel_with_actions_1_step() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0
    qvel[2] = 0.5
    qvel[3] = -0.3
    qvel[5] = 0.7
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act[0] = 0.5
    act[1] = -0.3
    act[2] = 0.2
    act[3] = 0.5
    act[4] = -0.3
    act[5] = 0.1
    compare_step(
        "Free fall, nonzero vel + actions (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


def test_free_fall_nonzero_vel_5_steps() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0
    qvel[2] = 0.5
    qvel[3] = -0.3
    qvel[5] = 0.7
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Free fall, nonzero vel (5 steps)",
        qpos,
        qvel,
        act,
        5,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


def test_ground_contact_zero_vel_1_step() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.2
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Ground contact, zero vel (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


def test_ground_contact_nonzero_vel_1_step() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.2
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0
    qvel[2] = 0.5
    qvel[3] = -0.3
    qvel[5] = 0.7
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Ground contact, nonzero vel (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


def test_ground_contact_with_actions_1_step() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.2
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act[0] = 0.5
    act[1] = -0.3
    act[2] = 0.2
    act[3] = 0.5
    act[4] = -0.3
    act[5] = 0.1
    compare_step(
        "Ground contact + actions (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
