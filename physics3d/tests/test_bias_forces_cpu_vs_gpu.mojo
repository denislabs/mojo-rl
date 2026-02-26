"""Test Bias Forces (RNE): CPU vs GPU.

Compares our CPU bias forces (RNE) with GPU bias forces for the HalfCheetah
model at multiple configurations. Both should produce identical results
(up to float32 precision).

The GPU pipeline is: FK -> cdof -> bias_forces_rne.

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_bias_forces_cpu_vs_gpu.mojo
"""

from testing import assert_true, TestSuite
from math import abs
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from gpu import block_idx

from physics3d.types import Model, Data, _max_one
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
    forward_kinematics_gpu,
)
from physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_cdof_gpu,
)
from physics3d.dynamics.bias_forces import (
    compute_bias_forces_rne,
    compute_bias_forces_rne_gpu,
)
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    integrator_workspace_size,
    ws_bias_offset,
)
from physics3d.gpu.buffer_utils import (
    create_state_buffer,
    copy_data_to_buffer,
)
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY  # 7
comptime NJOINT = HalfCheetahModel.NJOINT  # 9
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS  # 20
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime WS_SIZE = integrator_workspace_size[NV, NBODY]()

comptime V_SIZE = _max_one[NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()

# Tolerance (float32)
comptime ABS_TOL: Float64 = 1e-2
comptime REL_TOL: Float64 = 1e-2


# =============================================================================
# GPU kernel: FK + cdof + bias_forces_rne
# =============================================================================


fn bias_forces_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    var env = Int(block_idx.x)
    if env >= BATCH:
        return

    # 1. Forward kinematics
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # 2. Compute cdof
    compute_cdof_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 3. Compute bias forces (reads cdof from workspace, writes bias to workspace)
    compute_bias_forces_rne_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)


# =============================================================================
# Comparison helper (runs one test scenario, reuses buffers)
# =============================================================================


fn compare_bias_forces(
    ctx: DeviceContext,
    test_name: String,
    test_qpos: InlineArray[Float64, NQ],
    test_qvel: InlineArray[Float64, NV],
    model_cpu: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE],
    model_buf: DeviceBuffer[DTYPE],
    mut state_host: HostBuffer[DTYPE],
    mut state_buf: DeviceBuffer[DTYPE],
    mut workspace_buf: DeviceBuffer[DTYPE],
    mut ws_host: HostBuffer[DTYPE],
) raises:
    print("--- Test:", test_name, "---")

    # === CPU pipeline ===
    var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](test_qpos[i])
    for i in range(NV):
        data_cpu.qvel[i] = Scalar[DTYPE](test_qvel[i])

    forward_kinematics(model_cpu, data_cpu)
    compute_body_velocities(model_cpu, data_cpu)

    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model_cpu, data_cpu, cdof
    )

    var bias_cpu = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        bias_cpu[i] = Scalar[DTYPE](0)
    compute_bias_forces_rne[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
    ](model_cpu, data_cpu, cdof, bias_cpu)

    # === GPU pipeline (reuse buffers) ===
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](test_qpos[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](test_qvel[i])
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())

    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    comptime kernel_fn = bias_forces_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ]

    var state_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model_tensor = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())
    var ws_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ](workspace_buf.unsafe_ptr())

    ctx.enqueue_function[kernel_fn, kernel_fn](
        state_tensor, model_tensor, ws_tensor,
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ctx.synchronize()

    ctx.enqueue_copy(ws_host.unsafe_ptr(), workspace_buf)
    ctx.synchronize()

    # === Compare ===
    comptime bias_off = ws_bias_offset[NV, NBODY]()
    var all_pass = True
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for i in range(NV):
        var cpu_val = Float64(bias_cpu[i])
        var gpu_val = Float64(ws_host[bias_off + i])
        var abs_err = abs(cpu_val - gpu_val)
        var ref_mag = abs(cpu_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        var ok = abs_err < ABS_TOL or rel_err < REL_TOL
        if not ok:
            print(
                "  FAIL bias[", i, "]",
                " cpu=", cpu_val,
                " gpu=", gpu_val,
                " abs_err=", abs_err,
                " rel_err=", rel_err,
            )
            fail_count += 1
            all_pass = False

    if all_pass:
        print(
            "  ALL OK  max_abs_err=", max_abs_err,
            " max_rel_err=", max_rel_err,
        )
    else:
        print(
            "  FAILED", fail_count, "elements  max_abs_err=", max_abs_err,
            " max_rel_err=", max_rel_err,
        )

    print("  CPU bias:", end="")
    for i in range(NV):
        print(" ", Float64(bias_cpu[i]), end="")
    print()
    print("  GPU bias:", end="")
    for i in range(NV):
        print(" ", Float64(ws_host[bias_off + i]), end="")
    print()

    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


fn test_default_qpos_zero_vel() raises:
    print("=" * 60)
    print("Bias Forces (RNE) Validation: CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NV=9)")
    print("Precision: float32")
    print("Tolerances: abs=", ABS_TOL, " rel=", REL_TOL)
    print()

    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_bias_forces(ctx, "Default qpos, zero vel (gravity only)", qpos, qvel, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_zero_qpos_zero_vel() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_bias_forces(ctx, "Zero qpos, zero vel", qpos, qvel, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_nonzero_joints_zero_vel() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0
    qpos[1] = 0.7
    qpos[2] = 0.3
    qpos[3] = -0.4
    qpos[4] = 0.5
    qpos[5] = -0.2
    qpos[6] = 0.6
    qpos[7] = -0.8
    qpos[8] = 0.3
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_bias_forces(ctx, "Non-zero joints, zero vel", qpos, qvel, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_nonzero_vel_gravity_coriolis() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[2] = 0.1
    qpos[3] = -0.3
    qpos[6] = 0.4
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0
    qvel[2] = 0.5
    qvel[3] = -1.0
    qvel[4] = 0.8
    qvel[6] = 1.2
    qvel[7] = -0.6
    compare_bias_forces(ctx, "Non-zero vel (gravity + Coriolis)", qpos, qvel, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_extreme_velocities() raises:
    var ctx = DeviceContext()
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, _setup_data)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[3] = -0.52
    qpos[6] = -1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 5.0
    qvel[1] = -2.0
    qvel[2] = 3.0
    qvel[3] = -5.0
    qvel[4] = 5.0
    qvel[5] = -3.0
    qvel[6] = 5.0
    qvel[7] = -5.0
    qvel[8] = 3.0
    compare_bias_forces(ctx, "Extreme velocities", qpos, qvel, model_cpu, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
