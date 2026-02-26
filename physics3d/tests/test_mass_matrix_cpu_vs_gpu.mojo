"""Test Mass Matrix: CPU vs GPU.

Compares our CPU mass matrix (CRBA) with GPU mass matrix for the HalfCheetah
model at multiple qpos configurations. Both should produce identical results
(up to float32 precision).

The GPU pipeline is: FK -> cdof -> composite_inertia -> mass_matrix.

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_mass_matrix_cpu_vs_gpu.mojo
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
    forward_kinematics_gpu,
)
from physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_composite_inertia,
    compute_cdof_gpu,
    compute_composite_inertia_gpu,
)
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    compute_mass_matrix_full_gpu,
)
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    integrator_workspace_size,
    ws_M_offset,
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

# CPU array sizes
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()

# Tolerance (float32)
comptime M_TOL: Float64 = 1e-3
comptime M_REL_TOL: Float64 = 1e-2


# =============================================================================
# GPU kernel: FK + cdof + CRB + mass matrix
# =============================================================================


fn mass_matrix_kernel[
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

    # 1. Forward kinematics (writes xpos, xquat, xipos to state)
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # 2. Compute cdof (writes to workspace at offset 0)
    compute_cdof_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 3. Compute composite inertia (writes to workspace at ws_crb_offset)
    compute_composite_inertia_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)

    # 4. Compute mass matrix (writes to workspace at ws_M_offset)
    compute_mass_matrix_full_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ](env, state, model, workspace)


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_mass_matrix(
    ctx: DeviceContext,
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    model_buf: DeviceBuffer[DTYPE],
) raises:
    """Compute mass matrix on CPU and GPU with identical qpos, compare."""
    print("--- Test:", test_name, "---")

    # === CPU: FK + cdof + CRB + mass matrix ===
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, data_cpu)
    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](qpos_values[i])

    forward_kinematics(model_cpu, data_cpu)

    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model_cpu, data_cpu, cdof
    )

    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
    ](model_cpu, data_cpu, crb)

    var M_cpu = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M_cpu[i] = Scalar[DTYPE](0)
    compute_mass_matrix_full[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        M_SIZE, CDOF_SIZE, CRB_SIZE,
    ](model_cpu, data_cpu, cdof, crb, M_cpu)

    # === GPU: run kernel ===
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH
    ](ctx)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_values[i])

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.synchronize()

    # Zero workspace
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # Launch kernel
    var state_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model_tensor = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())
    var ws_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ](workspace_buf.unsafe_ptr())

    comptime kernel_fn = mass_matrix_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
    ]

    ctx.enqueue_function[kernel_fn, kernel_fn](
        state_tensor, model_tensor, ws_tensor,
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ctx.synchronize()

    # Copy workspace back to read M
    ctx.enqueue_copy(ws_host.unsafe_ptr(), workspace_buf)
    ctx.synchronize()

    # === Compare M element by element ===
    comptime M_off = ws_M_offset[NV, NBODY]()
    var all_pass = True
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for i in range(NV):
        for j in range(NV):
            var cpu_val = Float64(M_cpu[i * NV + j])
            var gpu_val = Float64(ws_host[M_off + i * NV + j])
            var abs_err = abs(cpu_val - gpu_val)
            var ref_mag = abs(cpu_val)
            var rel_err: Float64 = 0.0
            if ref_mag > 1e-10:
                rel_err = abs_err / ref_mag

            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err

            var ok = abs_err < M_TOL or rel_err < M_REL_TOL
            if not ok:
                if fail_count < 10:
                    print(
                        "  FAIL M[", i, ",", j, "]",
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

    # Print diagonals for inspection
    print("  CPU M diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(M_cpu[i * NV + i]), end="")
    print()
    print("  GPU M diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(ws_host[M_off + i * NV + i]), end="")
    print()

    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


# =============================================================================
# Test cases (same configs as test_mass_matrix_vs_mujoco.mojo)
# =============================================================================


fn test_default_qpos() raises:
    print("=" * 60)
    print("Mass Matrix Validation: CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NV=9)")
    print("Precision: float32")
    print("Tolerances: abs=", M_TOL, " rel=", M_REL_TOL)
    print()

    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    compare_mass_matrix(ctx, "Default qpos (rootz=0.7)", qpos, model_buf)
    print()


fn test_zero_qpos() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    compare_mass_matrix(ctx, "Zero qpos", qpos, model_buf)
    print()


fn test_nonzero_joints() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0   # rootx
    qpos[1] = 0.7   # rootz
    qpos[2] = 0.3   # rooty
    qpos[3] = -0.4  # bthigh
    qpos[4] = 0.5   # bshin
    qpos[5] = -0.2  # bfoot
    qpos[6] = 0.6   # fthigh
    qpos[7] = -0.8  # fshin
    qpos[8] = 0.3   # ffoot
    compare_mass_matrix(ctx, "Non-zero joints", qpos, model_buf)
    print()


fn test_extreme_joints() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[3] = -0.52   # bthigh min
    qpos[4] = 0.785   # bshin max
    qpos[5] = -0.4    # bfoot min
    qpos[6] = -1.0    # fthigh min
    qpos[7] = 0.87    # fshin max
    qpos[8] = -0.5    # ffoot min
    compare_mass_matrix(ctx, "Extreme joint angles", qpos, model_buf)
    print()


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
