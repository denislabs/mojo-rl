"""Test Forward Kinematics: CPU vs GPU.

Compares our CPU FK output (xpos, xquat, xipos) with GPU FK output for the
HalfCheetah model at multiple qpos configurations. Both should produce
identical results (up to float32 precision).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_fk_cpu_vs_gpu.mojo
"""

from testing import assert_true, TestSuite
from math import abs
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from gpu import block_idx

from physics3d.types import Model, Data
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    xpos_offset,
    xquat_offset,
    xipos_offset,
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

# float32 tolerance (GPU runs float32)
comptime POS_TOL: Float64 = 1e-4
comptime QUAT_TOL: Float64 = 1e-4


# =============================================================================
# GPU FK kernel — just calls forward_kinematics_gpu
# =============================================================================


fn fk_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
](
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= BATCH:
        return
    forward_kinematics_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
    ](env, state, model)


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_fk(
    ctx: DeviceContext,
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    model_buf: DeviceBuffer[DTYPE],
) raises:
    """Run FK on CPU and GPU with identical qpos, compare results."""
    print("--- Test:", test_name, "---")

    # === CPU FK (float64 for reference, then compare as float32) ===
    var model_cpu = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model_cpu, data_cpu)
    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](qpos_values[i])

    forward_kinematics(model_cpu, data_cpu)

    # === GPU FK ===
    # Create state buffer and set qpos
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH
    ](ctx)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_values[i])

    # Copy to GPU
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.synchronize()

    # Launch FK kernel
    var state_tensor = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var model_tensor = LayoutTensor[
        DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())

    comptime kernel_fn = fk_kernel[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
    ]

    ctx.enqueue_function[
        kernel_fn,
        kernel_fn,
    ](
        state_tensor,
        model_tensor,
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ctx.synchronize()

    # Copy results back
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # === Compare body by body ===
    var all_pass = True
    var body_names = List[String]()
    body_names.append("torso")
    body_names.append("bthigh")
    body_names.append("bshin")
    body_names.append("bfoot")
    body_names.append("fthigh")
    body_names.append("fshin")
    body_names.append("ffoot")

    comptime xpos_off = xpos_offset[NQ, NV, NBODY]()
    comptime xquat_off = xquat_offset[NQ, NV, NBODY]()
    comptime xipos_off = xipos_offset[NQ, NV, NBODY]()

    for b in range(NBODY):
        # --- xpos ---
        var cpu_px = Float64(data_cpu.xpos[b * 3 + 0])
        var cpu_py = Float64(data_cpu.xpos[b * 3 + 1])
        var cpu_pz = Float64(data_cpu.xpos[b * 3 + 2])

        var gpu_px = Float64(state_host[xpos_off + b * 3 + 0])
        var gpu_py = Float64(state_host[xpos_off + b * 3 + 1])
        var gpu_pz = Float64(state_host[xpos_off + b * 3 + 2])

        var pos_err = (
            abs(cpu_px - gpu_px) + abs(cpu_py - gpu_py) + abs(cpu_pz - gpu_pz)
        )

        if pos_err > POS_TOL:
            print("  FAIL xpos ", body_names[b], " err=", pos_err)
            print("    cpu:", cpu_px, cpu_py, cpu_pz)
            print("    gpu:", gpu_px, gpu_py, gpu_pz)
            all_pass = False
        else:
            print("  OK   xpos ", body_names[b], " err=", pos_err)

        # --- xquat ---
        var cpu_qx = Float64(data_cpu.xquat[b * 4 + 0])
        var cpu_qy = Float64(data_cpu.xquat[b * 4 + 1])
        var cpu_qz = Float64(data_cpu.xquat[b * 4 + 2])
        var cpu_qw = Float64(data_cpu.xquat[b * 4 + 3])

        var gpu_qx = Float64(state_host[xquat_off + b * 4 + 0])
        var gpu_qy = Float64(state_host[xquat_off + b * 4 + 1])
        var gpu_qz = Float64(state_host[xquat_off + b * 4 + 2])
        var gpu_qw = Float64(state_host[xquat_off + b * 4 + 3])

        # Quaternions q and -q represent same rotation
        var diff_pos = (
            abs(cpu_qx - gpu_qx)
            + abs(cpu_qy - gpu_qy)
            + abs(cpu_qz - gpu_qz)
            + abs(cpu_qw - gpu_qw)
        )
        var diff_neg = (
            abs(cpu_qx + gpu_qx)
            + abs(cpu_qy + gpu_qy)
            + abs(cpu_qz + gpu_qz)
            + abs(cpu_qw + gpu_qw)
        )
        var quat_err = diff_pos if diff_pos < diff_neg else diff_neg

        if quat_err > QUAT_TOL:
            print("  FAIL xquat", body_names[b], " err=", quat_err)
            print("    cpu (x,y,z,w):", cpu_qx, cpu_qy, cpu_qz, cpu_qw)
            print("    gpu (x,y,z,w):", gpu_qx, gpu_qy, gpu_qz, gpu_qw)
            all_pass = False
        else:
            print("  OK   xquat", body_names[b], " err=", quat_err)

        # --- xipos ---
        var cpu_xi_x = Float64(data_cpu.xipos[b * 3 + 0])
        var cpu_xi_y = Float64(data_cpu.xipos[b * 3 + 1])
        var cpu_xi_z = Float64(data_cpu.xipos[b * 3 + 2])

        var gpu_xi_x = Float64(state_host[xipos_off + b * 3 + 0])
        var gpu_xi_y = Float64(state_host[xipos_off + b * 3 + 1])
        var gpu_xi_z = Float64(state_host[xipos_off + b * 3 + 2])

        var xipos_err = (
            abs(cpu_xi_x - gpu_xi_x)
            + abs(cpu_xi_y - gpu_xi_y)
            + abs(cpu_xi_z - gpu_xi_z)
        )

        if xipos_err > POS_TOL:
            print("  FAIL xipos", body_names[b], " err=", xipos_err)
            print("    cpu:", cpu_xi_x, cpu_xi_y, cpu_xi_z)
            print("    gpu:", gpu_xi_x, gpu_xi_y, gpu_xi_z)
            all_pass = False
        else:
            print("  OK   xipos", body_names[b], " err=", xipos_err)

    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


# =============================================================================
# Test cases (same configs as test_fk_vs_mujoco.mojo)
# =============================================================================


fn test_fk_default_qpos() raises:
    print("=" * 60)
    print("FK Validation: CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NBODY=7, NQ=9)")
    print("Precision: float32")
    print("Tolerances: pos=", POS_TOL, " quat=", QUAT_TOL)
    print()

    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7  # rootz
    compare_fk(ctx, "Default qpos (rootz=0.7)", qpos, model_buf)
    print()


fn test_fk_zero_qpos() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    compare_fk(ctx, "Zero qpos (robot at origin)", qpos, model_buf)
    print()


fn test_fk_nonzero_joints() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0  # rootx
    qpos[1] = 0.7  # rootz
    qpos[2] = 0.3  # rooty
    qpos[3] = -0.4  # bthigh
    qpos[4] = 0.5  # bshin
    qpos[5] = -0.2  # bfoot
    qpos[6] = 0.6  # fthigh
    qpos[7] = -0.8  # fshin
    qpos[8] = 0.3  # ffoot
    compare_fk(ctx, "Non-zero joints", qpos, model_buf)
    print()


fn test_fk_extreme_joints() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7  # rootz
    qpos[3] = -0.52  # bthigh min
    qpos[4] = 0.785  # bshin max
    qpos[5] = -0.4  # bfoot min
    qpos[6] = -1.0  # fthigh min
    qpos[7] = 0.87  # fshin max
    qpos[8] = -0.5  # ffoot min
    compare_fk(ctx, "Extreme joint angles (at limits)", qpos, model_buf)
    print()


fn test_fk_large_rootx() raises:
    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 100.0  # rootx = 100m
    qpos[1] = 0.7  # rootz
    qpos[3] = 0.5  # bthigh
    qpos[6] = -0.5  # fthigh
    compare_fk(ctx, "Large rootx (100m)", qpos, model_buf)
    print()


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
