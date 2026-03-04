"""Test Forward Kinematics: CPU vs GPU for Ant.

Compares our CPU FK output (xpos, xquat, xipos) with GPU FK output for the
Ant model at multiple qpos configurations. Both should produce identical
results (up to float32 precision).

Ant: NQ=15, NV=14, NBODY=14 (worldbody + torso + 4 legs × 3 bodies)
  Free joint (7 qpos DOFs: x, y, z, qw, qx, qy, qz) + 8 hinge joints.
  This is the first GPU FK test with a 3D free joint (full quaternion FK).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_ant_fk_cpu_vs_gpu.mojo
"""

from testing import assert_true
from std.math import abs
from std.collections import InlineArray
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from std.gpu import block_idx

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
from physics3d.gpu.buffer_utils import create_state_buffer
from envs.ant.ant_xml import AntModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = AntModel.NQ  # 15 (7 free-joint + 8 hinge)
comptime NV = AntModel.NV  # 14 (6 free-joint + 8 hinge)
comptime NBODY = AntModel.NBODY  # 14 (worldbody + torso + 4 legs × 3 bodies)
comptime NJOINT = AntModel.NJOINT  # 9 (1 free + 8 hinge)
comptime NGEOM = AntModel.NGEOM  # 15
comptime MAX_CONTACTS = AntModel.MAX_CONTACTS  # 40
comptime NSITE = AntModel.NSITE  # 0
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()

# float32 tolerance
comptime POS_TOL: Float64 = 1e-4
comptime QUAT_TOL: Float64 = 1e-4


# =============================================================================
# GPU FK kernel
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

    # === CPU FK ===
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        AntModel.MAX_EQUALITY,
        AntModel.CONE_TYPE,
        AntModel.MAX_TENDON,
        AntModel.NSITE,
    ]()
    var data_cpu = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, AntModel.NSITE
    ]()
    AntModel.setup_model_and_data[DTYPE](model_cpu, data_cpu)
    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](qpos_values[i])
    forward_kinematics(model_cpu, data_cpu)

    # === GPU FK ===
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH
    ](ctx)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_values[i])

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.synchronize()

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
    ctx.enqueue_function[kernel_fn, kernel_fn](
        state_tensor,
        model_tensor,
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ctx.synchronize()

    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # === Compare body by body ===
    # NBODY=14: worldbody(0), torso(1),
    #   front_left_leg(2), aux_1(3), ankle_1_body(4),
    #   front_right_leg(5), aux_2(6), ankle_2_body(7),
    #   back_leg(8), aux_3(9), ankle_3_body(10),
    #   right_back_leg(11), aux_4(12), ankle_4_body(13)
    var body_names = List[String]()
    body_names.append("worldbody")
    body_names.append("torso")
    body_names.append("front_left_leg")
    body_names.append("aux_1")
    body_names.append("ankle_1_body")
    body_names.append("front_right_leg")
    body_names.append("aux_2")
    body_names.append("ankle_2_body")
    body_names.append("back_leg")
    body_names.append("aux_3")
    body_names.append("ankle_3_body")
    body_names.append("right_back_leg")
    body_names.append("aux_4")
    body_names.append("ankle_4_body")

    comptime xpos_off = xpos_offset[NQ, NV, NBODY]()
    comptime xquat_off = xquat_offset[NQ, NV, NBODY]()
    comptime xipos_off = xipos_offset[NQ, NV, NBODY]()

    var all_pass = True

    for b in range(NBODY):
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

        var cpu_qx = Float64(data_cpu.xquat[b * 4 + 0])
        var cpu_qy = Float64(data_cpu.xquat[b * 4 + 1])
        var cpu_qz = Float64(data_cpu.xquat[b * 4 + 2])
        var cpu_qw = Float64(data_cpu.xquat[b * 4 + 3])
        var gpu_qx = Float64(state_host[xquat_off + b * 4 + 0])
        var gpu_qy = Float64(state_host[xquat_off + b * 4 + 1])
        var gpu_qz = Float64(state_host[xquat_off + b * 4 + 2])
        var gpu_qw = Float64(state_host[xquat_off + b * 4 + 3])
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
            print("    cpu:", cpu_qx, cpu_qy, cpu_qz, cpu_qw)
            print("    gpu:", gpu_qx, gpu_qy, gpu_qz, gpu_qw)
            all_pass = False
        else:
            print("  OK   xquat", body_names[b], " err=", quat_err)

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

    assert_true(all_pass, "CPU vs GPU FK mismatch for: " + test_name)


# =============================================================================
# Test cases (same configs as test_ant_fk_vs_mujoco.mojo)
# =============================================================================


fn test_fk_ant() raises:
    print("=" * 60)
    print("FK Validation: CPU vs GPU — Ant")
    print("=" * 60)
    print("Model: Ant (NBODY=14, NQ=15, free joint + 8 hinge)")
    print("Precision: float32")
    print("Tolerances: pos=", POS_TOL, " quat=", QUAT_TOL)
    print()

    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    AntModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    # Config 1: default init_qpos from XML
    # [0, 0, 0.55, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    qpos1[2] = 0.55  # z (torso above ground)
    qpos1[3] = 1.0  # qw (identity quaternion)
    qpos1[7] = 0.0  # qx
    qpos1[8] = 1.0  # hip_1 (from init_qpos: 0)
    # Use exact init_qpos values
    qpos1[0] = 0.0
    qpos1[1] = 0.0
    qpos1[2] = 0.55
    qpos1[3] = 1.0
    qpos1[4] = 0.0
    qpos1[5] = 0.0
    qpos1[6] = 0.0
    qpos1[7] = 0.0
    qpos1[8] = 1.0
    qpos1[9] = 0.0
    qpos1[10] = -1.0
    qpos1[11] = 0.0
    qpos1[12] = -1.0
    qpos1[13] = 0.0
    qpos1[14] = 1.0
    compare_fk(
        ctx, "Default init_qpos (z=0.55, identity quat)", qpos1, model_buf
    )
    print()

    # Config 2: raised torso, identity quaternion, all joints = 0
    var qpos2 = InlineArray[Float64, NQ](fill=0.0)
    qpos2[2] = 2.0  # z raised
    qpos2[3] = 1.0  # qw (identity)
    compare_fk(ctx, "Raised torso (z=2.0, identity quat)", qpos2, model_buf)
    print()

    # Config 3: nonzero joint angles
    var qpos3 = InlineArray[Float64, NQ](fill=0.0)
    qpos3[0] = 1.0  # x
    qpos3[1] = 0.5  # y
    qpos3[2] = 0.55  # z
    qpos3[3] = 1.0  # qw
    qpos3[7] = 0.3  # hip_1
    qpos3[8] = 0.5  # ankle_1
    qpos3[9] = -0.3  # hip_2
    qpos3[10] = 0.5  # ankle_2
    qpos3[11] = 0.2  # hip_3
    qpos3[12] = -0.4  # ankle_3
    qpos3[13] = -0.2  # hip_4
    qpos3[14] = 0.4  # ankle_4
    compare_fk(ctx, "Nonzero joint angles", qpos3, model_buf)
    print()

    # Config 4: rotated torso (30° around z-axis)
    var qpos4 = InlineArray[Float64, NQ](fill=0.0)
    qpos4[2] = 0.55
    qpos4[3] = 0.866  # qw = cos(30°)
    qpos4[6] = 0.5  # qz = sin(30°) — rotation about z-axis
    compare_fk(ctx, "Rotated torso (30 deg around z)", qpos4, model_buf)
    print()

    # Config 5: extreme joint angles (at limits: hip ±30°, ankle 30–70°)
    var qpos5 = InlineArray[Float64, NQ](fill=0.0)
    qpos5[2] = 0.55
    qpos5[3] = 1.0  # qw identity
    qpos5[7] = 0.52  # hip_1 max ~30° = 0.52 rad
    qpos5[8] = 1.22  # ankle_1 max 70° = 1.22 rad
    qpos5[9] = -0.52  # hip_2 min
    qpos5[10] = -0.52  # ankle_2 min -30°
    qpos5[11] = 0.52  # hip_3
    qpos5[12] = -0.52  # ankle_3
    qpos5[13] = -0.52  # hip_4
    qpos5[14] = 1.22  # ankle_4 max
    compare_fk(ctx, "Extreme joint angles (at limits)", qpos5, model_buf)
    print()

    print("All Ant FK CPU vs GPU tests passed.")


fn main() raises:
    test_fk_ant()
