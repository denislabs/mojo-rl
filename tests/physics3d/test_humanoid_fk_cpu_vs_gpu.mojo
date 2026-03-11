"""Test Forward Kinematics: CPU vs GPU for Humanoid.

Compares our CPU FK output (xpos, xquat, xipos) with GPU FK output for the
Humanoid model at multiple qpos configurations. Both should produce identical
results (up to float32 precision).

Humanoid: NQ=24, NV=23, NBODY=14 (worldbody + 13 bodies)
  Free joint + 17 hinge joints. 2 tendons. Most complex model tested.
  Default standing: qpos[2]=1.4 (torso z), qpos[3]=1.0 (qw, identity).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_humanoid_fk_cpu_vs_gpu.mojo
"""

from std.testing import assert_true
from std.math import abs
from std.collections import InlineArray
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from std.gpu import block_idx

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    xpos_offset,
    xquat_offset,
    xipos_offset,
)
from mojo_rl.physics3d.gpu.buffer_utils import create_state_buffer
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HumanoidModel.NQ  # 24 (7 free + 17 hinge)
comptime NV = HumanoidModel.NV  # 23 (6 free + 17 hinge)
comptime NBODY = HumanoidModel.NBODY  # 14 (worldbody + 13 bodies)
comptime NJOINT = HumanoidModel.NJOINT  # 18 (1 free + 17 hinge)
comptime NGEOM = HumanoidModel.NGEOM  # 18
comptime MAX_CONTACTS = HumanoidModel.MAX_CONTACTS  # 50
comptime NSITE = HumanoidModel.NSITE  # 0
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()

# float32 tolerance — slightly relaxed vs Ant because of the deeply-nested
# pelvis→thigh→shin→foot chain and lwaist body quat (~0.1° rotation)
comptime POS_TOL: Float64 = 1e-3
comptime QUAT_TOL: Float64 = 1e-3


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
        HumanoidModel.MAX_EQUALITY,
        HumanoidModel.CONE_TYPE,
        HumanoidModel.MAX_TENDON,
        HumanoidModel.NSITE,
    ]()
    var data_cpu = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HumanoidModel.NSITE
    ]()
    HumanoidModel.setup_model_and_data[DTYPE](model_cpu, data_cpu)
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
    # NBODY=14: worldbody(0), torso(1), lwaist(2), pelvis(3),
    #   right_thigh(4), right_shin(5), right_foot(6),
    #   left_thigh(7), left_shin(8), left_foot(9),
    #   right_upper_arm(10), right_lower_arm(11),
    #   left_upper_arm(12), left_lower_arm(13)
    var body_names = List[String]()
    body_names.append("worldbody")
    body_names.append("torso")
    body_names.append("lwaist")
    body_names.append("pelvis")
    body_names.append("right_thigh")
    body_names.append("right_shin")
    body_names.append("right_foot")
    body_names.append("left_thigh")
    body_names.append("left_shin")
    body_names.append("left_foot")
    body_names.append("right_upper_arm")
    body_names.append("right_lower_arm")
    body_names.append("left_upper_arm")
    body_names.append("left_lower_arm")

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
# Test cases (same configs as test_humanoid_fk_vs_mujoco.mojo)
# =============================================================================


fn test_fk_humanoid() raises:
    print("=" * 60)
    print("FK Validation: CPU vs GPU — Humanoid")
    print("=" * 60)
    print("Model: Humanoid (NBODY=14, NQ=24, free joint + 17 hinge)")
    print("Precision: float32")
    print("Tolerances: pos=", POS_TOL, " quat=", QUAT_TOL)
    print()

    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HumanoidModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    # Config 1: default standing (torso at z=1.4, identity quaternion)
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    qpos1[2] = 1.4  # z (torso height)
    qpos1[3] = 1.0  # qw (identity quaternion)
    compare_fk(ctx, "Default standing (z=1.4, identity quat)", qpos1, model_buf)
    print()

    # Config 2: bent knees
    var qpos2 = InlineArray[Float64, NQ](fill=0.0)
    qpos2[2] = 1.4
    qpos2[3] = 1.0
    qpos2[10] = -0.5  # right_knee
    qpos2[13] = -0.5  # left_knee
    compare_fk(
        ctx, "Bent knees (right_knee=-0.5, left_knee=-0.5)", qpos2, model_buf
    )
    print()

    # Config 3: arms extended
    var qpos3 = InlineArray[Float64, NQ](fill=0.0)
    qpos3[2] = 1.4
    qpos3[3] = 1.0
    qpos3[14] = 0.8  # right_shoulder1
    qpos3[15] = 0.3  # right_shoulder2
    qpos3[17] = 0.8  # left_shoulder1
    qpos3[18] = -0.3  # left_shoulder2
    compare_fk(ctx, "Arms extended", qpos3, model_buf)
    print()

    # Config 4: rotated torso (45° around z-axis)
    var qpos4 = InlineArray[Float64, NQ](fill=0.0)
    qpos4[2] = 1.4
    qpos4[3] = 0.924  # qw = cos(22.5°) for 45° rotation
    qpos4[6] = 0.383  # qz = sin(22.5°)
    compare_fk(ctx, "Rotated torso (45 deg around z)", qpos4, model_buf)
    print()

    # Config 5: full body pose
    var qpos5 = InlineArray[Float64, NQ](fill=0.0)
    qpos5[0] = 0.5  # x
    qpos5[1] = 0.2  # y
    qpos5[2] = 1.4  # z
    qpos5[3] = 0.99  # qw
    qpos5[4] = 0.1  # qx
    qpos5[5] = 0.05  # qy
    qpos5[6] = 0.0  # qz
    qpos5[7] = 0.2  # abdomen_z
    qpos5[8] = 0.1  # abdomen_y
    qpos5[9] = -0.1  # abdomen_x
    qpos5[10] = -0.3  # right_hip_x
    qpos5[11] = 0.2  # right_hip_z
    qpos5[12] = -0.5  # right_hip_y
    qpos5[13] = -0.8  # right_knee
    qpos5[14] = 0.5  # right_shoulder1
    qpos5[15] = 0.3  # right_shoulder2
    qpos5[16] = 0.2  # right_elbow
    compare_fk(ctx, "Full body pose", qpos5, model_buf)
    print()

    print("All Humanoid FK CPU vs GPU tests passed.")


fn main() raises:
    test_fk_humanoid()
