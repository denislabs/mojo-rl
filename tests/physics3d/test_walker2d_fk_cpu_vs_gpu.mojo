"""Test Forward Kinematics: CPU vs GPU for Walker2D.

Compares our CPU FK output (xpos, xquat, xipos) with GPU FK output for the
Walker2D model at multiple qpos configurations. Both should produce identical
results (up to float32 precision).

Walker2D: NQ=9, NV=9, NBODY=8 (worldbody + torso + 2 × thigh/leg/foot)
  rootx (slide), rootz (slide, ref=1.25), rooty (hinge), + 6 leg joint hinges.
  Default standing pose: qpos[1]=1.25 (rootz at rest height).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_walker2d_fk_cpu_vs_gpu.mojo
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
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ  # 9
comptime NV = Walker2dModel.NV  # 9
comptime NBODY = Walker2dModel.NBODY  # 8 (worldbody + torso + 2×(thigh+leg+foot))
comptime NJOINT = Walker2dModel.NJOINT  # 9
comptime NGEOM = Walker2dModel.NGEOM  # 8
comptime MAX_CONTACTS = Walker2dModel.MAX_CONTACTS  # 20
comptime NSITE = Walker2dModel.NSITE  # 0
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
        Walker2dModel.MAX_EQUALITY,
        Walker2dModel.CONE_TYPE,
        Walker2dModel.MAX_TENDON,
        Walker2dModel.NSITE,
    ]()
    var data_cpu = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, Walker2dModel.NSITE
    ]()
    Walker2dModel.setup_model_and_data[DTYPE](model_cpu, data_cpu)
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
    # NBODY=8: worldbody(0), torso(1), thigh(2), leg(3), foot(4),
    #          thigh_left(5), leg_left(6), foot_left(7)
    var body_names = List[String]()
    body_names.append("worldbody")
    body_names.append("torso")
    body_names.append("thigh")
    body_names.append("leg")
    body_names.append("foot")
    body_names.append("thigh_left")
    body_names.append("leg_left")
    body_names.append("foot_left")

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
# Test cases (same configs as test_walker2d_fk_vs_mujoco.mojo)
# =============================================================================


fn test_fk_walker2d() raises:
    print("=" * 60)
    print("FK Validation: CPU vs GPU — Walker2D")
    print("=" * 60)
    print("Model: Walker2D (NBODY=8, NQ=9)")
    print("Precision: float32")
    print("Tolerances: pos=", POS_TOL, " quat=", QUAT_TOL)
    print("Note: qpos[1]=rootz, ref=1.25 — standing height is rootz=1.25")
    print()

    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    Walker2dModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    # Config 1: default standing pose (rootz=1.25 = qpos0[rootz])
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    qpos1[1] = 1.25  # rootz at standing height
    compare_fk(ctx, "Default standing (rootz=1.25)", qpos1, model_buf)
    print()

    # Config 2: large rootx displacement
    var qpos2 = InlineArray[Float64, NQ](fill=0.0)
    qpos2[0] = 10.0  # rootx
    qpos2[1] = 1.25  # rootz
    compare_fk(ctx, "Large rootx (x=10, rootz=1.25)", qpos2, model_buf)
    print()

    # Config 3: bent right leg
    var qpos3 = InlineArray[Float64, NQ](fill=0.0)
    qpos3[1] = 1.25  # rootz
    qpos3[3] = 0.5  # thigh_joint (right)
    qpos3[4] = -0.8  # leg_joint (right)
    qpos3[5] = 0.3  # foot_joint (right)
    compare_fk(
        ctx, "Bent right leg (thigh=0.5, leg=-0.8, foot=0.3)", qpos3, model_buf
    )
    print()

    # Config 4: symmetric gait (both legs bent)
    var qpos4 = InlineArray[Float64, NQ](fill=0.0)
    qpos4[1] = 1.25  # rootz
    qpos4[2] = 0.1  # rooty
    qpos4[3] = 0.3  # thigh_joint (right)
    qpos4[4] = -0.5  # leg_joint (right)
    qpos4[6] = -0.3  # thigh_joint_left
    qpos4[7] = -0.5  # leg_joint_left
    compare_fk(ctx, "Symmetric gait", qpos4, model_buf)
    print()

    # Config 5: extreme joint angles
    var qpos5 = InlineArray[Float64, NQ](fill=0.0)
    qpos5[1] = 1.25  # rootz
    qpos5[2] = 0.5  # rooty (lean forward)
    qpos5[3] = 1.0  # thigh_joint
    qpos5[4] = -1.2  # leg_joint
    qpos5[5] = 0.6  # foot_joint
    qpos5[6] = -1.0  # thigh_joint_left
    qpos5[7] = 1.2  # leg_joint_left
    qpos5[8] = -0.6  # foot_joint_left
    compare_fk(ctx, "Extreme joint angles", qpos5, model_buf)
    print()

    print("All Walker2D FK CPU vs GPU tests passed.")


fn main() raises:
    test_fk_walker2d()
