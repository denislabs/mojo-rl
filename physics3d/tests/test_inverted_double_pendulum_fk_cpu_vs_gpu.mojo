"""Test Forward Kinematics: CPU vs GPU for InvertedDoublePendulum.

Compares our CPU FK output (xpos, xquat, xipos) with GPU FK output for the
InvertedDoublePendulum model at multiple qpos configurations. Both should produce
identical results (up to float32 precision).

InvertedDoublePendulum: NQ=3, NV=3, NBODY=4 (worldbody + cart + pole + pole2)
  slider (slide x), hinge (hinge y), hinge2 (hinge y). NSITE=1 ("tip" at top of pole2).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_inverted_double_pendulum_fk_cpu_vs_gpu.mojo
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
from envs.inverted_double_pendulum.inverted_double_pendulum_xml import (
    InvertedDoublePendulumModel,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = InvertedDoublePendulumModel.NQ  # 3
comptime NV = InvertedDoublePendulumModel.NV  # 3
comptime NBODY = InvertedDoublePendulumModel.NBODY  # 4 (worldbody + cart + pole + pole2)
comptime NJOINT = InvertedDoublePendulumModel.NJOINT  # 3
comptime NGEOM = InvertedDoublePendulumModel.NGEOM  # 5
comptime MAX_CONTACTS = InvertedDoublePendulumModel.MAX_CONTACTS  # 5
comptime NSITE = InvertedDoublePendulumModel.NSITE  # 1 ("tip")
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
        InvertedDoublePendulumModel.MAX_EQUALITY,
        InvertedDoublePendulumModel.CONE_TYPE,
        InvertedDoublePendulumModel.MAX_TENDON,
        InvertedDoublePendulumModel.NSITE,
    ]()
    var data_cpu = Data[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        InvertedDoublePendulumModel.NSITE,
    ]()
    InvertedDoublePendulumModel.setup_model_and_data[DTYPE](model_cpu, data_cpu)
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
    var body_names = List[String]()
    body_names.append("worldbody")
    body_names.append("cart")
    body_names.append("pole")
    body_names.append("pole2")

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
# Test cases (same configs as test_inverted_double_pendulum_fk_vs_mujoco.mojo)
# =============================================================================


fn test_fk_idp() raises:
    print("=" * 60)
    print("FK Validation: CPU vs GPU — InvertedDoublePendulum")
    print("=" * 60)
    print("Model: InvertedDoublePendulum (NBODY=4, NQ=3, NSITE=1)")
    print("Precision: float32")
    print("Tolerances: pos=", POS_TOL, " quat=", QUAT_TOL)
    print()

    var ctx = DeviceContext()
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    InvertedDoublePendulumModel.init_model_gpu(ctx, model_buf)
    ctx.synchronize()

    # Config 1: default qpos (all zeros — cart at origin, poles upright)
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    compare_fk(ctx, "Default qpos (all zeros)", qpos1, model_buf)
    print()

    # Config 2: displaced cart
    var qpos2 = InlineArray[Float64, NQ](fill=0.0)
    qpos2[0] = 2.0  # slider (cart x)
    compare_fk(ctx, "Displaced cart (x=2.0)", qpos2, model_buf)
    print()

    # Config 3: first hinge deflected
    var qpos3 = InlineArray[Float64, NQ](fill=0.0)
    qpos3[1] = 0.5  # hinge (first pole angle)
    compare_fk(ctx, "First hinge deflected (q1=0.5)", qpos3, model_buf)
    print()

    # Config 4: both hinges deflected
    var qpos4 = InlineArray[Float64, NQ](fill=0.0)
    qpos4[1] = 0.5  # hinge
    qpos4[2] = -0.3  # hinge2
    compare_fk(ctx, "Both hinges deflected (q1=0.5, q2=-0.3)", qpos4, model_buf)
    print()

    # Config 5: large tilt
    var qpos5 = InlineArray[Float64, NQ](fill=0.0)
    qpos5[0] = -1.5  # cart displaced
    qpos5[1] = 1.0  # hinge (large tilt)
    qpos5[2] = 0.5  # hinge2
    compare_fk(ctx, "Large tilt (cart=-1.5, q1=1.0, q2=0.5)", qpos5, model_buf)
    print()

    print("All InvertedDoublePendulum FK CPU vs GPU tests passed.")


fn main() raises:
    test_fk_idp()
