"""Test Forward Kinematics against MuJoCo reference for Swimmer.

Compares our FK output (xpos, xquat, xipos) with MuJoCo's for the Swimmer
model at multiple qpos configurations. Uses Python interop to call MuJoCo.

The Swimmer is a 3-body chain (torso → mid → back) with:
  - 2 slide joints (x, y translation, no quaternion DOF)
  - 1 free rotation hinge (torso yaw)
  - 2 motor hinges (motor1_rot, motor2_rot)
NQ=5, NV=5, NBODY=4 (worldbody + 3 bodies).

All geoms have contype=0 so no contacts are expected.
MuJoCo fluid dynamics (viscosity=0.1, density=4000) affects forces but not FK.
FK depends only on joint positions, not velocities or forces.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_swimmer_fk_vs_mujoco.mojo
"""

from testing import assert_true, TestSuite
from python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from physics3d.types import Model, Data
from physics3d.kinematics.forward_kinematics import forward_kinematics
from envs.swimmer.swimmer_xml import SwimmerModel
from envs.swimmer.swimmer_config import SwimmerConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = SwimmerModel.NQ  # 5 (slider1, slider2, free_body_rot, motor1_rot, motor2_rot)
comptime NV = SwimmerModel.NV  # 5
comptime NBODY = SwimmerModel.NBODY  # 4 (worldbody, torso, mid, back)
comptime NJOINT = SwimmerModel.NJOINT  # 5
comptime NGEOM = SwimmerModel.NGEOM  # 3 capsules
comptime MAX_CONTACTS = SwimmerModel.MAX_CONTACTS  # 5

# Tolerance for comparison (float64)
comptime POS_TOL: Float64 = 1e-6
comptime QUAT_TOL: Float64 = 1e-5


# =============================================================================
# Comparison: run FK in both engines, compare results
# =============================================================================


fn compare_fk(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
) raises:
    """Run FK in both engines with identical qpos, compare results."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        SwimmerModel.MAX_EQUALITY,
        SwimmerModel.CONE_TYPE,
        SwimmerModel.MAX_TENDON,
        SwimmerModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SwimmerModel.NSITE
    ]()
    SwimmerModel.setup_model_and_data(model, data)

    # Set qpos
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])

    # Run our FK
    forward_kinematics(model, data)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/swimmer.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    # Set qpos in MuJoCo
    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]

    # Run MuJoCo forward kinematics
    mujoco.mj_forward(mj_model, mj_data)

    # Flatten MuJoCo arrays
    var mj_xpos_flat = mj_data.xpos.flatten().tolist()
    var mj_xquat_flat = mj_data.xquat.flatten().tolist()
    var mj_xipos_flat = mj_data.xipos.flatten().tolist()

    # === Compare body by body (skip worldbody at index 0) ===
    var body_names = List[String]()
    body_names.append("torso")
    body_names.append("mid")
    body_names.append("back")

    var all_pass = True

    for bi in range(NBODY - 1):
        var b = bi + 1  # skip worldbody (body 0)
        var bname = body_names[bi]

        # --- xpos ---
        var mj_px = Float64(py=mj_xpos_flat[b * 3 + 0])
        var mj_py = Float64(py=mj_xpos_flat[b * 3 + 1])
        var mj_pz = Float64(py=mj_xpos_flat[b * 3 + 2])

        var our_px = Float64(data.xpos[b * 3 + 0])
        var our_py = Float64(data.xpos[b * 3 + 1])
        var our_pz = Float64(data.xpos[b * 3 + 2])

        var pos_err = (
            abs(our_px - mj_px) + abs(our_py - mj_py) + abs(our_pz - mj_pz)
        )

        if pos_err > POS_TOL:
            print("  FAIL xpos ", bname, " err=", pos_err)
            print("    ours:  ", our_px, our_py, our_pz)
            print("    mujoco:", mj_px, mj_py, mj_pz)
            all_pass = False
        else:
            print("  OK   xpos ", bname, " err=", pos_err)

        # --- xquat (our: x,y,z,w — MuJoCo: w,x,y,z) ---
        var our_qx = Float64(data.xquat[b * 4 + 0])
        var our_qy = Float64(data.xquat[b * 4 + 1])
        var our_qz = Float64(data.xquat[b * 4 + 2])
        var our_qw = Float64(data.xquat[b * 4 + 3])

        var mj_qw = Float64(py=mj_xquat_flat[b * 4 + 0])
        var mj_qx = Float64(py=mj_xquat_flat[b * 4 + 1])
        var mj_qy = Float64(py=mj_xquat_flat[b * 4 + 2])
        var mj_qz = Float64(py=mj_xquat_flat[b * 4 + 3])

        # Quaternions q and -q represent the same rotation
        var diff_pos = (
            abs(our_qx - mj_qx)
            + abs(our_qy - mj_qy)
            + abs(our_qz - mj_qz)
            + abs(our_qw - mj_qw)
        )
        var diff_neg = (
            abs(our_qx + mj_qx)
            + abs(our_qy + mj_qy)
            + abs(our_qz + mj_qz)
            + abs(our_qw + mj_qw)
        )
        var quat_err = diff_pos if diff_pos < diff_neg else diff_neg

        if quat_err > QUAT_TOL:
            print("  FAIL xquat", bname, " err=", quat_err)
            print("    ours (x,y,z,w):  ", our_qx, our_qy, our_qz, our_qw)
            print("    mujoco (w,x,y,z):", mj_qw, mj_qx, mj_qy, mj_qz)
            all_pass = False
        else:
            print("  OK   xquat", bname, " err=", quat_err)

        # --- xipos ---
        var mj_xi_x = Float64(py=mj_xipos_flat[b * 3 + 0])
        var mj_xi_y = Float64(py=mj_xipos_flat[b * 3 + 1])
        var mj_xi_z = Float64(py=mj_xipos_flat[b * 3 + 2])

        var our_xi_x = Float64(data.xipos[b * 3 + 0])
        var our_xi_y = Float64(data.xipos[b * 3 + 1])
        var our_xi_z = Float64(data.xipos[b * 3 + 2])

        var xipos_err = (
            abs(our_xi_x - mj_xi_x)
            + abs(our_xi_y - mj_xi_y)
            + abs(our_xi_z - mj_xi_z)
        )

        if xipos_err > POS_TOL:
            print("  FAIL xipos", bname, " err=", xipos_err)
            print("    ours:  ", our_xi_x, our_xi_y, our_xi_z)
            print("    mujoco:", mj_xi_x, mj_xi_y, mj_xi_z)
            all_pass = False
        else:
            print("  OK   xipos", bname, " err=", xipos_err)

    assert_true(all_pass, "compare_fk failed for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


fn test_fk_all_zeros() raises:
    """FK at all-zero qpos: torso at origin, all joints at 0.
    Swimmer torso capsule extends along x-axis with body at z=0."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    compare_fk("All-zero qpos", qpos)


fn test_fk_nonzero_position() raises:
    """FK with the swimmer displaced in the x-y plane.
    Slide joints move the body; body orientations should be unchanged."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 3.0  # slider1 (x translation)
    qpos[1] = -2.0  # slider2 (y translation)
    compare_fk("Nonzero x-y position (slider1=3, slider2=-2)", qpos)


fn test_fk_bent_joints() raises:
    """FK with motor joints bent — exercises the 3-body chain FK.
    motor1_rot bends mid relative to torso, motor2_rot bends back relative to mid.
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[3] = 0.5  # motor1_rot (mid bent ~28.6 deg relative to torso)
    qpos[4] = -0.5  # motor2_rot (back bent ~28.6 deg relative to mid)
    compare_fk("Bent joints (motor1=0.5, motor2=-0.5 rad)", qpos)


fn test_fk_rotated_and_bent() raises:
    """FK with torso rotated + both motor joints bent.
    Tests composition of rotation through the full body chain."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0  # x position
    qpos[1] = 0.5  # y position
    qpos[2] = 0.785  # free_body_rot = 45 deg = pi/4
    qpos[3] = 0.3  # motor1_rot
    qpos[4] = 0.3  # motor2_rot (S-curve shape)
    compare_fk("Rotated torso + bent joints (45 deg + 0.3/0.3 rad)", qpos)


fn test_fk_large_position() raises:
    """FK with large x displacement — tests FK locality (position offset
    should not affect body orientations)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 100.0  # slider1 far in x
    qpos[3] = 0.8  # motor1_rot
    qpos[4] = -0.8  # motor2_rot (C-curve shape)
    compare_fk("Large x position (100m) + C-curve joints", qpos)


fn test_fk_near_joint_limits() raises:
    """FK near the joint limits of motor1_rot and motor2_rot (±100 degrees).
    Range is ±100 deg = ±1.745 rad."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[3] = 1.5  # motor1_rot near limit (~86 deg)
    qpos[4] = -1.5  # motor2_rot near lower limit
    compare_fk("Near joint limits (±1.5 rad)", qpos)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
