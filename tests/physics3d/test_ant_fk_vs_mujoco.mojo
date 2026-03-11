"""Test Forward Kinematics against MuJoCo reference for Ant.

Compares our FK output (xpos, xquat, xipos) with MuJoCo's for the Ant
model at multiple qpos configurations. Uses Python interop to call MuJoCo.

The Ant is the first test with a 3D free joint (7 qpos DOFs: x, y, z, qw, qx, qy, qz)
and a 4-leg symmetric body tree. This exercises quaternion FK in full 3D,
unlike HalfCheetah/Hopper which are planar.

Ant uses ELLIPTIC cone (default), RK4 integrator, NQ=15, NV=14.
Free joint quaternion in qpos is stored as (w, x, y, z) matching MuJoCo.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_ant_fk_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.envs.ant.ant_xml import AntModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = AntModel.NQ  # 15 (7 free-joint + 8 hinge)
comptime NV = AntModel.NV  # 14 (6 free-joint + 8 hinge)
comptime NBODY = AntModel.NBODY  # 14 (worldbody + torso + 4 legs × 3 bodies)
comptime NJOINT = AntModel.NJOINT  # 9 (1 free + 8 hinge)
comptime NGEOM = AntModel.NGEOM
comptime MAX_CONTACTS = AntModel.MAX_CONTACTS  # 40

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
        AntModel.MAX_EQUALITY,
        AntModel.CONE_TYPE,
        AntModel.MAX_TENDON,
        AntModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, AntModel.NSITE
    ]()
    AntModel.setup_model_and_data(model, data)

    # Set qpos
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])

    # Run our FK
    forward_kinematics(model, data)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/ant.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    # Set qpos in MuJoCo (same values — free-joint quat in (w,x,y,z) for both)
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


fn test_fk_default_qpos() raises:
    """FK at Ant's default init_qpos: torso at z=0.75, identity quaternion,
    legs at their default angles from the XML custom/numeric init_qpos."""
    # From XML: <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    # Free joint: x=0, y=0, z=0.55, qw=1, qx=0, qy=0, qz=0 (identity quaternion)
    qpos[0] = 0.0  # x
    qpos[1] = 0.0  # y
    qpos[2] = 0.55  # z (torso height above ground)
    qpos[3] = 1.0  # qw (unit quaternion — upright torso)
    qpos[4] = 0.0  # qx
    qpos[5] = 0.0  # qy
    qpos[6] = 0.0  # qz
    # Hinge joints from init_qpos: [0, 1, 0, -1, 0, -1, 0, 1]
    qpos[7] = 0.0  # hip_1
    qpos[8] = 1.0  # ankle_1
    qpos[9] = 0.0  # hip_2
    qpos[10] = -1.0  # ankle_2
    qpos[11] = 0.0  # hip_3
    qpos[12] = -1.0  # ankle_3
    qpos[13] = 0.0  # hip_4
    qpos[14] = 1.0  # ankle_4
    compare_fk("Default init_qpos (z=0.55, identity quat)", qpos)


fn test_fk_zero_joints() raises:
    """FK with all hinge joints at 0, torso at default height."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 0.55  # z
    qpos[3] = 1.0  # qw (identity quaternion)
    compare_fk("All-zero joints, z=0.55", qpos)


fn test_fk_bent_legs() raises:
    """FK with legs bent symmetrically — exercises multi-level hinge chains."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 0.55  # z
    qpos[3] = 1.0  # qw
    # Hip joints at +15 deg (0.26 rad), ankle joints at +45 deg (0.79 rad)
    qpos[7] = 0.26  # hip_1
    qpos[8] = 0.79  # ankle_1
    qpos[9] = -0.26  # hip_2 (opposite direction)
    qpos[10] = 0.79  # ankle_2
    qpos[11] = 0.26  # hip_3
    qpos[12] = 0.79  # ankle_3
    qpos[13] = -0.26  # hip_4
    qpos[14] = 0.79  # ankle_4
    compare_fk("Bent legs (hips 15 deg, ankles 45 deg)", qpos)


fn test_fk_rotated_torso() raises:
    """FK with torso rotated 45 degrees around the z-axis.
    This exercises the full 3D quaternion propagation through the body tree."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 0.55  # z height
    # 45 deg rotation about z-axis: qw=cos(22.5°)=0.924, qz=sin(22.5°)=0.383
    qpos[3] = 0.9239  # qw
    qpos[4] = 0.0  # qx
    qpos[5] = 0.0  # qy
    qpos[6] = 0.3827  # qz
    compare_fk("Torso rotated 45 deg about z-axis", qpos)


fn test_fk_elevated_and_tilted() raises:
    """FK with elevated torso and small tilt — simulates mid-jump or landing."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 2.0  # x displacement
    qpos[1] = 1.0  # y displacement
    qpos[2] = 1.5  # elevated z
    # Small tilt around x-axis: ~15 deg → qw=cos(7.5°)=0.991, qx=sin(7.5°)=0.131
    qpos[3] = 0.9914  # qw
    qpos[4] = 0.1305  # qx
    qpos[5] = 0.0  # qy
    qpos[6] = 0.0  # qz
    qpos[7] = 0.3  # hip_1
    qpos[8] = 0.5  # ankle_1
    qpos[9] = -0.3  # hip_2
    qpos[10] = 0.5  # ankle_2
    compare_fk("Elevated and tilted torso", qpos)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
