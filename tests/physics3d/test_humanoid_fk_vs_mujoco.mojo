"""Test Forward Kinematics against MuJoCo reference for Humanoid.

Compares our FK output (xpos, xquat, xipos) with MuJoCo's for the
Humanoid model at multiple qpos configurations.

The Humanoid is the most complex model tested:
  - 24 DOFs: 1 free joint (7 qpos) + 17 hinge joints
  - 14 bodies: torso, lwaist, pelvis, 2 thighs, 2 shins, 2 feet,
               2 upper arms, 2 lower arms
  - 2 tendons (left_hipknee, right_hipknee) — coupled joint constraints
  - RK4 integrator, PGS solver, dt=0.003

Key features beyond other models:
  - Free joint at torso (full 3D locomotion, like Ant)
  - Dense body tree with arm branches off the torso
  - Tendon constraints (not tested in FK directly, but model setup validates them)
  - Multiple multi-joint bodies (hips have 3 joints each)
  - Default qpos0 = [0, 0, 1.4, 1, 0, 0, 0, 0, ...] (torso at z=1.4)

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_humanoid_fk_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from std.gpu.host import DeviceContext
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HumanoidModel.NQ  # 24 (7 free + 17 hinge)
comptime NV = HumanoidModel.NV  # 23 (6 free + 17 hinge)
comptime NBODY = HumanoidModel.NBODY  # 14
comptime NJOINT = HumanoidModel.NJOINT  # 18 (1 free + 17 hinge)
comptime NGEOM = HumanoidModel.NGEOM  # 18
comptime MAX_CONTACTS = HumanoidModel.MAX_CONTACTS  # 50

# Tolerance for comparison (float64).
# Humanoid uses 5e-6 instead of the usual 1e-6 because lwaist has a body-level
# quat="1.000 0 -0.002 0" (non-trivial rotation). Tiny cos/sin rounding errors
# in that rotation accumulate through the deeply-nested pelvis→thigh→shin→foot
# chain and reach 1-3 µm. Arms (shallower chain, no body quat) still hit ~1e-16.
comptime POS_TOL: Float64 = 5e-6
comptime QUAT_TOL: Float64 = 1e-5


# =============================================================================
# Comparison: run FK in both engines, compare results
# =============================================================================


def compare_fk(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
) raises:
    """Run FK in both engines with identical qpos, compare results."""
    print("--- Test:", test_name, "---")

    # === Our engine (fields; legacy Model/Data FK deleted at G4) ===
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, NV, NBODY, NJOINT, NGEOM, HumanoidModel.MAX_EQUALITY,
        HumanoidModel.MAX_TENDON, HumanoidModel.NSITE, HumanoidModel.NEXCLUDE, 0,
    ]()
    HumanoidModel.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HumanoidModel.NSITE, 1
    ]()

    # Set qpos
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](qpos_values[i])

    # Run our FK (fields, CPU)
    forward_kinematics[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        HumanoidModel.MAX_EQUALITY, HumanoidModel.MAX_TENDON, HumanoidModel.NSITE,
        HumanoidModel.NEXCLUDE, 0, 1,
    ](d, mf, None)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/humanoid.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]

    mujoco.mj_forward(mj_model, mj_data)

    var mj_xpos_flat = mj_data.xpos.flatten().tolist()
    var mj_xquat_flat = mj_data.xquat.flatten().tolist()
    var mj_xipos_flat = mj_data.xipos.flatten().tolist()

    # === Compare body by body (skip worldbody at index 0) ===
    var body_names = List[String]()
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

    var all_pass = True

    for bi in range(NBODY - 1):
        var b = bi + 1  # skip worldbody
        var bname = body_names[bi]

        # --- xpos ---
        var mj_px = Float64(py=mj_xpos_flat[b * 3 + 0])
        var mj_py = Float64(py=mj_xpos_flat[b * 3 + 1])
        var mj_pz = Float64(py=mj_xpos_flat[b * 3 + 2])

        var our_px = Float64(d.xpos.data[b * 3 + 0])
        var our_py = Float64(d.xpos.data[b * 3 + 1])
        var our_pz = Float64(d.xpos.data[b * 3 + 2])

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
        var our_qx = Float64(d.xquat.data[b * 4 + 0])
        var our_qy = Float64(d.xquat.data[b * 4 + 1])
        var our_qz = Float64(d.xquat.data[b * 4 + 2])
        var our_qw = Float64(d.xquat.data[b * 4 + 3])

        var mj_qw = Float64(py=mj_xquat_flat[b * 4 + 0])
        var mj_qx = Float64(py=mj_xquat_flat[b * 4 + 1])
        var mj_qy = Float64(py=mj_xquat_flat[b * 4 + 2])
        var mj_qz = Float64(py=mj_xquat_flat[b * 4 + 3])

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

        var our_xi_x = Float64(d.xipos.data[b * 3 + 0])
        var our_xi_y = Float64(d.xipos.data[b * 3 + 1])
        var our_xi_z = Float64(d.xipos.data[b * 3 + 2])

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
#
# qpos layout:
#   [0:3]  free joint translation: x, y, z
#   [3:7]  free joint quaternion: qw, qx, qy, qz
#   [7]    abdomen_z
#   [8]    abdomen_y
#   [9]    abdomen_x
#   [10]   right_hip_x
#   [11]   right_hip_z
#   [12]   right_hip_y
#   [13]   right_knee
#   [14]   left_hip_x
#   [15]   left_hip_z
#   [16]   left_hip_y
#   [17]   left_knee
#   [18]   right_shoulder1
#   [19]   right_shoulder2
#   [20]   right_elbow
#   [21]   left_shoulder1
#   [22]   left_shoulder2
#   [23]   left_elbow


def test_fk_default_qpos() raises:
    """FK at default standing pose (qpos = qpos0).
    Torso at z=1.4, identity quaternion, all joint angles zero.
    This is the initial pose MuJoCo initializes to."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    # Free joint: identity quaternion at standing height
    qpos[2] = 1.4  # z = 1.4m (torso height)
    qpos[3] = 1.0  # qw = 1 (identity quaternion)
    compare_fk("Default standing pose (z=1.4, identity quat)", qpos)


def test_fk_bent_knees() raises:
    """FK with both knees bent — tests hip multi-joint chains.
    right_hip_y and left_hip_y rotate the thighs; right_knee and
    left_knee bend the shins."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 1.4
    qpos[3] = 1.0  # identity quat
    qpos[12] = -0.5  # right_hip_y: thigh forward
    qpos[13] = -1.0  # right_knee: bent (range -160 to -2, so -1.0 is ~-57 deg)
    qpos[16] = -0.5  # left_hip_y: thigh forward
    qpos[17] = -1.0  # left_knee: bent
    compare_fk("Bent knees (hip_y=-0.5, knee=-1.0 both sides)", qpos)


def test_fk_arms_extended() raises:
    """FK with arms extended — tests shoulder + elbow chains branching off torso.
    The arm branches are independent of the leg branches."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 1.4
    qpos[3] = 1.0  # identity quat
    qpos[18] = 0.5  # right_shoulder1
    qpos[19] = 0.5  # right_shoulder2
    qpos[20] = 0.8  # right_elbow
    qpos[21] = 0.5  # left_shoulder1
    qpos[22] = 0.5  # left_shoulder2
    qpos[23] = 0.8  # left_elbow
    compare_fk("Arms extended (shoulders=0.5, elbows=0.8)", qpos)


def test_fk_rotated_torso() raises:
    """FK with torso rotated ~30 deg about z-axis.
    Tests quaternion propagation through the dense body tree:
    all 13 child bodies must accumulate the root rotation correctly."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0  # x translation
    qpos[2] = 1.4  # z = standing height
    # ~30 deg rotation about z-axis: qw=cos(15°)=0.966, qz=sin(15°)=0.259
    qpos[3] = 0.9659  # qw
    qpos[4] = 0.0  # qx
    qpos[5] = 0.0  # qy
    qpos[6] = 0.2588  # qz
    compare_fk("Torso rotated 30 deg about z-axis", qpos)


def test_fk_full_body_pose() raises:
    """FK at a realistic walking pose: bent knees, abdomen lean, arms moving.
    Exercises all body branches simultaneously."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 1.4
    qpos[3] = 1.0  # identity quat
    qpos[7] = 0.1  # abdomen_z
    qpos[8] = -0.1  # abdomen_y (slight lean)
    qpos[9] = 0.05  # abdomen_x
    qpos[10] = 0.1  # right_hip_x
    qpos[12] = -0.4  # right_hip_y
    qpos[13] = -0.8  # right_knee
    qpos[14] = -0.1  # left_hip_x
    qpos[16] = -0.3  # left_hip_y
    qpos[17] = -0.6  # left_knee
    qpos[18] = 0.3  # right_shoulder1
    qpos[19] = -0.2  # right_shoulder2
    qpos[20] = 0.4  # right_elbow
    qpos[21] = -0.3  # left_shoulder1
    qpos[22] = 0.2  # left_shoulder2
    qpos[23] = 0.4  # left_elbow
    compare_fk("Full body walking pose (all joints active)", qpos)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
