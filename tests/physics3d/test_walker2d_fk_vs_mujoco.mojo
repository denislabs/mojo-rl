"""Test Forward Kinematics against MuJoCo reference for Walker2D.

Compares our FK output (xpos, xquat, xipos) with MuJoCo's for the
Walker2D model at multiple qpos configurations.

Walker2D is a biped with two symmetric legs (complements Hopper's single leg):
  - 9 DOFs: rootx (slide), rootz (slide, ref=1.25), rooty (hinge),
    thigh/leg/foot joints × 2 legs
  - 8 bodies: torso + 3 bodies per leg × 2 legs
  - RK4 integrator, dt=0.002, armature=0.01

Key features tested beyond HalfCheetah/Hopper:
  - Two-leg symmetric body tree (left/right)
  - leg_joint and foot_joint have off-center jnt_pos (tests cdof anchor)
  - rootz has ref="1.25" → qpos0[rootz]=1.25 (default standing height)

Note: At qpos=all_zeros, torso is at z=0 (not 1.25). The natural standing
pose uses qpos[rootz]=1.25 (= qpos0[rootz]).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_walker2d_fk_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = Walker2dModel.NQ  # 9
comptime NV = Walker2dModel.NV  # 9
comptime NBODY = Walker2dModel.NBODY  # 8
comptime NJOINT = Walker2dModel.NJOINT  # 9
comptime NGEOM = Walker2dModel.NGEOM  # 8
comptime MAX_CONTACTS = Walker2dModel.MAX_CONTACTS  # 20

# Tolerance for comparison (float64)
comptime POS_TOL: Float64 = 1e-6
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

    # === Our engine ===
    var model = Model[
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
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, Walker2dModel.NSITE
    ]()
    Walker2dModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])

    forward_kinematics(model, data)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/walker2d.xml"
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
    body_names.append("thigh")
    body_names.append("leg")
    body_names.append("foot")
    body_names.append("thigh_left")
    body_names.append("leg_left")
    body_names.append("foot_left")

    var all_pass = True

    for bi in range(NBODY - 1):
        var b = bi + 1  # skip worldbody
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


def test_fk_default_qpos() raises:
    """FK at default standing pose (qpos = qpos0).
    rootz=1.25 places torso at z=1.25m — the natural standing height.
    All joint angles are zero (straight legs)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.25  # rootz = qpos0 → torso at z=1.25
    compare_fk("Default standing pose (rootz=1.25)", qpos)


def test_fk_large_rootx() raises:
    """FK with large horizontal displacement — torso moved 5m forward.
    Validates that translation doesn't accumulate floating-point errors."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 5.0  # rootx: 5m forward
    qpos[1] = 1.25  # rootz: standing height
    compare_fk("Large rootx (5m)", qpos)


def test_fk_bent_right_leg() raises:
    """FK with right leg bent: thigh forward, leg bent back, foot angled.
    Tests off-center jnt_pos for leg_joint (pos='0 0 0.25') and
    foot_joint (pos='-0.2 0 0.1') — the same structure that exposed
    the cdof anchor bug in Hopper."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.25  # rootz
    qpos[3] = -0.5  # thigh_joint (backward bend, axis=-y)
    qpos[4] = 0.5  # leg_joint (forward flex)
    qpos[5] = -0.2  # foot_joint
    compare_fk("Right leg bent (thigh=-0.5, leg=0.5, foot=-0.2)", qpos)


def test_fk_symmetric_gait() raises:
    """FK with both legs bent symmetrically — typical walking pose.
    Tests the full body tree with all joints active."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0  # rootx: 1m forward
    qpos[1] = 1.25  # rootz: standing height
    qpos[2] = 0.1  # rooty: slight forward lean
    qpos[3] = -0.5  # thigh_joint (right, backward)
    qpos[4] = 0.5  # leg_joint (right, forward)
    qpos[5] = -0.2  # foot_joint (right)
    qpos[6] = -0.3  # thigh_left_joint (stepping forward)
    qpos[7] = 0.3  # leg_left_joint
    qpos[8] = -0.1  # foot_left_joint
    compare_fk("Symmetric gait pose (both legs bent)", qpos)


def test_fk_extreme_joints() raises:
    """FK near joint limits: thighs at max backward bend, feet angled.
    Tests large rotations in the multi-level hinge chain."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.25  # rootz
    qpos[2] = -0.3  # rooty: lean back
    qpos[
        3
    ] = -2.0  # thigh_joint: near -150 deg limit (in radians: -2.0 ≈ -114 deg)
    qpos[4] = 1.5  # leg_joint: bent far forward
    qpos[5] = 0.7  # foot_joint: near +45 deg limit
    qpos[6] = -2.0  # thigh_left_joint
    qpos[7] = 1.5  # leg_left_joint
    qpos[8] = 0.7  # foot_left_joint
    compare_fk("Extreme joints (near limits)", qpos)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
