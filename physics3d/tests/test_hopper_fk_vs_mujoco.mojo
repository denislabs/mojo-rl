"""Test Forward Kinematics against MuJoCo reference for Hopper.

Compares our FK output (xpos, xquat, xipos) with MuJoCo's for the Hopper
model at multiple qpos configurations. Uses Python interop to call MuJoCo.

Hopper uses ELLIPTIC cone (default), providing coverage for non-pyramidal cone.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_hopper_fk_vs_mujoco.mojo
"""

from testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from physics3d.types import Model, Data
from physics3d.kinematics.forward_kinematics import forward_kinematics
from envs.hopper.hopper_xml import HopperModel
from envs.hopper.hopper_config import HopperConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HopperModel.NQ  # 6
comptime NV = HopperModel.NV  # 6
comptime NBODY = HopperModel.NBODY  # 5
comptime NJOINT = HopperModel.NJOINT  # 6
comptime NGEOM = HopperModel.NGEOM  # 5
comptime MAX_CONTACTS = HopperConfig.MAX_CONTACTS  # 20

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
        HopperModel.MAX_EQUALITY,
        HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON,
        HopperModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    HopperModel.setup_model_and_data(model, data)

    # Set qpos
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])

    # Run our FK
    forward_kinematics(model, data)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    # Set qpos in MuJoCo
    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]

    # Run MuJoCo forward (computes FK + dynamics)
    mujoco.mj_forward(mj_model, mj_data)

    # Flatten MuJoCo arrays
    var mj_xpos_flat = mj_data.xpos.flatten().tolist()
    var mj_xquat_flat = mj_data.xquat.flatten().tolist()
    var mj_xipos_flat = mj_data.xipos.flatten().tolist()

    # === Compare body by body ===
    var all_pass = True
    var body_names = List[String]()
    body_names.append("torso")
    body_names.append("thigh")
    body_names.append("leg")
    body_names.append("foot")

    for bi in range(len(body_names)):
        var b = bi + 1  # Skip worldbody (body 0)

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
            print("  FAIL xpos ", body_names[bi], " err=", pos_err)
            print("    ours:  ", our_px, our_py, our_pz)
            print("    mujoco:", mj_px, mj_py, mj_pz)
            all_pass = False
        else:
            print("  OK   xpos ", body_names[bi], " err=", pos_err)

        # --- xquat ---
        # MuJoCo uses (w,x,y,z), our engine uses (x,y,z,w)
        var our_qx = Float64(data.xquat[b * 4 + 0])
        var our_qy = Float64(data.xquat[b * 4 + 1])
        var our_qz = Float64(data.xquat[b * 4 + 2])
        var our_qw = Float64(data.xquat[b * 4 + 3])

        var mj_qw = Float64(py=mj_xquat_flat[b * 4 + 0])  # MuJoCo: (w,x,y,z)
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
            print("  FAIL xquat", body_names[bi], " err=", quat_err)
            print("    ours (x,y,z,w):  ", our_qx, our_qy, our_qz, our_qw)
            print("    mujoco (w,x,y,z):", mj_qw, mj_qx, mj_qy, mj_qz)
            all_pass = False
        else:
            print("  OK   xquat", body_names[bi], " err=", quat_err)

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
            print("  FAIL xipos", body_names[bi], " err=", xipos_err)
            print("    ours:  ", our_xi_x, our_xi_y, our_xi_z)
            print("    mujoco:", mj_xi_x, mj_xi_y, mj_xi_z)
            all_pass = False
        else:
            print("  OK   xipos", body_names[bi], " err=", xipos_err)

    assert_true(all_pass, "compare_fk failed for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


fn test_fk_default_qpos() raises:
    """Test FK at default qpos (all zeros — torso at body_pos height 1.25)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    compare_fk("Default qpos (torso at 1.25)", qpos)


fn test_fk_nonzero_rootz() raises:
    """Test FK with nonzero rootz (jumping)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.5  # rootz offset => torso at 1.75
    compare_fk("Nonzero rootz (jumping)", qpos)


fn test_fk_nonzero_joints() raises:
    """Test FK with non-zero joint angles."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0  # rootx = 1m forward
    qpos[1] = 0.0  # rootz at default
    qpos[2] = 0.3  # rooty pitch
    qpos[3] = -0.4  # thigh_joint
    qpos[4] = 0.5  # leg_joint
    qpos[5] = -0.2  # foot_joint
    compare_fk("Non-zero joints", qpos)


fn test_fk_extreme_joints() raises:
    """Test FK at or near joint limits."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = -0.3  # rooty negative pitch
    qpos[3] = -2.0  # thigh_joint (large backward bend)
    qpos[4] = -0.005  # leg_joint near lower limit
    qpos[5] = -0.7  # foot_joint
    compare_fk("Extreme joint angles", qpos)


fn test_fk_large_rootx() raises:
    """Test FK with large horizontal displacement."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 100.0  # rootx far forward
    qpos[3] = 0.5  # thigh_joint
    qpos[5] = -0.3  # foot_joint
    compare_fk("Large rootx (100m)", qpos)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
