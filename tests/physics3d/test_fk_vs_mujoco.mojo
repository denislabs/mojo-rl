"""Test Forward Kinematics against MuJoCo reference.

Compares our FK output (xpos, xquat, xipos) with MuJoCo's for the HalfCheetah
model at multiple qpos configurations. Uses Python interop to call MuJoCo.

Run with:
    cd mojo-rl && pixi run mojo run -I . test_fk_vs_mujoco.mojo
"""

from std.python import Python, PythonObject
from std.math import abs, pi
from std.collections import InlineArray
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY  # 7
comptime NJOINT = HalfCheetahModel.NJOINT  # 9
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS  # 20

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
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model, data)

    # Set qpos
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])

    # Run our FK
    forward_kinematics(model, data)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    # Set qpos in MuJoCo via numpy
    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]

    # Run MuJoCo forward (computes FK + dynamics)
    mujoco.mj_forward(mj_model, mj_data)

    # Flatten MuJoCo arrays to Python lists for easy extraction
    var mj_xpos_flat = mj_data.xpos.flatten().tolist()
    var mj_xquat_flat = mj_data.xquat.flatten().tolist()
    var mj_xipos_flat = mj_data.xipos.flatten().tolist()

    # === Compare body by body ===
    # Both engines: body 0 = worldbody, body 1 = torso, etc.
    var all_pass = True
    var body_names = List[String]()
    body_names.append("torso")
    body_names.append("bthigh")
    body_names.append("bshin")
    body_names.append("bfoot")
    body_names.append("fthigh")
    body_names.append("fshin")
    body_names.append("ffoot")

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

        # Quaternions q and -q represent the same rotation — check both signs
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

    assert_true(all_pass, "FK mismatch for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


def test_fk_default_qpos() raises:
    """Test FK at MuJoCo default qpos: rootz=0.7, all others zero."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7  # rootz
    compare_fk("Default qpos (rootz=0.7)", qpos)


def test_fk_zero_qpos() raises:
    """Test FK at qpos=0 (robot at origin)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    compare_fk("Zero qpos (robot at origin)", qpos)


def test_fk_nonzero_joints() raises:
    """Test FK with non-zero joint angles."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0  # rootx = 1m forward
    qpos[1] = 0.7  # rootz = 0.7m height
    qpos[2] = 0.3  # rooty = 0.3 rad pitch
    qpos[3] = -0.4  # bthigh
    qpos[4] = 0.5  # bshin
    qpos[5] = -0.2  # bfoot
    qpos[6] = 0.6  # fthigh
    qpos[7] = -0.8  # fshin
    qpos[8] = 0.3  # ffoot
    compare_fk("Non-zero joints", qpos)


def test_fk_extreme_joints() raises:
    """Test FK at joint limits."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7  # rootz
    qpos[3] = -0.52  # bthigh min
    qpos[4] = 0.785  # bshin max
    qpos[5] = -0.4  # bfoot min
    qpos[6] = -1.0  # fthigh min
    qpos[7] = 0.87  # fshin max
    qpos[8] = -0.5  # ffoot min
    compare_fk("Extreme joint angles (at limits)", qpos)


def test_fk_large_rootx() raises:
    """Test FK with large horizontal displacement."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 100.0  # rootx = 100m forward
    qpos[1] = 0.7  # rootz
    qpos[3] = 0.5  # bthigh
    qpos[6] = -0.5  # fthigh
    compare_fk("Large rootx (100m)", qpos)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
