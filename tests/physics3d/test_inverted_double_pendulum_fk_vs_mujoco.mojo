"""Test Forward Kinematics against MuJoCo reference for InvertedDoublePendulum.

Compares our FK output (xpos, xquat, xipos) with MuJoCo's for the
InvertedDoublePendulum model at multiple qpos configurations.

The InvertedDoublePendulum is the simplest model tested:
  - 3 DOFs: slider (slide), hinge (hinge), hinge2 (hinge)
  - 3 bodies: cart, pole, pole2
  - All geoms have contype=0 (no contacts)
  - RK4 integrator, dt=0.01
  - 1 site: "tip" at the top of pole2

This is a quick sanity check for a slide+hinge chain, smaller than
HalfCheetah and without the complexity of a free joint (Ant).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_inverted_double_pendulum_fk_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from physics3d.types import Model, Data
from physics3d.kinematics.forward_kinematics import forward_kinematics
from envs.inverted_double_pendulum.inverted_double_pendulum_xml import (
    InvertedDoublePendulumModel,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = InvertedDoublePendulumModel.NQ  # 3
comptime NV = InvertedDoublePendulumModel.NV  # 3
comptime NBODY = InvertedDoublePendulumModel.NBODY  # 4
comptime NJOINT = InvertedDoublePendulumModel.NJOINT  # 3
comptime NGEOM = InvertedDoublePendulumModel.NGEOM  # 5
comptime MAX_CONTACTS = InvertedDoublePendulumModel.MAX_CONTACTS  # 5

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
        InvertedDoublePendulumModel.MAX_EQUALITY,
        InvertedDoublePendulumModel.CONE_TYPE,
        InvertedDoublePendulumModel.MAX_TENDON,
        InvertedDoublePendulumModel.NSITE,
    ]()
    var data = Data[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        InvertedDoublePendulumModel.NSITE,
    ]()
    InvertedDoublePendulumModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])

    forward_kinematics(model, data)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/inverted_double_pendulum.xml"
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
    body_names.append("cart")
    body_names.append("pole")
    body_names.append("pole2")

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


fn test_fk_default_qpos() raises:
    """FK at default qpos (all zeros): cart at origin, pendulums upright."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    compare_fk("Default qpos (cart at origin, pendulums upright)", qpos)


fn test_fk_displaced_cart() raises:
    """FK with cart displaced to x=0.5 — tests slide joint translation."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.5  # slider displacement
    compare_fk("Displaced cart (x=0.5)", qpos)


fn test_fk_first_hinge_only() raises:
    """FK with only the first hinge bent — pole tilted ~17 deg, pole2 follows.
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.3  # hinge: ~17 degrees tilt
    compare_fk("First hinge only (hinge=0.3 rad)", qpos)


fn test_fk_both_hinges_bent() raises:
    """FK with both hinges bent in opposite directions.
    Tests quaternion accumulation for a two-link chain."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.2  # cart displaced
    qpos[1] = 0.4  # first hinge: pole bent ~23 deg
    qpos[2] = -0.3  # second hinge: pole2 bent back ~17 deg
    compare_fk("Both hinges bent (hinge=0.4, hinge2=-0.3)", qpos)


fn test_fk_large_tilt() raises:
    """FK with large first hinge tilt near the observation limit.
    Tests nonlinear rotation accumulation in the double pendulum chain."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = -0.5  # cart left
    qpos[1] = 1.0  # ~57 deg tilt (large but not at limit)
    qpos[2] = -0.6  # pole2 bent back
    compare_fk("Large tilt (hinge=1.0, hinge2=-0.6)", qpos)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
