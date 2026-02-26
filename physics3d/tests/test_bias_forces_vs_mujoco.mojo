"""Test Bias Forces (RNE) against MuJoCo reference.

Compares our bias forces (Coriolis + gravity via Recursive Newton-Euler) with
MuJoCo's qfrc_bias for the HalfCheetah model at multiple configurations.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_bias_forces_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.jacobian import compute_cdof
from physics3d.dynamics.bias_forces import compute_bias_forces_rne
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


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

comptime V_SIZE = _max_one[NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()

# Tolerance
comptime ABS_TOL: Float64 = 1e-4
comptime REL_TOL: Float64 = 1e-3


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_bias_forces(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    qvel_values: InlineArray[Float64, NV],
) raises -> Bool:
    """Compute bias forces in both engines with identical state, compare."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model, data)
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_values[i])

    # Run FK + body velocities (needed for RNE)
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    # Compute cdof
    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model, data, cdof
    )

    # Compute bias forces via RNE
    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        bias[i] = Scalar[DTYPE](0)
    compute_bias_forces_rne[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
    ](model, data, cdof, bias)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_values[i]

    mujoco.mj_forward(mj_model, mj_data)

    var mj_bias_flat = mj_data.qfrc_bias.flatten().tolist()

    # === Compare element by element ===
    var all_pass = True
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for i in range(NV):
        var our_val = Float64(bias[i])
        var mj_val = Float64(py=mj_bias_flat[i])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        var ok = abs_err < ABS_TOL or rel_err < REL_TOL
        if not ok:
            print(
                "  FAIL bias[", i, "]",
                " ours=", our_val,
                " mj=", mj_val,
                " abs_err=", abs_err,
                " rel_err=", rel_err,
            )
            fail_count += 1
            all_pass = False

    if all_pass:
        print("  ALL OK  max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err)
    else:
        print(
            "  FAILED", fail_count, "elements  max_abs_err=", max_abs_err,
            " max_rel_err=", max_rel_err,
        )

    # Print values for inspection
    print("  Our bias: ", end="")
    for i in range(NV):
        print(" ", Float64(bias[i]), end="")
    print()
    print("  MuJoCo:   ", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_bias_flat[i]), end="")
    print()

    return all_pass


# =============================================================================
# Test cases
# =============================================================================


fn test_default_qpos_zero_vel() raises -> Bool:
    """Bias at default qpos, zero velocity (gravity only)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7  # rootz
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_bias_forces("Default qpos, zero vel (gravity only)", qpos, qvel)


fn test_zero_qpos_zero_vel() raises -> Bool:
    """Bias at qpos=0, zero velocity."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_bias_forces("Zero qpos, zero vel", qpos, qvel)


fn test_nonzero_joints_zero_vel() raises -> Bool:
    """Bias with non-zero joints, zero velocity (gravity only)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0   # rootx
    qpos[1] = 0.7   # rootz
    qpos[2] = 0.3   # rooty
    qpos[3] = -0.4  # bthigh
    qpos[4] = 0.5   # bshin
    qpos[5] = -0.2  # bfoot
    qpos[6] = 0.6   # fthigh
    qpos[7] = -0.8  # fshin
    qpos[8] = 0.3   # ffoot
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_bias_forces("Non-zero joints, zero vel", qpos, qvel)


fn test_nonzero_vel() raises -> Bool:
    """Bias with non-zero velocity (includes Coriolis)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7   # rootz
    qpos[2] = 0.1   # rooty
    qpos[3] = -0.3  # bthigh
    qpos[6] = 0.4   # fthigh
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0   # rootx vel (running)
    qvel[2] = 0.5   # rooty vel (pitching)
    qvel[3] = -1.0  # bthigh vel
    qvel[4] = 0.8   # bshin vel
    qvel[6] = 1.2   # fthigh vel
    qvel[7] = -0.6  # fshin vel
    return compare_bias_forces("Non-zero vel (gravity + Coriolis)", qpos, qvel)


fn test_extreme_vel() raises -> Bool:
    """Bias with large velocities (stress test Coriolis)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[3] = -0.52  # bthigh min
    qpos[6] = -1.0   # fthigh min
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 5.0    # fast running
    qvel[1] = -2.0   # falling
    qvel[2] = 3.0    # fast pitch
    qvel[3] = -5.0   # bthigh fast
    qvel[4] = 5.0    # bshin fast
    qvel[5] = -3.0   # bfoot fast
    qvel[6] = 5.0    # fthigh fast
    qvel[7] = -5.0   # fshin fast
    qvel[8] = 3.0    # ffoot fast
    return compare_bias_forces("Extreme velocities", qpos, qvel)


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("Bias Forces (RNE) Validation: Mojo Engine vs MuJoCo Reference")
    print("=" * 60)
    print("Model: HalfCheetah (NV=9)")
    print("Tolerances: abs=", ABS_TOL, " rel=", REL_TOL)
    print()

    var num_pass = 0
    var num_fail = 0

    if test_default_qpos_zero_vel():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_zero_qpos_zero_vel():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_nonzero_joints_zero_vel():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_nonzero_vel():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_extreme_vel():
        num_pass += 1
    else:
        num_fail += 1
    print()

    print("=" * 60)
    print(
        "Results:",
        num_pass,
        "passed,",
        num_fail,
        "failed out of",
        num_pass + num_fail,
    )
    if num_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
