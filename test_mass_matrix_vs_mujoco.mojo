"""Test Mass Matrix against MuJoCo reference.

Compares our full mass matrix (CRBA) output with MuJoCo's mj_fullM for the
HalfCheetah model at multiple qpos configurations. Uses Python interop to
call MuJoCo.

Run with:
    cd mojo-rl && pixi run mojo run test_mass_matrix_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.dynamics.jacobian import compute_cdof, compute_composite_inertia
from physics3d.dynamics.mass_matrix import compute_mass_matrix_full
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahGeoms,
    HalfCheetahParams,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY  # 7
comptime NJOINT = HalfCheetahModel.NJOINT  # 9
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS  # 20

comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()

# Tolerance for comparison (float64)
comptime M_TOL: Float64 = 1e-4  # Mass matrix elements
comptime M_REL_TOL: Float64 = 1e-3  # Relative tolerance for large values


# =============================================================================
# Comparison: compute mass matrix in both engines, compare
# =============================================================================


fn compare_mass_matrix(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
) raises -> Bool:
    """Compute mass matrix in both engines with identical qpos, compare."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.01),
    )
    HalfCheetahBodies.setup_model(model)
    HalfCheetahJoints.setup_model(model)
    HalfCheetahGeoms.setup_model(model)

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()

    # Set qpos
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])

    # Run FK (required before mass matrix)
    forward_kinematics(model, data)

    # Compute cdof (spatial motion axes)
    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE
    ](model, data, cdof)

    # Compute composite rigid body inertia
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
    ](model, data, crb)

    # Compute full mass matrix
    var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M[i] = Scalar[DTYPE](0)
    compute_mass_matrix_full[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        M_SIZE, CDOF_SIZE, CRB_SIZE,
    ](model, data, cdof, crb, M)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    # Set qpos in MuJoCo
    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]

    # Run MuJoCo forward
    mujoco.mj_forward(mj_model, mj_data)

    # Get full mass matrix from MuJoCo
    var nv = Int(py=mj_model.nv)
    var mj_M = np.zeros(nv * nv).reshape(nv, nv)
    mujoco.mj_fullM(mj_model, mj_M, mj_data.qM)

    # Flatten for easy access
    var mj_M_flat = mj_M.flatten().tolist()

    # Add armature to our diagonal (MuJoCo includes it in mj_fullM)
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof = joint.dof_adr
        M[dof * NV + dof] += Scalar[DTYPE](joint.armature)

    # === Compare element by element ===
    var all_pass = True
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for i in range(NV):
        for j in range(NV):
            var our_val = Float64(M[i * NV + j])
            var mj_val = Float64(py=mj_M_flat[i * nv + j])
            var abs_err = abs(our_val - mj_val)
            var ref_mag = abs(mj_val)
            var rel_err: Float64 = 0.0
            if ref_mag > 1e-10:
                rel_err = abs_err / ref_mag

            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err

            # Check: either absolute OR relative tolerance must pass
            var ok = abs_err < M_TOL or rel_err < M_REL_TOL
            if not ok:
                if fail_count < 10:  # Limit output
                    print(
                        "  FAIL M[", i, ",", j, "]",
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
        print("  FAILED", fail_count, "elements  max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err)

    # Print our matrix for debugging
    print("  Our M diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(M[i * NV + i]), end="")
    print()
    print("  MuJoCo diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_M_flat[i * nv + i]), end="")
    print()

    return all_pass


# =============================================================================
# Test cases
# =============================================================================


fn test_default_qpos() raises -> Bool:
    """Mass matrix at default qpos (rootz=0.7)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    return compare_mass_matrix("Default qpos (rootz=0.7)", qpos)


fn test_zero_qpos() raises -> Bool:
    """Mass matrix at qpos=0."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    return compare_mass_matrix("Zero qpos", qpos)


fn test_nonzero_joints() raises -> Bool:
    """Mass matrix with non-zero joint angles."""
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
    return compare_mass_matrix("Non-zero joints", qpos)


fn test_extreme_joints() raises -> Bool:
    """Mass matrix at joint limits."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[3] = -0.52   # bthigh min
    qpos[4] = 0.785   # bshin max
    qpos[5] = -0.4    # bfoot min
    qpos[6] = -1.0    # fthigh min
    qpos[7] = 0.87    # fshin max
    qpos[8] = -0.5    # ffoot min
    return compare_mass_matrix("Extreme joint angles", qpos)


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("Mass Matrix Validation: Mojo Engine vs MuJoCo Reference")
    print("=" * 60)
    print("Model: HalfCheetah (NV=9)")
    print("Tolerances: abs=", M_TOL, " rel=", M_REL_TOL)
    print()

    var num_pass = 0
    var num_fail = 0

    if test_default_qpos():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_zero_qpos():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_nonzero_joints():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_extreme_joints():
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
