"""Test Mass Matrix against MuJoCo reference.

Compares our full mass matrix (CRBA) output with MuJoCo's mj_fullM for the
HalfCheetah model at multiple qpos configurations. Uses Python interop to
call MuJoCo.

Run with:
    cd mojo-rl && pixi run mojo run -I . test_mass_matrix_vs_mujoco.mojo
"""

from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.types import Model, Data, _max_one
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_composite_inertia,
)
from mojo_rl.physics3d.dynamics.mass_matrix import compute_mass_matrix_full
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

comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()

# Tolerance for comparison (float64)
comptime M_TOL: Float64 = 1e-4  # Mass matrix elements
comptime M_REL_TOL: Float64 = 1e-3  # Relative tolerance for large values


# =============================================================================
# Comparison: compute mass matrix in both engines, compare
# =============================================================================


def compare_mass_matrix(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
) raises:
    """Compute mass matrix in both engines with identical qpos, compare."""
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

    # Run FK (required before mass matrix)
    forward_kinematics(model, data)

    # Compute cdof (spatial motion axes)
    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)

    # Compute composite rigid body inertia
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model, data, crb)

    # Compute full mass matrix
    var M = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(model, data, cdof, crb, M)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
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
                        "  FAIL M[",
                        i,
                        ",",
                        j,
                        "]",
                        " ours=",
                        our_val,
                        " mj=",
                        mj_val,
                        " abs_err=",
                        abs_err,
                        " rel_err=",
                        rel_err,
                    )
                fail_count += 1
                all_pass = False

    if all_pass:
        print(
            "  ALL OK  max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err
        )
    else:
        print(
            "  FAILED",
            fail_count,
            "elements  max_abs_err=",
            max_abs_err,
            " max_rel_err=",
            max_rel_err,
        )

    # Print our matrix for debugging
    print("  Our M diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(M[i * NV + i]), end="")
    print()
    print("  MuJoCo diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_M_flat[i * nv + i]), end="")
    print()

    assert_true(all_pass, "Mass matrix mismatch for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


def test_default_qpos() raises:
    """Mass matrix at default qpos (rootz=0.7)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    compare_mass_matrix("Default qpos (rootz=0.7)", qpos)


def test_zero_qpos() raises:
    """Mass matrix at qpos=0."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    compare_mass_matrix("Zero qpos", qpos)


def test_nonzero_joints() raises:
    """Mass matrix with non-zero joint angles."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0  # rootx
    qpos[1] = 0.7  # rootz
    qpos[2] = 0.3  # rooty
    qpos[3] = -0.4  # bthigh
    qpos[4] = 0.5  # bshin
    qpos[5] = -0.2  # bfoot
    qpos[6] = 0.6  # fthigh
    qpos[7] = -0.8  # fshin
    qpos[8] = 0.3  # ffoot
    compare_mass_matrix("Non-zero joints", qpos)


def test_extreme_joints() raises:
    """Mass matrix at joint limits."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[3] = -0.52  # bthigh min
    qpos[4] = 0.785  # bshin max
    qpos[5] = -0.4  # bfoot min
    qpos[6] = -1.0  # fthigh min
    qpos[7] = 0.87  # fshin max
    qpos[8] = -0.5  # ffoot min
    compare_mass_matrix("Extreme joint angles", qpos)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
