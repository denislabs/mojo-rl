"""Diagnostic: Compare qDeriv between Mojo and MuJoCo.

This test isolates the RNE velocity derivative computation by comparing
the qDeriv matrix directly. MuJoCo exposes d.qDeriv after mj_forward()
with opt.integrator = mjINT_IMPLICIT (2).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_qderiv_vs_mujoco.mojo
"""

from testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from physics3d.types import Model, Data, ConeType
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.jacobian import compute_cdof, compute_composite_inertia
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
)
from physics3d.dynamics.velocity_derivatives import compute_rne_vel_derivative
from physics3d.joint_types import JNT_FREE, JNT_BALL
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS
comptime M_SIZE = NV * NV
comptime CDOF_SIZE = NV * 6
comptime CRB_SIZE = NBODY * 10


fn compare_qderiv(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
) raises:
    """Compare qDeriv between our engine and MuJoCo."""
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
    HalfCheetahModel.setup_model_and_data(model, data)
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    # Run FK + body velocities
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    # Compute cdof
    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)

    # Initialize qDeriv with passive damping (like ImplicitIntegrator does)
    var qDeriv = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        qDeriv.append(Scalar[DTYPE](0))

    # Passive damping: qDeriv[i,i] = -damping[i]
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var damp = joint.damping
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                qDeriv[(dof_adr + d) * NV + (dof_adr + d)] = -damp
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                qDeriv[(dof_adr + d) * NV + (dof_adr + d)] = -damp
        else:
            qDeriv[dof_adr * NV + dof_adr] = -damp

    # Compute RNE velocity derivative (subtracts from qDeriv)
    compute_rne_vel_derivative(model, data, cdof, qDeriv)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.cone = 0  # mjCONE_PYRAMIDAL (matches HalfCheetahModel)
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.integrator = (
        2  # mjINT_IMPLICIT (triggers full qDeriv with RNE)
    )
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]

    # Force dense Jacobian storage so qDeriv is NV*NV
    mj_model.opt.jacobian = 0  # mjJAC_DENSE

    # Re-create data after changing model options
    mj_data = mujoco.MjData(mj_model)
    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]

    # qDeriv is computed inside mj_implicit() during mj_step, NOT during mj_forward.
    # Call mj_step to trigger implicit integrator which populates qDeriv.
    mujoco.mj_step(mj_model, mj_data)

    # Print first few raw qDeriv values
    var qd_check = mj_data.qDeriv.flatten()
    print("  MJ qDeriv raw (first 10):", end="")
    for ii in range(min(10, Int(py=qd_check.shape[0]))):
        print(" ", Float64(py=qd_check[ii]), end="")
    print()

    # MuJoCo stores qDeriv in sparse format (band/sparse depending on tree).
    # Reconstruct full NV*NV dense matrix from sparse representation.
    var qd_sparse = mj_data.qDeriv.flatten().tolist()
    var D_rownnz = mj_model.D_rownnz.flatten().tolist()
    var D_rowadr = mj_model.D_rowadr.flatten().tolist()
    var D_colind = mj_model.D_colind.flatten().tolist()

    print("  MJ qDeriv sparse size:", len(qd_sparse))
    print("  D_rownnz:", D_rownnz)
    print("  D_rowadr:", D_rowadr)

    # Reconstruct full dense NV*NV from sparse
    var mj_qDeriv_dense = List[Float64]()
    for i in range(NV * NV):
        mj_qDeriv_dense.append(0.0)

    for i in range(NV):
        var nnz = Int(py=D_rownnz[i])
        var adr = Int(py=D_rowadr[i])
        for k in range(nnz):
            var j = Int(py=D_colind[adr + k])
            var val = Float64(py=qd_sparse[adr + k])
            mj_qDeriv_dense[i * NV + j] = val

    var mj_qDeriv = mj_qDeriv_dense^

    # === Compare ===
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var num_fail = 0
    var total_checked = 0

    print("  qDeriv comparison (NV x NV =", NV, "x", NV, "):")
    for i in range(NV):
        for j in range(NV):
            var our_val = Float64(qDeriv[i * NV + j])
            var mj_val = mj_qDeriv[i * NV + j]
            var abs_err = abs(our_val - mj_val)
            var ref_mag = abs(mj_val)
            var rel_err: Float64 = 0.0
            if ref_mag > 1e-10:
                rel_err = abs_err / ref_mag

            if abs_err > max_abs:
                max_abs = abs_err
            if rel_err > max_rel:
                max_rel = rel_err

            total_checked += 1
            if abs_err > 1e-6 and rel_err > 1e-4:
                num_fail += 1
                if num_fail <= 10:
                    print(
                        "    [",
                        i,
                        ",",
                        j,
                        "]",
                        " ours=",
                        our_val,
                        " mj=",
                        mj_val,
                        " abs=",
                        abs_err,
                        " rel=",
                        rel_err,
                    )

    print("  Summary: checked", total_checked, "entries, failed:", num_fail)
    print("  max_abs=", max_abs, " max_rel=", max_rel)

    # Print full matrices for inspection (first 5 rows)
    print("  Our qDeriv (first 5 rows):")
    for i in range(min(5, NV)):
        print("    row", i, ":", end="")
        for j in range(NV):
            print("", Float64(qDeriv[i * NV + j]), end="")
        print()

    print("  MJ  qDeriv (first 5 rows):")
    for i in range(min(5, NV)):
        print("    row", i, ":", end="")
        for j in range(NV):
            print("", mj_qDeriv[i * NV + j], end="")
        print()

    if num_fail == 0:
        print("  PASS")
    else:
        print("  FAIL")
        assert_true(False, "compare_qderiv failed for: " + test_name)


fn test_zero_velocity() raises:
    """Zero velocities (qDeriv should just be -damping diagonal)."""
    var qpos0 = InlineArray[Float64, NQ](fill=0.0)
    qpos0[1] = 1.5
    var qvel0 = InlineArray[Float64, NV](fill=0.0)
    compare_qderiv("Zero velocity", qpos0, qvel0)


fn test_moving_moderate_vel() raises:
    """Nonzero velocities (RNE derivative should add off-diagonal terms)."""
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    qpos1[1] = 1.5
    qpos1[2] = 0.1
    qpos1[3] = -0.3
    qpos1[6] = 0.4
    var qvel1 = InlineArray[Float64, NV](fill=0.0)
    qvel1[0] = 2.0
    qvel1[2] = 0.5
    qvel1[3] = -1.0
    qvel1[6] = 1.2
    compare_qderiv("Moving (moderate vel)", qpos1, qvel1)


fn test_fast_spinning() raises:
    """High angular velocities."""
    var qpos2 = InlineArray[Float64, NQ](fill=0.0)
    qpos2[1] = 1.5
    qpos2[2] = 0.3
    qpos2[3] = -0.5
    qpos2[4] = 0.4
    qpos2[6] = 0.5
    qpos2[7] = -0.3
    var qvel2 = InlineArray[Float64, NV](fill=0.0)
    qvel2[0] = 3.0
    qvel2[2] = 2.0
    qvel2[3] = -3.0
    qvel2[4] = 2.5
    qvel2[6] = 3.0
    qvel2[7] = -2.0
    compare_qderiv("Fast spinning", qpos2, qvel2)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
