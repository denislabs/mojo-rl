"""Diagnostic: compare diagonal vs LDL solve for HalfCheetah mass matrix."""

from physics3d.types import Model, Data, _max_one
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix,
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
)
from physics3d.dynamics.bias_forces import compute_bias_forces
from physics3d.dynamics.jacobian import compute_cdof, compute_composite_inertia
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.joint_types import JNT_FREE, JNT_BALL
from envs.half_cheetah import HalfCheetah
from std.math import abs

comptime NQ = 10
comptime NV = 10
comptime NBODY = 8
comptime NJOINT = 10
comptime MAX_CONTACTS = 20


fn main():
    print("=" * 60)
    print("HalfCheetah Mass Matrix Diagnostic")
    print("=" * 60)

    # Create HalfCheetah environment
    var env = HalfCheetah()
    var state = env.reset()

    # Use model/data via env reference
    forward_kinematics(env.model, env.data)
    compute_body_velocities(env.model, env.data)

    comptime M_SIZE2 = _max_one[NV * NV]()
    comptime V_SIZE2 = _max_one[NV]()
    comptime CDOF_SIZE2 = _max_one[NV * 6]()
    comptime CRB_SIZE2 = _max_one[NBODY * 10]()

    # Old diagonal
    var M_old = InlineArray[Scalar[DType.float64], M_SIZE2](uninitialized=True)
    for i in range(M_SIZE2):
        M_old[i] = Scalar[DType.float64](0)
    compute_mass_matrix[
        DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, M_SIZE2
    ](env.model, env.data, M_old)

    print("\nOLD diagonal mass matrix:")
    for i in range(NV):
        print("  M_old[", i, ",", i, "] =", M_old[i * NV + i])

    # New full mass matrix
    var cdof = InlineArray[Scalar[DType.float64], CDOF_SIZE2](
        uninitialized=True
    )
    compute_cdof[
        DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE2
    ](env.model, env.data, cdof)

    var crb = InlineArray[Scalar[DType.float64], CRB_SIZE2](uninitialized=True)
    for i in range(CRB_SIZE2):
        crb[i] = Scalar[DType.float64](0)
    compute_composite_inertia[
        DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE2
    ](env.model, env.data, crb)

    var M_new = InlineArray[Scalar[DType.float64], M_SIZE2](uninitialized=True)
    for i in range(M_SIZE2):
        M_new[i] = Scalar[DType.float64](0)
    compute_mass_matrix_full[
        DType.float64,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        M_SIZE2,
        CDOF_SIZE2,
        CRB_SIZE2,
    ](env.model, env.data, cdof, crb, M_new)

    print("\nNEW full mass matrix (diagonal):")
    for i in range(NV):
        print("  M_new[", i, ",", i, "] =", M_new[i * NV + i])

    print("\nFull mass matrix (non-zero entries):")
    for i in range(NV):
        for j in range(NV):
            var val = M_new[i * NV + j]
            if abs(val) > 1e-10:
                print("  M[", i, ",", j, "] =", val)

    # Check diagonal positive
    print("\nDiagonal range:")
    var min_d = M_new[0]
    var max_d = M_new[0]
    for i in range(NV):
        var d = M_new[i * NV + i]
        if d < min_d:
            min_d = d
        if d > max_d:
            max_d = d
    print(
        "  Min:",
        min_d,
        "Max:",
        max_d,
        "Ratio:",
        max_d / min_d if min_d > 1e-20 else Float64(0),
    )

    # Bias forces
    var bias = InlineArray[Scalar[DType.float64], V_SIZE2](uninitialized=True)
    for i in range(V_SIZE2):
        bias[i] = Scalar[DType.float64](0)
    compute_bias_forces[
        DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE2
    ](env.model, env.data, bias)

    var f_net = InlineArray[Scalar[DType.float64], V_SIZE2](uninitialized=True)
    for i in range(NV):
        f_net[i] = -bias[i]

    print("\nf_net (= -bias):")
    for i in range(NV):
        print("  f_net[", i, "] =", f_net[i])

    # Apply armature to mass matrix diagonal (matching integrator)
    print("\nApplying armature to M diagonal:")
    for j in range(env.model.num_joints):
        var joint = env.model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        if arm > 0:
            print("  joint", j, "dof_adr=", dof_adr, "armature=", arm)
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                M_new[(dof_adr + d) * NV + (dof_adr + d)] = (
                    M_new[(dof_adr + d) * NV + (dof_adr + d)] + arm
                )
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                M_new[(dof_adr + d) * NV + (dof_adr + d)] = (
                    M_new[(dof_adr + d) * NV + (dof_adr + d)] + arm
                )
        else:
            M_new[dof_adr * NV + dof_adr] = M_new[dof_adr * NV + dof_adr] + arm

    print("\nM diagonal AFTER armature:")
    for i in range(NV):
        print("  M[", i, ",", i, "] =", M_new[i * NV + i])

    # LDL solve
    var L = InlineArray[Scalar[DType.float64], M_SIZE2](uninitialized=True)
    var D = InlineArray[Scalar[DType.float64], V_SIZE2](uninitialized=True)
    ldl_factor[DType.float64, NV, M_SIZE2, V_SIZE2](M_new, L, D)

    print("\nLDL D diagonal:")
    for i in range(NV):
        print("  D[", i, "] =", D[i])

    var qacc_ldl = InlineArray[Scalar[DType.float64], V_SIZE2](
        uninitialized=True
    )
    for i in range(NV):
        qacc_ldl[i] = Scalar[DType.float64](0)
    ldl_solve[DType.float64, NV, M_SIZE2, V_SIZE2](L, D, f_net, qacc_ldl)

    print("\nComparison (diag vs LDL):")
    for i in range(NV):
        var m_ii = M_old[i * NV + i]
        var qacc_d = f_net[i] / m_ii if m_ii > 1e-10 else Float64(0)
        print("  DOF", i, ": diag =", qacc_d, " LDL =", qacc_ldl[i])

    # Verify M * qacc = f_net
    print("\nVerify M*qacc_ldl = f_net:")
    var max_err = Float64(0)
    for i in range(NV):
        var s = Scalar[DType.float64](0)
        for j in range(NV):
            s += M_new[i * NV + j] * qacc_ldl[j]
        var err = abs(s - f_net[i])
        if err > max_err:
            max_err = err
    print("  Max error:", max_err)
