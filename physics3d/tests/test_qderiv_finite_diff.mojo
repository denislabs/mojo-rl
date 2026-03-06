"""Verify qDeriv via finite differences.

Computes d(bias_forces)/d(qvel) analytically and via finite differences
to check if our RNE velocity derivative implementation is correct.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_qderiv_finite_diff.mojo
"""

from std.testing import assert_true, TestSuite
from std.math import abs
from std.collections import InlineArray

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.jacobian import compute_cdof
from physics3d.dynamics.bias_forces import compute_bias_forces_rne
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


fn compute_bias_at_qvel(
    model: Model[
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
    ],
    qpos: InlineArray[Float64, NQ],
    qvel: InlineArray[Float64, NV],
) raises -> List[Scalar[DTYPE]]:
    """Compute full RNE bias forces (gravity + Coriolis) for given qpos, qvel.
    """
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel[i])

    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)

    var bias = List[Scalar[DTYPE]](capacity=NV)
    for _ in range(NV):
        bias.append(Scalar[DTYPE](0))
    compute_bias_forces_rne(model, data, cdof, bias)

    return bias^


fn test_qderiv_finite_diff() raises:
    print("=" * 60)
    print("qDeriv Finite Difference Verification")
    print("=" * 60)
    print()

    # Setup model
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

    # Test configuration with nonzero velocities
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    qpos[2] = 0.1
    qpos[3] = -0.3
    qpos[6] = 0.4
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0
    qvel[2] = 0.5
    qvel[3] = -1.0
    qvel[6] = 1.2

    # === Analytical qDeriv ===
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel[i])

    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)

    # Initialize qDeriv with passive damping
    var qDeriv_analytical = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        qDeriv_analytical.append(Scalar[DTYPE](0))

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var damp = joint.damping
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                qDeriv_analytical[(dof_adr + d) * NV + (dof_adr + d)] = -damp
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                qDeriv_analytical[(dof_adr + d) * NV + (dof_adr + d)] = -damp
        else:
            qDeriv_analytical[dof_adr * NV + dof_adr] = -damp

    compute_rne_vel_derivative(model, data, cdof, qDeriv_analytical)

    # === Finite difference qDeriv ===
    var eps: Float64 = 1e-6
    var bias_ref = compute_bias_at_qvel(model, qpos, qvel)

    var qDeriv_fd = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        qDeriv_fd.append(Scalar[DTYPE](0))

    for k in range(NV):
        # Perturb qvel[k] by +eps
        var qvel_plus = InlineArray[Float64, NV](fill=0.0)
        for i in range(NV):
            qvel_plus[i] = qvel[i]
        qvel_plus[k] += eps

        var bias_plus = compute_bias_at_qvel(model, qpos, qvel_plus)

        # d(bias[i])/d(qvel[k]) ≈ (bias_plus[i] - bias_ref[i]) / eps
        for i in range(NV):
            var deriv = (Float64(bias_plus[i]) - Float64(bias_ref[i])) / eps
            # Note: qDeriv = -damping_diag - d(bias)/d(qvel)
            # The RNE part is SUBTRACTED: qDeriv -= d(bias)/d(qvel)
            # So qDeriv_fd = -damping - d(bias)/d(qvel)
            qDeriv_fd[i * NV + k] = Scalar[DTYPE](-deriv)

    # Add damping diagonal to fd result
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var damp = joint.damping
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                qDeriv_fd[(dof_adr + d) * NV + (dof_adr + d)] -= Scalar[DTYPE](
                    damp
                )
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                qDeriv_fd[(dof_adr + d) * NV + (dof_adr + d)] -= Scalar[DTYPE](
                    damp
                )
        else:
            qDeriv_fd[dof_adr * NV + dof_adr] -= Scalar[DTYPE](damp)

    # === Compare ===
    print("Analytical vs Finite Difference qDeriv:")
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var num_fail = 0

    for i in range(NV):
        for j in range(NV):
            var a_val = Float64(qDeriv_analytical[i * NV + j])
            var fd_val = Float64(qDeriv_fd[i * NV + j])
            var abs_err = abs(a_val - fd_val)
            var ref_mag = abs(fd_val)
            var rel_err: Float64 = 0.0
            if ref_mag > 1e-8:
                rel_err = abs_err / ref_mag

            if abs_err > max_abs:
                max_abs = abs_err
            if rel_err > max_rel:
                max_rel = rel_err

            if abs_err > 1e-4 and rel_err > 1e-3:
                num_fail += 1
                if num_fail <= 15:
                    print(
                        "  [",
                        i,
                        ",",
                        j,
                        "]",
                        " analytical=",
                        a_val,
                        " fd=",
                        fd_val,
                        " abs=",
                        abs_err,
                        " rel=",
                        rel_err,
                    )

    print("  Summary: checked", NV * NV, "entries, failed:", num_fail)
    print("  max_abs=", max_abs, " max_rel=", max_rel)

    # Print full matrices
    print()
    print("  Analytical qDeriv:")
    for i in range(NV):
        print("    row", i, ":", end="")
        for j in range(NV):
            print(" ", Float64(qDeriv_analytical[i * NV + j]), end="")
        print()

    print()
    print("  Finite Diff qDeriv:")
    for i in range(NV):
        print("    row", i, ":", end="")
        for j in range(NV):
            print(" ", Float64(qDeriv_fd[i * NV + j]), end="")
        print()

    if num_fail == 0:
        print()
        print("  PASS — analytical derivative matches finite differences")
    else:
        print()
        print("  FAIL — analytical derivative has errors")
    assert_true(
        num_fail == 0, "qDeriv analytical vs finite difference mismatch"
    )


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
