"""Quick diagnostic: print qDeriv at zero velocity.
Check if RNE velocity derivative produces non-zero off-diagonal terms."""

from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.jacobian import compute_cdof
from physics3d.dynamics.velocity_derivatives import compute_rne_vel_derivative
from physics3d.joint_types import JNT_FREE, JNT_BALL
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahGeoms,
    HalfCheetahParams,
)


comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS
comptime M_SIZE = NV * NV
comptime CDOF_SIZE = NV * 6


fn main() raises:
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, ConeType.ELLIPTIC
    ](
    )
    HalfCheetahBodies.setup_model(model)
    HalfCheetahJoints.setup_model(model)
    HalfCheetahGeoms.setup_model(model)

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    data.qpos[1] = 1.5  # rootz

    # Zero velocity
    for i in range(NV):
        data.qvel[i] = 0.0

    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof(model, data, cdof)

    # Initialize qDeriv with damping
    var qDeriv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        qDeriv[i] = Scalar[DTYPE](0)

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

    print("Before RNE derivative (damping only):")
    print("  Diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(qDeriv[i * NV + i]), end="")
    print()

    # Compute RNE velocity derivative
    compute_rne_vel_derivative(model, data, cdof, qDeriv)

    print("\nAfter RNE derivative:")
    print("  Diagonal:", end="")
    for i in range(NV):
        print(" ", Float64(qDeriv[i * NV + i]), end="")
    print()

    var max_offdiag: Float64 = 0.0
    var max_any: Float64 = 0.0
    for i in range(NV):
        for j in range(NV):
            var val = abs(Float64(qDeriv[i * NV + j]))
            if val > max_any:
                max_any = val
            if i != j and val > max_offdiag:
                max_offdiag = val

    print("  Max off-diagonal:", max_offdiag)
    print("  Max any:", max_any)

    print("\nFull qDeriv:")
    for i in range(NV):
        print("  row", i, ":", end="")
        for j in range(NV):
            print(" ", Float64(qDeriv[i * NV + j]), end="")
        print()
