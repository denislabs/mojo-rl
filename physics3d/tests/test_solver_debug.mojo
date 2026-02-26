"""Debug test: traces constraint solver internals step by step.

Runs HalfCheetah free fall with ImplicitFastIntegrator[NewtonSolver]
(matching the real env), printing:
- Pre-solver: qacc (unconstrained), qvel (current)
- Constraint builder: per-contact penetration, v_n, imp, bias, inv_K, K
- Solver: per-iteration delta and lambda for each contact
- Post-solver: qacc (constrained), qvel (new)

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_solver_debug.mojo
"""

from math import sqrt
from builtin.math import abs

from envs.half_cheetah import HalfCheetah
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_def import (
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
)

from physics3d.types import Model, Data, _max_one
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    compute_M_inv_from_ldl,
)
from physics3d.dynamics.bias_forces import compute_bias_forces_rne
from physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_composite_inertia,
    compute_contact_jacobian_row,
)
from physics3d.collision.contact_detection import (
    detect_contacts,
    normalize_qpos_quaternions,
)
from physics3d.constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)
from physics3d.constraints.constraint_builder import (
    build_constraints,
    writeback_forces,
)
from physics3d.solver import NewtonSolver
from physics3d.solver.pgs_solver import PGSSolver


fn body_name(id: Int) -> String:
    if id == 0:
        return "torso "
    elif id == 1:
        return "bthigh"
    elif id == 2:
        return "bshin "
    elif id == 3:
        return "bfoot "
    elif id == 4:
        return "fthigh"
    elif id == 5:
        return "fshin "
    elif id == 6:
        return "ffoot "
    elif id == 7:
        return "head  "
    return "???"


fn constraint_type_name(t: Int) -> String:
    if t == CNSTR_NORMAL:
        return "NORMAL"
    elif t == CNSTR_FRICTION_T1:
        return "FRIC_T1"
    elif t == CNSTR_FRICTION_T2:
        return "FRIC_T2"
    elif t == CNSTR_LIMIT:
        return "LIMIT"
    return "???"


fn debug_step[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    step_num: Int,
    verbose: Bool = True,
) where DTYPE.is_floating_point():
    """One physics step with full debug output."""
    var dt = model.timestep
    comptime M_SIZE = _max_one[NV * NV]()
    comptime V_SIZE = _max_one[NV]()
    comptime CDOF_SIZE = _max_one[NV * 6]()
    comptime CRB_SIZE = _max_one[NBODY * 10]()

    # 1. Forward kinematics + collision
    forward_kinematics(model, data)
    compute_body_velocities(model, data)
    detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )

    if verbose:
        print("  [FK] contacts:", data.num_contacts)
        for c in range(Int(data.num_contacts)):
            var ct = data.contacts[c]
            print(
                "    c",
                c,
                ": body_a=",
                body_name(Int(ct.body_a)),
                "body_b=",
                Int(ct.body_b),
                "dist=",
                Float64(ct.dist),
                "pen=",
                -Float64(ct.dist),
            )

    # 2. Compute cdof + crb + mass matrix
    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model, data, cdof
    )
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
    ](model, data, crb)

    var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M[i] = Scalar[DTYPE](0)
    compute_mass_matrix_full[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        M_SIZE,
        CDOF_SIZE,
        CRB_SIZE,
    ](model, data, cdof, crb, M)

    # Add armature + implicit damping (M_hat = M + arm - dt*qDeriv)
    # For passive system: qDeriv[i,i] = -damping[i], so M_hat = M + arm + dt*damp
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        var damp = joint.damping
        var diag_add = arm + dt * damp
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                M[(dof_adr + d) * NV + (dof_adr + d)] = (
                    M[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
                )
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                M[(dof_adr + d) * NV + (dof_adr + d)] = (
                    M[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
                )
        else:
            M[dof_adr * NV + dof_adr] = M[dof_adr * NV + dof_adr] + diag_add

    # 3. LDL factorize and solve for unconstrained qacc
    var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var D_diag = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D_diag)

    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        bias[i] = Scalar[DTYPE](0)
    compute_bias_forces_rne[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
    ](model, data, cdof, bias)

    var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        f_net[i] = data.qfrc[i] - bias[i]

    # Passive forces: damping + stiffness + frictionloss
    for j in range(model.num_joints):
        var joint_d = model.joints[j]
        var dof_adr_d = joint_d.dof_adr
        var damp_d = joint_d.damping
        if damp_d > Scalar[DTYPE](0):
            if joint_d.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr_d + d] = (
                        f_net[dof_adr_d + d]
                        - damp_d * data.qvel[dof_adr_d + d]
                    )
            elif joint_d.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr_d + d] = (
                        f_net[dof_adr_d + d]
                        - damp_d * data.qvel[dof_adr_d + d]
                    )
            else:
                f_net[dof_adr_d] = (
                    f_net[dof_adr_d] - damp_d * data.qvel[dof_adr_d]
                )

    # Stiffness
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var qpos_adr = joint.qpos_adr
        var stiff = joint.stiffness
        var sref = joint.springref
        if stiff > Scalar[DTYPE](0):
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr + d] = f_net[dof_adr + d] - stiff * (
                        data.qpos[qpos_adr + d] - sref
                    )
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr + d] = f_net[dof_adr + d] - stiff * (
                        data.qpos[qpos_adr + d] - sref
                    )
            else:
                f_net[dof_adr] = f_net[dof_adr] - stiff * (
                    data.qpos[qpos_adr] - sref
                )

    var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc[i] = Scalar[DTYPE](0)
    ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D_diag, f_net, qacc)

    # 4. M_inv for constraint solver
    var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M_inv[i] = Scalar[DTYPE](0)
    compute_M_inv_from_ldl[DTYPE, NV, M_SIZE, V_SIZE](L, D_diag, M_inv)

    if verbose:
        print("  [PRE-SOLVER]")
        print("    qpos:", end="")
        for i in range(NQ):
            print(" ", Float64(data.qpos[i]), end="")
        print("")
        print("    qvel:", end="")
        for i in range(NV):
            print(" ", Float64(data.qvel[i]), end="")
        print("")
        print("    qacc_unconstrained:", end="")
        for i in range(NV):
            print(" ", Float64(qacc[i]), end="")
        print("")
        print("    f_net:", end="")
        for i in range(NV):
            print(" ", Float64(f_net[i]), end="")
        print("")
        print("    qfrc:", end="")
        for i in range(NV):
            print(" ", Float64(data.qfrc[i]), end="")
        print("")

    # 5. Build constraints
    comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT
    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        MAX_ROWS,
        V_SIZE,
        M_SIZE,
        CDOF_SIZE,
    ](model, data, cdof, M_inv, qacc, dt, constraints)

    if verbose:
        print("  [CONSTRAINTS] num_rows:", constraints.num_rows,
              "normals:", constraints.num_normals,
              "friction:", constraints.num_friction,
              "limits:", constraints.num_limits)
        for r in range(constraints.num_rows):
            var row = constraints.rows[r]
            print(
                "    row", r, ":",
                constraint_type_name(Int(row.constraint_type)),
                " K=", Float64(row.K),
                " bias=", Float64(row.bias),
                " inv_K_imp=", Float64(row.inv_K_imp),
                " lambda=", Float64(row.lambda_val),
                " lo=", Float64(row.lo),
                " hi=", Float64(row.hi),
            )
            if Int(row.constraint_type) == CNSTR_NORMAL:
                # Show J · qvel (velocity) and J · qacc (acceleration)
                var j_dot_qvel: Float64 = 0
                var j_dot_qacc: Float64 = 0
                for i in range(NV):
                    j_dot_qvel += Float64(constraints.J[r * NV + i]) * Float64(data.qvel[i])
                    j_dot_qacc += Float64(constraints.J[r * NV + i]) * Float64(qacc[i])
                print(
                    "      J·qvel=", j_dot_qvel,
                    " J·qacc=", j_dot_qacc,
                    " (a_n + bias)=", j_dot_qacc + Float64(row.bias),
                )
                if Int(row.source_contact_idx) >= 0:
                    var ci = Int(row.source_contact_idx)
                    print(
                        "      contact[", ci, "]: pen=",
                        -Float64(data.contacts[ci].dist),
                        " friction_coef=", Float64(row.friction_coef),
                    )

    # 6. Solve constraints
    if verbose:
        print("  [SOLVING with NewtonSolver]")
        print("    qacc before solve:", end="")
        for i in range(NV):
            print(" ", Float64(qacc[i]), end="")
        print("")

    NewtonSolver.solve[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        MAX_ROWS,
        V_SIZE,
        M_SIZE,
    ](model, data, M_inv, constraints, qacc, dt)

    if verbose:
        print("    qacc after solve:", end="")
        for i in range(NV):
            print(" ", Float64(qacc[i]), end="")
        print("")

        # Show final constraint forces
        print("    final lambdas:", end="")
        for r in range(constraints.num_rows):
            if Int(constraints.rows[r].constraint_type) == CNSTR_NORMAL:
                print(" n[", r, "]=", Float64(constraints.rows[r].lambda_val), end="")
        print("")

        # Show J·qacc after solve (should be close to aref for active contacts)
        for r in range(constraints.num_rows):
            if Int(constraints.rows[r].constraint_type) == CNSTR_NORMAL:
                var j_dot_qacc_post: Float64 = 0
                for i in range(NV):
                    j_dot_qacc_post += Float64(constraints.J[r * NV + i]) * Float64(qacc[i])
                print(
                    "    row", r, ": J·qacc_post=", j_dot_qacc_post,
                    " bias=", Float64(constraints.rows[r].bias),
                    " (a+bias)=", j_dot_qacc_post + Float64(constraints.rows[r].bias),
                )

    # 7. Writeback
    writeback_forces[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        MAX_ROWS,
    ](constraints, data)

    # 8. Integrate: qvel = old_qvel + qacc * dt
    comptime MAX_QVEL: Scalar[DTYPE] = 10.0
    for i in range(NV):
        data.qacc[i] = qacc[i]
        data.qvel[i] = data.qvel[i] + qacc[i] * dt

    # Clamp velocities
    for i in range(NV):
        if data.qvel[i] > MAX_QVEL:
            data.qvel[i] = MAX_QVEL
        elif data.qvel[i] < -MAX_QVEL:
            data.qvel[i] = -MAX_QVEL

    for i in range(NQ):
        if i < NV:
            data.qpos[i] = data.qpos[i] + data.qvel[i] * dt

    normalize_qpos_quaternions(model, data)

    if verbose:
        print("  [POST-INTEGRATION]")
        print("    qvel_new:", end="")
        for i in range(NV):
            print(" ", Float64(data.qvel[i]), end="")
        print("")
        print("    rootz=", Float64(data.qpos[JOINT_ROOTZ]),
              " vz=", Float64(data.qvel[JOINT_ROOTZ]))


fn main():
    print("")
    print("=" * 80)
    print("Constraint Solver Debug — HalfCheetah Free Fall")
    print("=" * 80)
    print("")

    var env = HalfCheetah[DType.float64, False]()
    _ = env.reset()

    var dt = Float64(env.model.timestep)
    print("dt =", dt, "s")
    print("solref_contact =", env.model.solref_contact[0], env.model.solref_contact[1])
    print("solimp_contact =", env.model.solimp_contact[0], env.model.solimp_contact[1], env.model.solimp_contact[2])
    print("solref_limit =", env.model.solref_limit[0], env.model.solref_limit[1])
    print("solimp_limit =", env.model.solimp_limit[0], env.model.solimp_limit[1], env.model.solimp_limit[2])
    print("")

    # Show initial state
    print("Initial qpos:", end="")
    for i in range(HalfCheetahModel.NQ):
        print(" ", Float64(env.data.qpos[i]), end="")
    print("")
    print("Initial qvel:", end="")
    for i in range(HalfCheetahModel.NV):
        print(" ", Float64(env.data.qvel[i]), end="")
    print("")
    print("")

    # Run steps — verbose for first few, then brief around contact, then summary
    var num_steps = 300
    var first_contact = -1

    for step in range(num_steps):
        env.data.clear_forces()

        # Decide verbosity
        var verbose = False
        if step < 3:
            verbose = True
        elif first_contact > 0 and step >= first_contact - 1 and step <= first_contact + 10:
            verbose = True

        if verbose:
            print("")
            print("=" * 60)
            print("STEP", step + 1, "(t =", Float64(step + 1) * dt, "s)")
            print("=" * 60)

        debug_step(env.model, env.data, step + 1, verbose)

        # Check for first contact
        if first_contact < 0 and Int(env.data.num_contacts) > 0:
            first_contact = step + 1
            if not verbose:
                print("\n*** FIRST CONTACT at step", first_contact, "***")

        # Brief summary every 25 steps
        if not verbose and (step + 1) % 25 == 0:
            var nc = Int(env.data.num_contacts)
            var rootz = Float64(env.data.qpos[JOINT_ROOTZ])
            var vz = Float64(env.data.qvel[JOINT_ROOTZ])
            var max_pen = Float64(0)
            for c in range(nc):
                var pen = -Float64(env.data.contacts[c].dist)
                if pen > max_pen:
                    max_pen = pen
            print(
                "  step", step + 1,
                ": rootz=", rootz,
                " vz=", vz,
                " contacts=", nc,
                " max_pen=", max_pen,
            )

    print("")
    print("=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
