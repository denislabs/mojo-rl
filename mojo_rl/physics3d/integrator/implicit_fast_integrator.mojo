"""Implicit-fast integrator matching MuJoCo 3.4.1+ implicitfast design.

Uses M_hat = M + armature + dt * dof_damping for the mass matrix, with
damping also explicit in forces (f_net -= D*v). This is algebraically
equivalent to the Euler integrator's implicit velocity damping scheme.

MuJoCo ImplicitFast (3.4.1+) calls mjd_smooth_vel(flg_bias=0), which
computes qDeriv from actuator and passive velocity derivatives only,
SKIPPING Coriolis/centripetal force derivatives (mjd_rne_vel).

For simple motor actuators without kv, qDeriv = -diag(dof_damping), so:
  M_hat = M + arm - dt*(-diag(D)) = M + arm + dt*D

Note: MuJoCo 3.3.6 includes Coriolis derivatives for ImplicitFast, causing
~1-5% velocity-dependent drift vs our implementation. This was changed in
3.4.x per the original design intent (see MuJoCo 3.0 changelog).
"""

from std.math import sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from mojo_rl.deep_agents.core.perf_timer import PerfTimer

from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from ..kinematics.quat_math import quat_normalize, quat_integrate, quat_rotate
from ..dynamics.mass_matrix import (
    compute_mass_matrix,
    compute_mass_matrix_full,
    compute_mass_matrix_full_gpu,
    ldl_factor,
    ldl_factor_gpu,
    ldl_solve,
    ldl_solve_gpu,
    ldl_solve_workspace_gpu,
    compute_M_inv_from_ldl,
    compute_M_inv_from_ldl_gpu,
    solve_linear_diagonal,
    build_sparse_pattern,
    compute_mass_matrix_sparse,
    ldl_factor_sparse,
    ldl_solve_sparse,
    sparse_to_dense,
    # Sparse GPU functions
    build_sparse_pattern_gpu,
    compute_mass_matrix_sparse_gpu,
    ldl_factor_sparse_gpu,
    ldl_solve_sparse_gpu,
    compute_M_inv_from_sparse_ldl_gpu,
    SparseMassMatrix,
    _ensure_positive,
)
from ..dynamics.bias_forces import (
    compute_bias_forces,
    compute_bias_forces_rne,
    compute_bias_forces_rne_gpu,
)
from ..dynamics.jacobian import (
    compute_subtree_com,
    compute_cdof,
    compute_subtree_com_gpu,
    compute_cdof_gpu,
    compute_composite_inertia,
    compute_composite_inertia_gpu,
)
from ..collision.contact_detection import (
    normalize_qpos_quaternions,
    normalize_qpos_quaternions_gpu,
)
from ..collision.broadphase_sap import (
    detect_contacts_auto,
    detect_contacts_auto_gpu,
)
from ..solver.pgs_solver import PGSSolver
from ..constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)
from ..constraints.constraint_builder import (
    build_constraints,
    writeback_forces,
)
from ..dynamics.cfrc_ext import compute_cfrc_ext
from ..traits.integrator import Integrator
from ..traits.solver import ConstraintSolver
from ..gpu.constants import (
    TPB,
    state_size,
    model_size,
    model_size_with_invweight,
    model_metadata_offset,
    model_joint_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_DAMPING,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_NJOINT,
    integrator_workspace_size,
    ws_M_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    metadata_offset,
    META_IDX_NUM_CONTACTS,
)


struct ImplicitFastIntegrator[SOLVER: ConstraintSolver](Integrator):
    """Implicit-fast integrator with qDeriv-based mass matrix modification.

    Uses M_hat = M + armature - dt * qDeriv where qDeriv = d(forces)/d(qvel).
    Currently qDeriv only includes passive damping; extensible for actuators.

    Parametrized by SOLVER type (PGSSolver, NewtonSolver, or CGSolver).

    Usage:
        # PGS (default):
        alias PGSImplicitFast = ImplicitFastIntegrator[PGSSolver]

        # Newton (most accurate, matches MuJoCo):
        alias NewtonImplicitFast = ImplicitFastIntegrator[NewtonSolver]
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    @staticmethod
    def step[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
    ](
        model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        verbose: Bool = False,
    ):
        """Execute one simulation step with implicit-fast integration.

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
            verbose: Whether to print debug information.
        """
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"
        var dt = model.timestep
        comptime M_SIZE = _max_one[NV * NV]()
        comptime V_SIZE = _max_one[NV]()
        comptime CDOF_SIZE = _max_one[NV * 6]()
        comptime CRB_SIZE = _max_one[NBODY * 10]()
        comptime NM_SAFE = _ensure_positive[NM]()

        # 1. Forward kinematics
        forward_kinematics(model, data)
        compute_body_velocities(model, data)

        # 2. Collision detection
        detect_contacts_auto[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
            model, data
        )

        if verbose:
            print("  [FK] body positions:")
            for b in range(NBODY):
                print(
                    "    body",
                    b,
                    "(",
                    model.get_body_name(b),
                    "): x=",
                    Float64(data.xpos[b * 3]),
                    " y=",
                    Float64(data.xpos[b * 3 + 1]),
                    " z=",
                    Float64(data.xpos[b * 3 + 2]),
                )
            print("  [FK] body CoM positions (xipos):")
            for b in range(NBODY):
                print(
                    "    body",
                    b,
                    "(",
                    model.get_body_name(b),
                    "): x=",
                    Float64(data.xipos[b * 3]),
                    " y=",
                    Float64(data.xipos[b * 3 + 1]),
                    " z=",
                    Float64(data.xipos[b * 3 + 2]),
                )
            print("  [FK] contacts:", data.num_contacts)
            for c in range(Int(data.num_contacts)):
                var ct = data.contacts[c]
                var ba = Int(ct.body_a)
                var bb = Int(ct.body_b)
                var ba_name = String("ground") if ba == 0 else String(
                    model.get_body_name(ba)
                )
                var bb_name = String("ground") if bb == 0 else String(
                    model.get_body_name(bb)
                )
                print(
                    "    c",
                    c,
                    ": body_a=",
                    ba_name,
                    "(",
                    ba,
                    ") body_b=",
                    bb_name,
                    "(",
                    bb,
                    ") dist=",
                    Float64(ct.dist),
                    "pen=",
                    -Float64(ct.dist),
                )

        # 3a. Compute subtree CoM (MuJoCo mj_comPos)
        var stcom_tmp = List[Scalar[DTYPE]](capacity=NBODY * 3)
        for _ in range(NBODY * 3):
            stcom_tmp.append(Scalar[DTYPE](0))
        compute_subtree_com(model, data, stcom_tmp)
        for sc_i in range(NBODY * 3):
            data.subtree_com[sc_i] = stcom_tmp[sc_i]
        data.has_subtree_com = True

        # 3b. Compute cdof (spatial motion axes per DOF)
        var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
        for _ in range(CDOF_SIZE):
            cdof.append(Scalar[DTYPE](0))
        compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
            model, data, cdof, stcom_tmp
        )

        # 4. Compute composite rigid body inertia
        var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
        for _ in range(CRB_SIZE):
            crb.append(Scalar[DTYPE](0))
        compute_composite_inertia[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
            model, data, crb
        )

        # 5. Compute mass matrix using CRBA
        var M = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            M.append(Scalar[DTYPE](0))
        var sM = SparseMassMatrix[DTYPE, NV, NM]()

        comptime if SPARSE:
            build_sparse_pattern[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                NM,
                NGEOM,
                MAX_EQUALITY,
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
            ](model, sM)
            compute_mass_matrix_sparse[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                NM,
                CDOF_SIZE,
                CRB_SIZE,
                NGEOM,
                MAX_EQUALITY,
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
            ](model, data, cdof, crb, sM)
        else:
            compute_mass_matrix_full(model, data, cdof, crb, M)

        # 5b. Add armature only to mass matrix diagonal
        # MuJoCo: constraint solver sees M + arm (no dt*D).
        # The dt*D is added later in the post-constraint re-solve step.
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var arm = joint.armature

            comptime if SPARSE:
                if joint.jnt_type == JNT_FREE:
                    for d in range(6):
                        sM.values[sM.diag_pos(dof_adr + d)] += arm
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        sM.values[sM.diag_pos(dof_adr + d)] += arm
                else:
                    sM.values[sM.diag_pos(dof_adr)] += arm
            else:
                if joint.jnt_type == JNT_FREE:
                    for d in range(6):
                        M[(dof_adr + d) * NV + (dof_adr + d)] = (
                            M[(dof_adr + d) * NV + (dof_adr + d)] + arm
                        )
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        M[(dof_adr + d) * NV + (dof_adr + d)] = (
                            M[(dof_adr + d) * NV + (dof_adr + d)] + arm
                        )
                else:
                    M[dof_adr * NV + dof_adr] = M[dof_adr * NV + dof_adr] + arm

        # 5c. Expand sparse to dense for M_hat (must be before ldl_factor_sparse mutates sM)
        comptime if SPARSE:
            sparse_to_dense[DTYPE, NV, NM](sM, M)

        # 6. LDL factorize M and solve for qacc
        var L = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            L.append(Scalar[DTYPE](0))
        var D = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            D.append(Scalar[DTYPE](0))

        comptime if SPARSE:
            ldl_factor_sparse(sM)
        else:
            ldl_factor[DTYPE, NV](M, L, D)

        var bias = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            bias.append(Scalar[DTYPE](0))
        compute_bias_forces_rne[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
            model, data, cdof, bias
        )

        var f_net = List[Scalar[DTYPE]](capacity=V_SIZE)
        for i in range(NV):
            f_net.append(data.qfrc[i] - bias[i])

        # 6b. Apply passive joint forces: damping + stiffness + frictionloss
        # Damping force: f -= damping * qvel (purely explicit in MuJoCo 3.3.6)
        # MuJoCo's qDeriv = 0 for simple motor actuators, so damping is NOT
        # incorporated into M_hat — it's handled entirely through this force term.
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

        # Stiffness: f -= stiffness * (qpos - springref)
        # Frictionloss: f -= frictionloss * sign(qvel)
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var qpos_adr = joint.qpos_adr
            var stiff = joint.stiffness
            var sref = joint.springref
            var floss = joint.frictionloss
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
            if floss > Scalar[DTYPE](0):
                comptime VEL_THRESH: Scalar[DTYPE] = 1e-4
                if joint.jnt_type == JNT_FREE:
                    for d in range(6):
                        var v = data.qvel[dof_adr + d]
                        if v > VEL_THRESH:
                            f_net[dof_adr + d] = f_net[dof_adr + d] - floss
                        elif v < -VEL_THRESH:
                            f_net[dof_adr + d] = f_net[dof_adr + d] + floss
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        var v = data.qvel[dof_adr + d]
                        if v > VEL_THRESH:
                            f_net[dof_adr + d] = f_net[dof_adr + d] - floss
                        elif v < -VEL_THRESH:
                            f_net[dof_adr + d] = f_net[dof_adr + d] + floss
                else:
                    var v = data.qvel[dof_adr]
                    if v > VEL_THRESH:
                        f_net[dof_adr] = f_net[dof_adr] - floss
                    elif v < -VEL_THRESH:
                        f_net[dof_adr] = f_net[dof_adr] + floss

        # qacc = M^-1 * f_net via LDL solve
        var qacc = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            qacc.append(Scalar[DTYPE](0))

        comptime if SPARSE:
            ldl_solve_sparse[DTYPE, NV, NM](sM, f_net, qacc)
        else:
            ldl_solve[DTYPE, NV](L, D, f_net, qacc)

        # 7. Compute full M_inv from LDL factors for constraint solver
        var M_inv = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            M_inv.append(Scalar[DTYPE](0))

        comptime if SPARSE:
            # Compute M_inv column-by-column: solve M * e_j = e_j for each j
            var e_col = List[Scalar[DTYPE]](capacity=V_SIZE)
            for _ in range(V_SIZE):
                e_col.append(Scalar[DTYPE](0))
            var col_result = List[Scalar[DTYPE]](capacity=V_SIZE)
            for _ in range(V_SIZE):
                col_result.append(Scalar[DTYPE](0))
            for col in range(NV):
                for k in range(NV):
                    e_col[k] = Scalar[DTYPE](1) if k == col else Scalar[DTYPE](
                        0
                    )
                ldl_solve_sparse[DTYPE, NV, NM](sM, e_col, col_result)
                for row in range(NV):
                    M_inv[row * NV + col] = col_result[row]
        else:
            compute_M_inv_from_ldl[DTYPE, NV](L, D, M_inv)

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
            print("    M_hat diagonal:", end="")
            for i in range(NV):
                print(" ", Float64(M[i * NV + i]), end="")
            print("")
            print("    bias (RNE):", end="")
            for i in range(NV):
                print(" ", Float64(bias[i]), end="")
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

        # 8. Build constraints and solve (modifies qacc in-place)
        comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT + 6 * MAX_EQUALITY + MAX_TENDON
        var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
        build_constraints[CONE_TYPE=CONE_TYPE, MAX_TENDON=MAX_TENDON](
            model, data, cdof, M_inv, dt, constraints
        )

        if verbose:
            print(
                "  [CONSTRAINTS] num_rows:",
                constraints.num_rows,
                "normals:",
                constraints.num_normals,
                "friction:",
                constraints.num_friction,
                "limits:",
                constraints.num_limits,
            )
            for r in range(constraints.num_rows):
                var row = constraints.rows[r]
                var ct_name: String
                if Int(row.constraint_type) == CNSTR_NORMAL:
                    ct_name = "NORMAL"
                elif Int(row.constraint_type) == CNSTR_FRICTION_T1:
                    ct_name = "FRIC_T1"
                elif Int(row.constraint_type) == CNSTR_FRICTION_T2:
                    ct_name = "FRIC_T2"
                elif Int(row.constraint_type) == CNSTR_LIMIT:
                    ct_name = "LIMIT"
                else:
                    ct_name = "???"
                print(
                    "    row",
                    r,
                    ":",
                    ct_name,
                    " K=",
                    Float64(row.K),
                    " bias=",
                    Float64(row.bias),
                    " inv_K_imp=",
                    Float64(row.inv_K_imp),
                    " lambda=",
                    Float64(row.lambda_val),
                    " lo=",
                    Float64(row.lo),
                    " hi=",
                    Float64(row.hi),
                )
                if Int(row.constraint_type) == CNSTR_NORMAL:
                    var j_dot_qvel: Float64 = 0
                    var j_dot_qacc: Float64 = 0
                    for i in range(NV):
                        j_dot_qvel += Float64(
                            constraints.J[r * NV + i]
                        ) * Float64(data.qvel[i])
                        j_dot_qacc += Float64(
                            constraints.J[r * NV + i]
                        ) * Float64(qacc[i])
                    print(
                        "      J·qvel=",
                        j_dot_qvel,
                        " J·qacc=",
                        j_dot_qacc,
                        " (a_n + bias)=",
                        j_dot_qacc + Float64(row.bias),
                    )
                    if Int(row.source_contact_idx) >= 0:
                        var ci = Int(row.source_contact_idx)
                        print(
                            "      contact[",
                            ci,
                            "]: pen=",
                            -Float64(data.contacts[ci].dist),
                            " friction_coef=",
                            Float64(row.friction_coef),
                        )

            print("  [SOLVING]")
            print("    qacc before solve:", end="")
            for i in range(NV):
                print(" ", Float64(qacc[i]), end="")
            print("")

        # Fill M_hat and qfrc_smooth for primal solvers
        # M_hat = M + arm (already computed as M above)
        for i in range(NV * NV):
            constraints.M_hat[i] = M[i]
        for i in range(NV):
            constraints.qfrc_smooth[i] = f_net[i]

        Self.SOLVER.solve[CONE_TYPE=CONE_TYPE](
            model, data, M_inv, constraints, qacc, dt
        )

        if verbose:
            print("    qacc after solve:", end="")
            for i in range(NV):
                print(" ", Float64(qacc[i]), end="")
            print("")

            # Show final constraint forces
            print("    final lambdas:", end="")
            for r in range(constraints.num_rows):
                if Int(constraints.rows[r].constraint_type) == CNSTR_NORMAL:
                    print(
                        " n[",
                        r,
                        "]=",
                        Float64(constraints.rows[r].lambda_val),
                        end="",
                    )
            print("")

            # Show J·qacc after solve with full KKT residual
            # At convergence: a + bias + R*lambda = 0 for active contacts
            for r in range(constraints.num_rows):
                if Int(constraints.rows[r].constraint_type) == CNSTR_NORMAL:
                    var j_dot_qacc_post: Float64 = 0
                    for i in range(NV):
                        j_dot_qacc_post += Float64(
                            constraints.J[r * NV + i]
                        ) * Float64(qacc[i])
                    var lam = Float64(constraints.rows[r].lambda_val)
                    var K_val = Float64(constraints.rows[r].K)
                    var inv_Ki = Float64(constraints.rows[r].inv_K_imp)
                    var R_val = 1.0 / inv_Ki - K_val if inv_Ki > 1e-14 else 0.0
                    var kkt_residual = (
                        j_dot_qacc_post
                        + Float64(constraints.rows[r].bias)
                        + R_val * lam
                    )
                    print(
                        "    row",
                        r,
                        ": lambda=",
                        lam,
                        " a+bias=",
                        j_dot_qacc_post + Float64(constraints.rows[r].bias),
                        " R*lam=",
                        R_val * lam,
                        " KKT=",
                        kkt_residual,
                    )

        writeback_forces[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_ROWS,
        ](constraints, data)

        # Compute cfrc_ext: contact forces per body in subtree CoM frame
        compute_cfrc_ext[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
        ](model, data)

        # 9. Post-constraint re-solve with M_hat = M + arm + dt*D
        # MuJoCo pattern: constraint solver uses M+arm, then the integrator
        # re-solves qacc = M_hat^{-1} * (qfrc_smooth + qfrc_constraint).
        # This is equivalent to mj_implicitSkip in MuJoCo.

        # 9a. Compute qfrc_constraint = sum(J^T * lambda) for all constraints
        var qfrc_constraint = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            qfrc_constraint.append(Scalar[DTYPE](0))
        for r in range(constraints.num_rows):
            var lam = constraints.rows[r].lambda_val
            for i in range(NV):
                qfrc_constraint[i] += constraints.J[r * NV + i] * lam

        # 9b. qfrc_total = qfrc_smooth + qfrc_constraint
        #     where qfrc_smooth = f_net (already computed: qfrc - bias - D*v - K*(q-qref))
        var qfrc_total = List[Scalar[DTYPE]](capacity=V_SIZE)
        for i in range(NV):
            qfrc_total.append(f_net[i] + qfrc_constraint[i])

        # 9c. Add dt*damping to M diagonal → M_hat = M + arm + dt*D
        comptime if not SPARSE:
            for j in range(model.num_joints):
                var joint = model.joints[j]
                var dof_adr = joint.dof_adr
                var damp = joint.damping
                if damp > Scalar[DTYPE](0):
                    if joint.jnt_type == JNT_FREE:
                        for d in range(6):
                            M[(dof_adr + d) * NV + (dof_adr + d)] += dt * damp
                    elif joint.jnt_type == JNT_BALL:
                        for d in range(3):
                            M[(dof_adr + d) * NV + (dof_adr + d)] += dt * damp
                    else:
                        M[dof_adr * NV + dof_adr] += dt * damp

        # 9d. Re-factor M_hat and solve qacc = M_hat^{-1} * qfrc_total
        for i in range(NV):
            qacc[i] = Scalar[DTYPE](0)

        comptime if SPARSE:
            # Recompute sparse M, add armature + dt*damping to diagonal, then factor
            compute_mass_matrix_sparse[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                NM,
                CDOF_SIZE,
                CRB_SIZE,
                NGEOM,
                MAX_EQUALITY,
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
            ](model, data, cdof, crb, sM)
            for j2 in range(model.num_joints):
                var joint2 = model.joints[j2]
                var dof2 = joint2.dof_adr
                var arm2 = joint2.armature
                var damp2 = joint2.damping
                var add2 = arm2 + dt * damp2
                if joint2.jnt_type == JNT_FREE:
                    for d in range(6):
                        sM.values[sM.diag_pos(dof2 + d)] += add2
                elif joint2.jnt_type == JNT_BALL:
                    for d in range(3):
                        sM.values[sM.diag_pos(dof2 + d)] += add2
                else:
                    sM.values[sM.diag_pos(dof2)] += add2
            ldl_factor_sparse(sM)
            ldl_solve_sparse[DTYPE, NV, NM](sM, qfrc_total, qacc)
        else:
            ldl_factor[DTYPE, NV](M, L, D)
            ldl_solve[DTYPE, NV](L, D, qfrc_total, qacc)

        # 9e. Integrate: qvel += dt * qacc, qpos += dt * qvel
        for i in range(NV):
            data.qacc[i] = qacc[i]
            data.qvel[i] = data.qvel[i] + qacc[i] * dt

        # Integrate position: qpos += qvel * dt (quaternion-aware)
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var qpos_adr = joint.qpos_adr
            var dof_adr = joint.dof_adr

            if joint.jnt_type == JNT_FREE:
                for d in range(3):
                    data.qpos[qpos_adr + d] = (
                        data.qpos[qpos_adr + d] + data.qvel[dof_adr + d] * dt
                    )
                var qx = data.qpos[qpos_adr + 3]
                var qy = data.qpos[qpos_adr + 4]
                var qz = data.qpos[qpos_adr + 5]
                var qw = data.qpos[qpos_adr + 6]
                var wx = data.qvel[dof_adr + 3]
                var wy = data.qvel[dof_adr + 4]
                var wz = data.qvel[dof_adr + 5]
                var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
                var norm = quat_normalize(
                    result[0], result[1], result[2], result[3]
                )
                data.qpos[qpos_adr + 3] = norm[0]
                data.qpos[qpos_adr + 4] = norm[1]
                data.qpos[qpos_adr + 5] = norm[2]
                data.qpos[qpos_adr + 6] = norm[3]

            elif joint.jnt_type == JNT_HINGE or joint.jnt_type == JNT_SLIDE:
                data.qpos[qpos_adr] = (
                    data.qpos[qpos_adr] + data.qvel[dof_adr] * dt
                )

        # 10. Normalize quaternions (handles remaining cases like BALL)
        normalize_qpos_quaternions(model, data)

        if verbose:
            print("  [POST-INTEGRATION]")
            print("    qvel_new:", end="")
            for i in range(NV):
                print(" ", Float64(data.qvel[i]), end="")
            print("")
            print("    qpos_new:", end="")
            for i in range(NQ):
                print(" ", Float64(data.qpos[i]), end="")
            print("")

        # 11. Joint limits now enforced as constraints inside the solver
        # (no post-step clamping needed)

    @staticmethod
    def simulate[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
    ](
        model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        num_steps: Int,
    ):
        """Run simulation for multiple steps on CPU."""
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"
        for _ in range(num_steps):
            Self.step[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                NGEOM,
                MAX_EQUALITY,
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
                NM,
                SPARSE,
            ](model, data)

    # =========================================================================
    # GPU Methods
    # =========================================================================

    @always_inline
    @staticmethod
    def step_kernel[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        BATCH: Int,
        WS_SIZE: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
    ):
        """Complete implicit-fast physics step kernel (pre-solver).

        Pipeline:
        1-6. Same as EulerIntegrator (FK, contacts, CRBA, mass matrix)
        6b. qDeriv-based mass matrix modification
        7-10. LDL factorize, bias forces, f_net with damping, predict velocity
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        comptime V_SIZE = _max_one[NV]()
        comptime M_idx = ws_M_offset[NV, NBODY]()
        comptime bias_idx = ws_bias_offset[NV, NBODY]()
        comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
        comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()
        comptime qacc_constrained_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime m_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime NM_SAFE = _ensure_positive[NM]()
        var sp_row_nnz = InlineArray[Int, _ensure_positive[NV]()](fill=0)
        var sp_row_adr = InlineArray[Int, _ensure_positive[NV]()](fill=0)
        var sp_col_ind = InlineArray[Int, NM_SAFE](fill=0)

        comptime if SPARSE:
            _ = build_sparse_pattern_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, MODEL_SIZE
            ](model, sp_row_nnz, sp_row_adr, sp_col_ind)

        # 1. Forward kinematics
        forward_kinematics_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
        ](env, state, model)

        # 2. Compute body velocities
        compute_body_velocities_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
        ](env, state, model)

        # 3. Detect contacts
        detect_contacts_auto_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            NGEOM,
            MAX_EQUALITY,
            MAX_TENDON,
            NSITE,
        ](env, state, model)

        # 3a. Compute subtree_com
        compute_subtree_com_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, BATCH,
        ](env, state, model)

        # 4. Compute cdof (writes to workspace at ws_cdof_offset)
        compute_cdof_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
        ](env, state, model, workspace)

        # 5. Compute composite rigid body inertia (writes to workspace at ws_crb_offset)
        compute_composite_inertia_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
        ](env, state, model, workspace)

        # 6. Compute mass matrix using CRBA (reads cdof/crb, writes M in workspace)
        comptime if SPARSE:
            compute_mass_matrix_sparse_gpu[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                NM,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
                WS_SIZE,
            ](env, state, model, workspace, sp_row_nnz, sp_row_adr, sp_col_ind)
        else:
            compute_mass_matrix_full_gpu[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
                WS_SIZE,
            ](env, state, model, workspace)

        # 6b. Add armature only to mass matrix diagonal
        # MuJoCo: constraint solver sees M + arm (no dt*D).
        # The dt*D is added later in step_finalize_kernel.
        var model_meta_off_early = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off_early + MODEL_META_IDX_TIMESTEP]
        )
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(model[0, joint_off + JOINT_IDX_TYPE])
            var dof_adr = Int(model[0, joint_off + JOINT_IDX_DOF_ADR])
            var arm = model[0, joint_off + JOINT_IDX_ARMATURE]
            if jnt_type == JNT_FREE:
                for d in range(6):
                    var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                    workspace[env, idx] += arm
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                    workspace[env, idx] += arm
            else:
                var idx = M_idx + dof_adr * NV + dof_adr
                workspace[env, idx] += arm

        # 7. LDL factorize M, conditionally compute M_inv
        comptime if SPARSE:
            ldl_factor_sparse_gpu[DTYPE, NV, NBODY, NM, BATCH, WS_SIZE](
                env, workspace, sp_row_nnz, sp_row_adr, sp_col_ind
            )
            comptime if Self.SOLVER.NEEDS_M_INV:
                compute_M_inv_from_sparse_ldl_gpu[
                    DTYPE, NV, NBODY, NM, BATCH, WS_SIZE
                ](env, workspace, sp_row_nnz, sp_row_adr, sp_col_ind)
        else:
            ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)
            comptime if Self.SOLVER.NEEDS_M_INV:
                compute_M_inv_from_ldl_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
                    env, workspace
                )

        # 8. Compute bias forces (reads cdof from workspace, writes bias to workspace)
        compute_bias_forces_rne_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
        ](env, state, model, workspace)

        # 9. Compute unconstrained acceleration via LDL solve
        var qvel_off = qvel_offset[NQ, NV]()
        var qacc_off = qacc_offset[NQ, NV]()
        var qfrc_off = qfrc_offset[NQ, NV]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        # f_net = qfrc - bias (write to workspace fnet region)
        for i in range(NV):
            var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
            var bias_val = rebind[Scalar[DTYPE]](workspace[env, bias_idx + i])
            workspace[env, fnet_idx + i] = qfrc - bias_val

        # 8b. Apply passive joint forces: damping + stiffness + frictionloss
        # Damping force: f -= damping * qvel (explicit part)
        for j in range(NJOINT):
            var joint_off_d = model_joint_offset[NBODY](j)
            var jnt_type_d = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off_d + JOINT_IDX_TYPE])
            )
            var dof_adr_d = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off_d + JOINT_IDX_DOF_ADR])
            )
            var damp_d = rebind[Scalar[DTYPE]](
                model[0, joint_off_d + JOINT_IDX_DAMPING]
            )
            if damp_d > Scalar[DTYPE](0):
                if jnt_type_d == JNT_FREE:
                    for d in range(6):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr_d + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr_d + d]
                        )
                        workspace[env, fnet_idx + dof_adr_d + d] = (
                            cur - damp_d * v
                        )
                elif jnt_type_d == JNT_BALL:
                    for d in range(3):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr_d + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr_d + d]
                        )
                        workspace[env, fnet_idx + dof_adr_d + d] = (
                            cur - damp_d * v
                        )
                else:
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr_d]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr_d]
                    )
                    workspace[env, fnet_idx + dof_adr_d] = cur - damp_d * v

        # Stiffness + frictionloss
        var qpos_off_stiff = qpos_offset[NQ, NV]()
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
            )
            var stiff = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_STIFFNESS]
            )
            var sref = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_SPRINGREF]
            )
            var floss = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_FRICTIONLOSS]
            )
            # Stiffness: f -= stiffness * (qpos - springref)
            if stiff > Scalar[DTYPE](0):
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var qpos_d = rebind[Scalar[DTYPE]](
                            state[env, qpos_off_stiff + qpos_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = cur - stiff * (
                            qpos_d - sref
                        )
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var qpos_d = rebind[Scalar[DTYPE]](
                            state[env, qpos_off_stiff + qpos_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = cur - stiff * (
                            qpos_d - sref
                        )
                else:
                    var qpos_d = rebind[Scalar[DTYPE]](
                        state[env, qpos_off_stiff + qpos_adr]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    workspace[env, fnet_idx + dof_adr] = cur - stiff * (
                        qpos_d - sref
                    )
            # Frictionloss: f -= frictionloss * sign(qvel)
            if floss > Scalar[DTYPE](0):
                comptime VEL_THRESH: Scalar[DTYPE] = 1e-4
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        if v > VEL_THRESH:
                            workspace[env, fnet_idx + dof_adr + d] = cur - floss
                        elif v < -VEL_THRESH:
                            workspace[env, fnet_idx + dof_adr + d] = cur + floss
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        if v > VEL_THRESH:
                            workspace[env, fnet_idx + dof_adr + d] = cur - floss
                        elif v < -VEL_THRESH:
                            workspace[env, fnet_idx + dof_adr + d] = cur + floss
                else:
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    if v > VEL_THRESH:
                        workspace[env, fnet_idx + dof_adr] = cur - floss
                    elif v < -VEL_THRESH:
                        workspace[env, fnet_idx + dof_adr] = cur + floss

        # LDL solve: reads f_net from workspace, writes qacc to workspace
        comptime if SPARSE:
            ldl_solve_sparse_gpu[DTYPE, NV, NBODY, NM, BATCH, WS_SIZE](
                env, workspace, sp_row_nnz, sp_row_adr, sp_col_ind
            )
        else:
            ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
                env, workspace
            )

        # 10. Warm-start: use previous step's constrained qacc if nonzero AND contacts exist.
        # state[env, qacc_off + i] still holds prev step's constrained qacc here
        # (written by step_finalize_kernel). Must read BEFORE overwriting below.
        # IMPORTANT: Only warm-start when there are active contacts. Without contacts,
        # warm-starting freezes qacc at the first step value (bias forces change with qpos
        # but step_finalize uses M*qacc_constrained which never updates for nc=0).
        comptime meta_off_ws = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        var nc_for_ws = Int(
            rebind[Scalar[DTYPE]](
                state[env, meta_off_ws + META_IDX_NUM_CONTACTS]
            )
        )
        if nc_for_ws > MAX_CONTACTS:
            nc_for_ws = MAX_CONTACTS
        var has_warmstart = False
        if nc_for_ws > 0:
            for i in range(NV):
                if rebind[Scalar[DTYPE]](state[env, qacc_off + i]) != Scalar[
                    DTYPE
                ](0):
                    has_warmstart = True
                    break

        if has_warmstart:
            for i in range(NV):
                workspace[env, qacc_constrained_idx + i] = state[
                    env, qacc_off + i
                ]
        else:
            for i in range(NV):
                workspace[env, qacc_constrained_idx + i] = workspace[
                    env, qacc_ws_idx + i
                ]

        # Write unconstrained qacc to state (overwrites previous warmstart slot)
        for i in range(NV):
            var qacc_val = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            state[env, qacc_off + i] = qacc_val

    @always_inline
    @staticmethod
    def step_finalize_kernel[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        BATCH: Int,
        WS_SIZE: Int,
        NM: Int = 0,
        SPARSE: Bool = False,
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
    ):
        """Finalize physics step: write back velocity, integrate position.

        Pipeline:
        9. Write back constrained velocity, integrate position
        10. Normalize quaternions
        11. Enforce joint limits
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var qvel_off = qvel_offset[NQ, NV]()
        var qacc_off = qacc_offset[NQ, NV]()
        var qacc_constrained_idx = ws_qacc_constrained_offset[NV, NBODY]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )

        # 9. Post-constraint re-solve with M_hat = M + arm + dt*D
        # MuJoCo pattern: constraint solver used M+arm, now re-solve with M_hat.
        # qacc_final = M_hat^{-1} * (qfrc_smooth + qfrc_constraint)
        #            = M_hat^{-1} * M * qacc_constrained
        # (since qacc_constrained = M_inv * (qfrc_smooth + qfrc_constraint))

        comptime M_idx = ws_M_offset[NV, NBODY]()
        comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
        comptime NM_SAFE = _ensure_positive[NM]()
        var sp_row_nnz = InlineArray[Int, _ensure_positive[NV]()](fill=0)
        var sp_row_adr = InlineArray[Int, _ensure_positive[NV]()](fill=0)
        var sp_col_ind = InlineArray[Int, NM_SAFE](fill=0)

        comptime if SPARSE:
            _ = build_sparse_pattern_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, MODEL_SIZE
            ](model, sp_row_nnz, sp_row_adr, sp_col_ind)

        # 9a. Compute qfrc_total = M * qacc_constrained (store in fnet workspace)
        for i in range(NV):
            var sum = Scalar[DTYPE](0)
            for j in range(NV):
                var m_ij = rebind[Scalar[DTYPE]](
                    workspace[env, M_idx + i * NV + j]
                )
                var qacc_j = rebind[Scalar[DTYPE]](
                    workspace[env, qacc_constrained_idx + j]
                )
                sum += m_ij * qacc_j
            workspace[env, fnet_idx + i] = sum

        # 9b. Add dt*damping to M diagonal → M_hat = M + arm + dt*D
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )
            var damp = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_DAMPING]
            )
            if damp > Scalar[DTYPE](0):
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                        var cur = rebind[Scalar[DTYPE]](workspace[env, idx])
                        workspace[env, idx] = cur + dt * damp
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                        var cur = rebind[Scalar[DTYPE]](workspace[env, idx])
                        workspace[env, idx] = cur + dt * damp
                else:
                    var idx = M_idx + dof_adr * NV + dof_adr
                    var cur = rebind[Scalar[DTYPE]](workspace[env, idx])
                    workspace[env, idx] = cur + dt * damp

        # 9c. Re-factor M_hat, solve qacc_final = M_hat^{-1} * qfrc_total
        comptime if SPARSE:
            ldl_factor_sparse_gpu[DTYPE, NV, NBODY, NM, BATCH, WS_SIZE](
                env, workspace, sp_row_nnz, sp_row_adr, sp_col_ind
            )
            ldl_solve_sparse_gpu[DTYPE, NV, NBODY, NM, BATCH, WS_SIZE](
                env, workspace, sp_row_nnz, sp_row_adr, sp_col_ind
            )
        else:
            ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)
            ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
                env, workspace
            )

        # 9d. Read re-solved qacc from workspace and integrate
        comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()
        var qpos_off = qpos_offset[NQ, NV]()

        for i in range(NV):
            var old_qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            var qacc_final = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            state[env, qacc_off + i] = qacc_final
            var new_qvel = old_qvel + qacc_final * dt
            state[env, qvel_off + i] = new_qvel

        # Integrate position: qpos += qvel * dt (quaternion-aware for free joints)
        var model_meta_off_pos = model_metadata_offset[NBODY, NJOINT]()
        var num_joints_pos = Int(
            rebind[Scalar[DTYPE]](
                model[0, model_meta_off_pos + MODEL_META_IDX_NJOINT]
            )
        )

        for j in range(num_joints_pos):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var jnt_qpos_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
            )
            var jnt_dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )

            if jnt_type == JNT_FREE:
                # Position: simple addition
                for d in range(3):
                    var qp = rebind[Scalar[DTYPE]](
                        state[env, qpos_off + jnt_qpos_adr + d]
                    )
                    var qv = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + jnt_dof_adr + d]
                    )
                    state[env, qpos_off + jnt_qpos_adr + d] = qp + qv * dt
                # Quaternion: exponential map integration
                var qx = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr + 3]
                )
                var qy = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr + 4]
                )
                var qz = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr + 5]
                )
                var qw = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr + 6]
                )
                var wx = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + jnt_dof_adr + 3]
                )
                var wy = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + jnt_dof_adr + 4]
                )
                var wz = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + jnt_dof_adr + 5]
                )
                var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
                state[env, qpos_off + jnt_qpos_adr + 3] = result[0]
                state[env, qpos_off + jnt_qpos_adr + 4] = result[1]
                state[env, qpos_off + jnt_qpos_adr + 5] = result[2]
                state[env, qpos_off + jnt_qpos_adr + 6] = result[3]

            elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
                var qp = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr]
                )
                var qv = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + jnt_dof_adr]
                )
                state[env, qpos_off + jnt_qpos_adr] = qp + qv * dt

        # 11. Joint limits now enforced as constraints inside the solver
        # (no post-step clamping needed)

    @staticmethod
    def step_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
        STEP_THREADS: Int = 1,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU with implicit-fast integration.

        Uses the parametrized SOLVER for contact constraint resolution.
        """
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
        comptime MODEL_SIZE = model_size_with_invweight[
            NBODY, NJOINT, NV, NGEOM, NEQUALITY=MAX_EQUALITY
        ]()
        comptime WS_SIZE = integrator_workspace_size[
            NV, NBODY
        ]() + NV * NV + Self.SOLVER.solver_workspace_size[NV, MAX_CONTACTS]()

        comptime THREADS = Self.SOLVER.solver_threads[
            NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ]()
        comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime SOLVER_THREADS_BLOCKS = (THREADS + THREADS - 1) // THREADS
        comptime SOLVER_ENV_TPB = TPB // THREADS
        comptime SOLVER_ENV_BLOCKS = (
            BATCH + SOLVER_ENV_TPB - 1
        ) // SOLVER_ENV_TPB

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf)

        var model = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf)

        var workspace = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ](workspace_buf)

        comptime kernel_wrapper = Self.step_kernel[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
            NM,
            SPARSE,
        ]

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        comptime V_SIZE = _max_one[NV]()

        comptime solver_wrapper = Self.SOLVER.solve_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            V_SIZE,
            BATCH,
            WS_SIZE,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]

        ctx.enqueue_function[solver_wrapper, solver_wrapper](
            state,
            model,
            workspace,
            grid_dim=(SOLVER_ENV_BLOCKS, SOLVER_THREADS_BLOCKS),
            block_dim=(SOLVER_ENV_TPB, THREADS),
        )

        comptime finalize_kernel_wrapper = Self.step_finalize_kernel[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
            NM,
            SPARSE,
        ]

        ctx.enqueue_function[finalize_kernel_wrapper, finalize_kernel_wrapper](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Profiling
    # =========================================================================

    @staticmethod
    def register_gpu_profile_slots(
        mut timer: PerfTimer[True], parent: Int = -1
    ) -> Int:
        """Register 3 profiling slots for ImplicitFast GPU step phases.

        Slots (relative to returned base):
            +0: dynamics  (step_kernel — FK, collision, mass matrix, bias, accel)
            +1: solver    (constraint solve)
            +2: finalize  (integration + normalization)

        Args:
            timer: PerfTimer to add slots to.
            parent: Parent slot index (-1 = top-level).

        Returns:
            Base slot index.
        """
        var base = timer.add_slot("dynamics", parent=parent)
        _ = timer.add_slot("solver", parent=parent)
        _ = timer.add_slot("finalize", parent=parent)
        return base

    @staticmethod
    def step_gpu_profiled[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
        STEP_THREADS: Int = 1,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        mut timer: PerfTimer[True],
        base: Int,
    ) raises:
        """Profiled GPU step — same as step_gpu but with per-phase timing.

        Call register_gpu_profile_slots() first to get the base slot index.
        Inserts GPU sync + timing between each kernel launch.
        """
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
        comptime MODEL_SIZE = model_size_with_invweight[
            NBODY, NJOINT, NV, NGEOM, NEQUALITY=MAX_EQUALITY
        ]()
        comptime WS_SIZE = integrator_workspace_size[
            NV, NBODY
        ]() + NV * NV + Self.SOLVER.solver_workspace_size[NV, MAX_CONTACTS]()

        comptime THREADS = Self.SOLVER.solver_threads[
            NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ]()
        comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime SOLVER_THREADS_BLOCKS = (THREADS + THREADS - 1) // THREADS
        comptime SOLVER_ENV_TPB = TPB // THREADS
        comptime SOLVER_ENV_BLOCKS = (
            BATCH + SOLVER_ENV_TPB - 1
        ) // SOLVER_ENV_TPB

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf)
        var model = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf)
        var workspace = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ](workspace_buf)

        # ---- Phase 0: Dynamics kernel (FK + collision + mass matrix + bias) ----
        timer.sync_and_mark(ctx)

        comptime kernel_wrapper = Self.step_kernel[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
            NM,
            SPARSE,
        ]
        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        timer.sync_and_accumulate(base + 0, ctx)

        # ---- Phase 1: Constraint solver ----
        timer.mark()

        comptime V_SIZE = _max_one[NV]()
        comptime solver_wrapper = Self.SOLVER.solve_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            V_SIZE,
            BATCH,
            WS_SIZE,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
        ctx.enqueue_function[solver_wrapper, solver_wrapper](
            state,
            model,
            workspace,
            grid_dim=(SOLVER_ENV_BLOCKS, SOLVER_THREADS_BLOCKS),
            block_dim=(SOLVER_ENV_TPB, THREADS),
        )

        timer.sync_and_accumulate(base + 1, ctx)

        # ---- Phase 2: Finalize (integration + normalization) ----
        timer.mark()

        comptime finalize_kernel_wrapper = Self.step_finalize_kernel[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
            NM,
            SPARSE,
        ]
        ctx.enqueue_function[finalize_kernel_wrapper, finalize_kernel_wrapper](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        timer.sync_and_accumulate(base + 2, ctx)

    @staticmethod
    def simulate_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
        STEP_THREADS: Int = 1,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        num_steps: Int,
    ) raises:
        """Run simulation for multiple steps on GPU."""
        for _ in range(num_steps):
            Self.step_gpu[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                BATCH,
                NGEOM,
                MAX_EQUALITY,
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
                NM,
                SPARSE,
            ](
                ctx,
                state_buf,
                model_buf,
                workspace_buf,
            )
