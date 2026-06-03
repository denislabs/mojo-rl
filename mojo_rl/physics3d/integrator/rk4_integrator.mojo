"""RK4 (4th-order Runge-Kutta) integrator for physics simulation.

Matches MuJoCo's mj_RungeKutta: runs full forward dynamics (including
constraint solver) at each of the 4 stages. Uses the standard Butcher tableau:

  c = [0, 1/2, 1/2, 1]
  b = [1/6, 1/3, 1/3, 1/6]

Pipeline per step (matching MuJoCo):
1. Save initial (qpos, qvel)
2. Stage 0: full forward dynamics + constraints at (q0, v0) -> A[0], C[0]=v0
3. Stage 1: full dynamics + constraints at (q0+dt/2*C[0], v0+dt/2*A[0]) -> A[1]
4. Stage 2: full dynamics + constraints at (q0+dt/2*C[1], v0+dt/2*A[1]) -> A[2]
5. Stage 3: full dynamics + constraints at (q0+dt*C[2], v0+dt*A[2]) -> A[3]
6. Combine: qacc = (A[0]+2*A[1]+2*A[2]+A[3])/6, v_rk4 = (C[0]+2*C[1]+2*C[2]+C[3])/6
7. Integrate: qvel = v0+qacc*dt, qpos = q0+v_rk4*dt (quaternion-aware)

Both CPU and GPU supported. GPU uses 9 kernel launches per step:
  4 × (stage_kernel + solver) + 1 combine_kernel
"""

from std.math import sqrt, abs
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim, barrier
from layout import LayoutTensor, Layout
from mojo_rl.deep_agents.core.perf_timer import PerfTimer

from ..types import Model, Data, _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
    forward_kinematics_gpu,
    forward_kinematics_gpu_mt,
    compute_body_velocities_gpu,
    compute_body_velocities_gpu_mt,
)
from ..kinematics.quat_math import quat_normalize, quat_integrate
from ..dynamics.mass_matrix import (
    compute_mass_matrix_full,
    compute_mass_matrix_full_gpu,
    compute_mass_matrix_full_gpu_mt,
    ldl_factor,
    ldl_factor_gpu,
    ldl_factor_gpu_mt,
    ldl_solve,
    ldl_solve_gpu,
    ldl_solve_workspace_gpu,
    compute_M_inv_from_ldl,
    compute_M_inv_from_ldl_gpu,
    compute_M_inv_from_ldl_gpu_mt,
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
    compute_bias_forces_rne,
    compute_bias_forces_rne_gpu,
    compute_bias_forces_rne_gpu_mt,
)
from ..dynamics.jacobian import (
    compute_subtree_com,
    compute_cdof,
    compute_subtree_com_gpu,
    compute_cdof_gpu,
    compute_cdof_gpu_mt,
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
from ..constraints.constraint_data import ConstraintData
from ..constraints.constraint_builder import build_constraints, writeback_forces
from ..dynamics.cfrc_ext import compute_cfrc_ext
from ..dynamics.fluid_forces import compute_fluid_forces
from ..traits.integrator import Integrator
from ..traits.solver import ConstraintSolver
from ..kinematics.quat_math import gpu_quat_rotate
from ..gpu.constants import (
    TPB,
    state_size,
    model_size,
    model_size_with_invweight,
    model_metadata_offset,
    model_joint_offset,
    model_body_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    xipos_offset,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_DAMPING,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_BODY_ID,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    subtree_com_offset,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    integrator_workspace_size,
    ws_M_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_cdof_offset,
    rk4_extra_workspace_size,
    ws_rk4_q0_offset,
    ws_rk4_v0_offset,
    ws_rk4_A_offset,
    ws_rk4_C1_offset,
    ws_rk4_C2_offset,
)


@always_inline
def _integrate_pos_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    q0_idx: Int,
    vel_idx: Int,
    dt: Scalar[DTYPE],
):
    """Integrate position on GPU: qpos = q0 + vel * dt (quaternion-aware).

    Reads base qpos from workspace[env, q0_idx+..], velocity from
    workspace[env, vel_idx+..], writes result to state[env, qpos_off+..].
    Handles FREE joint quaternion integration properly.
    """
    var qpos_off = qpos_offset[NQ, NV]()
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )

    for j in range(num_joints):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )

        if jnt_type == JNT_FREE:
            # Position: simple addition
            for d in range(3):
                var q0_d = rebind[Scalar[DTYPE]](
                    workspace[env, q0_idx + qpos_adr + d]
                )
                var v_d = rebind[Scalar[DTYPE]](
                    workspace[env, vel_idx + dof_adr + d]
                )
                state[env, qpos_off + qpos_adr + d] = q0_d + v_d * dt
            # Quaternion: exponential map integration.
            # MuJoCo qpos layout: [tx, ty, tz, qw, qx, qy, qz]
            # Our internal convention: (x, y, z, w)
            var qw = rebind[Scalar[DTYPE]](
                workspace[env, q0_idx + qpos_adr + 3]
            )  # MuJoCo qpos[3] = qw
            var qx = rebind[Scalar[DTYPE]](
                workspace[env, q0_idx + qpos_adr + 4]
            )  # MuJoCo qpos[4] = qx
            var qy = rebind[Scalar[DTYPE]](
                workspace[env, q0_idx + qpos_adr + 5]
            )  # MuJoCo qpos[5] = qy
            var qz = rebind[Scalar[DTYPE]](
                workspace[env, q0_idx + qpos_adr + 6]
            )  # MuJoCo qpos[6] = qz
            var wx = rebind[Scalar[DTYPE]](
                workspace[env, vel_idx + dof_adr + 3]
            )
            var wy = rebind[Scalar[DTYPE]](
                workspace[env, vel_idx + dof_adr + 4]
            )
            var wz = rebind[Scalar[DTYPE]](
                workspace[env, vel_idx + dof_adr + 5]
            )
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(
                result[0], result[1], result[2], result[3]
            )
            # Write back in MuJoCo qpos layout: [qw, qx, qy, qz]
            state[env, qpos_off + qpos_adr + 3] = norm[3]  # qw
            state[env, qpos_off + qpos_adr + 4] = norm[0]  # qx
            state[env, qpos_off + qpos_adr + 5] = norm[1]  # qy
            state[env, qpos_off + qpos_adr + 6] = norm[2]  # qz

        elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
            var q0_val = rebind[Scalar[DTYPE]](
                workspace[env, q0_idx + qpos_adr]
            )
            var v_val = rebind[Scalar[DTYPE]](workspace[env, vel_idx + dof_adr])
            state[env, qpos_off + qpos_adr] = q0_val + v_val * dt


def _forward_dynamics[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int,
    V_SIZE: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
    CRB_SIZE: Int,
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
    mut qacc_out: List[Scalar[DTYPE]],
    mut cdof_out: List[Scalar[DTYPE]],
    mut M_inv_out: List[Scalar[DTYPE]],
    mut M_out: List[Scalar[DTYPE]],
):
    """Compute unconstrained acceleration from current (qpos, qvel) in data.

    Runs the full dynamics pipeline:
    FK -> body velocities -> collision -> cdof -> CRBA -> M -> LDL -> bias -> passive -> solve.

    Returns qacc, cdof, M_inv, and M (with armature, for constraint solver M_hat).
    """
    # 1. Forward kinematics + body velocities
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    # 2. Collision detection
    detect_contacts_auto[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )

    # 3. Compute subtree CoM and store in data (MuJoCo mj_comPos)
    var stcom_tmp = List[Scalar[DTYPE]](capacity=NBODY * 3)
    for _ in range(NBODY * 3):
        stcom_tmp.append(Scalar[DTYPE](0))
    compute_subtree_com[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        model, data, stcom_tmp
    )
    for sc_i in range(NBODY * 3):
        data.subtree_com[sc_i] = stcom_tmp[sc_i]
    data.has_subtree_com = True

    # 4. Compute cdof (with subtree_com reference)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        model, data, cdof_out, data.subtree_com
    )

    # 5. Composite rigid body inertia
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        model, data, crb
    )

    comptime NM_SAFE = _ensure_positive[NM]()

    # 5. Full mass matrix
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
        ](model, data, cdof_out, crb, sM)
    else:
        for i in range(M_SIZE):
            M_out[i] = Scalar[DTYPE](0)
        compute_mass_matrix_full[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
        ](model, data, cdof_out, crb, M_out)

    # 5b. Armature only (no implicit damping for RK4 — damping is explicit)
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
                    M_out[(dof_adr + d) * NV + (dof_adr + d)] = (
                        M_out[(dof_adr + d) * NV + (dof_adr + d)] + arm
                    )
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    M_out[(dof_adr + d) * NV + (dof_adr + d)] = (
                        M_out[(dof_adr + d) * NV + (dof_adr + d)] + arm
                    )
            else:
                M_out[dof_adr * NV + dof_adr] = (
                    M_out[dof_adr * NV + dof_adr] + arm
                )

    # 5c. Expand sparse to dense for M_out (must be before ldl_factor_sparse mutates sM)
    comptime if SPARSE:
        sparse_to_dense[DTYPE, NV, NM](sM, M_out)

    # 6. LDL factorize
    var L = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[DTYPE](0))
    var D = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D.append(Scalar[DTYPE](0))

    comptime if SPARSE:
        ldl_factor_sparse(sM)
    else:
        ldl_factor[
            DTYPE,
            NV,
        ](M_out, L, D)

    # 7. Bias forces
    var bias = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        bias.append(Scalar[DTYPE](0))
    compute_bias_forces_rne[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
    ](model, data, cdof_out, bias)

    # 8. Net force = external - bias - passive
    var f_net = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        f_net.append(Scalar[DTYPE](0))
    for i in range(NV):
        f_net[i] = data.qfrc[i] - bias[i]

    # Damping: f -= damping * qvel (fully explicit in RK4)
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var damp = joint.damping
        if damp > Scalar[DTYPE](0):
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr + d] = (
                        f_net[dof_adr + d] - damp * data.qvel[dof_adr + d]
                    )
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr + d] = (
                        f_net[dof_adr + d] - damp * data.qvel[dof_adr + d]
                    )
            else:
                f_net[dof_adr] = f_net[dof_adr] - damp * data.qvel[dof_adr]

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

    # 8c. Fluid forces: viscous + pressure drag (disabled when density=viscosity=0)
    compute_fluid_forces[
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
    ](model, data, cdof_out, f_net, stcom_tmp)

    # 9. qacc = M^-1 * f_net
    for i in range(NV):
        qacc_out[i] = Scalar[DTYPE](0)

    comptime if SPARSE:
        ldl_solve_sparse[DTYPE, NV, NM](sM, f_net, qacc_out)
    else:
        ldl_solve[DTYPE, NV](L, D, f_net, qacc_out)

    # 10. M_inv for constraint solver
    for i in range(M_SIZE):
        M_inv_out[i] = Scalar[DTYPE](0)

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
                e_col[k] = Scalar[DTYPE](1) if k == col else Scalar[DTYPE](0)
            ldl_solve_sparse[DTYPE, NV, NM](sM, e_col, col_result)
            for row in range(NV):
                M_inv_out[row * NV + col] = col_result[row]
    else:
        compute_M_inv_from_ldl[DTYPE, NV](L, D, M_inv_out)


def _solve_constraints[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int,
    V_SIZE: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
    CONE_TYPE: Int,
    MAX_TENDON: Int,
    SOLVER: ConstraintSolver,
    NSITE: Int = 0,
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
    cdof: List[Scalar[DTYPE]],
    M_inv: List[Scalar[DTYPE]],
    M: List[Scalar[DTYPE]],
    mut qacc: List[Scalar[DTYPE]],
    dt: Scalar[DTYPE],
    is_last_stage: Bool,
):
    """Build and solve constraints, modifying qacc in place.

    Separate function so ConstraintData is allocated/freed in its own stack frame.
    This prevents stack overflow from accumulating 4 ConstraintData instances.
    """
    comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT + 6 * MAX_EQUALITY + MAX_TENDON
    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints[CONE_TYPE=CONE_TYPE, MAX_TENDON=MAX_TENDON](
        model, data, cdof, M_inv, dt, constraints
    )

    # Fill M_hat and qfrc_smooth for primal solvers
    for i in range(NV * NV):
        constraints.M_hat[i] = M[i]
    for i in range(NV):
        var f_i = Scalar[DTYPE](0)
        for j in range(NV):
            f_i = f_i + M[i * NV + j] * qacc[j]
        constraints.qfrc_smooth[i] = f_i

    SOLVER.solve[CONE_TYPE=CONE_TYPE](model, data, M_inv, constraints, qacc, dt)

    # Only write back forces on the last stage (for data.qfrc_constraint)
    if is_last_stage:
        writeback_forces[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_ROWS,
        ](constraints, data)


def _integrate_pos[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
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
    qpos_base: List[Scalar[DTYPE]],
    vel: List[Scalar[DTYPE]],
    dt: Scalar[DTYPE],
    mut qpos_out: List[Scalar[DTYPE]],
):
    """Integrate position: qpos_out = qpos_base + vel * dt.

    Uses quaternion exponential map for FREE and BALL joints,
    simple addition for HINGE and SLIDE joints.
    """
    # Start with a copy of base
    for i in range(NQ):
        qpos_out[i] = qpos_base[i]

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var qpos_adr = joint.qpos_adr
        var dof_adr = joint.dof_adr

        if joint.jnt_type == JNT_FREE:
            # Linear position: simple addition
            for d in range(3):
                qpos_out[qpos_adr + d] = (
                    qpos_base[qpos_adr + d] + vel[dof_adr + d] * dt
                )
            # Quaternion: exponential map integration.
            # MuJoCo qpos layout: [tx, ty, tz, qw, qx, qy, qz]
            # Our internal convention: (x, y, z, w)
            var qw = qpos_base[qpos_adr + 3]  # MuJoCo qpos[3] = qw
            var qx = qpos_base[qpos_adr + 4]  # MuJoCo qpos[4] = qx
            var qy = qpos_base[qpos_adr + 5]  # MuJoCo qpos[5] = qy
            var qz = qpos_base[qpos_adr + 6]  # MuJoCo qpos[6] = qz
            var wx = vel[dof_adr + 3]
            var wy = vel[dof_adr + 4]
            var wz = vel[dof_adr + 5]
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(
                result[0], result[1], result[2], result[3]
            )
            # Write back in MuJoCo qpos layout: [qw, qx, qy, qz]
            qpos_out[qpos_adr + 3] = norm[3]  # qw
            qpos_out[qpos_adr + 4] = norm[0]  # qx
            qpos_out[qpos_adr + 5] = norm[1]  # qy
            qpos_out[qpos_adr + 6] = norm[2]  # qz

        elif joint.jnt_type == JNT_BALL:
            # Quaternion: exponential map integration
            var qx = qpos_base[qpos_adr]
            var qy = qpos_base[qpos_adr + 1]
            var qz = qpos_base[qpos_adr + 2]
            var qw = qpos_base[qpos_adr + 3]
            var wx = vel[dof_adr]
            var wy = vel[dof_adr + 1]
            var wz = vel[dof_adr + 2]
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(
                result[0], result[1], result[2], result[3]
            )
            qpos_out[qpos_adr] = norm[0]
            qpos_out[qpos_adr + 1] = norm[1]
            qpos_out[qpos_adr + 2] = norm[2]
            qpos_out[qpos_adr + 3] = norm[3]

        else:
            # HINGE / SLIDE: simple addition
            qpos_out[qpos_adr] = qpos_base[qpos_adr] + vel[dof_adr] * dt


# 5a: when True, RK4Integrator.step_gpu launches the Newton solver
# one-environment-per-block (grid=(BATCH,1), block=(1, THREADS)) instead of
# packing many envs per block. Bit-identical to the packed launch (same kernel,
# same per-env math) but isolates per-env Newton iteration-count divergence
# across independently-scheduled blocks. Module-level (not a step_gpu param) to
# preserve the Integrator trait signature; mirrors USE_NEWTON_SIMD. See
# docs/PHYSICS3D_BLOCKED_SOLVER.md.
comptime RK4_BLOCKED_SOLVER: Bool = True


# When True (and STEP_THREADS>1, dense, solver needs M_inv), the dense M^-1
# computation in rk4_stage_kernel is distributed across the STEP_THREADS threads
# that would otherwise sit idle after the mass matrix (each thread solves its
# share of the NV independent M^-1 columns). Bit-identical to the serial path.
# See docs/PHYSICS3D_BLOCKED_SOLVER.md.
comptime RK4_PARALLEL_MINV: Bool = True


# Sub-gate of RK4_PARALLEL_MINV: when True, the LDL factorization of the mass
# matrix is also cooperative (each column's off-diagonal entries distributed
# across the idle threads, block-wide barrier per column) instead of running on
# tid 0. Bit-identical. Set False to keep LDL serial while M^-1 stays parallel.
comptime RK4_PARALLEL_LDL: Bool = True


# When True (and STEP_THREADS>1), forward kinematics in rk4_stage_kernel is
# computed level-parallel instead of redundantly on every thread: bodies are
# processed by kinematic-tree depth, and bodies at the same depth (independent —
# none is another's parent) are striped across the STEP_THREADS threads. Tree
# levels are derived in-kernel from body_parent (no model/parser change).
# Bit-identical to the serial tree walk (same fk_body_gpu per body). Lever 1
# (branch/level-parallel forward dynamics). NVIDIA Humanoid measured 2.6×
# (64µs→25µs standalone, profile_rk4_phases_humanoid.mojo 2026-06-03) → ON by
# default. See docs/PHYSICS3D_BLOCKED_SOLVER.md.
comptime RK4_PARALLEL_FK: Bool = True


# When True (and STEP_THREADS>1), body velocities are computed level-parallel
# (same recipe as FK: vel_body_gpu per body, distributed by tree depth). The
# serial root→leaf walk is the oracle. Within float32 tolerance vs serial
# (~1e-9 — well under the 1e-6 parallel-path bar; FK happened to be byte-exact,
# velocities is within-tol). Default ON (velocities is the same structure as FK,
# expect a similar win). See docs/PHYSICS3D_BLOCKED_SOLVER.md.
comptime RK4_PARALLEL_VEL: Bool = True


# When True (and STEP_THREADS>1), cdof (spatial motion axes) is computed
# flat-parallel: bodies are independent (each writes its own DOFs' cdof from FK
# state + subtree_com), so threads just stripe over bodies — no level ordering,
# 2 barriers (after zero-init, after the body sweep). Within float32 tolerance
# of the serial walk. Default ON. See docs/PHYSICS3D_BLOCKED_SOLVER.md.
comptime RK4_PARALLEL_CDOF: Bool = True


# When True (and the parallel M-inv tail runs, i.e. STEP_THREADS>1 + dense), RNE
# bias forces are computed cooperatively: cinert flat + forward cvel/cacc
# level-parallel (rne_fwd_body) + cfrc flat + backward tid0 (cheap ~NBODY adds) +
# qfrc flat. Borrows mujoco_warp's 4-pass decomposition but skips its per-level
# atomic backward (our one-block-per-env model makes tid0-serial backward free).
# Within float32 tolerance of the serial walk. Default ON. Requires RK4_PARALLEL_MINV
# (RNE-mt runs before the tid0 drop-out in that tail). See docs/PHYSICS3D_BLOCKED_SOLVER.md.
comptime RK4_PARALLEL_RNE: Bool = True


struct RK4Integrator[SOLVER: ConstraintSolver](Integrator):
    """4th-order Runge-Kutta integrator with configurable constraint solver.

    Matches MuJoCo's mj_RungeKutta: runs full forward dynamics including
    constraint solver at each of the 4 RK4 stages. The constrained
    accelerations from all stages are combined with RK4 weights.

    GPU support: 9 kernel launches per step (4 stage + 4 solver + 1 combine).
    Workspace includes extra RK4 storage (q0, v0, A[0..3], C1, C2).

    Usage:
        alias RK4PGS = RK4Integrator[PGSSolver]
        RK4PGS.step(model, data)        # CPU
        RK4PGS.step_gpu[...](ctx, ...)  # GPU
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
        """Execute one RK4 simulation step (MuJoCo-compatible).

        Runs full forward dynamics + constraint solver at each of 4 stages,
        then combines with RK4 weights. Matches MuJoCo's mj_RungeKutta.
        """
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"
        var dt = model.timestep
        comptime Q_SIZE = _max_one[NQ]()
        comptime V_SIZE = _max_one[NV]()
        comptime M_SIZE = _max_one[NV * NV]()
        comptime CDOF_SIZE = _max_one[NV * 6]()
        comptime CRB_SIZE = _max_one[NBODY * 10]()

        # Save initial state
        var q0 = List[Scalar[DTYPE]](capacity=Q_SIZE)
        for _ in range(Q_SIZE):
            q0.append(Scalar[DTYPE](0))
        var v0 = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            v0.append(Scalar[DTYPE](0))
        for i in range(NQ):
            q0[i] = data.qpos[i]
        for i in range(NV):
            v0[i] = data.qvel[i]

        # RK4 stage results: A[i] = constrained qacc, C[i] = velocity at stage
        var a0 = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            a0.append(Scalar[DTYPE](0))
        var a1 = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            a1.append(Scalar[DTYPE](0))
        var a2 = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            a2.append(Scalar[DTYPE](0))
        var a3 = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            a3.append(Scalar[DTYPE](0))

        # Workspace reused across stages
        var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
        for _ in range(CDOF_SIZE):
            cdof.append(Scalar[DTYPE](0))
        var M_inv = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            M_inv.append(Scalar[DTYPE](0))
        var M = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            M.append(Scalar[DTYPE](0))

        var half_dt = dt * Scalar[DTYPE](0.5)
        var q_stage = List[Scalar[DTYPE]](capacity=Q_SIZE)
        for _ in range(Q_SIZE):
            q_stage.append(Scalar[DTYPE](0))

        # =================================================================
        # Stage 0: evaluate at (q0, v0) — full pipeline
        # =================================================================
        # data already has (q0, v0)
        _forward_dynamics[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            CRB_SIZE,
            NM=NM,
            SPARSE=SPARSE,
        ](model, data, a0, cdof, M_inv, M)

        _solve_constraints[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            CONE_TYPE,
            MAX_TENDON,
            Self.SOLVER,
        ](model, data, cdof, M_inv, M, a0, dt, False)
        # a0 is now CONSTRAINED qacc
        # C[0] = v0 (saved above)

        # =================================================================
        # Stage 1: evaluate at (q0 + dt/2*C[0], v0 + dt/2*A[0])
        # =================================================================
        # C[0] = v0
        for i in range(NV):
            data.qvel[i] = v0[i] + half_dt * a0[i]
        _integrate_pos[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
        ](model, q0, v0, half_dt, q_stage)
        for i in range(NQ):
            data.qpos[i] = q_stage[i]

        _forward_dynamics[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            CRB_SIZE,
            NM=NM,
            SPARSE=SPARSE,
        ](model, data, a1, cdof, M_inv, M)

        _solve_constraints[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            CONE_TYPE,
            MAX_TENDON,
            Self.SOLVER,
        ](model, data, cdof, M_inv, M, a1, dt, False)
        # a1 is now CONSTRAINED qacc
        # C[1] = v0 + dt/2*A[0] = data.qvel (set above)

        # =================================================================
        # Stage 2: evaluate at (q0 + dt/2*C[1], v0 + dt/2*A[1])
        # =================================================================
        # C[1] = v0 + dt/2*a0
        var c1 = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            c1.append(Scalar[DTYPE](0))
        for i in range(NV):
            c1[i] = v0[i] + half_dt * a0[i]

        for i in range(NV):
            data.qvel[i] = v0[i] + half_dt * a1[i]
        _integrate_pos[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
        ](model, q0, c1, half_dt, q_stage)
        for i in range(NQ):
            data.qpos[i] = q_stage[i]

        _forward_dynamics[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            CRB_SIZE,
            NM=NM,
            SPARSE=SPARSE,
        ](model, data, a2, cdof, M_inv, M)

        _solve_constraints[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            CONE_TYPE,
            MAX_TENDON,
            Self.SOLVER,
        ](model, data, cdof, M_inv, M, a2, dt, False)
        # a2 is now CONSTRAINED qacc
        # C[2] = v0 + dt/2*A[1] = data.qvel (set above)

        # =================================================================
        # Stage 3: evaluate at (q0 + dt*C[2], v0 + dt*A[2])
        # =================================================================
        # C[2] = v0 + dt/2*a1
        var c2 = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            c2.append(Scalar[DTYPE](0))
        for i in range(NV):
            c2[i] = v0[i] + half_dt * a1[i]

        for i in range(NV):
            data.qvel[i] = v0[i] + dt * a2[i]
        _integrate_pos[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
        ](model, q0, c2, dt, q_stage)
        for i in range(NQ):
            data.qpos[i] = q_stage[i]

        _forward_dynamics[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            CRB_SIZE,
            NM=NM,
            SPARSE=SPARSE,
        ](model, data, a3, cdof, M_inv, M)

        _solve_constraints[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            CONE_TYPE,
            MAX_TENDON,
            Self.SOLVER,
        ](model, data, cdof, M_inv, M, a3, dt, True)
        # a3 is now CONSTRAINED qacc
        # C[3] = v0 + dt*A[2] = data.qvel (set above)

        # =================================================================
        # Combine with RK4 weights: b = [1/6, 1/3, 1/3, 1/6]
        # =================================================================
        comptime ONE_SIXTH: Scalar[DTYPE] = 1.0 / 6.0
        comptime ONE_THIRD: Scalar[DTYPE] = 1.0 / 3.0

        # qacc_combined = (A[0] + 2*A[1] + 2*A[2] + A[3]) / 6
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qacc[i] = (
                ONE_SIXTH * a0[i]
                + ONE_THIRD * a1[i]
                + ONE_THIRD * a2[i]
                + ONE_SIXTH * a3[i]
            )

        # v_combined = (C[0] + 2*C[1] + 2*C[2] + C[3]) / 6
        # C[0] = v0, C[1] = v0+dt/2*a0, C[2] = v0+dt/2*a1, C[3] = v0+dt*a2
        var c3 = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            c3.append(Scalar[DTYPE](0))
        for i in range(NV):
            c3[i] = v0[i] + dt * a2[i]

        var v_combined = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            v_combined.append(Scalar[DTYPE](0))
        for i in range(NV):
            v_combined[i] = (
                ONE_SIXTH * v0[i]
                + ONE_THIRD * c1[i]
                + ONE_THIRD * c2[i]
                + ONE_SIXTH * c3[i]
            )

        # =================================================================
        # Final integration (MuJoCo's mj_advance)
        # =================================================================
        # qvel = v0 + qacc * dt
        for i in range(NV):
            data.qacc[i] = qacc[i]
            data.qvel[i] = v0[i] + qacc[i] * dt

        # qpos = q0 + v_combined * dt
        _integrate_pos(model, q0, v_combined, dt, q_stage)
        for i in range(NQ):
            data.qpos[i] = q_stage[i]

        # Normalize quaternions
        normalize_qpos_quaternions(model, data)

        # Compute cfrc_ext: contact forces per body in subtree CoM frame
        # Uses forces from the last stage (stage 3) written back above.
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
    def rk4_stage_kernel[
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
        NGEOM: Int,
        SOLVER_WS_SIZE: Int,
        STAGE: Int,
        NM: Int = 0,
        SPARSE: Bool = False,
        STEP_THREADS: Int = 1,
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
        """RK4 stage kernel: sets up intermediate state, runs forward dynamics.

        STAGE 0: save q0/v0 to workspace, run dynamics on original state.
        STAGE 1: save A[0] from qacc_constrained, set state = (q0+dt/2*v0, v0+dt/2*A[0]).
        STAGE 2: save A[1], set state = (q0+dt/2*C[1], v0+dt/2*A[1]).
        STAGE 3: save A[2], set state = (q0+dt*C[2], v0+dt*A[2]).

        When STEP_THREADS > 1, uses 2D blocks (envs, STEP_THREADS) to
        parallelize mass matrix computation across threads.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var tid: Int
        comptime if STEP_THREADS > 1:
            tid = Int(thread_idx.y)
        else:
            tid = 0
        var valid_env = env < BATCH
        comptime if STEP_THREADS <= 1:
            if not valid_env:
                return

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

        # RK4 workspace offsets
        comptime q0_idx = ws_rk4_q0_offset[NV, NBODY](SOLVER_WS_SIZE)
        comptime v0_idx = ws_rk4_v0_offset[NV, NBODY, NQ](SOLVER_WS_SIZE)
        comptime A0_idx = ws_rk4_A_offset[NV, NBODY, NQ](SOLVER_WS_SIZE, 0)
        comptime A1_idx = ws_rk4_A_offset[NV, NBODY, NQ](SOLVER_WS_SIZE, 1)
        comptime A2_idx = ws_rk4_A_offset[NV, NBODY, NQ](SOLVER_WS_SIZE, 2)
        comptime c1_idx = ws_rk4_C1_offset[NV, NBODY, NQ](SOLVER_WS_SIZE)
        comptime c2_idx = ws_rk4_C2_offset[NV, NBODY, NQ](SOLVER_WS_SIZE)

        var qpos_off = qpos_offset[NQ, NV]()
        var qvel_off = qvel_offset[NQ, NV]()
        var qacc_off = qacc_offset[NQ, NV]()
        var qfrc_off = qfrc_offset[NQ, NV]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )
        var half_dt = dt * Scalar[DTYPE](0.5)

        # ---- Pre-stage: save A[prev] and set intermediate state ----
        # Note: when STEP_THREADS > 1, all threads run the serial phases
        # redundantly (idempotent writes to same env). Only mass matrix
        # is actually distributed across threads via compute_mass_matrix_full_gpu_mt.
        # Invalid envs (env >= BATCH) must still reach barriers but skip all
        # state/workspace writes to avoid out-of-bounds GPU memory corruption.

        if valid_env:
            comptime if STAGE == 0:
                # Save initial state to workspace
                for i in range(NQ):
                    workspace[env, q0_idx + i] = state[env, qpos_off + i]
                for i in range(NV):
                    workspace[env, v0_idx + i] = state[env, qvel_off + i]
            elif STAGE == 1:
                # Save A[0] from qacc_constrained
                for i in range(NV):
                    workspace[env, A0_idx + i] = workspace[
                        env, qacc_constrained_idx + i
                    ]
                # Set intermediate state: qpos = q0 + dt/2 * v0 (C[0] = v0)
                # qvel = v0 + dt/2 * A[0]
                _integrate_pos_gpu[
                    DTYPE,
                    NQ,
                    NV,
                    NBODY,
                    NJOINT,
                    STATE_SIZE,
                    MODEL_SIZE,
                    BATCH,
                    WS_SIZE,
                ](env, state, model, workspace, q0_idx, v0_idx, half_dt)
                for i in range(NV):
                    var v0_i = rebind[Scalar[DTYPE]](workspace[env, v0_idx + i])
                    var a0_i = rebind[Scalar[DTYPE]](workspace[env, A0_idx + i])
                    state[env, qvel_off + i] = v0_i + half_dt * a0_i
            elif STAGE == 2:
                # Save A[1] from qacc_constrained
                for i in range(NV):
                    workspace[env, A1_idx + i] = workspace[
                        env, qacc_constrained_idx + i
                    ]
                # C[1] = v0 + dt/2 * A[0] — save to workspace for combine kernel
                for i in range(NV):
                    var v0_i = rebind[Scalar[DTYPE]](workspace[env, v0_idx + i])
                    var a0_i = rebind[Scalar[DTYPE]](workspace[env, A0_idx + i])
                    workspace[env, c1_idx + i] = v0_i + half_dt * a0_i
                # Set intermediate state: qpos = q0 + dt/2 * C[1]
                # qvel = v0 + dt/2 * A[1]
                _integrate_pos_gpu[
                    DTYPE,
                    NQ,
                    NV,
                    NBODY,
                    NJOINT,
                    STATE_SIZE,
                    MODEL_SIZE,
                    BATCH,
                    WS_SIZE,
                ](env, state, model, workspace, q0_idx, c1_idx, half_dt)
                for i in range(NV):
                    var v0_i = rebind[Scalar[DTYPE]](workspace[env, v0_idx + i])
                    var a1_i = rebind[Scalar[DTYPE]](workspace[env, A1_idx + i])
                    state[env, qvel_off + i] = v0_i + half_dt * a1_i
            elif STAGE == 3:
                # Save A[2] from qacc_constrained
                for i in range(NV):
                    workspace[env, A2_idx + i] = workspace[
                        env, qacc_constrained_idx + i
                    ]
                # C[2] = v0 + dt/2 * A[1] — save to workspace for combine kernel
                for i in range(NV):
                    var v0_i = rebind[Scalar[DTYPE]](workspace[env, v0_idx + i])
                    var a1_i = rebind[Scalar[DTYPE]](workspace[env, A1_idx + i])
                    workspace[env, c2_idx + i] = v0_i + half_dt * a1_i
                # Set intermediate state: qpos = q0 + dt * C[2]
                # qvel = v0 + dt * A[2]
                _integrate_pos_gpu[
                    DTYPE,
                    NQ,
                    NV,
                    NBODY,
                    NJOINT,
                    STATE_SIZE,
                    MODEL_SIZE,
                    BATCH,
                    WS_SIZE,
                ](env, state, model, workspace, q0_idx, c2_idx, dt)
                for i in range(NV):
                    var v0_i = rebind[Scalar[DTYPE]](workspace[env, v0_idx + i])
                    var a2_i = rebind[Scalar[DTYPE]](workspace[env, A2_idx + i])
                    state[env, qvel_off + i] = v0_i + dt * a2_i

        # ---- Forward dynamics pipeline (same as EulerIntegrator.step_kernel) ----

        # 1. Forward kinematics. When RK4_PARALLEL_FK (and STEP_THREADS>1) the
        # cooperative level-parallel variant is called UNCONDITIONALLY so every
        # thread (incl. invalid-env / other packed envs) reaches its internal
        # barriers; per-body writes are guarded by valid_env inside it. Otherwise
        # the serial walk runs redundantly per valid-env thread (as before).
        comptime USE_PAR_FK = RK4_PARALLEL_FK and (STEP_THREADS > 1)
        comptime if USE_PAR_FK:
            forward_kinematics_gpu_mt[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
            ](env, tid, STEP_THREADS, valid_env, state, model)
        else:
            if valid_env:
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

        # 2. Body velocities. When RK4_PARALLEL_VEL (and STEP_THREADS>1) the
        # cooperative level-parallel variant is called UNCONDITIONALLY (for its
        # barriers); per-body writes guarded by valid_env. Otherwise the serial
        # walk runs redundantly per valid-env thread. FK above already published
        # xpos/xquat (mt FK ends with a barrier; serial FK is per-thread redundant).
        comptime USE_PAR_VEL = RK4_PARALLEL_VEL and (STEP_THREADS > 1)
        comptime if USE_PAR_VEL:
            compute_body_velocities_gpu_mt[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
            ](env, tid, STEP_THREADS, valid_env, state, model)
        else:
            if valid_env:
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

        if valid_env:
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
            ](env, state, model)

            # 3a. Compute subtree_com
            compute_subtree_com_gpu[
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

        # 4. Compute cdof. Flat-parallel (mt) called UNCONDITIONALLY for its
        # barriers when RK4_PARALLEL_CDOF; serial in valid_env otherwise. Reads
        # FK state + subtree_com, both written redundantly per-thread just above
        # (each thread has its own copy) → no extra barrier needed before this.
        comptime USE_PAR_CDOF = RK4_PARALLEL_CDOF and (STEP_THREADS > 1)
        comptime if USE_PAR_CDOF:
            compute_cdof_gpu_mt[
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
            ](env, tid, STEP_THREADS, valid_env, state, model, workspace)
        else:
            if valid_env:
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

        if valid_env:
            # 5. Composite rigid body inertia
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

        # 6. Full mass matrix (multi-threaded when STEP_THREADS > 1)
        comptime if STEP_THREADS > 1:
            barrier()
        comptime if SPARSE:
            if tid == 0 and valid_env:
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
                ](
                    env,
                    state,
                    model,
                    workspace,
                    sp_row_nnz,
                    sp_row_adr,
                    sp_col_ind,
                )
        else:
            comptime if STEP_THREADS > 1:
                if valid_env:
                    compute_mass_matrix_full_gpu_mt[
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
                    ](env, tid, STEP_THREADS, state, model, workspace)
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
        comptime if STEP_THREADS > 1:
            barrier()

        # After the mass matrix the remaining work (armature, LDL, M_inv, RNE,
        # accel) is serial per env. M_inv, however, is NV independent column
        # solves — when supported we distribute those across the STEP_THREADS
        # threads that would otherwise idle after `if tid != 0: return`.
        comptime USE_PAR_MINV = (
            RK4_PARALLEL_MINV
            and (STEP_THREADS > 1)
            and (not SPARSE)
            and Self.SOLVER.NEEDS_M_INV
        )

        # 6b. Armature only (no implicit damping for RK4) — tid 0 modifies the
        # mass-matrix diagonal before LDL.
        if valid_env and tid == 0:
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

        comptime if USE_PAR_MINV:
            # Make tid 0's armature writes to M visible before the cooperative
            # LDL/M_inv read it.
            barrier()
            # LDL: cooperative (per-column distributed, internal barriers) or
            # serial on tid 0, then all threads solve their share of M^-1 cols.
            comptime if RK4_PARALLEL_LDL:
                ldl_factor_gpu_mt[DTYPE, NV, NBODY, BATCH, WS_SIZE](
                    env, tid, STEP_THREADS, valid_env, workspace
                )
            else:
                if valid_env and tid == 0:
                    ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
                        env, workspace
                    )
                barrier()
            if valid_env:
                compute_M_inv_from_ldl_gpu_mt[
                    DTYPE, NV, NBODY, BATCH, WS_SIZE
                ](env, tid, STEP_THREADS, workspace)
            barrier()
            # RNE bias forces — cooperative across all threads BEFORE the tid 0
            # drop-out (it has internal barriers). When off, the serial RNE runs
            # on tid 0 below (gated by USE_PAR_MINV+RK4_PARALLEL_RNE).
            comptime if RK4_PARALLEL_RNE:
                compute_bias_forces_rne_gpu_mt[
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
                ](env, tid, STEP_THREADS, valid_env, state, model, workspace)
            # Only tid 0 runs the remaining serial work (f_net + accel).
            if tid != 0 or not valid_env:
                return
        else:
            # Serial tail on tid 0 (existing behavior).
            comptime if STEP_THREADS > 1:
                if tid != 0:
                    return
            if not valid_env:
                return

            # 7. LDL factorize M, compute M_inv
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
                    compute_M_inv_from_ldl_gpu[
                        DTYPE, NV, NBODY, BATCH, WS_SIZE
                    ](env, workspace)

        # 8. Bias forces (serial, tid 0) — skipped when the cooperative RNE-mt
        # already ran in the parallel tail (RK4_PARALLEL_RNE + USE_PAR_MINV).
        comptime if not (RK4_PARALLEL_RNE and USE_PAR_MINV):
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

        # 9. f_net = qfrc - bias
        for i in range(NV):
            var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
            var bias_val = rebind[Scalar[DTYPE]](workspace[env, bias_idx + i])
            workspace[env, fnet_idx + i] = qfrc - bias_val

        # 9b. Passive forces: damping + stiffness + frictionloss (explicit for RK4)
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
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = cur - damp * v
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = cur - damp * v
                else:
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    workspace[env, fnet_idx + dof_adr] = cur - damp * v

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
            var qpos_adr_j = Int(
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
            if stiff > Scalar[DTYPE](0):
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var qpos_d = rebind[Scalar[DTYPE]](
                            state[env, qpos_off_stiff + qpos_adr_j + d]
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
                            state[env, qpos_off_stiff + qpos_adr_j + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = cur - stiff * (
                            qpos_d - sref
                        )
                else:
                    var qpos_d = rebind[Scalar[DTYPE]](
                        state[env, qpos_off_stiff + qpos_adr_j]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    workspace[env, fnet_idx + dof_adr] = cur - stiff * (
                        qpos_d - sref
                    )
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

        # 9c. Fluid forces: inertia-box viscous + pressure drag (GPU)
        # Enabled when density > 0 or viscosity > 0 (stored in model metadata).
        var model_meta_off_fl = model_metadata_offset[NBODY, NJOINT]()
        var rho_fl = rebind[Scalar[DTYPE]](
            model[0, model_meta_off_fl + MODEL_META_IDX_DENSITY]
        )
        var mu_fl = rebind[Scalar[DTYPE]](
            model[0, model_meta_off_fl + MODEL_META_IDX_VISCOSITY]
        )
        if rho_fl > Scalar[DTYPE](0) or mu_fl > Scalar[DTYPE](0):
            comptime PI_FL: Scalar[DTYPE] = 3.14159265358979323846
            comptime xquat_off_fl = xquat_offset[NQ, NV, NBODY]()
            comptime xvel_off_fl = xvel_offset[NQ, NV, NBODY]()
            comptime xangvel_off_fl = xangvel_offset[NQ, NV, NBODY]()
            comptime xipos_off_fl = xipos_offset[NQ, NV, NBODY]()
            comptime cdof_off_fl = ws_cdof_offset()

            for b in range(1, NBODY):
                var body_off_b = model_body_offset(b)
                var mass_b = rebind[Scalar[DTYPE]](
                    model[0, body_off_b + BODY_IDX_MASS]
                )
                if mass_b <= Scalar[DTYPE](1e-10):
                    continue

                # Box from diagonal inertia
                var Ixx = rebind[Scalar[DTYPE]](
                    model[0, body_off_b + BODY_IDX_IXX]
                )
                var Iyy = rebind[Scalar[DTYPE]](
                    model[0, body_off_b + BODY_IDX_IYY]
                )
                var Izz = rebind[Scalar[DTYPE]](
                    model[0, body_off_b + BODY_IDX_IZZ]
                )
                var bx2 = Scalar[DTYPE](6) * (Iyy + Izz - Ixx) / mass_b
                var by2 = Scalar[DTYPE](6) * (Ixx + Izz - Iyy) / mass_b
                var bz2 = Scalar[DTYPE](6) * (Ixx + Iyy - Izz) / mass_b
                var bx = sqrt(max(bx2, Scalar[DTYPE](0)))
                var by = sqrt(max(by2, Scalar[DTYPE](0)))
                var bz = sqrt(max(bz2, Scalar[DTYPE](0)))

                # World-frame body velocity (at body origin)
                var vx_w = rebind[Scalar[DTYPE]](
                    state[env, xvel_off_fl + b * 3 + 0]
                )
                var vy_w = rebind[Scalar[DTYPE]](
                    state[env, xvel_off_fl + b * 3 + 1]
                )
                var vz_w = rebind[Scalar[DTYPE]](
                    state[env, xvel_off_fl + b * 3 + 2]
                )
                var wx_w = rebind[Scalar[DTYPE]](
                    state[env, xangvel_off_fl + b * 3 + 0]
                )
                var wy_w = rebind[Scalar[DTYPE]](
                    state[env, xangvel_off_fl + b * 3 + 1]
                )
                var wz_w = rebind[Scalar[DTYPE]](
                    state[env, xangvel_off_fl + b * 3 + 2]
                )

                # Rotate to body local frame (conjugate of xquat)
                var qx_b = rebind[Scalar[DTYPE]](
                    state[env, xquat_off_fl + b * 4 + 0]
                )
                var qy_b = rebind[Scalar[DTYPE]](
                    state[env, xquat_off_fl + b * 4 + 1]
                )
                var qz_b = rebind[Scalar[DTYPE]](
                    state[env, xquat_off_fl + b * 4 + 2]
                )
                var qw_b = rebind[Scalar[DTYPE]](
                    state[env, xquat_off_fl + b * 4 + 3]
                )

                var vloc_b = gpu_quat_rotate[DTYPE](
                    -qx_b, -qy_b, -qz_b, qw_b, vx_w, vy_w, vz_w
                )
                var wloc_b = gpu_quat_rotate[DTYPE](
                    -qx_b, -qy_b, -qz_b, qw_b, wx_w, wy_w, wz_w
                )
                var vx = vloc_b[0]
                var vy = vloc_b[1]
                var vz = vloc_b[2]
                var wx = wloc_b[0]
                var wy = wloc_b[1]
                var wz = wloc_b[2]

                var diam = (bx + by + bz) / Scalar[DTYPE](3)

                var lfx = Scalar[DTYPE](0)
                var lfy = Scalar[DTYPE](0)
                var lfz = Scalar[DTYPE](0)
                var ltx = Scalar[DTYPE](0)
                var lty = Scalar[DTYPE](0)
                var ltz = Scalar[DTYPE](0)

                if mu_fl > Scalar[DTYPE](0):
                    var visc_lin = Scalar[DTYPE](3) * PI_FL * diam * mu_fl
                    lfx = lfx - visc_lin * vx
                    lfy = lfy - visc_lin * vy
                    lfz = lfz - visc_lin * vz
                    var d3 = diam * diam * diam
                    var visc_ang = PI_FL * d3 * mu_fl
                    ltx = ltx - visc_ang * wx
                    lty = lty - visc_ang * wy
                    ltz = ltz - visc_ang * wz

                if rho_fl > Scalar[DTYPE](0):
                    var half_rho = Scalar[DTYPE](0.5) * rho_fl
                    lfx = lfx - half_rho * by * bz * abs(vx) * vx
                    lfy = lfy - half_rho * bx * bz * abs(vy) * vy
                    lfz = lfz - half_rho * bx * by * abs(vz) * vz
                    var bx4 = bx * bx * bx * bx
                    var by4 = by * by * by * by
                    var bz4 = bz * bz * bz * bz
                    ltx = ltx - rho_fl * bx * (by4 + bz4) * abs(
                        wx
                    ) * wx / Scalar[DTYPE](64)
                    lty = lty - rho_fl * by * (bx4 + bz4) * abs(
                        wy
                    ) * wy / Scalar[DTYPE](64)
                    ltz = ltz - rho_fl * bz * (bx4 + by4) * abs(
                        wz
                    ) * wz / Scalar[DTYPE](64)

                # Rotate forces to world frame
                var fw_b = gpu_quat_rotate[DTYPE](
                    qx_b, qy_b, qz_b, qw_b, lfx, lfy, lfz
                )
                var tw_b = gpu_quat_rotate[DTYPE](
                    qx_b, qy_b, qz_b, qw_b, ltx, lty, ltz
                )
                var fx_w = fw_b[0]
                var fy_w = fw_b[1]
                var fz_w = fw_b[2]
                var tx_w = tw_b[0]
                var ty_w = tw_b[1]
                var tz_w = tw_b[2]

                # Apply wrench at xipos via Jacobian transpose (kinematic tree walk)
                # Transport wrench to subtree_com[rootid] (cdof reference point)
                comptime stcom_off_fl = subtree_com_offset[
                    NQ, NV, NBODY, MAX_CONTACTS
                ]()
                var px_b = rebind[Scalar[DTYPE]](
                    state[env, xipos_off_fl + b * 3 + 0]
                )
                var py_b = rebind[Scalar[DTYPE]](
                    state[env, xipos_off_fl + b * 3 + 1]
                )
                var pz_b = rebind[Scalar[DTYPE]](
                    state[env, xipos_off_fl + b * 3 + 2]
                )
                var rootid_b = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, body_off_b + BODY_IDX_ROOTID]
                    )
                )
                var dx_b = px_b - rebind[Scalar[DTYPE]](
                    state[env, stcom_off_fl + rootid_b * 3 + 0]
                )
                var dy_b = py_b - rebind[Scalar[DTYPE]](
                    state[env, stcom_off_fl + rootid_b * 3 + 1]
                )
                var dz_b = pz_b - rebind[Scalar[DTYPE]](
                    state[env, stcom_off_fl + rootid_b * 3 + 2]
                )
                var tau_ox = tx_w + dy_b * fz_w - dz_b * fy_w
                var tau_oy = ty_w + dz_b * fx_w - dx_b * fz_w
                var tau_oz = tz_w + dx_b * fy_w - dy_b * fx_w

                var anc = b
                while anc > 0:
                    for j2 in range(NJOINT):
                        var jo2 = model_joint_offset[NBODY](j2)
                        var bid2 = Int(
                            rebind[Scalar[DTYPE]](
                                model[0, jo2 + JOINT_IDX_BODY_ID]
                            )
                        )
                        if bid2 != anc:
                            continue
                        var jt2 = Int(
                            rebind[Scalar[DTYPE]](
                                model[0, jo2 + JOINT_IDX_TYPE]
                            )
                        )
                        var da2 = Int(
                            rebind[Scalar[DTYPE]](
                                model[0, jo2 + JOINT_IDX_DOF_ADR]
                            )
                        )
                        var nd2 = 1
                        if jt2 == JNT_FREE:
                            nd2 = 6
                        elif jt2 == JNT_BALL:
                            nd2 = 3
                        for d2 in range(nd2):
                            var di2 = da2 + d2
                            var ca0 = rebind[Scalar[DTYPE]](
                                workspace[env, cdof_off_fl + di2 * 6 + 0]
                            )
                            var ca1 = rebind[Scalar[DTYPE]](
                                workspace[env, cdof_off_fl + di2 * 6 + 1]
                            )
                            var ca2 = rebind[Scalar[DTYPE]](
                                workspace[env, cdof_off_fl + di2 * 6 + 2]
                            )
                            var cl0 = rebind[Scalar[DTYPE]](
                                workspace[env, cdof_off_fl + di2 * 6 + 3]
                            )
                            var cl1 = rebind[Scalar[DTYPE]](
                                workspace[env, cdof_off_fl + di2 * 6 + 4]
                            )
                            var cl2 = rebind[Scalar[DTYPE]](
                                workspace[env, cdof_off_fl + di2 * 6 + 5]
                            )
                            var cur2 = rebind[Scalar[DTYPE]](
                                workspace[env, fnet_idx + di2]
                            )
                            workspace[env, fnet_idx + di2] = (
                                cur2
                                + cl0 * fx_w
                                + cl1 * fy_w
                                + cl2 * fz_w
                                + ca0 * tau_ox
                                + ca1 * tau_oy
                                + ca2 * tau_oz
                            )
                    var anc_off = model_body_offset(anc)
                    anc = Int(
                        rebind[Scalar[DTYPE]](
                            model[0, anc_off + BODY_IDX_PARENT]
                        )
                    )

        # 10. LDL solve: f_net → qacc
        comptime if SPARSE:
            ldl_solve_sparse_gpu[DTYPE, NV, NBODY, NM, BATCH, WS_SIZE](
                env, workspace, sp_row_nnz, sp_row_adr, sp_col_ind
            )
        else:
            ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
                env, workspace
            )

        # Write qacc to state and qacc_constrained for solver
        for i in range(NV):
            var qacc_val = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            state[env, qacc_off + i] = qacc_val
            workspace[env, qacc_constrained_idx + i] = qacc_val

    @always_inline
    @staticmethod
    def rk4_combine_kernel[
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
        SOLVER_WS_SIZE: Int,
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
        """RK4 combine kernel: save A[3], combine all stages, integrate.

        Reads A[0..2] from workspace, A[3] from qacc_constrained.
        Computes RK4-weighted qacc and velocity, integrates position.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        comptime qacc_constrained_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime q0_idx = ws_rk4_q0_offset[NV, NBODY](SOLVER_WS_SIZE)
        comptime v0_idx = ws_rk4_v0_offset[NV, NBODY, NQ](SOLVER_WS_SIZE)
        comptime A0_idx = ws_rk4_A_offset[NV, NBODY, NQ](SOLVER_WS_SIZE, 0)
        comptime A1_idx = ws_rk4_A_offset[NV, NBODY, NQ](SOLVER_WS_SIZE, 1)
        comptime A2_idx = ws_rk4_A_offset[NV, NBODY, NQ](SOLVER_WS_SIZE, 2)
        comptime c1_idx = ws_rk4_C1_offset[NV, NBODY, NQ](SOLVER_WS_SIZE)
        comptime c2_idx = ws_rk4_C2_offset[NV, NBODY, NQ](SOLVER_WS_SIZE)

        var qpos_off = qpos_offset[NQ, NV]()
        var qvel_off = qvel_offset[NQ, NV]()
        var qacc_off = qacc_offset[NQ, NV]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )

        comptime ONE_SIXTH = Scalar[DTYPE](1.0 / 6.0)
        comptime ONE_THIRD = Scalar[DTYPE](1.0 / 3.0)

        # Read A[3] from qacc_constrained (stage 3 solver just ran)
        # Compute qacc_combined = (A[0] + 2*A[1] + 2*A[2] + A[3]) / 6
        # Compute C[3] = v0 + dt * A[2]
        # v_combined = (C[0] + 2*C[1] + 2*C[2] + C[3]) / 6
        #   where C[0] = v0

        # First pass: compute qacc_combined, v_combined, update qvel/qacc.
        # Store v_combined in A0 workspace (no longer needed after this).
        for i in range(NV):
            var a0_i = rebind[Scalar[DTYPE]](workspace[env, A0_idx + i])
            var a1_i = rebind[Scalar[DTYPE]](workspace[env, A1_idx + i])
            var a2_i = rebind[Scalar[DTYPE]](workspace[env, A2_idx + i])
            var a3_i = rebind[Scalar[DTYPE]](
                workspace[env, qacc_constrained_idx + i]
            )
            var v0_i = rebind[Scalar[DTYPE]](workspace[env, v0_idx + i])
            var c1_i = rebind[Scalar[DTYPE]](workspace[env, c1_idx + i])
            var c2_i = rebind[Scalar[DTYPE]](workspace[env, c2_idx + i])

            # Combined acceleration
            var qacc_i = (
                ONE_SIXTH * a0_i
                + ONE_THIRD * a1_i
                + ONE_THIRD * a2_i
                + ONE_SIXTH * a3_i
            )

            # C[3] = v0 + dt * A[2]
            var c3_i = v0_i + dt * a2_i

            # Combined velocity — store in A0 workspace for position integration
            # NaN guard + clamp: if any stage solver produced NaN qacc, c1/c2/c3
            # are NaN; clamp v_combined to prevent NaN qpos integration.
            var v_combined_i = (
                ONE_SIXTH * v0_i
                + ONE_THIRD * c1_i
                + ONE_THIRD * c2_i
                + ONE_SIXTH * c3_i
            )
            var vpos_max = Scalar[DTYPE](100.0)
            if v_combined_i != v_combined_i:  # NaN guard: no position change
                v_combined_i = Scalar[DTYPE](0.0)
            elif v_combined_i > vpos_max:
                v_combined_i = vpos_max
            elif v_combined_i < -vpos_max:
                v_combined_i = -vpos_max
            workspace[env, A0_idx + i] = v_combined_i

            # Integrate: qvel = v0 + qacc * dt  (with NaN guard + velocity clamp)
            var qvel_new = v0_i + qacc_i * dt
            var qvel_max = Scalar[DTYPE](100.0)
            if qvel_new != qvel_new:  # NaN guard: reset to zero
                qvel_new = Scalar[DTYPE](0.0)
            elif qvel_new > qvel_max:
                qvel_new = qvel_max
            elif qvel_new < -qvel_max:
                qvel_new = -qvel_max
            state[env, qvel_off + i] = qvel_new
            state[env, qacc_off + i] = qacc_i

        # Second pass: integrate position using v_combined (quaternion-aware).
        # v_combined is stored in A0_idx workspace.
        _integrate_pos_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
        ](env, state, model, workspace, q0_idx, A0_idx, dt)

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
        """Perform one RK4 physics step on GPU.

        Launches 9 kernels: 4 × (stage + solver) + 1 combine.
        Workspace must include RK4 extra space beyond the standard layout.

        Solver launch is gated by the module-level RK4_BLOCKED_SOLVER flag
        (5a): when True the Newton solver runs one-environment-per-block instead
        of packing many envs per block. Bit-identical to the packed launch (same
        kernel, same per-env math) but isolates per-env Newton iteration-count
        divergence across independently-scheduled blocks. See
        docs/PHYSICS3D_BLOCKED_SOLVER.md.
        """
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
        comptime MODEL_SIZE = model_size_with_invweight[
            NBODY,
            NJOINT,
            NV,
            NGEOM,
            NEQUALITY=MAX_EQUALITY,
            NTENDON=MAX_TENDON,
            NSITE=NSITE,
        ]()
        comptime SOLVER_WS = Self.SOLVER.solver_workspace_size[
            NV, MAX_CONTACTS
        ]()
        comptime WS_SIZE = (
            integrator_workspace_size[NV, NBODY]()
            + NV * NV
            + SOLVER_WS
            + rk4_extra_workspace_size[NQ, NV]()
        )

        comptime V_SIZE = _max_one[NV]()
        comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime THREADS = Self.SOLVER.solver_threads[
            NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ]()
        comptime SOLVER_THREADS_BLOCKS = (THREADS + THREADS - 1) // THREADS
        comptime SOLVER_ENV_TPB = TPB // THREADS
        comptime SOLVER_ENV_BLOCKS = (
            BATCH + SOLVER_ENV_TPB - 1
        ) // SOLVER_ENV_TPB

        # 5a: RK4_BLOCKED_SOLVER launches solve_gpu one-env-per-block. With
        # block_dim.x == 1, env = block_idx.x and contact_tid = thread_idx.y,
        # so the per-env math is byte-for-byte identical to the packed launch.
        # 5b: for PYRAMIDAL cones the dedicated solve_gpu_blocked kernel runs
        # cooperatively across the block's threads (one env per block, 1D
        # block_dim=(THREADS,)). Non-PYRAMIDAL cones keep the 5a serial launch.
        comptime USE_BLOCKED_PYR = RK4_BLOCKED_SOLVER and (
            CONE_TYPE == ConeType.PYRAMIDAL
        )
        comptime SOLVER_GRID_X = BATCH if RK4_BLOCKED_SOLVER else SOLVER_ENV_BLOCKS
        comptime SOLVER_GRID_Y = 1 if RK4_BLOCKED_SOLVER else SOLVER_THREADS_BLOCKS
        comptime SOLVER_BLOCK_X = 1 if RK4_BLOCKED_SOLVER else SOLVER_ENV_TPB

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf)

        var model = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf)

        var workspace = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ](workspace_buf)

        # Grid configuration for stage kernels (multi-threaded)
        comptime STEP_ENV_TPB = TPB // STEP_THREADS
        comptime STEP_ENV_BLOCKS = (BATCH + STEP_ENV_TPB - 1) // STEP_ENV_TPB

        # --- Stage 0: forward dynamics at (q0, v0) ---
        comptime stage0_kernel = Self.rk4_stage_kernel[
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
            SOLVER_WS,
            0,
            NM,
            SPARSE,
            STEP_THREADS,
        ]
        comptime if STEP_THREADS > 1:
            ctx.enqueue_function[stage0_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        else:
            ctx.enqueue_function[stage0_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

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
        comptime solver_blocked_wrapper = Self.SOLVER.solve_gpu_blocked[
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
        comptime if USE_BLOCKED_PYR:
            ctx.enqueue_function[solver_blocked_wrapper](
                state,
                model,
                workspace,
                grid_dim=(BATCH, 1),
                block_dim=(THREADS,),
            )
        else:
            ctx.enqueue_function[solver_wrapper](
                state,
                model,
                workspace,
                grid_dim=(SOLVER_GRID_X, SOLVER_GRID_Y),
                block_dim=(SOLVER_BLOCK_X, THREADS),
            )

        # --- Stage 1: forward dynamics at (q0+dt/2*C[0], v0+dt/2*A[0]) ---
        comptime stage1_kernel = Self.rk4_stage_kernel[
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
            SOLVER_WS,
            1,
            NM,
            SPARSE,
            STEP_THREADS,
        ]
        comptime if STEP_THREADS > 1:
            ctx.enqueue_function[stage1_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        else:
            ctx.enqueue_function[stage1_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        comptime if USE_BLOCKED_PYR:
            ctx.enqueue_function[solver_blocked_wrapper](
                state,
                model,
                workspace,
                grid_dim=(BATCH, 1),
                block_dim=(THREADS,),
            )
        else:
            ctx.enqueue_function[solver_wrapper](
                state,
                model,
                workspace,
                grid_dim=(SOLVER_GRID_X, SOLVER_GRID_Y),
                block_dim=(SOLVER_BLOCK_X, THREADS),
            )

        # --- Stage 2: forward dynamics at (q0+dt/2*C[1], v0+dt/2*A[1]) ---
        comptime stage2_kernel = Self.rk4_stage_kernel[
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
            SOLVER_WS,
            2,
            NM,
            SPARSE,
            STEP_THREADS,
        ]
        comptime if STEP_THREADS > 1:
            ctx.enqueue_function[stage2_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        else:
            ctx.enqueue_function[stage2_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        comptime if USE_BLOCKED_PYR:
            ctx.enqueue_function[solver_blocked_wrapper](
                state,
                model,
                workspace,
                grid_dim=(BATCH, 1),
                block_dim=(THREADS,),
            )
        else:
            ctx.enqueue_function[solver_wrapper](
                state,
                model,
                workspace,
                grid_dim=(SOLVER_GRID_X, SOLVER_GRID_Y),
                block_dim=(SOLVER_BLOCK_X, THREADS),
            )

        # --- Stage 3: forward dynamics at (q0+dt*C[2], v0+dt*A[2]) ---
        comptime stage3_kernel = Self.rk4_stage_kernel[
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
            SOLVER_WS,
            3,
            NM,
            SPARSE,
            STEP_THREADS,
        ]
        comptime if STEP_THREADS > 1:
            ctx.enqueue_function[stage3_kernel](
                state,
                model,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        else:
            ctx.enqueue_function[stage3_kernel](
                state,
                model,
                workspace,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        comptime if USE_BLOCKED_PYR:
            ctx.enqueue_function[solver_blocked_wrapper](
                state,
                model,
                workspace,
                grid_dim=(BATCH, 1),
                block_dim=(THREADS,),
            )
        else:
            ctx.enqueue_function[solver_wrapper](
                state,
                model,
                workspace,
                grid_dim=(SOLVER_GRID_X, SOLVER_GRID_Y),
                block_dim=(SOLVER_BLOCK_X, THREADS),
            )

        # --- Combine: weighted average + integrate ---
        comptime combine_kernel = Self.rk4_combine_kernel[
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
            SOLVER_WS,
        ]
        ctx.enqueue_function[combine_kernel](
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
        """Register 5 profiling slots for RK4 GPU step phases.

        Slots (relative to returned base):
            +0: stage0  (dynamics + solver at (q0, v0))
            +1: stage1  (dynamics + solver at half-step from stage0)
            +2: stage2  (dynamics + solver at half-step from stage1)
            +3: stage3  (dynamics + solver at full-step from stage2)
            +4: combine  (RK4 weighted average + integration)

        Args:
            timer: PerfTimer to add slots to.
            parent: Parent slot index (-1 = top-level).

        Returns:
            Base slot index.
        """
        var base = timer.add_slot("stage0", parent=parent)
        _ = timer.add_slot("stage1", parent=parent)
        _ = timer.add_slot("stage2", parent=parent)
        _ = timer.add_slot("stage3", parent=parent)
        _ = timer.add_slot("combine", parent=parent)
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
        """Profiled GPU step — same as step_gpu but with per-stage timing.

        Call register_gpu_profile_slots() first to get the base slot index.
        Each slot covers a full RK4 stage (dynamics + solver).
        """
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
        comptime MODEL_SIZE = model_size_with_invweight[
            NBODY,
            NJOINT,
            NV,
            NGEOM,
            NEQUALITY=MAX_EQUALITY,
            NTENDON=MAX_TENDON,
            NSITE=NSITE,
        ]()
        comptime SOLVER_WS = Self.SOLVER.solver_workspace_size[
            NV, MAX_CONTACTS
        ]()
        comptime WS_SIZE = (
            integrator_workspace_size[NV, NBODY]()
            + NV * NV
            + SOLVER_WS
            + rk4_extra_workspace_size[NQ, NV]()
        )

        comptime V_SIZE = _max_one[NV]()
        comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime THREADS = Self.SOLVER.solver_threads[
            NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ]()
        comptime SOLVER_THREADS_BLOCKS = (THREADS + THREADS - 1) // THREADS
        comptime SOLVER_ENV_TPB = TPB // THREADS
        comptime SOLVER_ENV_BLOCKS = (
            BATCH + SOLVER_ENV_TPB - 1
        ) // SOLVER_ENV_TPB

        # 5a: RK4_BLOCKED_SOLVER launches solve_gpu one-env-per-block (see step_gpu).
        # 5b: PYRAMIDAL cones route to the cooperative solve_gpu_blocked kernel.
        comptime USE_BLOCKED_PYR = RK4_BLOCKED_SOLVER and (
            CONE_TYPE == ConeType.PYRAMIDAL
        )
        comptime SOLVER_GRID_X = BATCH if RK4_BLOCKED_SOLVER else SOLVER_ENV_BLOCKS
        comptime SOLVER_GRID_Y = 1 if RK4_BLOCKED_SOLVER else SOLVER_THREADS_BLOCKS
        comptime SOLVER_BLOCK_X = 1 if RK4_BLOCKED_SOLVER else SOLVER_ENV_TPB

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf)
        var model = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf)
        var workspace = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ](workspace_buf)

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
        comptime solver_blocked_wrapper = Self.SOLVER.solve_gpu_blocked[
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

        # ---- Stage 0 ----
        timer.sync_and_mark(ctx)

        comptime stage0_kernel = Self.rk4_stage_kernel[
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
            SOLVER_WS,
            0,
            NM,
            SPARSE,
        ]
        ctx.enqueue_function[stage0_kernel](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if USE_BLOCKED_PYR:
            ctx.enqueue_function[solver_blocked_wrapper](
                state,
                model,
                workspace,
                grid_dim=(BATCH, 1),
                block_dim=(THREADS,),
            )
        else:
            ctx.enqueue_function[solver_wrapper](
                state,
                model,
                workspace,
                grid_dim=(SOLVER_GRID_X, SOLVER_GRID_Y),
                block_dim=(SOLVER_BLOCK_X, THREADS),
            )

        timer.sync_and_accumulate(base + 0, ctx)

        # ---- Stage 1 ----
        timer.mark()

        comptime stage1_kernel = Self.rk4_stage_kernel[
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
            SOLVER_WS,
            1,
            NM,
            SPARSE,
        ]
        ctx.enqueue_function[stage1_kernel](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if USE_BLOCKED_PYR:
            ctx.enqueue_function[solver_blocked_wrapper](
                state,
                model,
                workspace,
                grid_dim=(BATCH, 1),
                block_dim=(THREADS,),
            )
        else:
            ctx.enqueue_function[solver_wrapper](
                state,
                model,
                workspace,
                grid_dim=(SOLVER_GRID_X, SOLVER_GRID_Y),
                block_dim=(SOLVER_BLOCK_X, THREADS),
            )

        timer.sync_and_accumulate(base + 1, ctx)

        # ---- Stage 2 ----
        timer.mark()

        comptime stage2_kernel = Self.rk4_stage_kernel[
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
            SOLVER_WS,
            2,
            NM,
            SPARSE,
        ]
        ctx.enqueue_function[stage2_kernel](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if USE_BLOCKED_PYR:
            ctx.enqueue_function[solver_blocked_wrapper](
                state,
                model,
                workspace,
                grid_dim=(BATCH, 1),
                block_dim=(THREADS,),
            )
        else:
            ctx.enqueue_function[solver_wrapper](
                state,
                model,
                workspace,
                grid_dim=(SOLVER_GRID_X, SOLVER_GRID_Y),
                block_dim=(SOLVER_BLOCK_X, THREADS),
            )

        timer.sync_and_accumulate(base + 2, ctx)

        # ---- Stage 3 ----
        timer.mark()

        comptime stage3_kernel = Self.rk4_stage_kernel[
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
            SOLVER_WS,
            3,
            NM,
            SPARSE,
        ]
        ctx.enqueue_function[stage3_kernel](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if USE_BLOCKED_PYR:
            ctx.enqueue_function[solver_blocked_wrapper](
                state,
                model,
                workspace,
                grid_dim=(BATCH, 1),
                block_dim=(THREADS,),
            )
        else:
            ctx.enqueue_function[solver_wrapper](
                state,
                model,
                workspace,
                grid_dim=(SOLVER_GRID_X, SOLVER_GRID_Y),
                block_dim=(SOLVER_BLOCK_X, THREADS),
            )

        timer.sync_and_accumulate(base + 3, ctx)

        # ---- Combine ----
        timer.mark()

        comptime combine_kernel = Self.rk4_combine_kernel[
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
            SOLVER_WS,
        ]
        ctx.enqueue_function[combine_kernel](
            state,
            model,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        timer.sync_and_accumulate(base + 4, ctx)

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
        """Run RK4 simulation for multiple steps on GPU."""
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
