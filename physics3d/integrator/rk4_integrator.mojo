"""RK4 (4th-order Runge-Kutta) integrator for physics simulation.

Provides 4th-order accuracy for smooth dynamics, with significantly better
energy conservation than Euler for conservative systems (undamped pendulums,
free-flight dynamics). Uses the standard Butcher tableau:

  c = [0, 1/2, 1/2, 1]
  b = [1/6, 1/3, 1/3, 1/6]

Pipeline per step:
1. Save initial (qpos, qvel)
2. Stage 1: evaluate forward_dynamics at (q0, v0) -> a1
3. Stage 2: evaluate at (q0 + dt/2*v0, v0 + dt/2*a1) -> a2
4. Stage 3: evaluate at (q0 + dt/2*v2, v0 + dt/2*a2) -> a3
5. Stage 4: evaluate at (q0 + dt*v3, v0 + dt*a3) -> a4
6. Combine: qacc = (a1 + 2*a2 + 2*a3 + a4) / 6
7. Constraint solve on combined acceleration (once)
8. Integrate: qvel += qacc*dt, qpos += qvel*dt (quaternion-aware)
9. Normalize quaternions

CPU only — 4x dynamics cost makes this impractical for batched GPU RL training.
Use for validation, trajectory comparison, and energy conservation testing.
"""

from math import sqrt
from gpu.host import DeviceContext, DeviceBuffer

from ..types import Model, Data, _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from ..kinematics.quat_math import quat_normalize, quat_integrate
from ..dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    compute_M_inv_from_ldl,
)
from ..dynamics.bias_forces import compute_bias_forces_rne
from ..dynamics.jacobian import compute_cdof, compute_composite_inertia
from ..collision.contact_detection import (
    detect_contacts,
    normalize_qpos_quaternions,
)
from ..constraints.constraint_data import ConstraintData
from ..constraints.constraint_builder import build_constraints, writeback_forces
from ..traits.integrator import Integrator
from ..traits.solver import ConstraintSolver


fn _forward_dynamics[
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
    ],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut qacc_out: InlineArray[Scalar[DTYPE], V_SIZE],
    mut cdof_out: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    mut M_inv_out: InlineArray[Scalar[DTYPE], M_SIZE],
) where DTYPE.is_floating_point():
    """Compute unconstrained acceleration from current (qpos, qvel) in data.

    Runs the full dynamics pipeline:
    FK -> body velocities -> collision -> cdof -> CRBA -> M -> LDL -> bias -> passive -> solve.

    Returns qacc, cdof (for constraint builder), and M_inv (for constraint solver).
    """
    var _ = (
        model.timestep
    )  # Not used — RK4 uses explicit damping (no dt*D in M)

    # 1. Forward kinematics + body velocities
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    # 2. Collision detection
    detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )

    # 3. Compute cdof
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model, data, cdof_out
    )

    # 4. Composite rigid body inertia
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
    ](model, data, crb)

    # 5. Full mass matrix
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
    ](model, data, cdof_out, crb, M)

    # 5b. Armature only (no implicit damping for RK4 — damping is explicit)
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
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

    # 6. LDL factorize
    var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var D = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

    # 7. Bias forces
    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        bias[i] = Scalar[DTYPE](0)
    compute_bias_forces_rne[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
    ](model, data, cdof_out, bias)

    # 8. Net force = external - bias - passive
    var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
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

    # 9. qacc = M^-1 * f_net
    for i in range(NV):
        qacc_out[i] = Scalar[DTYPE](0)
    ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D, f_net, qacc_out)

    # 10. M_inv for constraint solver (only needed at final stage, but
    # we compute it here since we have L,D — caller can ignore if not needed)
    for i in range(M_SIZE):
        M_inv_out[i] = Scalar[DTYPE](0)
    compute_M_inv_from_ldl[DTYPE, NV, M_SIZE, V_SIZE](L, D, M_inv_out)


fn _integrate_pos[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
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
    ],
    qpos_base: InlineArray[Scalar[DTYPE], _max_one[NQ]()],
    vel: InlineArray[Scalar[DTYPE], _max_one[NV]()],
    dt: Scalar[DTYPE],
    mut qpos_out: InlineArray[Scalar[DTYPE], _max_one[NQ]()],
) where DTYPE.is_floating_point():
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
            # Quaternion: exponential map integration
            var qx = qpos_base[qpos_adr + 3]
            var qy = qpos_base[qpos_adr + 4]
            var qz = qpos_base[qpos_adr + 5]
            var qw = qpos_base[qpos_adr + 6]
            var wx = vel[dof_adr + 3]
            var wy = vel[dof_adr + 4]
            var wz = vel[dof_adr + 5]
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(
                result[0], result[1], result[2], result[3]
            )
            qpos_out[qpos_adr + 3] = norm[0]
            qpos_out[qpos_adr + 4] = norm[1]
            qpos_out[qpos_adr + 5] = norm[2]
            qpos_out[qpos_adr + 6] = norm[3]

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


struct RK4Integrator[SOLVER: ConstraintSolver](Integrator):
    """4th-order Runge-Kutta integrator with configurable constraint solver.

    Provides 4th-order accuracy for smooth dynamics with significantly better
    energy conservation than Euler integration. Constraint solver runs once
    per step on the combined RK4 acceleration.

    CPU only — GPU methods are stubs (4x dynamics cost is impractical for
    batched RL training).

    Usage:
        alias RK4PGS = RK4Integrator[PGSSolver]
        RK4PGS.step(model, data)
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    @staticmethod
    fn step[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
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
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        verbose: Bool = False,
    ) where DTYPE.is_floating_point():
        """Execute one RK4 simulation step.

        Evaluates forward dynamics 4 times at different (q, v) states,
        combines with RK4 weights, then runs constraint solver once.
        """
        var dt = model.timestep
        comptime Q_SIZE = _max_one[NQ]()
        comptime V_SIZE = _max_one[NV]()
        comptime M_SIZE = _max_one[NV * NV]()
        comptime CDOF_SIZE = _max_one[NV * 6]()
        comptime CRB_SIZE = _max_one[NBODY * 10]()

        # Save initial state
        var q0 = InlineArray[Scalar[DTYPE], Q_SIZE](uninitialized=True)
        var v0 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NQ):
            q0[i] = data.qpos[i]
        for i in range(NV):
            v0[i] = data.qvel[i]

        # Accumulators for RK4 weighted sum
        var a1 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var a2 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var a3 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var a4 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Workspace for _forward_dynamics (cdof and M_inv from last stage used for constraints)
        var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
        var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)

        # =====================================================================
        # Stage 1: evaluate at (q0, v0)
        # =====================================================================
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
        ](model, data, a1, cdof, M_inv)

        # =====================================================================
        # Stage 2: evaluate at (q0 + dt/2 * v0, v0 + dt/2 * a1)
        # =====================================================================
        var half_dt = dt * Scalar[DTYPE](0.5)

        # Set velocities for stage 2
        for i in range(NV):
            data.qvel[i] = v0[i] + half_dt * a1[i]

        # Set positions for stage 2: q0 + dt/2 * v0
        var q_stage = InlineArray[Scalar[DTYPE], Q_SIZE](uninitialized=True)
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
        ](model, data, a2, cdof, M_inv)

        # =====================================================================
        # Stage 3: evaluate at (q0 + dt/2 * v2, v0 + dt/2 * a2)
        # =====================================================================
        # v2 = v0 + dt/2 * a2 (the velocity AT stage 2 evaluation point... but
        # MuJoCo RK4 uses k_q = velocity at the stage, so:
        # k2_q = v2 = v0 + dt/2 * a1  (the velocity we set for stage 2)
        # Stage 3 pos = q0 + dt/2 * k2_q = q0 + dt/2 * (v0 + dt/2 * a1)
        # But standard RK4 for q' = v, v' = a(q,v) is:
        #   k1_q = v0,              k1_v = a1
        #   k2_q = v0 + dt/2 * a1,  k2_v = a2
        #   k3_q = v0 + dt/2 * a2,  k3_v = a3
        #   k4_q = v0 + dt * a3,    k4_v = a4
        # Position update: q = q0 + dt/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)

        # v_stage3 for evaluation = v0 + dt/2 * a2
        for i in range(NV):
            data.qvel[i] = v0[i] + half_dt * a2[i]

        # k2_q = v0 + dt/2 * a1 (already computed as data.qvel from stage 2)
        var v_stage2 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            v_stage2[i] = v0[i] + half_dt * a1[i]

        # q_stage3 = q0 + dt/2 * v_stage2
        _integrate_pos[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
        ](model, q0, v_stage2, half_dt, q_stage)
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
        ](model, data, a3, cdof, M_inv)

        # =====================================================================
        # Stage 4: evaluate at (q0 + dt * v3, v0 + dt * a3)
        # =====================================================================
        # v_stage3 = v0 + dt/2 * a2 (k3_q)
        var v_stage3 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            v_stage3[i] = v0[i] + half_dt * a2[i]

        for i in range(NV):
            data.qvel[i] = v0[i] + dt * a3[i]

        # q_stage4 = q0 + dt * v_stage3
        _integrate_pos[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
        ](model, q0, v_stage3, dt, q_stage)
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
        ](model, data, a4, cdof, M_inv)

        # =====================================================================
        # Combine: qacc = (a1 + 2*a2 + 2*a3 + a4) / 6
        # =====================================================================
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        comptime ONE_SIXTH: Scalar[DTYPE] = 1.0 / 6.0
        comptime ONE_THIRD: Scalar[DTYPE] = 1.0 / 3.0
        for i in range(NV):
            qacc[i] = (
                ONE_SIXTH * a1[i]
                + ONE_THIRD * a2[i]
                + ONE_THIRD * a3[i]
                + ONE_SIXTH * a4[i]
            )

        # =====================================================================
        # Restore state for constraint solving
        # =====================================================================
        # Restore original qpos/qvel so constraint solver sees the initial state
        for i in range(NQ):
            data.qpos[i] = q0[i]
        for i in range(NV):
            data.qvel[i] = v0[i]

        # Re-run FK and collision for constraint solver (needs current geometry)
        forward_kinematics(model, data)
        compute_body_velocities(model, data)
        detect_contacts(model, data)

        # Re-compute cdof for constraint builder
        compute_cdof(model, data, cdof)

        # Re-compute M_inv for constraint solver (at initial state)
        var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
        for i in range(CRB_SIZE):
            crb[i] = Scalar[DTYPE](0)
        compute_composite_inertia[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
        ](model, data, crb)

        var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M[i] = Scalar[DTYPE](0)
        compute_mass_matrix_full(model, data, cdof, crb, M)

        # Add armature to M diagonal
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var arm = joint.armature
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

        var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var D_ldl = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D_ldl)
        for i in range(M_SIZE):
            M_inv[i] = Scalar[DTYPE](0)
        compute_M_inv_from_ldl[DTYPE, NV, M_SIZE, V_SIZE](L, D_ldl, M_inv)

        # =====================================================================
        # Constraint solve on combined acceleration
        # =====================================================================
        comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT + 6 * MAX_EQUALITY
        var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
        build_constraints[CONE_TYPE=CONE_TYPE,](
            model, data, cdof, M_inv, qacc, dt, constraints
        )

        Self.SOLVER.solve[CONE_TYPE=CONE_TYPE](
            model, data, M_inv, constraints, qacc, dt
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

        # =====================================================================
        # Final integration
        # =====================================================================
        # qvel = v0 + qacc * dt
        for i in range(NV):
            data.qacc[i] = qacc[i]
            data.qvel[i] = v0[i] + qacc[i] * dt

        # Position integration: combine velocity k-vectors with RK4 weights
        # k1_q = v0, k2_q = v0+dt/2*a1, k3_q = v0+dt/2*a2, k4_q = v0+dt*a3
        var v_combined = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        # k1_q = v0, k2_q = v_stage2, k3_q = v_stage3, k4_q = v0 + dt*a3
        for i in range(NV):
            var k4_q_i = v0[i] + dt * a3[i]
            v_combined[i] = (
                ONE_SIXTH * v0[i]
                + ONE_THIRD * v_stage2[i]
                + ONE_THIRD * v_stage3[i]
                + ONE_SIXTH * k4_q_i
            )

        _integrate_pos(model, q0, v_combined, dt, q_stage)
        for i in range(NQ):
            data.qpos[i] = q_stage[i]

        # Normalize quaternions
        normalize_qpos_quaternions(model, data)

    @staticmethod
    fn simulate[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
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
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        num_steps: Int,
    ) where DTYPE.is_floating_point():
        """Run simulation for multiple steps on CPU."""
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
            ](model, data)

    # =========================================================================
    # GPU Methods (stubs — RK4 is CPU only)
    # =========================================================================

    @staticmethod
    fn step_gpu[
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
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """GPU step is not supported for RK4Integrator.

        RK4 requires 4x dynamics evaluations per step, making it impractical
        for batched GPU simulation. Use EulerIntegrator or ImplicitFastIntegrator
        for GPU workloads.
        """
        raise "RK4Integrator does not support GPU execution. Use EulerIntegrator or ImplicitFastIntegrator for GPU."

    @staticmethod
    fn simulate_gpu[
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
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        num_steps: Int,
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """GPU simulate is not supported for RK4Integrator."""
        raise "RK4Integrator does not support GPU execution. Use EulerIntegrator or ImplicitFastIntegrator for GPU."
