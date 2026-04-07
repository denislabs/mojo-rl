"""Diagnostic: Hopper RK4 physics forces comparison at default standing pose.

Isolates the source of the ~2x angular velocity error on rooty seen in
trajectory comparisons. Compares intermediate quantities after a single
RK4 step (and also after 4 steps = 1 frame_skip):
  - Mass matrix (M)
  - Bias forces (Coriolis + gravity)
  - Contact constraint forces
  - Unconstrained qacc (M^{-1} * (qfrc - bias))
  - Constrained qacc (after solver)
  - Final qpos/qvel

Uses MuJoCo defaults: PYRAMIDAL cone (0), Newton solver (2), RK4 integrator (1).

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_hopper_rk4_forces_diagnostic.mojo
"""

from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, _max_one, ConeType
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_composite_inertia,
)
from mojo_rl.physics3d.dynamics.bias_forces import compute_bias_forces_rne
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    compute_M_inv_from_ldl,
)
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.constraints.constraint_builder import (
    build_constraints,
    writeback_forces,
)
from mojo_rl.physics3d.constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.joint_types import (
    JNT_HINGE,
    JNT_SLIDE,
    JNT_BALL,
    JNT_FREE,
)
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.envs.hopper.hopper_config import HopperConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HopperModel.NQ  # 6
comptime NV = HopperModel.NV  # 6
comptime NBODY = HopperModel.NBODY  # 5
comptime NJOINT = HopperModel.NJOINT  # 6
comptime NGEOM = HopperModel.NGEOM  # 5
comptime MAX_CONTACTS = HopperConfig.MAX_CONTACTS  # 20
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3
comptime FRAME_SKIP = HopperConfig.FRAME_SKIP  # 4

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()
comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT


# =============================================================================
# Helper: print vector comparison
# =============================================================================


def compare_vectors(
    label: String,
    our: InlineArray[Float64, NV],
    mj: InlineArray[Float64, NV],
):
    var max_abs: Float64 = 0.0
    var max_idx = 0
    print("  " + label + ":")
    for i in range(NV):
        var err = abs(our[i] - mj[i])
        if err > max_abs:
            max_abs = err
            max_idx = i
        print(
            "    [" + String(i) + "] ours=",
            our[i],
            " mj=",
            mj[i],
            " err=",
            err,
        )
    print("    max_err=" + String(max_abs) + " at [" + String(max_idx) + "]")


# =============================================================================
# Diagnostic 1: Intermediate forces at default standing pose
# =============================================================================


def diagnose_forces(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
) raises:
    """Compare intermediate physics quantities before any time integration."""
    print("=" * 70)
    print("DIAGNOSTIC:", test_name)
    print("=" * 70)

    # === Our engine ===
    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HopperModel.MAX_EQUALITY,
        HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON,
        HopperModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    HopperModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    # Apply actions to qfrc
    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])
    for i in range(NV):
        data.qfrc[i] = Scalar[DTYPE](0)
    HopperModel.apply_actions(data, action_list)

    print()
    print("--- Applied forces (qfrc) ---")
    for i in range(NV):
        print("  qfrc[" + String(i) + "] =", Float64(data.qfrc[i]))

    # 1. FK + velocities + cdof
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)

    # 2. Contacts
    detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )
    print()
    print("--- Contacts ---")
    print("  num_contacts:", Int(data.num_contacts))
    for c in range(Int(data.num_contacts)):
        print(
            "  contact[",
            c,
            "]: body_a=",
            Int(data.contacts[c].body_a),
            " body_b=",
            Int(data.contacts[c].body_b),
            " dist=",
            Float64(data.contacts[c].dist),
        )

    # 3. Mass matrix
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model, data, crb)

    var M = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(model, data, cdof, crb, M)

    # Add armature
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                M[(dof_adr + d) * NV + (dof_adr + d)] += arm
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                M[(dof_adr + d) * NV + (dof_adr + d)] += arm
        else:
            M[dof_adr * NV + dof_adr] += arm

    # 4. Bias forces
    var bias = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        bias.append(Scalar[DTYPE](0))
    compute_bias_forces_rne(model, data, cdof, bias)

    # 5. LDL
    var L = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[DTYPE](0))
    var D_ldl = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D_ldl.append(Scalar[DTYPE](0))
    ldl_factor[DTYPE, NV](M, L, D_ldl)

    var M_inv = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_inv.append(Scalar[DTYPE](0))
    compute_M_inv_from_ldl[DTYPE, NV](L, D_ldl, M_inv)

    # 6. f_net = qfrc - bias + passive
    var f_net = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(NV):
        f_net.append(data.qfrc[i] - bias[i])

    # Apply passive forces (damping)
    for j in range(model.num_joints):
        var joint_d = model.joints[j]
        var dof_adr_d = joint_d.dof_adr
        var damp_d = joint_d.damping
        if damp_d > Scalar[DTYPE](0):
            if joint_d.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr_d + d] -= damp_d * data.qvel[dof_adr_d + d]
            elif joint_d.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr_d + d] -= damp_d * data.qvel[dof_adr_d + d]
            else:
                f_net[dof_adr_d] -= damp_d * data.qvel[dof_adr_d]

    # 7. qacc_unconstrained
    var qacc0 = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        qacc0.append(Scalar[DTYPE](0))
    ldl_solve[DTYPE, NV](L, D_ldl, f_net, qacc0)

    # 8. Build + solve constraints
    var dt = Scalar[DTYPE](0.002)
    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints[CONE_TYPE=HopperModel.CONE_TYPE](
        model, data, cdof, M_inv, dt, constraints
    )

    for i in range(NV * NV):
        constraints.M_hat[i] = M[i]
    for i in range(NV):
        constraints.qfrc_smooth[i] = f_net[i]

    var qacc = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(NV):
        qacc.append(qacc0[i])

    NewtonSolver.solve[CONE_TYPE=HopperModel.CONE_TYPE](
        model, data, M_inv, constraints, qacc, dt
    )

    # Our qfrc_constraint = J^T * lambda
    var our_qfrc_c = InlineArray[Float64, NV](fill=0.0)
    for r in range(constraints.num_rows):
        var lam = Float64(constraints.rows[r].lambda_val)
        for i in range(NV):
            our_qfrc_c[i] += lam * Float64(constraints.J[r * NV + i])

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Use MuJoCo defaults (matching Gymnasium): PYRAMIDAL, Newton, RK4
    # Note: mj_forward doesn't use integrator, but set for consistency
    mj_model.opt.cone = 0  # mjCONE_PYRAMIDAL (MuJoCo default)
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.integrator = 1  # mjINT_RK4
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    # mj_forward computes forces without integrating
    mujoco.mj_forward(mj_model, mj_data)

    # Extract MuJoCo quantities
    var mj_bias_flat = mj_data.qfrc_bias.flatten().tolist()
    var mj_qacc_flat = mj_data.qacc.flatten().tolist()
    var mj_qfrc_c_flat = mj_data.qfrc_constraint.flatten().tolist()
    var mj_qfrc_act_flat = mj_data.qfrc_actuator.flatten().tolist()
    var mj_qfrc_pas_flat = mj_data.qfrc_passive.flatten().tolist()

    # Mass matrix
    var mj_M_dense = np.zeros(NV * NV).reshape(NV, NV)
    mujoco.mj_fullM(mj_model, mj_M_dense, mj_data.qM)
    var mj_M_flat = mj_M_dense.flatten().tolist()

    # Build comparison arrays
    var our_bias = InlineArray[Float64, NV](fill=0.0)
    var mj_bias = InlineArray[Float64, NV](fill=0.0)
    var our_qacc0 = InlineArray[Float64, NV](fill=0.0)
    var our_qacc_arr = InlineArray[Float64, NV](fill=0.0)
    var mj_qacc = InlineArray[Float64, NV](fill=0.0)
    var mj_qfrc_c = InlineArray[Float64, NV](fill=0.0)
    var mj_qfrc_act = InlineArray[Float64, NV](fill=0.0)
    var mj_qfrc_pas = InlineArray[Float64, NV](fill=0.0)

    for i in range(NV):
        our_bias[i] = Float64(bias[i])
        mj_bias[i] = Float64(py=mj_bias_flat[i])
        our_qacc0[i] = Float64(qacc0[i])
        our_qacc_arr[i] = Float64(qacc[i])
        mj_qacc[i] = Float64(py=mj_qacc_flat[i])
        mj_qfrc_c[i] = Float64(py=mj_qfrc_c_flat[i])
        mj_qfrc_act[i] = Float64(py=mj_qfrc_act_flat[i])
        mj_qfrc_pas[i] = Float64(py=mj_qfrc_pas_flat[i])

    # === Print comparisons ===
    print()
    print("--- Mass matrix diagonal ---")
    for i in range(NV):
        var our_mii = Float64(M[i * NV + i])
        var mj_mii = Float64(py=mj_M_flat[i * NV + i])
        print(
            "  M[" + String(i) + "," + String(i) + "] ours=",
            our_mii,
            " mj=",
            mj_mii,
            " err=",
            abs(our_mii - mj_mii),
        )

    # Full M comparison (off-diagonal)
    print()
    print("--- Mass matrix max off-diagonal error ---")
    var m_max_err: Float64 = 0.0
    for i in range(NV):
        for j in range(NV):
            var our_m = Float64(M[i * NV + j])
            var mj_m = Float64(py=mj_M_flat[i * NV + j])
            var err = abs(our_m - mj_m)
            if err > m_max_err:
                m_max_err = err
    print("  max_err =", m_max_err)

    print()
    print("--- Bias forces (qfrc_bias) ---")
    compare_vectors("bias", our_bias, mj_bias)

    print()
    print("--- MuJoCo qfrc_actuator ---")
    for i in range(NV):
        print("  mj_qfrc_actuator[" + String(i) + "] =", mj_qfrc_act[i])

    print()
    print("--- MuJoCo qfrc_passive ---")
    for i in range(NV):
        print("  mj_qfrc_passive[" + String(i) + "] =", mj_qfrc_pas[i])

    print()
    print("--- Constrained qacc (after solver) ---")
    compare_vectors("qacc", our_qacc_arr, mj_qacc)

    print()
    print("--- qfrc_constraint ---")
    compare_vectors("qfrc_constraint", our_qfrc_c, mj_qfrc_c)

    print()
    print("--- Contact count ---")
    print("  ours:", Int(data.num_contacts))
    print("  mj:  ", Int(py=mj_data.ncon))

    var mj_ncon = Int(py=mj_data.ncon)
    var mj_nefc = Int(py=mj_data.nefc)
    print("  mj_nefc:", mj_nefc)
    print("  our_rows:", constraints.num_rows)


# =============================================================================
# Diagnostic 2: Full RK4 step comparison
# =============================================================================


def diagnose_rk4_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int,
) raises:
    """Compare qpos/qvel after N RK4 steps between our engine and MuJoCo."""
    print()
    print("=" * 70)
    print("RK4 STEP:", test_name, "(", num_steps, "steps)")
    print("=" * 70)

    # === Our engine ===
    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HopperModel.MAX_EQUALITY,
        HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON,
        HopperModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    HopperModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    for _ in range(num_steps):
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)
        HopperModel.apply_actions(data, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    # === MuJoCo ===
    var mujoco = Python.import_module("mujoco")
    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.cone = 0  # PYRAMIDAL
    mj_model.opt.solver = 2  # Newton
    mj_model.opt.integrator = 1  # RK4
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)

    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()

    print()
    print("--- qpos after " + String(num_steps) + " steps ---")
    var our_qpos = InlineArray[Float64, NV](fill=0.0)
    var mj_qpos_arr = InlineArray[Float64, NV](fill=0.0)
    for i in range(NQ):
        var our_val = Float64(data.qpos[i])
        var mj_val = Float64(py=mj_qpos[i])
        our_qpos[i] = our_val
        mj_qpos_arr[i] = mj_val
        print(
            "  qpos[" + String(i) + "] ours=",
            our_val,
            " mj=",
            mj_val,
            " err=",
            abs(our_val - mj_val),
        )

    print()
    print("--- qvel after " + String(num_steps) + " steps ---")
    var our_qvel = InlineArray[Float64, NV](fill=0.0)
    var mj_qvel_arr = InlineArray[Float64, NV](fill=0.0)
    for i in range(NV):
        var our_val = Float64(data.qvel[i])
        var mj_val = Float64(py=mj_qvel[i])
        our_qvel[i] = our_val
        mj_qvel_arr[i] = mj_val
        print(
            "  qvel[" + String(i) + "] ours=",
            our_val,
            " mj=",
            mj_val,
            " err=",
            abs(our_val - mj_val),
        )


# =============================================================================
# Main
# =============================================================================


def main() raises:
    print("Hopper RK4 Forces Diagnostic")
    print("HopperModel.CONE_TYPE =", HopperModel.CONE_TYPE)
    print("ConeType.PYRAMIDAL =", ConeType.PYRAMIDAL)
    print("ConeType.ELLIPTIC =", ConeType.ELLIPTIC)
    print()

    # Default standing pose, zero actions (isolates gravity + contacts)
    var qpos_default = InlineArray[Float64, NQ](fill=0.0)
    var qvel_zero = InlineArray[Float64, NV](fill=0.0)
    var actions_zero = InlineArray[Float64, ACTION_DIM](fill=0.0)

    diagnose_forces(
        "Default standing, zero actions",
        qpos_default,
        qvel_zero,
        actions_zero,
    )

    # Single RK4 step, zero actions
    diagnose_rk4_step(
        "Default standing, zero actions, 1 step",
        qpos_default,
        qvel_zero,
        actions_zero,
        1,
    )

    # 4 RK4 steps = 1 frame_skip, zero actions
    diagnose_rk4_step(
        "Default standing, zero actions, 4 steps (1 frame_skip)",
        qpos_default,
        qvel_zero,
        actions_zero,
        4,
    )

    # Default standing with moderate actions
    var actions_mod = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions_mod[0] = 0.5  # thigh
    actions_mod[1] = -0.3  # leg
    actions_mod[2] = 0.2  # foot

    diagnose_forces(
        "Default standing, moderate actions",
        qpos_default,
        qvel_zero,
        actions_mod,
    )

    diagnose_rk4_step(
        "Default standing, moderate actions, 1 step",
        qpos_default,
        qvel_zero,
        actions_mod,
        1,
    )

    diagnose_rk4_step(
        "Default standing, moderate actions, 4 steps (1 frame_skip)",
        qpos_default,
        qvel_zero,
        actions_mod,
        4,
    )

    # Low rootz (contacts)
    var qpos_low = InlineArray[Float64, NQ](fill=0.0)
    qpos_low[1] = -0.8  # low rootz

    diagnose_forces(
        "Low rootz, zero actions",
        qpos_low,
        qvel_zero,
        actions_zero,
    )

    diagnose_rk4_step(
        "Low rootz, zero actions, 1 step",
        qpos_low,
        qvel_zero,
        actions_zero,
        1,
    )
