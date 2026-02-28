"""Diagnostic test: Compare intermediate values between our ImplicitFast and MuJoCo.

Pinpoints where the 1-5% velocity drift originates by comparing:
1. M_hat diagonal
2. qfrc_smooth (f_net)
3. qacc_unconstrained
4. MuJoCo Euler vs MuJoCo ImplicitFast (to check if drift is inherent)

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_implicitfast_diagnostics.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.integrator.implicit_fast_integrator import ImplicitFastIntegrator
from physics3d.solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
)
from physics3d.dynamics.bias_forces import compute_bias_forces_rne
from physics3d.dynamics.jacobian import compute_cdof, compute_composite_inertia
from physics3d.collision.contact_detection import detect_contacts
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
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM
comptime M_SIZE = NV * NV
comptime V_SIZE = NV
comptime CDOF_SIZE = NV * 6
comptime CRB_SIZE = NBODY * 10


fn main() raises:
    print("=" * 70)
    print("ImplicitFast Diagnostics: Intermediate Value Comparison")
    print("=" * 70)

    # =====================================================================
    # Config: "Moving with actions" — the one that shows 1.9% drift
    # =====================================================================
    var qpos_init = InlineArray[Float64, NQ](fill=0.0)
    qpos_init[1] = 1.5  # high (no contacts)
    qpos_init[2] = 0.1  # slight pitch
    qpos_init[3] = -0.3  # bthigh
    qpos_init[6] = 0.4  # fthigh
    var qvel_init = InlineArray[Float64, NV](fill=0.0)
    qvel_init[0] = 2.0  # rootx vel
    qvel_init[2] = 0.5  # rooty vel
    qvel_init[3] = -1.0  # bthigh vel
    qvel_init[6] = 1.2  # fthigh vel
    var actions_init = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions_init[0] = 1.0
    actions_init[1] = -0.5
    actions_init[3] = 1.0
    actions_init[4] = -0.5

    # =====================================================================
    # Part 1: MuJoCo reference — both Euler AND ImplicitFast
    # =====================================================================
    print("\n--- Part 1: MuJoCo Euler vs MuJoCo ImplicitFast ---")

    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )

    # --- MuJoCo Euler ---
    var mj_model_e = mujoco.MjModel.from_xml_path(xml_path)
    mj_model_e.opt.cone = 1  # elliptic
    mj_model_e.opt.solver = 2  # Newton
    mj_model_e.opt.integrator = 0  # Euler
    var mj_data_e = mujoco.MjData(mj_model_e)
    for i in range(NQ):
        mj_data_e.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data_e.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data_e.ctrl[i] = actions_init[i]
    mujoco.mj_step(mj_model_e, mj_data_e)
    var mj_qvel_euler = mj_data_e.qvel.flatten().tolist()

    # --- MuJoCo ImplicitFast ---
    var mj_model_if = mujoco.MjModel.from_xml_path(xml_path)
    mj_model_if.opt.cone = 1
    mj_model_if.opt.solver = 2
    mj_model_if.opt.integrator = 2  # ImplicitFast
    var mj_data_if = mujoco.MjData(mj_model_if)
    for i in range(NQ):
        mj_data_if.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data_if.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data_if.ctrl[i] = actions_init[i]

    # Run sub-steps to get intermediate values
    mujoco.mj_forward(mj_model_if, mj_data_if)

    var mj_qfrc_smooth = mj_data_if.qfrc_smooth.flatten().tolist()
    var mj_qacc = mj_data_if.qacc.flatten().tolist()
    var mj_qfrc_passive = mj_data_if.qfrc_passive.flatten().tolist()
    var mj_qfrc_bias = mj_data_if.qfrc_bias.flatten().tolist()
    var mj_qfrc_actuator = mj_data_if.qfrc_actuator.flatten().tolist()

    # Now step
    mujoco.mj_step(mj_model_if, mj_data_if)
    var mj_qvel_if = mj_data_if.qvel.flatten().tolist()

    print("  MuJoCo Euler vs ImplicitFast qvel comparison:")
    var max_euler_if_diff: Float64 = 0.0
    for i in range(NV):
        var v_e = Float64(py=mj_qvel_euler[i])
        var v_if = Float64(py=mj_qvel_if[i])
        var diff = abs(v_e - v_if)
        if diff > max_euler_if_diff:
            max_euler_if_diff = diff
        if diff > 1e-10:
            print(
                "    qvel[", i, "]: Euler=", v_e, " IF=", v_if, " diff=", diff
            )
    print("  Max Euler vs IF diff:", max_euler_if_diff)
    if max_euler_if_diff < 1e-10:
        print("  => MuJoCo Euler and ImplicitFast are IDENTICAL (no contacts)")
    else:
        print("  => MuJoCo Euler and ImplicitFast DIFFER!")

    # =====================================================================
    # Part 2: Our engine — compute intermediate values manually
    # =====================================================================
    print("\n--- Part 2: Our engine intermediate values ---")

    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, ConeType.ELLIPTIC, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions_init[i])

    # Apply actuator forces
    for i in range(NV):
        data.qfrc[i] = Scalar[DTYPE](0)
    HalfCheetahModel.apply_actions(data, action_list)

    # Replicate ImplicitFast step manually
    var dt = model.timestep

    # FK
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    # Contact detection (should find 0 contacts at height 1.5)
    detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )
    print("  Contacts detected:", Int(data.num_contacts))

    # cdof
    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)

    # CRB
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model, data, crb)

    # Mass matrix (raw CRBA — no armature, no dt*damp)
    var M_raw = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_raw.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(model, data, cdof, crb, M_raw)

    print("\n  M_raw diagonal (CRBA only, no armature):")
    print("    ", end="")
    for i in range(NV):
        print(Float64(M_raw[i * NV + i]), " ", end="")
    print()

    # Copy M for M_hat computation
    var M_hat = List[Scalar[DTYPE]](capacity=M_SIZE)
    for i in range(M_SIZE):
        M_hat.append(M_raw[i])

    # Add armature + dt*damp to diagonal (ImplicitFast M_hat)
    print("\n  Joint damping/armature values:")
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        var damp = joint.damping
        var diag_add = arm + dt * damp
        if diag_add > Scalar[DTYPE](0):
            print(
                "    joint",
                j,
                " dof=",
                Int(dof_adr),
                " arm=",
                Float64(arm),
                " damp=",
                Float64(damp),
                " dt*damp=",
                Float64(dt * damp),
            )
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                M_hat[(dof_adr + d) * NV + (dof_adr + d)] = (
                    M_hat[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
                )
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                M_hat[(dof_adr + d) * NV + (dof_adr + d)] = (
                    M_hat[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
                )
        else:
            M_hat[dof_adr * NV + dof_adr] = (
                M_hat[dof_adr * NV + dof_adr] + diag_add
            )

    print("\n  M_hat diagonal (M + arm + dt*damp):")
    print("    ", end="")
    for i in range(NV):
        print(Float64(M_hat[i * NV + i]), " ", end="")
    print()

    # Bias forces
    var bias = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        bias.append(Scalar[DTYPE](0))
    compute_bias_forces_rne(model, data, cdof, bias)

    # f_net = qfrc - bias
    var f_net = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(NV):
        f_net.append(data.qfrc[i] - bias[i])

    # Add damping forces
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

    # Add stiffness forces
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

    # LDL factorize M_hat and solve for qacc
    var L = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[DTYPE](0))
    var D = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D.append(Scalar[DTYPE](0))
    ldl_factor[DTYPE, NV](M_hat, L, D)

    var qacc = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        qacc.append(Scalar[DTYPE](0))
    ldl_solve[DTYPE, NV](L, D, f_net, qacc)

    # =====================================================================
    # Part 3: Compare intermediate values
    # =====================================================================
    print("\n--- Part 3: Compare intermediate values ---")

    # Compare qfrc (actuator forces)
    print("\n  Actuator forces (our qfrc vs MJ qfrc_actuator):")
    for i in range(NV):
        var ours = Float64(data.qfrc[i])
        var mj = Float64(py=mj_qfrc_actuator[i])
        var diff = abs(ours - mj)
        if diff > 1e-12 or ours != 0 or mj != 0:
            print(
                "    dof[", i, "]: ours=", ours, " mj=", mj, " diff=", diff
            )

    # Compare bias forces
    print("\n  Bias forces (our bias vs MJ qfrc_bias):")
    var max_bias_diff: Float64 = 0.0
    for i in range(NV):
        var ours = Float64(bias[i])
        var mj = Float64(py=mj_qfrc_bias[i])
        var diff = abs(ours - mj)
        if diff > max_bias_diff:
            max_bias_diff = diff
        if diff > 1e-6:
            print(
                "    dof[", i, "]: ours=", ours, " mj=", mj, " diff=", diff
            )
    print("    Max bias diff:", max_bias_diff)

    # Compare passive forces
    print("\n  Passive forces (our vs MJ qfrc_passive):")
    for i in range(NV):
        var mj = Float64(py=mj_qfrc_passive[i])
        # Our passive = -damp*qvel - stiff*(q-qref)
        var our_passive: Float64 = 0.0
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = Int(joint.dof_adr)
            if dof_adr == i:
                var damp = Float64(joint.damping)
                var stiff = Float64(joint.stiffness)
                var sref = Float64(joint.springref)
                if damp > 0:
                    our_passive -= damp * Float64(data.qvel[i])
                if stiff > 0:
                    our_passive -= stiff * (
                        Float64(data.qpos[Int(joint.qpos_adr)]) - sref
                    )
        var diff = abs(our_passive - mj)
        if diff > 1e-6 or abs(mj) > 1e-10:
            print(
                "    dof[",
                i,
                "]: ours=",
                our_passive,
                " mj=",
                mj,
                " diff=",
                diff,
            )

    # Compare f_net vs qfrc_smooth
    print("\n  f_net vs MJ qfrc_smooth:")
    var max_fnet_diff: Float64 = 0.0
    for i in range(NV):
        var ours = Float64(f_net[i])
        var mj = Float64(py=mj_qfrc_smooth[i])
        var diff = abs(ours - mj)
        if diff > max_fnet_diff:
            max_fnet_diff = diff
        print(
            "    dof[", i, "]: ours=", ours, " mj=", mj, " diff=", diff
        )
    print("    Max f_net diff:", max_fnet_diff)

    # Compare qacc
    print("\n  qacc (unconstrained) comparison:")
    var max_qacc_diff: Float64 = 0.0
    for i in range(NV):
        var ours = Float64(qacc[i])
        var mj = Float64(py=mj_qacc[i])
        var diff = abs(ours - mj)
        if diff > max_qacc_diff:
            max_qacc_diff = diff
        var rel: Float64 = 0.0
        if abs(mj) > 1e-10:
            rel = diff / abs(mj)
        print(
            "    dof[",
            i,
            "]: ours=",
            ours,
            " mj=",
            mj,
            " abs=",
            diff,
            " rel=",
            rel,
        )
    print("    Max qacc diff:", max_qacc_diff)

    # Compare final qvel
    print("\n  Final qvel (v_old + dt*qacc):")
    var max_qvel_diff: Float64 = 0.0
    for i in range(NV):
        var v_new = Float64(data.qvel[i]) + Float64(qacc[i]) * Float64(dt)
        var mj = Float64(py=mj_qvel_if[i])
        var diff = abs(v_new - mj)
        if diff > max_qvel_diff:
            max_qvel_diff = diff
        var rel: Float64 = 0.0
        if abs(mj) > 1e-10:
            rel = diff / abs(mj)
        if diff > 1e-6:
            print(
                "    dof[",
                i,
                "]: ours=",
                v_new,
                " mj=",
                mj,
                " abs=",
                diff,
                " rel=",
                rel,
            )
    print("    Max qvel diff:", max_qvel_diff)

    # =====================================================================
    # Part 4: Check active constraints and mass matrix
    # =====================================================================
    print("\n--- Part 4: Active constraints and mass matrix ---")

    # Re-run mj_forward to get constraint info
    var mj_model_d = mujoco.MjModel.from_xml_path(xml_path)
    mj_model_d.opt.cone = 1
    mj_model_d.opt.solver = 2
    mj_model_d.opt.integrator = 2  # implicitfast
    var mj_data_d = mujoco.MjData(mj_model_d)
    for i in range(NQ):
        mj_data_d.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data_d.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data_d.ctrl[i] = actions_init[i]
    mujoco.mj_forward(mj_model_d, mj_data_d)

    var nefc = Int(py=mj_data_d.nefc)
    var ncon = Int(py=mj_data_d.ncon)
    print("  MuJoCo nefc (total active constraints):", nefc)
    print("  MuJoCo ncon (contacts):", ncon)

    # Check constraint types
    if nefc > 0:
        var efc_type = mj_data_d.efc_type.flatten().tolist()
        for c in range(nefc):
            var ct = Int(py=efc_type[c])
            var type_name: String
            if ct == 0:
                type_name = "EQUALITY"
            elif ct == 1:
                type_name = "FRICTION_DOF"
            elif ct == 2:
                type_name = "LIMIT"
            elif ct == 3:
                type_name = "CONTACT_PYRAMIDAL"
            elif ct == 4:
                type_name = "CONTACT_ELLIPTIC"
            else:
                type_name = "UNKNOWN(" + String(ct) + ")"
            print("    constraint", c, ":", type_name)

    # Verify qacc vs qvel change
    print("\n  Verifying MuJoCo qacc vs actual qvel change:")
    var mj_qacc_d = mj_data_d.qacc.flatten().tolist()
    var mj_qacc_smooth = mj_data_d.qacc_smooth.flatten().tolist()
    print("    qacc (constrained) vs qacc_smooth (unconstrained):")
    for i in range(NV):
        var qa = Float64(py=mj_qacc_d[i])
        var qas = Float64(py=mj_qacc_smooth[i])
        var diff = abs(qa - qas)
        if diff > 1e-10:
            print(
                "      dof[", i, "]: qacc=", qa, " qacc_smooth=", qas,
                " diff=", diff
            )
    print("    (Only showing diffs > 1e-10)")

    # Now check: is qacc from mj_forward the same as qacc_smooth?
    print("\n  qacc_smooth (=M_hat^-1 * qfrc_smooth, NO constraint solver):")
    for i in range(NV):
        var qas = Float64(py=mj_qacc_smooth[i])
        print("    dof[", i, "]:", qas)

    # Get dense M using mj_fullM with correct API
    var nv = Int(py=mj_model_d.nv)
    var mj_M_dense = np.zeros(nv * nv).reshape(nv, nv)
    mujoco.mj_fullM(mj_model_d, mj_M_dense, mj_data_d.qM)
    print("\n  MuJoCo M diagonal (from mj_fullM on qM):")
    var any_nonzero = False
    for i in range(NV):
        var v = Float64(py=mj_M_dense[i][i])
        if abs(v) > 1e-10:
            any_nonzero = True
        print("    M[", i, ",", i, "] =", v)
    if not any_nonzero:
        print("  WARNING: mj_fullM returned all zeros — trying alternative")
        # Try accessing M directly as flat array and reconstructing
        var mj_M_raw = mj_data_d.qM.flatten().tolist()
        var mj_M_len = Int(py=Python.evaluate("len")(mj_M_raw))
        print("  qM length:", mj_M_len, "(expected nC sparse entries)")

        # Try with d.M (band format in MuJoCo 3.x)
        try:
            var mj_M_band = mj_data_d.M.flatten().tolist()
            var mj_M_band_len = Int(py=Python.evaluate("len")(mj_M_band))
            print("  d.M (band) length:", mj_M_band_len)
        except:
            print("  d.M not available (older MuJoCo)")

    # Print armature values from MuJoCo
    print("\n  MuJoCo dof_armature:")
    var mj_arm = mj_model_d.dof_armature.flatten().tolist()
    for i in range(NV):
        var a = Float64(py=mj_arm[i])
        if a > 0:
            print("    dof[", i, "]:", a)

    # Print damping values from MuJoCo
    print("  MuJoCo dof_damping:")
    var mj_damp = mj_model_d.dof_damping.flatten().tolist()
    for i in range(NV):
        var d_val = Float64(py=mj_damp[i])
        if d_val > 0:
            print("    dof[", i, "]:", d_val)

    # =====================================================================
    # Part 5: Verify M_hat by computing M_hat * qacc_mj vs f_mj
    # =====================================================================
    print("\n--- Part 5: M_hat verification ---")

    # Our M_hat includes armature + dt*damp.
    # MuJoCo's M_hat = M_mj + dt*damp = (M_raw + arm) + dt*damp
    # Since we add arm+dt*damp to M_raw, our M_hat should = MuJoCo's M_hat.
    #
    # Test: does our M_hat * mj_qacc_smooth = mj_qfrc_smooth?
    print("  M_hat_ours * qacc_mj vs f_mj (qfrc_smooth):")
    var mj_qacc_s = mj_data_d.qacc_smooth.flatten().tolist()
    var mj_fsmooth = mj_data_d.qfrc_smooth.flatten().tolist()
    var max_residual: Float64 = 0.0
    for i in range(NV):
        var Mq: Float64 = 0.0
        for j in range(NV):
            Mq += Float64(M_hat[i * NV + j]) * Float64(py=mj_qacc_s[j])
        var f_mj = Float64(py=mj_fsmooth[i])
        var residual = abs(Mq - f_mj)
        if residual > max_residual:
            max_residual = residual
        if residual > 0.01:
            print(
                "    row", i, ": M_hat*qacc_mj=", Mq, " f_mj=", f_mj,
                " residual=", residual
            )
    print("  Max residual (M_hat_ours * qacc_mj - f_mj):", max_residual)
    if max_residual > 1.0:
        print("  => LARGE residual — M_hat matrices DIFFER!")
    else:
        print("  => M_hat matrices match (residual from f_net rounding)")

    # Also check: does M_hat_ours * qacc_ours = f_ours?
    print("\n  Sanity check: M_hat_ours * qacc_ours vs f_ours:")
    var max_residual2: Float64 = 0.0
    for i in range(NV):
        var Mq2: Float64 = 0.0
        for j in range(NV):
            Mq2 += Float64(M_hat[i * NV + j]) * Float64(qacc[j])
        var f_o = Float64(f_net[i])
        var residual2 = abs(Mq2 - f_o)
        if residual2 > max_residual2:
            max_residual2 = residual2
    print("  Max residual (M_hat_ours * qacc_ours - f_ours):", max_residual2)

    # Compare full M matrices (ours + armature vs MuJoCo)
    print("\n  Full M comparison (our M_raw + armature vs MuJoCo M):")
    var max_M_diff: Float64 = 0.0
    var worst_i = 0
    var worst_j2 = 0
    for i in range(NV):
        for j in range(NV):
            var ours_val = Float64(M_raw[i * NV + j])
            # Add armature to our diagonal
            if i == j:
                for k in range(model.num_joints):
                    var jt = model.joints[k]
                    if Int(jt.dof_adr) == i:
                        ours_val += Float64(jt.armature)
            var mj_val = Float64(py=mj_M_dense[i][j])
            var diff = abs(ours_val - mj_val)
            if diff > max_M_diff:
                max_M_diff = diff
                worst_i = i
                worst_j2 = j
    print("  Max M diff:", max_M_diff, "at [", worst_i, ",", worst_j2, "]")
    if max_M_diff > 1e-4:
        print("  => M matrices DIFFER significantly!")
        print("  Showing all diffs > 1e-6:")
        for i in range(NV):
            for j in range(NV):
                var ours_val = Float64(M_raw[i * NV + j])
                if i == j:
                    for k in range(model.num_joints):
                        var jt = model.joints[k]
                        if Int(jt.dof_adr) == i:
                            ours_val += Float64(jt.armature)
                var mj_val = Float64(py=mj_M_dense[i][j])
                var diff = abs(ours_val - mj_val)
                if diff > 1e-6:
                    print(
                        "    M[", i, ",", j, "]: ours=", ours_val,
                        " mj=", mj_val, " diff=", diff
                    )

    # =====================================================================
    # Part 6: Reconstruct M_hat from MuJoCo's M, solve with numpy
    # =====================================================================
    print("\n--- Part 6: Solve with MuJoCo's M + dt*damp via numpy ---")

    # Construct M_hat from MuJoCo's dense M (from mj_fullM)
    var mj_M_hat = mj_M_dense.copy()
    var mj_damp_arr = mj_model_d.dof_damping.flatten()
    var dt_val = Float64(py=mj_model_d.opt.timestep)
    print("  MuJoCo timestep:", dt_val)
    for i in range(NV):
        var d_val = Float64(py=mj_damp_arr[i])
        mj_M_hat[i][i] = mj_M_hat[i][i] + dt_val * d_val

    # Solve qacc = M_hat^-1 * qfrc_smooth using numpy
    var mj_f_arr = np.array(mj_fsmooth)
    var numpy_qacc = np.linalg.solve(mj_M_hat, mj_f_arr)

    print("  qacc comparison (numpy solve with MJ's M vs MJ actual vs ours):")
    for i in range(NV):
        var np_q = Float64(py=numpy_qacc[i])
        var mj_q = Float64(py=mj_qacc_s[i])
        var our_q = Float64(qacc[i])
        print(
            "    dof[", i, "]: numpy=", np_q, " mj=", mj_q, " ours=", our_q,
            " np-mj=", abs(np_q - mj_q), " np-ours=", abs(np_q - our_q),
        )

    # If numpy matches ours but not MuJoCo's, then mj_fullM(qM) ≠ d->M
    print("\n  If numpy≈ours but ≠mj: mj_fullM(qM) differs from d->M used internally")
    print("  If numpy≈mj but ≠ours: our M computation differs")

    # =====================================================================
    # Part 7: Reconstruct dense M from d.M using CSR metadata
    # =====================================================================
    print("\n--- Part 7: Reconstruct dense M from d.M (CSR format) ---")

    try:
        var M_csr = mj_data_d.M.flatten().tolist()
        var M_rownnz = mj_model_d.M_rownnz.flatten().tolist()
        var M_rowadr = mj_model_d.M_rowadr.flatten().tolist()
        var M_colind = mj_model_d.M_colind.flatten().tolist()

        print("  CSR metadata:")
        print("    M_rownnz:", M_rownnz)
        print("    M_rowadr:", M_rowadr)
        print("    M_colind:", M_colind)
        print("    d.M entries:", M_csr)

        # Reconstruct dense matrix from CSR lower-triangular
        var M_from_csr = np.zeros(nv * nv).reshape(nv, nv)
        for row in range(NV):
            var nnz = Int(py=M_rownnz[row])
            var adr = Int(py=M_rowadr[row])
            for k in range(nnz):
                var col = Int(py=M_colind[adr + k])
                var val = Float64(py=M_csr[adr + k])
                M_from_csr[row][col] = val
                M_from_csr[col][row] = val  # symmetric

        print("\n  Dense M from CSR vs from mj_fullM(qM):")
        var max_csr_diff: Float64 = 0.0
        for i in range(NV):
            for j in range(NV):
                var csr_val = Float64(py=M_from_csr[i][j])
                var qm_val = Float64(py=mj_M_dense[i][j])
                var diff = abs(csr_val - qm_val)
                if diff > max_csr_diff:
                    max_csr_diff = diff
                if diff > 1e-10:
                    print(
                        "    M[", i, ",", j, "]: CSR=", csr_val, " qM=",
                        qm_val, " diff=", diff,
                    )
        print("  Max CSR vs qM diff:", max_csr_diff)

        if max_csr_diff < 1e-6:
            print("  => d.M and d.qM represent the SAME matrix")
        else:
            print("  => d.M and d.qM DIFFER!")

        # Solve with CSR-reconstructed M
        var M_hat_csr = M_from_csr.copy()
        for i in range(NV):
            var d_val2 = Float64(py=mj_damp_arr[i])
            M_hat_csr[i][i] = M_hat_csr[i][i] + dt_val * d_val2
        var numpy_qacc_csr = np.linalg.solve(M_hat_csr, mj_f_arr)

        print("\n  qacc from CSR-M solve vs MuJoCo:")
        var max_csr_qacc_diff: Float64 = 0.0
        for i in range(NV):
            var np_q = Float64(py=numpy_qacc_csr[i])
            var mj_q = Float64(py=mj_qacc_s[i])
            var diff = abs(np_q - mj_q)
            if diff > max_csr_qacc_diff:
                max_csr_qacc_diff = diff
            print(
                "    dof[", i, "]: CSR_solve=", np_q, " mj=", mj_q,
                " diff=", diff,
            )
        print("  Max CSR qacc diff:", max_csr_qacc_diff)

        # Also check qH (the actual matrix MuJoCo factorizes)
        print("\n  d.qH (factored implicitfast matrix):")
        var qH_data = mj_data_d.qH.flatten().tolist()
        print("    qH entries:", qH_data)

        # Reconstruct dense qH from CSR
        var qH_dense = np.zeros(nv * nv).reshape(nv, nv)
        for row in range(NV):
            var nnz = Int(py=M_rownnz[row])
            var adr = Int(py=M_rowadr[row])
            for k in range(nnz):
                var col = Int(py=M_colind[adr + k])
                var val = Float64(py=qH_data[adr + k])
                qH_dense[row][col] = val
                qH_dense[col][row] = val

        # qH is already factored in-place, so reconstruct pre-factored
        # by building M_hat = M_csr + dt*damp manually
        print("\n  Pre-factor qH (= M + dt*damp) from CSR-M:")
        var M_hat_prefactor = M_from_csr.copy()
        for i in range(NV):
            var d_val3 = Float64(py=mj_damp_arr[i])
            M_hat_prefactor[i][i] = M_hat_prefactor[i][i] + dt_val * d_val3
        print("    Diagonal:")
        for i in range(NV):
            print(
                "    M_hat[", i, ",", i, "] =",
                Float64(py=M_hat_prefactor[i][i]),
            )

        # Also check qDeriv directly
        print("\n  d.qDeriv:")
        var qDeriv = mj_data_d.qDeriv.flatten().tolist()
        var qDeriv_len = Int(py=Python.evaluate("len")(qDeriv))
        print("    qDeriv length:", qDeriv_len)
        print("    qDeriv entries:", qDeriv)

        # Reconstruct dense qDeriv using D format metadata
        var D_rownnz = mj_model_d.D_rownnz.flatten().tolist()
        var D_rowadr = mj_model_d.D_rowadr.flatten().tolist()
        var D_colind = mj_model_d.D_colind.flatten().tolist()
        print("    D_rownnz:", D_rownnz)
        print("    D_rowadr:", D_rowadr)

        var qDeriv_dense = np.zeros(nv * nv).reshape(nv, nv)
        for row in range(NV):
            var nnz = Int(py=D_rownnz[row])
            var adr = Int(py=D_rowadr[row])
            for k in range(nnz):
                var col = Int(py=D_colind[adr + k])
                var val = Float64(py=qDeriv[adr + k])
                qDeriv_dense[row][col] = val

        print("    qDeriv dense (should be diagonal with -damping):")
        for i in range(NV):
            for j in range(NV):
                var v = Float64(py=qDeriv_dense[i][j])
                if abs(v) > 1e-10:
                    print("      qDeriv[", i, ",", j, "] =", v)

        # Final: build M_hat = M - dt*qDeriv, solve, compare
        print("\n  Final solve: M_hat = M_csr - dt*qDeriv, numpy.solve:")
        var M_hat_final = M_from_csr - dt_val * qDeriv_dense
        # Symmetrize (qDeriv_dense may not be symmetric)
        var M_hat_sym = (M_hat_final + M_hat_final.T) / 2.0
        var numpy_qacc_final = np.linalg.solve(M_hat_sym, mj_f_arr)

        print("  qacc: final_solve vs MuJoCo vs ours:")
        for i in range(NV):
            var np_q = Float64(py=numpy_qacc_final[i])
            var mj_q = Float64(py=mj_qacc_s[i])
            var our_q = Float64(qacc[i])
            print(
                "    dof[", i, "]: final=", np_q, " mj=", mj_q,
                " ours=", our_q,
                " final-mj=", abs(np_q - mj_q),
            )

    except e:
        print("  Error:", e)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
