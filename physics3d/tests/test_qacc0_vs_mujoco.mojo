"""Test Unconstrained Acceleration (qacc0) against MuJoCo reference.

Compares our qacc0 = M_arm^{-1} * f_net (acceleration before constraints)
with MuJoCo's qacc0 for the HalfCheetah model at multiple configurations.

This validates that mass matrix, bias forces, actuator forces, and passive
forces (damping, stiffness) combine correctly into the right acceleration.
It's the last "pre-solver" check — if qacc0 is wrong, no solver can produce
correct results.

MuJoCo reference: mj_data.qacc_smooth (available after mj_forward)
  qacc_smooth = (M + armature)^{-1} * qfrc_smooth
  qfrc_smooth = -qfrc_bias + qfrc_passive + qfrc_actuator  (MuJoCo 3.x convention)

Our engine equivalent:
  M_arm = CRBA mass matrix + armature (NO implicit dt*D)
  f_net = qfrc_actuator - bias_rne - damping*qvel - stiffness*(qpos-springref)
  qacc0 = M_arm^{-1} * f_net  (via LDL solve)

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_qacc0_vs_mujoco.mojo
"""

from testing import assert_true, TestSuite
from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.jacobian import compute_cdof, compute_composite_inertia
from physics3d.dynamics.bias_forces import compute_bias_forces_rne
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
)
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()

# Tolerance — qacc0 accumulates errors from M, bias, and forces
comptime ABS_TOL: Float64 = 1e-4
comptime REL_TOL: Float64 = 1e-3


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_qacc0(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    qvel_values: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
) raises:
    """Compute qacc0 in both engines with identical state, compare."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model, data)
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_values[i])

    # 1. FK + body velocities + cdof
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)

    # 2. Bias forces (RNE)
    var bias = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        bias.append(Scalar[DTYPE](0))
    compute_bias_forces_rne(model, data, cdof, bias)

    # 3. Mass matrix (CRBA)
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model, data, crb)

    var M = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(model, data, cdof, crb, M)

    # 4. Add armature ONLY (no dt*D — MuJoCo qacc0 uses M+arm, not M+arm+dt*D)
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

    # 5. Apply actuator forces
    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])
    HalfCheetahModel.apply_actions[DTYPE](data, action_list)

    # 6. Compute f_net = qfrc - bias (matches our integrator convention)
    var f_net = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(NV):
        f_net.append(data.qfrc[i] - bias[i])

    # 7. Add passive forces: damping + stiffness + frictionloss
    # (same code as euler_integrator.mojo but without implicit dt*D in M)
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var damp = joint.damping
        if damp > Scalar[DTYPE](0):
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr + d] -= damp * data.qvel[dof_adr + d]
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr + d] -= damp * data.qvel[dof_adr + d]
            else:
                f_net[dof_adr] -= damp * data.qvel[dof_adr]

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
                    f_net[dof_adr + d] -= stiff * (
                        data.qpos[qpos_adr + d] - sref
                    )
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr + d] -= stiff * (
                        data.qpos[qpos_adr + d] - sref
                    )
            else:
                f_net[dof_adr] -= stiff * (data.qpos[qpos_adr] - sref)
        if floss > Scalar[DTYPE](0):
            comptime VEL_THRESH: Scalar[DTYPE] = 1e-4
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    var v = data.qvel[dof_adr + d]
                    if v > VEL_THRESH:
                        f_net[dof_adr + d] -= floss
                    elif v < -VEL_THRESH:
                        f_net[dof_adr + d] += floss
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    var v = data.qvel[dof_adr + d]
                    if v > VEL_THRESH:
                        f_net[dof_adr + d] -= floss
                    elif v < -VEL_THRESH:
                        f_net[dof_adr + d] += floss
            else:
                var v = data.qvel[dof_adr]
                if v > VEL_THRESH:
                    f_net[dof_adr] -= floss
                elif v < -VEL_THRESH:
                    f_net[dof_adr] += floss

    # 8. LDL factorize M_arm and solve for qacc0
    var L = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[DTYPE](0))
    var D_ldl = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D_ldl.append(Scalar[DTYPE](0))
    ldl_factor[DTYPE, NV](M, L, D_ldl)

    var qacc0 = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        qacc0.append(Scalar[DTYPE](0))
    ldl_solve[DTYPE, NV](L, D_ldl, f_net, qacc0)

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_values[i]
    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    # mj_forward runs: FK + collision + constraint setup + solver
    # After mj_forward: qacc_smooth is available (unconstrained acceleration)
    mujoco.mj_forward(mj_model, mj_data)

    var mj_qacc0_flat = mj_data.qacc_smooth.flatten().tolist()

    # Also grab intermediate values for debugging
    var mj_bias_flat = mj_data.qfrc_bias.flatten().tolist()
    var mj_passive_flat = mj_data.qfrc_passive.flatten().tolist()
    var mj_actuator_flat = mj_data.qfrc_actuator.flatten().tolist()
    var mj_smooth_flat = mj_data.qfrc_smooth.flatten().tolist()

    # === Compare qacc0 ===
    var all_pass = True
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for i in range(NV):
        var our_val = Float64(qacc0[i])
        var mj_val = Float64(py=mj_qacc0_flat[i])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        var ok = abs_err < ABS_TOL or rel_err < REL_TOL
        if not ok:
            print(
                "  FAIL qacc0[", i, "]",
                " ours=", our_val,
                " mj=", mj_val,
                " abs_err=", abs_err,
                " rel_err=", rel_err,
            )
            fail_count += 1
            all_pass = False

    if all_pass:
        print("  ALL OK  max_abs_err=", max_abs_err, " max_rel_err=", max_rel_err)
    else:
        print(
            "  FAILED", fail_count, "elements  max_abs_err=", max_abs_err,
            " max_rel_err=", max_rel_err,
        )
        assert_true(False, "compare_qacc0 failed for: " + test_name)

    # Print qacc0 values
    print("  Our qacc0:", end="")
    for i in range(NV):
        print(" ", Float64(qacc0[i]), end="")
    print()
    print("  MJ  qacc0:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_qacc0_flat[i]), end="")
    print()

    # Print intermediate values for debugging
    print("  Our f_net:", end="")
    for i in range(NV):
        print(" ", Float64(f_net[i]), end="")
    print()
    print("  MJ smooth:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_smooth_flat[i]), end="")
    print()

    print("  Our bias: ", end="")
    for i in range(NV):
        print(" ", Float64(bias[i]), end="")
    print()
    print("  MJ  bias: ", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_bias_flat[i]), end="")
    print()

    print("  MJ passive:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_passive_flat[i]), end="")
    print()

    print("  Our qfrc: ", end="")
    for i in range(NV):
        print(" ", Float64(data.qfrc[i]), end="")
    print()
    print("  MJ  actu: ", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_actuator_flat[i]), end="")
    print()


# =============================================================================
# Test cases
# =============================================================================


fn test_gravity_only() raises:
    """Default pose, zero vel, zero actions — qacc0 = M^{-1} * gravity_bias."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7  # rootz
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_qacc0("Gravity only (default pose)", qpos, qvel, actions)


fn test_with_actions() raises:
    """Default pose, zero vel, with actions — qacc0 includes actuator forces."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7  # rootz
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0  # bthigh (gear=120)
    actions[1] = -0.5  # bshin (gear=90)
    actions[2] = 0.3  # bfoot (gear=60)
    actions[3] = 1.0  # fthigh (gear=120)
    actions[4] = -0.5  # fshin (gear=60)
    actions[5] = 0.3  # ffoot (gear=30)
    compare_qacc0("With actions", qpos, qvel, actions)


fn test_nonzero_vel() raises:
    """Non-zero joints + velocity — qacc0 includes Coriolis + damping."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 1.0   # rootx
    qpos[1] = 0.7   # rootz
    qpos[2] = 0.1   # rooty
    qpos[3] = -0.3  # bthigh
    qpos[6] = 0.4   # fthigh
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0   # rootx vel (running)
    qvel[2] = 0.5   # rooty vel (pitching)
    qvel[3] = -1.0  # bthigh vel
    qvel[4] = 0.8   # bshin vel
    qvel[6] = 1.2   # fthigh vel
    qvel[7] = -0.6  # fshin vel
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_qacc0("Nonzero vel (Coriolis + damping)", qpos, qvel, actions)


fn test_full_combo() raises:
    """Nonzero joints + velocity + actions — tests everything together."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.7
    qpos[2] = 0.1
    qpos[3] = -0.3
    qpos[4] = 0.5
    qpos[6] = 0.4
    qpos[7] = -0.8
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0
    qvel[1] = -0.5
    qvel[2] = 0.5
    qvel[3] = -1.0
    qvel[4] = 0.8
    qvel[5] = -0.3
    qvel[6] = 1.2
    qvel[7] = -0.6
    qvel[8] = 0.4
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8
    actions[1] = -0.5
    actions[2] = 0.3
    actions[3] = 0.8
    actions[4] = -0.5
    actions[5] = 0.3
    compare_qacc0("Full combo (vel + actions)", qpos, qvel, actions)


fn test_ground_contact_pose() raises:
    """Ground contact pose — qacc0 should not be affected by contacts.
    (contacts affect qacc via solver, but qacc0 = unconstrained acceleration)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz low — touches ground
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_qacc0("Ground contact pose (qacc0 unaffected)", qpos, qvel, actions)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
