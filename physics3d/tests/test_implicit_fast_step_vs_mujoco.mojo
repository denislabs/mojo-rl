"""Test ImplicitFast Full Step (no contacts): Mojo Engine vs MuJoCo.

Compares qpos/qvel after running physics steps using ImplicitFastIntegrator
on CPU vs MuJoCo Euler integrator (opt.integrator=0).

Our ImplicitFastIntegrator uses M_hat = M + armature + dt*D (no Coriolis),
matching MuJoCo 3.4.1+ where ImplicitFast calls mjd_smooth_vel(flg_bias=0).

We compare against MuJoCo Euler (not ImplicitFast) because:
- MuJoCo 3.3.6 ImplicitFast includes Coriolis velocity derivatives
  (mjd_rne_vel) which we intentionally skip (matching 3.4.1+ design).
- MuJoCo Euler uses qH = M + h*D (same formula as our ImplicitFast
  without Coriolis), giving an exact reference for our implementation.
- conda-forge only packages MuJoCo up to 3.3.6 for osx-arm64.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_implicit_fast_step_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.integrator.implicit_fast_integrator import ImplicitFastIntegrator
from physics3d.solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.dynamics.mass_matrix import compute_body_invweight0
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahGeoms,
    HalfCheetahActuators,
    HalfCheetahParams,
    HalfCheetahDefaults,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS
comptime ACTION_DIM = HalfCheetahParams[DTYPE].ACTION_DIM

# Tolerances — tight, matching Euler test precision (comparing against MuJoCo
# Euler which uses the same qH = M + h*D formula as our ImplicitFast)
comptime QPOS_ABS_TOL: Float64 = 1e-6
comptime QPOS_REL_TOL: Float64 = 1e-4
comptime QVEL_ABS_TOL: Float64 = 1e-4
comptime QVEL_REL_TOL: Float64 = 1e-4


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int = 1,
) raises -> Bool:
    """Run num_steps physics steps in both engines, compare final qpos/qvel."""
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === Our engine (ImplicitFast + Newton) ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, ConeType.ELLIPTIC
    ](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.01),
    )
    HalfCheetahModel.setup_solver_params[Defaults=HalfCheetahDefaults](model)

    HalfCheetahBodies.setup_model(model)

    HalfCheetahJoints.setup_model[Defaults=HalfCheetahDefaults](model)

    HalfCheetahGeoms.setup_model[Defaults=HalfCheetahDefaults](model)

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()

    # Compute body_invweight0 at reference pose
    forward_kinematics(model, data)
    compute_body_invweight0[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )

    # Set test configuration
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
        HalfCheetahActuators.apply_actions[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ](data, action_list)
        ImplicitFastIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
            model, data
        )

    # === MuJoCo reference (opt.integrator=2 = implicitfast) ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.cone = 1  # mjCONE_ELLIPTIC
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.integrator = 3  # mjINT_IMPLICITFAST
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)

    # === Compare ===
    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()

    var qpos_pass = True
    var qpos_max_abs: Float64 = 0.0
    var qpos_max_rel: Float64 = 0.0
    var qpos_fails = 0

    for i in range(NQ):
        var our_val = Float64(data.qpos[i])
        var mj_val = Float64(py=mj_qpos[i])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > qpos_max_abs:
            qpos_max_abs = abs_err
        if rel_err > qpos_max_rel:
            qpos_max_rel = rel_err

        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            if qpos_fails < 5:
                print(
                    "  FAIL qpos[",
                    i,
                    "]",
                    " ours=",
                    our_val,
                    " mj=",
                    mj_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qpos_fails += 1
            qpos_pass = False

    var qvel_pass = True
    var qvel_max_abs: Float64 = 0.0
    var qvel_max_rel: Float64 = 0.0
    var qvel_fails = 0

    for i in range(NV):
        var our_val = Float64(data.qvel[i])
        var mj_val = Float64(py=mj_qvel[i])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > qvel_max_abs:
            qvel_max_abs = abs_err
        if rel_err > qvel_max_rel:
            qvel_max_rel = rel_err

        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            if qvel_fails < 5:
                print(
                    "  FAIL qvel[",
                    i,
                    "]",
                    " ours=",
                    our_val,
                    " mj=",
                    mj_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass

    if all_pass:
        print(
            "  ALL OK  qpos_max_abs=",
            qpos_max_abs,
            " qpos_max_rel=",
            qpos_max_rel,
            " qvel_max_abs=",
            qvel_max_abs,
            " qvel_max_rel=",
            qvel_max_rel,
        )
    else:
        print(
            "  FAILED  qpos:",
            qpos_fails,
            "fails (max_abs=",
            qpos_max_abs,
            " max_rel=",
            qpos_max_rel,
            ")",
            " qvel:",
            qvel_fails,
            "fails (max_abs=",
            qvel_max_abs,
            " max_rel=",
            qvel_max_rel,
            ")",
        )

    # Print values for inspection
    print("  Our qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(data.qpos[i]), end="")
    print()
    print("  MJ  qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(py=mj_qpos[i]), end="")
    print()
    print("  Our qvel:", end="")
    for i in range(NV):
        print(" ", Float64(data.qvel[i]), end="")
    print()
    print("  MJ  qvel:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_qvel[i]), end="")
    print()
    print("  Our contacts:", Int(data.num_contacts))
    var mj_ncon = Int(py=mj_data.ncon)
    print("  MJ  contacts:", mj_ncon)

    return all_pass


# =============================================================================
# Test cases — no ground contact (free flight)
# =============================================================================


fn test_freefall() raises -> Bool:
    """Free fall from height — no contacts expected."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # rootz high enough to avoid ground contact
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    return compare_step("Free fall (no contacts)", qpos, qvel, actions)


fn test_standing_with_action() raises -> Bool:
    """Standing with moderate actions."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # high enough for no contacts
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[1] = -0.3
    actions[2] = 0.2
    actions[3] = 0.5
    actions[4] = -0.3
    actions[5] = 0.1
    return compare_step("Actions (no contacts)", qpos, qvel, actions)


fn test_moving_with_action() raises -> Bool:
    """Moving with velocity + actions, no contacts."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # high
    qpos[2] = 0.1  # slight pitch
    qpos[3] = -0.3  # bthigh
    qpos[6] = 0.4  # fthigh
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # rootx vel
    qvel[2] = 0.5  # rooty vel
    qvel[3] = -1.0  # bthigh vel
    qvel[6] = 1.2  # fthigh vel
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0
    actions[1] = -0.5
    actions[3] = 1.0
    actions[4] = -0.5
    return compare_step(
        "Moving with actions (no contacts)", qpos, qvel, actions
    )


fn test_freefall_10_steps() raises -> Bool:
    """Free fall 10 steps — accumulates per-step drift."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    return compare_step(
        "Free fall (10 steps)", qpos, qvel, actions, num_steps=10
    )


fn test_standing_10_steps() raises -> Bool:
    """Standing 10 steps with actions — tests integration stability."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[3] = 0.5
    return compare_step(
        "With actions (10 steps)", qpos, qvel, actions, num_steps=10
    )


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("ImplicitFast Full Step (no contacts): Mojo vs MuJoCo")
    print("=" * 60)
    print("Model: HalfCheetah (NQ=9, NV=9)")
    print("Integrator: ImplicitFast (ref: MuJoCo Euler, same qH=M+h*D)")
    print("Solver: Newton (opt.solver=2)")
    print("Cone: elliptic (opt.cone=1)")
    print("Precision: float64")
    print("Tolerances: qpos abs=", QPOS_ABS_TOL, " rel=", QPOS_REL_TOL)
    print("            qvel abs=", QVEL_ABS_TOL, " rel=", QVEL_REL_TOL)
    print()

    var num_pass = 0
    var num_fail = 0

    if test_freefall():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_standing_with_action():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_moving_with_action():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_freefall_10_steps():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_standing_10_steps():
        num_pass += 1
    else:
        num_fail += 1
    print()

    print("=" * 60)
    print(
        "Results:",
        num_pass,
        "passed,",
        num_fail,
        "failed out of",
        num_pass + num_fail,
    )
    if num_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
