"""Test Full Implicit Step (no contacts): Mojo Engine vs MuJoCo.

Compares qpos/qvel after running physics steps using ImplicitIntegrator
on CPU vs MuJoCo Implicit integrator (opt.integrator=2 = mjINT_IMPLICIT).

Our ImplicitIntegrator uses M_hat = M + armature - dt * qDeriv where qDeriv
includes the FULL RNE velocity derivative (d(Coriolis)/d(qvel)), making
M_hat non-symmetric. Uses LU factorization instead of LDL.

MuJoCo reference: opt.integrator=2 (mjINT_IMPLICIT) which also includes
the full RNE velocity derivative via mjd_rne_vel.

Note: MuJoCo 3.3.6 enum values:
  0 = mjINT_EULER
  1 = mjINT_RK4
  2 = mjINT_IMPLICIT (full, with RNE derivative)
  3 = mjINT_IMPLICITFAST (no RNE derivative)

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_implicit_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.implicit_integrator import ImplicitIntegrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


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
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM

# Tolerances — tight since qDeriv is now verified to match MuJoCo.
comptime QPOS_ABS_TOL: Float64 = 1e-6
comptime QPOS_REL_TOL: Float64 = 1e-4
comptime QVEL_ABS_TOL: Float64 = 1e-4
comptime QVEL_REL_TOL: Float64 = 1e-4


# =============================================================================
# Comparison helper
# =============================================================================


def compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int = 1,
) raises:
    """Run num_steps physics steps in both engines, compare final qpos/qvel."""
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === Our engine (Implicit + Newton) ===
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
    HalfCheetahModel.setup_model_and_data(model, data)

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
        HalfCheetahModel.apply_actions(data, action_list)
        ImplicitIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    # === MuJoCo reference (opt.integrator=1 = mjINT_IMPLICIT) ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.cone = 0  # mjCONE_PYRAMIDAL (matches HalfCheetahModel)
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.integrator = (
        2  # mjINT_IMPLICIT (full implicit with RNE vel derivative)
    )
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
    assert_true(all_pass, "compare_step failed for: " + test_name)


# =============================================================================
# Test cases — no ground contact (free flight)
# =============================================================================


def test_freefall() raises:
    """Free fall from height — no contacts expected."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # rootz high enough to avoid ground contact
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall (no contacts)", qpos, qvel, actions)


def test_standing_with_action() raises:
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
    compare_step("Actions (no contacts)", qpos, qvel, actions)


def test_moving_with_action() raises:
    """Moving with velocity + actions, no contacts.

    This is the key test for the full implicit integrator — with nonzero
    velocities, the RNE velocity derivative (Coriolis terms) is nonzero,
    making qDeriv dense and M_hat non-symmetric. ImplicitFast would get
    different results because it skips these terms.
    """
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
    compare_step("Moving with actions (no contacts)", qpos, qvel, actions)


def test_fast_spinning() raises:
    """High angular velocities — maximizes Coriolis/gyroscopic effects.

    This test has high joint velocities which produce significant RNE
    velocity derivative terms, maximizing the difference between full
    implicit and implicit-fast integrators.
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # high
    qpos[2] = 0.3  # pitch
    qpos[3] = -0.5  # bthigh bent
    qpos[4] = 0.4  # bshin
    qpos[6] = 0.5  # fthigh bent
    qpos[7] = -0.3  # fshin
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 3.0  # rootx vel
    qvel[2] = 2.0  # rooty angular vel
    qvel[3] = -3.0  # bthigh fast
    qvel[4] = 2.5  # bshin fast
    qvel[6] = 3.0  # fthigh fast
    qvel[7] = -2.0  # fshin fast
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0
    actions[1] = -1.0
    actions[2] = 0.5
    actions[3] = 1.0
    actions[4] = -1.0
    actions[5] = 0.5
    compare_step("Fast spinning (high angular vel)", qpos, qvel, actions)


def test_freefall_10_steps() raises:
    """Free fall 10 steps — accumulates per-step drift."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall (10 steps)", qpos, qvel, actions, num_steps=10)


def test_moving_10_steps() raises:
    """Moving with velocity 10 steps — Coriolis effects accumulate."""
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
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[3] = 0.5
    compare_step(
        "Moving with actions (10 steps)", qpos, qvel, actions, num_steps=10
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
