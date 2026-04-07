"""Test RK4 Full Step with Contacts: Mojo Engine vs MuJoCo for Hopper.

Tests the actual integrator used by the Hopper training env (RK4 + Newton),
NOT Euler as in the existing contact tests. This is the config that matters
for the SAC reward gap.

Hopper uses PYRAMIDAL cone (default) and condim=1 (frictionless contacts).
MuJoCo reference: opt.integrator=1 (mjINT_RK4), opt.cone=0 (PYRAMIDAL),
opt.solver=2 (NEWTON).

Scenarios:
  1. Free flight (no contacts) — baseline RK4 accuracy
  2. Ground contact (foot touching) — constraint solver interaction
  3. Ground contact + actions — full training-like scenario
  4. Multi-step (4 = one frame_skip) — error accumulation
  5. Deep penetration — stress test
  6. Moving with contacts — dynamic contact switching

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_hopper_rk4_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
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

# Tolerances — start moderate, tighten based on results.
# RK4 with contacts can diverge more than Euler due to 4x contact detection.
comptime QPOS_ABS_TOL: Float64 = 5e-4
comptime QPOS_REL_TOL: Float64 = 5e-4
comptime QVEL_ABS_TOL: Float64 = 5e-3
comptime QVEL_REL_TOL: Float64 = 5e-3


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
    """Run num_steps RK4 physics steps in both engines, compare final qpos/qvel."""
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === Our engine (RK4 + Newton) ===
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

    # === MuJoCo reference (RK4 + Newton) ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.cone = 0  # mjCONE_PYRAMIDAL (matches HopperModel default)
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.integrator = 1  # mjINT_RK4 (NOT Euler!)
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

    # Print values
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

    # Print contact details
    var our_ncon = Int(data.num_contacts)
    if our_ncon > 0:
        print("  --- Our contact details ---")
        for c in range(our_ncon):
            print(
                "  contact[",
                c,
                "]: body_a=",
                Int(data.contacts[c].body_a),
                " body_b=",
                Int(data.contacts[c].body_b),
                " pos=(",
                Float64(data.contacts[c].pos_x),
                ",",
                Float64(data.contacts[c].pos_y),
                ",",
                Float64(data.contacts[c].pos_z),
                ")",
                " dist=",
                Float64(data.contacts[c].dist),
                " force_n=",
                Float64(data.contacts[c].force_n),
            )

    if mj_ncon > 0:
        var mj_contacts = mj_data.contact
        for c in range(mj_ncon):
            var mj_c = mj_contacts[c]
            var mj_dist = Float64(py=mj_c.dist)
            var mj_pos = mj_c.pos.flatten().tolist()
            var mj_geom = mj_c.geom.flatten().tolist()
            print(
                "  MJ  contact[",
                c,
                "]: geom=(",
                Int(py=mj_geom[0]),
                ",",
                Int(py=mj_geom[1]),
                ")",
                " pos=(",
                Float64(py=mj_pos[0]),
                ",",
                Float64(py=mj_pos[1]),
                ",",
                Float64(py=mj_pos[2]),
                ")",
                " dist=",
                mj_dist,
            )

    assert_true(all_pass, "compare_step failed for: " + test_name)


# =============================================================================
# Test cases — no contacts (baseline RK4 accuracy)
# =============================================================================


def test_freefall_rk4() raises:
    """Free fall from default height — no contacts.
    Pure gravity, tests basic RK4 integration accuracy."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    # Hopper default: torso at 1.25. qpos[1]=0 means rootz at ref=1.25.
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall (no contacts)", qpos, qvel, actions)


def test_freefall_with_actions_rk4() raises:
    """Free fall with actions — tests force integration without contacts."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8  # thigh
    actions[1] = -0.5  # leg
    actions[2] = 0.3  # foot
    compare_step("Free fall + actions", qpos, qvel, actions)


# =============================================================================
# Test cases — ground contact (the critical scenarios for SAC training)
# =============================================================================


def test_ground_contact_rk4() raises:
    """Robot low enough for ground contact (foot touching).
    Hopper default: torso at 1.25, foot ~0.6m below. rootz=-0.8 pushes down.
    RK4 re-detects contacts at each of 4 stages."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8  # rootz — pushes robot down from 1.25
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Ground contact (low rootz)", qpos, qvel, actions)


def test_ground_contact_with_actions_rk4() raises:
    """Robot on ground with actions — full constraint solver + RK4 test.
    This is closest to what happens during SAC training."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8  # rootz — pushes robot down
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8  # thigh
    actions[1] = -0.5  # leg
    actions[2] = 0.3  # foot
    compare_step("Ground contact + actions", qpos, qvel, actions)


def test_ground_contact_4_steps_rk4() raises:
    """4 RK4 steps = 1 frame_skip in training. Tests error accumulation
    over one effective training step."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[1] = -0.3
    actions[2] = 0.2
    compare_step(
        "Ground contact + actions (4 steps = 1 frame_skip)",
        qpos,
        qvel,
        actions,
        num_steps=4,
    )


def test_ground_contact_20_steps_rk4() raises:
    """20 RK4 steps = 5 frame_skips. Longer horizon to see divergence grow."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[1] = -0.3
    actions[2] = 0.2
    compare_step(
        "Ground contact + actions (20 steps)",
        qpos,
        qvel,
        actions,
        num_steps=20,
    )


def test_deep_penetration_rk4() raises:
    """Deep ground penetration — stress test for contact solver under RK4."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -1.1  # very low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Deep penetration", qpos, qvel, actions)


def test_moving_with_contacts_rk4() raises:
    """Moving robot making contact — RK4 evaluates dynamics at 4 intermediate
    positions, so contact state can change between stages."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0  # rootx vel (forward)
    qvel[1] = -1.0  # rootz vel (falling)
    qvel[2] = -0.5  # rooty vel (tilting)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[1] = -0.3
    actions[2] = 0.2
    compare_step("Moving + contacts", qpos, qvel, actions)


def test_standing_default_rk4() raises:
    """Default standing pose — this is the initial state in training.
    Foot should be just touching the ground."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    # All zeros = default standing (rootz at 1.25 via ref)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.3
    actions[1] = -0.2
    actions[2] = 0.1
    compare_step(
        "Standing default + actions (4 steps)",
        qpos,
        qvel,
        actions,
        num_steps=4,
    )


def test_bent_joints_contact_rk4() raises:
    """Hopper with bent joints on ground — tests joint limit + contact interaction."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8  # low
    qpos[3] = -0.5  # thigh bent
    qpos[4] = -0.3  # leg bent
    qpos[5] = 0.2  # foot angled
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0
    actions[1] = -1.0
    actions[2] = 0.5
    compare_step("Bent joints + contact", qpos, qvel, actions)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
