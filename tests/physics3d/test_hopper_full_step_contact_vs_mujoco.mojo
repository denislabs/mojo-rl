"""Test Full Physics Step with Ground Contacts: Mojo Engine vs MuJoCo for Hopper.

Tests scenarios where the Hopper makes ground contact, exercising the
full constraint solver pipeline (contact detection + Jacobians + solver).

Hopper uses ELLIPTIC cone (default) and condim=1 (frictionless contacts).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_hopper_full_step_contact_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
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

# Tolerances — relaxed for contact scenarios
comptime QPOS_ABS_TOL: Float64 = 2e-4
comptime QPOS_REL_TOL: Float64 = 2e-4
comptime QVEL_ABS_TOL: Float64 = 2e-4
comptime QVEL_REL_TOL: Float64 = 2e-4


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int = 1,
) raises:
    """Run num_steps physics steps in both engines, compare final qpos/qvel."""
    print("--- Test:", test_name, "---")
    print("  Steps:", num_steps)

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
        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.cone = 1  # mjCONE_ELLIPTIC (matches HopperModel)
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.integrator = 0  # mjINT_EULER
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
        print("  --- Contact details ---")
        for c in range(our_ncon):
            print(
                "  Our contact[",
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
# Test cases — all involve ground contact
# =============================================================================


fn test_ground_contact() raises:
    """Robot low enough to have ground contact (foot touching).
    Hopper default: torso at 1.25, foot about 0.6m below. rootz=-0.8 pushes down.
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8  # rootz — pushes robot down from 1.25
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Ground contact (low rootz)", qpos, qvel, actions)


fn test_ground_contact_with_action() raises:
    """Robot on ground with actions — full constraint solver test."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8  # rootz — pushes robot down
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8  # thigh
    actions[1] = -0.5  # leg
    actions[2] = 0.3  # foot
    compare_step("Ground contact with action", qpos, qvel, actions)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
