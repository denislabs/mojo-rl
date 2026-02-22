"""Test cfrc_ext (contact forces per body) vs MuJoCo for Hopper.

Validates that compute_cfrc_ext() matches MuJoCo's d.cfrc_ext after a physics
step where the Hopper foot makes ground contact.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_cfrc_ext_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, ConeType
from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver import NewtonSolver
from envs.hopper.hopper_def import (
    HopperModel,
    HopperParams,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HopperModel.NQ
comptime NV = HopperModel.NV
comptime NBODY = HopperModel.NBODY
comptime NJOINT = HopperModel.NJOINT
comptime NGEOM = HopperModel.NGEOM
comptime MAX_CONTACTS = HopperParams[DTYPE].MAX_CONTACTS
comptime ACTION_DIM = HopperParams[DTYPE].ACTION_DIM

# Tolerances
comptime TOL: Float64 = 1.0    # N or Nm absolute tolerance
comptime FRAC_TOL: Float64 = 0.5  # 50% relative tolerance for large forces


fn run_test(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int = 1,
) raises -> Bool:
    """Run num_steps, compare cfrc_ext[1:] vs MuJoCo."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, HopperModel.CONE_TYPE
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
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

    # Print our cfrc_ext
    print("  Our cfrc_ext (bodies 1..NBODY-1):")
    for b in range(1, NBODY):
        print(
            "    body",
            b,
            ": torque=[",
            Float64(data.cfrc_ext[b * 6 + 0]),
            Float64(data.cfrc_ext[b * 6 + 1]),
            Float64(data.cfrc_ext[b * 6 + 2]),
            "]  force=[",
            Float64(data.cfrc_ext[b * 6 + 3]),
            Float64(data.cfrc_ext[b * 6 + 4]),
            Float64(data.cfrc_ext[b * 6 + 5]),
            "]",
        )

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = String("../Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml")
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.cone = 1     # mjCONE_ELLIPTIC
    mj_model.opt.solver = 2   # mjSOL_NEWTON
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

    # Compute cfrc_ext (MuJoCo only computes it on demand via rnePostConstraint)
    mujoco.mj_rnePostConstraint(mj_model, mj_data)

    var mj_cfrc = mj_data.cfrc_ext  # shape (NBODY, 6)

    print("  MuJoCo cfrc_ext (bodies 1..NBODY-1):")
    for b in range(1, NBODY):
        var row = mj_cfrc[b]
        print(
            "    body",
            b,
            ": torque=[",
            Float64(py=row[0]),
            Float64(py=row[1]),
            Float64(py=row[2]),
            "]  force=[",
            Float64(py=row[3]),
            Float64(py=row[4]),
            Float64(py=row[5]),
            "]",
        )

    # Compare
    var max_abs_err: Float64 = 0.0
    var passed = True
    for b in range(1, NBODY):
        var row = mj_cfrc[b]
        for k in range(6):
            var our_val = Float64(data.cfrc_ext[b * 6 + k])
            var mj_val = Float64(py=row[k])
            var err = abs(our_val - mj_val)
            if err > max_abs_err:
                max_abs_err = err
            var abs_mj = abs(mj_val)
            if abs_mj > 1.0:
                var rel = err / abs_mj
                if rel > FRAC_TOL and err > TOL:
                    print(
                        "  MISMATCH body",
                        b,
                        "component",
                        k,
                        ": ours=",
                        our_val,
                        " mj=",
                        mj_val,
                        " rel_err=",
                        rel,
                    )
                    passed = False
            elif err > TOL:
                print(
                    "  MISMATCH body",
                    b,
                    "component",
                    k,
                    ": ours=",
                    our_val,
                    " mj=",
                    mj_val,
                    " abs_err=",
                    err,
                )
                passed = False

    print("  Max absolute error:", max_abs_err)
    if passed:
        print("  PASS")
    else:
        print("  FAIL")
    return passed


fn main() raises:
    print("=" * 60)
    print("cfrc_ext vs MuJoCo tests (Hopper)")
    print("=" * 60)

    var all_passed = True

    # Test 1: Standing pose, no action — foot in contact with ground
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    qpos1[1] = 1.25  # rootz = default standing height
    var qvel1 = InlineArray[Float64, NV](fill=0.0)
    var act0 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    all_passed = run_test("Standing, no action, 1 step", qpos1, qvel1, act0, 1) and all_passed

    # Test 2: Standing pose, strong action (larger contact forces)
    var act1 = InlineArray[Float64, ACTION_DIM](fill=1.0)
    all_passed = run_test("Standing, max action, 1 step", qpos1, qvel1, act1, 1) and all_passed

    # Test 3: After 5 steps with action
    all_passed = run_test("Standing, max action, 5 steps", qpos1, qvel1, act1, 5) and all_passed

    # Test 4: Slightly compressed (foot deeper in ground → bigger contact force)
    var qpos2 = InlineArray[Float64, NQ](fill=0.0)
    qpos2[1] = 1.0  # lower than default
    all_passed = run_test("Low pose, max action, 1 step", qpos2, qvel1, act1, 1) and all_passed

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
