"""HalfCheetah Torque Diagnostic — physics behavior with applied forces.

Tests the physics with constant torques to see if ground penetration
worsens or bouncing occurs when actuation is applied.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_cheetah_torque_diag.mojo
"""

from math import sqrt, pi
from builtin.math import abs

from envs.half_cheetah_gc import HalfCheetahGC
from envs.half_cheetah_gc.constants_gc import (
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    BODY_HEAD,
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    JOINT_BTHIGH,
    JOINT_BSHIN,
    JOINT_BFOOT,
    JOINT_FTHIGH,
    JOINT_FSHIN,
    JOINT_FFOOT,
    CAPSULE_RADIUS,
    BTHIGH_GEAR,
    BSHIN_GEAR,
    BFOOT_GEAR,
    FTHIGH_GEAR,
    FSHIN_GEAR,
    FFOOT_GEAR,
    FRAME_SKIP,
    DT,
)

from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver.pgs_solver import PGSSolver
from physics3d.solver.newton_solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import forward_kinematics


fn body_name(id: Int) -> String:
    if id == 0:
        return "torso "
    elif id == 1:
        return "bthigh"
    elif id == 2:
        return "bshin "
    elif id == 3:
        return "bfoot "
    elif id == 4:
        return "fthigh"
    elif id == 5:
        return "fshin "
    elif id == 6:
        return "ffoot "
    elif id == 7:
        return "head  "
    return "???"


fn run_test(label: String, actions: InlineArray[Float64, 6]):
    """Run 200 env steps (1000 physics steps) with given actions."""
    print("")
    print("=" * 80)
    print(label)
    print("=" * 80)
    print("Actions (pre-gear):", actions[0], actions[1], actions[2], actions[3], actions[4], actions[5])
    print("Torques (post-gear):",
        actions[0] * BTHIGH_GEAR, actions[1] * BSHIN_GEAR, actions[2] * BFOOT_GEAR,
        actions[3] * FTHIGH_GEAR, actions[4] * FSHIN_GEAR, actions[5] * FFOOT_GEAR)
    print("")

    var env = HalfCheetahGC[DType.float64, False]()
    _ = env.reset()

    var max_pen = Float64(0.0)
    var max_rootz = Float64(env.data.qpos[JOINT_ROOTZ])
    var min_rootz = Float64(env.data.qpos[JOINT_ROOTZ])
    var max_vz = Float64(0.0)
    var min_vz = Float64(0.0)
    var max_imp_n = Float64(0.0)

    var dt = Float64(env.model.timestep)
    var env_steps = 200
    var physics_steps_per_env = FRAME_SKIP

    print("  step | rootz    | vz       | rooty    | bfoot_z  | ffoot_z  | contacts | max_pen  | max_imp_n | qvel_max")
    print("  " + "-" * 120)

    for step in range(env_steps):
        # Apply actions (like the environment does)
        env.data.qfrc[JOINT_BTHIGH] = Scalar[DType.float64](actions[0] * BTHIGH_GEAR)
        env.data.qfrc[JOINT_BSHIN] = Scalar[DType.float64](actions[1] * BSHIN_GEAR)
        env.data.qfrc[JOINT_BFOOT] = Scalar[DType.float64](actions[2] * BFOOT_GEAR)
        env.data.qfrc[JOINT_FTHIGH] = Scalar[DType.float64](actions[3] * FTHIGH_GEAR)
        env.data.qfrc[JOINT_FSHIN] = Scalar[DType.float64](actions[4] * FSHIN_GEAR)
        env.data.qfrc[JOINT_FFOOT] = Scalar[DType.float64](actions[5] * FFOOT_GEAR)

        # Physics sub-steps (matching env.step)
        for _ in range(physics_steps_per_env):
            EulerIntegrator[NewtonSolver].step(env.model, env.data)

        var rootz = Float64(env.data.qpos[JOINT_ROOTZ])
        var vz = Float64(env.data.qvel[JOINT_ROOTZ])
        var rooty = Float64(env.data.qpos[JOINT_ROOTY])
        var nc = Int(env.data.num_contacts)
        var bfoot_z = Float64(env.data.xpos[BODY_BFOOT * 3 + 2])
        var ffoot_z = Float64(env.data.xpos[BODY_FFOOT * 3 + 2])

        # Find max penetration and impulse
        var step_max_pen = Float64(0.0)
        var step_max_imp = Float64(0.0)
        for c in range(nc):
            var pen = -Float64(env.data.contacts[c].dist)
            var imp = Float64(env.data.contacts[c].impulse_n)
            if pen > step_max_pen:
                step_max_pen = pen
            if imp > step_max_imp:
                step_max_imp = imp

        # Find max absolute qvel
        var qvel_max = Float64(0.0)
        for i in range(NV):
            var v = Float64(env.data.qvel[i])
            if v > qvel_max:
                qvel_max = v
            if -v > qvel_max:
                qvel_max = -v

        if step_max_pen > max_pen:
            max_pen = step_max_pen
        if rootz > max_rootz:
            max_rootz = rootz
        if rootz < min_rootz:
            min_rootz = rootz
        if vz > max_vz:
            max_vz = vz
        if vz < min_vz:
            min_vz = vz
        if step_max_imp > max_imp_n:
            max_imp_n = step_max_imp

        # Print every 10 env steps, plus first 5
        if step < 5 or (step + 1) % 10 == 0 or step == env_steps - 1:
            print(
                "  ",
                step + 1,
                " | ",
                rootz,
                " | ",
                vz,
                " | ",
                rooty,
                " | ",
                bfoot_z,
                " | ",
                ffoot_z,
                " | ",
                nc,
                " | ",
                step_max_pen,
                " | ",
                step_max_imp,
                " | ",
                qvel_max,
            )

    print("")
    print("  SUMMARY:")
    print("    Max penetration:", max_pen, "m")
    print("    RootZ range: [", min_rootz, ",", max_rootz, "]")
    print("    Vz range: [", min_vz, ",", max_vz, "]")
    print("    Max impulse_n:", max_imp_n)


fn main():
    # Test 1: Zero actions (baseline)
    var zero = InlineArray[Float64, 6](0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    run_test("Test 1: Zero actions (free fall + settle)", zero)

    # Test 2: Small constant actions (mild torques)
    var small = InlineArray[Float64, 6](0.1, -0.1, 0.1, -0.1, 0.1, -0.1)
    run_test("Test 2: Small constant actions (0.1)", small)

    # Test 3: Max actions (worst case — all max torque)
    var max_act = InlineArray[Float64, 6](1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    run_test("Test 3: Max alternating actions (1.0)", max_act)

    # Test 4: All legs push down (should cause max ground force)
    var down = InlineArray[Float64, 6](-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    run_test("Test 4: All legs push down (-1.0)", down)

    # Test 5: All legs push up (should lift cheetah)
    var up = InlineArray[Float64, 6](1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    run_test("Test 5: All legs push up (1.0)", up)
