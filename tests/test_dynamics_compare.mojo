"""Direct comparison of our dynamics vs MuJoCo.

Dumps M_hat diagonal, bias forces, f_net, qacc at known states.
Compare output with tests/mujoco_dynamics_compare.py.

Run with:
    cd mojo-rl && pixi run mojo build tests/test_dynamics_compare.mojo -o /tmp/test_dynamics && /tmp/test_dynamics
"""

from envs.half_cheetah import HalfCheetah, HalfCheetahParams
from envs.half_cheetah.half_cheetah_def import (
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    HalfCheetahJoints,
    HalfCheetahModel,
)
from physics3d.integrator import ImplicitFastIntegrator
from physics3d.solver import NewtonSolver


comptime C = HalfCheetahParams[DType.float64]
comptime dtype = DType.float64


fn main() raises:
    print("=" * 70)
    print("Dynamics Comparison Test — HalfCheetah vs MuJoCo")
    print("=" * 70)

    var env = HalfCheetah()
    _ = env.reset()

    print("Our engine: NQ =", C.NQ, "NV =", C.NV, "NBODY =", C.NUM_BODIES)
    print("MuJoCo:     NQ = 9, NV = 9, NBODY = 8 (incl world)")
    print("dt =", Float64(env.model.timestep))
    print()

    # Print joint info
    for j in range(env.model.num_joints):
        var joint = env.model.joints[j]
        print(
            "  Joint",
            j,
            ": dof=",
            joint.dof_adr,
            " stiffness=",
            Float64(joint.stiffness),
            " damping=",
            Float64(joint.damping),
            " armature=",
            Float64(joint.armature),
        )
    print()

    # ===== TEST 1: Default state (zero velocity) =====
    print("=" * 70)
    print("TEST 1: Default state (zero velocity)")
    print("=" * 70)

    # Reset to default state
    _ = env.reset()
    # Zero out any noise
    for i in range(C.NQ):
        env.data.qpos[i] = 0
    env.data.qpos[JOINT_ROOTZ] = 0.7  # Our convention: absolute z
    for i in range(C.NV):
        env.data.qvel[i] = 0
        env.data.qfrc[i] = 0

    print("qpos:", end="")
    for i in range(C.NQ):
        print(" ", Float64(env.data.qpos[i]), end="")
    print("")
    print("qvel:", end="")
    for i in range(C.NV):
        print(" ", Float64(env.data.qvel[i]), end="")
    print("")

    # Run one step with verbose
    print("\n--- Substep 1 ---")
    ImplicitFastIntegrator[SOLVER=NewtonSolver].step[
        NGEOM = C.NGEOM
    ](env.model, env.data, verbose=True)

    print("\nAfter 1 substep:")
    print("qpos:", end="")
    for i in range(C.NQ):
        print(" ", Float64(env.data.qpos[i]), end="")
    print("")
    print("qvel:", end="")
    for i in range(C.NV):
        print(" ", Float64(env.data.qvel[i]), end="")
    print("")

    # ===== TEST 2: Moderate velocities =====
    print("\n" + "=" * 70)
    print("TEST 2: Moderate velocities")
    print("=" * 70)

    _ = env.reset()
    for i in range(C.NQ):
        env.data.qpos[i] = 0
    env.data.qpos[JOINT_ROOTZ] = 0.7
    for i in range(C.NV):
        env.data.qfrc[i] = 0

    # Set moderate velocities matching MuJoCo test
    var test_vel = List[Float64]()
    test_vel.append(1.0)
    test_vel.append(-2.0)
    test_vel.append(0.5)
    test_vel.append(3.0)
    test_vel.append(-1.5)
    test_vel.append(2.0)
    test_vel.append(-3.0)
    test_vel.append(1.0)
    test_vel.append(-0.5)
    for i in range(min(C.NV, 9)):
        env.data.qvel[i] = test_vel[i]

    print("qpos:", end="")
    for i in range(C.NQ):
        print(" ", Float64(env.data.qpos[i]), end="")
    print("")
    print("qvel:", end="")
    for i in range(C.NV):
        print(" ", Float64(env.data.qvel[i]), end="")
    print("")

    print("\n--- Substep 1 ---")
    ImplicitFastIntegrator[SOLVER=NewtonSolver].step[
        NGEOM = C.NGEOM
    ](env.model, env.data, verbose=True)

    print("\nAfter 1 substep:")
    print("qpos:", end="")
    for i in range(C.NQ):
        print(" ", Float64(env.data.qpos[i]), end="")
    print("")
    print("qvel:", end="")
    for i in range(C.NV):
        print(" ", Float64(env.data.qvel[i]), end="")
    print("")

    # ===== TEST 3: With action =====
    print("\n" + "=" * 70)
    print("TEST 3: Default state with action [1, -1, 0.5, -0.5, 1, -1]")
    print("=" * 70)

    _ = env.reset()
    for i in range(C.NQ):
        env.data.qpos[i] = 0
    env.data.qpos[JOINT_ROOTZ] = 0.7
    for i in range(C.NV):
        env.data.qvel[i] = 0
        env.data.qfrc[i] = 0

    # Apply action [1, -1, 0.5, -0.5, 1, -1]
    # MuJoCo: ctrl * gear gives force
    # Gears: [120, 90, 60, 120, 60, 30]
    # So forces: [120, -90, 30, -60, 60, -30]
    var actions = List[Float64]()
    actions.append(1.0)
    actions.append(-1.0)
    actions.append(0.5)
    actions.append(-0.5)
    actions.append(1.0)
    actions.append(-1.0)
    HalfCheetahJoints.apply_actions(env.data, actions)

    print("qfrc (after apply_actions):", end="")
    for i in range(C.NV):
        print(" ", Float64(env.data.qfrc[i]), end="")
    print("")

    print("\n--- Substep 1 ---")
    ImplicitFastIntegrator[SOLVER=NewtonSolver].step[
        NGEOM = C.NGEOM
    ](env.model, env.data, verbose=True)

    print("\nAfter 1 substep:")
    print("qpos:", end="")
    for i in range(C.NQ):
        print(" ", Float64(env.data.qpos[i]), end="")
    print("")
    print("qvel:", end="")
    for i in range(C.NV):
        print(" ", Float64(env.data.qvel[i]), end="")
    print("")
