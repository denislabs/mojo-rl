"""Zero-action stability test for acceleration-level solver.

Runs HalfCheetah with ZERO torques (no policy) for 1000+ steps.
If the physics engine is correct, the robot should simply fall under gravity,
make contact with the ground, and settle to a stable resting state.

Any instability here = physics engine bug (not policy incompatibility).

Run with:
    cd mojo-rl && pixi run mojo run tests/test_zero_action_stability.mojo
"""

from random import seed
from math import abs

from envs.half_cheetah import HalfCheetah, HalfCheetahParams
from envs.half_cheetah.half_cheetah_def import (
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    HalfCheetahJoints,
)


comptime C = HalfCheetahParams[DType.float32]
comptime dtype = DType.float32


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Zero-Action Stability Test — HalfCheetah (Acceleration-Level Solver)")
    print("=" * 70)

    var env = HalfCheetah()
    _ = env.reset()

    print("dt =", Float64(env.model.timestep))
    print("solref_contact =", env.model.solref_contact[0], env.model.solref_contact[1])
    print("solimp_contact =", env.model.solimp_contact[0], env.model.solimp_contact[1], env.model.solimp_contact[2])
    print()

    # Zero actions
    var action_list = List[Scalar[dtype]]()
    for _ in range(C.ACTION_DIM):
        action_list.append(Scalar[dtype](0))

    comptime MAX_STEPS = 2000
    var max_pen_overall: Float64 = 0
    var max_vel_overall: Float64 = 0
    var anomaly_found = False

    for step in range(MAX_STEPS):
        # Enable verbose for first 3 steps and around any anomaly
        var do_verbose = step < 3

        var result = env.step_continuous_vec[dtype](
            action_list, verbose=do_verbose
        )
        var reward = result[1]
        var done = result[2]

        # Extract diagnostic values
        var rootz = Float64(env.data.qpos[JOINT_ROOTZ])
        var rootx = Float64(env.data.qpos[JOINT_ROOTX])
        var pitch = Float64(env.data.qpos[JOINT_ROOTY])
        var nc = Int(env.data.num_contacts)

        # Max penetration
        var max_pen: Float64 = 0
        for c in range(nc):
            var pen = -Float64(env.data.contacts[c].dist)
            if pen > max_pen:
                max_pen = pen
        if max_pen > max_pen_overall:
            max_pen_overall = max_pen

        # Max absolute velocity
        var max_vel: Float64 = 0
        for i in range(C.NV):
            var v = abs(Float64(env.data.qvel[i]))
            if v > max_vel:
                max_vel = v
        if max_vel > max_vel_overall:
            max_vel_overall = max_vel

        # Print every 50 steps
        if (step + 1) % 50 == 0 or step < 5:
            print(
                "  step", step + 1,
                ": rootz=", rootz,
                " rootx=", rootx,
                " pitch=", pitch,
                " contacts=", nc,
                " max_pen=", max_pen,
                " max_vel=", max_vel,
            )

        # Check for anomalies
        if max_pen > 0.05:  # 50mm
            if not anomaly_found:
                anomaly_found = True
                print("!!! ANOMALY: Large penetration", max_pen, "at step", step + 1)
                print("    qpos:", end="")
                for i in range(C.NQ):
                    print(" ", Float64(env.data.qpos[i]), end="")
                print("")
                print("    qvel:", end="")
                for i in range(C.NV):
                    print(" ", Float64(env.data.qvel[i]), end="")
                print("")

        if max_vel > 9.5:  # Near MAX_QVEL clamp
            if not anomaly_found:
                anomaly_found = True
                print("!!! ANOMALY: Velocity explosion", max_vel, "at step", step + 1)

        if rootz < -0.5:
            if not anomaly_found:
                anomaly_found = True
                print("!!! ANOMALY: Deep underground rootz=", rootz, "at step", step + 1)

        if done:
            print("  Episode ended at step", step + 1)
            break

    print()
    print("=" * 70)
    print("RESULTS:")
    print("  Max penetration overall:", max_pen_overall, "m")
    print("  Max velocity overall:   ", max_vel_overall, "m/s or rad/s")
    print("  Final rootz:            ", Float64(env.data.qpos[JOINT_ROOTZ]))
    print("  Final pitch:            ", Float64(env.data.qpos[JOINT_ROOTY]))
    if anomaly_found:
        print("  STATUS: UNSTABLE — physics bug detected!")
    else:
        print("  STATUS: STABLE — physics engine is correct")
    print("=" * 70)
