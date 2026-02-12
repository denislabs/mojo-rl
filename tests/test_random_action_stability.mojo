"""Random-action stress test for acceleration-level solver.

Runs HalfCheetah with RANDOM large torques for 1000 steps.
A correct physics engine should handle ANY torques without:
- Ground penetration > reasonable threshold
- Velocity explosion
- Flying (rootz >> initial height)

If instability occurs here, the solver has a robustness bug.

Run with:
    cd mojo-rl && pixi run mojo run tests/test_random_action_stability.mojo
"""

from random import seed, random_float64
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
    print("Random-Action Stress Test — HalfCheetah (Accel-Level Solver)")
    print("=" * 70)

    var env = HalfCheetah()
    _ = env.reset()

    print("dt =", Float64(env.model.timestep))
    print()

    comptime MAX_STEPS = 1000
    var max_pen_overall: Float64 = 0
    var max_vel_overall: Float64 = 0
    var max_rootz: Float64 = 0
    var min_rootz: Float64 = 1.0
    var anomaly_count = 0

    for step in range(MAX_STEPS):
        # Random actions in [-1, 1] (full torque range)
        var action_list = List[Scalar[dtype]]()
        for _ in range(C.ACTION_DIM):
            var a = random_float64(-1.0, 1.0)
            action_list.append(Scalar[dtype](a))

        # Enable verbose around anomalies
        var do_verbose = False
        if anomaly_count > 0 and anomaly_count <= 3:
            do_verbose = True
            anomaly_count += 1

        var result = env.step_continuous_vec[dtype](
            action_list, verbose=do_verbose
        )
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

        if rootz > max_rootz:
            max_rootz = rootz
        if rootz < min_rootz:
            min_rootz = rootz

        # Print every 50 steps
        if (step + 1) % 50 == 0 or step < 5:
            print(
                "  step", step + 1,
                ": rootz=", rootz,
                " pitch=", pitch,
                " contacts=", nc,
                " max_pen=", max_pen,
                " max_vel=", max_vel,
            )

        # Detect anomalies
        if max_pen > 0.05 and anomaly_count == 0:
            anomaly_count = 1
            print("!!! PENETRATION", max_pen, "at step", step + 1)
            print("    rootz=", rootz, " pitch=", pitch, " contacts=", nc)
            print("    qpos:", end="")
            for i in range(C.NQ):
                print(" ", Float64(env.data.qpos[i]), end="")
            print("")
            print("    qvel:", end="")
            for i in range(C.NV):
                print(" ", Float64(env.data.qvel[i]), end="")
            print("")

        if rootz > 3.0 and anomaly_count == 0:
            anomaly_count = 1
            print("!!! FLYING rootz=", rootz, "at step", step + 1)

        if done:
            print("  Episode ended at step", step + 1, "(unhealthy termination)")
            # Reset and keep going to test more
            _ = env.reset()

    print()
    print("=" * 70)
    print("RESULTS:")
    print("  Max penetration: ", max_pen_overall, "m")
    print("  Max velocity:    ", max_vel_overall)
    print("  Max rootz:       ", max_rootz)
    print("  Min rootz:       ", min_rootz)
    if max_pen_overall > 0.05:
        print("  STATUS: UNSTABLE — large penetration under random torques!")
    elif max_vel_overall > 9.5:
        print("  STATUS: UNSTABLE — velocity explosion under random torques!")
    else:
        print("  STATUS: ROBUST — physics handles random torques correctly")
    print("=" * 70)
