"""Minimal pyramidal cone debug test."""

from random import seed
from envs.half_cheetah import HalfCheetah


fn main() raises:
    seed(42)
    var env = HalfCheetah()

    # Reset
    var obs = env.reset()
    print("=== Pyramidal Cone Debug ===")
    print("Initial qpos[1] (z):", env.data.qpos[1])

    # Do steps with zero action, print z position each step
    var action = List[Scalar[DType.float64]]()
    for _ in range(6):
        action.append(Scalar[DType.float64](0.0))

    for step in range(20):
        var result = env.step_continuous_vec(action, verbose=False)
        var z = Float64(env.data.qpos[1])
        var vz = Float64(env.data.qvel[1])
        var vx = Float64(env.data.qvel[0])
        print(
            "step",
            step,
            " z=",
            z,
            " vz=",
            vz,
            " vx=",
            vx,
        )
        if z > 2.0:
            print("FLYING! Aborting.")
            break
