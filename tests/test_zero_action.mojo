"""Test: zero actions should produce stable free fall + ground contact."""
from random import seed
from envs.half_cheetah import HalfCheetah


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    # Zero actions — robot should just fall and settle
    var action = env.ActionType()
    for i in range(6):
        action[i] = 0.0

    for step in range(100):
        _ = env.step(action)
        if step % 10 == 0 or step < 10:
            print(
                "Step",
                step + 1,
                "rootz=",
                env.data.qpos[1],
                "vz=",
                env.data.qvel[1],
                "nc=",
                env.data.num_contacts,
            )

    # Now test with torques applied while airborne
    print("\n=== Test airborne with +1.0 actions ===")
    var env2 = HalfCheetah()
    _ = env2.reset()
    # Set initial height very high (no contacts possible)
    env2.data.qpos[1] = 10.0  # 10m above ground
    for i in range(9):
        env2.data.qvel[i] = 0.0  # zero velocity (NV=9)

    var action2 = env2.ActionType()
    for i in range(6):
        action2[i] = 1.0

    for step in range(20):
        _ = env2.step(action2)
        # CoM should only have gravity: az = -9.81
        print(
            "Step",
            step + 1,
            "rootz=",
            env2.data.qpos[1],
            "vz=",
            env2.data.qvel[1],
            "az_approx=",
            env2.data.qacc[1],
            "nc=",
            env2.data.num_contacts,
        )
