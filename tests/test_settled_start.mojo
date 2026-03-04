"""Test: settle robot first, then apply actions."""
from std.random import seed
from envs.half_cheetah import HalfCheetah


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    # Phase 1: Settle with zero actions (100 env steps = 500 physics steps)
    var zero_action = env.ActionType()
    for i in range(6):
        zero_action[i] = 0.0

    for step in range(100):
        _ = env.step(zero_action)

    print("=== After settling (100 zero-action steps) ===")
    print(
        "rootx=",
        env.data.qpos[0],
        "rootz=",
        env.data.qpos[1],
        "rooty=",
        env.data.qpos[2],
    )
    print(
        "vx=",
        env.data.qvel[0],
        "vz=",
        env.data.qvel[1],
        "vy=",
        env.data.qvel[2],
    )
    print("nc=", env.data.num_contacts)
    print()

    # Phase 2: Apply +1.0 actions from settled state
    var action = env.ActionType()
    for i in range(6):
        action[i] = 1.0

    print("=== Now applying +1.0 actions ===")
    for step in range(100):
        _ = env.step(action)
        if step < 10 or step % 10 == 0:
            print(
                "Step",
                step + 1,
                "rootx=",
                env.data.qpos[0],
                "rootz=",
                env.data.qpos[1],
                "vx=",
                env.data.qvel[0],
                "vz=",
                env.data.qvel[1],
                "nc=",
                env.data.num_contacts,
            )
