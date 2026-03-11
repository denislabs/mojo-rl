"""Test: check joint positions and limits with +1.0 actions."""
from std.random import seed
from mojo_rl.envs.half_cheetah import HalfCheetah


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    var action = env.ActionType()
    for i in range(6):
        action[i] = 1.0

    # Joint order: rootx(0), rootz(1), rooty(2), bthigh(3), bshin(4), bfoot(5), fthigh(6), fshin(7), ffoot(8)
    # Joint ranges from XML:
    # bthigh: [-0.52, 1.05], bshin: [-0.785, 0.785], bfoot: [-0.4, 0.785]
    # fthigh: [-1.0, 0.7], fshin: [-1.2, 0.87], ffoot: [-0.5, 0.5]

    for step in range(30):
        _ = env.step(action)
        if step % 5 == 0 or step < 5:
            print("Step", step + 1)
            print(
                "  rootx=",
                env.data.qpos[0],
                "rootz=",
                env.data.qpos[1],
                "rooty=",
                env.data.qpos[2],
            )
            print("  bthigh=", env.data.qpos[3], "[-0.52, 1.05]")
            print("  bshin=", env.data.qpos[4], "[-0.785, 0.785]")
            print("  bfoot=", env.data.qpos[5], "[-0.4, 0.785]")
            print("  fthigh=", env.data.qpos[6], "[-1.0, 0.7]")
            print("  fshin=", env.data.qpos[7], "[-1.2, 0.87]")
            print("  ffoot=", env.data.qpos[8], "[-0.5, 0.5]")
            print("  vx=", env.data.qvel[0], "vz=", env.data.qvel[1])
            print("  num_contacts=", env.data.num_contacts)
            print()
