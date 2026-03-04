"""Quick stability test: 200 env steps with all +1.0 actions."""
from std.random import seed
from envs.half_cheetah import HalfCheetah


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    var action = env.ActionType()
    for i in range(6):
        action[i] = 1.0

    for step in range(200):
        _ = env.step(action)
        if step % 20 == 0 or step < 10:
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
