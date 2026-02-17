"""Debug pyramidal cone: check settling and contact detection."""
from random import seed
from envs.half_cheetah import HalfCheetah


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    print("cone_type =", env.model.cone_type)

    var zero_action = env.ActionType()
    for i in range(6):
        zero_action[i] = 0.0

    for step in range(20):
        _ = env.step(zero_action)
        print(
            "Step", step,
            "rootx=", env.data.qpos[0],
            "rootz=", env.data.qpos[1],
            "vx=", env.data.qvel[0],
            "vz=", env.data.qvel[1],
            "nc=", env.data.num_contacts,
        )
        for c in range(env.data.num_contacts):
            var ct = env.data.contacts[c]
            print(
                "  c", c,
                "fn=", ct.force_n,
                "ft1=", ct.force_t1,
                "ft2=", ct.force_t2,
                "dist=", ct.dist,
            )
