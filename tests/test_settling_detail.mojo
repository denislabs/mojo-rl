"""Detailed settling diagnostic: print every 5 env steps during settling."""
from random import seed
from envs.half_cheetah import HalfCheetah


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    var zero_action = env.ActionType()
    for i in range(6):
        zero_action[i] = 0.0

    for step in range(100):
        _ = env.step(zero_action)
        if step < 20 or step % 10 == 0:
            print(
                "Step", step + 1,
                "rootx=", env.data.qpos[0],
                "rootz=", env.data.qpos[1],
                "rooty=", env.data.qpos[2],
                "vx=", env.data.qvel[0],
                "vz=", env.data.qvel[1],
                "nc=", env.data.num_contacts,
            )
            # Print contact details
            for c in range(env.data.num_contacts):
                var ct = env.data.contacts[c]
                print(
                    "  c", c,
                    "ba=", ct.body_a, "bb=", ct.body_b,
                    "dist=", ct.dist,
                    "fn=", ct.force_n,
                    "ft1=", ct.force_t1,
                    "ft2=", ct.force_t2,
                )
