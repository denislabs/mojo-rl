"""Quick Hopper test: verify friction geom changes compile and run."""
from random import seed
from envs.hopper import Hopper


fn main():
    seed(42)
    var env = Hopper[DType.float64]()
    _ = env.reset()
    var action = env.ActionType()
    for i in range(3):
        action[i] = 0.0
    for step in range(20):
        _ = env.step(action)
        print(
            "Step", step,
            "rootz=", env.data.qpos[1],
            "vz=", env.data.qvel[1],
            "nc=", env.data.num_contacts,
        )
        for c in range(env.data.num_contacts):
            var ct = env.data.contacts[c]
            print(
                "  c", c,
                "fn=", ct.force_n,
                "ft2=", ct.force_t2,
                "dist=", ct.dist,
            )
