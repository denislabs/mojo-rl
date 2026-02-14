"""Debug solver: check constraint building and solving for a single contact."""
from random import seed
from envs.hopper import Hopper
from physics3d.constraints.constraint_data import CNSTR_NORMAL, CNSTR_FRICTION_T1, CNSTR_FRICTION_T2, CNSTR_LIMIT


fn main():
    seed(42)
    var env = Hopper[DType.float64]()
    _ = env.reset()

    # Zero actions — just let it fall and settle
    var action = env.ActionType()
    for i in range(3):
        action[i] = 0.0

    print("=== Solver debug ===")
    print("Hopper: DT=0.002, FRAME_SKIP=4, Newton solver")
    print()

    for step in range(30):
        _ = env.step(action)
        var nc = env.data.num_contacts
        if nc > 0 or step < 5:
            print(
                "Step", step,
                "rootz=", env.data.qpos[1],
                "vz=", env.data.qvel[1],
                "nc=", nc,
            )
            for c in range(nc):
                var ct = env.data.contacts[c]
                print(
                    "  c", c,
                    "body_a=", ct.body_a,
                    "body_b=", ct.body_b,
                    "fn=", ct.force_n,
                    "ft1=", ct.force_t1,
                    "ft2=", ct.force_t2,
                    "dist=", ct.dist,
                    "condim=", ct.condim,
                )
