"""Debug friction: print constraint Jacobians and solver internals."""
from std.random import seed
from envs.half_cheetah import HalfCheetah
from physics3d.constraints.constraint_data import (
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
)


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    # Settle
    var zero_action = env.ActionType()
    for i in range(6):
        zero_action[i] = 0.0
    for _ in range(100):
        _ = env.step(zero_action)

    # Inject horizontal velocity
    env.data.qvel[0] = 1.0

    print("=== Injected vx=1.0, zero actions ===")
    for step in range(20):
        _ = env.step(zero_action)
        print(
            "Step",
            step + 1,
            "vx=",
            env.data.qvel[0],
            "nc=",
            env.data.num_contacts,
        )
        for c in range(env.data.num_contacts):
            var ct = env.data.contacts[c]
            print(
                "  c",
                c,
                "fn=",
                ct.force_n,
                "ft1=",
                ct.force_t1,
                "ft2=",
                ct.force_t2,
                "dist=",
                ct.dist,
            )
