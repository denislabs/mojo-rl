"""Test: match MuJoCo initial state (rootz=0) and compare step-by-step."""
from random import seed
from envs.half_cheetah import HalfCheetah


fn main():
    seed(42)
    var env = HalfCheetah()
    _ = env.reset()

    # Set initial state to match MuJoCo: all qpos=0 (rootz=0 instead of 0.7)
    for i in range(9):
        env.data.qpos[i] = 0.0
    for i in range(9):
        env.data.qvel[i] = 0.0

    var action = env.ActionType()
    for i in range(6):
        action[i] = 1.0

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
                "rooty=",
                env.data.qpos[2],
                "vx=",
                env.data.qvel[0],
                "vz=",
                env.data.qvel[1],
                "nc=",
                env.data.num_contacts,
            )
            if step == 4:  # Step 5 — compare with MuJoCo
                print("  qpos:", end="")
                for i in range(9):
                    print(" ", env.data.qpos[i], end="")
                print()
                print("  qvel:", end="")
                for i in range(9):
                    print(" ", env.data.qvel[i], end="")
                print()
                print("  qacc:", end="")
                for i in range(9):
                    print(" ", env.data.qacc[i], end="")
                print()
                for c in range(env.data.num_contacts):
                    var ct = env.data.contacts[c]
                    print(
                        "  contact",
                        c,
                        ": body_a=",
                        ct.body_a,
                        "body_b=",
                        ct.body_b,
                        " dist=",
                        ct.dist,
                        " fn=",
                        ct.force_n,
                        " ft1=",
                        ct.force_t1,
                        " ft2=",
                        ct.force_t2,
                    )
