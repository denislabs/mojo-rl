"""Debug friction: run one verbose step after injecting vx=1.0."""
from std.random import seed
from mojo_rl.envs.half_cheetah import HalfCheetah


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

    print(
        "After settling: rootx=", env.data.qpos[0], "rootz=", env.data.qpos[1]
    )
    print("vx=", env.data.qvel[0], "nc=", env.data.num_contacts)

    # Inject horizontal velocity
    env.data.qvel[0] = 1.0
    print("\n=== Injected vx=1.0, verbose step ===")
    _ = env.step(zero_action, verbose=True)
    print("\nAfter step: vx=", env.data.qvel[0])
