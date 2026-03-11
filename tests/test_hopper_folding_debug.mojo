"""Debug Hopper body folding: check body-body collisions and joint limits."""
from std.random import seed
from mojo_rl.envs.hopper import Hopper


fn main():
    seed(42)
    var env = Hopper[DType.float64]()
    _ = env.reset()

    # Apply extreme actions to cause folding
    var action = env.ActionType()
    # Push all joints to extremes
    action[0] = 1.0  # thigh torque (200 Nm)
    action[1] = 1.0  # leg torque (200 Nm)
    action[2] = 1.0  # foot torque (200 Nm)

    print("=== Hopper folding debug ===")
    print("Joint limits:")
    print("  thigh: [-2.618, 0.0]")
    print("  leg:   [-2.618, 0.0]")
    print("  foot:  [-0.785, 0.785]")
    print()

    for step in range(200):
        _ = env.step(action)
        var q_thigh = env.data.qpos[3]
        var q_leg = env.data.qpos[4]
        var q_foot = env.data.qpos[5]
        var rootz = env.data.qpos[1]
        var rooty = env.data.qpos[2]

        # Check if any joints exceed limits
        var thigh_violated = q_thigh < -2.618 or q_thigh > 0.0
        var leg_violated = q_leg < -2.618 or q_leg > 0.0
        var foot_violated = q_foot < -0.785 or q_foot > 0.785

        if (
            step < 30
            or step % 20 == 0
            or thigh_violated
            or leg_violated
            or foot_violated
        ):
            print(
                "Step",
                step,
                "rootz=",
                rootz,
                "rooty=",
                rooty,
                "q_thigh=",
                q_thigh,
                "q_leg=",
                q_leg,
                "q_foot=",
                q_foot,
                "nc=",
                env.data.num_contacts,
            )
            if thigh_violated:
                print("  !!! THIGH LIMIT VIOLATED")
            if leg_violated:
                print("  !!! LEG LIMIT VIOLATED")
            if foot_violated:
                print("  !!! FOOT LIMIT VIOLATED")

            for c in range(env.data.num_contacts):
                var ct = env.data.contacts[c]
                print(
                    "  c",
                    c,
                    "body_a=",
                    ct.body_a,
                    "body_b=",
                    ct.body_b,
                    "fn=",
                    ct.force_n,
                    "ft2=",
                    ct.force_t2,
                    "dist=",
                    ct.dist,
                )
