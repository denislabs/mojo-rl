"""Debug script to verify Half Cheetah GC body positions.

Prints the computed xpos/xquat values after reset to verify FK is working correctly.
"""

from envs.half_cheetah import HalfCheetah
from envs.half_cheetah.half_cheetah_def import (
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    HalfCheetahParams,
)

comptime TORSO_HALF_LENGTH: Float64 = 0.5
comptime INIT_HEIGHT: Float64 = HalfCheetahParams[DType.float64].INITIAL_Z


fn main() raises:
    print("=" * 60)
    print("Half Cheetah GC Body Position Debug")
    print("=" * 60)

    # Create environment
    var env = HalfCheetah()
    print("\nEnvironment created.")

    # Reset to initial state
    _ = env.reset()

    print("\n--- Initial qpos values ---")
    var qpos = env.get_qpos()
    print("rootx (qpos[0]):", qpos[0])
    print("rootz (qpos[1]):", qpos[1])
    print("rooty (qpos[2]):", qpos[2])
    print("bthigh (qpos[3]):", qpos[3])
    print("bshin (qpos[4]):", qpos[4])
    print("bfoot (qpos[5]):", qpos[5])
    print("fthigh (qpos[6]):", qpos[6])
    print("fshin (qpos[7]):", qpos[7])
    print("ffoot (qpos[8]):", qpos[8])

    print("\n--- Body world positions (xpos) ---")
    var body_names = List[String]()
    body_names.append("Torso")
    body_names.append("BThigh")
    body_names.append("BShin")
    body_names.append("BFoot")
    body_names.append("FThigh")
    body_names.append("FShin")
    body_names.append("FFoot")

    for i in range(7):
        var pos = env.get_body_position(i)
        print(
            body_names[i],
            ": x=",
            pos[0],
            ", y=",
            pos[1],
            ", z=",
            pos[2],
        )

    print("\n--- Body world orientations (xquat as x,y,z,w) ---")
    for i in range(7):
        var quat = env.get_body_quaternion(i)
        print(
            body_names[i],
            ": x=",
            quat[0],
            ", y=",
            quat[1],
            ", z=",
            quat[2],
            ", w=",
            quat[3],
        )

    print("\n--- Expected values ---")
    print("Torso should be at: x=0, y=0, z=", INIT_HEIGHT)
    print("Torso quat should be 90° Y: x=0, y=0.707, z=0, w=0.707")
    print(
        "Back leg X should be: x=",
        -TORSO_HALF_LENGTH,
        " (back of torso)",
    )
    print(
        "Front leg X should be: x=",
        TORSO_HALF_LENGTH,
        " (front of torso)",
    )

    print("\n--- Verification ---")
    var torso_pos = env.get_body_position(BODY_TORSO)
    var bthigh_pos = env.get_body_position(BODY_BTHIGH)
    var fthigh_pos = env.get_body_position(BODY_FTHIGH)

    var torso_ok = (
        abs(Float64(torso_pos[0])) < 0.01
        and abs(Float64(torso_pos[1])) < 0.01
        and abs(Float64(torso_pos[2]) - INIT_HEIGHT) < 0.01
    )
    print("Torso position OK:", torso_ok)

    var bthigh_x_ok = abs(Float64(bthigh_pos[0]) + TORSO_HALF_LENGTH) < 0.01
    print(
        "BThigh at back of torso (x=-0.5):",
        bthigh_x_ok,
        " (actual x=",
        bthigh_pos[0],
        ")",
    )

    var fthigh_x_ok = abs(Float64(fthigh_pos[0]) - TORSO_HALF_LENGTH) < 0.01
    print(
        "FThigh at front of torso (x=+0.5):",
        fthigh_x_ok,
        " (actual x=",
        fthigh_pos[0],
        ")",
    )

    # Check quaternions
    var torso_quat = env.get_body_quaternion(BODY_TORSO)
    var torso_quat_ok = (
        abs(Float64(torso_quat[0])) < 0.01  # x ≈ 0
        and abs(Float64(torso_quat[1]) - 0.707) < 0.01  # y ≈ 0.707
        and abs(Float64(torso_quat[2])) < 0.01  # z ≈ 0
        and abs(Float64(torso_quat[3]) - 0.707) < 0.01  # w ≈ 0.707
    )
    print("Torso 90° Y rotation OK:", torso_quat_ok)

    var bthigh_quat = env.get_body_quaternion(BODY_BTHIGH)
    var bthigh_vertical = (
        abs(Float64(bthigh_quat[0])) < 0.01
        and abs(Float64(bthigh_quat[1])) < 0.01
        and abs(Float64(bthigh_quat[2])) < 0.01
        and abs(Float64(bthigh_quat[3]) - 1.0) < 0.01
    )
    print("BThigh vertical (identity quat) OK:", bthigh_vertical)

    # Run a few simulation steps with zero action and check connectivity
    print("\n--- Running 10 simulation steps with zero action ---")
    from core import ContAction

    var zero_action = ContAction[6]()  # All zeros

    for step in range(10):
        _ = env.step(zero_action)

        var t_pos = env.get_body_position(BODY_TORSO)
        var bt_pos = env.get_body_position(BODY_BTHIGH)
        var ft_pos = env.get_body_position(BODY_FTHIGH)

        # Check back thigh is ~0.5 behind torso (accounting for torso rotation)
        var torso_quat = env.get_body_quaternion(BODY_TORSO)
        # For 90° Y rotation, back leg should be at torso_x - 0.5*cos(rooty)

        # Simple check: back leg should be within 1m of torso
        var bt_dist = (
            (Float64(bt_pos[0]) - Float64(t_pos[0])) ** 2
            + (Float64(bt_pos[1]) - Float64(t_pos[1])) ** 2
            + (Float64(bt_pos[2]) - Float64(t_pos[2])) ** 2
        ) ** 0.5

        var ft_dist = (
            (Float64(ft_pos[0]) - Float64(t_pos[0])) ** 2
            + (Float64(ft_pos[1]) - Float64(t_pos[1])) ** 2
            + (Float64(ft_pos[2]) - Float64(t_pos[2])) ** 2
        ) ** 0.5

        if step == 0 or step == 9:
            print(
                "Step",
                step,
                ": Torso at (",
                t_pos[0],
                ",",
                t_pos[2],
                "), BThigh dist=",
                bt_dist,
                ", FThigh dist=",
                ft_dist,
            )

        # Alert if legs are too far from torso
        if bt_dist > 2.0 or ft_dist > 2.0:
            print("WARNING: Leg seems disconnected at step", step)
            print("  Torso:", t_pos[0], t_pos[1], t_pos[2])
            print("  BThigh:", bt_pos[0], bt_pos[1], bt_pos[2])
            print("  FThigh:", ft_pos[0], ft_pos[1], ft_pos[2])

    print("\n--- Running 100 steps with random-ish action ---")
    var action = ContAction[6]()
    action[0] = 0.5  # bthigh
    action[1] = -0.3  # bshin
    action[2] = 0.2  # bfoot
    action[3] = -0.5  # fthigh
    action[4] = 0.3  # fshin
    action[5] = -0.2  # ffoot

    for step in range(100):
        _ = env.step(action)

    var final_t = env.get_body_position(BODY_TORSO)
    var final_bt = env.get_body_position(BODY_BTHIGH)
    var final_ft = env.get_body_position(BODY_FTHIGH)

    print("After 100 steps:")
    print("  Torso:", final_t[0], final_t[1], final_t[2])
    print("  BThigh:", final_bt[0], final_bt[1], final_bt[2])
    print("  FThigh:", final_ft[0], final_ft[1], final_ft[2])

    # Print torso orientation
    var final_t_quat = env.get_body_quaternion(BODY_TORSO)
    print(
        "  Torso quat:",
        final_t_quat[0],
        final_t_quat[1],
        final_t_quat[2],
        final_t_quat[3],
    )

    # Check qpos to see rooty (pitch) angle
    var final_qpos = env.get_qpos()
    print("  rooty (pitch angle):", final_qpos[2], "radians")

    # Print ALL body positions
    print("\n  All body positions after 100 steps:")
    for i in range(7):
        var pos = env.get_body_position(i)
        var quat = env.get_body_quaternion(i)
        print(
            "   ",
            body_names[i],
            ": pos=(",
            pos[0],
            ",",
            pos[1],
            ",",
            pos[2],
            ") quat=(",
            quat[0],
            ",",
            quat[1],
            ",",
            quat[2],
            ",",
            quat[3],
            ")",
        )

    # Print joint angles
    print("\n  Joint angles (qpos[3:9]):")
    print("    bthigh:", final_qpos[3])
    print("    bshin:", final_qpos[4])
    print("    bfoot:", final_qpos[5])
    print("    fthigh:", final_qpos[6])
    print("    fshin:", final_qpos[7])
    print("    ffoot:", final_qpos[8])

    var final_bt_dist = (
        (Float64(final_bt[0]) - Float64(final_t[0])) ** 2
        + (Float64(final_bt[1]) - Float64(final_t[1])) ** 2
        + (Float64(final_bt[2]) - Float64(final_t[2])) ** 2
    ) ** 0.5
    var final_ft_dist = (
        (Float64(final_ft[0]) - Float64(final_t[0])) ** 2
        + (Float64(final_ft[1]) - Float64(final_t[1])) ** 2
        + (Float64(final_ft[2]) - Float64(final_t[2])) ** 2
    ) ** 0.5

    print("  BThigh distance from torso:", final_bt_dist)
    print("  FThigh distance from torso:", final_ft_dist)

    if final_bt_dist > 2.0 or final_ft_dist > 2.0:
        print("WARNING: Legs appear disconnected!")
    else:
        print("Legs appear properly connected to torso.")

    env.close()
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)
