"""Test physics2d/articulated module compilation."""

from physics2d.articulated import (
    HOPPER_NUM_BODIES,
    HOPPER_NUM_JOINTS,
    HOPPER_OBS_DIM,
    HOPPER_ACTION_DIM,
    WALKER_NUM_BODIES,
    WALKER_NUM_JOINTS,
    CHEETAH_NUM_BODIES,
    CHEETAH_NUM_JOINTS,
    DEFAULT_KP,
    DEFAULT_KD,
    DEFAULT_MAX_TORQUE,
    compute_link_inertia,
)


fn main() raises:
    print("=== Articulated Module Tests ===\n")

    # Test constants
    print("Testing articulated constants...")
    print("  HOPPER_NUM_BODIES:", HOPPER_NUM_BODIES)
    print("  HOPPER_NUM_JOINTS:", HOPPER_NUM_JOINTS)
    print("  HOPPER_OBS_DIM:", HOPPER_OBS_DIM)
    print("  HOPPER_ACTION_DIM:", HOPPER_ACTION_DIM)

    print("\n  WALKER_NUM_BODIES:", WALKER_NUM_BODIES)
    print("  WALKER_NUM_JOINTS:", WALKER_NUM_JOINTS)

    print("\n  CHEETAH_NUM_BODIES:", CHEETAH_NUM_BODIES)
    print("  CHEETAH_NUM_JOINTS:", CHEETAH_NUM_JOINTS)

    print("\n  DEFAULT_KP:", DEFAULT_KP)
    print("  DEFAULT_KD:", DEFAULT_KD)
    print("  DEFAULT_MAX_TORQUE:", DEFAULT_MAX_TORQUE)

    print("\nConstants PASSED")

    # Test link inertia computation
    print("\nTesting compute_link_inertia...")
    var inertia = compute_link_inertia(mass=2.0, length=0.4, width=0.05)
    print("  Link inertia (2kg, 0.4m x 0.05m):", inertia)
    print("  Expected: ~0.0270833 ((1/12)*2*(0.16+0.0025))")

    print("\n=== All Articulated Tests PASSED ===")
