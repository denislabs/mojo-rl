"""Diagnostic: test physics stability with constant max-torque actions.

Tests if the physics diverges under extreme (but constant) action inputs,
to distinguish between a physics bug and a policy mismatch issue.
"""

from random import seed
from envs.half_cheetah import HalfCheetah
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    JOINT_BTHIGH,
    JOINT_BSHIN,
    JOINT_BFOOT,
    JOINT_FTHIGH,
    JOINT_FSHIN,
    JOINT_FFOOT,
)

comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV


fn print_state(env: HalfCheetah, step: Int):
    """Print key state variables."""
    print("Step", step, end="")
    print("  rootx=", env.data.qpos[JOINT_ROOTX], end="")
    print("  rootz=", env.data.qpos[JOINT_ROOTZ], end="")
    print("  rooty=", env.data.qpos[JOINT_ROOTY], end="")
    print("  bthigh=", env.data.qpos[JOINT_BTHIGH], end="")
    print("  bshin=", env.data.qpos[JOINT_BSHIN], end="")
    print("  fthigh=", env.data.qpos[JOINT_FTHIGH], end="")
    print()
    # Print velocities
    print("       ", end="")
    print("  vx=", env.data.qvel[JOINT_ROOTX], end="")
    print("  vz=", env.data.qvel[JOINT_ROOTZ], end="")
    print("  vy=", env.data.qvel[JOINT_ROOTY], end="")
    print("  v_bthigh=", env.data.qvel[JOINT_BTHIGH], end="")
    print("  v_bshin=", env.data.qvel[JOINT_BSHIN], end="")
    print("  v_fthigh=", env.data.qvel[JOINT_FTHIGH], end="")
    print()

    # Check for NaN or Inf
    for i in range(NV):
        var v = env.data.qvel[i]
        var p = env.data.qpos[i]
        if v != v or p != p:  # NaN check
            print("  *** NaN detected at DOF", i, "***")
        if v > 1e10 or v < -1e10 or p > 1e10 or p < -1e10:
            print("  *** OVERFLOW at DOF", i, "qpos=", p, "qvel=", v, "***")


fn test_constant_actions(action_val: Float64, label: String):
    """Test with constant actions."""
    print("\n" + "=" * 70)
    print("Test:", label, "  action_value=", action_val)
    print("=" * 70)

    var env = HalfCheetah()
    _ = env.reset()

    var action = env.ActionType()
    for i in range(6):
        action[i] = action_val

    print_state(env, 0)

    for step in range(1, 201):
        _ = env.step(action)
        if step <= 10 or step % 20 == 0:
            print_state(env, step)


fn test_alternating_actions():
    """Test with alternating max-torque actions (worst case for resonance)."""
    print("\n" + "=" * 70)
    print("Test: Alternating +1/-1 actions (resonance test)")
    print("=" * 70)

    var env = HalfCheetah()
    _ = env.reset()

    print_state(env, 0)

    for step in range(1, 201):
        var sign = Float64(1.0) if step % 2 == 0 else Float64(-1.0)
        var action = env.ActionType()
        for i in range(6):
            action[i] = sign
        _ = env.step(action)
        if step <= 10 or step % 20 == 0:
            print_state(env, step)


fn test_extreme_single_joint():
    """Test with max torque on single joint (isolate coupling effects)."""
    print("\n" + "=" * 70)
    print("Test: Max torque on bthigh only (coupling test)")
    print("=" * 70)

    var env = HalfCheetah()
    _ = env.reset()

    print_state(env, 0)

    for step in range(1, 201):
        var action = env.ActionType()
        action[0] = 1.0  # bthigh only
        _ = env.step(action)
        if step <= 10 or step % 20 == 0:
            print_state(env, step)


fn main():
    seed(42)
    print("HalfCheetah Physics Stability Diagnostic")
    print("NQ=", NQ, "NV=", NV)
    print("Tests whether physics diverges with extreme actions\n")

    # Test 1: All max positive torques
    test_constant_actions(1.0, "All actions = +1.0 (max positive)")

    # Test 2: All max negative torques
    test_constant_actions(-1.0, "All actions = -1.0 (max negative)")

    # Test 3: Alternating (worst case for resonance)
    test_alternating_actions()

    # Test 4: Single joint max torque
    test_extreme_single_joint()

    print("\n\nDiagnostic complete.")
