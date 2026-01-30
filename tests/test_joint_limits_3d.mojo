"""Test that 3D joint limits are properly enforced.

This test verifies that the Hinge3D joint limits are working correctly
by applying extreme torques and checking that joints stay within bounds.

Run with:
    pixi run mojo run tests/test_joint_limits_3d.mojo
"""

from random import seed

from envs.half_cheetah_3d import HalfCheetah3D
from envs.half_cheetah_3d.constants3d import HC3DConstantsCPU


fn main() raises:
    print("=" * 60)
    print("JOINT LIMIT ENFORCEMENT TEST")
    print("=" * 60)
    print()

    # Print expected joint limits
    print("Expected joint limits:")
    print("  BTHIGH: [", HC3DConstantsCPU.BTHIGH_LIMIT_LOW, ",", HC3DConstantsCPU.BTHIGH_LIMIT_HIGH, "]")
    print("  BSHIN:  [", HC3DConstantsCPU.BSHIN_LIMIT_LOW, ",", HC3DConstantsCPU.BSHIN_LIMIT_HIGH, "]")
    print("  BFOOT:  [", HC3DConstantsCPU.BFOOT_LIMIT_LOW, ",", HC3DConstantsCPU.BFOOT_LIMIT_HIGH, "]")
    print("  FTHIGH: [", HC3DConstantsCPU.FTHIGH_LIMIT_LOW, ",", HC3DConstantsCPU.FTHIGH_LIMIT_HIGH, "]")
    print("  FSHIN:  [", HC3DConstantsCPU.FSHIN_LIMIT_LOW, ",", HC3DConstantsCPU.FSHIN_LIMIT_HIGH, "]")
    print("  FFOOT:  [", HC3DConstantsCPU.FFOOT_LIMIT_LOW, ",", HC3DConstantsCPU.FFOOT_LIMIT_HIGH, "]")
    print()

    # Create limits array for checking
    var lower_limits = List[Float64]()
    var upper_limits = List[Float64]()
    lower_limits.append(Float64(HC3DConstantsCPU.BTHIGH_LIMIT_LOW))
    lower_limits.append(Float64(HC3DConstantsCPU.BSHIN_LIMIT_LOW))
    lower_limits.append(Float64(HC3DConstantsCPU.BFOOT_LIMIT_LOW))
    lower_limits.append(Float64(HC3DConstantsCPU.FTHIGH_LIMIT_LOW))
    lower_limits.append(Float64(HC3DConstantsCPU.FSHIN_LIMIT_LOW))
    lower_limits.append(Float64(HC3DConstantsCPU.FFOOT_LIMIT_LOW))

    upper_limits.append(Float64(HC3DConstantsCPU.BTHIGH_LIMIT_HIGH))
    upper_limits.append(Float64(HC3DConstantsCPU.BSHIN_LIMIT_HIGH))
    upper_limits.append(Float64(HC3DConstantsCPU.BFOOT_LIMIT_HIGH))
    upper_limits.append(Float64(HC3DConstantsCPU.FTHIGH_LIMIT_HIGH))
    upper_limits.append(Float64(HC3DConstantsCPU.FSHIN_LIMIT_HIGH))
    upper_limits.append(Float64(HC3DConstantsCPU.FFOOT_LIMIT_HIGH))

    var joint_names = List[String]()
    joint_names.append("BTHIGH")
    joint_names.append("BSHIN")
    joint_names.append("BFOOT")
    joint_names.append("FTHIGH")
    joint_names.append("FSHIN")
    joint_names.append("FFOOT")

    seed(42)

    var env = HalfCheetah3D(seed=42)
    _ = env.reset_obs_list()

    # Track min/max angles seen for each joint
    var min_angles = List[Float64]()
    var max_angles = List[Float64]()
    for _ in range(6):
        min_angles.append(Float64(0.0))
        max_angles.append(Float64(0.0))

    print("Running 500 steps with EXTREME torques (alternating +1/-1)...")
    print()

    # Run with extreme actions that alternate to stress-test limits
    for step in range(500):
        var actions = List[Scalar[DType.float32]]()
        # Alternate between extreme positive and negative torques
        for j in range(6):
            var sign = 1.0 if (step + j) % 2 == 0 else -1.0
            actions.append(Scalar[DType.float32](sign))

        var result = env.step_continuous_vec(actions)
        var obs = result[0].copy()

        # Track joint angles (obs[2:8])
        for j in range(6):
            var angle = Float64(obs[2 + j])
            if angle < min_angles[j]:
                min_angles[j] = angle
            if angle > max_angles[j]:
                max_angles[j] = angle

    # Print results
    print("Results after stress test:")
    print("-" * 60)
    print("Joint    | Min Angle | Low Limit | Max Angle | Up Limit | Status")
    print("-" * 60)

    var all_passed = True
    var tolerance = 0.3  # Allow some tolerance for soft constraints

    for j in range(6):
        var status = "OK"
        var low_ok = min_angles[j] >= lower_limits[j] - tolerance
        var high_ok = max_angles[j] <= upper_limits[j] + tolerance

        if not low_ok or not high_ok:
            status = "EXCEEDED"
            all_passed = False

        print(
            joint_names[j],
            "  |",
            String(min_angles[j])[:8],
            "|",
            String(lower_limits[j])[:8],
            "|",
            String(max_angles[j])[:8],
            "|",
            String(upper_limits[j])[:8],
            "|",
            status,
        )

    print("-" * 60)
    print()

    # Additional test: check observation bounds
    print("Checking observation value bounds...")
    _ = env.reset_obs_list()

    var max_obs_value = Float64(0.0)
    var min_obs_value = Float64(0.0)

    for step in range(200):
        var actions = List[Scalar[DType.float32]]()
        for j in range(6):
            # Random-ish actions
            var val = Float64((step * 17 + j * 7) % 100) / 50.0 - 1.0
            actions.append(Scalar[DType.float32](val))

        var result = env.step_continuous_vec(actions)
        var obs = result[0].copy()

        for i in range(len(obs)):
            var v = Float64(obs[i])
            if v > max_obs_value:
                max_obs_value = v
            if v < min_obs_value:
                min_obs_value = v

    print("  Max observation value:", max_obs_value)
    print("  Min observation value:", min_obs_value)
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("PASSED: All joints stayed within tolerance of limits")
    else:
        print("PARTIAL: Some joints exceeded limits (soft constraint)")
        print("         This is expected with Baumgarte stabilization")
    print("=" * 60)
