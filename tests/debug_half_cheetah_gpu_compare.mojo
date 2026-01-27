"""Compare CPU and GPU physics step for HalfCheetah V2.

This test runs a single physics step on both CPU and GPU and compares results.
"""

from random import seed

from gpu.host import DeviceContext, DeviceBuffer
from layout import LayoutTensor, Layout

from envs.half_cheetah import HalfCheetahPlanarV2, HalfCheetahPlanarAction, HCConstants
from physics2d.constants import dtype, BODY_STATE_SIZE, IDX_X, IDX_Y, IDX_ANGLE, IDX_VX, IDX_VY, IDX_OMEGA


fn main() raises:
    print("=" * 60)
    print("Comparing CPU vs GPU Physics for HalfCheetah V2")
    print("=" * 60)

    # Verify layout constants
    print("\nLayout Constants:")
    print("  STATE_SIZE:", HCConstants.STATE_SIZE_VAL)
    print("  OBS_DIM:", HCConstants.OBS_DIM_VAL)
    print("  NUM_BODIES:", HCConstants.NUM_BODIES)
    print("  BODIES_OFFSET:", HCConstants.BODIES_OFFSET)
    print("  FORCES_OFFSET:", HCConstants.FORCES_OFFSET)
    print("  JOINTS_OFFSET:", HCConstants.JOINTS_OFFSET)
    print("  JOINT_COUNT_OFFSET:", HCConstants.JOINT_COUNT_OFFSET)
    print("  METADATA_OFFSET:", HCConstants.METADATA_OFFSET)

    # Create CPU environment
    print("\n--- CPU Environment ---")
    var cpu_env = HalfCheetahPlanarV2[dtype]()
    var cpu_state = cpu_env.reset()

    print("Initial CPU state:")
    print("  Torso Y:", cpu_state.torso_z)
    print("  Torso Angle:", cpu_state.torso_angle)
    print("  Vel X:", cpu_state.vel_x)

    # Run CPU steps
    print("\n--- CPU Steps ---")
    var cpu_action = HalfCheetahPlanarAction[dtype](0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for step in range(5):
        var result = cpu_env._step_cpu_continuous(cpu_action)
        var reward = Float64(result[0])
        print("Step", step + 1, ": reward =", reward,
              ", torso_y =", cpu_env.cached_state.torso_z,
              ", vel_x =", cpu_env.cached_state.vel_x)

    # GPU comparison
    print("\n--- GPU Environment ---")

    with DeviceContext() as ctx:
        # Allocate GPU buffers
        comptime BATCH = 1
        comptime STATE_SIZE = HCConstants.STATE_SIZE_VAL
        comptime OBS_DIM = HCConstants.OBS_DIM_VAL
        comptime ACTION_DIM = 6

        var states_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
        var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH)

        # Reset on GPU
        HalfCheetahPlanarV2[dtype].reset_kernel_gpu[BATCH, STATE_SIZE](
            ctx, states_buf, rng_seed=42
        )
        ctx.synchronize()

        # Read back initial state
        var host_state = List[Scalar[dtype]](capacity=STATE_SIZE)
        for _ in range(STATE_SIZE):
            host_state.append(Scalar[dtype](0))
        states_buf.enqueue_copy_to(host_state.unsafe_ptr())
        ctx.synchronize()

        # Print initial GPU state
        var torso_off = HCConstants.BODIES_OFFSET
        print("Initial GPU state:")
        print("  Torso X:", host_state[torso_off + IDX_X])
        print("  Torso Y:", host_state[torso_off + IDX_Y])
        print("  Torso Angle:", host_state[torso_off + IDX_ANGLE])
        print("  Torso VX:", host_state[torso_off + IDX_VX])
        print("  Torso VY:", host_state[torso_off + IDX_VY])

        # Print body shape indices
        print("\nGPU Body Shape Indices:")
        for body in range(HCConstants.NUM_BODIES):
            var body_off = HCConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
            print("  Body", body, "shape:", host_state[body_off + 12])  # IDX_SHAPE = 12

        # Print joint count
        print("\nJoint count:", host_state[HCConstants.JOINT_COUNT_OFFSET])

        # Print joint data for first joint
        print("\nJoint 0 data:")
        var joint_off = HCConstants.JOINTS_OFFSET
        from physics2d.constants import (
            JOINT_TYPE, JOINT_BODY_A, JOINT_BODY_B,
            JOINT_ANCHOR_AX, JOINT_ANCHOR_AY, JOINT_ANCHOR_BX, JOINT_ANCHOR_BY,
            JOINT_REF_ANGLE, JOINT_LOWER_LIMIT, JOINT_UPPER_LIMIT,
            JOINT_MAX_MOTOR_TORQUE, JOINT_MOTOR_SPEED, JOINT_FLAGS,
            JOINT_DATA_SIZE
        )
        print("  Type:", host_state[joint_off + JOINT_TYPE], "(should be 0 for REVOLUTE)")
        print("  Body A:", host_state[joint_off + JOINT_BODY_A])
        print("  Body B:", host_state[joint_off + JOINT_BODY_B])
        print("  Anchor A (x,y):", host_state[joint_off + JOINT_ANCHOR_AX], ",", host_state[joint_off + JOINT_ANCHOR_AY])
        print("  Anchor B (x,y):", host_state[joint_off + JOINT_ANCHOR_BX], ",", host_state[joint_off + JOINT_ANCHOR_BY])
        print("  Ref angle:", host_state[joint_off + JOINT_REF_ANGLE])
        print("  Limits:", host_state[joint_off + JOINT_LOWER_LIMIT], "to", host_state[joint_off + JOINT_UPPER_LIMIT])
        print("  Max motor torque:", host_state[joint_off + JOINT_MAX_MOTOR_TORQUE])
        print("  Motor speed:", host_state[joint_off + JOINT_MOTOR_SPEED])
        print("  Flags:", host_state[joint_off + JOINT_FLAGS], "(should have bits for limit+motor enabled)")

        # Also print forces to see if they're being cleared
        print("\nForces for torso (should be 0,0,0):")
        var force_off = HCConstants.FORCES_OFFSET
        print("  fx:", host_state[force_off + 0])
        print("  fy:", host_state[force_off + 1])
        print("  tau:", host_state[force_off + 2])

        # Set zero actions
        var host_actions = List[Scalar[dtype]](capacity=ACTION_DIM)
        for _ in range(ACTION_DIM):
            host_actions.append(Scalar[dtype](0))
        actions_buf.enqueue_copy_from(host_actions.unsafe_ptr())
        ctx.synchronize()

        # Run GPU steps
        print("\n--- GPU Steps ---")
        var host_rewards = List[Scalar[dtype]](capacity=1)
        host_rewards.append(Scalar[dtype](0))

        # Now test with non-zero action
        print("\n--- Testing with action = 0.5 ---")
        for i in range(ACTION_DIM):
            host_actions[i] = Scalar[dtype](0.5)
        actions_buf.enqueue_copy_from(host_actions.unsafe_ptr())
        ctx.synchronize()

        for step in range(5):
            # Step on GPU
            HalfCheetahPlanarV2[dtype].step_kernel_gpu[
                BATCH, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                obs_buf,
                rng_seed=42
            )
            ctx.synchronize()

            # Read reward
            rewards_buf.enqueue_copy_to(host_rewards.unsafe_ptr())
            ctx.synchronize()

            # Read state
            states_buf.enqueue_copy_to(host_state.unsafe_ptr())
            ctx.synchronize()

            print("Step", step + 1, ": reward =", host_rewards[0],
                  ", torso_y =", host_state[torso_off + IDX_Y],
                  ", vel_x =", host_state[torso_off + IDX_VX],
                  ", torso_omega =", host_state[torso_off + IDX_OMEGA])

        # Final state comparison
        print("\n--- Final GPU Body States ---")
        for body in range(HCConstants.NUM_BODIES):
            var body_off = HCConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
            print("Body", body, ":")
            print("  X:", host_state[body_off + IDX_X])
            print("  Y:", host_state[body_off + IDX_Y])
            print("  Angle:", host_state[body_off + IDX_ANGLE])
            print("  VX:", host_state[body_off + IDX_VX])
            print("  VY:", host_state[body_off + IDX_VY])
            print("  Omega:", host_state[body_off + IDX_OMEGA])

        # Check joint anchor alignment
        print("\n--- Joint Anchor Alignment Check ---")
        from math import cos, sin
        # Joint 0: Torso (0) to BThigh (1)
        # Anchor A on torso: (-0.5, 0)
        # Anchor B on bthigh: (0, 0.0725)
        var torso_x = Float64(host_state[torso_off + IDX_X])
        var torso_y = Float64(host_state[torso_off + IDX_Y])
        var torso_angle = Float64(host_state[torso_off + IDX_ANGLE])

        var bthigh_off = HCConstants.BODIES_OFFSET + 1 * BODY_STATE_SIZE
        var bthigh_x = Float64(host_state[bthigh_off + IDX_X])
        var bthigh_y = Float64(host_state[bthigh_off + IDX_Y])
        var bthigh_angle = Float64(host_state[bthigh_off + IDX_ANGLE])

        # Torso anchor in world
        var anchor_a_local_x = -0.5
        var anchor_a_local_y = 0.0
        var cos_t = cos(torso_angle)
        var sin_t = sin(torso_angle)
        var anchor_a_world_x = torso_x + anchor_a_local_x * cos_t - anchor_a_local_y * sin_t
        var anchor_a_world_y = torso_y + anchor_a_local_x * sin_t + anchor_a_local_y * cos_t

        # Bthigh anchor in world
        var anchor_b_local_x = 0.0
        var anchor_b_local_y = 0.0725
        var cos_b = cos(bthigh_angle)
        var sin_b = sin(bthigh_angle)
        var anchor_b_world_x = bthigh_x + anchor_b_local_x * cos_b - anchor_b_local_y * sin_b
        var anchor_b_world_y = bthigh_y + anchor_b_local_x * sin_b + anchor_b_local_y * cos_b

        print("Joint 0 (torso-bthigh):")
        print("  Anchor A world: (", anchor_a_world_x, ",", anchor_a_world_y, ")")
        print("  Anchor B world: (", anchor_b_world_x, ",", anchor_b_world_y, ")")
        print("  Separation: dx =", anchor_b_world_x - anchor_a_world_x,
              ", dy =", anchor_b_world_y - anchor_a_world_y)

        # Check inv_mass values
        print("\nBody inverse masses:")
        for body in range(HCConstants.NUM_BODIES):
            var b_off = HCConstants.BODIES_OFFSET + body * BODY_STATE_SIZE
            print("  Body", body, "inv_mass:", host_state[b_off + 10],
                  "inv_inertia:", host_state[b_off + 11])

    print("\n" + "=" * 60)
    print("Comparison Complete")
    print("=" * 60)
