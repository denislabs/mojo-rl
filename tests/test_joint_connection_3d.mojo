"""Test that 3D joints keep bodies connected.

This tests that the hinge joint anchor points stay together.
If bodies fly apart, there's a fundamental issue with the joint solver.

Run with:
    pixi run mojo run tests/test_joint_connection_3d.mojo
"""

from random import seed
from math import sqrt

from envs.half_cheetah_3d import HalfCheetah3D
from envs.half_cheetah_3d.constants3d import HC3DConstantsCPU
from physics3d.constants import (
    BODY_STATE_SIZE_3D,
    IDX_PX, IDX_PY, IDX_PZ,
    IDX_QW, IDX_QX, IDX_QY, IDX_QZ,
    JOINT_DATA_SIZE_3D,
    JOINT3D_ANCHOR_AX, JOINT3D_ANCHOR_AY, JOINT3D_ANCHOR_AZ,
    JOINT3D_ANCHOR_BX, JOINT3D_ANCHOR_BY, JOINT3D_ANCHOR_BZ,
    JOINT3D_BODY_A, JOINT3D_BODY_B,
)


fn rotate_vec_by_quat(
    qw: Float64, qx: Float64, qy: Float64, qz: Float64,
    vx: Float64, vy: Float64, vz: Float64
) -> Tuple[Float64, Float64, Float64]:
    """Rotate vector by quaternion."""
    # v' = v + 2*qw*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
    var cx = qy * vz - qz * vy
    var cy = qz * vx - qx * vz
    var cz = qx * vy - qy * vx
    var ccx = qy * cz - qz * cy
    var ccy = qz * cx - qx * cz
    var ccz = qx * cy - qy * cx
    return (
        vx + 2.0 * qw * cx + 2.0 * ccx,
        vy + 2.0 * qw * cy + 2.0 * ccy,
        vz + 2.0 * qw * cz + 2.0 * ccz,
    )


fn main() raises:
    print("=" * 60)
    print("JOINT CONNECTION TEST")
    print("=" * 60)
    print()
    print("Testing that joint anchor points stay connected...")
    print()

    seed(42)

    var env = HalfCheetah3D(seed=42)
    _ = env.reset_obs_list()

    var max_separation = Float64(0.0)
    var worst_joint = -1
    var worst_step = -1

    print("Running 200 steps with random actions...")
    print()

    for step in range(200):
        # Random actions
        var actions = List[Scalar[DType.float32]]()
        for j in range(6):
            var val = Float64((step * 17 + j * 7) % 100) / 50.0 - 1.0
            actions.append(Scalar[DType.float32](val))

        _ = env.step_continuous_vec(actions)

        # Check all joint anchor separations
        for joint_idx in range(6):
            var joint_off = HC3DConstantsCPU.JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

            var body_a = Int(env.state[joint_off + JOINT3D_BODY_A])
            var body_b = Int(env.state[joint_off + JOINT3D_BODY_B])

            var body_a_off = HC3DConstantsCPU.BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
            var body_b_off = HC3DConstantsCPU.BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

            # Get body positions
            var pa_x = Float64(env.state[body_a_off + IDX_PX])
            var pa_y = Float64(env.state[body_a_off + IDX_PY])
            var pa_z = Float64(env.state[body_a_off + IDX_PZ])
            var pb_x = Float64(env.state[body_b_off + IDX_PX])
            var pb_y = Float64(env.state[body_b_off + IDX_PY])
            var pb_z = Float64(env.state[body_b_off + IDX_PZ])

            # Get body orientations
            var qa_w = Float64(env.state[body_a_off + IDX_QW])
            var qa_x = Float64(env.state[body_a_off + IDX_QX])
            var qa_y = Float64(env.state[body_a_off + IDX_QY])
            var qa_z = Float64(env.state[body_a_off + IDX_QZ])
            var qb_w = Float64(env.state[body_b_off + IDX_QW])
            var qb_x = Float64(env.state[body_b_off + IDX_QX])
            var qb_y = Float64(env.state[body_b_off + IDX_QY])
            var qb_z = Float64(env.state[body_b_off + IDX_QZ])

            # Get local anchors
            var anchor_a_local_x = Float64(env.state[joint_off + JOINT3D_ANCHOR_AX])
            var anchor_a_local_y = Float64(env.state[joint_off + JOINT3D_ANCHOR_AY])
            var anchor_a_local_z = Float64(env.state[joint_off + JOINT3D_ANCHOR_AZ])
            var anchor_b_local_x = Float64(env.state[joint_off + JOINT3D_ANCHOR_BX])
            var anchor_b_local_y = Float64(env.state[joint_off + JOINT3D_ANCHOR_BY])
            var anchor_b_local_z = Float64(env.state[joint_off + JOINT3D_ANCHOR_BZ])

            # Transform to world space
            var ra = rotate_vec_by_quat(qa_w, qa_x, qa_y, qa_z, anchor_a_local_x, anchor_a_local_y, anchor_a_local_z)
            var rb = rotate_vec_by_quat(qb_w, qb_x, qb_y, qb_z, anchor_b_local_x, anchor_b_local_y, anchor_b_local_z)

            var anchor_a_world_x = pa_x + ra[0]
            var anchor_a_world_y = pa_y + ra[1]
            var anchor_a_world_z = pa_z + ra[2]
            var anchor_b_world_x = pb_x + rb[0]
            var anchor_b_world_y = pb_y + rb[1]
            var anchor_b_world_z = pb_z + rb[2]

            # Compute separation
            var dx = anchor_b_world_x - anchor_a_world_x
            var dy = anchor_b_world_y - anchor_a_world_y
            var dz = anchor_b_world_z - anchor_a_world_z
            var separation = sqrt(dx * dx + dy * dy + dz * dz)

            if separation > max_separation:
                max_separation = separation
                worst_joint = joint_idx
                worst_step = step

        # Print progress every 50 steps
        if step % 50 == 49:
            print("Step", step + 1, "| Max separation so far:", String(max_separation)[:10])

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("Max anchor separation:", max_separation)
    print("Worst joint:", worst_joint)
    print("At step:", worst_step)
    print()

    if max_separation < 0.01:
        print("PASSED: Joints stay connected (< 0.01m separation)")
    elif max_separation < 0.1:
        print("WARNING: Some joint drift (< 0.1m) - may affect physics")
    else:
        print("FAILED: Joints are disconnecting (> 0.1m separation)!")
        print("This explains why the cheetah flies apart.")
