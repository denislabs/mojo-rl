"""Contact detection for physics engine.

Provides ground contact detection, body-body contact detection,
and quaternion normalization utilities.
"""

from math import sqrt
from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_normalize, quat_rotate


fn normalize_qpos_quaternions[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Normalize quaternions in qpos for BALL and FREE joints."""
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var qpos_adr = joint.qpos_adr

        if joint.jnt_type == JNT_FREE:
            # Quaternion at qpos_adr + 3..6
            var qx = data.qpos[qpos_adr + 3]
            var qy = data.qpos[qpos_adr + 4]
            var qz = data.qpos[qpos_adr + 5]
            var qw = data.qpos[qpos_adr + 6]

            var normalized = quat_normalize(qx, qy, qz, qw)
            data.qpos[qpos_adr + 3] = normalized[0]
            data.qpos[qpos_adr + 4] = normalized[1]
            data.qpos[qpos_adr + 5] = normalized[2]
            data.qpos[qpos_adr + 6] = normalized[3]

        elif joint.jnt_type == JNT_BALL:
            # Quaternion at qpos_adr + 0..3
            var qx = data.qpos[qpos_adr + 0]
            var qy = data.qpos[qpos_adr + 1]
            var qz = data.qpos[qpos_adr + 2]
            var qw = data.qpos[qpos_adr + 3]

            var normalized = quat_normalize(qx, qy, qz, qw)
            data.qpos[qpos_adr + 0] = normalized[0]
            data.qpos[qpos_adr + 1] = normalized[1]
            data.qpos[qpos_adr + 2] = normalized[2]
            data.qpos[qpos_adr + 3] = normalized[3]


fn detect_ground_contacts[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Detect contacts between bodies and ground plane.

    For capsules, checks both endpoints (center ± half_length along axis).
    The capsule axis is determined by the body's world orientation.
    """
    data.num_contacts = 0
    var ground_z = model.ground_z

    for body in range(NBODY):
        var px = data.xpos[body * 3 + 0]
        var py = data.xpos[body * 3 + 1]
        var pz = data.xpos[body * 3 + 2]
        var radius = model.body_radius[body]
        var half_length = model.body_half_length[body]

        # Get body orientation
        var qx = data.xquat[body * 4 + 0]
        var qy = data.xquat[body * 4 + 1]
        var qz = data.xquat[body * 4 + 2]
        var qw = data.xquat[body * 4 + 3]

        # Capsule axis in local frame is (0, 0, 1) - along Z
        # Transform to world frame
        var axis_world = quat_rotate(qx, qy, qz, qw,
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1))
        var axis_x = axis_world[0]
        var axis_y = axis_world[1]
        var axis_z = axis_world[2]

        # For spheres (half_length = 0), just check center - radius
        if half_length <= Scalar[DTYPE](0.0001):
            var dist = pz - radius - ground_z
            if dist < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = px
                    data.contacts[idx].pos_y = py
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist
                    data.num_contacts += 1
        else:
            # Capsule: check both endpoints
            # Endpoint 1: center + half_length * axis
            var e1_x = px + half_length * axis_x
            var e1_y = py + half_length * axis_y
            var e1_z = pz + half_length * axis_z
            var dist1 = e1_z - radius - ground_z

            # Endpoint 2: center - half_length * axis
            var e2_x = px - half_length * axis_x
            var e2_y = py - half_length * axis_y
            var e2_z = pz - half_length * axis_z
            var dist2 = e2_z - radius - ground_z

            # Check endpoint 1
            if dist1 < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = e1_x
                    data.contacts[idx].pos_y = e1_y
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist1
                    data.num_contacts += 1

            # Check endpoint 2
            if dist2 < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = e2_x
                    data.contacts[idx].pos_y = e2_y
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist2
                    data.num_contacts += 1


fn detect_body_body_contacts[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Detect body-body contacts and append to existing contact list.

    Does NOT reset num_contacts — appends to contacts from ground detection.
    O(N^2) pair iteration, skipping parent-child pairs.
    Dispatches to collision primitives based on geometry types.
    """
    from ..constants import GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX
    from ..collision.collision_primitives import (
        sphere_sphere,
        capsule_sphere,
        capsule_capsule,
        box_sphere,
        box_capsule,
    )

    for i in range(NBODY):
        for j in range(i + 1, NBODY):
            # Skip parent-child pairs (connected bodies)
            if model.body_parent[j] == i or model.body_parent[i] == j:
                continue

            # Skip if already at max contacts
            if data.num_contacts >= MAX_CONTACTS:
                return

            var gi = model.body_geom_type[i]
            var gj = model.body_geom_type[j]

            # Get positions
            var pi_x = data.xpos[i * 3 + 0]
            var pi_y = data.xpos[i * 3 + 1]
            var pi_z = data.xpos[i * 3 + 2]
            var pj_x = data.xpos[j * 3 + 0]
            var pj_y = data.xpos[j * 3 + 1]
            var pj_z = data.xpos[j * 3 + 2]

            # Get quaternions
            var qi_x = data.xquat[i * 4 + 0]
            var qi_y = data.xquat[i * 4 + 1]
            var qi_z = data.xquat[i * 4 + 2]
            var qi_w = data.xquat[i * 4 + 3]
            var qj_x = data.xquat[j * 4 + 0]
            var qj_y = data.xquat[j * 4 + 1]
            var qj_z = data.xquat[j * 4 + 2]
            var qj_w = data.xquat[j * 4 + 3]

            var dist: Scalar[DTYPE] = 1.0  # Default: no contact
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
            var swap = False  # Whether we swapped body order for the primitive

            # Dispatch based on geometry pair
            if gi == GEOM_SPHERE and gj == GEOM_SPHERE:
                var result = sphere_sphere[DTYPE](
                    pi_x, pi_y, pi_z, model.body_radius[i],
                    pj_x, pj_y, pj_z, model.body_radius[j],
                )
                dist = result[0]; cx = result[1]; cy = result[2]; cz = result[3]
                nx = result[4]; ny = result[5]; nz = result[6]

            elif gi == GEOM_CAPSULE and gj == GEOM_SPHERE:
                var result = capsule_sphere[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                    model.body_half_length[i], model.body_radius[i],
                    pj_x, pj_y, pj_z, model.body_radius[j],
                )
                dist = result[0]; cx = result[1]; cy = result[2]; cz = result[3]
                nx = result[4]; ny = result[5]; nz = result[6]

            elif gi == GEOM_SPHERE and gj == GEOM_CAPSULE:
                # Swap: call capsule_sphere with j as capsule, negate normal
                var result = capsule_sphere[DTYPE](
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                    model.body_half_length[j], model.body_radius[j],
                    pi_x, pi_y, pi_z, model.body_radius[i],
                )
                dist = result[0]; cx = result[1]; cy = result[2]; cz = result[3]
                nx = -result[4]; ny = -result[5]; nz = -result[6]
                swap = True

            elif gi == GEOM_CAPSULE and gj == GEOM_CAPSULE:
                var result = capsule_capsule[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                    model.body_half_length[i], model.body_radius[i],
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                    model.body_half_length[j], model.body_radius[j],
                )
                dist = result[0]; cx = result[1]; cy = result[2]; cz = result[3]
                nx = result[4]; ny = result[5]; nz = result[6]

            elif gi == GEOM_BOX and gj == GEOM_SPHERE:
                var result = box_sphere[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                    model.body_half_x[i], model.body_half_y[i], model.body_half_z[i],
                    pj_x, pj_y, pj_z, model.body_radius[j],
                )
                dist = result[0]; cx = result[1]; cy = result[2]; cz = result[3]
                nx = result[4]; ny = result[5]; nz = result[6]

            elif gi == GEOM_SPHERE and gj == GEOM_BOX:
                # Swap: call box_sphere with j as box, negate normal
                var result = box_sphere[DTYPE](
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                    model.body_half_x[j], model.body_half_y[j], model.body_half_z[j],
                    pi_x, pi_y, pi_z, model.body_radius[i],
                )
                dist = result[0]; cx = result[1]; cy = result[2]; cz = result[3]
                nx = -result[4]; ny = -result[5]; nz = -result[6]
                swap = True

            elif gi == GEOM_BOX and gj == GEOM_CAPSULE:
                var result = box_capsule[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                    model.body_half_x[i], model.body_half_y[i], model.body_half_z[i],
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                    model.body_half_length[j], model.body_radius[j],
                )
                dist = result[0]; cx = result[1]; cy = result[2]; cz = result[3]
                nx = result[4]; ny = result[5]; nz = result[6]

            elif gi == GEOM_CAPSULE and gj == GEOM_BOX:
                # Swap: call box_capsule with j as box, negate normal
                var result = box_capsule[DTYPE](
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                    model.body_half_x[j], model.body_half_y[j], model.body_half_z[j],
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                    model.body_half_length[i], model.body_radius[i],
                )
                dist = result[0]; cx = result[1]; cy = result[2]; cz = result[3]
                nx = -result[4]; ny = -result[5]; nz = -result[6]
                swap = True

            # Store contact if penetrating
            if dist < Scalar[DTYPE](0) and data.num_contacts < MAX_CONTACTS:
                var idx = data.num_contacts
                if swap:
                    data.contacts[idx].body_a = j
                    data.contacts[idx].body_b = i
                else:
                    data.contacts[idx].body_a = i
                    data.contacts[idx].body_b = j
                data.contacts[idx].pos_x = cx
                data.contacts[idx].pos_y = cy
                data.contacts[idx].pos_z = cz
                data.contacts[idx].normal_x = nx
                data.contacts[idx].normal_y = ny
                data.contacts[idx].normal_z = nz
                data.contacts[idx].dist = dist
                data.num_contacts += 1
