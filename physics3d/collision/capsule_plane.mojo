"""Capsule vs Plane collision detection.

Detects collision between a capsule and an infinite plane (ground).
Capsules are commonly used for limbs in MuJoCo environments.

GPU support follows the three-method hierarchy pattern for writing
contacts directly to flat buffer arrays.
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from math3d import Vec3, Quat
from .contact3d import Contact3D, ContactManifold

from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE_3D,
    CONTACT_DATA_SIZE_3D,
    IDX_PX, IDX_PY, IDX_PZ,
    IDX_QW, IDX_QX, IDX_QY, IDX_QZ,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    SHAPE_CAPSULE,
    CONTACT_BODY_A_3D, CONTACT_BODY_B_3D,
    CONTACT_POINT_X, CONTACT_POINT_Y, CONTACT_POINT_Z,
    CONTACT_NORMAL_X, CONTACT_NORMAL_Y, CONTACT_NORMAL_Z,
    CONTACT_DEPTH_3D,
    CONTACT_IMPULSE_N, CONTACT_IMPULSE_T1, CONTACT_IMPULSE_T2,
    CONTACT_TANGENT1_X, CONTACT_TANGENT1_Y, CONTACT_TANGENT1_Z,
)


struct CapsulePlaneCollision:
    """Capsule vs Plane collision detection and contact generation.

    A capsule is defined by:
    - Center position
    - Orientation (quaternion)
    - Radius
    - Half-height (of cylindrical part)
    - Axis (0=X, 1=Y, 2=Z in local frame)
    """

    @staticmethod
    fn get_capsule_endpoints(
        center: Vec3,
        orientation: Quat,
        half_height: Float64,
        axis: Int,
    ) -> Tuple[Vec3, Vec3]:
        """Get world-space endpoints of capsule axis.

        Args:
            center: Capsule center position.
            orientation: Capsule orientation.
            half_height: Half-height of cylindrical part.
            axis: Local axis direction (0=X, 1=Y, 2=Z).

        Returns:
            Tuple of (endpoint_a, endpoint_b) in world space.
        """
        # Get local axis direction
        var local_axis: Vec3
        if axis == 0:
            local_axis = Vec3.unit_x()
        elif axis == 1:
            local_axis = Vec3.unit_y()
        else:
            local_axis = Vec3.unit_z()

        # Transform to world space
        var world_axis = orientation.rotate_vec(local_axis)

        var endpoint_a = center - world_axis * half_height
        var endpoint_b = center + world_axis * half_height

        return (endpoint_a, endpoint_b)

    @staticmethod
    fn detect(
        center: Vec3,
        orientation: Quat,
        radius: Float64,
        half_height: Float64,
        axis: Int,
        plane_normal: Vec3,
        plane_offset: Float64,
        body_idx: Int,
    ) -> ContactManifold:
        """Detect collision between capsule and plane.

        A capsule can generate 1-2 contact points depending on orientation.

        Args:
            center: Capsule center position.
            orientation: Capsule orientation quaternion.
            radius: Capsule radius.
            half_height: Half-height of cylindrical part.
            axis: Local axis (0=X, 1=Y, 2=Z).
            plane_normal: Plane normal (normalized).
            plane_offset: Plane offset from origin.
            body_idx: Index of capsule body.

        Returns:
            ContactManifold with 0-2 contact points.
        """
        var manifold = ContactManifold(body_idx, -1)

        # Get capsule endpoints
        var endpoints = Self.get_capsule_endpoints(center, orientation, half_height, axis)
        var p0 = endpoints[0]
        var p1 = endpoints[1]

        var n = plane_normal.normalized()

        # Check each endpoint (as sphere)
        for i in range(2):
            var point = p0 if i == 0 else p1

            # Distance from endpoint to plane
            var dist = n.dot(point) + plane_offset
            var depth = radius - dist

            if depth > 0.0:
                # Contact point on capsule surface toward plane
                var contact_point = point - n * dist

                var contact = Contact3D(
                    body_a=body_idx,
                    body_b=-1,
                    point=contact_point,
                    normal=n,
                    depth=depth,
                )
                manifold.add_contact(contact^)

        return manifold^

    @staticmethod
    fn detect_ground(
        center: Vec3,
        orientation: Quat,
        radius: Float64,
        half_height: Float64,
        axis: Int,
        ground_height: Float64,
        body_idx: Int,
    ) -> ContactManifold:
        """Simplified detection for horizontal ground plane.

        Args:
            center: Capsule center.
            orientation: Capsule orientation.
            radius: Capsule radius.
            half_height: Half-height of cylindrical part.
            axis: Local axis (0=X, 1=Y, 2=Z).
            ground_height: Z-coordinate of ground.
            body_idx: Body index.

        Returns:
            ContactManifold with contacts.
        """
        return Self.detect(
            center,
            orientation,
            radius,
            half_height,
            axis,
            Vec3.unit_z(),
            -ground_height,
            body_idx,
        )

    @staticmethod
    fn detect_simple(
        center: Vec3,
        radius: Float64,
        ground_height: Float64,
        body_idx: Int,
    ) -> Contact3D:
        """Simplified spherical approximation of capsule.

        Treats the capsule as a sphere centered at its midpoint.
        Useful for quick-and-dirty collision detection.

        Args:
            center: Capsule center.
            radius: Capsule radius (used as sphere radius).
            ground_height: Z-coordinate of ground.
            body_idx: Body index.

        Returns:
            Single contact point.
        """
        var sphere_bottom = center.z - radius
        var depth = ground_height - sphere_bottom

        if depth <= 0.0:
            return Contact3D()

        var contact_point = Vec3(center.x, center.y, ground_height)

        return Contact3D(
            body_a=body_idx,
            body_b=-1,
            point=contact_point,
            normal=Vec3.unit_z(),
            depth=depth,
        )


# =============================================================================
# Batch Collision Detection
# =============================================================================


fn detect_capsules_ground(
    centers: List[Vec3],
    orientations: List[Quat],
    radii: List[Float64],
    half_heights: List[Float64],
    axes: List[Int],
    body_indices: List[Int],
    ground_height: Float64,
) -> List[Contact3D]:
    """Detect ground collisions for multiple capsules.

    Returns flat list of all contacts across all capsules.
    """
    var contacts = List[Contact3D]()

    for i in range(len(centers)):
        var manifold = CapsulePlaneCollision.detect_ground(
            centers[i],
            orientations[i],
            radii[i],
            half_heights[i],
            axes[i],
            ground_height,
            body_indices[i],
        )

        # Add all contacts from manifold
        for j in range(len(manifold.contacts)):
            contacts.append(manifold.contacts[j])

    return contacts


# =============================================================================
# GPU Implementation
# =============================================================================


struct CapsulePlaneCollisionGPU:
    """GPU-compatible Capsule vs Plane collision detection.

    Writes contact data directly to flat state buffer arrays.
    Capsules can generate 1-2 contact points (one per endpoint).
    """

    # =========================================================================
    # Helper: Rotate vector by quaternion (inline)
    # =========================================================================

    @always_inline
    @staticmethod
    fn rotate_vec_by_quat(
        qw: Scalar[dtype],
        qx: Scalar[dtype],
        qy: Scalar[dtype],
        qz: Scalar[dtype],
        vx: Scalar[dtype],
        vy: Scalar[dtype],
        vz: Scalar[dtype],
    ) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
        """Rotate a vector by a quaternion: q * v * q^-1.

        Uses the formula: v' = v + 2*q_w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
        """
        # Cross product: q_xyz x v
        var cx1 = qy * vz - qz * vy
        var cy1 = qz * vx - qx * vz
        var cz1 = qx * vy - qy * vx

        # Cross product: q_xyz x (q_xyz x v)
        var cx2 = qy * cz1 - qz * cy1
        var cy2 = qz * cx1 - qx * cz1
        var cz2 = qx * cy1 - qy * cx1

        # v' = v + 2*w*c1 + 2*c2
        var two = Scalar[dtype](2.0)
        var rx = vx + two * qw * cx1 + two * cx2
        var ry = vy + two * qw * cy1 + two * cy2
        var rz = vz + two * qw * cz1 + two * cz2

        return (rx, ry, rz)

    # =========================================================================
    # Single-Environment Methods (can be called from fused kernels)
    # =========================================================================

    @always_inline
    @staticmethod
    fn detect_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shape_types: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        shape_radii: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        shape_half_heights: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        shape_axes: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ],
        ground_height: Scalar[dtype],
    ):
        """Detect capsule-ground collisions for a single environment.

        Generates up to 2 contacts per capsule (one for each endpoint).

        Args:
            env: Environment index.
            state: State buffer [BATCH, STATE_SIZE].
            shape_types: Shape type for each body.
            shape_radii: Radius for each body.
            shape_half_heights: Half-height of cylindrical part for capsules.
            shape_axes: Local axis for capsules (0=X, 1=Y, 2=Z).
            contacts: Contact buffer.
            contact_counts: Contact count per environment.
            ground_height: Z-coordinate of ground plane.
        """
        var contact_count = Int(contact_counts[env])

        @parameter
        for body in range(NUM_BODIES):
            # Skip if not a capsule
            if Int(shape_types[body]) != SHAPE_CAPSULE:
                continue

            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE_3D

            # Skip if not dynamic
            var body_type = Int(state[env, body_off + IDX_BODY_TYPE])
            if body_type != BODY_DYNAMIC:
                continue

            # Get capsule center position
            var px = state[env, body_off + IDX_PX]
            var py = state[env, body_off + IDX_PY]
            var pz = state[env, body_off + IDX_PZ]

            # Get quaternion orientation
            var qw = state[env, body_off + IDX_QW]
            var qx = state[env, body_off + IDX_QX]
            var qy = state[env, body_off + IDX_QY]
            var qz = state[env, body_off + IDX_QZ]

            # Get capsule parameters
            var radius = shape_radii[body]
            var half_height = shape_half_heights[body]
            var axis = Int(shape_axes[body])

            # Get local axis direction
            var local_ax = Scalar[dtype](0)
            var local_ay = Scalar[dtype](0)
            var local_az = Scalar[dtype](0)
            if axis == 0:
                local_ax = Scalar[dtype](1)  # X axis
            elif axis == 1:
                local_ay = Scalar[dtype](1)  # Y axis
            else:
                local_az = Scalar[dtype](1)  # Z axis

            # Rotate to world space
            var world_axis = Self.rotate_vec_by_quat(qw, qx, qy, qz, local_ax, local_ay, local_az)
            var world_ax = world_axis[0]
            var world_ay = world_axis[1]
            var world_az = world_axis[2]

            # Compute endpoints
            var p0_x = px - world_ax * half_height
            var p0_y = py - world_ay * half_height
            var p0_z = pz - world_az * half_height

            var p1_x = px + world_ax * half_height
            var p1_y = py + world_ay * half_height
            var p1_z = pz + world_az * half_height

            # Check each endpoint for collision with ground
            # Endpoint 0
            var dist0 = p0_z - ground_height
            var depth0 = radius - dist0
            if depth0 > Scalar[dtype](0) and contact_count < MAX_CONTACTS:
                var contact_off = contact_count

                contacts[env, contact_off, CONTACT_BODY_A_3D] = Scalar[dtype](body)
                contacts[env, contact_off, CONTACT_BODY_B_3D] = Scalar[dtype](-1)

                contacts[env, contact_off, CONTACT_POINT_X] = p0_x
                contacts[env, contact_off, CONTACT_POINT_Y] = p0_y
                contacts[env, contact_off, CONTACT_POINT_Z] = ground_height

                contacts[env, contact_off, CONTACT_NORMAL_X] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_NORMAL_Y] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_NORMAL_Z] = Scalar[dtype](1)

                contacts[env, contact_off, CONTACT_DEPTH_3D] = depth0

                contacts[env, contact_off, CONTACT_IMPULSE_N] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_IMPULSE_T1] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_IMPULSE_T2] = Scalar[dtype](0)

                contacts[env, contact_off, CONTACT_TANGENT1_X] = Scalar[dtype](1)
                contacts[env, contact_off, CONTACT_TANGENT1_Y] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_TANGENT1_Z] = Scalar[dtype](0)

                contact_count += 1

            # Endpoint 1
            var dist1 = p1_z - ground_height
            var depth1 = radius - dist1
            if depth1 > Scalar[dtype](0) and contact_count < MAX_CONTACTS:
                var contact_off = contact_count

                contacts[env, contact_off, CONTACT_BODY_A_3D] = Scalar[dtype](body)
                contacts[env, contact_off, CONTACT_BODY_B_3D] = Scalar[dtype](-1)

                contacts[env, contact_off, CONTACT_POINT_X] = p1_x
                contacts[env, contact_off, CONTACT_POINT_Y] = p1_y
                contacts[env, contact_off, CONTACT_POINT_Z] = ground_height

                contacts[env, contact_off, CONTACT_NORMAL_X] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_NORMAL_Y] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_NORMAL_Z] = Scalar[dtype](1)

                contacts[env, contact_off, CONTACT_DEPTH_3D] = depth1

                contacts[env, contact_off, CONTACT_IMPULSE_N] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_IMPULSE_T1] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_IMPULSE_T2] = Scalar[dtype](0)

                contacts[env, contact_off, CONTACT_TANGENT1_X] = Scalar[dtype](1)
                contacts[env, contact_off, CONTACT_TANGENT1_Y] = Scalar[dtype](0)
                contacts[env, contact_off, CONTACT_TANGENT1_Z] = Scalar[dtype](0)

                contact_count += 1

        contact_counts[env] = Scalar[dtype](contact_count)

    # =========================================================================
    # GPU Kernel Entry Point
    # =========================================================================

    @always_inline
    @staticmethod
    fn detect_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shape_types: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        shape_radii: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        shape_half_heights: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        shape_axes: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ],
        ground_height: Scalar[dtype],
    ):
        """GPU kernel for capsule-plane collision detection."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        CapsulePlaneCollisionGPU.detect_single_env[
            BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
        ](env, state, shape_types, shape_radii, shape_half_heights, shape_axes,
          contacts, contact_counts, ground_height)

    # =========================================================================
    # Public GPU API
    # =========================================================================

    @staticmethod
    fn detect_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        state_buf: DeviceBuffer[dtype],
        shape_types_buf: DeviceBuffer[dtype],
        shape_radii_buf: DeviceBuffer[dtype],
        shape_half_heights_buf: DeviceBuffer[dtype],
        shape_axes_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        ground_height: Scalar[dtype],
    ) raises:
        """Launch capsule-plane collision detection kernel on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            shape_types_buf: Shape types [NUM_BODIES].
            shape_radii_buf: Shape radii [NUM_BODIES].
            shape_half_heights_buf: Half-heights [NUM_BODIES].
            shape_axes_buf: Local axes [NUM_BODIES].
            contacts_buf: Contact buffer.
            contact_counts_buf: Contact counts [BATCH].
            ground_height: Z-coordinate of ground plane.
        """
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ](state_buf.unsafe_ptr())
        var shape_types = LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ](shape_types_buf.unsafe_ptr())
        var shape_radii = LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ](shape_radii_buf.unsafe_ptr())
        var shape_half_heights = LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ](shape_half_heights_buf.unsafe_ptr())
        var shape_axes = LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ](shape_axes_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
            shape_types: LayoutTensor[dtype, Layout.row_major(NUM_BODIES), MutAnyOrigin],
            shape_radii: LayoutTensor[dtype, Layout.row_major(NUM_BODIES), MutAnyOrigin],
            shape_half_heights: LayoutTensor[dtype, Layout.row_major(NUM_BODIES), MutAnyOrigin],
            shape_axes: LayoutTensor[dtype, Layout.row_major(NUM_BODIES), MutAnyOrigin],
            contacts: LayoutTensor[dtype, Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D), MutAnyOrigin],
            contact_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            ground_height: Scalar[dtype],
        ):
            CapsulePlaneCollisionGPU.detect_kernel[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](state, shape_types, shape_radii, shape_half_heights, shape_axes,
              contacts, contact_counts, ground_height)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state, shape_types, shape_radii, shape_half_heights, shape_axes,
            contacts, contact_counts, ground_height,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
