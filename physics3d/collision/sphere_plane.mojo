"""Sphere vs Plane collision detection.

Detects collision between a sphere and an infinite plane (ground).

GPU support follows the three-method hierarchy pattern for writing
contacts directly to flat buffer arrays.
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from math3d import Vec3
from .contact3d import Contact3D

from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE_3D,
    CONTACT_DATA_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    SHAPE_SPHERE,
    CONTACT_BODY_A_3D,
    CONTACT_BODY_B_3D,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_POINT_Z,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_NORMAL_Z,
    CONTACT_DEPTH_3D,
    CONTACT_IMPULSE_N,
    CONTACT_IMPULSE_T1,
    CONTACT_IMPULSE_T2,
    CONTACT_TANGENT1_X,
    CONTACT_TANGENT1_Y,
    CONTACT_TANGENT1_Z,
)


struct SpherePlaneCollision:
    """Sphere vs Plane collision detection and contact generation."""

    @staticmethod
    fn detect(
        sphere_center: Vec3,
        sphere_radius: Float64,
        plane_normal: Vec3,
        plane_offset: Float64,  # Distance from origin along normal
        body_idx: Int,
    ) -> Contact3D:
        """Detect collision between sphere and plane.

        The plane equation is: normal · p + offset = 0
        For ground plane at y=0 with normal (0,0,1): normal=(0,0,1), offset=0

        Args:
            sphere_center: Center of the sphere in world coordinates.
            sphere_radius: Radius of the sphere.
            plane_normal: Normal of the plane (pointing outward).
            plane_offset: Signed distance from origin to plane.
            body_idx: Index of the sphere's body.

        Returns:
            Contact3D with collision info. body_b = -1 for ground.
            Check contact.is_valid() or contact.depth > 0 for collision.
        """
        # Signed distance from sphere center to plane
        # d = normal · center + offset
        var n = plane_normal.normalized()
        var dist = n.dot(sphere_center) + plane_offset

        # Penetration depth
        var depth = sphere_radius - dist

        if depth <= 0.0:
            # No collision
            return Contact3D()

        # Contact point is on sphere surface toward plane
        var contact_point = sphere_center - n * dist

        return Contact3D(
            body_a=body_idx,
            body_b=-1,  # Ground/static
            point=contact_point,
            normal=n,  # Normal points from ground toward sphere
            depth=depth,
        )

    @staticmethod
    fn detect_ground(
        sphere_center: Vec3,
        sphere_radius: Float64,
        ground_height: Float64,
        body_idx: Int,
    ) -> Contact3D:
        """Simplified detection for horizontal ground plane.

        Assumes ground plane at z = ground_height with normal (0, 0, 1).

        Args:
            sphere_center: Center of the sphere.
            sphere_radius: Radius of the sphere.
            ground_height: Z-coordinate of ground plane.
            body_idx: Index of the sphere's body.

        Returns:
            Contact3D with collision info.
        """
        # Distance from sphere bottom to ground
        var sphere_bottom = sphere_center.z - sphere_radius
        var depth = ground_height - sphere_bottom

        if depth <= 0.0:
            return Contact3D()

        # Contact point on ground directly below sphere center
        var contact_point = Vec3(
            sphere_center.x, sphere_center.y, ground_height
        )

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


fn detect_spheres_ground(
    centers: List[Vec3],
    radii: List[Float64],
    body_indices: List[Int],
    ground_height: Float64,
) -> List[Contact3D]:
    """Detect ground collisions for multiple spheres.

    Args:
        centers: List of sphere center positions.
        radii: List of sphere radii.
        body_indices: List of body indices corresponding to each sphere.
        ground_height: Z-coordinate of ground plane.

    Returns:
        List of valid contacts (only spheres that are colliding).
    """
    var contacts = List[Contact3D]()

    for i in range(len(centers)):
        var contact = SpherePlaneCollision.detect_ground(
            centers[i], radii[i], ground_height, body_indices[i]
        )
        if contact.is_valid():
            contacts.append(contact.copy())

    return contacts^


# =============================================================================
# GPU Implementation
# =============================================================================


struct SpherePlaneCollisionGPU:
    """GPU-compatible Sphere vs Plane collision detection.

    Writes contact data directly to flat state buffer arrays instead of
    returning Contact3D objects. Designed for use in fused kernels.
    """

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
        CONTACTS_OFFSET: Int,
        CONTACT_COUNT_OFFSET: Int,
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
        ground_height: Scalar[dtype],
    ):
        """Detect sphere-ground collisions for a single environment.

        Writes contacts to the state buffer at CONTACTS_OFFSET.
        Updates contact count at CONTACT_COUNT_OFFSET.

        Args:
            env: Environment index.
            state: State buffer [BATCH, STATE_SIZE].
            shape_types: Shape type for each body (SHAPE_SPHERE, etc.).
            shape_radii: Radius for each body.
            ground_height: Z-coordinate of ground plane.
        """
        # Get current contact count
        var contact_count = Int(state[env, CONTACT_COUNT_OFFSET])

        @parameter
        for body in range(NUM_BODIES):
            # Skip if not a sphere
            if Int(shape_types[body]) != SHAPE_SPHERE:
                continue

            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE_3D

            # Skip if not dynamic
            var body_type = Int(state[env, body_off + IDX_BODY_TYPE])
            if body_type != BODY_DYNAMIC:
                continue

            # Get sphere center
            var px = state[env, body_off + IDX_PX]
            var py = state[env, body_off + IDX_PY]
            var pz = state[env, body_off + IDX_PZ]

            # Get radius
            var radius = shape_radii[body]

            # Distance from sphere bottom to ground
            var sphere_bottom = pz - radius
            var depth = ground_height - sphere_bottom

            # Skip if no collision
            if depth <= Scalar[dtype](0):
                continue

            # Skip if contact buffer is full
            if contact_count >= MAX_CONTACTS:
                continue

            # Write contact to state buffer
            var contact_off = (
                CONTACTS_OFFSET + contact_count * CONTACT_DATA_SIZE_3D
            )

            # Body indices
            state[env, contact_off + CONTACT_BODY_A_3D] = Scalar[dtype](body)
            state[env, contact_off + CONTACT_BODY_B_3D] = Scalar[dtype](
                -1
            )  # Ground

            # Contact point (on ground directly below sphere center)
            state[env, contact_off + CONTACT_POINT_X] = px
            state[env, contact_off + CONTACT_POINT_Y] = py
            state[env, contact_off + CONTACT_POINT_Z] = ground_height

            # Contact normal (pointing up from ground toward sphere)
            state[env, contact_off + CONTACT_NORMAL_X] = Scalar[dtype](0)
            state[env, contact_off + CONTACT_NORMAL_Y] = Scalar[dtype](0)
            state[env, contact_off + CONTACT_NORMAL_Z] = Scalar[dtype](1)

            # Penetration depth
            state[env, contact_off + CONTACT_DEPTH_3D] = depth

            # Initialize impulses to zero (or warm start if available)
            state[env, contact_off + CONTACT_IMPULSE_N] = Scalar[dtype](0)
            state[env, contact_off + CONTACT_IMPULSE_T1] = Scalar[dtype](0)
            state[env, contact_off + CONTACT_IMPULSE_T2] = Scalar[dtype](0)

            # Compute tangent basis (arbitrary orthonormal basis to normal)
            # For vertical normal (0,0,1), tangents are (1,0,0) and (0,1,0)
            state[env, contact_off + CONTACT_TANGENT1_X] = Scalar[dtype](1)
            state[env, contact_off + CONTACT_TANGENT1_Y] = Scalar[dtype](0)
            state[env, contact_off + CONTACT_TANGENT1_Z] = Scalar[dtype](0)

            contact_count += 1

        # Write back contact count
        state[env, CONTACT_COUNT_OFFSET] = Scalar[dtype](contact_count)

    @always_inline
    @staticmethod
    fn detect_single_env_with_separate_contacts[
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
        """Detect sphere-ground collisions with separate contact buffer.

        Alternative version that writes to separate contacts tensor.

        Args:
            env: Environment index.
            state: State buffer [BATCH, STATE_SIZE] (bodies only).
            shape_types: Shape type for each body.
            shape_radii: Radius for each body.
            contacts: Contact buffer [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D].
            contact_counts: Contact count per environment [BATCH].
            ground_height: Z-coordinate of ground plane.
        """
        var contact_count = Int(contact_counts[env])

        @parameter
        for body in range(NUM_BODIES):
            # Skip if not a sphere
            if Int(shape_types[body]) != SHAPE_SPHERE:
                continue

            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE_3D

            # Skip if not dynamic
            var body_type = Int(state[env, body_off + IDX_BODY_TYPE])
            if body_type != BODY_DYNAMIC:
                continue

            # Get sphere center
            var px = state[env, body_off + IDX_PX]
            var py = state[env, body_off + IDX_PY]
            var pz = state[env, body_off + IDX_PZ]

            var radius = shape_radii[body]

            var sphere_bottom = pz - radius
            var depth = ground_height - sphere_bottom

            if depth <= Scalar[dtype](0):
                continue

            if contact_count >= MAX_CONTACTS:
                continue

            # Write to contacts tensor
            contacts[env, contact_count, CONTACT_BODY_A_3D] = Scalar[dtype](
                body
            )
            contacts[env, contact_count, CONTACT_BODY_B_3D] = Scalar[dtype](-1)

            contacts[env, contact_count, CONTACT_POINT_X] = px
            contacts[env, contact_count, CONTACT_POINT_Y] = py
            contacts[env, contact_count, CONTACT_POINT_Z] = ground_height

            contacts[env, contact_count, CONTACT_NORMAL_X] = Scalar[dtype](0)
            contacts[env, contact_count, CONTACT_NORMAL_Y] = Scalar[dtype](0)
            contacts[env, contact_count, CONTACT_NORMAL_Z] = Scalar[dtype](1)

            contacts[env, contact_count, CONTACT_DEPTH_3D] = depth

            contacts[env, contact_count, CONTACT_IMPULSE_N] = Scalar[dtype](0)
            contacts[env, contact_count, CONTACT_IMPULSE_T1] = Scalar[dtype](0)
            contacts[env, contact_count, CONTACT_IMPULSE_T2] = Scalar[dtype](0)

            contacts[env, contact_count, CONTACT_TANGENT1_X] = Scalar[dtype](1)
            contacts[env, contact_count, CONTACT_TANGENT1_Y] = Scalar[dtype](0)
            contacts[env, contact_count, CONTACT_TANGENT1_Z] = Scalar[dtype](0)

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
        """GPU kernel for sphere-plane collision detection."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        SpherePlaneCollisionGPU.detect_single_env_with_separate_contacts[
            BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
        ](
            env,
            state,
            shape_types,
            shape_radii,
            contacts,
            contact_counts,
            ground_height,
        )

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
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        ground_height: Scalar[dtype],
    ) raises:
        """Launch sphere-plane collision detection kernel on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            shape_types_buf: Shape types [NUM_BODIES].
            shape_radii_buf: Shape radii [NUM_BODIES].
            contacts_buf: Contact buffer [BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE_3D].
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
            SpherePlaneCollisionGPU.detect_kernel[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](
                state,
                shape_types,
                shape_radii,
                contacts,
                contact_counts,
                ground_height,
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            shape_types,
            shape_radii,
            contacts,
            contact_counts,
            ground_height,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
