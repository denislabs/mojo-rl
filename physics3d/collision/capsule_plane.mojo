"""Capsule vs Plane collision detection.

Detects collision between a capsule and an infinite plane (ground).
Capsules are commonly used for limbs in MuJoCo environments.
"""

from math import sqrt
from math3d import Vec3, Quat
from .contact3d import Contact3D, ContactManifold


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
