"""Sphere vs Plane collision detection.

Detects collision between a sphere and an infinite plane (ground).
"""

from math import sqrt
from math3d import Vec3
from .contact3d import Contact3D


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
        var contact_point = Vec3(sphere_center.x, sphere_center.y, ground_height)

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
            contacts.append(contact)

    return contacts
