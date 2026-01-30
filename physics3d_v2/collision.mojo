"""Physics3D v2 collision detection - Sphere-plane collision.

Phase 2: Simple sphere-ground plane collision detection.
"""

from .constants import GEOM_SPHERE
from .types import Model, Data


fn detect_sphere_plane[
    DTYPE: DType
](model: Model[DTYPE], mut data: Data[DTYPE]):
    """Detect sphere-ground collision.

    The ground plane is at z = model.ground_z, normal pointing up.
    A sphere penetrates when its center z - radius < ground_z.

    Sets data.contact fields:
    - active: True if penetrating
    - depth: penetration depth (positive = penetrating)
    - normal: [0, 0, 1, 0] (up)
    - pos: contact point on ground
    """
    # Only handle sphere geometry
    if model.geom.type != GEOM_SPHERE:
        data.contact.active = False
        return

    var sphere_z = data.xpos_z
    var radius = model.geom.size
    var ground_z = model.ground_z

    # Penetration depth: how far the sphere bottom is below ground
    # depth = radius - (sphere_center_z - ground_z)
    # depth > 0 means penetrating
    var depth = radius - (sphere_z - ground_z)

    if depth > Scalar[DTYPE](0):
        data.contact.active = True
        data.contact.depth = depth
        # Normal points up (from ground toward sphere)
        data.contact.normal_x = Scalar[DTYPE](0)
        data.contact.normal_y = Scalar[DTYPE](0)
        data.contact.normal_z = Scalar[DTYPE](1)
        # Contact point is on the ground directly below sphere center
        data.contact.pos_x = data.xpos_x
        data.contact.pos_y = data.xpos_y
        data.contact.pos_z = ground_z
    else:
        data.contact.active = False
        data.contact.depth = Scalar[DTYPE](0)
