"""
Physics: Minimal 2D Physics Engine for LunarLander

A lightweight physics engine implementing only the Box2D features
needed for LunarLander simulation.
"""

from physics_legacy.vec2 import (
    Vec2,
    vec2,
    dot,
    cross,
    cross_sv,
    cross_vs,
    length,
    normalize,
    distance,
    min_vec,
    max_vec,
    clamp_vec,
    clamp_length,
)
from physics_legacy.shape import (
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    SHAPE_EDGE,
    MAX_POLYGON_VERTICES,
    PolygonShape,
    CircleShape,
    EdgeShape,
)
from physics_legacy.body import (
    BODY_STATIC,
    BODY_DYNAMIC,
    MassData,
    Transform,
    Body,
)
from physics_legacy.fixture import (
    CATEGORY_GROUND,
    CATEGORY_LANDER,
    CATEGORY_LEG,
    CATEGORY_PARTICLE,
    Filter,
    AABB,
    Fixture,
)
from physics_legacy.collision import (
    ContactPoint,
    ContactManifold,
    Contact,
    collide_edge_polygon,
    collide_edge_circle,
    collide_polygon_polygon,
)
from physics_legacy.joint import RevoluteJoint
from physics_legacy.world import World, ContactListener
from physics_legacy.raycast import RaycastResult, raycast_edge, raycast_polygon
