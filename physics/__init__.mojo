"""
Physics: Minimal 2D Physics Engine for LunarLander

A lightweight physics engine implementing only the Box2D features
needed for LunarLander simulation.
"""

from physics.vec2 import (
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
from physics.shape import (
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    SHAPE_EDGE,
    MAX_POLYGON_VERTICES,
    PolygonShape,
    CircleShape,
    EdgeShape,
)
from physics.body import (
    BODY_STATIC,
    BODY_DYNAMIC,
    MassData,
    Transform,
    Body,
)
from physics.fixture import (
    CATEGORY_GROUND,
    CATEGORY_LANDER,
    CATEGORY_LEG,
    CATEGORY_PARTICLE,
    Filter,
    AABB,
    Fixture,
)
from physics.collision import (
    ContactPoint,
    ContactManifold,
    Contact,
    collide_edge_polygon,
    collide_edge_circle,
)
from physics.joint import RevoluteJoint
from physics.world import World, ContactListener
