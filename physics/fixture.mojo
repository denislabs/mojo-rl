"""
Fixture: Shape-Body Attachment for Physics Engine

Fixtures attach shapes to bodies and define material properties
and collision filtering.
"""

from physics.vec2 import Vec2, vec2, min_vec, max_vec
from physics.shape import (
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    SHAPE_EDGE,
    PolygonShape,
    CircleShape,
    EdgeShape,
)
from physics.body import Body, Transform


# ===== Collision Filtering =====

# LunarLander collision categories
comptime CATEGORY_GROUND: UInt16 = 0x0001
comptime CATEGORY_LANDER: UInt16 = 0x0010
comptime CATEGORY_LEG: UInt16 = 0x0020
comptime CATEGORY_PARTICLE: UInt16 = 0x0100


@register_passable("trivial")
struct Filter(Copyable, Movable):
    """Collision filter using category/mask bits.

    Two fixtures collide if:
    - (filterA.category & filterB.mask) != 0 AND
    - (filterB.category & filterA.mask) != 0
    """

    var category_bits: UInt16
    var mask_bits: UInt16

    fn __init__(
        out self, category: UInt16 = 0x0001, mask: UInt16 = 0xFFFF
    ):
        """Create filter with category and mask."""
        self.category_bits = category
        self.mask_bits = mask

    @always_inline
    fn should_collide(self, other: Self) -> Bool:
        """Check if two filters allow collision."""
        return (
            (self.category_bits & other.mask_bits) != 0
            and (other.category_bits & self.mask_bits) != 0
        )

    @staticmethod
    fn ground() -> Self:
        """Create filter for ground (collides with everything)."""
        return Self(CATEGORY_GROUND, 0xFFFF)

    @staticmethod
    fn lander() -> Self:
        """Create filter for lander (collides with ground only)."""
        return Self(CATEGORY_LANDER, CATEGORY_GROUND)

    @staticmethod
    fn leg() -> Self:
        """Create filter for legs (collides with ground only)."""
        return Self(CATEGORY_LEG, CATEGORY_GROUND)

    @staticmethod
    fn particle() -> Self:
        """Create filter for particles (collides with ground only)."""
        return Self(CATEGORY_PARTICLE, CATEGORY_GROUND)


# ===== AABB for Broad Phase =====


@register_passable("trivial")
struct AABB(Copyable, Movable):
    """Axis-Aligned Bounding Box for broad phase collision detection."""

    var lower: Vec2
    var upper: Vec2

    fn __init__(out self):
        """Create empty AABB."""
        self.lower = Vec2.zero()
        self.upper = Vec2.zero()

    fn __init__(out self, lower: Vec2, upper: Vec2):
        """Create AABB from corners."""
        self.lower = lower
        self.upper = upper

    @always_inline
    fn overlaps(self, other: Self) -> Bool:
        """Check if two AABBs overlap."""
        return (
            self.lower.x <= other.upper.x
            and self.upper.x >= other.lower.x
            and self.lower.y <= other.upper.y
            and self.upper.y >= other.lower.y
        )

    @always_inline
    fn contains(self, point: Vec2) -> Bool:
        """Check if point is inside AABB."""
        return (
            point.x >= self.lower.x
            and point.x <= self.upper.x
            and point.y >= self.lower.y
            and point.y <= self.upper.y
        )

    fn combine(self, other: Self) -> Self:
        """Create AABB that contains both AABBs."""
        return Self(min_vec(self.lower, other.lower), max_vec(self.upper, other.upper))

    fn expand(self, amount: Float64) -> Self:
        """Expand AABB by amount in all directions."""
        var delta = Vec2(amount, amount)
        return Self(self.lower - delta, self.upper + delta)


# ===== Fixture =====


struct Fixture(Copyable, Movable):
    """Attaches a shape to a body with material properties."""

    var body_idx: Int  # Index in World.bodies
    var shape_type: Int

    # Shape data (only one is valid based on shape_type)
    var polygon: PolygonShape
    var circle: CircleShape
    var edge: EdgeShape

    # Material properties
    var density: Float64
    var friction: Float64
    var restitution: Float64

    # Collision filtering
    var filter: Filter

    # AABB (computed and cached)
    var aabb: AABB

    fn __init__(out self):
        """Create empty fixture."""
        self.body_idx = -1
        self.shape_type = SHAPE_POLYGON
        self.polygon = PolygonShape()
        self.circle = CircleShape(0.0)
        self.edge = EdgeShape()
        self.density = 1.0
        self.friction = 0.2
        self.restitution = 0.0
        self.filter = Filter()
        self.aabb = AABB()

    @staticmethod
    fn from_polygon(
        body_idx: Int,
        var polygon: PolygonShape,
        density: Float64 = 1.0,
        friction: Float64 = 0.2,
        restitution: Float64 = 0.0,
        filter: Filter = Filter(),
    ) -> Self:
        """Create fixture with polygon shape."""
        var f = Self()
        f.body_idx = body_idx
        f.shape_type = SHAPE_POLYGON
        f.polygon = polygon^
        f.density = density
        f.friction = friction
        f.restitution = restitution
        f.filter = filter
        return f^

    @staticmethod
    fn from_circle(
        body_idx: Int,
        var circle: CircleShape,
        density: Float64 = 1.0,
        friction: Float64 = 0.2,
        restitution: Float64 = 0.0,
        filter: Filter = Filter(),
    ) -> Self:
        """Create fixture with circle shape."""
        var f = Self()
        f.body_idx = body_idx
        f.shape_type = SHAPE_CIRCLE
        f.circle = circle^
        f.density = density
        f.friction = friction
        f.restitution = restitution
        f.filter = filter
        return f^

    @staticmethod
    fn from_edge(
        body_idx: Int,
        var edge: EdgeShape,
        density: Float64 = 0.0,
        friction: Float64 = 0.1,
        restitution: Float64 = 0.0,
        filter: Filter = Filter(),
    ) -> Self:
        """Create fixture with edge shape."""
        var f = Self()
        f.body_idx = body_idx
        f.shape_type = SHAPE_EDGE
        f.edge = edge^
        f.density = density
        f.friction = friction
        f.restitution = restitution
        f.filter = filter
        return f^

    fn compute_aabb(mut self, transform: Transform):
        """Compute AABB for this fixture given body transform."""
        if self.shape_type == SHAPE_POLYGON:
            self._compute_polygon_aabb(transform)
        elif self.shape_type == SHAPE_CIRCLE:
            self._compute_circle_aabb(transform)
        else:  # SHAPE_EDGE
            self._compute_edge_aabb(transform)

    fn _compute_polygon_aabb(mut self, transform: Transform):
        """Compute AABB for polygon shape."""
        if self.polygon.count == 0:
            self.aabb = AABB()
            return

        var p = transform.apply(self.polygon.vertices[0])
        var lower = p
        var upper = p

        for i in range(1, self.polygon.count):
            p = transform.apply(self.polygon.vertices[i])
            lower = min_vec(lower, p)
            upper = max_vec(upper, p)

        self.aabb = AABB(lower, upper)

    fn _compute_circle_aabb(mut self, transform: Transform):
        """Compute AABB for circle shape."""
        var center = transform.apply(self.circle.center)
        var r = Vec2(self.circle.radius, self.circle.radius)
        self.aabb = AABB(center - r, center + r)

    fn _compute_edge_aabb(mut self, transform: Transform):
        """Compute AABB for edge shape."""
        var v1 = transform.apply(self.edge.v1)
        var v2 = transform.apply(self.edge.v2)
        self.aabb = AABB(min_vec(v1, v2), max_vec(v1, v2))

    fn compute_mass(self) -> Tuple[Float64, Float64, Vec2]:
        """Compute mass properties based on shape and density.

        Returns: (mass, inertia, center)
        """
        if self.shape_type == SHAPE_POLYGON:
            return self.polygon.compute_mass(self.density)
        elif self.shape_type == SHAPE_CIRCLE:
            return self.circle.compute_mass(self.density)
        else:  # SHAPE_EDGE - edges have no mass
            return (0.0, 0.0, Vec2.zero())
