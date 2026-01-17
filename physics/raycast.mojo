"""
Raycast: Ray Casting for Physics Engine

Implements ray casting against shapes for lidar simulation.
Used by BipedalWalker's 10-beam lidar sensor.
"""

from math import sqrt
from physics.vec2 import Vec2, vec2, dot
from physics.shape import PolygonShape, EdgeShape
from physics.body import Transform


struct RaycastResult[dtype: DType](Copyable, ImplicitlyCopyable, Movable):
    """Result of a raycast query."""

    var hit: Bool  # True if ray hit something
    var point: Vec2[Self.dtype]  # World-space hit point
    var normal: Vec2[Self.dtype]  # Surface normal at hit point
    var fraction: Scalar[Self.dtype]  # Distance fraction along ray (0.0-1.0)
    var fixture_idx: Int  # Index of hit fixture (-1 if no hit)

    fn __init__(out self):
        """Create empty (no hit) result."""
        self.hit = False
        self.point = Vec2[Self.dtype].zero()
        self.normal = Vec2[Self.dtype].zero()
        self.fraction = 1.0
        self.fixture_idx = -1

    fn __init__(
        out self,
        hit: Bool,
        point: Vec2[Self.dtype],
        normal: Vec2[Self.dtype],
        fraction: Scalar[Self.dtype],
        fixture_idx: Int = -1,
    ):
        """Create raycast result with given values."""
        self.hit = hit
        self.point = point
        self.normal = normal
        self.fraction = fraction
        self.fixture_idx = fixture_idx

    @staticmethod
    fn no_hit() -> Self:
        """Create a no-hit result."""
        return Self()

    fn __copyinit__(out self, existing: Self):
        self.hit = existing.hit
        self.point = existing.point
        self.normal = existing.normal
        self.fraction = existing.fraction
        self.fixture_idx = existing.fixture_idx

    fn __moveinit__(out self, deinit existing: Self):
        self.hit = existing.hit
        self.point = existing.point
        self.normal = existing.normal
        self.fraction = existing.fraction
        self.fixture_idx = existing.fixture_idx


fn raycast_edge[
    dtype: DType
](
    ray_start: Vec2[dtype],
    ray_end: Vec2[dtype],
    edge: EdgeShape[dtype],
    xf: Transform[dtype],
) -> RaycastResult[dtype]:
    """Cast ray against an edge shape.

    Uses parametric line intersection:
    Ray: P(t) = ray_start + t * (ray_end - ray_start), t in [0, 1]
    Edge: Q(s) = v1 + s * (v2 - v1), s in [0, 1]

    Args:
        ray_start: Ray origin in world space
        ray_end: Ray end point in world space
        edge: Edge shape in local coordinates
        xf: Transform from local to world space

    Returns:
        RaycastResult with hit info (fraction is distance along ray)
    """
    # Transform edge vertices to world space
    var v1 = xf.apply(edge.v1)
    var v2 = xf.apply(edge.v2)

    # Ray direction
    var d = ray_end - ray_start

    # Edge direction
    var e = v2 - v1

    # Cross product d x e (scalar in 2D)
    var cross_de = d.x * e.y - d.y * e.x

    # Check if parallel (cross product near zero)
    comptime EPSILON: Scalar[dtype] = 1e-10
    if abs(cross_de) < EPSILON:
        return RaycastResult[dtype].no_hit()

    # Vector from ray start to edge start
    var f = v1 - ray_start

    # Parametric solutions
    # t = (f x e) / (d x e)
    # s = (f x d) / (d x e)
    var t = (f.x * e.y - f.y * e.x) / cross_de
    var s = (f.x * d.y - f.y * d.x) / cross_de

    # Check if intersection is within ray segment [0, 1] and edge segment [0, 1]
    if t >= 0.0 and t <= 1.0 and s >= 0.0 and s <= 1.0:
        # Compute hit point
        var hit_point = ray_start + d * t

        # Transform normal to world space (rotation only)
        var normal = xf.apply_rotation(edge.normal)

        # Make sure normal points toward ray origin (for one-sided collision)
        # If ray is hitting from behind the edge, flip normal
        if d.dot(normal) > 0.0:
            normal = Vec2(-normal.x, -normal.y)

        return RaycastResult(
            hit=True,
            point=hit_point,
            normal=normal,
            fraction=t,
        )

    return RaycastResult[dtype].no_hit()


fn raycast_polygon[
    dtype: DType
](
    ray_start: Vec2[dtype],
    ray_end: Vec2[dtype],
    polygon: PolygonShape[dtype],
    xf: Transform[dtype],
) -> RaycastResult[dtype]:
    """Cast ray against a convex polygon shape.

    Tests ray against each polygon edge and returns closest hit.

    Args:
        ray_start: Ray origin in world space
        ray_end: Ray end point in world space
        polygon: Polygon shape in local coordinates
        xf: Transform from local to world space

    Returns:
        RaycastResult with hit info (fraction is distance along ray)
    """
    if polygon.count < 3:
        return RaycastResult[dtype].no_hit()

    # Ray direction
    var d = ray_end - ray_start
    var d_len_sq = d.length_squared()
    if d_len_sq < 1e-20:
        return RaycastResult[dtype].no_hit()

    # Transform polygon vertices to world space
    var world_verts = InlineArray[Vec2[dtype], 8](Vec2[dtype].zero())
    for i in range(polygon.count):
        world_verts[i] = xf.apply(polygon.vertices[i])

    # Test ray against each edge, find closest hit
    var best_fraction: Scalar[dtype] = 1.0 + 1e-6  # Start slightly > 1.0
    var best_normal = Vec2[dtype].zero()
    var hit_found = False

    for i in range(polygon.count):
        var i2 = (i + 1) % polygon.count
        var v1 = world_verts[i]
        var v2 = world_verts[i2]

        # Edge direction
        var e = v2 - v1

        # Cross product d x e
        var cross_de = d.x * e.y - d.y * e.x

        comptime EPSILON: Scalar[dtype] = 1e-10
        if abs(cross_de) < EPSILON:
            continue  # Parallel, skip this edge

        # Vector from ray start to edge start
        var f = v1 - ray_start

        # Parametric solutions
        var t = (f.x * e.y - f.y * e.x) / cross_de
        var s = (f.x * d.y - f.y * d.x) / cross_de

        # Check valid intersection
        if t >= 0.0 and t <= 1.0 and s >= 0.0 and s <= 1.0:
            if t < best_fraction:
                best_fraction = t
                # Compute outward normal for this edge
                var edge_normal = Vec2(-e.y, e.x).normalize()
                # Ensure normal points toward ray (opposite to ray direction)
                if d.dot(edge_normal) > 0.0:
                    edge_normal = Vec2(-edge_normal.x, -edge_normal.y)
                best_normal = edge_normal
                hit_found = True

    if hit_found and best_fraction <= 1.0:
        var hit_point = ray_start + d * best_fraction
        return RaycastResult(
            hit=True,
            point=hit_point,
            normal=best_normal,
            fraction=best_fraction,
        )

    return RaycastResult[dtype].no_hit()
