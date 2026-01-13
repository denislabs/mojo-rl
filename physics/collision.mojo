"""
Collision: Contact Detection for Physics Engine

Implements collision detection between shapes:
- Edge vs Polygon (terrain vs lander/legs)
- Edge vs Circle (terrain vs particles)

Uses Separating Axis Theorem (SAT) for edge-polygon collisions.
"""

from math import sqrt
from physics.vec2 import Vec2, vec2, dot, cross, min_vec, max_vec
from physics.shape import (
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    SHAPE_EDGE,
    PolygonShape,
    CircleShape,
    EdgeShape,
)
from physics.body import Body, Transform


# Maximum contact points per manifold
comptime MAX_MANIFOLD_POINTS: Int = 2


@register_passable("trivial")
struct ContactPoint(Copyable, Movable):
    """Single contact point."""

    var point: Vec2  # World space contact point
    var normal: Vec2  # Normal from fixture A to fixture B
    var separation: Float64  # Negative = penetration depth

    fn __init__(out self):
        self.point = Vec2.zero()
        self.normal = Vec2.zero()
        self.separation = 0.0

    fn __init__(out self, point: Vec2, normal: Vec2, separation: Float64):
        self.point = point
        self.normal = normal
        self.separation = separation


struct ContactManifold(Copyable, Movable):
    """Collection of contact points between two fixtures."""

    var points: InlineArray[ContactPoint, MAX_MANIFOLD_POINTS]
    var count: Int
    var fixture_a_idx: Int
    var fixture_b_idx: Int

    fn __init__(out self):
        self.points = InlineArray[ContactPoint, MAX_MANIFOLD_POINTS](ContactPoint())
        self.count = 0
        self.fixture_a_idx = -1
        self.fixture_b_idx = -1

    fn add_point(mut self, point: Vec2, normal: Vec2, separation: Float64):
        """Add a contact point if space available."""
        if self.count < MAX_MANIFOLD_POINTS:
            self.points[self.count] = ContactPoint(point, normal, separation)
            self.count += 1


struct Contact(Copyable, Movable):
    """Contact between two fixtures with solver state."""

    var manifold: ContactManifold

    # Bodies involved
    var body_a_idx: Int
    var body_b_idx: Int

    # Contact state
    var touching: Bool
    var is_new: Bool  # For BeginContact callback

    # Solver data (accumulated impulses for warm starting)
    var normal_impulse: Float64
    var tangent_impulse: Float64

    fn __init__(out self):
        self.manifold = ContactManifold()
        self.body_a_idx = -1
        self.body_b_idx = -1
        self.touching = False
        self.is_new = False
        self.normal_impulse = 0.0
        self.tangent_impulse = 0.0


# ===== Collision Detection Functions =====


fn collide_edge_polygon(
    edge: EdgeShape,
    xf_edge: Transform,
    polygon: PolygonShape,
    xf_polygon: Transform,
) -> ContactManifold:
    """Detect collision between edge and polygon.

    Uses simplified SAT: test edge normal and polygon edge normals.
    Returns manifold with contact points if collision found.
    """
    var manifold = ContactManifold()

    # Transform edge to world space
    var v1 = xf_edge.apply(edge.v1)
    var v2 = xf_edge.apply(edge.v2)
    var edge_normal = xf_edge.apply_rotation(edge.normal)

    # Edge direction
    var edge_dir = (v2 - v1).normalize()

    # Find polygon vertices in world space
    var poly_verts = InlineArray[Vec2, 8](Vec2.zero())
    for i in range(polygon.count):
        poly_verts[i] = xf_polygon.apply(polygon.vertices[i])

    # Test 1: Separate along edge normal
    var min_sep = find_min_separation_edge(v1, edge_normal, poly_verts, polygon.count)
    if min_sep > 0.0:
        return manifold^  # Separated

    # Test 2: Separate along each polygon edge normal
    for i in range(polygon.count):
        var poly_normal = xf_polygon.apply_rotation(polygon.normals[i])
        var sep = find_min_separation_polygon(
            poly_verts[i], poly_normal, v1, v2
        )
        if sep > 0.0:
            return manifold^  # Separated

    # No separation found - collision!
    # Generate contact points by clipping polygon against edge region

    # Find polygon vertices inside edge region (between v1 and v2 along edge normal)
    for i in range(polygon.count):
        var vertex = poly_verts[i]

        # Check if vertex is on collision side of edge
        var sep = (vertex - v1).dot(edge_normal)
        if sep < 0.0:  # Penetrating
            # Check if vertex is between edge endpoints
            var t = (vertex - v1).dot(edge_dir)
            var edge_len = (v2 - v1).length()
            if t >= 0.0 and t <= edge_len:
                manifold.add_point(vertex, edge_normal, sep)
                if manifold.count >= MAX_MANIFOLD_POINTS:
                    break

    # If no vertex contacts, check edge endpoints against polygon
    if manifold.count == 0:
        # Check if edge endpoints are inside polygon
        var _ = xf_polygon.apply(polygon.centroid)  # Unused but shows intent

        for edge_pt_idx in range(2):
            var edge_pt = v1 if edge_pt_idx == 0 else v2

            # Simple point-in-polygon test using normals
            var inside = True
            var min_dist: Float64 = 1e10
            var closest_normal = Vec2.zero()

            for i in range(polygon.count):
                var dist = (edge_pt - poly_verts[i]).dot(
                    xf_polygon.apply_rotation(polygon.normals[i])
                )
                if dist > 0.0:
                    inside = False
                    break
                if dist > -min_dist:
                    min_dist = -dist
                    closest_normal = xf_polygon.apply_rotation(polygon.normals[i])

            if inside:
                manifold.add_point(edge_pt, -closest_normal, -min_dist)
                if manifold.count >= MAX_MANIFOLD_POINTS:
                    break

    return manifold^


fn collide_edge_circle(
    edge: EdgeShape,
    xf_edge: Transform,
    circle: CircleShape,
    xf_circle: Transform,
) -> ContactManifold:
    """Detect collision between edge and circle.

    Projects circle center onto edge and checks distance.
    """
    var manifold = ContactManifold()

    # Transform shapes to world space
    var v1 = xf_edge.apply(edge.v1)
    var v2 = xf_edge.apply(edge.v2)
    var center = xf_circle.apply(circle.center)
    var radius = circle.radius

    # Find closest point on edge to circle center
    var edge_vec = v2 - v1
    var edge_len_sq = edge_vec.length_squared()

    var closest: Vec2
    var normal: Vec2

    if edge_len_sq < 1e-10:
        # Degenerate edge
        closest = v1
        normal = (center - v1).normalize_safe(Vec2.unit_y())
    else:
        # Project center onto edge line
        var t = (center - v1).dot(edge_vec) / edge_len_sq

        if t <= 0.0:
            # Closest to v1
            closest = v1
            normal = (center - v1).normalize_safe(Vec2.unit_y())
        elif t >= 1.0:
            # Closest to v2
            closest = v2
            normal = (center - v2).normalize_safe(Vec2.unit_y())
        else:
            # Closest to interior of edge
            closest = v1 + edge_vec * t
            # Use edge normal for interior contacts
            normal = xf_edge.apply_rotation(edge.normal)

    # Check distance
    var dist = (center - closest).length()
    var separation = dist - radius

    if separation <= 0.0:
        # Collision!
        var contact_point = center - normal * radius
        manifold.add_point(contact_point, normal, separation)

    return manifold^


fn collide_polygon_polygon(
    poly_a: PolygonShape,
    xf_a: Transform,
    poly_b: PolygonShape,
    xf_b: Transform,
) -> ContactManifold:
    """Detect collision between two convex polygons.

    Uses Separating Axis Theorem (SAT): test all face normals from both polygons.
    Returns manifold with contact points if collision found.
    """
    var manifold = ContactManifold()

    if poly_a.count < 3 or poly_b.count < 3:
        return manifold^

    # Transform polygon vertices to world space
    var verts_a = InlineArray[Vec2, 8](Vec2.zero())
    var verts_b = InlineArray[Vec2, 8](Vec2.zero())

    for i in range(poly_a.count):
        verts_a[i] = xf_a.apply(poly_a.vertices[i])

    for i in range(poly_b.count):
        verts_b[i] = xf_b.apply(poly_b.vertices[i])

    # Find separating axis on poly_a faces
    var sep_a: Float64 = -1e10
    var best_idx_a = 0

    for i in range(poly_a.count):
        var normal = xf_a.apply_rotation(poly_a.normals[i])
        var min_sep: Float64 = 1e10

        # Find min separation along this normal
        for j in range(poly_b.count):
            var sep = (verts_b[j] - verts_a[i]).dot(normal)
            if sep < min_sep:
                min_sep = sep

        if min_sep > sep_a:
            sep_a = min_sep
            best_idx_a = i

        # Early out: separation found
        if min_sep > 0.0:
            return manifold^

    # Find separating axis on poly_b faces
    var sep_b: Float64 = -1e10
    var best_idx_b = 0

    for i in range(poly_b.count):
        var normal = xf_b.apply_rotation(poly_b.normals[i])
        var min_sep: Float64 = 1e10

        # Find min separation along this normal
        for j in range(poly_a.count):
            var sep = (verts_a[j] - verts_b[i]).dot(normal)
            if sep < min_sep:
                min_sep = sep

        if min_sep > sep_b:
            sep_b = min_sep
            best_idx_b = i

        # Early out: separation found
        if min_sep > 0.0:
            return manifold^

    # No separation found - collision!
    # Use the axis with smallest penetration as contact normal
    var contact_normal: Vec2
    var ref_verts: InlineArray[Vec2, 8]
    var ref_count: Int
    var inc_verts: InlineArray[Vec2, 8]
    var inc_count: Int
    var flip: Bool

    # Determine reference and incident edges
    comptime RELATIVE_TOL: Float64 = 0.98
    comptime ABSOLUTE_TOL: Float64 = 0.001

    if sep_a > sep_b * RELATIVE_TOL + ABSOLUTE_TOL:
        # Use poly_a face as reference
        contact_normal = xf_a.apply_rotation(poly_a.normals[best_idx_a])
        ref_verts = verts_a
        ref_count = poly_a.count
        inc_verts = verts_b
        inc_count = poly_b.count
        flip = False
    else:
        # Use poly_b face as reference
        contact_normal = xf_b.apply_rotation(poly_b.normals[best_idx_b])
        ref_verts = verts_b
        ref_count = poly_b.count
        inc_verts = verts_a
        inc_count = poly_a.count
        flip = True

    # Find incident edge vertices that are on the collision side
    for i in range(inc_count):
        var vertex = inc_verts[i]

        # Find penetration depth
        var min_dist: Float64 = 1e10
        for j in range(ref_count):
            var dist = (vertex - ref_verts[j]).dot(contact_normal)
            if dist < min_dist:
                min_dist = dist

        if min_dist < 0.0:
            # This vertex is penetrating
            var final_normal = contact_normal
            if flip:
                final_normal = Vec2(-contact_normal.x, -contact_normal.y)

            manifold.add_point(vertex, final_normal, min_dist)
            if manifold.count >= MAX_MANIFOLD_POINTS:
                break

    return manifold^


# ===== Helper Functions =====


fn find_min_separation_edge(
    edge_v1: Vec2,
    edge_normal: Vec2,
    poly_verts: InlineArray[Vec2, 8],
    poly_count: Int,
) -> Float64:
    """Find minimum separation between edge and polygon along edge normal."""
    var min_sep: Float64 = 1e10

    for i in range(poly_count):
        var sep = (poly_verts[i] - edge_v1).dot(edge_normal)
        if sep < min_sep:
            min_sep = sep

    return min_sep


fn find_min_separation_polygon(
    poly_vertex: Vec2,
    poly_normal: Vec2,
    edge_v1: Vec2,
    edge_v2: Vec2,
) -> Float64:
    """Find minimum separation between polygon edge and edge shape."""
    var sep1 = (edge_v1 - poly_vertex).dot(poly_normal)
    var sep2 = (edge_v2 - poly_vertex).dot(poly_normal)
    return min_f64(sep1, sep2)


fn min_f64(a: Float64, b: Float64) -> Float64:
    """Minimum of two floats."""
    return a if a < b else b


fn max_f64(a: Float64, b: Float64) -> Float64:
    """Maximum of two floats."""
    return a if a > b else b


fn abs_f64(x: Float64) -> Float64:
    """Absolute value."""
    return x if x >= 0.0 else -x


fn clamp(x: Float64, low: Float64, high: Float64) -> Float64:
    """Clamp value to range."""
    if x < low:
        return low
    if x > high:
        return high
    return x


# ===== Contact Velocity Solver =====


fn compute_contact_velocity_at_point(
    body_a: Body,
    body_b: Body,
    point: Vec2,
) -> Vec2:
    """Compute relative velocity at contact point (B relative to A)."""
    var vel_a = body_a.get_linear_velocity_at_point(point)
    var vel_b = body_b.get_linear_velocity_at_point(point)
    return vel_b - vel_a


fn solve_contact_velocity(
    mut body_a: Body,
    mut body_b: Body,
    contact_point: Vec2,
    normal: Vec2,
    restitution: Float64,
    mut accumulated_impulse: Float64,
) -> Float64:
    """Solve velocity constraint for a single contact point.

    Returns the impulse magnitude applied.
    """
    # Compute relative velocity
    var rel_vel = compute_contact_velocity_at_point(body_a, body_b, contact_point)
    var vel_along_normal = rel_vel.dot(normal)

    # Only resolve if velocities are separating
    if vel_along_normal > 0.0:
        return 0.0

    # Compute impulse scalar
    var r_a = contact_point - body_a.position
    var r_b = contact_point - body_b.position

    var r_a_cross_n = r_a.cross(normal)
    var r_b_cross_n = r_b.cross(normal)

    var inv_mass_sum = (
        body_a.mass_data.inv_mass
        + body_b.mass_data.inv_mass
        + r_a_cross_n * r_a_cross_n * body_a.mass_data.inv_inertia
        + r_b_cross_n * r_b_cross_n * body_b.mass_data.inv_inertia
    )

    if inv_mass_sum == 0.0:
        return 0.0

    var impulse_magnitude = -(1.0 + restitution) * vel_along_normal / inv_mass_sum

    # Clamp accumulated impulse (sequential impulse method)
    var old_impulse = accumulated_impulse
    accumulated_impulse = max_f64(old_impulse + impulse_magnitude, 0.0)
    impulse_magnitude = accumulated_impulse - old_impulse

    # Apply impulse
    var impulse = normal * impulse_magnitude
    body_a.apply_linear_impulse(impulse * -1.0, contact_point)
    body_b.apply_linear_impulse(impulse, contact_point)

    return impulse_magnitude


fn solve_contact_position(
    mut body_a: Body,
    mut body_b: Body,
    contact_point: Vec2,
    normal: Vec2,
    penetration: Float64,
) -> Bool:
    """Solve position constraint for a single contact point.

    Returns True if constraint is satisfied.
    """
    comptime SLOP: Float64 = 0.005  # Allowed penetration
    comptime BAUMGARTE: Float64 = 0.2  # Position correction factor
    comptime MAX_CORRECTION: Float64 = 0.2  # Maximum correction per step

    var separation = penetration + SLOP

    if separation >= 0.0:
        return True  # No correction needed

    var r_a = contact_point - body_a.position
    var r_b = contact_point - body_b.position

    var r_a_cross_n = r_a.cross(normal)
    var r_b_cross_n = r_b.cross(normal)

    var inv_mass_sum = (
        body_a.mass_data.inv_mass
        + body_b.mass_data.inv_mass
        + r_a_cross_n * r_a_cross_n * body_a.mass_data.inv_inertia
        + r_b_cross_n * r_b_cross_n * body_b.mass_data.inv_inertia
    )

    if inv_mass_sum == 0.0:
        return True

    var correction = clamp(-BAUMGARTE * separation / inv_mass_sum, -MAX_CORRECTION, 0.0)

    # Apply position correction
    var correction_vec = normal * correction

    body_a.position -= correction_vec * body_a.mass_data.inv_mass
    body_a.angle -= r_a.cross(correction_vec) * body_a.mass_data.inv_inertia

    body_b.position += correction_vec * body_b.mass_data.inv_mass
    body_b.angle += r_b.cross(correction_vec) * body_b.mass_data.inv_inertia

    # Update transforms
    body_a.transform = Transform(body_a.position, body_a.angle)
    body_b.transform = Transform(body_b.position, body_b.angle)

    return separation >= -SLOP * 3.0
