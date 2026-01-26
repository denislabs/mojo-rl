"""
Shape: Geometric Shapes for Physics Engine

Defines three shape types used by LunarLander:
- PolygonShape: Convex polygon (lander body, legs)
- CircleShape: Circle (particles)
- EdgeShape: Line segment (terrain)
"""

from math import sqrt, pi
from physics.vec2 import Vec2, vec2, cross


# Shape type constants
comptime SHAPE_POLYGON: Int = 0
comptime SHAPE_CIRCLE: Int = 1
comptime SHAPE_EDGE: Int = 2

# Maximum vertices for polygon (lander uses 6, we allow up to 8)
comptime MAX_POLYGON_VERTICES: Int = 8


struct PolygonShape[dtype: DType](Copyable, Movable):
    """Convex polygon shape with up to MAX_POLYGON_VERTICES vertices.

    Vertices are stored in counter-clockwise order.
    Normals point outward from edges.
    """

    var vertices: InlineArray[Vec2[Self.dtype], MAX_POLYGON_VERTICES]
    var normals: InlineArray[Vec2[Self.dtype], MAX_POLYGON_VERTICES]
    var count: Int
    var centroid: Vec2[Self.dtype]

    fn __init__(out self):
        """Create empty polygon."""
        self.vertices = InlineArray[Vec2[Self.dtype], MAX_POLYGON_VERTICES](
            Vec2[Self.dtype].zero()
        )
        self.normals = InlineArray[Vec2[Self.dtype], MAX_POLYGON_VERTICES](
            Vec2[Self.dtype].zero()
        )
        self.count = 0
        self.centroid = Vec2[Self.dtype].zero()

    fn __init__(out self, vertices: List[Vec2[Self.dtype]]):
        """Create polygon from list of vertices (counter-clockwise order)."""
        self.vertices = InlineArray[Vec2[Self.dtype], MAX_POLYGON_VERTICES](
            Vec2[Self.dtype].zero()
        )
        self.normals = InlineArray[Vec2[Self.dtype], MAX_POLYGON_VERTICES](
            Vec2[Self.dtype].zero()
        )
        self.count = min(len(vertices), MAX_POLYGON_VERTICES)
        self.centroid = Vec2[Self.dtype].zero()

        # Copy vertices
        for i in range(self.count):
            self.vertices[i] = vertices[i]

        # Compute normals and centroid
        self._compute_normals()
        self._compute_centroid()

    @staticmethod
    fn from_box(
        half_width: Scalar[Self.dtype], half_height: Scalar[Self.dtype]
    ) -> Self:
        """Create box polygon centered at origin."""
        var result = Self()
        result.count = 4
        result.vertices[0] = Vec2(-half_width, -half_height)
        result.vertices[1] = Vec2(half_width, -half_height)
        result.vertices[2] = Vec2(half_width, half_height)
        result.vertices[3] = Vec2(-half_width, half_height)
        result._compute_normals()
        result._compute_centroid()
        return result^

    @staticmethod
    fn from_box_offset(
        half_width: Scalar[Self.dtype],
        half_height: Scalar[Self.dtype],
        center: Vec2[Self.dtype],
        angle: Scalar[Self.dtype],
    ) -> Self:
        """Create box polygon with center offset and rotation."""
        from math import cos, sin

        var result = Self()
        result.count = 4

        # Box vertices relative to center
        var verts = InlineArray[Vec2[Self.dtype], 4](Vec2[Self.dtype].zero())
        verts[0] = Vec2(-half_width, -half_height)
        verts[1] = Vec2(half_width, -half_height)
        verts[2] = Vec2(half_width, half_height)
        verts[3] = Vec2(-half_width, half_height)

        # Rotate and translate
        var c = cos(angle)
        var s = sin(angle)
        for i in range(4):
            var v = verts[i]
            result.vertices[i] = Vec2(
                center.x + (v.x * c - v.y * s), center.y + (v.x * s + v.y * c)
            )

        result._compute_normals()
        result._compute_centroid()
        return result^

    fn _compute_normals(mut self):
        """Compute outward-facing normals for each edge."""
        for i in range(self.count):
            var i2 = (i + 1) % self.count
            var edge = self.vertices[i2] - self.vertices[i]
            # Normal is perpendicular to edge, pointing outward (CCW winding)
            self.normals[i] = Vec2(edge.y, -edge.x).normalize()

    fn _compute_centroid(mut self):
        """Compute centroid using signed area formula."""
        if self.count < 3:
            if self.count > 0:
                self.centroid = self.vertices[0]
            return

        var area: Scalar[Self.dtype] = 0.0
        var cx: Scalar[Self.dtype] = 0.0
        var cy: Scalar[Self.dtype] = 0.0

        for i in range(self.count):
            var p1 = self.vertices[i]
            var p2 = self.vertices[(i + 1) % self.count]
            var cross_val = p1.x * p2.y - p2.x * p1.y
            area += cross_val
            cx += (p1.x + p2.x) * cross_val
            cy += (p1.y + p2.y) * cross_val

        area *= 0.5
        if abs(area) > 1e-10:
            var inv_6area = 1.0 / (6.0 * area)
            self.centroid = Vec2(cx * inv_6area, cy * inv_6area)
        else:
            # Degenerate polygon, use average
            cx = 0.0
            cy = 0.0
            for i in range(self.count):
                cx += self.vertices[i].x
                cy += self.vertices[i].y
            self.centroid = Vec2(
                cx / Scalar[Self.dtype](self.count),
                cy / Scalar[Self.dtype](self.count),
            )

    fn compute_mass(
        self, density: Scalar[Self.dtype]
    ) -> Tuple[Scalar[Self.dtype], Scalar[Self.dtype], Vec2[Self.dtype]]:
        """Compute mass, inertia, and center of mass.

        Returns: (mass, inertia, center)
        """
        if self.count < 3:
            return (0.0, 0.0, Vec2[Self.dtype].zero())

        var area: Scalar[Self.dtype] = 0.0
        var inertia: Scalar[Self.dtype] = 0.0
        var cx: Scalar[Self.dtype] = 0.0
        var cy: Scalar[Self.dtype] = 0.0

        # Use triangulation from centroid
        for i in range(self.count):
            var p1 = self.vertices[i]
            var p2 = self.vertices[(i + 1) % self.count]

            # Triangle area
            var cross_val = p1.x * p2.y - p2.x * p1.y
            var tri_area = 0.5 * cross_val
            area += tri_area

            # Centroid contribution
            cx += tri_area * (p1.x + p2.x) / 3.0
            cy += tri_area * (p1.y + p2.y) / 3.0

            # Inertia contribution (about origin)
            var ex1 = p1.x
            var ey1 = p1.y
            var ex2 = p2.x
            var ey2 = p2.y
            inertia += (
                0.25
                * cross_val
                * (
                    ex1 * ex1
                    + ex2 * ex1
                    + ex2 * ex2
                    + ey1 * ey1
                    + ey2 * ey1
                    + ey2 * ey2
                )
                / 3.0
            )

        area = abs(area)
        if area < 1e-10:
            return (0.0, 0.0, Vec2[Self.dtype].zero())

        var center = Vec2(cx / area, cy / area)
        var mass = density * area
        inertia = density * inertia

        # Shift inertia to center of mass using parallel axis theorem
        inertia -= mass * center.length_squared()
        inertia = abs(inertia)

        return (mass, inertia, center)

    fn get_support(self, direction: Vec2[Self.dtype]) -> Vec2[Self.dtype]:
        """Get the vertex farthest along the given direction (for SAT)."""
        var best_index = 0
        var best_value = self.vertices[0].dot(direction)

        for i in range(1, self.count):
            var value = self.vertices[i].dot(direction)
            if value > best_value:
                best_value = value
                best_index = i

        return self.vertices[best_index]

    fn get_support_index(self, direction: Vec2[Self.dtype]) -> Int:
        """Get index of vertex farthest along the given direction."""
        var best_index = 0
        var best_value = self.vertices[0].dot(direction)

        for i in range(1, self.count):
            var value = self.vertices[i].dot(direction)
            if value > best_value:
                best_value = value
                best_index = i

        return best_index


struct CircleShape[dtype: DType](Copyable, Movable):
    """Circle shape with radius and local center offset."""

    var radius: Scalar[Self.dtype]
    var center: Vec2[Self.dtype]  # Local offset from body center

    fn __init__(
        out self,
        radius: Scalar[Self.dtype],
        center: Vec2[Self.dtype] = Vec2[Self.dtype].zero(),
    ):
        """Create circle with given radius and optional center offset."""
        self.radius = radius
        self.center = center

    fn compute_mass(
        self, density: Scalar[Self.dtype]
    ) -> Tuple[Scalar[Self.dtype], Scalar[Self.dtype], Vec2[Self.dtype]]:
        """Compute mass, inertia, and center of mass.

        Returns: (mass, inertia, center)
        """
        var mass = density * pi * self.radius * self.radius
        # Inertia of disk about center: I = 0.5 * m * r^2
        var inertia = 0.5 * mass * self.radius * self.radius
        # Add parallel axis theorem for offset center
        inertia += mass * self.center.length_squared()
        return (mass, inertia, self.center)


struct EdgeShape[dtype: DType](Copyable, Movable):
    """Edge (line segment) shape for static terrain.

    Edges are one-sided - collision only happens on the normal side.
    """

    var v1: Vec2[Self.dtype]  # Start vertex
    var v2: Vec2[Self.dtype]  # End vertex
    var normal: Vec2[Self.dtype]  # Outward normal (computed from vertices)

    fn __init__(out self, v1: Vec2[Self.dtype], v2: Vec2[Self.dtype]):
        """Create edge from two vertices. Normal points to the left of v1->v2.
        """
        self.v1 = v1
        self.v2 = v2
        # Compute normal (perpendicular to edge, pointing left/outward)
        var edge = v2 - v1
        self.normal = Vec2(-edge.y, edge.x).normalize()

    fn __init__(out self):
        """Create empty edge."""
        self.v1 = Vec2[Self.dtype].zero()
        self.v2 = Vec2[Self.dtype].zero()
        self.normal = Vec2[Self.dtype].unit_y()

    fn length(self) -> Scalar[Self.dtype]:
        """Get edge length."""
        return (self.v2 - self.v1).length()

    fn direction(self) -> Vec2[Self.dtype]:
        """Get normalized edge direction from v1 to v2."""
        return (self.v2 - self.v1).normalize()

    fn closest_point(self, point: Vec2[Self.dtype]) -> Vec2[Self.dtype]:
        """Get closest point on edge to given point."""
        var edge = self.v2 - self.v1
        var len_sq = edge.length_squared()

        if len_sq < 1e-10:
            return self.v1

        # Project point onto edge line
        var t = (point - self.v1).dot(edge) / len_sq
        # Clamp to edge segment
        if t < 0.0:
            return self.v1
        elif t > 1.0:
            return self.v2
        else:
            return self.v1 + edge * t
