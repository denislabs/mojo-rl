"""3D Wireframe Shapes.

Generates vertices and edges for wireframe rendering of common 3D shapes.
"""

from math import sin, cos
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


struct WireframeLine(Copyable, Movable):
    """A single line segment for wireframe rendering."""

    var start: Vec3
    var end: Vec3

    fn __init__(out self, start: Vec3, end: Vec3):
        self.start = start
        self.end = end

    fn __copyinit__(out self, other: Self):
        self.start = other.start
        self.end = other.end

    fn __moveinit__(out self, deinit other: Self):
        self.start = other.start
        self.end = other.end


struct WireframeSphere:
    """Generate wireframe lines for a sphere.

    Creates latitude/longitude lines for a sphere approximation.
    """

    var center: Vec3
    var radius: Float64
    var segments: Int
    var rings: Int

    fn __init__(
        out self,
        center: Vec3 = Vec3.zero(),
        radius: Float64 = 1.0,
        segments: Int = 12,  # Longitude divisions
        rings: Int = 8,  # Latitude divisions
    ):
        """Initialize sphere wireframe.

        Args:
            center: Sphere center position.
            radius: Sphere radius.
            segments: Number of longitude segments.
            rings: Number of latitude rings.
        """
        self.center = center
        self.radius = radius
        self.segments = segments
        self.rings = rings

    fn get_lines(self) -> List[WireframeLine]:
        """Generate wireframe lines for the sphere.

        Returns:
            List of line segments.
        """
        var lines = List[WireframeLine]()
        var pi = 3.14159265358979

        # Generate longitude lines (vertical circles)
        for i in range(self.segments):
            var theta = Float64(i) * 2.0 * pi / Float64(self.segments)
            var cos_theta = cos(theta)
            var sin_theta = sin(theta)

            for j in range(self.rings):
                var phi1 = Float64(j) * pi / Float64(self.rings)
                var phi2 = Float64(j + 1) * pi / Float64(self.rings)

                var p1 = self.center + Vec3(
                    self.radius * sin(phi1) * cos_theta,
                    self.radius * sin(phi1) * sin_theta,
                    self.radius * cos(phi1),
                )
                var p2 = self.center + Vec3(
                    self.radius * sin(phi2) * cos_theta,
                    self.radius * sin(phi2) * sin_theta,
                    self.radius * cos(phi2),
                )
                lines.append(WireframeLine(p1, p2))

        # Generate latitude lines (horizontal circles)
        for j in range(1, self.rings):
            var phi = Float64(j) * pi / Float64(self.rings)
            var sin_phi = sin(phi)
            var cos_phi = cos(phi)
            var r = self.radius * sin_phi

            for i in range(self.segments):
                var theta1 = Float64(i) * 2.0 * pi / Float64(self.segments)
                var theta2 = Float64(i + 1) * 2.0 * pi / Float64(self.segments)

                var p1 = self.center + Vec3(
                    r * cos(theta1),
                    r * sin(theta1),
                    self.radius * cos_phi,
                )
                var p2 = self.center + Vec3(
                    r * cos(theta2),
                    r * sin(theta2),
                    self.radius * cos_phi,
                )
                lines.append(WireframeLine(p1, p2))

        return lines^


struct WireframeCapsule:
    """Generate wireframe lines for a capsule.

    A capsule is a cylinder with hemispherical caps.
    """

    var center: Vec3
    var orientation: Quat
    var radius: Float64
    var half_height: Float64
    var axis: Int  # 0=X, 1=Y, 2=Z in local space
    var segments: Int

    fn __init__(
        out self,
        center: Vec3 = Vec3.zero(),
        orientation: Quat = Quat.identity(),
        radius: Float64 = 0.05,
        half_height: Float64 = 0.2,
        axis: Int = 2,  # Z-axis default
        segments: Int = 12,
    ):
        """Initialize capsule wireframe.

        Args:
            center: Capsule center position.
            orientation: Capsule orientation.
            radius: Capsule radius.
            half_height: Half-height of cylindrical part.
            axis: Local axis direction (0=X, 1=Y, 2=Z).
            segments: Number of circular segments.
        """
        self.center = center
        self.orientation = orientation
        self.radius = radius
        self.half_height = half_height
        self.axis = axis
        self.segments = segments

    fn get_lines(self) -> List[WireframeLine]:
        """Generate wireframe lines for the capsule.

        Returns:
            List of line segments.
        """
        var lines = List[WireframeLine]()
        var pi = 3.14159265358979

        # Get local axis direction
        var local_axis: Vec3
        var local_u: Vec3
        var local_v: Vec3

        if self.axis == 0:
            local_axis = Vec3.unit_x()
            local_u = Vec3.unit_y()
            local_v = Vec3.unit_z()
        elif self.axis == 1:
            local_axis = Vec3.unit_y()
            local_u = Vec3.unit_z()
            local_v = Vec3.unit_x()
        else:
            local_axis = Vec3.unit_z()
            local_u = Vec3.unit_x()
            local_v = Vec3.unit_y()

        # Transform to world space
        var world_axis = self.orientation.rotate_vec(local_axis)
        var world_u = self.orientation.rotate_vec(local_u)
        var world_v = self.orientation.rotate_vec(local_v)

        # End points of cylinder axis
        var top = self.center + world_axis * self.half_height
        var bottom = self.center - world_axis * self.half_height

        # Draw vertical lines (cylinder edges)
        for i in range(self.segments):
            var theta = Float64(i) * 2.0 * pi / Float64(self.segments)
            var offset = world_u * (self.radius * cos(theta)) + world_v * (
                self.radius * sin(theta)
            )

            var p_top = top + offset
            var p_bottom = bottom + offset

            lines.append(WireframeLine(p_top, p_bottom))

        # Draw circles at top and bottom of cylinder
        for i in range(self.segments):
            var theta1 = Float64(i) * 2.0 * pi / Float64(self.segments)
            var theta2 = Float64(i + 1) * 2.0 * pi / Float64(self.segments)

            var offset1 = world_u * (self.radius * cos(theta1)) + world_v * (
                self.radius * sin(theta1)
            )
            var offset2 = world_u * (self.radius * cos(theta2)) + world_v * (
                self.radius * sin(theta2)
            )

            # Top circle
            lines.append(WireframeLine(top + offset1, top + offset2))

            # Bottom circle
            lines.append(WireframeLine(bottom + offset1, bottom + offset2))

        # Draw hemisphere arcs (simplified: just main meridians)
        var num_arcs = 4
        for i in range(num_arcs):
            var theta = Float64(i) * pi / Float64(num_arcs)
            var arc_u = world_u * cos(theta) + world_v * sin(theta)

            # Top hemisphere
            for j in range(self.segments // 2):
                var phi1 = Float64(j) * pi / Float64(self.segments)
                var phi2 = Float64(j + 1) * pi / Float64(self.segments)

                var p1 = (
                    top
                    + arc_u * (self.radius * sin(phi1))
                    + world_axis * (self.radius * cos(phi1))
                )
                var p2 = (
                    top
                    + arc_u * (self.radius * sin(phi2))
                    + world_axis * (self.radius * cos(phi2))
                )
                lines.append(WireframeLine(p1, p2))

            # Bottom hemisphere
            for j in range(self.segments // 2):
                var phi1 = Float64(j) * pi / Float64(self.segments)
                var phi2 = Float64(j + 1) * pi / Float64(self.segments)

                var p1 = (
                    bottom
                    + arc_u * (self.radius * sin(phi1))
                    - world_axis * (self.radius * cos(phi1))
                )
                var p2 = (
                    bottom
                    + arc_u * (self.radius * sin(phi2))
                    - world_axis * (self.radius * cos(phi2))
                )
                lines.append(WireframeLine(p1, p2))

        return lines^


struct WireframeBox:
    """Generate wireframe lines for a box.

    An axis-aligned or oriented box.
    """

    var center: Vec3
    var orientation: Quat
    var half_extents: Vec3

    fn __init__(
        out self,
        center: Vec3 = Vec3.zero(),
        orientation: Quat = Quat.identity(),
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
    ):
        """Initialize box wireframe.

        Args:
            center: Box center position.
            orientation: Box orientation.
            half_extents: Half-extents along local X, Y, Z axes.
        """
        self.center = center
        self.orientation = orientation
        self.half_extents = half_extents

    fn get_lines(self) -> List[WireframeLine]:
        """Generate wireframe lines for the box.

        Returns:
            List of 12 line segments (edges of the box).
        """
        var lines = List[WireframeLine]()

        # Local corners (all 8 combinations of +/- half_extents)
        var corners = List[Vec3]()

        for i in range(8):
            var sx = 1.0 if (i & 1) else -1.0
            var sy = 1.0 if (i & 2) else -1.0
            var sz = 1.0 if (i & 4) else -1.0

            var local = Vec3(
                sx * self.half_extents.x,
                sy * self.half_extents.y,
                sz * self.half_extents.z,
            )

            var world = self.center + self.orientation.rotate_vec(local)
            corners.append(world)

        # Connect corners with edges (12 edges total)
        # Bottom face (z = -hz)
        lines.append(WireframeLine(corners[0], corners[1]))  # 000 -> 001
        lines.append(WireframeLine(corners[1], corners[3]))  # 001 -> 011
        lines.append(WireframeLine(corners[3], corners[2]))  # 011 -> 010
        lines.append(WireframeLine(corners[2], corners[0]))  # 010 -> 000

        # Top face (z = +hz)
        lines.append(WireframeLine(corners[4], corners[5]))  # 100 -> 101
        lines.append(WireframeLine(corners[5], corners[7]))  # 101 -> 111
        lines.append(WireframeLine(corners[7], corners[6]))  # 111 -> 110
        lines.append(WireframeLine(corners[6], corners[4]))  # 110 -> 100

        # Vertical edges
        lines.append(WireframeLine(corners[0], corners[4]))  # 000 -> 100
        lines.append(WireframeLine(corners[1], corners[5]))  # 001 -> 101
        lines.append(WireframeLine(corners[2], corners[6]))  # 010 -> 110
        lines.append(WireframeLine(corners[3], corners[7]))  # 011 -> 111

        return lines^


fn create_ground_grid(
    size: Float64 = 5.0,
    divisions: Int = 10,
    height: Float64 = 0.0,
) -> List[WireframeLine]:
    """Create a wireframe grid for the ground plane.

    Args:
        size: Half-size of the grid (grid goes from -size to +size).
        divisions: Number of divisions per side.
        height: Z-coordinate of the grid.

    Returns:
        List of line segments forming the grid.
    """
    var lines = List[WireframeLine]()
    var step = 2.0 * size / Float64(divisions)

    # Lines parallel to X axis
    for i in range(divisions + 1):
        var y = -size + Float64(i) * step
        lines.append(
            WireframeLine(
                Vec3(-size, y, height),
                Vec3(size, y, height),
            )
        )

    # Lines parallel to Y axis
    for i in range(divisions + 1):
        var x = -size + Float64(i) * step
        lines.append(
            WireframeLine(
                Vec3(x, -size, height),
                Vec3(x, size, height),
            )
        )

    return lines^


fn create_axes(
    origin: Vec3 = Vec3.zero(),
    length: Float64 = 1.0,
) -> List[WireframeLine]:
    """Create coordinate axes visualization.

    Args:
        origin: Origin point.
        length: Length of each axis.

    Returns:
        List of 3 line segments (X, Y, Z axes).
    """
    var lines = List[WireframeLine]()

    # X axis (red)
    lines.append(WireframeLine(origin, origin + Vec3(length, 0.0, 0.0)))
    # Y axis (green)
    lines.append(WireframeLine(origin, origin + Vec3(0.0, length, 0.0)))
    # Z axis (blue)
    lines.append(WireframeLine(origin, origin + Vec3(0.0, 0.0, length)))

    return lines^
