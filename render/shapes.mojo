"""Common shape definitions for rendering.

Provides factory functions for creating vertex lists for common shapes
used in RL environment visualization (rectangles, polygons, arrows, etc.).
"""

from math import cos, sin, pi
from .transform import Vec2


# =============================================================================
# Basic Shape Factories
# =============================================================================


fn make_rect(
    width: Float64, height: Float64, centered: Bool = True
) -> List[Vec2]:
    """Create rectangle vertices.

    Args:
        width: Rectangle width.
        height: Rectangle height.
        centered: If True, center is at origin; else bottom-left is at origin.

    Returns:
        List of 4 vertices (counter-clockwise from bottom-left).
    """
    var vertices = List[Vec2]()
    if centered:
        var hw = width / 2.0
        var hh = height / 2.0
        vertices.append(Vec2(-hw, -hh))
        vertices.append(Vec2(hw, -hh))
        vertices.append(Vec2(hw, hh))
        vertices.append(Vec2(-hw, hh))
    else:
        vertices.append(Vec2(0, 0))
        vertices.append(Vec2(width, 0))
        vertices.append(Vec2(width, height))
        vertices.append(Vec2(0, height))
    return vertices^


fn make_box(half_width: Float64, half_height: Float64) -> List[Vec2]:
    """Create box vertices centered at origin.

    Args:
        half_width: Half width.
        half_height: Half height.

    Returns:
        List of 4 vertices.
    """
    return make_rect(half_width * 2.0, half_height * 2.0, centered=True)^


fn make_triangle(
    base: Float64, height: Float64, centered: Bool = True
) -> List[Vec2]:
    """Create isoceles triangle vertices pointing up.

    Args:
        base: Triangle base width.
        height: Triangle height.
        centered: If True, centroid is at origin.

    Returns:
        List of 3 vertices.
    """
    var vertices = List[Vec2]()
    if centered:
        # Centroid is at 1/3 height from base
        var centroid_y = height / 3.0
        vertices.append(Vec2(-base / 2.0, -centroid_y))
        vertices.append(Vec2(base / 2.0, -centroid_y))
        vertices.append(Vec2(0, height - centroid_y))
    else:
        vertices.append(Vec2(0, 0))
        vertices.append(Vec2(base, 0))
        vertices.append(Vec2(base / 2.0, height))
    return vertices^^


fn make_circle(radius: Float64, segments: Int = 32) -> List[Vec2]:
    """Create circle vertices.

    Args:
        radius: Circle radius.
        segments: Number of segments (more = smoother).

    Returns:
        List of vertices forming a circle.
    """
    var vertices = List[Vec2]()
    var angle_step = 2.0 * pi / Float64(segments)
    for i in range(segments):
        var angle = Float64(i) * angle_step
        vertices.append(Vec2(radius * cos(angle), radius * sin(angle)))
    return vertices^


fn make_regular_polygon(radius: Float64, sides: Int) -> List[Vec2]:
    """Create regular polygon vertices.

    Args:
        radius: Distance from center to vertices.
        sides: Number of sides.

    Returns:
        List of vertices.
    """
    var vertices = List[Vec2]()
    var angle_step = 2.0 * pi / Float64(sides)
    # Start at top (angle = pi/2) for visual consistency
    var start_angle = pi / 2.0
    for i in range(sides):
        var angle = start_angle + Float64(i) * angle_step
        vertices.append(Vec2(radius * cos(angle), radius * sin(angle)))
    return vertices^


fn make_hexagon(radius: Float64) -> List[Vec2]:
    """Create hexagon vertices.

    Args:
        radius: Distance from center to vertices.

    Returns:
        List of 6 vertices.
    """
    return make_regular_polygon(radius, 6)^


# =============================================================================
# Arrow and Direction Indicators
# =============================================================================


fn make_arrow(
    length: Float64, head_size: Float64, shaft_width: Float64
) -> List[Vec2]:
    """Create arrow vertices pointing right (positive X).

    Args:
        length: Total arrow length.
        head_size: Size of arrowhead.
        shaft_width: Width of arrow shaft.

    Returns:
        List of vertices forming arrow polygon.
    """
    var vertices = List[Vec2]()
    var hw = shaft_width / 2.0
    var shaft_length = length - head_size

    # Shaft (4 vertices)
    vertices.append(Vec2(0, -hw))
    vertices.append(Vec2(shaft_length, -hw))
    # Arrowhead notch
    vertices.append(Vec2(shaft_length, -head_size / 2.0))
    # Arrowhead tip
    vertices.append(Vec2(length, 0))
    # Other side of arrowhead
    vertices.append(Vec2(shaft_length, head_size / 2.0))
    vertices.append(Vec2(shaft_length, hw))
    vertices.append(Vec2(0, hw))

    return vertices^


fn make_simple_arrow_head(size: Float64) -> List[Vec2]:
    """Create simple triangular arrowhead pointing right.

    Args:
        size: Size of arrowhead.

    Returns:
        List of 3 vertices.
    """
    var vertices = List[Vec2]()
    vertices.append(Vec2(0, -size / 2.0))
    vertices.append(Vec2(size, 0))
    vertices.append(Vec2(0, size / 2.0))
    return vertices^


fn make_chevron(size: Float64, thickness: Float64) -> List[Vec2]:
    """Create chevron (>) shape pointing right.

    Args:
        size: Overall size.
        thickness: Line thickness.

    Returns:
        List of vertices.
    """
    var vertices = List[Vec2]()
    var ht = thickness / 2.0

    # Outer chevron
    vertices.append(Vec2(-size / 2.0 + ht, -size / 2.0))
    vertices.append(Vec2(size / 2.0, 0))
    vertices.append(Vec2(-size / 2.0 + ht, size / 2.0))
    # Inner chevron
    vertices.append(Vec2(-size / 2.0 - ht, size / 2.0 - thickness))
    vertices.append(Vec2(size / 2.0 - thickness * 1.5, 0))
    vertices.append(Vec2(-size / 2.0 - ht, -size / 2.0 + thickness))

    return vertices^


# =============================================================================
# Vehicle/Robot Parts
# =============================================================================


fn make_wheel(radius: Float64, segments: Int = 16) -> List[Vec2]:
    """Create wheel vertices (same as circle but with fewer segments).

    Args:
        radius: Wheel radius.
        segments: Number of segments.

    Returns:
        List of vertices.
    """
    return make_circle(radius, segments)^


fn make_capsule(
    length: Float64, radius: Float64, segments: Int = 8
) -> List[Vec2]:
    """Create capsule/pill shape vertices (rectangle with semicircle ends).

    Args:
        length: Total length (including caps).
        radius: Cap radius (also half-width).
        segments: Segments per semicircle.

    Returns:
        List of vertices.
    """
    var vertices = List[Vec2]()
    var half_length = (length - 2.0 * radius) / 2.0

    # Right semicircle
    for i in range(segments + 1):
        var angle = -pi / 2.0 + pi * Float64(i) / Float64(segments)
        vertices.append(
            Vec2(half_length + radius * cos(angle), radius * sin(angle))
        )

    # Left semicircle
    for i in range(segments + 1):
        var angle = pi / 2.0 + pi * Float64(i) / Float64(segments)
        vertices.append(
            Vec2(-half_length + radius * cos(angle), radius * sin(angle))
        )

    return vertices^


fn make_car_body(
    length: Float64, width: Float64, cabin_ratio: Float64 = 0.5
) -> List[Vec2]:
    """Create simple car body silhouette.

    Args:
        length: Car length.
        width: Car width.
        cabin_ratio: How much of length is cabin vs hood/trunk.

    Returns:
        List of vertices.
    """
    var vertices = List[Vec2]()
    var hl = length / 2.0
    var hw = width / 2.0
    var cabin_start = -hl + length * (1.0 - cabin_ratio) / 2.0
    var cabin_end = hl - length * (1.0 - cabin_ratio) / 2.0
    var cabin_height = width * 0.6

    # Bottom edge (front to back)
    vertices.append(Vec2(-hl, -hw))
    vertices.append(Vec2(hl, -hw))
    # Back
    vertices.append(Vec2(hl, hw))
    # Roof line with cabin
    vertices.append(Vec2(cabin_end, hw))
    vertices.append(Vec2(cabin_end, hw + cabin_height))
    vertices.append(Vec2(cabin_start, hw + cabin_height))
    vertices.append(Vec2(cabin_start, hw))
    # Front
    vertices.append(Vec2(-hl, hw))

    return vertices^


fn make_lander_body() -> List[Vec2]:
    """Create lunar lander body vertices (hexagonal shape).

    Returns:
        List of 6 vertices matching LunarLander env.
    """
    var vertices = List[Vec2]()
    # Vertices based on Gymnasium LunarLander
    vertices.append(Vec2(-14, 17))
    vertices.append(Vec2(-17, 0))
    vertices.append(Vec2(-17, -10))
    vertices.append(Vec2(17, -10))
    vertices.append(Vec2(17, 0))
    vertices.append(Vec2(14, 17))
    return vertices^


fn make_leg_box(width: Float64, height: Float64) -> List[Vec2]:
    """Create leg segment box vertices.

    Args:
        width: Leg width.
        height: Leg height/length.

    Returns:
        List of 4 vertices.
    """
    return make_rect(width, height, centered=True)


# =============================================================================
# UI Elements
# =============================================================================


fn make_flag(
    pole_height: Float64, flag_width: Float64, flag_height: Float64
) -> Tuple[List[Vec2], List[Vec2]]:
    """Create flag pole and flag vertices.

    Args:
        pole_height: Height of pole.
        flag_width: Width of flag.
        flag_height: Height of flag.

    Returns:
        Tuple of (pole_vertices, flag_vertices).
    """
    # Pole is a thin rectangle
    var pole = List[Vec2]()
    pole.append(Vec2(-1, 0))
    pole.append(Vec2(1, 0))
    pole.append(Vec2(1, pole_height))
    pole.append(Vec2(-1, pole_height))

    # Flag is a triangle
    var flag = List[Vec2]()
    flag.append(Vec2(0, pole_height))
    flag.append(Vec2(flag_width, pole_height - flag_height / 2.0))
    flag.append(Vec2(0, pole_height - flag_height))

    return (pole^, flag^)


fn make_star(
    outer_radius: Float64, inner_radius: Float64, points: Int = 5
) -> List[Vec2]:
    """Create star shape vertices.

    Args:
        outer_radius: Radius to outer points.
        inner_radius: Radius to inner points.
        points: Number of points.

    Returns:
        List of vertices.
    """
    var vertices = List[Vec2]()
    var angle_step = pi / Float64(points)
    var start_angle = pi / 2.0  # Start at top

    for i in range(points * 2):
        var angle = start_angle + Float64(i) * angle_step
        var radius = outer_radius if i % 2 == 0 else inner_radius
        vertices.append(Vec2(radius * cos(angle), radius * sin(angle)))

    return vertices^


fn make_cross(size: Float64, thickness: Float64) -> List[Vec2]:
    """Create plus/cross shape vertices.

    Args:
        size: Overall size.
        thickness: Arm thickness.

    Returns:
        List of 12 vertices.
    """
    var vertices = List[Vec2]()
    var hs = size / 2.0
    var ht = thickness / 2.0

    # Going clockwise from top-left of top arm
    vertices.append(Vec2(-ht, hs))
    vertices.append(Vec2(ht, hs))
    vertices.append(Vec2(ht, ht))
    vertices.append(Vec2(hs, ht))
    vertices.append(Vec2(hs, -ht))
    vertices.append(Vec2(ht, -ht))
    vertices.append(Vec2(ht, -hs))
    vertices.append(Vec2(-ht, -hs))
    vertices.append(Vec2(-ht, -ht))
    vertices.append(Vec2(-hs, -ht))
    vertices.append(Vec2(-hs, ht))
    vertices.append(Vec2(-ht, ht))

    return vertices^


# =============================================================================
# Terrain Generation
# =============================================================================


fn make_terrain_line(
    func: fn (Float64) -> Float64,
    x_min: Float64,
    x_max: Float64,
    num_points: Int,
) -> List[Vec2]:
    """Generate terrain line from a height function.

    Args:
        func: Function that returns height for x position.
        x_min: Start X position.
        x_max: End X position.
        num_points: Number of sample points.

    Returns:
        List of points along terrain.
    """
    var points = List[Vec2]()
    var step = (x_max - x_min) / Float64(num_points - 1)

    for i in range(num_points):
        var x = x_min + Float64(i) * step
        var y = func(x)
        points.append(Vec2(x, y))

    return points^


fn make_filled_terrain(
    terrain_line: List[Vec2], bottom_y: Float64
) -> List[Vec2]:
    """Convert terrain line to filled polygon.

    Args:
        terrain_line: Line points along terrain surface.
        bottom_y: Y coordinate for bottom of polygon.

    Returns:
        Polygon vertices (terrain line + bottom edge).
    """
    var vertices = List[Vec2]()

    # Add terrain line points
    for i in range(len(terrain_line)):
        vertices.append(terrain_line[i])

    # Add bottom edge (right to left)
    if len(terrain_line) > 0:
        vertices.append(Vec2(terrain_line[len(terrain_line) - 1].x, bottom_y))
        vertices.append(Vec2(terrain_line[0].x, bottom_y))

    return vertices^


# =============================================================================
# Transform Utilities
# =============================================================================


fn offset_vertices(vertices: List[Vec2], offset: Vec2) -> List[Vec2]:
    """Offset all vertices by a fixed amount.

    Args:
        vertices: Original vertices.
        offset: Offset to apply.

    Returns:
        New list with offset vertices.
    """
    var result = List[Vec2]()
    for i in range(len(vertices)):
        result.append(vertices[i] + offset)
    return result^


fn scale_vertices(vertices: List[Vec2], scale: Float64) -> List[Vec2]:
    """Scale all vertices uniformly.

    Args:
        vertices: Original vertices.
        scale: Scale factor.

    Returns:
        New list with scaled vertices.
    """
    var result = List[Vec2]()
    for i in range(len(vertices)):
        result.append(vertices[i] * scale)
    return result^


fn rotate_vertices(vertices: List[Vec2], angle: Float64) -> List[Vec2]:
    """Rotate all vertices around origin.

    Args:
        vertices: Original vertices.
        angle: Rotation angle in radians.

    Returns:
        New list with rotated vertices.
    """
    var result = List[Vec2]()
    for i in range(len(vertices)):
        result.append(vertices[i].rotated(angle))
    return result^


fn flip_vertices_y(vertices: List[Vec2]) -> List[Vec2]:
    """Flip vertices vertically (negate Y).

    Args:
        vertices: Original vertices.

    Returns:
        New list with flipped vertices.
    """
    var result = List[Vec2]()
    for i in range(len(vertices)):
        result.append(Vec2(vertices[i].x, -vertices[i].y))
    return result^


fn flip_vertices_x(vertices: List[Vec2]) -> List[Vec2]:
    """Flip vertices horizontally (negate X).

    Args:
        vertices: Original vertices.

    Returns:
        New list with flipped vertices.
    """
    var result = List[Vec2]()
    for i in range(len(vertices)):
        result.append(Vec2(-vertices[i].x, vertices[i].y))
    return result^
