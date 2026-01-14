"""Transform utilities for 2D rendering.

Provides Transform2D for object positioning/rotation and Camera for
viewport management with world-to-screen coordinate conversion.
"""

from math import cos, sin
from .sdl2 import SDL_Point


@fieldwise_init
struct Vec2(ImplicitlyCopyable, Movable):
    """2D vector for positions and directions."""

    var x: Float64
    var y: Float64

    fn __add__(self, other: Self) -> Self:
        return Self(self.x + other.x, self.y + other.y)

    fn __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y)

    fn __mul__(self, scalar: Float64) -> Self:
        return Self(self.x * scalar, self.y * scalar)

    fn __neg__(self) -> Self:
        return Self(-self.x, -self.y)

    fn length(self) -> Float64:
        return (self.x * self.x + self.y * self.y) ** 0.5

    fn normalized(self) -> Self:
        var len = self.length()
        if len > 0:
            return Self(self.x / len, self.y / len)
        return Self(0, 0)

    fn rotated(self, angle: Float64) -> Self:
        """Rotate vector by angle (radians)."""
        var c = cos(angle)
        var s = sin(angle)
        return Self(self.x * c - self.y * s, self.x * s + self.y * c)

    fn to_point(self) -> SDL_Point:
        """Convert to SDL_Point (integer coordinates)."""
        return SDL_Point(Int32(self.x), Int32(self.y))


@fieldwise_init
struct Transform2D(ImplicitlyCopyable, Movable):
    """2D transformation with position, rotation, and scale.

    Used to transform local coordinates to world coordinates.
    Rotation is applied first, then scale, then translation.
    """

    var x: Float64
    var y: Float64
    var angle: Float64  # radians
    var scale_x: Float64
    var scale_y: Float64

    fn __init__(out self, x: Float64, y: Float64, angle: Float64 = 0.0):
        """Create transform with uniform scale of 1.0."""
        self.x = x
        self.y = y
        self.angle = angle
        self.scale_x = 1.0
        self.scale_y = 1.0

    fn __init__(
        out self,
        x: Float64,
        y: Float64,
        angle: Float64,
        scale: Float64,
    ):
        """Create transform with uniform scale."""
        self.x = x
        self.y = y
        self.angle = angle
        self.scale_x = scale
        self.scale_y = scale

    fn apply(self, point: Vec2) -> Vec2:
        """Transform a local point to world coordinates.

        Args:
            point: Point in local coordinates.

        Returns:
            Point in world coordinates.
        """
        var c = cos(self.angle)
        var s = sin(self.angle)
        # Rotate, then scale, then translate
        var rx = point.x * c - point.y * s
        var ry = point.x * s + point.y * c
        return Vec2(
            self.x + rx * self.scale_x,
            self.y + ry * self.scale_y,
        )

    fn apply_direction(self, direction: Vec2) -> Vec2:
        """Transform a direction vector (no translation).

        Args:
            direction: Direction in local coordinates.

        Returns:
            Direction in world coordinates.
        """
        var c = cos(self.angle)
        var s = sin(self.angle)
        var rx = direction.x * c - direction.y * s
        var ry = direction.x * s + direction.y * c
        return Vec2(rx * self.scale_x, ry * self.scale_y)

    fn position(self) -> Vec2:
        """Get position as Vec2."""
        return Vec2(self.x, self.y)

    fn with_offset(self, offset: Vec2) -> Self:
        """Return new transform with position offset in world space."""
        return Self(self.x + offset.x, self.y + offset.y, self.angle, self.scale_x, self.scale_y)

    fn with_local_offset(self, offset: Vec2) -> Self:
        """Return new transform with position offset in local space."""
        var world_offset = self.apply_direction(offset)
        return Self(self.x + world_offset.x, self.y + world_offset.y, self.angle, self.scale_x, self.scale_y)


struct Camera(ImplicitlyCopyable, Movable):
    """Camera for viewport management and coordinate conversion.

    Handles world-to-screen coordinate transformation with optional
    Y-axis flipping (common in 2D games where Y increases upward in
    world space but downward in screen space).
    """

    var x: Float64  # World position (center of viewport)
    var y: Float64
    var zoom: Float64  # Pixels per world unit
    var screen_width: Int
    var screen_height: Int
    var flip_y: Bool  # If True, Y increases upward in world space

    fn __init__(
        out self,
        x: Float64,
        y: Float64,
        zoom: Float64,
        screen_width: Int,
        screen_height: Int,
        flip_y: Bool = True,
    ):
        """Create a camera.

        Args:
            x: World X position (center of viewport).
            y: World Y position (center of viewport).
            zoom: Scale factor (pixels per world unit).
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.
            flip_y: If True, Y increases upward in world (default True).
        """
        self.x = x
        self.y = y
        self.zoom = zoom
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.flip_y = flip_y

    fn world_to_screen(self, world_x: Float64, world_y: Float64) -> SDL_Point:
        """Convert world coordinates to screen coordinates.

        Args:
            world_x: X position in world space.
            world_y: Y position in world space.

        Returns:
            Screen position as SDL_Point.
        """
        var sx = (world_x - self.x) * self.zoom + Float64(self.screen_width) / 2.0
        var sy: Float64
        if self.flip_y:
            sy = Float64(self.screen_height) / 2.0 - (world_y - self.y) * self.zoom
        else:
            sy = (world_y - self.y) * self.zoom + Float64(self.screen_height) / 2.0
        return SDL_Point(Int32(sx), Int32(sy))

    fn world_to_screen_vec(self, world: Vec2) -> Vec2:
        """Convert world coordinates to screen coordinates as Vec2.

        Args:
            world: Position in world space.

        Returns:
            Screen position as Vec2.
        """
        var sx = (world.x - self.x) * self.zoom + Float64(self.screen_width) / 2.0
        var sy: Float64
        if self.flip_y:
            sy = Float64(self.screen_height) / 2.0 - (world.y - self.y) * self.zoom
        else:
            sy = (world.y - self.y) * self.zoom + Float64(self.screen_height) / 2.0
        return Vec2(sx, sy)

    fn screen_to_world(self, screen_x: Int, screen_y: Int) -> Vec2:
        """Convert screen coordinates to world coordinates.

        Args:
            screen_x: X position in screen space.
            screen_y: Y position in screen space.

        Returns:
            World position as Vec2.
        """
        var wx = (Float64(screen_x) - Float64(self.screen_width) / 2.0) / self.zoom + self.x
        var wy: Float64
        if self.flip_y:
            wy = (Float64(self.screen_height) / 2.0 - Float64(screen_y)) / self.zoom + self.y
        else:
            wy = (Float64(screen_y) - Float64(self.screen_height) / 2.0) / self.zoom + self.y
        return Vec2(wx, wy)

    fn world_to_screen_scale(self, world_size: Float64) -> Int:
        """Convert a world-space size to screen pixels.

        Args:
            world_size: Size in world units.

        Returns:
            Size in pixels.
        """
        return Int(world_size * self.zoom)

    fn follow(mut self, target: Vec2, smoothing: Float64 = 1.0):
        """Move camera toward target position.

        Args:
            target: Target world position.
            smoothing: Interpolation factor (0-1, 1 = instant).
        """
        self.x += (target.x - self.x) * smoothing
        self.y += (target.y - self.y) * smoothing

    fn set_position(mut self, x: Float64, y: Float64):
        """Set camera position directly."""
        self.x = x
        self.y = y

    fn get_viewport_bounds(self) -> Tuple[Vec2, Vec2]:
        """Get the world-space bounds of the visible viewport.

        Returns:
            Tuple of (min_corner, max_corner) in world coordinates.
        """
        var half_w = Float64(self.screen_width) / 2.0 / self.zoom
        var half_h = Float64(self.screen_height) / 2.0 / self.zoom
        if self.flip_y:
            return (
                Vec2(self.x - half_w, self.y - half_h),
                Vec2(self.x + half_w, self.y + half_h),
            )
        else:
            return (
                Vec2(self.x - half_w, self.y - half_h),
                Vec2(self.x + half_w, self.y + half_h),
            )

    fn is_visible(self, world_pos: Vec2, margin: Float64 = 0.0) -> Bool:
        """Check if a world position is visible in the viewport.

        Args:
            world_pos: Position to check.
            margin: Extra margin around viewport.

        Returns:
            True if position is visible.
        """
        var bounds = self.get_viewport_bounds()
        var min_corner = bounds[0]
        var max_corner = bounds[1]
        return (
            world_pos.x >= min_corner.x - margin
            and world_pos.x <= max_corner.x + margin
            and world_pos.y >= min_corner.y - margin
            and world_pos.y <= max_corner.y + margin
        )


# Factory functions for common camera setups


fn make_centered_camera(
    screen_width: Int,
    screen_height: Int,
    zoom: Float64,
    flip_y: Bool = True,
) -> Camera:
    """Create a camera centered at origin.

    Args:
        screen_width: Screen width in pixels.
        screen_height: Screen height in pixels.
        zoom: Scale factor (pixels per world unit).
        flip_y: If True, Y increases upward.

    Returns:
        Camera centered at (0, 0).
    """
    return Camera(0.0, 0.0, zoom, screen_width, screen_height, flip_y)


fn make_offset_camera(
    screen_width: Int,
    screen_height: Int,
    zoom: Float64,
    offset_x: Float64,
    offset_y: Float64,
    flip_y: Bool = True,
) -> Camera:
    """Create a camera with custom center offset.

    Args:
        screen_width: Screen width in pixels.
        screen_height: Screen height in pixels.
        zoom: Scale factor (pixels per world unit).
        offset_x: World X at screen center.
        offset_y: World Y at screen center.
        flip_y: If True, Y increases upward.

    Returns:
        Camera at specified position.
    """
    return Camera(offset_x, offset_y, zoom, screen_width, screen_height, flip_y)


struct RotatingCamera(ImplicitlyCopyable, Movable):
    """Camera with rotation support for top-down views.

    Used for environments like CarRacing where the camera rotates
    to follow the player's orientation.
    """

    var x: Float64  # World position (center of viewport)
    var y: Float64
    var angle: Float64  # Camera rotation in radians
    var zoom: Float64  # Pixels per world unit
    var screen_width: Int
    var screen_height: Int
    var screen_center_x: Float64  # Screen X where camera center is drawn
    var screen_center_y: Float64  # Screen Y where camera center is drawn

    fn __init__(
        out self,
        x: Float64,
        y: Float64,
        angle: Float64,
        zoom: Float64,
        screen_width: Int,
        screen_height: Int,
        screen_center_x: Float64 = -1.0,
        screen_center_y: Float64 = -1.0,
    ):
        """Create a rotating camera.

        Args:
            x: World X position (center of viewport).
            y: World Y position (center of viewport).
            angle: Camera rotation in radians.
            zoom: Scale factor (pixels per world unit).
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.
            screen_center_x: Screen X for camera center (-1 = screen center).
            screen_center_y: Screen Y for camera center (-1 = screen center).
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.zoom = zoom
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_center_x if screen_center_x >= 0 else Float64(screen_width) / 2.0
        self.screen_center_y = screen_center_y if screen_center_y >= 0 else Float64(screen_height) / 2.0

    fn world_to_screen(self, world_x: Float64, world_y: Float64) -> SDL_Point:
        """Convert world coordinates to screen coordinates with rotation.

        Args:
            world_x: X position in world space.
            world_y: Y position in world space.

        Returns:
            Screen position as SDL_Point.
        """
        # Translate relative to camera
        var dx = world_x - self.x
        var dy = world_y - self.y

        # Rotate by camera angle
        var c = cos(self.angle)
        var s = sin(self.angle)
        var rx = dx * c - dy * s
        var ry = dx * s + dy * c

        # Scale and offset to screen center
        var screen_x = self.screen_center_x + rx * self.zoom
        var screen_y = self.screen_center_y - ry * self.zoom  # Flip Y

        return SDL_Point(Int32(screen_x), Int32(screen_y))

    fn world_to_screen_vec(self, world: Vec2) -> Vec2:
        """Convert world coordinates to screen coordinates as Vec2.

        Args:
            world: Position in world space.

        Returns:
            Screen position as Vec2.
        """
        var dx = world.x - self.x
        var dy = world.y - self.y
        var c = cos(self.angle)
        var s = sin(self.angle)
        var rx = dx * c - dy * s
        var ry = dx * s + dy * c
        return Vec2(
            self.screen_center_x + rx * self.zoom,
            self.screen_center_y - ry * self.zoom,
        )

    fn world_to_screen_scale(self, world_size: Float64) -> Int:
        """Convert a world-space size to screen pixels.

        Args:
            world_size: Size in world units.

        Returns:
            Size in pixels.
        """
        return Int(world_size * self.zoom)

    fn set_position(mut self, x: Float64, y: Float64):
        """Set camera position directly."""
        self.x = x
        self.y = y

    fn set_angle(mut self, angle: Float64):
        """Set camera rotation angle."""
        self.angle = angle

    fn follow(mut self, target_x: Float64, target_y: Float64, target_angle: Float64, smoothing: Float64 = 1.0):
        """Move camera toward target position and angle.

        Args:
            target_x: Target world X position.
            target_y: Target world Y position.
            target_angle: Target rotation angle.
            smoothing: Interpolation factor (0-1, 1 = instant).
        """
        self.x += (target_x - self.x) * smoothing
        self.y += (target_y - self.y) * smoothing
        self.angle += (target_angle - self.angle) * smoothing


fn make_rotating_camera(
    screen_width: Int,
    screen_height: Int,
    zoom: Float64,
) -> RotatingCamera:
    """Create a rotating camera centered on screen.

    Args:
        screen_width: Screen width in pixels.
        screen_height: Screen height in pixels.
        zoom: Scale factor (pixels per world unit).

    Returns:
        RotatingCamera centered at origin with no rotation.
    """
    return RotatingCamera(0.0, 0.0, 0.0, zoom, screen_width, screen_height)
