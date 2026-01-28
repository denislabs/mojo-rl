"""3D Wireframe Renderer.

Software wireframe renderer using SDL2 for display.
Projects 3D shapes to 2D and draws them as line segments.
"""

from math import sqrt
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render.sdl2 import SDL2, SDL_Event, SDL_QUIT, SDL_Point
from .camera3d import Camera3D
from .shapes3d import (
    WireframeLine,
    WireframeSphere,
    WireframeCapsule,
    WireframeBox,
    create_ground_grid,
    create_axes,
)

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


struct Color3D:
    """RGB color for wireframe rendering."""

    var r: UInt8
    var g: UInt8
    var b: UInt8

    fn __init__(out self, r: UInt8, g: UInt8, b: UInt8):
        self.r = r
        self.g = g
        self.b = b

    @staticmethod
    fn white() -> Self:
        return Self(255, 255, 255)

    @staticmethod
    fn black() -> Self:
        return Self(0, 0, 0)

    @staticmethod
    fn red() -> Self:
        return Self(255, 0, 0)

    @staticmethod
    fn green() -> Self:
        return Self(0, 255, 0)

    @staticmethod
    fn blue() -> Self:
        return Self(0, 0, 255)

    @staticmethod
    fn yellow() -> Self:
        return Self(255, 255, 0)

    @staticmethod
    fn cyan() -> Self:
        return Self(0, 255, 255)

    @staticmethod
    fn gray() -> Self:
        return Self(128, 128, 128)

    @staticmethod
    fn dark_gray() -> Self:
        return Self(64, 64, 64)


struct Renderer3D:
    """3D wireframe renderer using SDL2.

    Projects 3D shapes to 2D screen coordinates and draws them
    as wireframe line segments.
    """

    var sdl: SDL2
    var camera: Camera3D
    var width: Int
    var height: Int
    var background_color: Color3D
    var draw_grid: Bool
    var draw_axes: Bool
    var should_quit: Bool

    fn __init__(
        out self,
        width: Int = 800,
        height: Int = 450,
        camera: Camera3D = Camera3D(),
        draw_grid: Bool = True,
        draw_axes: Bool = True,
    ) raises:
        """Initialize the 3D renderer.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            camera: Camera for viewing the scene.
            draw_grid: Whether to draw ground grid.
            draw_axes: Whether to draw coordinate axes.
        """
        self.sdl = SDL2()
        self.width = width
        self.height = height
        self.background_color = Color3D(32, 32, 48)  # Dark blue-gray
        self.draw_grid = draw_grid
        self.draw_axes = draw_axes
        self.should_quit = False

        # Copy camera and update screen size
        self.camera = Camera3D(
            eye=camera.eye,
            target=camera.target,
            up=camera.up,
            fov=camera.fov
            * 180.0
            / 3.14159265358979,  # Convert back to degrees
            aspect=Float64(width) / Float64(height),
            near=camera.near,
            far=camera.far,
            screen_width=width,
            screen_height=height,
        )

    fn init(mut self, mut title: String):
        """Initialize SDL2 and create window.

        Args:
            title: Window title.
        """
        _ = self.sdl.init()
        _ = self.sdl.create_window(title, self.width, self.height)
        _ = self.sdl.create_renderer()

    fn close(mut self):
        """Close the renderer and cleanup SDL2."""
        self.sdl.quit()

    fn begin_frame(self):
        """Begin a new frame (clear the screen)."""
        self.sdl.set_draw_color(
            self.background_color.r,
            self.background_color.g,
            self.background_color.b,
        )
        self.sdl.clear()

    fn end_frame(self):
        """End the frame (present to screen)."""
        self.sdl.present()

    fn draw_line_3d(self, line: WireframeLine, color: Color3D):
        """Draw a 3D line segment.

        Args:
            line: Line segment in world space.
            color: Line color.
        """
        var start = self.camera.project_to_screen(line.start)
        var end = self.camera.project_to_screen(line.end)

        # Only draw if both endpoints are visible
        if start[2] and end[2]:
            self.sdl.set_draw_color(color.r, color.g, color.b)
            self.sdl.draw_line(start[0], start[1], end[0], end[1])

    fn draw_lines_3d(self, lines: List[WireframeLine], color: Color3D):
        """Draw multiple 3D line segments.

        Args:
            lines: List of line segments.
            color: Line color.
        """
        self.sdl.set_draw_color(color.r, color.g, color.b)

        for i in range(len(lines)):
            var start = self.camera.project_to_screen(lines[i].start)
            var end = self.camera.project_to_screen(lines[i].end)

            if start[2] and end[2]:
                self.sdl.draw_line(start[0], start[1], end[0], end[1])

    fn draw_sphere(
        self,
        center: Vec3,
        radius: Float64,
        color: Color3D = Color3D.white(),
        segments: Int = 12,
        rings: Int = 8,
    ):
        """Draw a wireframe sphere.

        Args:
            center: Sphere center.
            radius: Sphere radius.
            color: Wireframe color.
            segments: Number of longitude segments.
            rings: Number of latitude rings.
        """
        var sphere = WireframeSphere(center, radius, segments, rings)
        var lines = sphere.get_lines()
        self.draw_lines_3d(lines, color)

    fn draw_capsule(
        self,
        center: Vec3,
        orientation: Quat,
        radius: Float64,
        half_height: Float64,
        axis: Int = 2,
        color: Color3D = Color3D.white(),
        segments: Int = 12,
    ):
        """Draw a wireframe capsule.

        Args:
            center: Capsule center.
            orientation: Capsule orientation.
            radius: Capsule radius.
            half_height: Half-height of cylindrical part.
            axis: Local axis (0=X, 1=Y, 2=Z).
            color: Wireframe color.
            segments: Number of circular segments.
        """
        var capsule = WireframeCapsule(
            center, orientation, radius, half_height, axis, segments
        )
        var lines = capsule.get_lines()
        self.draw_lines_3d(lines, color)

    fn draw_box(
        self,
        center: Vec3,
        orientation: Quat,
        half_extents: Vec3,
        color: Color3D = Color3D.white(),
    ):
        """Draw a wireframe box.

        Args:
            center: Box center.
            orientation: Box orientation.
            half_extents: Half-extents along local X, Y, Z.
            color: Wireframe color.
        """
        var box = WireframeBox(center, orientation, half_extents)
        var lines = box.get_lines()
        self.draw_lines_3d(lines, color)

    fn draw_ground_grid(
        self,
        size: Float64 = 5.0,
        divisions: Int = 10,
        height: Float64 = 0.0,
        color: Color3D = Color3D.dark_gray(),
    ):
        """Draw a ground plane grid.

        Args:
            size: Half-size of the grid.
            divisions: Number of divisions.
            height: Z-coordinate of the grid.
            color: Grid color.
        """
        var lines = create_ground_grid(size, divisions, height)
        self.draw_lines_3d(lines, color)

    fn draw_coordinate_axes(
        self,
        origin: Vec3 = Vec3.zero(),
        length: Float64 = 1.0,
    ):
        """Draw coordinate axes.

        X = red, Y = green, Z = blue.

        Args:
            origin: Origin point.
            length: Length of each axis.
        """
        var lines = create_axes(origin, length)

        # X axis - red
        self.draw_line_3d(lines[0], Color3D.red())
        # Y axis - green
        self.draw_line_3d(lines[1], Color3D.green())
        # Z axis - blue
        self.draw_line_3d(lines[2], Color3D.blue())

    fn render_scene(mut self):
        """Render the default scene elements (grid and axes)."""
        if self.draw_grid:
            self.draw_ground_grid()

        if self.draw_axes:
            self.draw_coordinate_axes()

    fn check_quit(mut self) -> Bool:
        """Check if user wants to quit.

        Polls events and returns True if quit event detected (window close).

        Returns:
            True if quit event detected.
        """
        var event = SDL_Event()

        while self.sdl.poll_event(event):
            if event.type == SDL_QUIT:
                self.should_quit = True
                return True

        return self.should_quit

    fn set_camera_position(mut self, eye: Vec3, target: Vec3):
        """Set camera position and target.

        Args:
            eye: Camera position.
            target: Look-at target.
        """
        self.camera.eye = eye
        self.camera.target = target

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64):
        """Orbit camera around target.

        Args:
            delta_theta: Horizontal rotation (radians).
            delta_phi: Vertical rotation (radians).
        """
        self.camera.orbit(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64):
        """Zoom camera in/out.

        Args:
            delta: Zoom amount.
        """
        self.camera.zoom(delta)

    fn delay(self, ms: Int):
        """Delay for given milliseconds.

        Args:
            ms: Milliseconds to delay.
        """
        self.sdl.delay(ms)

    fn draw_filled_quad_3d(
        self,
        p0: Vec3,
        p1: Vec3,
        p2: Vec3,
        p3: Vec3,
        color: Color3D,
    ):
        """Draw a filled quadrilateral in 3D.

        Args:
            p0, p1, p2, p3: Four corners of the quad (in order).
            color: Fill color.
        """
        # Project all 4 corners to screen
        var s0 = self.camera.project_to_screen(p0)
        var s1 = self.camera.project_to_screen(p1)
        var s2 = self.camera.project_to_screen(p2)
        var s3 = self.camera.project_to_screen(p3)

        # Check if all visible
        if not (s0[2] and s1[2] and s2[2] and s3[2]):
            return

        # Create polygon points
        var points = List[SDL_Point]()
        points.append(SDL_Point(Int32(s0[0]), Int32(s0[1])))
        points.append(SDL_Point(Int32(s1[0]), Int32(s1[1])))
        points.append(SDL_Point(Int32(s2[0]), Int32(s2[1])))
        points.append(SDL_Point(Int32(s3[0]), Int32(s3[1])))

        self.sdl.set_draw_color(color.r, color.g, color.b)
        self.sdl.fill_polygon(points)

    fn draw_filled_circle_2d(
        self,
        screen_x: Int,
        screen_y: Int,
        radius: Int,
        color: Color3D,
    ):
        """Draw a filled circle in screen space.

        Args:
            screen_x: Screen X coordinate.
            screen_y: Screen Y coordinate.
            radius: Circle radius in pixels.
            color: Fill color.
        """
        self.sdl.set_draw_color(color.r, color.g, color.b)
        self.sdl.fill_circle(screen_x, screen_y, radius)

    fn draw_shaded_sphere_2d(
        self,
        screen_x: Int,
        screen_y: Int,
        radius: Int,
        color: Color3D,
    ):
        """Draw a shaded sphere with 3D volume effect.

        Uses concentric circles with gradient colors to simulate
        a lit sphere with ambient and diffuse lighting.

        Args:
            screen_x: Screen X coordinate.
            screen_y: Screen Y coordinate.
            radius: Sphere radius in pixels.
            color: Base color.
        """
        if radius < 2:
            self.sdl.set_draw_color(color.r, color.g, color.b)
            self.sdl.fill_circle(screen_x, screen_y, radius)
            return

        # Light direction (upper-left) - offset for highlight
        var light_offset_x = -radius // 4
        var light_offset_y = -radius // 4

        # Draw concentric circles from outside-in for gradient effect
        # Dark edge -> base color -> bright highlight
        var num_rings = min(radius, 20)

        for i in range(num_rings, -1, -1):
            var t = Float64(i) / Float64(num_rings)  # 1.0 at edge, 0.0 at center
            var ring_radius = Int(Float64(radius) * t)

            # Calculate color: dark at edge, bright toward center
            # Edge is 40% of base color, center is 120% (clamped)
            var brightness = 0.4 + 0.8 * (1.0 - t)

            var r = Int(Float64(color.r) * brightness)
            var g = Int(Float64(color.g) * brightness)
            var b = Int(Float64(color.b) * brightness)

            # Clamp to 255
            r = min(r, 255)
            g = min(g, 255)
            b = min(b, 255)

            self.sdl.set_draw_color(UInt8(r), UInt8(g), UInt8(b))
            self.sdl.fill_circle(screen_x, screen_y, ring_radius)

        # Add specular highlight spot
        var highlight_x = screen_x + light_offset_x
        var highlight_y = screen_y + light_offset_y
        var highlight_radius = max(radius // 5, 2)

        # White-ish highlight blended with color
        var hr = min(Int(Float64(color.r) * 0.3 + 180), 255)
        var hg = min(Int(Float64(color.g) * 0.3 + 180), 255)
        var hb = min(Int(Float64(color.b) * 0.3 + 180), 255)

        self.sdl.set_draw_color(UInt8(hr), UInt8(hg), UInt8(hb))
        self.sdl.fill_circle(highlight_x, highlight_y, highlight_radius)

    fn draw_filled_capsule_2d(
        self,
        center: Vec3,
        orientation: Quat,
        radius: Float64,
        half_height: Float64,
        axis: Int = 2,
        color: Color3D = Color3D.white(),
    ):
        """Draw a filled capsule by projecting to 2D.

        Creates a filled capsule by projecting the 3D capsule endpoints
        and drawing a filled rectangle with circular caps.

        Args:
            center: Capsule center in world space.
            orientation: Capsule orientation.
            radius: Capsule radius.
            half_height: Half-height of cylindrical part.
            axis: Local axis (0=X, 1=Y, 2=Z).
            color: Fill color.
        """
        # Get local axis direction
        var local_axis: Vec3
        if axis == 0:
            local_axis = Vec3.unit_x()
        elif axis == 1:
            local_axis = Vec3.unit_y()
        else:
            local_axis = Vec3.unit_z()

        # Transform to world space
        var world_axis = orientation.rotate_vec(local_axis)

        # End points of capsule
        var top = center + world_axis * half_height
        var bottom = center - world_axis * half_height

        # Project endpoints to screen
        var s_top = self.camera.project_to_screen(top)
        var s_bottom = self.camera.project_to_screen(bottom)
        var s_center = self.camera.project_to_screen(center)

        if not (s_top[2] and s_bottom[2]):
            return

        # Calculate screen-space radius (approximate based on center distance)
        var view_center = self.camera.get_view_matrix().transform_point(center)
        var depth = -view_center.z
        if depth <= 0.1:
            return

        # Approximate screen radius based on perspective
        # Use camera FOV to compute proper scaling
        var fov_scale = 1.0 / (depth * 0.7)  # Adjusted for typical viewing angles
        var screen_radius = Int(radius * Float64(self.height) * fov_scale)
        screen_radius = max(screen_radius, 3)  # Minimum visible size

        # Direction vector on screen
        var dx = Float64(s_top[0] - s_bottom[0])
        var dy = Float64(s_top[1] - s_bottom[1])
        var length = sqrt(dx * dx + dy * dy)

        if length < 1.0:
            # Too small, just draw a circle
            self.sdl.set_draw_color(color.r, color.g, color.b)
            self.sdl.fill_circle(s_center[0], s_center[1], screen_radius)
            return

        # Perpendicular direction for width
        var perp_x = -dy / length * Float64(screen_radius)
        var perp_y = dx / length * Float64(screen_radius)

        # Draw filled quadrilateral for the body
        var points = List[SDL_Point]()
        points.append(
            SDL_Point(
                Int32(s_top[0] + Int(perp_x)), Int32(s_top[1] + Int(perp_y))
            )
        )
        points.append(
            SDL_Point(
                Int32(s_top[0] - Int(perp_x)), Int32(s_top[1] - Int(perp_y))
            )
        )
        points.append(
            SDL_Point(
                Int32(s_bottom[0] - Int(perp_x)),
                Int32(s_bottom[1] - Int(perp_y)),
            )
        )
        points.append(
            SDL_Point(
                Int32(s_bottom[0] + Int(perp_x)),
                Int32(s_bottom[1] + Int(perp_y)),
            )
        )

        self.sdl.set_draw_color(color.r, color.g, color.b)
        self.sdl.fill_polygon(points)

        # Draw filled circles at the caps
        self.sdl.fill_circle(s_top[0], s_top[1], screen_radius)
        self.sdl.fill_circle(s_bottom[0], s_bottom[1], screen_radius)

    fn draw_shaded_capsule_2d(
        self,
        center: Vec3,
        orientation: Quat,
        radius: Float64,
        half_height: Float64,
        axis: Int = 2,
        color: Color3D = Color3D.white(),
    ):
        """Draw a shaded capsule with 3D volume effect.

        Creates a capsule with gradient shading to simulate lighting,
        giving the appearance of a 3D cylinder with hemispherical caps.

        Args:
            center: Capsule center in world space.
            orientation: Capsule orientation.
            radius: Capsule radius.
            half_height: Half-height of cylindrical part.
            axis: Local axis (0=X, 1=Y, 2=Z).
            color: Base color.
        """
        # Get local axis direction
        var local_axis: Vec3
        if axis == 0:
            local_axis = Vec3.unit_x()
        elif axis == 1:
            local_axis = Vec3.unit_y()
        else:
            local_axis = Vec3.unit_z()

        # Transform to world space
        var world_axis = orientation.rotate_vec(local_axis)

        # End points of capsule
        var top = center + world_axis * half_height
        var bottom = center - world_axis * half_height

        # Project endpoints to screen
        var s_top = self.camera.project_to_screen(top)
        var s_bottom = self.camera.project_to_screen(bottom)
        var s_center = self.camera.project_to_screen(center)

        if not (s_top[2] and s_bottom[2]):
            return

        # Calculate screen-space radius
        var view_center = self.camera.get_view_matrix().transform_point(center)
        var depth = -view_center.z
        if depth <= 0.1:
            return

        var fov_scale = 1.0 / (depth * 0.7)
        var screen_radius = Int(radius * Float64(self.height) * fov_scale)
        screen_radius = max(screen_radius, 3)

        # Direction vector on screen
        var dx = Float64(s_top[0] - s_bottom[0])
        var dy = Float64(s_top[1] - s_bottom[1])
        var length = sqrt(dx * dx + dy * dy)

        if length < 1.0:
            # Too small, draw as shaded sphere
            self.draw_shaded_sphere_2d(s_center[0], s_center[1], screen_radius, color)
            return

        # Perpendicular direction for width (this is "up" relative to capsule)
        var perp_x = -dy / length
        var perp_y = dx / length

        # Draw multiple layers for gradient shading
        # From outside (dark) to center (bright)
        var num_layers = min(screen_radius, 15)

        for layer in range(num_layers, -1, -1):
            var t = Float64(layer) / Float64(num_layers)  # 1.0 at edge, 0.0 at center
            var layer_radius = Int(Float64(screen_radius) * t)
            if layer_radius < 1:
                layer_radius = 1

            # Calculate brightness: dark at edge, bright toward center
            var brightness = 0.4 + 0.7 * (1.0 - t)

            var r = Int(Float64(color.r) * brightness)
            var g = Int(Float64(color.g) * brightness)
            var b = Int(Float64(color.b) * brightness)
            r = min(r, 255)
            g = min(g, 255)
            b = min(b, 255)

            self.sdl.set_draw_color(UInt8(r), UInt8(g), UInt8(b))

            # Draw the cylindrical body as a quad
            var offset_x = perp_x * Float64(layer_radius)
            var offset_y = perp_y * Float64(layer_radius)

            var points = List[SDL_Point]()
            points.append(SDL_Point(
                Int32(s_top[0] + Int(offset_x)),
                Int32(s_top[1] + Int(offset_y))
            ))
            points.append(SDL_Point(
                Int32(s_top[0] - Int(offset_x)),
                Int32(s_top[1] - Int(offset_y))
            ))
            points.append(SDL_Point(
                Int32(s_bottom[0] - Int(offset_x)),
                Int32(s_bottom[1] - Int(offset_y))
            ))
            points.append(SDL_Point(
                Int32(s_bottom[0] + Int(offset_x)),
                Int32(s_bottom[1] + Int(offset_y))
            ))

            self.sdl.fill_polygon(points)

            # Draw caps at this layer
            self.sdl.fill_circle(s_top[0], s_top[1], layer_radius)
            self.sdl.fill_circle(s_bottom[0], s_bottom[1], layer_radius)

        # Add highlight strip along the cylinder
        var highlight_offset = screen_radius // 3
        var hr = min(Int(Float64(color.r) * 0.3 + 180), 255)
        var hg = min(Int(Float64(color.g) * 0.3 + 180), 255)
        var hb = min(Int(Float64(color.b) * 0.3 + 180), 255)
        self.sdl.set_draw_color(UInt8(hr), UInt8(hg), UInt8(hb))

        # Draw highlight line along one edge
        var hl_x = perp_x * Float64(highlight_offset)
        var hl_y = perp_y * Float64(highlight_offset)
        self.sdl.draw_line(
            s_top[0] - Int(hl_x), s_top[1] - Int(hl_y),
            s_bottom[0] - Int(hl_x), s_bottom[1] - Int(hl_y),
        )

        # Add highlight spots on caps
        var cap_highlight_radius = max(screen_radius // 5, 2)
        self.sdl.fill_circle(
            s_top[0] - Int(hl_x), s_top[1] - Int(hl_y), cap_highlight_radius
        )
        self.sdl.fill_circle(
            s_bottom[0] - Int(hl_x), s_bottom[1] - Int(hl_y), cap_highlight_radius
        )
