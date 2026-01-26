"""3D Wireframe Renderer.

Software wireframe renderer using SDL2 for display.
Projects 3D shapes to 2D and draws them as line segments.
"""

from math3d import Vec3, Quat
from render.sdl2 import SDL2, SDL_Event, SDL_QUIT
from .camera3d import Camera3D
from .shapes3d import (
    WireframeLine,
    WireframeSphere,
    WireframeCapsule,
    WireframeBox,
    create_ground_grid,
    create_axes,
)


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
            fov=camera.fov * 180.0 / 3.14159265358979,  # Convert back to degrees
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
        var capsule = WireframeCapsule(center, orientation, radius, half_height, axis, segments)
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
