"""Physics3D v2 Renderer using the render3d wireframe renderer.

Provides visualization of physics3d_v2 simulations with:
- Shaded sphere rendering for sphere geometries
- Ground plane with chessboard pattern
- Coordinate axes
- Velocity indicator
- Interactive camera control

Example:
    from physics3d_v2 import Model, Data, Body, Geom, step
    from physics3d_v2.render import Physics3DRenderer

    # Setup physics
    var body = Body.create_sphere(mass=1.0)
    var geom = Geom.sphere(0.1)
    var model = Model.create(body, geom)
    var data = Data()
    data.set_position(0, 0, 1)

    # Setup renderer
    var renderer = Physics3DRenderer()
    renderer.init()

    # Simulation loop
    while not renderer.check_quit():
        step(model, data)
        renderer.render(model, data)
        renderer.delay(10)

    renderer.close()
"""

from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render3d import Renderer3D, Camera3D, Color3D
from .types import Model, Data
from .constants import GEOM_SPHERE, GEOM_PLANE

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# Color Scheme
# =============================================================================


struct Physics3DColors:
    """Color scheme for Physics3D v2 visualization."""

    @staticmethod
    fn sphere() -> Color3D:
        """Sphere color - orange."""
        return Color3D(255, 140, 60)

    @staticmethod
    fn ground_light() -> Color3D:
        """Ground light tile color."""
        return Color3D(140, 140, 120)

    @staticmethod
    fn ground_dark() -> Color3D:
        """Ground dark tile color."""
        return Color3D(80, 80, 70)

    @staticmethod
    fn velocity() -> Color3D:
        """Velocity indicator color - cyan."""
        return Color3D(0, 255, 255)

    @staticmethod
    fn shadow() -> Color3D:
        """Shadow color."""
        return Color3D(30, 30, 30)

    @staticmethod
    fn contact() -> Color3D:
        """Contact point indicator - red."""
        return Color3D(255, 50, 50)


# =============================================================================
# Physics3D Renderer
# =============================================================================


struct Physics3DRenderer(Movable):
    """Renderer for Physics3D v2 simulations.

    Uses render3d for 3D visualization with shaded spheres,
    ground plane, and velocity indicators.
    """

    var renderer: Renderer3D
    var initialized: Bool
    var show_velocity: Bool
    var show_shadows: Bool
    var show_contact: Bool
    var follow_object: Bool

    fn __init__(
        out self,
        width: Int = 800,
        height: Int = 600,
        show_velocity: Bool = True,
        show_shadows: Bool = True,
        show_contact: Bool = True,
        follow_object: Bool = False,
    ) raises:
        """Initialize the Physics3D v2 renderer.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            show_velocity: Whether to show velocity indicator.
            show_shadows: Whether to show shadows.
            show_contact: Whether to show contact points.
            follow_object: Whether camera follows the object.
        """
        # Camera setup - isometric-ish view from the side
        var camera = Camera3D(
            eye=Vec3(2.0, -3.0, 2.0),  # View from side and above
            target=Vec3(0.0, 0.0, 0.5),  # Look at center above ground
            up=Vec3(0.0, 0.0, 1.0),  # Z-up
            fov=50.0,
            aspect=Float64(width) / Float64(height),
            near=0.1,
            far=100.0,
            screen_width=width,
            screen_height=height,
        )

        self.renderer = Renderer3D(
            width=width,
            height=height,
            camera=camera,
            draw_grid=False,  # We'll draw our own ground
            draw_axes=True,
        )
        self.initialized = False
        self.show_velocity = show_velocity
        self.show_shadows = show_shadows
        self.show_contact = show_contact
        self.follow_object = follow_object

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.renderer = other.renderer^
        self.initialized = other.initialized
        self.show_velocity = other.show_velocity
        self.show_shadows = other.show_shadows
        self.show_contact = other.show_contact
        self.follow_object = other.follow_object

    fn init(mut self) raises -> None:
        """Initialize the renderer window."""
        var title = String("Physics3D v2")
        self.renderer.init(title)
        self.initialized = True

    fn close(mut self) raises -> None:
        """Close the renderer."""
        if self.initialized:
            self.renderer.close()
            self.initialized = False

    fn check_quit(mut self) -> Bool:
        """Check if user wants to quit.

        Returns:
            True if quit event detected.
        """
        return self.renderer.check_quit()

    fn render(mut self, model: Model, data: Data):
        """Render the physics state.

        Args:
            model: Physics model configuration.
            data: Current physics state.
        """
        if not self.initialized:
            return

        # Get object position
        var pos = Vec3(
            Float64(data.xpos_x),
            Float64(data.xpos_y),
            Float64(data.xpos_z),
        )

        # Update camera to follow object if enabled
        if self.follow_object:
            self.renderer.camera.target = pos
            self.renderer.camera.eye = Vec3(
                pos.x + 2.0, pos.y - 3.0, pos.z + 1.5
            )

        # Begin frame
        self.renderer.begin_frame()

        # Draw ground (chessboard pattern)
        self._draw_ground(Float64(model.ground_z))

        # Draw coordinate axes
        self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.01), 0.5)

        # Draw shadow if enabled
        if self.show_shadows:
            self._draw_shadow(model, data)

        # Draw the geometry
        if model.geom.type == GEOM_SPHERE:
            self._draw_sphere(model, data)

        # Draw contact point if active and enabled
        if self.show_contact and data.contact.active:
            self._draw_contact(data)

        # Draw velocity indicator if enabled
        if self.show_velocity:
            self._draw_velocity(data)

        # End frame
        self.renderer.end_frame()

    fn _draw_ground(self, ground_z: Float64):
        """Draw chessboard ground pattern."""
        var tile_size = 0.5
        var num_tiles = 10  # 5 tiles in each direction

        for i in range(-num_tiles, num_tiles):
            for j in range(-num_tiles, num_tiles):
                var x0 = Float64(i) * tile_size
                var y0 = Float64(j) * tile_size
                var x1 = x0 + tile_size
                var y1 = y0 + tile_size

                # Chessboard pattern
                var is_light = (i + j) % 2 == 0

                var color = (
                    Physics3DColors.ground_light() if is_light else Physics3DColors.ground_dark()
                )

                self.renderer.draw_filled_quad_3d(
                    Vec3(x0, y0, ground_z),
                    Vec3(x1, y0, ground_z),
                    Vec3(x1, y1, ground_z),
                    Vec3(x0, y1, ground_z),
                    color,
                )

    fn _draw_shadow(self, model: Model, data: Data):
        """Draw shadow on ground for the object."""
        if model.geom.type != GEOM_SPHERE:
            return

        var shadow_z = Float64(model.ground_z) + 0.001  # Slightly above ground
        var pos_x = Float64(data.xpos_x)
        var pos_y = Float64(data.xpos_y)
        var radius = Float64(model.geom.size) * 1.2  # Shadow slightly larger

        # Draw shadow as circle segments
        var num_segments = 16
        from math import cos, sin, pi

        for i in range(num_segments):
            var angle0 = 2.0 * pi * Float64(i) / Float64(num_segments)
            var angle1 = 2.0 * pi * Float64(i + 1) / Float64(num_segments)

            var x0 = pos_x + radius * cos(angle0)
            var y0 = pos_y + radius * sin(angle0)
            var x1 = pos_x + radius * cos(angle1)
            var y1 = pos_y + radius * sin(angle1)

            self.renderer.draw_filled_quad_3d(
                Vec3(pos_x, pos_y, shadow_z),
                Vec3(x0, y0, shadow_z),
                Vec3(x1, y1, shadow_z),
                Vec3(pos_x, pos_y, shadow_z),
                Physics3DColors.shadow(),
            )

    fn _draw_sphere(self, model: Model, data: Data):
        """Draw the sphere geometry with shading."""
        var pos = Vec3(
            Float64(data.xpos_x),
            Float64(data.xpos_y),
            Float64(data.xpos_z),
        )
        var radius = Float64(model.geom.size)

        # Sphere has no visible orientation, so quaternion not needed
        # (keeping comment for future capsule/box support)

        # Project sphere center to get screen coordinates
        var screen_pos = self.renderer.camera.project_to_screen(pos)

        if screen_pos[2]:  # Visible
            # Calculate screen radius based on depth
            var view_pos = (
                self.renderer.camera.get_view_matrix().transform_point(pos)
            )
            var depth = -view_pos.z
            if depth > 0.1:
                var fov_scale = 1.0 / (depth * 0.7)
                var screen_radius = Int(
                    radius * Float64(self.renderer.height) * fov_scale
                )
                screen_radius = max(screen_radius, 5)

                self.renderer.draw_shaded_sphere_2d(
                    screen_pos[0],
                    screen_pos[1],
                    screen_radius,
                    Physics3DColors.sphere(),
                )

    fn _draw_contact(self, data: Data):
        """Draw contact point indicator."""
        var contact_pos = Vec3(
            Float64(data.contact.pos_x),
            Float64(data.contact.pos_y),
            Float64(data.contact.pos_z) + 0.01,  # Slightly above ground
        )

        # Draw small marker at contact point
        var screen_pos = self.renderer.camera.project_to_screen(contact_pos)
        if screen_pos[2]:
            self.renderer.draw_filled_circle_2d(
                screen_pos[0],
                screen_pos[1],
                5,
                Physics3DColors.contact(),
            )

    fn _draw_velocity(self, data: Data):
        """Draw velocity indicator arrow."""
        var pos = Vec3(
            Float64(data.xpos_x),
            Float64(data.xpos_y),
            Float64(data.xpos_z),
        )

        var vel = Vec3(
            Float64(data.qvel[0]),
            Float64(data.qvel[1]),
            Float64(data.qvel[2]),
        )

        # Scale velocity for display
        var scale = 0.1
        var arrow_end = pos + vel * scale

        # Draw velocity arrow
        from render3d.shapes3d import WireframeLine

        var lines = List[WireframeLine]()
        lines.append(WireframeLine(pos, arrow_end))

        # Add arrowhead if velocity is significant
        var vel_mag = (
            Float64(data.qvel[0]) ** 2
            + Float64(data.qvel[1]) ** 2
            + Float64(data.qvel[2]) ** 2
        )
        if vel_mag > 0.1:
            var head_size = 0.03
            # Simple arrowhead pointing in z direction
            if Float64(data.qvel[2]) != 0:
                var direction = 1.0 if Float64(data.qvel[2]) > 0 else -1.0
                lines.append(
                    WireframeLine(
                        arrow_end,
                        Vec3(
                            arrow_end.x + head_size,
                            arrow_end.y,
                            arrow_end.z - head_size * direction,
                        ),
                    )
                )
                lines.append(
                    WireframeLine(
                        arrow_end,
                        Vec3(
                            arrow_end.x - head_size,
                            arrow_end.y,
                            arrow_end.z - head_size * direction,
                        ),
                    )
                )

        self.renderer.draw_lines_3d(lines, Physics3DColors.velocity())

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64) -> None:
        """Orbit camera around target."""
        self.renderer.orbit_camera(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64) -> None:
        """Zoom camera in/out."""
        self.renderer.zoom_camera(delta)

    fn delay(self, ms: Int) -> None:
        """Delay for given milliseconds."""
        self.renderer.delay(ms)

    fn is_open(self) -> Bool:
        """Check if renderer window is still open."""
        return self.initialized
