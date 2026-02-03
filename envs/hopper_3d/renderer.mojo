"""Hopper3D Renderer using the render3d wireframe renderer.

Provides visualization of the Hopper3D environment with:
- Capsule bodies for each body segment
- Color-coded body parts (torso, thigh, leg, foot)
- Ground plane and coordinate axes
- Orbital camera control for interactive viewing

Implements EnvRenderer3D trait for integration with evaluation code.
"""

from math import cos, sin, pi
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render3d import Renderer3D, Camera3D, Color3D
from core import EnvRenderer3D

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# Color Scheme for Hopper3D
# =============================================================================


struct HopperColors:
    """Color scheme for Hopper3D visualization."""

    @staticmethod
    fn torso() -> Color3D:
        """Torso color - blue."""
        return Color3D(60, 120, 200)

    @staticmethod
    fn thigh() -> Color3D:
        """Thigh color - green."""
        return Color3D(80, 200, 80)

    @staticmethod
    fn leg() -> Color3D:
        """Leg color - orange."""
        return Color3D(220, 140, 60)

    @staticmethod
    fn foot() -> Color3D:
        """Foot color - red."""
        return Color3D(220, 80, 80)

    @staticmethod
    fn ground() -> Color3D:
        """Ground grid color."""
        return Color3D(60, 80, 60)

    @staticmethod
    fn velocity() -> Color3D:
        """Velocity indicator color - cyan."""
        return Color3D(0, 255, 255)


# =============================================================================
# Hopper3D Renderer
# =============================================================================


struct Hopper3DRenderer(EnvRenderer3D, Movable):
    """Renderer for Hopper3D environment.

    Uses filled capsules to visualize each body segment of the hopper.
    Supports interactive camera control for orbit, zoom, and pan.

    Implements EnvRenderer3D trait for integration with evaluation code.
    """

    var renderer: Renderer3D
    var initialized: Bool
    var follow_hopper: Bool
    var show_velocity: Bool
    var show_shadows: Bool

    # Body dimensions (matching physics - set in render call or use defaults)
    var torso_radius: Float64
    var torso_half_length: Float64
    var thigh_radius: Float64
    var thigh_half_length: Float64
    var leg_radius: Float64
    var leg_half_length: Float64
    var foot_radius: Float64
    var foot_half_length: Float64

    # Visual scale factor for capsule radii (physics radii are small for visualization)
    comptime VISUAL_RADIUS_SCALE: Float64 = 1.5

    fn __init__(
        out self,
        width: Int = 1024,
        height: Int = 576,
        follow_hopper: Bool = True,
        show_velocity: Bool = True,
        show_shadows: Bool = True,
        # Body dimensions (matching MuJoCo Hopper defaults)
        torso_radius: Float64 = 0.05,
        torso_half_length: Float64 = 0.2,
        thigh_radius: Float64 = 0.05,
        thigh_half_length: Float64 = 0.225,
        leg_radius: Float64 = 0.04,
        leg_half_length: Float64 = 0.25,
        foot_radius: Float64 = 0.06,
        foot_half_length: Float64 = 0.195,
    ) raises:
        """Initialize the Hopper3D renderer.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            follow_hopper: Whether camera follows the hopper's x position.
            show_velocity: Whether to show velocity indicator.
            show_shadows: Whether to show shadows on ground.
            torso_radius: Torso capsule radius.
            torso_half_length: Torso capsule half-length.
            thigh_radius: Thigh capsule radius.
            thigh_half_length: Thigh capsule half-length.
            leg_radius: Leg capsule radius.
            leg_half_length: Leg capsule half-length.
            foot_radius: Foot capsule radius.
            foot_half_length: Foot capsule half-length.
        """
        # Store body dimensions
        self.torso_radius = torso_radius
        self.torso_half_length = torso_half_length
        self.thigh_radius = thigh_radius
        self.thigh_half_length = thigh_half_length
        self.leg_radius = leg_radius
        self.leg_half_length = leg_half_length
        self.foot_radius = foot_radius
        self.foot_half_length = foot_half_length

        # Camera setup for side view
        var camera = Camera3D(
            eye=Vec3(0.0, -2.5, 1.2),  # Side view with slight elevation
            target=Vec3(0.0, 0.0, 0.8),  # Look at hopper center
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
            draw_grid=True,
            draw_axes=True,
        )
        self.initialized = False
        self.follow_hopper = follow_hopper
        self.show_velocity = show_velocity
        self.show_shadows = show_shadows

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor - transfers ownership of renderer."""
        self.renderer = other.renderer^
        self.initialized = other.initialized
        self.follow_hopper = other.follow_hopper
        self.show_velocity = other.show_velocity
        self.show_shadows = other.show_shadows
        self.torso_radius = other.torso_radius
        self.torso_half_length = other.torso_half_length
        self.thigh_radius = other.thigh_radius
        self.thigh_half_length = other.thigh_half_length
        self.leg_radius = other.leg_radius
        self.leg_half_length = other.leg_half_length
        self.foot_radius = other.foot_radius
        self.foot_half_length = other.foot_half_length

    fn init(mut self) raises -> None:
        """Initialize the renderer window."""
        var title = String("Hopper3D Environment")
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

    fn render(
        mut self,
        torso_pos: Vec3,
        torso_quat: Quat,
        thigh_pos: Vec3,
        thigh_quat: Quat,
        leg_pos: Vec3,
        leg_quat: Quat,
        foot_pos: Vec3,
        foot_quat: Quat,
        vel_x: Float64 = 0.0,
    ):
        """Render the Hopper3D state.

        Args:
            torso_pos: Torso position.
            torso_quat: Torso orientation.
            thigh_pos: Thigh position.
            thigh_quat: Thigh orientation.
            leg_pos: Leg position.
            leg_quat: Leg orientation.
            foot_pos: Foot position.
            foot_quat: Foot orientation.
            vel_x: Current forward velocity (for velocity indicator).
        """
        if not self.initialized:
            return

        # Update camera to follow hopper
        if self.follow_hopper:
            self.renderer.camera.target = Vec3(torso_pos.x, 0.0, 0.8)
            self.renderer.camera.eye = Vec3(torso_pos.x, -2.5, 1.2)

        # Begin frame
        self.renderer.begin_frame()

        # Draw ground grid (centered on hopper if following)
        var grid_center_x = torso_pos.x if self.follow_hopper else 0.0
        self._draw_ground_grid(grid_center_x)

        # Draw coordinate axes
        if self.follow_hopper:
            self.renderer.draw_coordinate_axes(
                Vec3(torso_pos.x - 0.8, 0.0, 0.0), 0.2
            )
        else:
            self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.0), 0.2)

        # Draw shadows first (under the hopper)
        if self.show_shadows:
            self._draw_shadows(torso_pos, thigh_pos, leg_pos, foot_pos)

        # Draw all body capsules
        self._draw_torso(torso_pos, torso_quat)
        self._draw_thigh(thigh_pos, thigh_quat)
        self._draw_leg(leg_pos, leg_quat)
        self._draw_foot(foot_pos, foot_quat)

        # Draw velocity indicator
        if self.show_velocity:
            self._draw_velocity_indicator(torso_pos, vel_x)

        # End frame
        self.renderer.end_frame()

    fn _draw_ground_grid(self, center_x: Float64):
        """Draw chessboard ground pattern centered at given x position."""
        var tile_size = 0.4
        var num_tiles_x = 16
        var num_tiles_y = 8

        var tile_center_x = Float64(Int(center_x / tile_size)) * tile_size

        for i in range(-num_tiles_x // 2, num_tiles_x // 2):
            for j in range(-num_tiles_y // 2, num_tiles_y // 2):
                var x0 = tile_center_x + Float64(i) * tile_size
                var y0 = Float64(j) * tile_size
                var x1 = x0 + tile_size
                var y1 = y0 + tile_size

                var is_light = (i + j) % 2 == 0

                if is_light:
                    self.renderer.draw_filled_quad_3d(
                        Vec3(x0, y0, 0.0),
                        Vec3(x1, y0, 0.0),
                        Vec3(x1, y1, 0.0),
                        Vec3(x0, y1, 0.0),
                        Color3D(140, 140, 120),
                    )
                else:
                    self.renderer.draw_filled_quad_3d(
                        Vec3(x0, y0, 0.0),
                        Vec3(x1, y0, 0.0),
                        Vec3(x1, y1, 0.0),
                        Vec3(x0, y1, 0.0),
                        Color3D(80, 80, 70),
                    )

    fn _draw_shadows(
        self,
        torso_pos: Vec3,
        thigh_pos: Vec3,
        leg_pos: Vec3,
        foot_pos: Vec3,
    ):
        """Draw shadows on the ground for all body parts."""
        var shadow_color = Color3D(30, 30, 30)
        var ground_z = 0.001

        # Torso shadow (elongated ellipse)
        self._draw_ellipse_shadow(
            torso_pos.x,
            torso_pos.y,
            ground_z,
            self.torso_half_length + 0.02,
            self.torso_radius * 2,
            shadow_color,
        )

        # Thigh shadow
        self._draw_circle_shadow(
            thigh_pos.x, thigh_pos.y, ground_z, self.thigh_radius * 1.5, shadow_color
        )

        # Leg shadow
        self._draw_circle_shadow(
            leg_pos.x, leg_pos.y, ground_z, self.leg_radius * 1.5, shadow_color
        )

        # Foot shadow (elongated)
        self._draw_ellipse_shadow(
            foot_pos.x,
            foot_pos.y,
            ground_z,
            self.foot_half_length + 0.02,
            self.foot_radius * 2,
            shadow_color,
        )

    fn _draw_circle_shadow(
        self,
        x: Float64,
        y: Float64,
        z: Float64,
        radius: Float64,
        color: Color3D,
    ):
        """Draw a circular shadow on the ground."""
        var num_segments = 12

        for i in range(num_segments):
            var angle0 = 2.0 * pi * Float64(i) / Float64(num_segments)
            var angle1 = 2.0 * pi * Float64(i + 1) / Float64(num_segments)

            var x0 = x + radius * cos(angle0)
            var y0 = y + radius * sin(angle0)
            var x1 = x + radius * cos(angle1)
            var y1 = y + radius * sin(angle1)

            self.renderer.draw_filled_quad_3d(
                Vec3(x, y, z),
                Vec3(x0, y0, z),
                Vec3(x1, y1, z),
                Vec3(x, y, z),
                color,
            )

    fn _draw_ellipse_shadow(
        self,
        x: Float64,
        y: Float64,
        z: Float64,
        radius_x: Float64,
        radius_y: Float64,
        color: Color3D,
    ):
        """Draw an elliptical shadow on the ground."""
        var num_segments = 16

        for i in range(num_segments):
            var angle0 = 2.0 * pi * Float64(i) / Float64(num_segments)
            var angle1 = 2.0 * pi * Float64(i + 1) / Float64(num_segments)

            var x0 = x + radius_x * cos(angle0)
            var y0 = y + radius_y * sin(angle0)
            var x1 = x + radius_x * cos(angle1)
            var y1 = y + radius_y * sin(angle1)

            self.renderer.draw_filled_quad_3d(
                Vec3(x, y, z),
                Vec3(x0, y0, z),
                Vec3(x1, y1, z),
                Vec3(x, y, z),
                color,
            )

    fn _draw_torso(self, pos: Vec3, quat: Quat):
        """Draw the torso capsule (vertical)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.torso_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.torso_half_length,
            axis=2,  # Z-axis (vertical)
            color=HopperColors.torso(),
        )

    fn _draw_thigh(self, pos: Vec3, quat: Quat):
        """Draw the thigh capsule (vertical)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.thigh_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.thigh_half_length,
            axis=2,  # Z-axis (vertical)
            color=HopperColors.thigh(),
        )

    fn _draw_leg(self, pos: Vec3, quat: Quat):
        """Draw the leg capsule (vertical)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.leg_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.leg_half_length,
            axis=2,  # Z-axis (vertical)
            color=HopperColors.leg(),
        )

    fn _draw_foot(self, pos: Vec3, quat: Quat):
        """Draw the foot capsule (horizontal, rotated 90° around Y)."""
        # The foot is already rotated by the quaternion from physics
        # So we just draw it along Z-axis and let the quat rotate it
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.foot_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.foot_half_length,
            axis=2,  # Z-axis, will be rotated by quat
            color=HopperColors.foot(),
        )

    fn _draw_velocity_indicator(self, torso_pos: Vec3, vel_x: Float64):
        """Draw a velocity indicator arrow above the torso."""
        var arrow_start = Vec3(torso_pos.x, torso_pos.y, torso_pos.z + 0.25)
        var arrow_length = vel_x * 0.15
        var arrow_end = Vec3(
            arrow_start.x + arrow_length, arrow_start.y, arrow_start.z
        )

        from render3d.shapes3d import WireframeLine

        var lines = List[WireframeLine]()
        lines.append(WireframeLine(arrow_start, arrow_end))

        # Add arrowhead if velocity is significant
        if abs(arrow_length) > 0.03:
            var head_size = 0.04
            var direction = 1.0 if arrow_length > 0 else -1.0
            lines.append(
                WireframeLine(
                    arrow_end,
                    Vec3(
                        arrow_end.x - head_size * direction,
                        arrow_end.y,
                        arrow_end.z + head_size,
                    ),
                )
            )
            lines.append(
                WireframeLine(
                    arrow_end,
                    Vec3(
                        arrow_end.x - head_size * direction,
                        arrow_end.y,
                        arrow_end.z - head_size,
                    ),
                )
            )

        self.renderer.draw_lines_3d(lines, HopperColors.velocity())

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64) -> None:
        """Orbit camera around target."""
        self.renderer.orbit_camera(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64) -> None:
        """Zoom camera in/out."""
        self.renderer.zoom_camera(delta)

    fn delay(self, ms: Int) -> None:
        """Delay for given milliseconds."""
        self.renderer.delay(ms)

    # =========================================================================
    # EnvRenderer Trait Implementation
    # =========================================================================

    fn is_open(self) -> Bool:
        """Check if renderer window is still open.

        Returns:
            True if renderer is initialized and window is open.
        """
        return self.initialized
