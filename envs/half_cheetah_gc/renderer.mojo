"""HalfCheetahGC Renderer using the render3d wireframe renderer.

Provides visualization of the HalfCheetahGC environment with:
- Capsule bodies for each body segment (8 total: torso, head, and 6 leg segments)
- Color-coded body parts (torso, head, back leg, front leg)
- Ground plane and coordinate axes
- Orbital camera control for interactive viewing

Implements EnvRenderer3D trait for integration with evaluation code.
"""

from math import cos, sin, pi
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render3d import Renderer3D, Camera3D, Color3D
from core import EnvRenderer3D

from .constants_gc import (
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    BODY_HEAD,
    NBODY,
    CAPSULE_RADIUS,
    TORSO_HALF_LENGTH,
    HEAD_HALF_LENGTH,
    HEAD_POS_X,
    HEAD_POS_Y,
    HEAD_POS_Z,
    HEAD_AXIS_ANGLE,
    BTHIGH_HALF_LENGTH,
    BSHIN_HALF_LENGTH,
    BFOOT_HALF_LENGTH,
    FTHIGH_HALF_LENGTH,
    FSHIN_HALF_LENGTH,
    FFOOT_HALF_LENGTH,
)

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# Color Scheme for HalfCheetahGC
# =============================================================================


struct HalfCheetahGCColors:
    """Color scheme for HalfCheetahGC visualization."""

    @staticmethod
    fn torso() -> Color3D:
        """Torso color - tan/brown matching MuJoCo."""
        return Color3D(204, 153, 102)

    @staticmethod
    fn head() -> Color3D:
        """Head color - same as torso."""
        return Color3D(204, 153, 102)

    @staticmethod
    fn back_thigh() -> Color3D:
        """Back thigh color - same as torso."""
        return Color3D(204, 153, 102)

    @staticmethod
    fn back_shin() -> Color3D:
        """Back shin color - slightly reddish (matching MuJoCo)."""
        return Color3D(230, 153, 153)

    @staticmethod
    fn back_foot() -> Color3D:
        """Back foot color - slightly reddish (matching MuJoCo)."""
        return Color3D(230, 153, 153)

    @staticmethod
    fn front_thigh() -> Color3D:
        """Front thigh color - same as torso."""
        return Color3D(204, 153, 102)

    @staticmethod
    fn front_shin() -> Color3D:
        """Front shin color - slightly reddish (matching MuJoCo)."""
        return Color3D(230, 153, 153)

    @staticmethod
    fn front_foot() -> Color3D:
        """Front foot color - slightly reddish (matching MuJoCo)."""
        return Color3D(230, 153, 153)

    @staticmethod
    fn ground() -> Color3D:
        """Ground grid color."""
        return Color3D(60, 80, 60)

    @staticmethod
    fn velocity() -> Color3D:
        """Velocity indicator color - cyan."""
        return Color3D(0, 255, 255)


# =============================================================================
# HalfCheetahGC Renderer
# =============================================================================


struct HalfCheetahGCRenderer(EnvRenderer3D, Movable):
    """Renderer for HalfCheetahGC environment.

    Uses filled capsules to visualize each body segment of the half cheetah.
    Supports interactive camera control for orbit, zoom, and pan.

    Implements EnvRenderer3D trait for integration with evaluation code.
    """

    var renderer: Renderer3D
    var initialized: Bool
    var follow_cheetah: Bool
    var show_velocity: Bool
    var show_shadows: Bool

    # Body dimensions (matching physics)
    var capsule_radius: Float64
    var torso_half_length: Float64
    var head_half_length: Float64
    var bthigh_half_length: Float64
    var bshin_half_length: Float64
    var bfoot_half_length: Float64
    var fthigh_half_length: Float64
    var fshin_half_length: Float64
    var ffoot_half_length: Float64

    # Visual scale factor for capsule radii (physics radii are small for visualization)
    comptime VISUAL_RADIUS_SCALE: Float64 = 2.0

    fn __init__(
        out self,
        width: Int = 1280,
        height: Int = 720,
        follow_cheetah: Bool = True,
        show_velocity: Bool = True,
        show_shadows: Bool = True,
        capsule_radius: Float64 = CAPSULE_RADIUS,
        torso_half_length: Float64 = TORSO_HALF_LENGTH,
        head_half_length: Float64 = HEAD_HALF_LENGTH,
        bthigh_half_length: Float64 = BTHIGH_HALF_LENGTH,
        bshin_half_length: Float64 = BSHIN_HALF_LENGTH,
        bfoot_half_length: Float64 = BFOOT_HALF_LENGTH,
        fthigh_half_length: Float64 = FTHIGH_HALF_LENGTH,
        fshin_half_length: Float64 = FSHIN_HALF_LENGTH,
        ffoot_half_length: Float64 = FFOOT_HALF_LENGTH,
    ) raises:
        """Initialize the HalfCheetahGC renderer.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            follow_cheetah: Whether camera follows the cheetah's x position.
            show_velocity: Whether to show velocity indicator.
            show_shadows: Whether to show shadows on ground.
            capsule_radius: Radius for all body capsules.
            torso_half_length: Torso capsule half-length.
            head_half_length: Head capsule half-length.
            bthigh_half_length: Back thigh capsule half-length.
            bshin_half_length: Back shin capsule half-length.
            bfoot_half_length: Back foot capsule half-length.
            fthigh_half_length: Front thigh capsule half-length.
            fshin_half_length: Front shin capsule half-length.
            ffoot_half_length: Front foot capsule half-length.
        """
        # Store body dimensions
        self.capsule_radius = capsule_radius
        self.torso_half_length = torso_half_length
        self.head_half_length = head_half_length
        self.bthigh_half_length = bthigh_half_length
        self.bshin_half_length = bshin_half_length
        self.bfoot_half_length = bfoot_half_length
        self.fthigh_half_length = fthigh_half_length
        self.fshin_half_length = fshin_half_length
        self.ffoot_half_length = ffoot_half_length

        # Camera setup for side view
        var camera = Camera3D(
            eye=Vec3(0.0, -3.0, 1.0),  # Side view with slight elevation
            target=Vec3(0.0, 0.0, 0.5),  # Look at cheetah center
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
        self.follow_cheetah = follow_cheetah
        self.show_velocity = show_velocity
        self.show_shadows = show_shadows

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor - transfers ownership of renderer."""
        self.renderer = other.renderer^
        self.initialized = other.initialized
        self.follow_cheetah = other.follow_cheetah
        self.show_velocity = other.show_velocity
        self.show_shadows = other.show_shadows
        self.capsule_radius = other.capsule_radius
        self.torso_half_length = other.torso_half_length
        self.head_half_length = other.head_half_length
        self.bthigh_half_length = other.bthigh_half_length
        self.bshin_half_length = other.bshin_half_length
        self.bfoot_half_length = other.bfoot_half_length
        self.fthigh_half_length = other.fthigh_half_length
        self.fshin_half_length = other.fshin_half_length
        self.ffoot_half_length = other.ffoot_half_length

    fn init(mut self) raises -> None:
        """Initialize the renderer window."""
        var title = String("HalfCheetahGC Environment")
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
        positions: List[Vec3],
        quaternions: List[Quat],
        vel_x: Float64 = 0.0,
    ):
        """Render the HalfCheetahGC state.

        Args:
            positions: List of 8 body positions (torso, bthigh, bshin, bfoot, fthigh, fshin, ffoot, head).
            quaternions: List of 8 body orientations.
            vel_x: Current forward velocity (for velocity indicator).
        """
        if not self.initialized:
            return

        if len(positions) < NBODY or len(quaternions) < NBODY:
            return

        var torso_pos = positions[BODY_TORSO]

        # Update camera to follow cheetah
        if self.follow_cheetah:
            self.renderer.camera.target = Vec3(torso_pos.x, 0.0, 0.5)
            self.renderer.camera.eye = Vec3(torso_pos.x, -3.0, 1.0)

        # Begin frame
        self.renderer.begin_frame()

        # Draw ground grid (centered on cheetah if following)
        var grid_center_x = torso_pos.x if self.follow_cheetah else 0.0
        self._draw_ground_grid(grid_center_x)

        # Draw coordinate axes
        if self.follow_cheetah:
            self.renderer.draw_coordinate_axes(
                Vec3(torso_pos.x - 1.5, 0.0, 0.0), 0.2
            )
        else:
            self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.0), 0.2)

        # Draw shadows first (under the cheetah)
        if self.show_shadows:
            self._draw_shadows(positions)

        # Draw all body capsules
        self._draw_torso(positions[BODY_TORSO], quaternions[BODY_TORSO])
        self._draw_head(positions[BODY_HEAD], quaternions[BODY_HEAD])
        self._draw_back_thigh(positions[BODY_BTHIGH], quaternions[BODY_BTHIGH])
        self._draw_back_shin(positions[BODY_BSHIN], quaternions[BODY_BSHIN])
        self._draw_back_foot(positions[BODY_BFOOT], quaternions[BODY_BFOOT])
        self._draw_front_thigh(positions[BODY_FTHIGH], quaternions[BODY_FTHIGH])
        self._draw_front_shin(positions[BODY_FSHIN], quaternions[BODY_FSHIN])
        self._draw_front_foot(positions[BODY_FFOOT], quaternions[BODY_FFOOT])

        # Draw velocity indicator
        if self.show_velocity:
            self._draw_velocity_indicator(torso_pos, vel_x)

        # End frame
        self.renderer.end_frame()

    fn _draw_ground_grid(self, center_x: Float64):
        """Draw chessboard ground pattern centered at given x position."""
        var tile_size = 0.5
        var num_tiles_x = 20
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

    fn _draw_shadows(self, positions: List[Vec3]):
        """Draw shadows on the ground for all body parts."""
        var shadow_color = Color3D(30, 30, 30)
        var ground_z = 0.001

        # Torso shadow (elongated ellipse - horizontal along X)
        self._draw_ellipse_shadow(
            positions[BODY_TORSO].x,
            positions[BODY_TORSO].y,
            ground_z,
            self.torso_half_length + 0.02,
            self.capsule_radius * 2,
            shadow_color,
        )

        # Head shadow (using actual head body position)
        self._draw_ellipse_shadow(
            positions[BODY_HEAD].x,
            positions[BODY_HEAD].y,
            ground_z,
            self.head_half_length + 0.01,
            self.capsule_radius * 1.5,
            shadow_color,
        )

        # Back leg shadows
        self._draw_circle_shadow(
            positions[BODY_BTHIGH].x,
            positions[BODY_BTHIGH].y,
            ground_z,
            self.capsule_radius * 1.5,
            shadow_color,
        )
        self._draw_circle_shadow(
            positions[BODY_BSHIN].x,
            positions[BODY_BSHIN].y,
            ground_z,
            self.capsule_radius * 1.5,
            shadow_color,
        )
        self._draw_ellipse_shadow(
            positions[BODY_BFOOT].x,
            positions[BODY_BFOOT].y,
            ground_z,
            self.bfoot_half_length + 0.01,
            self.capsule_radius * 2,
            shadow_color,
        )

        # Front leg shadows
        self._draw_circle_shadow(
            positions[BODY_FTHIGH].x,
            positions[BODY_FTHIGH].y,
            ground_z,
            self.capsule_radius * 1.5,
            shadow_color,
        )
        self._draw_circle_shadow(
            positions[BODY_FSHIN].x,
            positions[BODY_FSHIN].y,
            ground_z,
            self.capsule_radius * 1.5,
            shadow_color,
        )
        self._draw_ellipse_shadow(
            positions[BODY_FFOOT].x,
            positions[BODY_FFOOT].y,
            ground_z,
            self.ffoot_half_length + 0.01,
            self.capsule_radius * 2,
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
        """Draw the torso capsule (horizontal along X-axis)."""
        # Torso is horizontal, so we draw along Z-axis and let the quat rotate it
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.torso_half_length,
            axis=2,  # Z-axis, will be rotated by quat to lie along X
            color=HalfCheetahGCColors.torso(),
        )

    fn _draw_head(self, pos: Vec3, quat: Quat):
        """Draw the head capsule using its physics position and orientation."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.head_half_length,
            axis=2,  # Z-axis, will be rotated by quat
            color=HalfCheetahGCColors.head(),
        )

    fn _draw_back_thigh(self, pos: Vec3, quat: Quat):
        """Draw the back thigh capsule (vertical)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.bthigh_half_length,
            axis=2,  # Z-axis (vertical)
            color=HalfCheetahGCColors.back_thigh(),
        )

    fn _draw_back_shin(self, pos: Vec3, quat: Quat):
        """Draw the back shin capsule (vertical)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.bshin_half_length,
            axis=2,  # Z-axis (vertical)
            color=HalfCheetahGCColors.back_shin(),
        )

    fn _draw_back_foot(self, pos: Vec3, quat: Quat):
        """Draw the back foot capsule (horizontal)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.bfoot_half_length,
            axis=2,  # Z-axis, will be rotated by quat to lie along X
            color=HalfCheetahGCColors.back_foot(),
        )

    fn _draw_front_thigh(self, pos: Vec3, quat: Quat):
        """Draw the front thigh capsule (vertical)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.fthigh_half_length,
            axis=2,  # Z-axis (vertical)
            color=HalfCheetahGCColors.front_thigh(),
        )

    fn _draw_front_shin(self, pos: Vec3, quat: Quat):
        """Draw the front shin capsule (vertical)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.fshin_half_length,
            axis=2,  # Z-axis (vertical)
            color=HalfCheetahGCColors.front_shin(),
        )

    fn _draw_front_foot(self, pos: Vec3, quat: Quat):
        """Draw the front foot capsule (horizontal)."""
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.ffoot_half_length,
            axis=2,  # Z-axis, will be rotated by quat to lie along X
            color=HalfCheetahGCColors.front_foot(),
        )

    fn _draw_velocity_indicator(self, torso_pos: Vec3, vel_x: Float64):
        """Draw a velocity indicator arrow above the torso."""
        var arrow_start = Vec3(torso_pos.x, torso_pos.y, torso_pos.z + 0.15)
        var arrow_length = vel_x * 0.1  # Scale velocity for display
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

        self.renderer.draw_lines_3d(lines, HalfCheetahGCColors.velocity())

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
