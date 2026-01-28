"""HalfCheetah3D Renderer using the render3d wireframe renderer.

Provides visualization of the HalfCheetah3D environment with:
- Wireframe capsule bodies for each body segment
- Color-coded body parts (torso, legs)
- Ground plane and coordinate axes
- Orbital camera control for interactive viewing
"""

from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render3d import Renderer3D, Camera3D, Color3D

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]

from physics3d import (
    dtype,
    BODY_STATE_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
)

from .constants3d import HC3DConstantsCPU


# =============================================================================
# Color Scheme for HalfCheetah3D
# =============================================================================


struct CheetahColors:
    """Color scheme for HalfCheetah3D visualization."""

    @staticmethod
    fn torso() -> Color3D:
        """Torso color - orange-brown."""
        return Color3D(210, 140, 60)

    @staticmethod
    fn back_leg() -> Color3D:
        """Back leg color - red/orange for visibility."""
        return Color3D(255, 80, 80)

    @staticmethod
    fn front_leg() -> Color3D:
        """Front leg color - green for visibility."""
        return Color3D(80, 255, 80)

    @staticmethod
    fn joint() -> Color3D:
        """Joint marker color - white."""
        return Color3D(255, 255, 255)

    @staticmethod
    fn ground() -> Color3D:
        """Ground grid color."""
        return Color3D(60, 80, 60)

    @staticmethod
    fn velocity() -> Color3D:
        """Velocity indicator color - cyan."""
        return Color3D(0, 255, 255)


# =============================================================================
# HalfCheetah3D Renderer
# =============================================================================


struct HalfCheetah3DRenderer:
    """Renderer for HalfCheetah3D environment.

    Uses filled capsules to visualize each body segment of the cheetah.
    Supports interactive camera control for orbit, zoom, and pan.
    """

    var renderer: Renderer3D
    var initialized: Bool
    var follow_cheetah: Bool
    var show_velocity: Bool
    var show_shadows: Bool
    # Visual scale factor for capsule radii (physics radii are small)
    comptime VISUAL_RADIUS_SCALE: Float64 = 1.4
    # Visual scale for capsule lengths (to make them more visible)
    comptime VISUAL_LENGTH_SCALE: Float64 = 1.3
    # Shadow parameters
    comptime SHADOW_ALPHA: Float64 = 0.4  # Shadow darkness (0-1)

    fn __init__(
        out self,
        width: Int = 1024,
        height: Int = 576,
        follow_cheetah: Bool = True,
        show_velocity: Bool = True,
        show_shadows: Bool = True,
    ) raises:
        """Initialize the HalfCheetah3D renderer.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            follow_cheetah: Whether camera follows the cheetah's x position.
            show_velocity: Whether to show velocity indicator.
        """
        # Camera setup for slightly angled side view
        # Position camera at an angle to see 3D depth properly
        var camera = Camera3D(
            eye=Vec3(0.0, -3.0, 2.0),  # Side view with slight elevation
            target=Vec3(0.0, 0.0, 0.4),  # Look at approximate cheetah center
            up=Vec3(0.0, 0.0, 1.0),  # Z-up
            fov=50.0,  # Slightly wider FOV
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

    fn init(mut self):
        """Initialize the renderer window."""
        var title = String("HalfCheetah3D Environment")
        self.renderer.init(title)
        self.initialized = True

    fn close(mut self):
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
        state: List[Scalar[dtype]],
        torso_x: Float64 = 0.0,
        vel_x: Float64 = 0.0,
    ):
        """Render the HalfCheetah3D state.

        Args:
            state: Physics state buffer containing body positions and orientations.
            torso_x: Current torso x position (for camera following).
            vel_x: Current forward velocity (for velocity indicator).
        """
        if not self.initialized:
            return

        # Update camera to follow cheetah
        if self.follow_cheetah:
            self.renderer.camera.target = Vec3(torso_x, 0.0, 0.4)
            self.renderer.camera.eye = Vec3(torso_x, -3.0, 2.0)

        # Begin frame
        self.renderer.begin_frame()

        # Draw ground grid (centered on cheetah if following)
        var grid_center_x = torso_x if self.follow_cheetah else 0.0
        self._draw_ground_grid(grid_center_x)

        # Draw coordinate axes at origin or at cheetah position
        if self.follow_cheetah:
            self.renderer.draw_coordinate_axes(
                Vec3(torso_x - 1.0, 0.0, 0.0), 0.5
            )
        else:
            self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.0), 0.5)

        # Draw shadows first (under the cheetah)
        if self.show_shadows:
            self._draw_shadows(state)

        # Draw all body capsules
        self._draw_torso(state)
        self._draw_back_leg(state)
        self._draw_front_leg(state)

        # Draw velocity indicator
        if self.show_velocity:
            self._draw_velocity_indicator(state, vel_x)

        # End frame
        self.renderer.end_frame()

    fn _draw_ground_grid(self, center_x: Float64):
        """Draw chessboard ground pattern centered at given x position."""
        # Draw a chessboard pattern that moves with the cheetah
        var tile_size = 0.5
        var num_tiles_x = 20  # 10 tiles in each direction
        var num_tiles_y = 10  # 5 tiles in each direction

        # Round center to nearest tile for smooth scrolling
        var tile_center_x = Float64(Int(center_x / tile_size)) * tile_size

        for i in range(-num_tiles_x // 2, num_tiles_x // 2):
            for j in range(-num_tiles_y // 2, num_tiles_y // 2):
                var x0 = tile_center_x + Float64(i) * tile_size
                var y0 = Float64(j) * tile_size
                var x1 = x0 + tile_size
                var y1 = y0 + tile_size

                # Chessboard pattern - select color based on position
                var is_light = ((i + j) % 2 == 0)

                # Draw filled quad at z=0 (ground level)
                if is_light:
                    self.renderer.draw_filled_quad_3d(
                        Vec3(x0, y0, 0.0),
                        Vec3(x1, y0, 0.0),
                        Vec3(x1, y1, 0.0),
                        Vec3(x0, y1, 0.0),
                        Color3D(140, 140, 120),  # Light brown/gray
                    )
                else:
                    self.renderer.draw_filled_quad_3d(
                        Vec3(x0, y0, 0.0),
                        Vec3(x1, y0, 0.0),
                        Vec3(x1, y1, 0.0),
                        Vec3(x0, y1, 0.0),
                        Color3D(80, 80, 70),  # Dark brown/gray
                    )

    fn _draw_shadows(self, state: List[Scalar[dtype]]):
        """Draw shadows on the ground for all body parts."""
        # Shadow color - dark semi-transparent
        var shadow_color = Color3D(30, 30, 30)
        var ground_z = 0.001  # Slightly above ground to avoid z-fighting

        # Draw shadow for each body (projected onto ground plane)
        # Torso shadow (elongated ellipse along X)
        var torso_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_TORSO)
        var torso_pos = torso_pose[0]
        self._draw_ellipse_shadow(
            torso_pos.x, torso_pos.y, ground_z,
            HC3DConstantsCPU.TORSO_LENGTH / 2 + 0.05,  # Long axis (X)
            HC3DConstantsCPU.TORSO_RADIUS * 2,  # Short axis (Y)
            shadow_color,
        )

        # Back leg shadows
        var bthigh_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_BTHIGH)
        self._draw_circle_shadow(bthigh_pose[0].x, bthigh_pose[0].y, ground_z, 0.08, shadow_color)

        var bshin_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_BSHIN)
        self._draw_circle_shadow(bshin_pose[0].x, bshin_pose[0].y, ground_z, 0.07, shadow_color)

        var bfoot_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_BFOOT)
        self._draw_circle_shadow(bfoot_pose[0].x, bfoot_pose[0].y, ground_z, 0.06, shadow_color)

        # Front leg shadows
        var fthigh_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_FTHIGH)
        self._draw_circle_shadow(fthigh_pose[0].x, fthigh_pose[0].y, ground_z, 0.08, shadow_color)

        var fshin_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_FSHIN)
        self._draw_circle_shadow(fshin_pose[0].x, fshin_pose[0].y, ground_z, 0.07, shadow_color)

        var ffoot_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_FFOOT)
        self._draw_circle_shadow(ffoot_pose[0].x, ffoot_pose[0].y, ground_z, 0.06, shadow_color)

    fn _draw_circle_shadow(
        self,
        x: Float64,
        y: Float64,
        z: Float64,
        radius: Float64,
        color: Color3D,
    ):
        """Draw a circular shadow on the ground."""
        # Draw as a filled polygon approximating a circle
        var num_segments = 12
        from math import cos, sin, pi

        for i in range(num_segments):
            var angle0 = 2.0 * pi * Float64(i) / Float64(num_segments)
            var angle1 = 2.0 * pi * Float64(i + 1) / Float64(num_segments)

            var x0 = x + radius * cos(angle0)
            var y0 = y + radius * sin(angle0)
            var x1 = x + radius * cos(angle1)
            var y1 = y + radius * sin(angle1)

            # Draw triangle from center to edge
            self.renderer.draw_filled_quad_3d(
                Vec3(x, y, z),
                Vec3(x0, y0, z),
                Vec3(x1, y1, z),
                Vec3(x, y, z),  # Degenerate quad = triangle
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
        from math import cos, sin, pi

        for i in range(num_segments):
            var angle0 = 2.0 * pi * Float64(i) / Float64(num_segments)
            var angle1 = 2.0 * pi * Float64(i + 1) / Float64(num_segments)

            var x0 = x + radius_x * cos(angle0)
            var y0 = y + radius_y * sin(angle0)
            var x1 = x + radius_x * cos(angle1)
            var y1 = y + radius_y * sin(angle1)

            # Draw triangle from center to edge
            self.renderer.draw_filled_quad_3d(
                Vec3(x, y, z),
                Vec3(x0, y0, z),
                Vec3(x1, y1, z),
                Vec3(x, y, z),  # Degenerate quad = triangle
                color,
            )

    fn _get_body_pose(
        self, state: List[Scalar[dtype]], body_idx: Int
    ) -> Tuple[Vec3, Quat]:
        """Extract position and orientation for a body from state.

        Args:
            state: Physics state buffer.
            body_idx: Body index.

        Returns:
            Tuple of (position, orientation).
        """
        var offset = HC3DConstantsCPU.BODIES_OFFSET + body_idx * BODY_STATE_SIZE_3D

        var pos = Vec3(
            Float64(state[offset + IDX_PX]),
            Float64(state[offset + IDX_PY]),
            Float64(state[offset + IDX_PZ]),
        )

        var quat = Quat(
            Float64(state[offset + IDX_QW]),
            Float64(state[offset + IDX_QX]),
            Float64(state[offset + IDX_QY]),
            Float64(state[offset + IDX_QZ]),
        )

        return (pos, quat)

    fn _draw_torso(self, state: List[Scalar[dtype]]):
        """Draw the torso capsule."""
        var pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_TORSO)
        var pos = pose[0]
        var quat = pose[1]

        # Torso is a horizontal capsule along X-axis
        self.renderer.draw_shaded_capsule_2d(
            center=pos,
            orientation=quat,
            radius=HC3DConstantsCPU.TORSO_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.TORSO_LENGTH / 2,
            axis=0,  # X-axis (horizontal)
            color=CheetahColors.torso(),
        )

    fn _draw_back_leg(self, state: List[Scalar[dtype]]):
        """Draw the back leg (thigh, shin, foot)."""
        # Back thigh
        var thigh_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_BTHIGH)

        self.renderer.draw_shaded_capsule_2d(
            center=thigh_pose[0],
            orientation=thigh_pose[1],
            radius=HC3DConstantsCPU.BTHIGH_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.BTHIGH_LENGTH / 2 * Self.VISUAL_LENGTH_SCALE,
            axis=2,  # Z-axis (vertical)
            color=CheetahColors.back_leg(),
        )

        # Back shin
        var shin_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_BSHIN)
        self.renderer.draw_shaded_capsule_2d(
            center=shin_pose[0],
            orientation=shin_pose[1],
            radius=HC3DConstantsCPU.BSHIN_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.BSHIN_LENGTH / 2 * Self.VISUAL_LENGTH_SCALE,
            axis=2,
            color=CheetahColors.back_leg(),
        )

        # Back foot
        var foot_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_BFOOT)
        self.renderer.draw_shaded_capsule_2d(
            center=foot_pose[0],
            orientation=foot_pose[1],
            radius=HC3DConstantsCPU.BFOOT_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.BFOOT_LENGTH / 2 * Self.VISUAL_LENGTH_SCALE,
            axis=2,
            color=CheetahColors.back_leg(),
        )

    fn _draw_front_leg(self, state: List[Scalar[dtype]]):
        """Draw the front leg (thigh, shin, foot)."""
        # Front thigh
        var thigh_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_FTHIGH)

        self.renderer.draw_shaded_capsule_2d(
            center=thigh_pose[0],
            orientation=thigh_pose[1],
            radius=HC3DConstantsCPU.FTHIGH_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.FTHIGH_LENGTH / 2 * Self.VISUAL_LENGTH_SCALE,
            axis=2,  # Z-axis (vertical)
            color=CheetahColors.front_leg(),
        )

        # Front shin
        var shin_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_FSHIN)
        self.renderer.draw_shaded_capsule_2d(
            center=shin_pose[0],
            orientation=shin_pose[1],
            radius=HC3DConstantsCPU.FSHIN_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.FSHIN_LENGTH / 2 * Self.VISUAL_LENGTH_SCALE,
            axis=2,
            color=CheetahColors.front_leg(),
        )

        # Front foot
        var foot_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_FFOOT)
        self.renderer.draw_shaded_capsule_2d(
            center=foot_pose[0],
            orientation=foot_pose[1],
            radius=HC3DConstantsCPU.FFOOT_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.FFOOT_LENGTH / 2 * Self.VISUAL_LENGTH_SCALE,
            axis=2,
            color=CheetahColors.front_leg(),
        )

    fn _draw_velocity_indicator(
        self, state: List[Scalar[dtype]], vel_x: Float64
    ):
        """Draw a velocity indicator arrow above the torso."""
        var torso_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_TORSO)
        var torso_pos = torso_pose[0]

        # Draw velocity arrow above torso
        var arrow_start = Vec3(torso_pos.x, torso_pos.y, torso_pos.z + 0.3)
        var arrow_length = vel_x * 0.2  # Scale velocity for display
        var arrow_end = Vec3(
            arrow_start.x + arrow_length, arrow_start.y, arrow_start.z
        )

        from render3d.shapes3d import WireframeLine

        var lines = List[WireframeLine]()
        lines.append(WireframeLine(arrow_start, arrow_end))

        # Add arrowhead if velocity is significant
        if abs(arrow_length) > 0.05:
            var head_size = 0.05
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

        self.renderer.draw_lines_3d(lines, CheetahColors.velocity())

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64):
        """Orbit camera around target.

        Args:
            delta_theta: Horizontal rotation (radians).
            delta_phi: Vertical rotation (radians).
        """
        self.renderer.orbit_camera(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64):
        """Zoom camera in/out.

        Args:
            delta: Zoom amount.
        """
        self.renderer.zoom_camera(delta)

    fn delay(self, ms: Int):
        """Delay for given milliseconds."""
        self.renderer.delay(ms)
