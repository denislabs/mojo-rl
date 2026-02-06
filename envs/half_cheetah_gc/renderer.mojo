"""HalfCheetahGC Renderer using the GPU-accelerated Renderer3D.

Provides visualization of the HalfCheetahGC environment with:
- GPU-rendered capsule bodies for each body segment (8 total: torso, head, and 6 leg segments)
- Color-coded body parts (torso, head, back leg, front leg)
- Procedural checkerboard ground plane
- Orbital camera control for interactive viewing

Implements EnvRenderer3D trait for integration with evaluation code.
"""

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
    fn velocity() -> Color3D:
        """Velocity indicator color - cyan."""
        return Color3D(0, 255, 255)


# =============================================================================
# HalfCheetahGC Renderer
# =============================================================================


struct HalfCheetahGCRenderer(EnvRenderer3D, Movable):
    """Renderer for HalfCheetahGC environment.

    Uses GPU-rendered capsules to visualize each body segment of the half cheetah.
    Supports interactive camera control for orbit, zoom, and pan.

    Implements EnvRenderer3D trait for integration with evaluation code.
    """

    var renderer: Renderer3D
    var initialized: Bool
    var follow_cheetah: Bool
    var show_velocity: Bool

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

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor - transfers ownership of renderer."""
        self.renderer = other.renderer^
        self.initialized = other.initialized
        self.follow_cheetah = other.follow_cheetah
        self.show_velocity = other.show_velocity
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

        # Draw ground grid (GPU shader checkerboard)
        # Offset ground slightly below z=0 to account for visual radius scaling
        # (visual capsule bottom extends below physics contact point)
        var ground_offset = -self.capsule_radius * (Self.VISUAL_RADIUS_SCALE - 1.0)
        var grid_center_x = torso_pos.x if self.follow_cheetah else 0.0
        self.renderer.draw_ground_grid(grid_center_x, height=ground_offset)

        # Draw coordinate axes
        if self.follow_cheetah:
            self.renderer.draw_coordinate_axes(
                Vec3(torso_pos.x - 1.5, 0.0, 0.0), 0.2
            )
        else:
            self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.0), 0.2)

        # Draw all body capsules
        try:
            self._draw_torso(positions[BODY_TORSO], quaternions[BODY_TORSO])
            self._draw_head(positions[BODY_HEAD], quaternions[BODY_HEAD])
            self._draw_back_thigh(positions[BODY_BTHIGH], quaternions[BODY_BTHIGH])
            self._draw_back_shin(positions[BODY_BSHIN], quaternions[BODY_BSHIN])
            self._draw_back_foot(positions[BODY_BFOOT], quaternions[BODY_BFOOT])
            self._draw_front_thigh(positions[BODY_FTHIGH], quaternions[BODY_FTHIGH])
            self._draw_front_shin(positions[BODY_FSHIN], quaternions[BODY_FSHIN])
            self._draw_front_foot(positions[BODY_FFOOT], quaternions[BODY_FFOOT])
        except:
            pass

        # Draw velocity indicator
        if self.show_velocity:
            self._draw_velocity_indicator(torso_pos, vel_x)

        # End frame
        try:
            self.renderer.end_frame()
        except:
            pass

    fn _draw_torso(mut self, pos: Vec3, quat: Quat) raises:
        """Draw the torso capsule (horizontal along X-axis)."""
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.torso_half_length,
            axis=2,  # Z-axis, will be rotated by quat to lie along X
            color=HalfCheetahGCColors.torso(),
        )

    fn _draw_head(mut self, pos: Vec3, quat: Quat) raises:
        """Draw the head capsule using its physics position and orientation."""
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.head_half_length,
            axis=2,  # Z-axis, will be rotated by quat
            color=HalfCheetahGCColors.head(),
        )

    fn _draw_back_thigh(mut self, pos: Vec3, quat: Quat) raises:
        """Draw the back thigh capsule (vertical)."""
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.bthigh_half_length,
            axis=2,  # Z-axis (vertical)
            color=HalfCheetahGCColors.back_thigh(),
        )

    fn _draw_back_shin(mut self, pos: Vec3, quat: Quat) raises:
        """Draw the back shin capsule (vertical)."""
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.bshin_half_length,
            axis=2,  # Z-axis (vertical)
            color=HalfCheetahGCColors.back_shin(),
        )

    fn _draw_back_foot(mut self, pos: Vec3, quat: Quat) raises:
        """Draw the back foot capsule (horizontal)."""
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.bfoot_half_length,
            axis=2,  # Z-axis, will be rotated by quat to lie along X
            color=HalfCheetahGCColors.back_foot(),
        )

    fn _draw_front_thigh(mut self, pos: Vec3, quat: Quat) raises:
        """Draw the front thigh capsule (vertical)."""
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.fthigh_half_length,
            axis=2,  # Z-axis (vertical)
            color=HalfCheetahGCColors.front_thigh(),
        )

    fn _draw_front_shin(mut self, pos: Vec3, quat: Quat) raises:
        """Draw the front shin capsule (vertical)."""
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.fshin_half_length,
            axis=2,  # Z-axis (vertical)
            color=HalfCheetahGCColors.front_shin(),
        )

    fn _draw_front_foot(mut self, pos: Vec3, quat: Quat) raises:
        """Draw the front foot capsule (horizontal)."""
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=self.capsule_radius * Self.VISUAL_RADIUS_SCALE,
            half_height=self.ffoot_half_length,
            axis=2,  # Z-axis, will be rotated by quat to lie along X
            color=HalfCheetahGCColors.front_foot(),
        )

    fn _draw_velocity_indicator(mut self, torso_pos: Vec3, vel_x: Float64):
        """Draw a velocity indicator arrow above the torso."""
        var arrow_start = Vec3(torso_pos.x, torso_pos.y, torso_pos.z + 0.15)
        var arrow_length = vel_x * 0.1  # Scale velocity for display
        var arrow_end = Vec3(
            arrow_start.x + arrow_length, arrow_start.y, arrow_start.z
        )

        # Main arrow line
        self.renderer.draw_line_3d(arrow_start, arrow_end, HalfCheetahGCColors.velocity())

        # Add arrowhead if velocity is significant
        if abs(arrow_length) > 0.03:
            var head_size = 0.04
            var direction = 1.0 if arrow_length > 0 else -1.0
            self.renderer.draw_line_3d(
                arrow_end,
                Vec3(
                    arrow_end.x - head_size * direction,
                    arrow_end.y,
                    arrow_end.z + head_size,
                ),
                HalfCheetahGCColors.velocity(),
            )
            self.renderer.draw_line_3d(
                arrow_end,
                Vec3(
                    arrow_end.x - head_size * direction,
                    arrow_end.y,
                    arrow_end.z - head_size,
                ),
                HalfCheetahGCColors.velocity(),
            )

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64) -> None:
        """Orbit camera around target."""
        self.renderer.orbit_camera(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64) -> None:
        """Zoom camera in/out."""
        self.renderer.zoom_camera(delta)

    fn delay(self, ms: Int) -> None:
        """Delay for given milliseconds."""
        try:
            self.renderer.delay_ms(ms)
        except:
            pass

    # =========================================================================
    # EnvRenderer Trait Implementation
    # =========================================================================

    fn is_open(self) -> Bool:
        """Check if renderer window is still open.

        Returns:
            True if renderer is initialized and window is open.
        """
        return self.initialized
