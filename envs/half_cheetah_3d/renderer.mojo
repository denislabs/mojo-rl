"""HalfCheetah3D Renderer using the GPU-accelerated Renderer3D.

Provides visualization of the HalfCheetah3D environment with:
- GPU-rendered capsule bodies for each body segment
- Color-coded body parts (torso, legs)
- Procedural checkerboard ground plane
- Orbital camera control for interactive viewing

Implements EnvRenderer3D trait for integration with evaluation code.
Uses ref-based borrowing for safe access to environment state.
"""

from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render3d import Renderer3D, Camera3D, Color3D
from core import EnvRenderer3D

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
    fn velocity() -> Color3D:
        """Velocity indicator color - cyan."""
        return Color3D(0, 255, 255)


# =============================================================================
# HalfCheetah3D Renderer
# =============================================================================


struct HalfCheetah3DRenderer(EnvRenderer3D, Movable):
    """Renderer for HalfCheetah3D environment.

    Uses GPU-rendered capsules to visualize each body segment of the cheetah.
    Supports interactive camera control for orbit, zoom, and pan.

    Implements EnvRenderer3D trait for integration with evaluation code.
    Can be used in two ways:
    1. Direct: renderer.render(state, torso_x, vel_x)
    2. Via env: env.render(renderer)  # renderer borrows env via ref
    """

    var renderer: Renderer3D
    var initialized: Bool
    var follow_cheetah: Bool
    var show_velocity: Bool
    # Visual scale factor for capsule radii (physics radii are small)
    comptime VISUAL_RADIUS_SCALE: Float64 = 1.4
    # Visual scale for capsule lengths (to make them more visible)
    comptime VISUAL_LENGTH_SCALE: Float64 = 1.3

    fn __init__(
        out self,
        width: Int = 1024,
        height: Int = 576,
        follow_cheetah: Bool = True,
        show_velocity: Bool = True,
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

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor - transfers ownership of renderer."""
        self.renderer = other.renderer^
        self.initialized = other.initialized
        self.follow_cheetah = other.follow_cheetah
        self.show_velocity = other.show_velocity

    fn init(mut self) raises -> None:
        """Initialize the renderer window."""
        var title = String("HalfCheetah3D Environment")
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

        # Draw ground grid (GPU shader checkerboard)
        # Offset ground slightly below z=0 to account for visual radius scaling
        var ground_offset = -Float64(HC3DConstantsCPU.BFOOT_RADIUS) * (Self.VISUAL_RADIUS_SCALE - 1.0)
        var grid_center_x = torso_x if self.follow_cheetah else 0.0
        self.renderer.draw_ground_grid(grid_center_x, height=ground_offset)

        # Draw coordinate axes at origin or at cheetah position
        if self.follow_cheetah:
            self.renderer.draw_coordinate_axes(
                Vec3(torso_x - 1.0, 0.0, 0.0), 0.5
            )
        else:
            self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.0), 0.5)

        # Draw all body capsules
        try:
            self._draw_torso(state)
            self._draw_back_leg(state)
            self._draw_front_leg(state)
        except:
            pass

        # Draw velocity indicator
        if self.show_velocity:
            self._draw_velocity_indicator(state, vel_x)

        # End frame
        try:
            self.renderer.end_frame()
        except:
            pass

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
        var offset = (
            HC3DConstantsCPU.BODIES_OFFSET + body_idx * BODY_STATE_SIZE_3D
        )

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

    fn _draw_torso(mut self, state: List[Scalar[dtype]]) raises:
        """Draw the torso capsule."""
        var pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_TORSO)
        var pos = pose[0]
        var quat = pose[1]

        # Torso is a horizontal capsule along X-axis
        self.renderer.draw_capsule(
            center=pos,
            orientation=quat,
            radius=HC3DConstantsCPU.TORSO_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.TORSO_LENGTH / 2,
            axis=0,  # X-axis (horizontal)
            color=CheetahColors.torso(),
        )

    fn _draw_back_leg(mut self, state: List[Scalar[dtype]]) raises:
        """Draw the back leg (thigh, shin, foot)."""
        # Back thigh
        var thigh_pose = self._get_body_pose(
            state, HC3DConstantsCPU.BODY_BTHIGH
        )

        self.renderer.draw_capsule(
            center=thigh_pose[0],
            orientation=thigh_pose[1],
            radius=HC3DConstantsCPU.BTHIGH_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.BTHIGH_LENGTH
            / 2
            * Self.VISUAL_LENGTH_SCALE,
            axis=2,  # Z-axis (vertical)
            color=CheetahColors.back_leg(),
        )

        # Back shin
        var shin_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_BSHIN)
        self.renderer.draw_capsule(
            center=shin_pose[0],
            orientation=shin_pose[1],
            radius=HC3DConstantsCPU.BSHIN_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.BSHIN_LENGTH
            / 2
            * Self.VISUAL_LENGTH_SCALE,
            axis=2,
            color=CheetahColors.back_leg(),
        )

        # Back foot
        var foot_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_BFOOT)
        self.renderer.draw_capsule(
            center=foot_pose[0],
            orientation=foot_pose[1],
            radius=HC3DConstantsCPU.BFOOT_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.BFOOT_LENGTH
            / 2
            * Self.VISUAL_LENGTH_SCALE,
            axis=2,
            color=CheetahColors.back_leg(),
        )

    fn _draw_front_leg(mut self, state: List[Scalar[dtype]]) raises:
        """Draw the front leg (thigh, shin, foot)."""
        # Front thigh
        var thigh_pose = self._get_body_pose(
            state, HC3DConstantsCPU.BODY_FTHIGH
        )

        self.renderer.draw_capsule(
            center=thigh_pose[0],
            orientation=thigh_pose[1],
            radius=HC3DConstantsCPU.FTHIGH_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.FTHIGH_LENGTH
            / 2
            * Self.VISUAL_LENGTH_SCALE,
            axis=2,  # Z-axis (vertical)
            color=CheetahColors.front_leg(),
        )

        # Front shin
        var shin_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_FSHIN)
        self.renderer.draw_capsule(
            center=shin_pose[0],
            orientation=shin_pose[1],
            radius=HC3DConstantsCPU.FSHIN_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.FSHIN_LENGTH
            / 2
            * Self.VISUAL_LENGTH_SCALE,
            axis=2,
            color=CheetahColors.front_leg(),
        )

        # Front foot
        var foot_pose = self._get_body_pose(state, HC3DConstantsCPU.BODY_FFOOT)
        self.renderer.draw_capsule(
            center=foot_pose[0],
            orientation=foot_pose[1],
            radius=HC3DConstantsCPU.FFOOT_RADIUS * Self.VISUAL_RADIUS_SCALE,
            half_height=HC3DConstantsCPU.FFOOT_LENGTH
            / 2
            * Self.VISUAL_LENGTH_SCALE,
            axis=2,
            color=CheetahColors.front_leg(),
        )

    fn _draw_velocity_indicator(
        mut self, state: List[Scalar[dtype]], vel_x: Float64
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

        # Main arrow line
        self.renderer.draw_line_3d(arrow_start, arrow_end, CheetahColors.velocity())

        # Add arrowhead if velocity is significant
        if abs(arrow_length) > 0.05:
            var head_size = 0.05
            var direction = 1.0 if arrow_length > 0 else -1.0
            self.renderer.draw_line_3d(
                arrow_end,
                Vec3(
                    arrow_end.x - head_size * direction,
                    arrow_end.y,
                    arrow_end.z + head_size,
                ),
                CheetahColors.velocity(),
            )
            self.renderer.draw_line_3d(
                arrow_end,
                Vec3(
                    arrow_end.x - head_size * direction,
                    arrow_end.y,
                    arrow_end.z - head_size,
                ),
                CheetahColors.velocity(),
            )

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64) -> None:
        """Orbit camera around target.

        Args:
            delta_theta: Horizontal rotation (radians).
            delta_phi: Vertical rotation (radians).
        """
        self.renderer.orbit_camera(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64) -> None:
        """Zoom camera in/out.

        Args:
            delta: Zoom amount.
        """
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
