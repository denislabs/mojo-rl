"""Generic robot renderer that draws capsule bodies from BodySpec definitions.

Parameterized by a variadic list of BodySpec types, iterates at compile time
to draw all bodies automatically. Eliminates per-environment renderer boilerplate.
"""

from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render3d import Renderer3D, Camera3D, Color3D
from core import EnvRenderer3D
from ..robot.body_spec import BodySpec

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


@fieldwise_init
struct RobotRenderer[*B: BodySpec](EnvRenderer3D, Movable):
    """Generic renderer for robots defined by BodySpec types.

    Draws capsule bodies using compile-time geometry from BodySpec (RADIUS,
    HALF_LENGTH). Camera follows torso (body 0) with configurable offsets.

    Parameters:
        B: Variadic list of BodySpec types defining the robot's bodies.
    """

    comptime body_types = Variadic.types[T=BodySpec, *Self.B]
    comptime NUM_BODIES: Int = Variadic.size(Self.body_types)

    var renderer: Renderer3D
    var initialized: Bool
    var follow: Bool
    var show_velocity: Bool
    var visual_radius_scale: Float64

    # Camera configuration
    var cam_eye_y: Float64
    var cam_eye_z: Float64
    var cam_target_z: Float64
    var axes_offset: Float64
    var vel_arrow_height: Float64
    var vel_arrow_scale: Float64

    # Velocity arrow color
    var vel_color: Color3D

    fn __init__(
        out self,
        width: Int = 1280,
        height: Int = 720,
        visual_radius_scale: Float64 = 2.0,
        cam_eye_y: Float64 = -3.0,
        cam_eye_z: Float64 = 1.0,
        cam_target_z: Float64 = 0.5,
        axes_offset: Float64 = 1.5,
        vel_arrow_height: Float64 = 0.15,
        vel_arrow_scale: Float64 = 0.1,
        vel_color: Color3D = Color3D(0, 255, 255),
        follow: Bool = True,
        show_velocity: Bool = True,
        title: String = String("Robot Environment"),
    ) raises:
        self.visual_radius_scale = visual_radius_scale
        self.cam_eye_y = cam_eye_y
        self.cam_eye_z = cam_eye_z
        self.cam_target_z = cam_target_z
        self.axes_offset = axes_offset
        self.vel_arrow_height = vel_arrow_height
        self.vel_arrow_scale = vel_arrow_scale
        self.vel_color = vel_color
        self.follow = follow
        self.show_velocity = show_velocity

        var camera = Camera3D(
            eye=Vec3(0.0, cam_eye_y, cam_eye_z),
            target=Vec3(0.0, 0.0, cam_target_z),
            up=Vec3(0.0, 0.0, 1.0),
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

    fn __moveinit__(out self, deinit other: Self):
        self.renderer = other.renderer^
        self.initialized = other.initialized
        self.follow = other.follow
        self.show_velocity = other.show_velocity
        self.visual_radius_scale = other.visual_radius_scale
        self.cam_eye_y = other.cam_eye_y
        self.cam_eye_z = other.cam_eye_z
        self.cam_target_z = other.cam_target_z
        self.axes_offset = other.axes_offset
        self.vel_arrow_height = other.vel_arrow_height
        self.vel_arrow_scale = other.vel_arrow_scale
        self.vel_color = other.vel_color

    fn init(mut self) raises -> None:
        var title = String("Robot Environment")
        self.renderer.init(title)
        self.initialized = True

    fn close(mut self) raises -> None:
        if self.initialized:
            self.renderer.close()
            self.initialized = False

    fn check_quit(mut self) -> Bool:
        return self.renderer.check_quit()

    fn is_open(self) -> Bool:
        return self.initialized

    fn delay(self, ms: Int) -> None:
        try:
            self.renderer.delay_ms(ms)
        except:
            pass

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64) -> None:
        self.renderer.orbit_camera(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64) -> None:
        self.renderer.zoom_camera(delta)

    fn render(
        mut self,
        positions: List[Vec3],
        quaternions: List[Quat],
        vel_x: Float64 = 0.0,
    ):
        """Render all robot bodies.

        Args:
            positions: World positions for each body (len >= NUM_BODIES).
            quaternions: World orientations for each body (len >= NUM_BODIES).
            vel_x: Forward velocity for indicator arrow.
        """
        if not self.initialized:
            return

        if (
            len(positions) < Self.NUM_BODIES
            or len(quaternions) < Self.NUM_BODIES
        ):
            return

        var torso_pos = positions[0]

        # Camera follow torso
        if self.follow:
            self.renderer.camera.target = Vec3(
                torso_pos.x, 0.0, self.cam_target_z
            )
            self.renderer.camera.eye = Vec3(
                torso_pos.x, self.cam_eye_y, self.cam_eye_z
            )

        self.renderer.begin_frame()

        # Ground grid — offset by max radius across all bodies × (scale - 1)
        var max_radius: Float64 = 0.0

        @parameter
        for i in range(Self.NUM_BODIES):
            comptime B = Self.body_types[i]
            if B.RADIUS > max_radius:
                max_radius = B.RADIUS

        var ground_offset = -max_radius * (self.visual_radius_scale - 1.0)
        var grid_center_x = torso_pos.x if self.follow else 0.0
        self.renderer.draw_ground_grid(grid_center_x, height=ground_offset)

        # Coordinate axes
        if self.follow:
            self.renderer.draw_coordinate_axes(
                Vec3(torso_pos.x - self.axes_offset, 0.0, 0.0), 0.2
            )
        else:
            self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.0), 0.2)

        # Draw all body capsules
        try:

            @parameter
            for i in range(Self.NUM_BODIES):
                comptime B = Self.body_types[i]
                self.renderer.draw_capsule(
                    center=positions[i],
                    orientation=quaternions[i],
                    radius=B.RADIUS * self.visual_radius_scale,
                    half_height=B.HALF_LENGTH,
                    axis=2,
                    color=B.COLOR,
                )
        except:
            pass

        # Velocity indicator
        if self.show_velocity:
            self._draw_velocity_indicator(torso_pos, vel_x)

        # End frame
        try:
            self.renderer.end_frame()
        except:
            pass

    fn _draw_velocity_indicator(mut self, torso_pos: Vec3, vel_x: Float64):
        """Draw a velocity indicator arrow above the torso."""
        var arrow_start = Vec3(
            torso_pos.x, torso_pos.y, torso_pos.z + self.vel_arrow_height
        )
        var arrow_length = vel_x * self.vel_arrow_scale
        var arrow_end = Vec3(
            arrow_start.x + arrow_length, arrow_start.y, arrow_start.z
        )

        self.renderer.draw_line_3d(arrow_start, arrow_end, self.vel_color)

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
                self.vel_color,
            )
            self.renderer.draw_line_3d(
                arrow_end,
                Vec3(
                    arrow_end.x - head_size * direction,
                    arrow_end.y,
                    arrow_end.z - head_size,
                ),
                self.vel_color,
            )
