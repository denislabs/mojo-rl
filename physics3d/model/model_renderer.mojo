"""Generic model renderer that draws geoms from GeomSpec definitions.

Parameterized by a variadic list of GeomSpec types, iterates at compile time
to draw all visible geoms automatically. Eliminates per-environment renderer boilerplate.

Supports all geom types: capsule, sphere, box, and plane (ground).
"""

from collections import InlineArray
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render import Renderer3D, Camera3D, Color
from core import EnvRenderer3D
from ..model.geom_spec import GeomSpec, GeomsLike
from ..model.camera_spec import CamerasLike, _EmptyCameras
from ..model.light_spec import LightsLike, _EmptyLights

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


@fieldwise_init
struct ModelRenderer[MODEL_DEF: ModelDefLike](EnvRenderer3D, Movable):
    """Generic renderer for models defined by GeomSpec types.

    Draws all geom types (capsule, sphere, box, plane) using compile-time
    geometry from GeomSpec. Camera follows torso (body 1) with configurable
    offsets.

    Parameters:
        MODEL_DEF: ModelDefLike type defining the model's definition.
    """

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
    var vel_color: Color

    # Light configuration
    var light_dir_x: Float64
    var light_dir_y: Float64
    var light_dir_z: Float64
    var light_color_r: Float64
    var light_color_g: Float64
    var light_color_b: Float64
    var light_ambient: Float64

    fn __init__(
        out self,
        width: Int = 1280,
        height: Int = 720,
        visual_radius_scale: Float64 = 2.0,
        axes_offset: Float64 = 1.5,
        vel_arrow_height: Float64 = 0.15,
        vel_arrow_scale: Float64 = 0.1,
        vel_color: Color = Color(0, 255, 255, 255),
        follow: Bool = True,
        show_velocity: Bool = True,
        title: String = String("Model Environment"),
    ) raises:
        var cameras = Self.MODEL_DEF.setup_cameras(width, height)
        var camera = cameras[0].copy()

        var lights = Self.MODEL_DEF.setup_lights()
        var light = lights[0].copy()

        self.visual_radius_scale = visual_radius_scale
        self.cam_eye_y = camera.eye.y
        self.cam_eye_z = camera.eye.z
        self.cam_target_z = camera.target.z
        self.axes_offset = axes_offset
        self.vel_arrow_height = vel_arrow_height
        self.vel_arrow_scale = vel_arrow_scale
        self.vel_color = vel_color
        self.follow = follow
        self.show_velocity = show_velocity
        self.light_dir_x = light.dir_x
        self.light_dir_y = light.dir_y
        self.light_dir_z = light.dir_z
        self.light_color_r = light.color_r
        self.light_color_g = light.color_g
        self.light_color_b = light.color_b
        self.light_ambient = light.ambient

        self.renderer = Renderer3D(
            width=width,
            height=height,
            camera=camera,
            draw_grid=True,
            draw_axes=True,
            light_dir_x=Float32(light.dir_x),
            light_dir_y=Float32(light.dir_y),
            light_dir_z=Float32(light.dir_z),
            light_color_r=Float32(light.color_r),
            light_color_g=Float32(light.color_g),
            light_color_b=Float32(light.color_b),
            light_ambient=Float32(light.ambient),
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
        self.light_dir_x = other.light_dir_x
        self.light_dir_y = other.light_dir_y
        self.light_dir_z = other.light_dir_z
        self.light_color_r = other.light_color_r
        self.light_color_g = other.light_color_g
        self.light_color_b = other.light_color_b
        self.light_ambient = other.light_ambient

    fn init(mut self) raises -> None:
        var title = String("Model Environment")
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

    fn render_from_body_state[
        DTYPE: DType, SIZE_POS: Int, SIZE_QUAT: Int
    ](
        mut self,
        xpos: InlineArray[Scalar[DTYPE], SIZE_POS],
        xquat: InlineArray[Scalar[DTYPE], SIZE_QUAT],
        num_bodies: Int,
        vel_x: Float64 = 0.0,
    ):
        """Render directly from physics Data body arrays.

        Extracts body positions and quaternions from raw xpos/xquat arrays
        and delegates to render(). Eliminates per-environment boilerplate.

        Args:
            xpos: Flat array of body positions [x0,y0,z0, x1,y1,z1, ...].
            xquat: Flat array of body quaternions [x0,y0,z0,w0, ...].
            num_bodies: Number of bodies in the arrays.
            vel_x: Forward velocity for indicator arrow.
        """
        var positions = List[Vec3](capacity=num_bodies)
        var quaternions = List[Quat](capacity=num_bodies)

        for i in range(num_bodies):
            positions.append(
                Vec3(
                    Float64(xpos[i * 3 + 0]),
                    Float64(xpos[i * 3 + 1]),
                    Float64(xpos[i * 3 + 2]),
                )
            )
            # xquat stored as [x, y, z, w], Quat constructor is (w, x, y, z)
            quaternions.append(
                Quat(
                    Float64(xquat[i * 4 + 3]),
                    Float64(xquat[i * 4 + 0]),
                    Float64(xquat[i * 4 + 1]),
                    Float64(xquat[i * 4 + 2]),
                )
            )

        self.render(positions, quaternions, vel_x)

    fn render(
        mut self,
        positions: List[Vec3],
        quaternions: List[Quat],
        vel_x: Float64 = 0.0,
    ):
        """Render all visible geoms.

        Args:
            positions: World positions for each body (indexed by BODY_IDX).
            quaternions: World orientations for each body (indexed by BODY_IDX).
            vel_x: Forward velocity for indicator arrow.
        """
        if not self.initialized:
            return

        var torso_pos = positions[1]  # Body 1 = torso (body 0 = worldbody)

        # Camera follow torso
        if self.follow:
            self.renderer.camera.target = Vec3(
                torso_pos.x, 0.0, self.cam_target_z
            )
            self.renderer.camera.eye = Vec3(
                torso_pos.x, self.cam_eye_y, self.cam_eye_z
            )

        self.renderer.begin_frame()

        # Render ground geoms (planes or fallback grid)
        try:
            Self.MODEL_DEF.render_ground_geoms(
                self.renderer,
                torso_pos.x,
                self.follow,
                self.visual_radius_scale,
            )
        except:
            pass

        # Coordinate axes
        if self.follow:
            self.renderer.draw_coordinate_axes(
                Vec3(torso_pos.x - self.axes_offset, 0.0, 0.0), 0.2
            )
        else:
            self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.0), 0.2)

        # Render body-attached geoms
        try:
            Self.MODEL_DEF.render_body_geoms(
                self.renderer,
                positions,
                quaternions,
                self.visual_radius_scale,
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
