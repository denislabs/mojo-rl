"""Generic model renderer that draws geoms from GeomSpec definitions.

Parameterized by a variadic list of GeomSpec types, iterates at compile time
to draw all visible geoms automatically. Eliminates per-environment renderer boilerplate.

Supports all geom types: capsule, sphere, box, and plane (ground).
"""

from std.collections import InlineArray
from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from mojo_rl.render import Renderer3D, Camera3D, Color
from mojo_rl.render.light import Light
from mojo_rl.core import EnvRenderer3D
from ..model.geom_spec import GeomSpec, GeomsLike
from ..model.camera_spec import CamerasLike, _EmptyCameras
from ..model.light_spec import LightsLike, _EmptyLights

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


@fieldwise_init
struct ModelRenderer[MODEL_DEF: ModelDefLike](EnvRenderer3D, Movable):
    """Generic renderer for models defined by GeomSpec types.

    Draws all geom types (capsule, sphere, box, plane) using compile-time
    geometry from GeomSpec. Supports multiple cameras (switch with 1-9 keys)
    and multiple lights.

    Parameters:
        MODEL_DEF: ModelDefLike type defining the model's definition.
    """

    var renderer: Renderer3D
    var initialized: Bool
    var follow: Bool
    var show_velocity: Bool
    var visual_radius_scale: Float64

    # Multi-camera support
    var cameras: List[Camera3D]
    var camera_modes: List[Int]  # CAM_TRACKCOM=0, CAM_FIXED=1
    var active_camera: Int

    var axes_offset: Float64
    var vel_arrow_height: Float64
    var vel_arrow_scale: Float64

    # Velocity arrow color
    var vel_color: Color

    # Site marker visibility
    var show_sites: Bool

    # HUD state
    var step_count: Int

    def __init__(
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
        show_sites: Bool = False,
        title: String = String("Model Environment"),
    ) raises:
        # Setup all cameras from spec (fallback to default if none defined)
        var cam_list = Self.MODEL_DEF.setup_cameras(width, height)
        var mode_list = Self.MODEL_DEF.setup_camera_modes()
        self.cameras = List[Camera3D]()
        self.camera_modes = List[Int]()
        if len(cam_list) == 0:
            # No cameras in XML — add a default orbit camera
            self.cameras.append(Camera3D(
                eye=Vec3(0.0, -2.5, 1.5),
                target=Vec3(0.0, 0.5, 0.3),
                up=Vec3(0.0, 0.0, 1.0),
                fov=45.0,
                aspect=Float64(width) / Float64(height),
                near=0.01,
                far=100.0,
                screen_width=width,
                screen_height=height,
            ))
            self.camera_modes.append(1)  # CAM_FIXED
        else:
            for i in range(len(cam_list)):
                self.cameras.append(cam_list[i].copy())
                if i < len(mode_list):
                    self.camera_modes.append(mode_list[i])
                else:
                    self.camera_modes.append(0)  # CAM_TRACKCOM fallback
        self.active_camera = 0

        # Setup all lights from spec (fallback to default if none defined)
        var lights = Self.MODEL_DEF.setup_lights()
        if len(lights) == 0:
            # No lights in XML — add default directional light
            lights.append(Light(
                mode=0,  # directional
                dir_x=0.5, dir_y=0.5, dir_z=-1.0,
                color_r=0.7, color_g=0.7, color_b=0.7,
                ambient=0.3,
                specular_intensity=0.3,
                specular_exponent=10.0,
                cast_shadow=True,
            ))

        self.visual_radius_scale = visual_radius_scale
        self.axes_offset = axes_offset
        self.vel_arrow_height = vel_arrow_height
        self.vel_arrow_scale = vel_arrow_scale
        self.vel_color = vel_color
        self.follow = follow
        self.show_velocity = show_velocity
        self.show_sites = show_sites

        var camera = self.cameras[0].copy()
        self.renderer = Renderer3D(
            width=width,
            height=height,
            camera=camera,
            draw_grid=True,
            draw_axes=False,
            lights=lights,
        )

        # Configure skybox from GradientTexture (if model defines one)
        var skybox = Self.MODEL_DEF.get_skybox_colors()
        if len(skybox) == 6:
            self.renderer.set_skybox(
                top_r=Float32(skybox[0]),
                top_g=Float32(skybox[1]),
                top_b=Float32(skybox[2]),
                bottom_r=Float32(skybox[3]),
                bottom_g=Float32(skybox[4]),
                bottom_b=Float32(skybox[5]),
            )

        # Configure ground checker from CheckerTexture (if model defines one)
        var checker = Self.MODEL_DEF.get_checker_colors()
        if len(checker) == 3:
            self.renderer.set_ground_checker_colors(
                r=Float32(checker[0]),
                g=Float32(checker[1]),
                b=Float32(checker[2]),
            )

        self.step_count = 0
        self.initialized = False

    def __init__(out self, *, deinit take: Self):
        self.renderer = take.renderer^
        self.initialized = take.initialized
        self.follow = take.follow
        self.show_velocity = take.show_velocity
        self.show_sites = take.show_sites
        self.visual_radius_scale = take.visual_radius_scale
        self.cameras = take.cameras^
        self.camera_modes = take.camera_modes^
        self.active_camera = take.active_camera
        self.axes_offset = take.axes_offset
        self.vel_arrow_height = take.vel_arrow_height
        self.vel_arrow_scale = take.vel_arrow_scale
        self.vel_color = take.vel_color
        self.step_count = take.step_count

    def init(mut self) raises -> None:
        var title = String("Model Environment")
        self.renderer.init(title)
        self.initialized = True

    def close(mut self) raises -> None:
        if self.initialized:
            self.renderer.close()
            self.initialized = False

    def check_quit(mut self) -> Bool:
        return self.renderer.check_quit()

    def is_open(self) -> Bool:
        return self.initialized

    def delay(self, ms: Int) -> None:
        try:
            self.renderer.delay_ms(ms)
        except:
            pass

    def orbit_camera(
        mut self, delta_theta: Float64, delta_phi: Float64
    ) -> None:
        self.renderer.orbit_camera(delta_theta, delta_phi)

    def zoom_camera(mut self, delta: Float64) -> None:
        self.renderer.zoom_camera(delta)

    def pan_camera(mut self, delta_x: Float64, delta_y: Float64) -> None:
        self.renderer.pan_camera(delta_x, delta_y)

    def reset_camera(mut self) -> None:
        self.renderer.reset_camera()

    def render_from_body_state[
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

    def render(
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

        # Handle camera switch request from mojo_rl.renderer3D (number keys 1-9)
        var cam_req = self.renderer.camera_switch_request
        if cam_req >= 0 and cam_req < len(self.cameras):
            self.active_camera = cam_req
            # Load the new camera settings into the renderer
            var new_cam = self.cameras[self.active_camera].copy()
            self.renderer.camera.eye = new_cam.eye
            self.renderer.camera.target = new_cam.target
            self.renderer.camera.up = new_cam.up
            self.renderer.camera.fov = new_cam.fov
            self.renderer.camera.near = new_cam.near
            self.renderer.camera.far = new_cam.far

        var torso_pos = positions[1]  # Body 1 = torso (body 0 = worldbody)

        # Camera follow torso (only for trackcom mode cameras)
        var cam_mode = self.camera_modes[self.active_camera]
        if self.follow and cam_mode == 0:  # CAM_TRACKCOM
            # Preserve the current eye-to-target offset so mouse orbit is respected.
            # Each frame we only translate both eye and target to follow the torso.
            var offset = self.renderer.camera.eye - self.renderer.camera.target
            self.renderer.camera.target = Vec3(torso_pos.x, 0.0, torso_pos.z)
            self.renderer.camera.eye = self.renderer.camera.target + offset

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

        # Render site markers (bright green spheres, optional)
        if self.show_sites:
            try:
                Self.MODEL_DEF.render_sites(
                    self.renderer, positions, quaternions
                )
            except:
                pass

        # Velocity indicator
        if self.show_velocity:
            self._draw_velocity_indicator(torso_pos, vel_x)

        # HUD overlay
        self._draw_hud()

        # Increment step counter AFTER drawing (so first frame shows 0)
        # Only increment when not paused (paused display should freeze the count)
        if not self.renderer.is_paused:
            self.step_count += 1

        # End frame
        try:
            self.renderer.end_frame()
        except:
            pass

    def _draw_hud(mut self):
        """Draw MuJoCo-style HUD: controls help, camera name, step counter, pause indicator.
        """
        var x0 = Float32(12)
        var y = Float32(12)
        var s = 2  # 2× scale → 16×16 px per char

        # Controls (dim white) — shadow then bright
        var dim = Color(180, 180, 180, 200)
        self.renderer.draw_text(
            x0 + 1, y + 1, "[Spc] Pause", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[Spc] Pause", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[->]  Step", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[->]  Step", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[1-9] Camera", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[1-9] Camera", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[R]   Reset cam", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[R]   Reset cam", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[S]   Screenshot", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[S]   Screenshot", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[V]   Record", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[V]   Record", dim, s)
        y += 28  # gap

        # Camera name (bright white)
        var cam_name = String("Cam ") + String(self.active_camera + 1)
        self.renderer.draw_text(x0 + 1, y + 1, cam_name, Color(0, 0, 0, 160), s)
        self.renderer.draw_text(x0, y, cam_name, Color(255, 255, 255, 255), s)
        y += 20

        # Step counter (yellow-white)
        var step_str = String("Step: ") + String(self.step_count)
        self.renderer.draw_text(x0 + 1, y + 1, step_str, Color(0, 0, 0, 160), s)
        self.renderer.draw_text(x0, y, step_str, Color(255, 255, 200, 255), s)
        y += 20

        # Pause indicator (yellow, only when paused)
        if self.renderer.is_paused:
            self.renderer.draw_text(
                x0 + 1, y + 1, "[PAUSED]", Color(0, 0, 0, 160), s
            )
            self.renderer.draw_text(
                x0, y, "[PAUSED]", Color(255, 220, 0, 255), s
            )
            y += 20

        # Recording indicator (red, only when recording)
        if self.renderer.recorder.is_recording:
            var rec_str = (
                "* REC  " + String(self.renderer.recorder.frame_count) + "f"
            )
            self.renderer.draw_text(
                x0 + 1, y + 1, rec_str, Color(0, 0, 0, 160), s
            )
            self.renderer.draw_text(x0, y, rec_str, Color(220, 40, 40, 255), s)

    def _draw_velocity_indicator(mut self, torso_pos: Vec3, vel_x: Float64):
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
