"""CameraSpec trait and concrete camera types for model-defined cameras.

MuJoCo XML defines cameras per-model (e.g., <camera name="track" mode="trackcom"
pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>). This module provides compile-time camera
specifications that environments use to configure their renderers.

Camera modes:
  - CAM_TRACKCOM (0): Camera follows the center of mass (torso) of the model.
  - CAM_FIXED (1): Camera stays at a fixed position in world space.

Usage:
    from physics3d.model.camera_spec import CameraSpec, TrackCamera

    # HalfCheetah camera: MuJoCo pos="0 -3 0.3"
    comptime HalfCheetahCamera = TrackCamera[pos_y=-3.0, pos_z=0.3]

    # Hopper camera: MuJoCo pos="0 -3 -0.25"
    comptime HopperCamera = TrackCamera[pos_y=-3.0, pos_z=-0.25]
"""


from render import Camera3D
from math3d import Vec3 as Vec3Generic

comptime Vec3 = Vec3Generic[DType.float64]

# Camera mode constants
comptime CAM_TRACKCOM: Int = 0
comptime CAM_FIXED: Int = 1


trait CameraSpec:
    """Compile-time specification for a camera.

    Defines camera position, target, and tracking mode. Used by ModelRenderer
    to configure the 3D camera at construction time.
    """

    comptime MODE: Int  # CAM_TRACKCOM or CAM_FIXED
    comptime POS_X: Float64
    comptime POS_Y: Float64
    comptime POS_Z: Float64
    comptime TARGET_Z: Float64  # Z height of the camera target point


@fieldwise_init
struct TrackCamera[
    pos_x: Float64 = 0.0,
    pos_y: Float64 = -3.0,
    pos_z: Float64 = 0.3,
    target_z: Float64 = 0.0,
](CameraSpec):
    """Trackcom camera that follows the model's torso.

    Matches MuJoCo's mode="trackcom" behavior: camera eye follows the torso's
    X position while maintaining fixed Y/Z offsets. Target tracks torso X at
    a configurable Z height.

    Default values match MuJoCo's common trackcom camera pattern.
    """

    comptime MODE: Int = CAM_TRACKCOM
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime TARGET_Z: Float64 = Self.target_z


# =============================================================================
# Cameras — variadic camera list (purely visual, no setup_model)
# =============================================================================


trait CamerasLike:
    """Trait for compile-time camera container types."""

    comptime N: Int

    @staticmethod
    fn setup_cameras(width: Int, height: Int) -> List[Camera3D]:
        ...


@fieldwise_init
struct Cameras[*C: CameraSpec](CamerasLike):
    """Compile-time list of camera specifications.

    Provides N (camera count) and type-level access to each camera via cam_types[i].
    Cameras are purely visual — no setup_model needed.
    """

    comptime cam_types = Variadic.types[T=CameraSpec, *Self.C]
    comptime N: Int = Variadic.size(Self.cam_types)

    @staticmethod
    fn setup_cameras(width: Int, height: Int) -> List[Camera3D]:
        var cameras = List[Camera3D]()

        @parameter
        for i in range(Self.N):
            comptime Cam = Self.cam_types[i]
            var camera = Camera3D(
                eye=Vec3(0.0, Cam.POS_Y, Cam.POS_Z),
                target=Vec3(0.0, 0.0, Cam.TARGET_Z),
                up=Vec3(0.0, 0.0, 1.0),
                fov=50.0,
                aspect=Float64(width) / Float64(height),
                near=0.1,
                far=100.0,
                screen_width=width,
                screen_height=height,
            )
            cameras.append(camera^)
        return cameras^


@fieldwise_init
struct _EmptyCameras(CamerasLike):
    comptime N: Int = 0

    @staticmethod
    fn setup_cameras(width: Int, height: Int) -> List[Camera3D]:
        return []
