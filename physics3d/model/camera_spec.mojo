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
