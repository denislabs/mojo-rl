"""HalfCheetah Renderer — thin wrapper around generic ModelRenderer.

Provides visualization of the HalfCheetah environment with color-coded
body parts. All rendering logic is handled by ModelRenderer, which
reads geometry (RADIUS, HALF_LENGTH) from GeomSpec at compile time.

Implements EnvRenderer3D trait for integration with evaluation code.
"""

from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render import Color3D
from core import EnvRenderer3D
from physics3d.model.model_renderer import ModelRenderer

from .half_cheetah_def import (
    GroundGeom,
    TorsoGeom,
    HeadGeom,
    BThighGeom,
    BShinGeom,
    BFootGeom,
    FThighGeom,
    FShinGeom,
    FFootGeom,
)

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# HalfCheetah Renderer
# =============================================================================


struct HalfCheetahRenderer(EnvRenderer3D, Movable):
    """Renderer for HalfCheetah environment.

    Thin wrapper around ModelRenderer parameterized by HalfCheetah geom types.
    """

    var inner: ModelRenderer[
        GroundGeom,
        TorsoGeom,
        HeadGeom,
        BThighGeom,
        BShinGeom,
        BFootGeom,
        FThighGeom,
        FShinGeom,
        FFootGeom,
    ]

    fn __init__(
        out self,
        width: Int = 1280,
        height: Int = 720,
        follow_cheetah: Bool = True,
        show_velocity: Bool = True,
    ) raises:
        self.inner = ModelRenderer[
            GroundGeom,
            TorsoGeom,
            HeadGeom,
            BThighGeom,
            BShinGeom,
            BFootGeom,
            FThighGeom,
            FShinGeom,
            FFootGeom,
        ](
            width=width,
            height=height,
            visual_radius_scale=2.0,
            cam_eye_y=-3.0,
            cam_eye_z=1.0,
            cam_target_z=0.5,
            axes_offset=1.5,
            vel_arrow_height=0.15,
            vel_arrow_scale=0.1,
            vel_color=Color3D(0, 255, 255),
            follow=follow_cheetah,
            show_velocity=show_velocity,
        )

    fn __moveinit__(out self, deinit other: Self):
        self.inner = other.inner^

    fn init(mut self) raises -> None:
        self.inner.init()

    fn close(mut self) raises -> None:
        self.inner.close()

    fn check_quit(mut self) -> Bool:
        return self.inner.check_quit()

    fn is_open(self) -> Bool:
        return self.inner.is_open()

    fn delay(self, ms: Int) -> None:
        self.inner.delay(ms)

    fn orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64) -> None:
        self.inner.orbit_camera(delta_theta, delta_phi)

    fn zoom_camera(mut self, delta: Float64) -> None:
        self.inner.zoom_camera(delta)

    fn render(
        mut self,
        positions: List[Vec3],
        quaternions: List[Quat],
        vel_x: Float64 = 0.0,
    ):
        self.inner.render(positions, quaternions, vel_x)
