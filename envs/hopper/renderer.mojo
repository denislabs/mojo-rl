"""Hopper Renderer — thin wrapper around generic ModelRenderer.

Provides visualization of the Hopper environment with color-coded
body parts. All rendering logic is handled by ModelRenderer, which
reads geometry (RADIUS, HALF_LENGTH) and COLOR from GeomSpec at compile time.

Implements EnvRenderer3D trait for integration with evaluation code.
"""

from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from render import Color3D
from core import EnvRenderer3D
from physics3d.model.model_renderer import ModelRenderer

from .hopper_def import (
    HopperGroundGeom,
    HopperTorsoGeom,
    HopperThighGeom,
    HopperLegGeom,
    HopperFootGeom,
)

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# Hopper Renderer
# =============================================================================


struct HopperRenderer(EnvRenderer3D, Movable):
    """Renderer for Hopper environment.

    Thin wrapper around ModelRenderer parameterized by Hopper geom types.
    Colors are defined on the GeomSpec types in hopper_def.mojo.
    """

    var inner: ModelRenderer[
        HopperGroundGeom,
        HopperTorsoGeom,
        HopperThighGeom,
        HopperLegGeom,
        HopperFootGeom,
    ]

    fn __init__(
        out self,
        width: Int = 1024,
        height: Int = 576,
        follow_hopper: Bool = True,
        show_velocity: Bool = True,
    ) raises:
        self.inner = ModelRenderer[
            HopperGroundGeom,
            HopperTorsoGeom,
            HopperThighGeom,
            HopperLegGeom,
            HopperFootGeom,
        ](
            width=width,
            height=height,
            visual_radius_scale=1.5,
            cam_eye_y=-2.5,
            cam_eye_z=1.2,
            cam_target_z=0.8,
            axes_offset=0.8,
            vel_arrow_height=0.25,
            vel_arrow_scale=0.15,
            vel_color=Color3D(0, 255, 255),
            follow=follow_hopper,
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
        """Render the Hopper state.

        Args:
            positions: List of 4 body positions (torso, thigh, leg, foot).
            quaternions: List of 4 body orientations.
            vel_x: Current forward velocity (for velocity indicator).
        """
        self.inner.render(positions, quaternions, vel_x)
