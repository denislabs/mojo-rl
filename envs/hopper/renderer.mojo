"""Hopper Renderer — type alias for ModelRenderer with Hopper geoms.

All rendering logic is handled by ModelRenderer, which reads geometry
(type, radius, half_length, color) and position from GeomSpec at compile time.
"""

from physics3d.model.model_renderer import ModelRenderer

from .hopper_def import (
    HopperGroundGeom,
    HopperTorsoGeom,
    HopperThighGeom,
    HopperLegGeom,
    HopperFootGeom,
    HopperCamera,
    HopperLight,
)

comptime HopperRenderer = ModelRenderer[
    HopperGroundGeom,
    HopperTorsoGeom,
    HopperThighGeom,
    HopperLegGeom,
    HopperFootGeom,
]
