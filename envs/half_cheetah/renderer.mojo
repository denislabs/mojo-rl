"""HalfCheetah Renderer — type alias for ModelRenderer with HalfCheetah geoms.

All rendering logic is handled by ModelRenderer, which reads geometry
(type, radius, half_length, color) and position from GeomSpec at compile time.
"""

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

comptime HalfCheetahRenderer = ModelRenderer[
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
