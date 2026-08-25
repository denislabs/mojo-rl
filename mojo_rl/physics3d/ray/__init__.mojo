"""Ray casting — `mju_rayGeom` and the routines that build on it.

A separate package from `collision/` on purpose. The two share a subject and
not a shape: `collision/` answers "do these two bodies overlap, and with what
manifold" against `Model`/`Data` records, while this answers "what does a ray
hit" for consumers that are not the solver at all — the `rangefinder` sensor,
studio picking, and the batched camera renderer. Keeping them apart is also
what lets this package stay free of `Model`, `Data` and `LayoutTensor`.
"""

from .geom import (
    RayBoxHit,
    RAY_MINVAL,
    RAY_NO_HIT,
    ray_map,
    ray_quad,
    ray_plane,
    ray_sphere,
    ray_capsule,
    ray_ellipsoid,
    ray_cylinder,
    ray_box,
    ray_box_all,
    ray_geom,
)
from .triangle import ray_basis, ray_triangle
from .hfield import ray_hfield
