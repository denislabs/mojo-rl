"""Physics3D model layer.

`ModelDefLike` (the model-definition trait; sole implementer =
`parser.ModelDefFromXML`), the inertia-from-geom pure-math helpers used by
the spec-direct fields build, and the generic `ModelRenderer`.

The compile-time spec layer (BodySpec/JointSpec/GeomSpec/ActuatorSpec/... +
the `ModelDef` compositor) was deleted at the G2/G4 fields sunset — every
model is XML-parsed.
"""

from .model_def import ModelDefLike
from .inertia_from_geom import geom_volume, geom_effective_mass, geom_inertia
from .model_renderer import ModelRenderer
