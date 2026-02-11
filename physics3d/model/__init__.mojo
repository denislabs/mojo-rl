"""Model definition package for compile-time body/joint/geom specifications.

Provides BodySpec, JointSpec, and GeomSpec traits with concrete implementations
(CapsuleBody, SphereBody, BoxBody, HingeJoint, SlideJoint, PlaneGeom) and
variadic containers (Bodies, Joints, WorldBody) plus a ModelDef compositor.
"""

from .body_spec import BodySpec, CapsuleBody, SphereBody, BoxBody
from .joint_spec import JointSpec, HingeJoint, SlideJoint
from .geom_spec import GeomSpec, PlaneGeom
from .model_def import Bodies, Joints, WorldBody, ModelDef
from .model_renderer import ModelRenderer
