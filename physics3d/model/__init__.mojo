"""Model definition package for compile-time body/joint/geom/actuator specifications.

Provides BodySpec, JointSpec, GeomSpec, and ActuatorSpec traits with concrete
implementations and variadic containers (Bodies, Joints, Geoms, Actuators)
plus a ModelDef compositor.
"""

from .body_spec import BodySpec, CapsuleBody, SphereBody, BoxBody
from .joint_spec import JointSpec, HingeJoint, SlideJoint
from .geom_spec import (
    GeomSpec,
    Plane,
    Sphere,
    Box,
    Capsule,
    FromToCapsule,
    PlaneGeom,
    SphereGeom,
    BoxGeom,
    CapsuleGeom,
    FromToCapsuleGeom,
    BodyCapsuleGeom,
    BodySphereGeom,
    BodyBoxGeom,
)
from .texture_spec import (
    TextureSpec,
    CheckerTexture,
    FlatTexture,
    GradientTexture,
    TEX_CHECKER,
    TEX_FLAT,
    TEX_GRADIENT,
)
from .material_spec import (
    MaterialSpec,
    Material,
    DefaultMaterial,
    PlaneMaterial,
    GeomMaterial,
)
from .equality_spec import EqualitySpec, ConnectConstraint, WeldConstraint
from .actuator_spec import (
    ActuatorSpec,
    MotorActuator,
    PositionActuator,
    VelocityActuator,
    GeneralActuator,
)
from .camera_spec import CameraSpec, TrackCamera, CAM_TRACKCOM, CAM_FIXED
from .light_spec import LightSpec, DirectionalLight, LIGHT_DIRECTIONAL, LIGHT_POINT
from .model_def import Bodies, Joints, Geoms, Equalities, Actuators, Cameras, Lights, Textures, Materials, ModelDef
from .inertia_from_geom import compute_inertia_from_geoms
from .model_renderer import ModelRenderer
