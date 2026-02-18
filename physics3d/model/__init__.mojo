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
from .equality_spec import EqualitySpec, ConnectConstraint, WeldConstraint
from .actuator_spec import (
    ActuatorSpec,
    MotorActuator,
    PositionActuator,
    VelocityActuator,
    GeneralActuator,
)
from .model_def import Bodies, Joints, Geoms, Equalities, Actuators, ModelDef
from .model_renderer import ModelRenderer
