"""Model definition package for compile-time body/joint/geom/actuator specifications.

Provides BodySpec, JointSpec, GeomSpec, and ActuatorSpec traits with concrete
implementations and variadic containers (Bodies, Joints, Geoms, Actuators)
plus a ModelDef compositor.
"""

from .body_spec import (
    BodySpec,
    CapsuleBody,
    SphereBody,
    BoxBody,
    BodiesLike,
    _EmptyBodies,
    Bodies,
)
from .joint_spec import (
    JointSpec,
    HingeJoint,
    SlideJoint,
    JointsLike,
    _EmptyJoints,
    Joints,
)
from .geom_spec import (
    GeomSpec,
    Plane,
    Sphere,
    Box,
    Capsule,
    FromToCapsule,
    GeomsLike,
    _EmptyGeoms,
    Geoms,
)
from .light_spec import (
    LightSpec,
    DirectionalLight,
    LIGHT_DIRECTIONAL,
    LIGHT_POINT,
    LightsLike,
    _EmptyLights,
    Lights,
)
from .defaults_spec import (
    ModelDefaultsLike,
    ModelDefaults,
)
from .texture_spec import (
    TextureSpec,
    CheckerTexture,
    FlatTexture,
    GradientTexture,
    TEX_CHECKER,
    TEX_FLAT,
    TEX_GRADIENT,
    TexturesLike,
    _EmptyTextures,
    Textures,
)
from .material_spec import (
    MaterialSpec,
    Material,
    DefaultMaterial,
    PlaneMaterial,
    GeomMaterial,
    MaterialsLike,
    _EmptyMaterials,
    Materials,
)
from .equality_spec import EqualitySpec, ConnectConstraint, WeldConstraint
from .site_spec import SiteSpec, Site, SitesLike, _EmptySites, Sites
from .actuator_spec import (
    ActuatorSpec,
    MotorActuator,
    PositionActuator,
    VelocityActuator,
    GeneralActuator,
    ActuatorsLike,
    _EmptyActuators,
    Actuators,
)
from .camera_spec import (
    CameraSpec,
    TrackCamera,
    FixedCamera,
    CAM_TRACKCOM,
    CAM_FIXED,
    CamerasLike,
    _EmptyCameras,
    Cameras,
)
from .model_def import ModelDef, ModelDefLike
from .inertia_from_geom import compute_inertia_from_geoms
from .model_renderer import ModelRenderer
