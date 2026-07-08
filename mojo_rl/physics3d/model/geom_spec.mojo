"""GeomSpec trait and concrete geom types for geometry specification.

Supports both worldbody (static) and body-attached geoms via `body_idx`:
  - body_idx=0: worldbody (static geom in world frame)
  - body_idx>=1: attached to that body (pos/quat in body's local frame)

Five geom shapes:
  - Plane: Infinite ground plane (body_idx always 0)
  - Sphere: Sphere geom (body_idx=0 for static, >=1 for body-attached)
  - Capsule: Capsule geom (body_idx=0 for static, >=1 for body-attached)
  - Box: Box geom (body_idx=0 for static, >=1 for body-attached)
  - Cylinder: Cylinder geom (body_idx=0 for static, >=1 for body-attached)

Fields that use sentinel value -1.0 (Float64) or -1 (Int) mean "use
ModelDefaults". Resolution happens at Geoms.setup_model time.

Usage:
    from mojo_rl.physics3d.model.geom_spec import GeomSpec, Plane, Capsule

    # Ground plane (friction from defaults)
    comptime MyPlane = Plane[z=0.0]

    # Capsule with explicit friction override
    comptime MyCap = Capsule[body_idx=1, radius=0.046, friction=0.4]
"""
from std.math import sqrt
from ..types import ConeType
from ..constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
)
from mojo_rl.render import Color, Renderer3D
from mojo_rl.math3d import Vec3 as _Vec3G, Quat as _QuatG

comptime _RVec3 = _Vec3G[DType.float64]
comptime _RQuat = _QuatG[DType.float64]
from ..model.defaults_spec import ModelDefaults, _resolve_f64, _resolve_int
from .inertia_from_geom import geom_volume, compute_inertia_from_geoms
from ..gpu.constants import (
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X as _GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y as _GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z as _GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X as _GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y as _GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z as _GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W as _GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS as _GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH as _GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X as _GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y as _GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z as _GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION as _GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE as _GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY as _GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM as _GEOM_IDX_CONDIM,
    GEOM_IDX_FRICTION_SPIN as _GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL as _GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND as _GEOM_IDX_RBOUND,
    GEOM_IDX_SOLREF_0 as _GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1 as _GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0 as _GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1 as _GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2 as _GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3 as _GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4 as _GEOM_IDX_SOLIMP_4,
    GEOM_IDX_MARGIN as _GEOM_IDX_MARGIN,
    model_geom_offset,
)
from std.gpu.host import HostBuffer

# Sentinel values for "use model default" (re-exported for convenience)
comptime _UNSET_F64: Float64 = -1.0
comptime _UNSET_INT: Int = -1


# =============================================================================
# Compile-time math helpers for fromto conversion
# =============================================================================


def _comptime_sqrt(x: Float64) -> Float64:
    """Newton's method sqrt for compile-time evaluation.

    Needed because math.sqrt may not be available at comptime.
    Converges to machine precision in ~10 iterations for typical values.
    """
    if x <= 0.0:
        return 0.0
    # Initial guess
    var r = x
    if x > 1.0:
        r = x / 2.0
    # Newton iterations (enough for Float64 precision)
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    r = (r + x / r) * 0.5
    return r


def _fromto_center_x(from_x: Float64, to_x: Float64) -> Float64:
    return (from_x + to_x) * 0.5


def _fromto_center_y(from_y: Float64, to_y: Float64) -> Float64:
    return (from_y + to_y) * 0.5


def _fromto_center_z(from_z: Float64, to_z: Float64) -> Float64:
    return (from_z + to_z) * 0.5


def _fromto_half_length(
    from_x: Float64,
    from_y: Float64,
    from_z: Float64,
    to_x: Float64,
    to_y: Float64,
    to_z: Float64,
) -> Float64:
    var dx = to_x - from_x
    var dy = to_y - from_y
    var dz = to_z - from_z
    return _comptime_sqrt(dx * dx + dy * dy + dz * dz) * 0.5


def _fromto_quat_component(
    from_x: Float64,
    from_y: Float64,
    from_z: Float64,
    to_x: Float64,
    to_y: Float64,
    to_z: Float64,
    component: Int,
) -> Float64:
    """Compute one component of the quaternion (x,y,z,w) rotating Z-axis to the fromto direction.

    component: 0=x, 1=y, 2=z, 3=w.
    Uses the half-angle formula: q = normalize(cross(Z, d), 1 + dot(Z, d))
    = normalize(-dy, dx, 0, 1 + dz) which avoids trig functions.
    """
    var dx = to_x - from_x
    var dy = to_y - from_y
    var dz = to_z - from_z
    var length = _comptime_sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-12:
        # Degenerate: zero-length capsule, return identity
        if component == 3:
            return 1.0
        return 0.0
    # Normalized direction
    var nx = dx / length
    var ny = dy / length
    var nz = dz / length
    # Half-angle quaternion: q = normalize(cross(Z, n), 1 + dot(Z, n))
    # cross(Z, n) = (0,0,1) x (nx,ny,nz) = (-ny, nx, 0)
    # dot(Z, n) = nz
    var qx = -ny
    var qy = nx
    var _ = 0.0  # qz is always 0 (cross(Z, n) has no z component)
    var qw = 1.0 + nz
    if qw < 1e-12:
        # Direction is -Z: 180° rotation around X
        if component == 0:
            return 1.0
        return 0.0
    # Normalize
    var qlen = _comptime_sqrt(qx * qx + qy * qy + qw * qw)
    if component == 0:
        return qx / qlen
    elif component == 1:
        return qy / qlen
    elif component == 2:
        return 0.0
    else:
        return qw / qlen


trait GeomSpec:
    """Compile-time specification for a geom (static or body-attached).

    Fields with value -1.0 (Float64) or -1 (Int) are "unset" and will
    be resolved from ModelDefaults during Geoms.setup_model().
    """

    comptime GEOM_TYPE: Int  # GEOM_PLANE, GEOM_BOX, GEOM_SPHERE, GEOM_CAPSULE
    # Body index (0 for worldbody/static geoms, >= 1 for body-attached)
    comptime BODY_IDX: Int
    # Position (world frame for static, local frame in body for attached)
    comptime POS_X: Float64
    comptime POS_Y: Float64
    comptime POS_Z: Float64
    # Orientation (world frame for static, local frame in body for attached)
    comptime QUAT_X: Float64
    comptime QUAT_Y: Float64
    comptime QUAT_Z: Float64
    comptime QUAT_W: Float64
    # Size (interpretation depends on GEOM_TYPE)
    comptime SIZE_X: Float64  # plane: extent_x; box: half_x
    comptime SIZE_Y: Float64  # plane: extent_y; box: half_y
    comptime SIZE_Z: Float64  # box: half_z; capsule: half_length
    comptime RADIUS: Float64  # sphere/capsule radius
    comptime HALF_LENGTH: Float64  # capsule half-length (explicit)
    comptime HALF_X: Float64  # box half-extents
    comptime HALF_Y: Float64
    comptime HALF_Z: Float64
    # Physics (-1.0/-1 = use ModelDefaults)
    comptime FRICTION: Float64
    comptime CONDIM: Int  # Contact dimensionality: 1, 3, 4, or 6
    comptime FRICTION_SPIN: Float64  # Torsional friction coefficient
    comptime FRICTION_ROLL: Float64  # Rolling friction coefficient
    # Collision filtering (-1 = use ModelDefaults)
    comptime CONTYPE: Int
    comptime CONAFFINITY: Int
    # Per-geom solref/solimp (-1.0 = use model-level defaults)
    comptime SOLREF_0: Float64  # timeconst
    comptime SOLREF_1: Float64  # dampratio
    comptime SOLIMP_0: Float64  # dmin
    comptime SOLIMP_1: Float64  # dmax
    comptime SOLIMP_2: Float64  # width
    comptime SOLIMP_3: Float64  # midpoint
    comptime SOLIMP_4: Float64  # power
    # Contact margin (MuJoCo-style: contacts activate when dist < margin)
    comptime MARGIN: Float64
    # Mass/density (-1.0 = use model default density)
    comptime DENSITY: Float64  # kg/m³ (-1.0 = use ModelDefaults.GEOM_DENSITY)
    comptime GEOM_MASS: Float64  # kg (-1.0 = derive from density * volume)
    # Visual
    comptime COLOR: Color
    # Material properties (-1.0 = use model/material default)
    comptime SHININESS: Float64  # Specular exponent scaling (0-1)
    comptime SPECULAR: Float64  # Specular intensity (0-1)
    comptime REFLECTANCE: Float64  # Reflectance coefficient (0-1)
    # Material name reference (MuJoCo-style: geom material="name")
    comptime MATERIAL_NAME: String  # "" = no material reference


# =============================================================================
# Plane — infinite ground plane (always worldbody)
# =============================================================================


@fieldwise_init
struct Plane[
    z: Float64 = 0.0,
    friction: Float64 = _UNSET_F64,
    condim: Int = _UNSET_INT,
    friction_spin: Float64 = _UNSET_F64,
    friction_roll: Float64 = _UNSET_F64,
    contype: Int = _UNSET_INT,
    conaffinity: Int = _UNSET_INT,
    solref_0: Float64 = _UNSET_F64,
    solref_1: Float64 = _UNSET_F64,
    solimp_0: Float64 = _UNSET_F64,
    solimp_1: Float64 = _UNSET_F64,
    solimp_2: Float64 = _UNSET_F64,
    solimp_3: Float64 = _UNSET_F64,
    solimp_4: Float64 = _UNSET_F64,
    margin: Float64 = _UNSET_F64,
    size_x: Float64 = 40.0,
    size_y: Float64 = 40.0,
    shininess: Float64 = _UNSET_F64,
    specular: Float64 = _UNSET_F64,
    reflectance: Float64 = _UNSET_F64,
    material_name: String = "",
](GeomSpec):
    """Infinite horizontal plane at height z.

    Matches MuJoCo <geom type="plane"/>. The plane normal is always +Z.
    size_x/size_y define the visual extent (collision is infinite).
    """

    comptime GEOM_TYPE: Int = GEOM_PLANE
    comptime BODY_IDX: Int = 0
    comptime POS_X: Float64 = 0.0
    comptime POS_Y: Float64 = 0.0
    comptime POS_Z: Float64 = Self.z
    comptime QUAT_X: Float64 = 0.0
    comptime QUAT_Y: Float64 = 0.0
    comptime QUAT_Z: Float64 = 0.0
    comptime QUAT_W: Float64 = 1.0
    comptime SIZE_X: Float64 = Self.size_x
    comptime SIZE_Y: Float64 = Self.size_y
    comptime SIZE_Z: Float64 = 0.0
    comptime RADIUS: Float64 = 0.0
    comptime HALF_LENGTH: Float64 = 0.0
    comptime HALF_X: Float64 = 0.0
    comptime HALF_Y: Float64 = 0.0
    comptime HALF_Z: Float64 = 0.0
    comptime FRICTION: Float64 = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Float64 = Self.friction_spin
    comptime FRICTION_ROLL: Float64 = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime MARGIN: Float64 = Self.margin
    comptime DENSITY: Float64 = 0.0  # Planes have no mass
    comptime GEOM_MASS: Float64 = 0.0  # Planes have no mass
    comptime COLOR: Color = Color(128, 128, 128, 255)
    comptime SHININESS: Float64 = Self.shininess
    comptime SPECULAR: Float64 = Self.specular
    comptime REFLECTANCE: Float64 = Self.reflectance
    comptime MATERIAL_NAME: String = Self.material_name


# =============================================================================
# Sphere — static (body_idx=0) or body-attached (body_idx>=1)
# =============================================================================


@fieldwise_init
struct Sphere[
    body_idx: Int = 0,
    radius: Float64 = 0.5,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    friction: Float64 = _UNSET_F64,
    condim: Int = _UNSET_INT,
    friction_spin: Float64 = _UNSET_F64,
    friction_roll: Float64 = _UNSET_F64,
    contype: Int = _UNSET_INT,
    conaffinity: Int = _UNSET_INT,
    solref_0: Float64 = _UNSET_F64,
    solref_1: Float64 = _UNSET_F64,
    solimp_0: Float64 = _UNSET_F64,
    solimp_1: Float64 = _UNSET_F64,
    solimp_2: Float64 = _UNSET_F64,
    solimp_3: Float64 = _UNSET_F64,
    solimp_4: Float64 = _UNSET_F64,
    margin: Float64 = _UNSET_F64,
    density: Float64 = _UNSET_F64,
    mass: Float64 = _UNSET_F64,
    color: Color = Color(100, 100, 200, 255),
    shininess: Float64 = _UNSET_F64,
    specular: Float64 = _UNSET_F64,
    reflectance: Float64 = _UNSET_F64,
    material_name: String = "",
](GeomSpec):
    """Sphere geom (static or body-attached).

    body_idx=0: static in world frame (pos is world position).
    body_idx>=1: attached to body (pos is local offset in body frame).
    Matches MuJoCo <geom type="sphere" size="radius"/>.
    """

    comptime GEOM_TYPE: Int = GEOM_SPHERE
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime QUAT_X: Float64 = 0.0
    comptime QUAT_Y: Float64 = 0.0
    comptime QUAT_Z: Float64 = 0.0
    comptime QUAT_W: Float64 = 1.0
    comptime SIZE_X: Float64 = 0.0
    comptime SIZE_Y: Float64 = 0.0
    comptime SIZE_Z: Float64 = 0.0
    comptime RADIUS: Float64 = Self.radius
    comptime HALF_LENGTH: Float64 = 0.0
    comptime HALF_X: Float64 = 0.0
    comptime HALF_Y: Float64 = 0.0
    comptime HALF_Z: Float64 = 0.0
    comptime FRICTION: Float64 = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Float64 = Self.friction_spin
    comptime FRICTION_ROLL: Float64 = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime MARGIN: Float64 = Self.margin
    comptime DENSITY: Float64 = Self.density
    comptime GEOM_MASS: Float64 = Self.mass
    comptime COLOR: Color = Self.color
    comptime SHININESS: Float64 = Self.shininess
    comptime SPECULAR: Float64 = Self.specular
    comptime REFLECTANCE: Float64 = Self.reflectance
    comptime MATERIAL_NAME: String = Self.material_name


# =============================================================================
# Capsule — static (body_idx=0) or body-attached (body_idx>=1)
# =============================================================================


@fieldwise_init
struct Capsule[
    body_idx: Int = 0,
    radius: Float64 = 0.25,
    half_length: Float64 = 0.5,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    friction: Float64 = _UNSET_F64,
    condim: Int = _UNSET_INT,
    friction_spin: Float64 = _UNSET_F64,
    friction_roll: Float64 = _UNSET_F64,
    contype: Int = _UNSET_INT,
    conaffinity: Int = _UNSET_INT,
    solref_0: Float64 = _UNSET_F64,
    solref_1: Float64 = _UNSET_F64,
    solimp_0: Float64 = _UNSET_F64,
    solimp_1: Float64 = _UNSET_F64,
    solimp_2: Float64 = _UNSET_F64,
    solimp_3: Float64 = _UNSET_F64,
    solimp_4: Float64 = _UNSET_F64,
    margin: Float64 = _UNSET_F64,
    density: Float64 = _UNSET_F64,
    mass: Float64 = _UNSET_F64,
    color: Color = Color(204, 153, 102, 255),
    shininess: Float64 = _UNSET_F64,
    specular: Float64 = _UNSET_F64,
    reflectance: Float64 = _UNSET_F64,
    material_name: String = "",
](GeomSpec):
    """Capsule geom (static or body-attached).

    body_idx=0: static in world frame (pos/quat is world pose).
    body_idx>=1: attached to body (pos/quat is local offset in body frame).
    Capsule axis is local Z, half_length defines the cylinder half-length
    (total capsule length = 2*half_length + 2*radius).
    Matches MuJoCo <geom type="capsule" size="radius hlength"/>.
    """

    comptime GEOM_TYPE: Int = GEOM_CAPSULE
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime QUAT_X: Float64 = Self.quat_x
    comptime QUAT_Y: Float64 = Self.quat_y
    comptime QUAT_Z: Float64 = Self.quat_z
    comptime QUAT_W: Float64 = Self.quat_w
    comptime SIZE_X: Float64 = 0.0
    comptime SIZE_Y: Float64 = 0.0
    comptime SIZE_Z: Float64 = Self.half_length
    comptime RADIUS: Float64 = Self.radius
    comptime HALF_LENGTH: Float64 = Self.half_length
    comptime HALF_X: Float64 = 0.0
    comptime HALF_Y: Float64 = 0.0
    comptime HALF_Z: Float64 = 0.0
    comptime FRICTION: Float64 = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Float64 = Self.friction_spin
    comptime FRICTION_ROLL: Float64 = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime MARGIN: Float64 = Self.margin
    comptime DENSITY: Float64 = Self.density
    comptime GEOM_MASS: Float64 = Self.mass
    comptime COLOR: Color = Self.color
    comptime SHININESS: Float64 = Self.shininess
    comptime SPECULAR: Float64 = Self.specular
    comptime REFLECTANCE: Float64 = Self.reflectance
    comptime MATERIAL_NAME: String = Self.material_name


# =============================================================================
# FromToCapsule — capsule defined by two endpoints (MuJoCo fromto="...")
# =============================================================================


@fieldwise_init
struct FromToCapsule[
    body_idx: Int = 0,
    radius: Float64 = 0.25,
    from_x: Float64 = 0.0,
    from_y: Float64 = 0.0,
    from_z: Float64 = 0.0,
    to_x: Float64 = 0.0,
    to_y: Float64 = 0.0,
    to_z: Float64 = 0.0,
    friction: Float64 = _UNSET_F64,
    condim: Int = _UNSET_INT,
    friction_spin: Float64 = _UNSET_F64,
    friction_roll: Float64 = _UNSET_F64,
    contype: Int = _UNSET_INT,
    conaffinity: Int = _UNSET_INT,
    solref_0: Float64 = _UNSET_F64,
    solref_1: Float64 = _UNSET_F64,
    solimp_0: Float64 = _UNSET_F64,
    solimp_1: Float64 = _UNSET_F64,
    solimp_2: Float64 = _UNSET_F64,
    solimp_3: Float64 = _UNSET_F64,
    solimp_4: Float64 = _UNSET_F64,
    margin: Float64 = _UNSET_F64,
    density: Float64 = _UNSET_F64,
    mass: Float64 = _UNSET_F64,
    color: Color = Color(204, 153, 102, 255),
    shininess: Float64 = _UNSET_F64,
    specular: Float64 = _UNSET_F64,
    reflectance: Float64 = _UNSET_F64,
    material_name: String = "",
](GeomSpec):
    """Capsule defined by two endpoints, matching MuJoCo's fromto="x1 y1 z1 x2 y2 z2".

    Automatically computes center position, half-length, and orientation quaternion
    from the two endpoints at compile time. The capsule axis runs from `from` to `to`.

    Example (MuJoCo XML: fromto="-.5 0 0 .5 0 0"):
        comptime TorsoGeom = FromToCapsule[
            body_idx=1, radius=0.046,
            from_x=-0.5, to_x=0.5,  # Y and Z default to 0
        ]
    """

    comptime GEOM_TYPE: Int = GEOM_CAPSULE
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = _fromto_center_x(Self.from_x, Self.to_x)
    comptime POS_Y: Float64 = _fromto_center_y(Self.from_y, Self.to_y)
    comptime POS_Z: Float64 = _fromto_center_z(Self.from_z, Self.to_z)
    comptime QUAT_X: Float64 = _fromto_quat_component(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
        0,
    )
    comptime QUAT_Y: Float64 = _fromto_quat_component(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
        1,
    )
    comptime QUAT_Z: Float64 = _fromto_quat_component(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
        2,
    )
    comptime QUAT_W: Float64 = _fromto_quat_component(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
        3,
    )
    comptime SIZE_X: Float64 = 0.0
    comptime SIZE_Y: Float64 = 0.0
    comptime SIZE_Z: Float64 = _fromto_half_length(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
    )
    comptime RADIUS: Float64 = Self.radius
    comptime HALF_LENGTH: Float64 = Self.SIZE_Z
    comptime HALF_X: Float64 = 0.0
    comptime HALF_Y: Float64 = 0.0
    comptime HALF_Z: Float64 = 0.0
    comptime FRICTION: Float64 = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Float64 = Self.friction_spin
    comptime FRICTION_ROLL: Float64 = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime MARGIN: Float64 = Self.margin
    comptime DENSITY: Float64 = Self.density
    comptime GEOM_MASS: Float64 = Self.mass
    comptime COLOR: Color = Self.color
    comptime SHININESS: Float64 = Self.shininess
    comptime SPECULAR: Float64 = Self.specular
    comptime REFLECTANCE: Float64 = Self.reflectance
    comptime MATERIAL_NAME: String = Self.material_name


# =============================================================================
# Box — static (body_idx=0) or body-attached (body_idx>=1)
# =============================================================================


@fieldwise_init
struct Box[
    body_idx: Int = 0,
    half_x: Float64 = 0.5,
    half_y: Float64 = 0.5,
    half_z: Float64 = 0.5,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    friction: Float64 = _UNSET_F64,
    condim: Int = _UNSET_INT,
    friction_spin: Float64 = _UNSET_F64,
    friction_roll: Float64 = _UNSET_F64,
    contype: Int = _UNSET_INT,
    conaffinity: Int = _UNSET_INT,
    solref_0: Float64 = _UNSET_F64,
    solref_1: Float64 = _UNSET_F64,
    solimp_0: Float64 = _UNSET_F64,
    solimp_1: Float64 = _UNSET_F64,
    solimp_2: Float64 = _UNSET_F64,
    solimp_3: Float64 = _UNSET_F64,
    solimp_4: Float64 = _UNSET_F64,
    margin: Float64 = _UNSET_F64,
    density: Float64 = _UNSET_F64,
    mass: Float64 = _UNSET_F64,
    color: Color = Color(100, 200, 100, 255),
    shininess: Float64 = _UNSET_F64,
    specular: Float64 = _UNSET_F64,
    reflectance: Float64 = _UNSET_F64,
    material_name: String = "",
](GeomSpec):
    """Box geom (static or body-attached).

    body_idx=0: static in world frame (pos/quat is world pose).
    body_idx>=1: attached to body (pos/quat is local offset in body frame).
    Matches MuJoCo <geom type="box" size="hx hy hz"/>.
    """

    comptime GEOM_TYPE: Int = GEOM_BOX
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime QUAT_X: Float64 = Self.quat_x
    comptime QUAT_Y: Float64 = Self.quat_y
    comptime QUAT_Z: Float64 = Self.quat_z
    comptime QUAT_W: Float64 = Self.quat_w
    comptime SIZE_X: Float64 = Self.half_x
    comptime SIZE_Y: Float64 = Self.half_y
    comptime SIZE_Z: Float64 = Self.half_z
    comptime RADIUS: Float64 = 0.0
    comptime HALF_LENGTH: Float64 = 0.0
    comptime HALF_X: Float64 = Self.half_x
    comptime HALF_Y: Float64 = Self.half_y
    comptime HALF_Z: Float64 = Self.half_z
    comptime FRICTION: Float64 = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Float64 = Self.friction_spin
    comptime FRICTION_ROLL: Float64 = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime MARGIN: Float64 = Self.margin
    comptime DENSITY: Float64 = Self.density
    comptime GEOM_MASS: Float64 = Self.mass
    comptime COLOR: Color = Self.color
    comptime SHININESS: Float64 = Self.shininess
    comptime SPECULAR: Float64 = Self.specular
    comptime REFLECTANCE: Float64 = Self.reflectance
    comptime MATERIAL_NAME: String = Self.material_name


# =============================================================================
# Cylinder — static (body_idx=0) or body-attached (body_idx>=1)
# =============================================================================


@fieldwise_init
struct Cylinder[
    body_idx: Int = 0,
    radius: Float64 = 0.25,
    half_length: Float64 = 0.5,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    friction: Float64 = _UNSET_F64,
    condim: Int = _UNSET_INT,
    friction_spin: Float64 = _UNSET_F64,
    friction_roll: Float64 = _UNSET_F64,
    contype: Int = _UNSET_INT,
    conaffinity: Int = _UNSET_INT,
    solref_0: Float64 = _UNSET_F64,
    solref_1: Float64 = _UNSET_F64,
    solimp_0: Float64 = _UNSET_F64,
    solimp_1: Float64 = _UNSET_F64,
    solimp_2: Float64 = _UNSET_F64,
    solimp_3: Float64 = _UNSET_F64,
    solimp_4: Float64 = _UNSET_F64,
    margin: Float64 = _UNSET_F64,
    density: Float64 = _UNSET_F64,
    mass: Float64 = _UNSET_F64,
    color: Color = Color(180, 120, 60, 255),
    shininess: Float64 = _UNSET_F64,
    specular: Float64 = _UNSET_F64,
    reflectance: Float64 = _UNSET_F64,
    material_name: String = "",
](GeomSpec):
    """Cylinder geom (static or body-attached).

    body_idx=0: static in world frame (pos/quat is world pose).
    body_idx>=1: attached to body (pos/quat is local offset in body frame).
    Cylinder axis is local Z, half_length defines the half-height.
    Like a capsule but with flat ends instead of hemispherical caps.
    Matches MuJoCo <geom type="cylinder" size="radius hlength"/>.
    """

    comptime GEOM_TYPE: Int = GEOM_CYLINDER
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime QUAT_X: Float64 = Self.quat_x
    comptime QUAT_Y: Float64 = Self.quat_y
    comptime QUAT_Z: Float64 = Self.quat_z
    comptime QUAT_W: Float64 = Self.quat_w
    comptime SIZE_X: Float64 = 0.0
    comptime SIZE_Y: Float64 = 0.0
    comptime SIZE_Z: Float64 = Self.half_length
    comptime RADIUS: Float64 = Self.radius
    comptime HALF_LENGTH: Float64 = Self.half_length
    comptime HALF_X: Float64 = 0.0
    comptime HALF_Y: Float64 = 0.0
    comptime HALF_Z: Float64 = 0.0
    comptime FRICTION: Float64 = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Float64 = Self.friction_spin
    comptime FRICTION_ROLL: Float64 = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime MARGIN: Float64 = Self.margin
    comptime DENSITY: Float64 = Self.density
    comptime GEOM_MASS: Float64 = Self.mass
    comptime COLOR: Color = Self.color
    comptime SHININESS: Float64 = Self.shininess
    comptime SPECULAR: Float64 = Self.specular
    comptime REFLECTANCE: Float64 = Self.reflectance
    comptime MATERIAL_NAME: String = Self.material_name


# =============================================================================
# FromToCylinder — cylinder defined by two endpoints (MuJoCo fromto="...")
# =============================================================================


@fieldwise_init
struct FromToCylinder[
    body_idx: Int = 0,
    radius: Float64 = 0.25,
    from_x: Float64 = 0.0,
    from_y: Float64 = 0.0,
    from_z: Float64 = 0.0,
    to_x: Float64 = 0.0,
    to_y: Float64 = 0.0,
    to_z: Float64 = 0.0,
    friction: Float64 = _UNSET_F64,
    condim: Int = _UNSET_INT,
    friction_spin: Float64 = _UNSET_F64,
    friction_roll: Float64 = _UNSET_F64,
    contype: Int = _UNSET_INT,
    conaffinity: Int = _UNSET_INT,
    solref_0: Float64 = _UNSET_F64,
    solref_1: Float64 = _UNSET_F64,
    solimp_0: Float64 = _UNSET_F64,
    solimp_1: Float64 = _UNSET_F64,
    solimp_2: Float64 = _UNSET_F64,
    solimp_3: Float64 = _UNSET_F64,
    solimp_4: Float64 = _UNSET_F64,
    margin: Float64 = _UNSET_F64,
    density: Float64 = _UNSET_F64,
    mass: Float64 = _UNSET_F64,
    color: Color = Color(180, 120, 60, 255),
    shininess: Float64 = _UNSET_F64,
    specular: Float64 = _UNSET_F64,
    reflectance: Float64 = _UNSET_F64,
    material_name: String = "",
](GeomSpec):
    """Cylinder defined by two endpoints, matching MuJoCo's fromto="x1 y1 z1 x2 y2 z2".

    Automatically computes center position, half-length, and orientation quaternion
    from the two endpoints at compile time. The cylinder axis runs from `from` to `to`.
    """

    comptime GEOM_TYPE: Int = GEOM_CYLINDER
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = _fromto_center_x(Self.from_x, Self.to_x)
    comptime POS_Y: Float64 = _fromto_center_y(Self.from_y, Self.to_y)
    comptime POS_Z: Float64 = _fromto_center_z(Self.from_z, Self.to_z)
    comptime QUAT_X: Float64 = _fromto_quat_component(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
        0,
    )
    comptime QUAT_Y: Float64 = _fromto_quat_component(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
        1,
    )
    comptime QUAT_Z: Float64 = _fromto_quat_component(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
        2,
    )
    comptime QUAT_W: Float64 = _fromto_quat_component(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
        3,
    )
    comptime SIZE_X: Float64 = 0.0
    comptime SIZE_Y: Float64 = 0.0
    comptime SIZE_Z: Float64 = _fromto_half_length(
        Self.from_x,
        Self.from_y,
        Self.from_z,
        Self.to_x,
        Self.to_y,
        Self.to_z,
    )
    comptime RADIUS: Float64 = Self.radius
    comptime HALF_LENGTH: Float64 = Self.SIZE_Z
    comptime HALF_X: Float64 = 0.0
    comptime HALF_Y: Float64 = 0.0
    comptime HALF_Z: Float64 = 0.0
    comptime FRICTION: Float64 = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Float64 = Self.friction_spin
    comptime FRICTION_ROLL: Float64 = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime SOLREF_0: Float64 = Self.solref_0
    comptime SOLREF_1: Float64 = Self.solref_1
    comptime SOLIMP_0: Float64 = Self.solimp_0
    comptime SOLIMP_1: Float64 = Self.solimp_1
    comptime SOLIMP_2: Float64 = Self.solimp_2
    comptime SOLIMP_3: Float64 = Self.solimp_3
    comptime SOLIMP_4: Float64 = Self.solimp_4
    comptime MARGIN: Float64 = Self.margin
    comptime DENSITY: Float64 = Self.density
    comptime GEOM_MASS: Float64 = Self.mass
    comptime COLOR: Color = Self.color
    comptime SHININESS: Float64 = Self.shininess
    comptime SPECULAR: Float64 = Self.specular
    comptime REFLECTANCE: Float64 = Self.reflectance
    comptime MATERIAL_NAME: String = Self.material_name


trait GeomsLike:
    """Trait for compile-time geom container types."""

    comptime N: Int

    @staticmethod
    def write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        NJOINT: Int,
        Defaults: ModelDefaultsLike,
    ](buffer: HostBuffer[DTYPE]):
        ...

    @staticmethod
    def compute_geom_masses[
        DTYPE: DType,
        Defaults: ModelDefaultsLike,
    ]() -> InlineArray[Scalar[DTYPE], Self.N]:
        ...

    @staticmethod
    def setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int,
        Defaults: ModelDefaultsLike,
        NSITE: Int = 0,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        ...

    @staticmethod
    def get_ground_rgba() -> List[Float64]:
        """Return [r, g, b] of the first plane geom's color, or empty list."""
        ...

    @staticmethod
    def render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        """Draw plane geoms as ground grids, or a fallback grid if no planes."""
        ...

    @staticmethod
    def render_body_geoms(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        """Draw all body-attached geoms (capsule, sphere, box, cylinder)."""
        ...


@fieldwise_init
struct _EmptyGeoms(GeomsLike):
    comptime N: Int = 0

    @staticmethod
    def write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        NJOINT: Int,
        Defaults: ModelDefaultsLike,
    ](buffer: HostBuffer[DTYPE]):
        pass

    @staticmethod
    def compute_geom_masses[
        DTYPE: DType,
        Defaults: ModelDefaultsLike,
    ]() -> InlineArray[Scalar[DTYPE], Self.N]:
        return InlineArray[Scalar[DTYPE], Self.N](fill=Scalar[DTYPE](0))

    @staticmethod
    def setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int,
        Defaults: ModelDefaultsLike,
        NSITE: Int = 0,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        pass

    @staticmethod
    def get_ground_rgba() -> List[Float64]:
        return List[Float64]()

    @staticmethod
    def render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        pass

    @staticmethod
    def render_body_geoms(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        pass


@fieldwise_init
struct Geoms[*G: GeomSpec](GeomsLike):
    """Compile-time list of geom specifications (static + body-attached).

    Provides N (total geom count), type-level access via geom_types[i],
    and helper counts for static vs dynamic geoms.
    """

    comptime geom_types = Self.G
    comptime N: Int = Self.geom_types.size

    @staticmethod
    def _count_static_geoms() -> Int:
        """Count of static (worldbody) geoms (BODY_IDX == 0)."""
        var total = 0

        comptime for i in range(Self.N):
            comptime if Self.geom_types[i].BODY_IDX == 0:
                total += 1
        return total

    @staticmethod
    def _count_plane_geoms() -> Int:
        """Count of plane geoms (GEOM_TYPE == GEOM_PLANE)."""
        var total = 0

        comptime for i in range(Self.N):
            comptime if Self.geom_types[i].GEOM_TYPE == GEOM_PLANE:
                total += 1
        return total

    @staticmethod
    def setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        Defaults: ModelDefaultsLike = ModelDefaults[],
        NSITE: Int = 0,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        """Populate model geom arrays from compile-time GeomSpec list.

        Resolves sentinel values (-1.0/-1) from ModelDefaults.
        Sets geom type, body index, position, orientation, size, collision
        filtering, friction, and per-geom solref/solimp.
        """

        comptime for i in range(Self.N):
            comptime G_item = Self.geom_types[i]

            # Geom arrays
            model.geom_type[i] = G_item.GEOM_TYPE
            model.geom_body[i] = G_item.BODY_IDX
            model.geom_pos[i * 3 + 0] = Scalar[DTYPE](G_item.POS_X)
            model.geom_pos[i * 3 + 1] = Scalar[DTYPE](G_item.POS_Y)
            model.geom_pos[i * 3 + 2] = Scalar[DTYPE](G_item.POS_Z)
            model.geom_quat[i * 4 + 0] = Scalar[DTYPE](G_item.QUAT_X)
            model.geom_quat[i * 4 + 1] = Scalar[DTYPE](G_item.QUAT_Y)
            model.geom_quat[i * 4 + 2] = Scalar[DTYPE](G_item.QUAT_Z)
            model.geom_quat[i * 4 + 3] = Scalar[DTYPE](G_item.QUAT_W)
            model.geom_radius[i] = Scalar[DTYPE](G_item.RADIUS)
            model.geom_half_length[i] = Scalar[DTYPE](G_item.HALF_LENGTH)
            model.geom_half_x[i] = Scalar[DTYPE](G_item.HALF_X)
            model.geom_half_y[i] = Scalar[DTYPE](G_item.HALF_Y)
            model.geom_half_z[i] = Scalar[DTYPE](G_item.HALF_Z)

            # Resolve physics fields via defaults
            model.geom_friction[i] = Scalar[DTYPE](
                _resolve_f64[G_item.FRICTION, Defaults.GEOM_FRICTION]()
            )
            model.geom_condim[i] = _resolve_int[
                G_item.CONDIM, Defaults.GEOM_CONDIM
            ]()
            model.geom_friction_spin[i] = Scalar[DTYPE](
                _resolve_f64[
                    G_item.FRICTION_SPIN, Defaults.GEOM_FRICTION_SPIN
                ]()
            )
            model.geom_friction_roll[i] = Scalar[DTYPE](
                _resolve_f64[
                    G_item.FRICTION_ROLL, Defaults.GEOM_FRICTION_ROLL
                ]()
            )
            model.geom_contype[i] = _resolve_int[
                G_item.CONTYPE, Defaults.GEOM_CONTYPE
            ]()
            model.geom_conaffinity[i] = _resolve_int[
                G_item.CONAFFINITY, Defaults.GEOM_CONAFFINITY
            ]()

            # Per-geom solref/solimp (resolved from defaults)
            model.geom_solref[i * 2 + 0] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLREF_0, Defaults.GEOM_SOLREF_0]()
            )
            model.geom_solref[i * 2 + 1] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLREF_1, Defaults.GEOM_SOLREF_1]()
            )
            model.geom_solimp[i * 5 + 0] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_0, Defaults.GEOM_SOLIMP_0]()
            )
            model.geom_solimp[i * 5 + 1] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_1, Defaults.GEOM_SOLIMP_1]()
            )
            model.geom_solimp[i * 5 + 2] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_2, Defaults.GEOM_SOLIMP_2]()
            )
            model.geom_solimp[i * 5 + 3] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_3, Defaults.GEOM_SOLIMP_3]()
            )
            model.geom_solimp[i * 5 + 4] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_4, Defaults.GEOM_SOLIMP_4]()
            )

            # Contact margin (resolved from defaults)
            model.geom_margin[i] = Scalar[DTYPE](
                _resolve_f64[G_item.MARGIN, Defaults.GEOM_MARGIN]()
            )

            # Compute bounding sphere radius
            comptime if G_item.GEOM_TYPE == GEOM_SPHERE:
                model.geom_rbound[i] = Scalar[DTYPE](G_item.RADIUS)
            elif G_item.GEOM_TYPE == GEOM_CAPSULE:
                model.geom_rbound[i] = Scalar[DTYPE](
                    G_item.HALF_LENGTH
                ) + Scalar[DTYPE](G_item.RADIUS)
            elif G_item.GEOM_TYPE == GEOM_CYLINDER:
                # Corner of cylinder to center: sqrt(half_length^2 + radius^2)
                model.geom_rbound[i] = sqrt(
                    Scalar[DTYPE](G_item.HALF_LENGTH)
                    * Scalar[DTYPE](G_item.HALF_LENGTH)
                    + Scalar[DTYPE](G_item.RADIUS)
                    * Scalar[DTYPE](G_item.RADIUS)
                )
            elif G_item.GEOM_TYPE == GEOM_BOX:
                model.geom_rbound[i] = sqrt(
                    Scalar[DTYPE](G_item.HALF_X) * Scalar[DTYPE](G_item.HALF_X)
                    + Scalar[DTYPE](G_item.HALF_Y)
                    * Scalar[DTYPE](G_item.HALF_Y)
                    + Scalar[DTYPE](G_item.HALF_Z)
                    * Scalar[DTYPE](G_item.HALF_Z)
                )
            elif G_item.GEOM_TYPE == GEOM_PLANE:
                model.geom_rbound[i] = Scalar[DTYPE](
                    1e10
                )  # Planes are infinite

            # Compute and store per-geom mass
            # Priority: explicit mass > explicit density > default density
            comptime if G_item.GEOM_TYPE == GEOM_PLANE:
                model.geom_mass[i] = Scalar[DTYPE](0)
            elif G_item.GEOM_MASS >= 0.0:
                # Explicit mass on the geom
                model.geom_mass[i] = Scalar[DTYPE](G_item.GEOM_MASS)
            else:
                # Compute volume
                var vol = geom_volume[DTYPE](
                    G_item.GEOM_TYPE,
                    Scalar[DTYPE](G_item.RADIUS),
                    Scalar[DTYPE](G_item.HALF_LENGTH),
                    Scalar[DTYPE](G_item.HALF_X),
                    Scalar[DTYPE](G_item.HALF_Y),
                    Scalar[DTYPE](G_item.HALF_Z),
                )

                comptime if G_item.DENSITY >= 0.0:
                    # Explicit density on the geom
                    model.geom_mass[i] = Scalar[DTYPE](G_item.DENSITY) * vol
                else:
                    # Use default density
                    model.geom_mass[i] = (
                        Scalar[DTYPE](Defaults.GEOM_DENSITY) * vol
                    )

    @staticmethod
    def write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        NJOINT: Int,
        Defaults: ModelDefaultsLike = ModelDefaults[],
    ](buffer: HostBuffer[DTYPE]):
        """Write geom data directly to GPU HostBuffer (no Model struct).

        Resolves sentinel values (-1.0/-1) from Defaults.
        Computes rbound and geom_mass per geom type.
        """

        comptime for i in range(Self.N):
            comptime G_item = Self.geom_types[i]
            var off = model_geom_offset[NBODY, NJOINT](i)

            buffer[off + GEOM_IDX_TYPE] = Scalar[DTYPE](G_item.GEOM_TYPE)
            buffer[off + GEOM_IDX_BODY] = Scalar[DTYPE](G_item.BODY_IDX)
            buffer[off + _GEOM_IDX_POS_X] = Scalar[DTYPE](G_item.POS_X)
            buffer[off + _GEOM_IDX_POS_Y] = Scalar[DTYPE](G_item.POS_Y)
            buffer[off + _GEOM_IDX_POS_Z] = Scalar[DTYPE](G_item.POS_Z)
            buffer[off + _GEOM_IDX_QUAT_X] = Scalar[DTYPE](G_item.QUAT_X)
            buffer[off + _GEOM_IDX_QUAT_Y] = Scalar[DTYPE](G_item.QUAT_Y)
            buffer[off + _GEOM_IDX_QUAT_Z] = Scalar[DTYPE](G_item.QUAT_Z)
            buffer[off + _GEOM_IDX_QUAT_W] = Scalar[DTYPE](G_item.QUAT_W)
            buffer[off + _GEOM_IDX_RADIUS] = Scalar[DTYPE](G_item.RADIUS)
            buffer[off + _GEOM_IDX_HALF_LENGTH] = Scalar[DTYPE](
                G_item.HALF_LENGTH
            )
            buffer[off + _GEOM_IDX_HALF_X] = Scalar[DTYPE](G_item.HALF_X)
            buffer[off + _GEOM_IDX_HALF_Y] = Scalar[DTYPE](G_item.HALF_Y)
            buffer[off + _GEOM_IDX_HALF_Z] = Scalar[DTYPE](G_item.HALF_Z)

            # Resolve physics fields via defaults
            buffer[off + _GEOM_IDX_FRICTION] = Scalar[DTYPE](
                _resolve_f64[G_item.FRICTION, Defaults.GEOM_FRICTION]()
            )
            buffer[off + _GEOM_IDX_CONTYPE] = Scalar[DTYPE](
                _resolve_int[G_item.CONTYPE, Defaults.GEOM_CONTYPE]()
            )
            buffer[off + _GEOM_IDX_CONAFFINITY] = Scalar[DTYPE](
                _resolve_int[G_item.CONAFFINITY, Defaults.GEOM_CONAFFINITY]()
            )
            buffer[off + _GEOM_IDX_CONDIM] = Scalar[DTYPE](
                _resolve_int[G_item.CONDIM, Defaults.GEOM_CONDIM]()
            )
            buffer[off + _GEOM_IDX_FRICTION_SPIN] = Scalar[DTYPE](
                _resolve_f64[
                    G_item.FRICTION_SPIN, Defaults.GEOM_FRICTION_SPIN
                ]()
            )
            buffer[off + _GEOM_IDX_FRICTION_ROLL] = Scalar[DTYPE](
                _resolve_f64[
                    G_item.FRICTION_ROLL, Defaults.GEOM_FRICTION_ROLL
                ]()
            )

            # Per-geom solref/solimp
            buffer[off + _GEOM_IDX_SOLREF_0] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLREF_0, Defaults.GEOM_SOLREF_0]()
            )
            buffer[off + _GEOM_IDX_SOLREF_1] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLREF_1, Defaults.GEOM_SOLREF_1]()
            )
            buffer[off + _GEOM_IDX_SOLIMP_0] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_0, Defaults.GEOM_SOLIMP_0]()
            )
            buffer[off + _GEOM_IDX_SOLIMP_1] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_1, Defaults.GEOM_SOLIMP_1]()
            )
            buffer[off + _GEOM_IDX_SOLIMP_2] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_2, Defaults.GEOM_SOLIMP_2]()
            )
            buffer[off + _GEOM_IDX_SOLIMP_3] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_3, Defaults.GEOM_SOLIMP_3]()
            )
            buffer[off + _GEOM_IDX_SOLIMP_4] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_4, Defaults.GEOM_SOLIMP_4]()
            )
            buffer[off + _GEOM_IDX_MARGIN] = Scalar[DTYPE](
                _resolve_f64[G_item.MARGIN, Defaults.GEOM_MARGIN]()
            )

            # Compute bounding sphere radius
            comptime if G_item.GEOM_TYPE == GEOM_SPHERE:
                buffer[off + _GEOM_IDX_RBOUND] = Scalar[DTYPE](G_item.RADIUS)
            elif G_item.GEOM_TYPE == GEOM_CAPSULE:
                buffer[off + _GEOM_IDX_RBOUND] = Scalar[DTYPE](
                    G_item.HALF_LENGTH
                ) + Scalar[DTYPE](G_item.RADIUS)
            elif G_item.GEOM_TYPE == GEOM_CYLINDER:
                buffer[off + _GEOM_IDX_RBOUND] = sqrt(
                    Scalar[DTYPE](G_item.HALF_LENGTH)
                    * Scalar[DTYPE](G_item.HALF_LENGTH)
                    + Scalar[DTYPE](G_item.RADIUS)
                    * Scalar[DTYPE](G_item.RADIUS)
                )
            elif G_item.GEOM_TYPE == GEOM_BOX:
                buffer[off + _GEOM_IDX_RBOUND] = sqrt(
                    Scalar[DTYPE](G_item.HALF_X) * Scalar[DTYPE](G_item.HALF_X)
                    + Scalar[DTYPE](G_item.HALF_Y)
                    * Scalar[DTYPE](G_item.HALF_Y)
                    + Scalar[DTYPE](G_item.HALF_Z)
                    * Scalar[DTYPE](G_item.HALF_Z)
                )
            elif G_item.GEOM_TYPE == GEOM_PLANE:
                buffer[off + _GEOM_IDX_RBOUND] = Scalar[DTYPE](1e10)

    @staticmethod
    def compute_geom_masses[
        DTYPE: DType,
        Defaults: ModelDefaultsLike = ModelDefaults[],
    ]() -> InlineArray[Scalar[DTYPE], Self.N]:
        """Compute geom masses from compile-time specs.

        Returns an InlineArray of per-geom masses matching the order in
        the Geoms list. Priority: explicit mass > explicit density > default density.
        """
        var masses = InlineArray[Scalar[DTYPE], Self.N](fill=Scalar[DTYPE](0))

        comptime for i in range(Self.N):
            comptime G_item = Self.geom_types[i]

            comptime if G_item.GEOM_TYPE == GEOM_PLANE:
                masses[i] = Scalar[DTYPE](0)
            elif G_item.GEOM_MASS >= 0.0:
                masses[i] = Scalar[DTYPE](G_item.GEOM_MASS)
            else:
                var vol = geom_volume[DTYPE](
                    G_item.GEOM_TYPE,
                    Scalar[DTYPE](G_item.RADIUS),
                    Scalar[DTYPE](G_item.HALF_LENGTH),
                    Scalar[DTYPE](G_item.HALF_X),
                    Scalar[DTYPE](G_item.HALF_Y),
                    Scalar[DTYPE](G_item.HALF_Z),
                )

                comptime if G_item.DENSITY >= 0.0:
                    masses[i] = Scalar[DTYPE](G_item.DENSITY) * vol
                else:
                    masses[i] = Scalar[DTYPE](Defaults.GEOM_DENSITY) * vol
        return masses^

    @staticmethod
    def get_ground_rgba() -> List[Float64]:
        """Return [r, g, b] of the first plane geom's color, or empty list."""
        var result = List[Float64]()
        comptime for i in range(Self.N):
            comptime GG = Self.geom_types[i]
            comptime if GG.GEOM_TYPE == 0:  # GEOM_PLANE
                if len(result) == 0:
                    result.append(Float64(GG.COLOR.r) / 255.0)
                    result.append(Float64(GG.COLOR.g) / 255.0)
                    result.append(Float64(GG.COLOR.b) / 255.0)
        return result^

    @staticmethod
    def render_ground_geoms(
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        """Draw plane geoms as ground grids, or a fallback grid if no planes."""
        var has_plane = False

        comptime for i in range(Self.N):
            comptime GG = Self.geom_types[i]

            comptime if GG.GEOM_TYPE == GEOM_PLANE:
                has_plane = True
                var max_radius: Float64 = 0.0

                comptime for j in range(Self.N):
                    comptime HH = Self.geom_types[j]

                    comptime if HH.BODY_IDX > 0:
                        if HH.RADIUS > max_radius:
                            max_radius = HH.RADIUS

                var ground_offset = GG.POS_Z - max_radius * (
                    visual_radius_scale - 1.0
                )
                var grid_center_x = torso_x if follow else 0.0
                renderer.draw_ground_grid(grid_center_x, height=ground_offset)

        if not has_plane:
            var max_radius: Float64 = 0.0

            comptime for i in range(Self.N):
                comptime GG = Self.geom_types[i]

                comptime if GG.BODY_IDX > 0:
                    if GG.RADIUS > max_radius:
                        max_radius = GG.RADIUS

            var ground_offset = -max_radius * (visual_radius_scale - 1.0)
            var grid_center_x = torso_x if follow else 0.0
            renderer.draw_ground_grid(grid_center_x, height=ground_offset)

    @staticmethod
    def render_body_geoms(
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        """Draw all body-attached geoms (capsule, sphere, box, cylinder)."""

        comptime for i in range(Self.N):
            comptime GG = Self.geom_types[i]

            # Skip worldbody geoms (planes rendered separately)
            comptime if GG.BODY_IDX > 0:
                var body_pos = positions[GG.BODY_IDX]
                var body_quat = quaternions[GG.BODY_IDX]

                # Apply local position offset
                var geom_pos: _RVec3

                comptime if GG.POS_X == 0.0 and GG.POS_Y == 0.0 and GG.POS_Z == 0.0:
                    geom_pos = body_pos
                else:
                    var local_pos = _RVec3(GG.POS_X, GG.POS_Y, GG.POS_Z)
                    geom_pos = body_pos + body_quat.rotate_vec(local_pos)

                # Apply local rotation
                var geom_quat: _RQuat

                comptime if (
                    GG.QUAT_X == 0.0
                    and GG.QUAT_Y == 0.0
                    and GG.QUAT_Z == 0.0
                    and GG.QUAT_W == 1.0
                ):
                    geom_quat = body_quat
                else:
                    var local_quat = _RQuat(
                        GG.QUAT_W, GG.QUAT_X, GG.QUAT_Y, GG.QUAT_Z
                    )
                    geom_quat = body_quat * local_quat

                # Resolve material properties (-1.0 sentinel = use defaults)
                comptime _shin: Float64 = 0.5 if GG.SHININESS < 0.0 else GG.SHININESS
                comptime _spec: Float64 = 0.5 if GG.SPECULAR < 0.0 else GG.SPECULAR
                comptime _refl: Float64 = 0.0 if GG.REFLECTANCE < 0.0 else GG.REFLECTANCE

                # Dispatch draw call by geom type
                comptime if GG.GEOM_TYPE == GEOM_CAPSULE:
                    renderer.draw_capsule(
                        center=geom_pos,
                        orientation=geom_quat,
                        radius=GG.RADIUS * visual_radius_scale,
                        half_height=GG.HALF_LENGTH,
                        axis=2,
                        color=GG.COLOR,
                        shininess=Float32(_shin),
                        specular=Float32(_spec),
                        reflectance=Float32(_refl),
                    )
                elif GG.GEOM_TYPE == GEOM_SPHERE:
                    renderer.draw_sphere(
                        center=geom_pos,
                        radius=GG.RADIUS * visual_radius_scale,
                        color=GG.COLOR,
                        shininess=Float32(_shin),
                        specular=Float32(_spec),
                        reflectance=Float32(_refl),
                    )
                elif GG.GEOM_TYPE == GEOM_BOX:
                    renderer.draw_box(
                        center=geom_pos,
                        orientation=geom_quat,
                        half_extents=_RVec3(GG.HALF_X, GG.HALF_Y, GG.HALF_Z),
                        color=GG.COLOR,
                        shininess=Float32(_shin),
                        specular=Float32(_spec),
                        reflectance=Float32(_refl),
                    )
                elif GG.GEOM_TYPE == GEOM_CYLINDER:
                    renderer.draw_cylinder(
                        center=geom_pos,
                        orientation=geom_quat,
                        radius=GG.RADIUS * visual_radius_scale,
                        half_height=GG.HALF_LENGTH,
                        axis=2,
                        color=GG.COLOR,
                        shininess=Float32(_shin),
                        specular=Float32(_spec),
                        reflectance=Float32(_refl),
                    )
