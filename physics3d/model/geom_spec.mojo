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
    from physics3d.model.geom_spec import GeomSpec, Plane, Capsule

    # Ground plane (friction from defaults)
    comptime MyPlane = Plane[z=0.0]

    # Capsule with explicit friction override
    comptime MyCap = Capsule[body_idx=1, radius=0.046, friction=0.4]
"""
from math import sqrt
from ..constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
)
from render import Color
from ..model.defaults_spec import ModelDefaults, _resolve_f64, _resolve_int
from .inertia_from_geom import geom_volume, compute_inertia_from_geoms

# Sentinel values for "use model default" (re-exported for convenience)
comptime _UNSET_F64: Float64 = -1.0
comptime _UNSET_INT: Int = -1


# =============================================================================
# Compile-time math helpers for fromto conversion
# =============================================================================


fn _comptime_sqrt(x: Float64) -> Float64:
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


fn _fromto_center_x(from_x: Float64, to_x: Float64) -> Float64:
    return (from_x + to_x) * 0.5


fn _fromto_center_y(from_y: Float64, to_y: Float64) -> Float64:
    return (from_y + to_y) * 0.5


fn _fromto_center_z(from_z: Float64, to_z: Float64) -> Float64:
    return (from_z + to_z) * 0.5


fn _fromto_half_length(
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


fn _fromto_quat_component(
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
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        Defaults: ModelDefaultsLike,
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
        ]
    ):
        ...


@fieldwise_init
struct _EmptyGeoms(GeomsLike):
    comptime N: Int = 0

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        Defaults: ModelDefaultsLike,
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
        ]
    ):
        pass

    @staticmethod
    fn get_geom_spec(index: Int) -> GeomSpec:
        pass


@fieldwise_init
struct Geoms[*G: GeomSpec](GeomsLike):
    """Compile-time list of geom specifications (static + body-attached).

    Provides N (total geom count), type-level access via geom_types[i],
    and helper counts for static vs dynamic geoms.
    """

    comptime geom_types = Variadic.types[T=GeomSpec, *Self.G]
    comptime N: Int = Variadic.size(Self.geom_types)

    @staticmethod
    fn _count_static_geoms() -> Int:
        """Count of static (worldbody) geoms (BODY_IDX == 0)."""
        var total = 0

        @parameter
        for i in range(Self.N):

            @parameter
            if Self.geom_types[i].BODY_IDX == 0:
                total += 1
        return total

    @staticmethod
    fn _count_plane_geoms() -> Int:
        """Count of plane geoms (GEOM_TYPE == GEOM_PLANE)."""
        var total = 0

        @parameter
        for i in range(Self.N):

            @parameter
            if Self.geom_types[i].GEOM_TYPE == GEOM_PLANE:
                total += 1
        return total

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        Defaults: ModelDefaultsLike = ModelDefaults[],
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
        ]
    ):
        """Populate model geom arrays from compile-time GeomSpec list.

        Resolves sentinel values (-1.0/-1) from ModelDefaults.
        Sets geom type, body index, position, orientation, size, collision
        filtering, friction, and per-geom solref/solimp.
        """

        @parameter
        for i in range(Self.N):
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
            model.geom_solimp[i * 3 + 0] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_0, Defaults.GEOM_SOLIMP_0]()
            )
            model.geom_solimp[i * 3 + 1] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_1, Defaults.GEOM_SOLIMP_1]()
            )
            model.geom_solimp[i * 3 + 2] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_2, Defaults.GEOM_SOLIMP_2]()
            )

            # Contact margin (resolved from defaults)
            model.geom_margin[i] = Scalar[DTYPE](
                _resolve_f64[G_item.MARGIN, Defaults.GEOM_MARGIN]()
            )

            # Compute bounding sphere radius
            @parameter
            if G_item.GEOM_TYPE == GEOM_SPHERE:
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
            @parameter
            if G_item.GEOM_TYPE == GEOM_PLANE:
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

                @parameter
                if G_item.DENSITY >= 0.0:
                    # Explicit density on the geom
                    model.geom_mass[i] = Scalar[DTYPE](G_item.DENSITY) * vol
                else:
                    # Use default density
                    model.geom_mass[i] = (
                        Scalar[DTYPE](Defaults.GEOM_DENSITY) * vol
                    )
