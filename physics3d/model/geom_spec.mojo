"""GeomSpec trait and concrete geom types for geometry specification.

Supports both worldbody (static) and body-attached geoms via `body_idx`:
  - body_idx=0: worldbody (static geom in world frame)
  - body_idx>=1: attached to that body (pos/quat in body's local frame)

Four geom shapes:
  - Plane: Infinite ground plane (body_idx always 0)
  - Sphere: Sphere geom (body_idx=0 for static, >=1 for body-attached)
  - Capsule: Capsule geom (body_idx=0 for static, >=1 for body-attached)
  - Box: Box geom (body_idx=0 for static, >=1 for body-attached)

Fields that use sentinel value -1.0 (Float64) or -1 (Int) mean "use
ModelDefaults". Resolution happens at Geoms.setup_model time.

Usage:
    from physics3d.model.geom_spec import GeomSpec, Plane, Capsule

    # Ground plane (friction from defaults)
    comptime MyPlane = Plane[z=0.0]

    # Capsule with explicit friction override
    comptime MyCap = Capsule[body_idx=1, radius=0.046, friction=0.4]
"""

from ..constants import GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX
from render3d import Color3D

# Sentinel values for "use model default" (re-exported for convenience)
comptime _UNSET_F64: Float64 = -1.0
comptime _UNSET_INT: Int = -1


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
    # Visual
    comptime COLOR: Color3D


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
    size_x: Float64 = 40.0,
    size_y: Float64 = 40.0,
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
    comptime COLOR: Color3D = Color3D(128, 128, 128)


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
    color: Color3D = Color3D(100, 100, 200),
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
    comptime COLOR: Color3D = Self.color


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
    color: Color3D = Color3D(204, 153, 102),
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
    comptime COLOR: Color3D = Self.color


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
    color: Color3D = Color3D(100, 200, 100),
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
    comptime COLOR: Color3D = Self.color


# =============================================================================
# Backwards-compatible aliases
# =============================================================================

comptime PlaneGeom = Plane
comptime SphereGeom = Sphere
comptime BoxGeom = Box
comptime CapsuleGeom = Capsule
comptime BodyCapsuleGeom = Capsule
comptime BodySphereGeom = Sphere
comptime BodyBoxGeom = Box
