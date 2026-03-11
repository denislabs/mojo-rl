"""GeomSpec trait and concrete geom types for geometry specification.

Supports both worldbody (static) and body-attached geoms via `body_idx`:
  - body_idx=0: worldbody (static geom in world frame)
  - body_idx>=1: attached to that body (pos/quat in body's local frame)

Four geom shapes:
  - Plane: Infinite ground plane (body_idx always 0)
  - Sphere: Sphere geom (body_idx=0 for static, >=1 for body-attached)
  - Capsule: Capsule geom (body_idx=0 for static, >=1 for body-attached)
  - Box: Box geom (body_idx=0 for static, >=1 for body-attached)

Usage:
    from mojo_rl.physics3d.model.geom_spec import GeomSpec, Plane, Capsule

    # Ground plane
    comptime MyPlane = Plane[z=0.0, friction=0.4]

    # Static capsule in world frame
    comptime StaticCap = Capsule[radius=0.25, half_length=0.5]

    # Capsule attached to body 1
    comptime BodyCap = Capsule[body_idx=1, radius=0.046, half_length=0.09]
"""

from mojo_rl.physics3d.constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
)
from mojo_rl.render import Color


trait GeomSpec:
    """Compile-time specification for a geom (static or body-attached)."""

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
    # Physics
    comptime FRICTION: Optional[Float64]
    comptime CONDIM: Int  # Contact dimensionality: 1, 3, 4, or 6
    comptime FRICTION_SPIN: Optional[Float64]  # Torsional friction coefficient
    comptime FRICTION_ROLL: Optional[Float64]  # Rolling friction coefficient
    # Collision filtering
    comptime CONTYPE: Optional[Int]
    comptime CONAFFINITY: Int
    # Visual
    comptime COLOR: Color


# =============================================================================
# Plane — infinite ground plane (always worldbody)
# =============================================================================


@fieldwise_init
struct Plane[
    z: Float64 = 0.0,
    friction: Float64 = 0.5,
    condim: Int = 3,
    friction_spin: Float64 = 0.005,
    friction_roll: Float64 = 0.0001,
    contype: Int = 1,
    conaffinity: Int = 1,
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
    comptime COLOR: Color = Color(128, 128, 128, 255)


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
    friction: Optional[Float64] = None,
    condim: Int = 3,
    friction_spin: Optional[Float64] = None,
    friction_roll: Optional[Float64] = None,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color = Color(100, 100, 200, 255),
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
    comptime FRICTION: Optional[Float64] = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Optional[Float64] = Self.friction_spin
    comptime FRICTION_ROLL: Optional[Float64] = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color = Self.color


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
    friction: Optional[Float64] = None,
    condim: Int = 3,
    friction_spin: Optional[Float64] = None,
    friction_roll: Optional[Float64] = None,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color = Color(204, 153, 102, 255),
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
    comptime FRICTION: Optional[Float64] = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Optional[Float64] = Self.friction_spin
    comptime FRICTION_ROLL: Optional[Float64] = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color = Self.color


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
    friction: Optional[Float64] = None,
    condim: Int = 3,
    friction_spin: Optional[Float64] = None,
    friction_roll: Optional[Float64] = None,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color = Color(100, 200, 100, 255),
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
    comptime FRICTION: Optional[Float64] = Self.friction
    comptime CONDIM: Int = Self.condim
    comptime FRICTION_SPIN: Optional[Float64] = Self.friction_spin
    comptime FRICTION_ROLL: Optional[Float64] = Self.friction_roll
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color = Self.color


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


trait ModelParams:
    comptime FRICTION: Optional[Float64]
    comptime FRICTION_SPIN: Optional[Float64]
    comptime FRICTION_ROLL: Optional[Float64]


struct HalfCheetahModelParams(ModelParams):
    comptime FRICTION: Float64 = 0.8
    comptime FRICTION_SPIN: Float64 = 0.005
    comptime FRICTION_ROLL: Float64 = 0.0001


comptime FRICTION_DEFAULT: Float64 = 0.5
comptime FRICTION_SPIN_DEFAULT: Float64 = 0.005
comptime FRICTION_ROLL_DEFAULT: Float64 = 0.0001


fn setup_friction_geom[
    friction_geom: Optional[Float64] = None,
    default_friction: Float64 = FRICTION_DEFAULT,
]() -> Float64:
    comptime if friction_geom:
        return friction_geom.value()
    else:
        return default_friction


fn setup_model_params():
    comptime TorsoGeom = Sphere[
        body_idx=1,
        radius=0.25,
    ]
    comptime TorsoFriction = setup_friction_geom[
        TorsoGeom.friction, HalfCheetahModelParams.FRICTION
    ]()
    print(String(TorsoFriction))


fn main():
    setup_model_params()
