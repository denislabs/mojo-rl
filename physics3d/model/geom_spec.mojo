"""GeomSpec trait and concrete geom types for geometry specification.

Supports both static (worldbody) geoms and body-attached geoms.
Static geoms have BODY_IDX = -1, body-attached geoms have BODY_IDX >= 0.

Geom types:
  - PlaneGeom: Infinite ground plane (static only)
  - SphereGeom: Static sphere obstacle
  - BoxGeom: Static box obstacle
  - CapsuleGeom: Static capsule obstacle
  - BodyCapsuleGeom: Capsule attached to a body
  - BodySphereGeom: Sphere attached to a body
  - BodyBoxGeom: Box attached to a body

Usage:
    from physics3d.model.geom_spec import GeomSpec, PlaneGeom, BodyCapsuleGeom

    # Ground plane
    comptime MyPlane = PlaneGeom[z=0.0, friction=0.4]

    # Capsule geom on body 0 (torso) at local offset
    comptime MyGeom = BodyCapsuleGeom[body_idx=0, radius=0.046, half_length=0.09]
"""

from ..constants import GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX
from render3d import Color3D


trait GeomSpec:
    """Compile-time specification for a geom (static or body-attached)."""

    comptime GEOM_TYPE: Int  # GEOM_PLANE, GEOM_BOX, GEOM_SPHERE, GEOM_CAPSULE
    # Body index (-1 for static/worldbody geoms, >= 0 for body-attached)
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
    comptime FRICTION: Float64
    # Collision filtering
    comptime CONTYPE: Int
    comptime CONAFFINITY: Int
    # Visual
    comptime COLOR: Color3D


# =============================================================================
# Static (worldbody) geom types — BODY_IDX = -1
# =============================================================================


@fieldwise_init
struct PlaneGeom[
    z: Float64 = 0.0,
    friction: Float64 = 0.5,
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
    comptime BODY_IDX: Int = -1
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
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Color3D(128, 128, 128)


@fieldwise_init
struct SphereGeom[
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    radius: Float64 = 0.5,
    friction: Float64 = 0.5,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(100, 100, 200),
](GeomSpec):
    """Static sphere geom at a fixed world position.

    Matches MuJoCo <geom type="sphere" pos="x y z" size="radius"/>.
    """

    comptime GEOM_TYPE: Int = GEOM_SPHERE
    comptime BODY_IDX: Int = -1
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
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color


@fieldwise_init
struct BoxGeom[
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    half_x: Float64 = 0.5,
    half_y: Float64 = 0.5,
    half_z: Float64 = 0.5,
    friction: Float64 = 0.5,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(100, 200, 100),
](GeomSpec):
    """Static box geom at a fixed world position and orientation.

    Matches MuJoCo <geom type="box" pos="x y z" quat="w x y z" size="hx hy hz"/>.
    """

    comptime GEOM_TYPE: Int = GEOM_BOX
    comptime BODY_IDX: Int = -1
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
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color


@fieldwise_init
struct CapsuleGeom[
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    half_length: Float64 = 0.5,
    radius: Float64 = 0.25,
    friction: Float64 = 0.5,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(200, 100, 100),
](GeomSpec):
    """Static capsule geom at a fixed world position and orientation.

    Matches MuJoCo <geom type="capsule" pos="x y z" size="radius hlength"/>.
    Capsule axis is local Z, half_length defines the cylinder half-length
    (total capsule length = 2*half_length + 2*radius).
    """

    comptime GEOM_TYPE: Int = GEOM_CAPSULE
    comptime BODY_IDX: Int = -1
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
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color


# =============================================================================
# Body-attached geom types — BODY_IDX >= 0
# =============================================================================


@fieldwise_init
struct BodyCapsuleGeom[
    body_idx: Int,
    radius: Float64 = 0.05,
    half_length: Float64 = 0.1,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    friction: Float64 = 0.5,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(204, 153, 102),
](GeomSpec):
    """Capsule geom attached to a body at a local offset.

    pos/quat define the geom frame relative to the body frame.
    When pos=(0,0,0) and quat=(0,0,0,1), the geom is at the body origin.
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
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color


@fieldwise_init
struct BodySphereGeom[
    body_idx: Int,
    radius: Float64 = 0.05,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    friction: Float64 = 0.5,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(204, 153, 102),
](GeomSpec):
    """Sphere geom attached to a body at a local offset."""

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
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color


@fieldwise_init
struct BodyBoxGeom[
    body_idx: Int,
    half_x: Float64 = 0.1,
    half_y: Float64 = 0.1,
    half_z: Float64 = 0.1,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    friction: Float64 = 0.5,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(204, 153, 102),
](GeomSpec):
    """Box geom attached to a body at a local offset."""

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
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color
