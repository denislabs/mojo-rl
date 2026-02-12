"""GeomSpec trait and concrete geom types for worldbody static geometry.

Mirrors MuJoCo <worldbody><geom .../> elements. Supports PlaneGeom
(infinite ground plane), SphereGeom, BoxGeom, and CapsuleGeom for
obstacles and terrain.

Usage:
    from physics3d.model.geom_spec import GeomSpec, PlaneGeom, SphereGeom

    # MuJoCo: <geom type="plane" friction=".4" conaffinity="1" size="40 40 40"/>
    comptime MyPlane = PlaneGeom[z=0.0, friction=0.4, conaffinity=1]

    # Static sphere obstacle at (2, 0, 0.5)
    comptime MySphere = SphereGeom[pos_x=2.0, pos_z=0.5, radius=0.3]
"""

from ..constants import GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX


trait GeomSpec:
    """Compile-time specification for a static (worldbody) geom."""

    comptime GEOM_TYPE: Int  # GEOM_PLANE, GEOM_BOX, GEOM_SPHERE, GEOM_CAPSULE
    # Position in world frame
    comptime POS_X: Float64
    comptime POS_Y: Float64
    comptime POS_Z: Float64
    # Orientation in world frame (quaternion [x, y, z, w])
    comptime QUAT_X: Float64
    comptime QUAT_Y: Float64
    comptime QUAT_Z: Float64
    comptime QUAT_W: Float64
    # Size (interpretation depends on GEOM_TYPE)
    comptime SIZE_X: Float64  # plane: extent_x; box: half_x
    comptime SIZE_Y: Float64  # plane: extent_y; box: half_y
    comptime SIZE_Z: Float64  # box: half_z; capsule: half_length
    comptime RADIUS: Float64  # sphere/capsule radius
    # Physics
    comptime FRICTION: Float64
    # Collision filtering
    comptime CONTYPE: Int
    comptime CONAFFINITY: Int


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
    comptime FRICTION: Float64 = Self.friction
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity


@fieldwise_init
struct SphereGeom[
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    radius: Float64 = 0.5,
    friction: Float64 = 0.5,
    contype: Int = 1,
    conaffinity: Int = 1,
](GeomSpec):
    """Static sphere geom at a fixed world position.

    Matches MuJoCo <geom type="sphere" pos="x y z" size="radius"/>.
    """

    comptime GEOM_TYPE: Int = GEOM_SPHERE
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
    comptime FRICTION: Float64 = Self.friction
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity


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
](GeomSpec):
    """Static box geom at a fixed world position and orientation.

    Matches MuJoCo <geom type="box" pos="x y z" quat="w x y z" size="hx hy hz"/>.
    """

    comptime GEOM_TYPE: Int = GEOM_BOX
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
    comptime FRICTION: Float64 = Self.friction
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity


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
](GeomSpec):
    """Static capsule geom at a fixed world position and orientation.

    Matches MuJoCo <geom type="capsule" pos="x y z" size="radius hlength"/>.
    Capsule axis is local Z, half_length defines the cylinder half-length
    (total capsule length = 2*half_length + 2*radius).
    """

    comptime GEOM_TYPE: Int = GEOM_CAPSULE
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
    comptime FRICTION: Float64 = Self.friction
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
