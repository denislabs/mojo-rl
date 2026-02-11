"""GeomSpec trait and PlaneGeom for worldbody static geometry.

Mirrors MuJoCo <worldbody><geom .../> elements. Phase 1 supports PlaneGeom
(infinite ground plane). Future phases will add BoxGeom, SphereGeom,
CapsuleGeom for obstacles and terrain.

Usage:
    from physics3d.model.geom_spec import GeomSpec, PlaneGeom

    # MuJoCo: <geom type="plane" friction=".4" conaffinity="1" size="40 40 40"/>
    comptime MyPlane = PlaneGeom[z=0.0, friction=0.4, conaffinity=1]
"""

from ..constants import GEOM_PLANE


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
