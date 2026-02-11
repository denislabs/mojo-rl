"""BodySpec trait and concrete body types for compile-time model definitions.

Defines body geometry, mass, kinematic tree, and collision properties as
compile-time constants. Inertia is auto-computed from geometry and mass.

Geometry types reuse constants from physics3d/constants.mojo:
  GEOM_CAPSULE = 2, GEOM_SPHERE = 1, GEOM_BOX = 3
"""

from ..constants import GEOM_CAPSULE, GEOM_SPHERE, GEOM_BOX
from render3d import Color3D

# =============================================================================
# BodySpec Trait
# =============================================================================


trait BodySpec:
    """Compile-time body specification for physics3d model definitions.

    All properties are compile-time constants matching Model body fields
    and GPU buffer layout. Inertia is auto-computed from geometry + mass.
    """

    # Geometry
    comptime GEOM_TYPE: Int  # GEOM_CAPSULE, GEOM_SPHERE, GEOM_BOX
    comptime RADIUS: Float64  # Collision radius
    comptime HALF_LENGTH: Float64  # Capsule half-length (0 for sphere/box)
    comptime HALF_X: Float64  # Box half-extents (0 for capsule/sphere)
    comptime HALF_Y: Float64
    comptime HALF_Z: Float64

    # Mass
    comptime MASS: Float64

    # Kinematic tree
    comptime PARENT: Int  # Parent body index (-1 for world)

    # Local frame in parent
    comptime POS_X: Float64
    comptime POS_Y: Float64
    comptime POS_Z: Float64
    comptime QUAT_X: Float64
    comptime QUAT_Y: Float64
    comptime QUAT_Z: Float64
    comptime QUAT_W: Float64

    # Collision filtering (MuJoCo-style)
    comptime CONTYPE: Int
    comptime CONAFFINITY: Int

    # Visual properties
    comptime COLOR: Color3D

    # Auto-computed inertia from geometry + mass
    @staticmethod
    fn ixx() -> Float64:
        ...

    @staticmethod
    fn iyy() -> Float64:
        ...

    @staticmethod
    fn izz() -> Float64:
        ...


# =============================================================================
# CapsuleBody
# =============================================================================


@fieldwise_init
struct CapsuleBody[
    parent: Int = -1,
    mass: Float64 = 1.0,
    radius: Float64 = 0.05,
    half_length: Float64 = 0.1,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(204, 153, 102),
](BodySpec):
    """Capsule body with auto-computed inertia.

    Inertia formula (cylinder + hemispherical caps approximation):
      L = 2 * half_length + 2 * radius  (total length)
      I_trans = mass * (3 * r^2 + L^2) / 12
      I_axial = 0.5 * mass * r^2
    """

    comptime GEOM_TYPE: Int = GEOM_CAPSULE
    comptime RADIUS: Float64 = Self.radius
    comptime HALF_LENGTH: Float64 = Self.half_length
    comptime HALF_X: Float64 = 0.0
    comptime HALF_Y: Float64 = 0.0
    comptime HALF_Z: Float64 = 0.0
    comptime MASS: Float64 = Self.mass
    comptime PARENT: Int = Self.parent
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime QUAT_X: Float64 = Self.quat_x
    comptime QUAT_Y: Float64 = Self.quat_y
    comptime QUAT_Z: Float64 = Self.quat_z
    comptime QUAT_W: Float64 = Self.quat_w
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color

    @staticmethod
    fn _total_length() -> Float64:
        return 2.0 * Self.HALF_LENGTH + 2.0 * Self.RADIUS

    @staticmethod
    fn ixx() -> Float64:
        """Transverse inertia."""
        var r2 = Self.RADIUS * Self.RADIUS
        var L = Self._total_length()
        return Self.MASS * (3.0 * r2 + L * L) / 12.0

    @staticmethod
    fn iyy() -> Float64:
        """Transverse inertia (same as ixx for capsule)."""
        return Self.ixx()

    @staticmethod
    fn izz() -> Float64:
        """Axial inertia."""
        return 0.5 * Self.MASS * Self.RADIUS * Self.RADIUS


# =============================================================================
# SphereBody
# =============================================================================


@fieldwise_init
struct SphereBody[
    parent: Int = -1,
    mass: Float64 = 1.0,
    radius: Float64 = 0.05,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(204, 153, 102),
](BodySpec):
    """Sphere body with auto-computed inertia.

    Inertia: I = (2/5) * mass * r^2 (uniform solid sphere)
    """

    comptime GEOM_TYPE: Int = GEOM_SPHERE
    comptime RADIUS: Float64 = Self.radius
    comptime HALF_LENGTH: Float64 = 0.0
    comptime HALF_X: Float64 = 0.0
    comptime HALF_Y: Float64 = 0.0
    comptime HALF_Z: Float64 = 0.0
    comptime MASS: Float64 = Self.mass
    comptime PARENT: Int = Self.parent
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime QUAT_X: Float64 = Self.quat_x
    comptime QUAT_Y: Float64 = Self.quat_y
    comptime QUAT_Z: Float64 = Self.quat_z
    comptime QUAT_W: Float64 = Self.quat_w
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color

    @staticmethod
    fn ixx() -> Float64:
        return 0.4 * Self.MASS * Self.RADIUS * Self.RADIUS

    @staticmethod
    fn iyy() -> Float64:
        return Self.ixx()

    @staticmethod
    fn izz() -> Float64:
        return Self.ixx()


# =============================================================================
# BoxBody
# =============================================================================


@fieldwise_init
struct BoxBody[
    parent: Int = -1,
    mass: Float64 = 1.0,
    half_x: Float64 = 0.1,
    half_y: Float64 = 0.1,
    half_z: Float64 = 0.1,
    radius: Float64 = 0.0,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    contype: Int = 1,
    conaffinity: Int = 1,
    color: Color3D = Color3D(204, 153, 102),
](BodySpec):
    """Box body with auto-computed inertia.

    Inertia: I_xx = (1/12) * mass * (4*hy^2 + 4*hz^2), etc.
    (using full dimensions = 2 * half-extents)
    """

    comptime GEOM_TYPE: Int = GEOM_BOX
    comptime RADIUS: Float64 = Self.radius
    comptime HALF_LENGTH: Float64 = 0.0
    comptime HALF_X: Float64 = Self.half_x
    comptime HALF_Y: Float64 = Self.half_y
    comptime HALF_Z: Float64 = Self.half_z
    comptime MASS: Float64 = Self.mass
    comptime PARENT: Int = Self.parent
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime QUAT_X: Float64 = Self.quat_x
    comptime QUAT_Y: Float64 = Self.quat_y
    comptime QUAT_Z: Float64 = Self.quat_z
    comptime QUAT_W: Float64 = Self.quat_w
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color

    @staticmethod
    fn ixx() -> Float64:
        var fy = 2.0 * Self.HALF_Y
        var fz = 2.0 * Self.HALF_Z
        return Self.MASS * (fy * fy + fz * fz) / 12.0

    @staticmethod
    fn iyy() -> Float64:
        var fx = 2.0 * Self.HALF_X
        var fz = 2.0 * Self.HALF_Z
        return Self.MASS * (fx * fx + fz * fz) / 12.0

    @staticmethod
    fn izz() -> Float64:
        var fx = 2.0 * Self.HALF_X
        var fy = 2.0 * Self.HALF_Y
        return Self.MASS * (fx * fx + fy * fy) / 12.0
