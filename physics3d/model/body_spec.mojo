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

    All properties are compile-time constants matching Model body fields.
    Inertia is auto-computed from geometry + mass (geometry fields are on
    concrete structs, not required by the trait).
    """

    # Body name
    comptime NAME: String

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

    # CoM offset from body origin (body frame)
    comptime IPOS_X: Float64
    comptime IPOS_Y: Float64
    comptime IPOS_Z: Float64

    # Inertia frame quaternion [x, y, z, w] in body frame
    comptime IQUAT_X: Float64
    comptime IQUAT_Y: Float64
    comptime IQUAT_Z: Float64
    comptime IQUAT_W: Float64

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
    name: String = "capsule",
    radius: Float64 = 0.05,
    half_length: Float64 = 0.1,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    ipos_x: Float64 = 0.0,
    ipos_y: Float64 = 0.0,
    ipos_z: Float64 = 0.0,
    iquat_x: Float64 = 0.0,
    iquat_y: Float64 = 0.0,
    iquat_z: Float64 = 0.0,
    iquat_w: Float64 = 1.0,
    ixx_override: Float64 = 0.0,
    iyy_override: Float64 = 0.0,
    izz_override: Float64 = 0.0,
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
    comptime IPOS_X: Float64 = Self.ipos_x
    comptime IPOS_Y: Float64 = Self.ipos_y
    comptime IPOS_Z: Float64 = Self.ipos_z
    comptime IQUAT_X: Float64 = Self.iquat_x
    comptime IQUAT_Y: Float64 = Self.iquat_y
    comptime IQUAT_Z: Float64 = Self.iquat_z
    comptime IQUAT_W: Float64 = Self.iquat_w
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color
    comptime NAME: String = Self.name

    @staticmethod
    fn _total_length() -> Float64:
        return 2.0 * Self.HALF_LENGTH + 2.0 * Self.RADIUS

    @staticmethod
    fn _auto_ixx() -> Float64:
        var r2 = Self.RADIUS * Self.RADIUS
        var L = Self._total_length()
        return Self.MASS * (3.0 * r2 + L * L) / 12.0

    @staticmethod
    fn ixx() -> Float64:
        """Transverse inertia (override or auto-computed)."""
        if Self.ixx_override != 0.0:
            return Self.ixx_override
        return Self._auto_ixx()

    @staticmethod
    fn iyy() -> Float64:
        """Transverse inertia (override or auto-computed)."""
        if Self.iyy_override != 0.0:
            return Self.iyy_override
        return Self._auto_ixx()

    @staticmethod
    fn izz() -> Float64:
        """Axial inertia (override or auto-computed)."""
        if Self.izz_override != 0.0:
            return Self.izz_override
        return 0.5 * Self.MASS * Self.RADIUS * Self.RADIUS


# =============================================================================
# SphereBody
# =============================================================================


@fieldwise_init
struct SphereBody[
    parent: Int = -1,
    mass: Float64 = 1.0,
    name: String = "sphere",
    radius: Float64 = 0.05,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    quat_x: Float64 = 0.0,
    quat_y: Float64 = 0.0,
    quat_z: Float64 = 0.0,
    quat_w: Float64 = 1.0,
    ipos_x: Float64 = 0.0,
    ipos_y: Float64 = 0.0,
    ipos_z: Float64 = 0.0,
    iquat_x: Float64 = 0.0,
    iquat_y: Float64 = 0.0,
    iquat_z: Float64 = 0.0,
    iquat_w: Float64 = 1.0,
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
    comptime IPOS_X: Float64 = Self.ipos_x
    comptime IPOS_Y: Float64 = Self.ipos_y
    comptime IPOS_Z: Float64 = Self.ipos_z
    comptime IQUAT_X: Float64 = Self.iquat_x
    comptime IQUAT_Y: Float64 = Self.iquat_y
    comptime IQUAT_Z: Float64 = Self.iquat_z
    comptime IQUAT_W: Float64 = Self.iquat_w
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color
    comptime NAME: String = Self.name

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
    name: String = "box",
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
    ipos_x: Float64 = 0.0,
    ipos_y: Float64 = 0.0,
    ipos_z: Float64 = 0.0,
    iquat_x: Float64 = 0.0,
    iquat_y: Float64 = 0.0,
    iquat_z: Float64 = 0.0,
    iquat_w: Float64 = 1.0,
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
    comptime IPOS_X: Float64 = Self.ipos_x
    comptime IPOS_Y: Float64 = Self.ipos_y
    comptime IPOS_Z: Float64 = Self.ipos_z
    comptime IQUAT_X: Float64 = Self.iquat_x
    comptime IQUAT_Y: Float64 = Self.iquat_y
    comptime IQUAT_Z: Float64 = Self.iquat_z
    comptime IQUAT_W: Float64 = Self.iquat_w
    comptime CONTYPE: Int = Self.contype
    comptime CONAFFINITY: Int = Self.conaffinity
    comptime COLOR: Color3D = Self.color
    comptime NAME: String = Self.name

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
