"""BodySpec trait and concrete body types for compile-time model definitions.

Defines body geometry, mass, kinematic tree, and collision properties as
compile-time constants. Inertia is auto-computed from geometry and mass.

Geometry types reuse constants from mojo_rl.physics3d/constants.mojo:
  GEOM_CAPSULE = 2, GEOM_SPHERE = 1, GEOM_BOX = 3
"""

from ..constants import GEOM_CAPSULE, GEOM_SPHERE, GEOM_BOX
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    model_body_offset,
)
from std.gpu.host import HostBuffer
from mojo_rl.render import Color

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
    comptime PARENT: Int  # Parent body index (0 for worldbody)

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
    comptime COLOR: Color

    # Auto-computed inertia from geometry + mass
    @staticmethod
    def ixx() -> Float64:
        ...

    @staticmethod
    def iyy() -> Float64:
        ...

    @staticmethod
    def izz() -> Float64:
        ...


# =============================================================================
# CapsuleBody
# =============================================================================


@fieldwise_init
struct CapsuleBody[
    parent: Int = 0,
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
    color: Color = Color(204, 153, 102, 255),
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
    comptime COLOR: Color = Self.color
    comptime NAME: String = Self.name

    @staticmethod
    def _total_length() -> Float64:
        return 2.0 * Self.HALF_LENGTH + 2.0 * Self.RADIUS

    @staticmethod
    def _auto_ixx() -> Float64:
        var r2 = Self.RADIUS * Self.RADIUS
        var L = Self._total_length()
        return Self.MASS * (3.0 * r2 + L * L) / 12.0

    @staticmethod
    def ixx() -> Float64:
        """Transverse inertia (override or auto-computed)."""
        if Self.ixx_override != 0.0:
            return Self.ixx_override
        return Self._auto_ixx()

    @staticmethod
    def iyy() -> Float64:
        """Transverse inertia (override or auto-computed)."""
        if Self.iyy_override != 0.0:
            return Self.iyy_override
        return Self._auto_ixx()

    @staticmethod
    def izz() -> Float64:
        """Axial inertia (override or auto-computed)."""
        if Self.izz_override != 0.0:
            return Self.izz_override
        return 0.5 * Self.MASS * Self.RADIUS * Self.RADIUS


# =============================================================================
# SphereBody
# =============================================================================


@fieldwise_init
struct SphereBody[
    parent: Int = 0,
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
    color: Color = Color(204, 153, 102, 255),
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
    comptime COLOR: Color = Self.color
    comptime NAME: String = Self.name

    @staticmethod
    def ixx() -> Float64:
        return 0.4 * Self.MASS * Self.RADIUS * Self.RADIUS

    @staticmethod
    def iyy() -> Float64:
        return Self.ixx()

    @staticmethod
    def izz() -> Float64:
        return Self.ixx()


# =============================================================================
# BoxBody
# =============================================================================


@fieldwise_init
struct BoxBody[
    parent: Int = 0,
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
    color: Color = Color(204, 153, 102, 255),
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
    comptime COLOR: Color = Self.color
    comptime NAME: String = Self.name

    @staticmethod
    def ixx() -> Float64:
        var fy = 2.0 * Self.HALF_Y
        var fz = 2.0 * Self.HALF_Z
        return Self.MASS * (fy * fy + fz * fz) / 12.0

    @staticmethod
    def iyy() -> Float64:
        var fx = 2.0 * Self.HALF_X
        var fz = 2.0 * Self.HALF_Z
        return Self.MASS * (fx * fx + fz * fz) / 12.0

    @staticmethod
    def izz() -> Float64:
        var fx = 2.0 * Self.HALF_X
        var fy = 2.0 * Self.HALF_Y
        return Self.MASS * (fx * fx + fy * fy) / 12.0


trait BodiesLike:
    """Trait for compile-time body container types."""

    comptime N: Int  # number of bodies (excluding worldbody)

    @staticmethod
    def write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
    ](buffer: HostBuffer[DTYPE]):
        ...

    @staticmethod
    def setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            Self.N + 1,
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


@fieldwise_init
struct _EmptyBodies(BodiesLike):
    comptime N: Int = 0

    @staticmethod
    def write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
    ](buffer: HostBuffer[DTYPE]):
        pass

    @staticmethod
    def setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            Self.N + 1,
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


# =============================================================================
# Bodies — variadic body list
# =============================================================================


@fieldwise_init
struct Bodies[*B: BodySpec](BodiesLike):
    """Compile-time list of body specifications.

    Provides N (body count) and type-level access to each body via body_types[i].
    """

    comptime body_types = Variadic.types[T=BodySpec, *Self.B]
    comptime N: Int = Variadic.size(Self.body_types)

    @staticmethod
    def setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            Self.N + 1,  # +1 for worldbody at index 0
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ]
    ):
        """Populate model body properties from compile-time BodySpec list.

        Iterates over all body specs and sets mass, inertia, geometry, parent,
        local frame, and collision filtering on the model. Body indices start
        at 1 (worldbody at index 0 is initialized by Model.__init__).
        """

        comptime for i in range(Self.N):
            comptime B = Self.body_types[i]
            # Body index i+1: worldbody is at index 0 (reserved)
            comptime body_idx = i + 1

            # Mass, inertia
            model.set_body(
                body_idx,
                name=B.NAME,
                mass=Scalar[DTYPE](B.MASS),
                inertia=(
                    Scalar[DTYPE](B.ixx()),
                    Scalar[DTYPE](B.iyy()),
                    Scalar[DTYPE](B.izz()),
                ),
            )

            # Kinematic tree
            model.set_body_parent(body_idx, B.PARENT)

            # Local frame in parent
            model.set_body_local_frame(
                body_idx,
                pos=(
                    Scalar[DTYPE](B.POS_X),
                    Scalar[DTYPE](B.POS_Y),
                    Scalar[DTYPE](B.POS_Z),
                ),
                quat=(
                    Scalar[DTYPE](B.QUAT_X),
                    Scalar[DTYPE](B.QUAT_Y),
                    Scalar[DTYPE](B.QUAT_Z),
                    Scalar[DTYPE](B.QUAT_W),
                ),
            )

            # CoM offset and inertia frame
            model.set_body_ipos_iquat(
                body_idx,
                ipos=(
                    Scalar[DTYPE](B.IPOS_X),
                    Scalar[DTYPE](B.IPOS_Y),
                    Scalar[DTYPE](B.IPOS_Z),
                ),
                iquat=(
                    Scalar[DTYPE](B.IQUAT_X),
                    Scalar[DTYPE](B.IQUAT_Y),
                    Scalar[DTYPE](B.IQUAT_Z),
                    Scalar[DTYPE](B.IQUAT_W),
                ),
            )

    @staticmethod
    def write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
    ](buffer: HostBuffer[DTYPE]):
        """Write body data directly to GPU HostBuffer (no Model struct).

        Worldbody (index 0) is zero-initialized with identity quaternion.
        Body specs are written at indices 1..N.
        """
        # Worldbody at index 0: mass=0, identity quat, zero inertia
        var wb_off = model_body_offset(0)
        buffer[wb_off + BODY_IDX_QUAT_W] = Scalar[DTYPE](1.0)
        buffer[wb_off + BODY_IDX_IQUAT_W] = Scalar[DTYPE](1.0)
        buffer[wb_off + BODY_IDX_PARENT] = Scalar[DTYPE](-1)

        comptime for i in range(Self.N):
            comptime B = Self.body_types[i]
            comptime body_idx = i + 1
            var off = model_body_offset(body_idx)

            # Mass and inertia
            buffer[off + BODY_IDX_MASS] = Scalar[DTYPE](B.MASS)
            buffer[off + BODY_IDX_INV_MASS] = Scalar[DTYPE](1.0 / B.MASS)
            buffer[off + BODY_IDX_IXX] = Scalar[DTYPE](B.ixx())
            buffer[off + BODY_IDX_IYY] = Scalar[DTYPE](B.iyy())
            buffer[off + BODY_IDX_IZZ] = Scalar[DTYPE](B.izz())
            buffer[off + BODY_IDX_INV_IXX] = Scalar[DTYPE](1.0 / B.ixx())
            buffer[off + BODY_IDX_INV_IYY] = Scalar[DTYPE](1.0 / B.iyy())
            buffer[off + BODY_IDX_INV_IZZ] = Scalar[DTYPE](1.0 / B.izz())

            # Position in parent frame
            buffer[off + BODY_IDX_POS_X] = Scalar[DTYPE](B.POS_X)
            buffer[off + BODY_IDX_POS_Y] = Scalar[DTYPE](B.POS_Y)
            buffer[off + BODY_IDX_POS_Z] = Scalar[DTYPE](B.POS_Z)

            # Quaternion in parent frame
            buffer[off + BODY_IDX_QUAT_X] = Scalar[DTYPE](B.QUAT_X)
            buffer[off + BODY_IDX_QUAT_Y] = Scalar[DTYPE](B.QUAT_Y)
            buffer[off + BODY_IDX_QUAT_Z] = Scalar[DTYPE](B.QUAT_Z)
            buffer[off + BODY_IDX_QUAT_W] = Scalar[DTYPE](B.QUAT_W)

            # Parent body index
            buffer[off + BODY_IDX_PARENT] = Scalar[DTYPE](B.PARENT)

            # CoM offset (body frame)
            buffer[off + BODY_IDX_IPOS_X] = Scalar[DTYPE](B.IPOS_X)
            buffer[off + BODY_IDX_IPOS_Y] = Scalar[DTYPE](B.IPOS_Y)
            buffer[off + BODY_IDX_IPOS_Z] = Scalar[DTYPE](B.IPOS_Z)

            # Inertia frame quaternion (body frame)
            buffer[off + BODY_IDX_IQUAT_X] = Scalar[DTYPE](B.IQUAT_X)
            buffer[off + BODY_IDX_IQUAT_Y] = Scalar[DTYPE](B.IQUAT_Y)
            buffer[off + BODY_IDX_IQUAT_Z] = Scalar[DTYPE](B.IQUAT_Z)
            buffer[off + BODY_IDX_IQUAT_W] = Scalar[DTYPE](B.IQUAT_W)
