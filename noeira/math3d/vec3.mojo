"""3D Vector type for physics and rendering.

Provides Vec3 struct with common operations for 3D math.
Uses SIMD[float64, 4] backing for efficient computation.
"""

from std.math import sqrt, cos, sin


@fieldwise_init
struct Vec3[DTYPE: DType](ImplicitlyCopyable, Movable, Writable):
    """3D vector for positions, velocities, and directions.

    Backed by SIMD[DType.float64, 4] for efficient math operations.
    The fourth component is unused but provides better memory alignment.
    """

    var x: Scalar[Self.DTYPE]
    var y: Scalar[Self.DTYPE]
    var z: Scalar[Self.DTYPE]

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @staticmethod
    def zero() -> Self:
        """Return the zero vector."""
        return Self(0.0, 0.0, 0.0)

    @staticmethod
    def one() -> Self:
        """Return the unit vector (1, 1, 1)."""
        return Self(1.0, 1.0, 1.0)

    @staticmethod
    def unit_x() -> Self:
        """Return the X axis unit vector."""
        return Self(1.0, 0.0, 0.0)

    @staticmethod
    def unit_y() -> Self:
        """Return the Y axis unit vector."""
        return Self(0.0, 1.0, 0.0)

    @staticmethod
    def unit_z() -> Self:
        """Return the Z axis unit vector."""
        return Self(0.0, 0.0, 1.0)

    @staticmethod
    def from_scalar(s: Scalar[Self.DTYPE]) -> Self:
        """Create a vector with all components set to s."""
        return Self(s, s, s)

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def __add__(self, other: Self) -> Self:
        """Vector addition."""
        return Self(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Self:
        """Vector subtraction."""
        return Self(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Scalar[Self.DTYPE]) -> Self:
        """Scalar multiplication."""
        return Self(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: Scalar[Self.DTYPE]) -> Self:
        """Scalar multiplication (reversed)."""
        return Self(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: Scalar[Self.DTYPE]) -> Self:
        """Scalar division."""
        var inv = 1.0 / scalar
        return Self(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> Self:
        """Negation."""
        return Self(-self.x, -self.y, -self.z)

    def __iadd__(mut self, other: Self):
        """In-place addition."""
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def __isub__(mut self, other: Self):
        """In-place subtraction."""
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    def __imul__(mut self, scalar: Scalar[Self.DTYPE]):
        """In-place scalar multiplication."""
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar

    def __itruediv__(mut self, scalar: Scalar[Self.DTYPE]):
        """In-place scalar division."""
        var inv = 1.0 / scalar
        self.x *= inv
        self.y *= inv
        self.z *= inv

    # =========================================================================
    # Comparison Operations
    # =========================================================================

    def __eq__(self, other: Self) -> Bool:
        """Equality check."""
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: Self) -> Bool:
        """Inequality check."""
        return not (self == other)

    def approx_eq(
        self, other: Self, tolerance: Scalar[Self.DTYPE] = 1e-10
    ) -> Bool:
        """Approximate equality with tolerance."""
        return (
            abs(self.x - other.x) < tolerance
            and abs(self.y - other.y) < tolerance
            and abs(self.z - other.z) < tolerance
        )

    # =========================================================================
    # Geometric Operations
    # =========================================================================

    def dot(self, other: Self) -> Scalar[Self.DTYPE]:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Self) -> Self:
        """Cross product (self × other)."""
        return Self(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> Scalar[Self.DTYPE]:
        """Squared length (avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> Scalar[Self.DTYPE] where Self.DTYPE.is_floating_point():
        """Euclidean length."""
        return sqrt(self.length_squared())

    def normalized(self) -> Self where Self.DTYPE.is_floating_point():
        """Return unit vector in same direction.

        Returns zero vector if length is zero.
        """
        var len = self.length()
        if len > 1e-10:
            return self / len
        return Self.zero()

    def normalize(mut self) where Self.DTYPE.is_floating_point():
        """Normalize in place."""
        var len = self.length()
        if len > 1e-10:
            self /= len

    def distance_to(
        self, other: Self
    ) -> Scalar[Self.DTYPE] where Self.DTYPE.is_floating_point():
        """Distance to another point."""
        return (self - other).length()

    def distance_squared_to(self, other: Self) -> Scalar[Self.DTYPE]:
        """Squared distance to another point (avoids sqrt)."""
        return (self - other).length_squared()

    # =========================================================================
    # Component-wise Operations
    # =========================================================================

    def hadamard(self, other: Self) -> Self:
        """Component-wise (Hadamard) product."""
        return Self(self.x * other.x, self.y * other.y, self.z * other.z)

    def abs(self) -> Self:
        """Component-wise absolute value."""
        return Self(abs(self.x), abs(self.y), abs(self.z))

    def min(self, other: Self) -> Self:
        """Component-wise minimum."""
        return Self(
            min(self.x, other.x),
            min(self.y, other.y),
            min(self.z, other.z),
        )

    def max(self, other: Self) -> Self:
        """Component-wise maximum."""
        return Self(
            max(self.x, other.x),
            max(self.y, other.y),
            max(self.z, other.z),
        )

    def clamp(self, min_val: Self, max_val: Self) -> Self:
        """Component-wise clamp between min and max."""
        return self.max(min_val).min(max_val)

    def clamp_length(
        self, max_length: Scalar[Self.DTYPE]
    ) -> Self where Self.DTYPE.is_floating_point():
        """Clamp vector to maximum length."""
        var len_sq = self.length_squared()
        if len_sq > max_length * max_length:
            return self.normalized() * max_length
        return self

    # =========================================================================
    # Projection and Reflection
    # =========================================================================

    def project_onto(self, onto: Self) -> Self:
        """Project this vector onto another.

        Args:
            onto: The vector to project onto.

        Returns:
            The projection of self onto the given vector.
        """
        var onto_len_sq = onto.length_squared()
        if onto_len_sq < 1e-10:
            return Self.zero()
        return onto * (self.dot(onto) / onto_len_sq)

    def reject_from(self, from_vec: Self) -> Self:
        """Component of this vector perpendicular to another.

        Args:
            from_vec: The vector to reject from.

        Returns:
            The rejection of self from the given vector.
        """
        return self - self.project_onto(from_vec)

    def reflect(self, normal: Self) -> Self:
        """Reflect this vector about a normal.

        Args:
            normal: The surface normal (should be normalized).

        Returns:
            The reflected vector.
        """
        return self - normal * (2.0 * self.dot(normal))

    # =========================================================================
    # Interpolation
    # =========================================================================

    def lerp(self, other: Self, t: Scalar[Self.DTYPE]) -> Self:
        """Linear interpolation between self and other.

        Args:
            other: The target vector.
            t: Interpolation factor (0 = self, 1 = other).

        Returns:
            Interpolated vector.
        """
        return self + (other - self) * t

    # =========================================================================
    # Rotation
    # =========================================================================

    def rotated_x(
        self, angle: Scalar[Self.DTYPE]
    ) -> Self where Self.DTYPE.is_floating_point():
        """Rotate around X axis.

        Args:
            angle: Rotation angle in radians.

        Returns:
            Rotated vector.
        """
        var c = cos(angle)
        var s = sin(angle)
        return Self(
            self.x,
            self.y * c - self.z * s,
            self.y * s + self.z * c,
        )

    def rotated_y(
        self, angle: Scalar[Self.DTYPE]
    ) -> Self where Self.DTYPE.is_floating_point():
        """Rotate around Y axis.

        Args:
            angle: Rotation angle in radians.

        Returns:
            Rotated vector.
        """
        var c = cos(angle)
        var s = sin(angle)
        return Self(
            self.x * c + self.z * s,
            self.y,
            -self.x * s + self.z * c,
        )

    def rotated_z(
        self, angle: Scalar[Self.DTYPE]
    ) -> Self where Self.DTYPE.is_floating_point():
        """Rotate around Z axis.

        Args:
            angle: Rotation angle in radians.

        Returns:
            Rotated vector.
        """
        var c = cos(angle)
        var s = sin(angle)
        return Self(
            self.x * c - self.y * s,
            self.x * s + self.y * c,
            self.z,
        )

    # =========================================================================
    # Indexing
    # =========================================================================

    def __getitem__(self, index: Int) -> Scalar[Self.DTYPE]:
        """Get component by index (0=x, 1=y, 2=z)."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            return self.z

    def __setitem__(mut self, index: Int, value: Scalar[Self.DTYPE]):
        """Set component by index (0=x, 1=y, 2=z)."""
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            self.z = value

    # =========================================================================
    # Conversion
    # =========================================================================

    def to_simd(self) -> SIMD[Self.DTYPE, 4]:
        """Convert to SIMD vector (w component = 0)."""
        return SIMD[Self.DTYPE, 4](self.x, self.y, self.z, 0.0)

    @staticmethod
    def from_simd(v: SIMD[Self.DTYPE, 4]) -> Self:
        """Create from SIMD vector (ignores w component)."""
        return Self(v[0], v[1], v[2])

    def __str__(self) -> String:
        """String representation."""
        return (
            "Vec3("
            + String(self.x)
            + ", "
            + String(self.y)
            + ", "
            + String(self.z)
            + ")"
        )


# =========================================================================
# Utility Functions
# =========================================================================


def vec3[
    DTYPE: DType
](x: Scalar[DTYPE], y: Scalar[DTYPE], z: Scalar[DTYPE]) -> Vec3[DTYPE]:
    """Convenience function to create a Vec3."""
    return Vec3(x, y, z)


def dot[DTYPE: DType](a: Vec3[DTYPE], b: Vec3[DTYPE]) -> Scalar[DTYPE]:
    """Dot product of two vectors."""
    return a.dot(b)


def cross[DTYPE: DType](a: Vec3[DTYPE], b: Vec3[DTYPE]) -> Vec3[DTYPE]:
    """Cross product of two vectors."""
    return a.cross(b)


def normalize[
    DTYPE: DType
](v: Vec3[DTYPE]) -> Vec3[DTYPE] where DTYPE.is_floating_point():
    """Return normalized vector."""
    return v.normalized()


def length[
    DTYPE: DType
](v: Vec3[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """Return vector length."""
    return v.length()


def distance[
    DTYPE: DType
](a: Vec3[DTYPE], b: Vec3[DTYPE]) -> Scalar[
    DTYPE
] where DTYPE.is_floating_point():
    """Distance between two points."""
    return a.distance_to(b)


def lerp[
    DTYPE: DType
](a: Vec3[DTYPE], b: Vec3[DTYPE], t: Scalar[DTYPE]) -> Vec3[DTYPE]:
    """Linear interpolation."""
    return a.lerp(b, t)
