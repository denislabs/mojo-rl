"""3D Vector type for physics and rendering.

Provides Vec3 struct with common operations for 3D math.
Uses SIMD[float64, 4] backing for efficient computation.
"""

from math import sqrt, cos, sin


@fieldwise_init
struct Vec3(ImplicitlyCopyable, Movable, Stringable):
    """3D vector for positions, velocities, and directions.

    Backed by SIMD[DType.float64, 4] for efficient math operations.
    The fourth component is unused but provides better memory alignment.
    """

    var x: Float64
    var y: Float64
    var z: Float64

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @staticmethod
    fn zero() -> Self:
        """Return the zero vector."""
        return Self(0.0, 0.0, 0.0)

    @staticmethod
    fn one() -> Self:
        """Return the unit vector (1, 1, 1)."""
        return Self(1.0, 1.0, 1.0)

    @staticmethod
    fn unit_x() -> Self:
        """Return the X axis unit vector."""
        return Self(1.0, 0.0, 0.0)

    @staticmethod
    fn unit_y() -> Self:
        """Return the Y axis unit vector."""
        return Self(0.0, 1.0, 0.0)

    @staticmethod
    fn unit_z() -> Self:
        """Return the Z axis unit vector."""
        return Self(0.0, 0.0, 1.0)

    @staticmethod
    fn from_scalar(s: Float64) -> Self:
        """Create a vector with all components set to s."""
        return Self(s, s, s)

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    fn __add__(self, other: Self) -> Self:
        """Vector addition."""
        return Self(self.x + other.x, self.y + other.y, self.z + other.z)

    fn __sub__(self, other: Self) -> Self:
        """Vector subtraction."""
        return Self(self.x - other.x, self.y - other.y, self.z - other.z)

    fn __mul__(self, scalar: Float64) -> Self:
        """Scalar multiplication."""
        return Self(self.x * scalar, self.y * scalar, self.z * scalar)

    fn __rmul__(self, scalar: Float64) -> Self:
        """Scalar multiplication (reversed)."""
        return Self(self.x * scalar, self.y * scalar, self.z * scalar)

    fn __truediv__(self, scalar: Float64) -> Self:
        """Scalar division."""
        var inv = 1.0 / scalar
        return Self(self.x * inv, self.y * inv, self.z * inv)

    fn __neg__(self) -> Self:
        """Negation."""
        return Self(-self.x, -self.y, -self.z)

    fn __iadd__(mut self, other: Self):
        """In-place addition."""
        self.x += other.x
        self.y += other.y
        self.z += other.z

    fn __isub__(mut self, other: Self):
        """In-place subtraction."""
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    fn __imul__(mut self, scalar: Float64):
        """In-place scalar multiplication."""
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar

    fn __itruediv__(mut self, scalar: Float64):
        """In-place scalar division."""
        var inv = 1.0 / scalar
        self.x *= inv
        self.y *= inv
        self.z *= inv

    # =========================================================================
    # Comparison Operations
    # =========================================================================

    fn __eq__(self, other: Self) -> Bool:
        """Equality check."""
        return self.x == other.x and self.y == other.y and self.z == other.z

    fn __ne__(self, other: Self) -> Bool:
        """Inequality check."""
        return not (self == other)

    fn approx_eq(self, other: Self, tolerance: Float64 = 1e-10) -> Bool:
        """Approximate equality with tolerance."""
        return (
            abs(self.x - other.x) < tolerance
            and abs(self.y - other.y) < tolerance
            and abs(self.z - other.z) < tolerance
        )

    # =========================================================================
    # Geometric Operations
    # =========================================================================

    fn dot(self, other: Self) -> Float64:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    fn cross(self, other: Self) -> Self:
        """Cross product (self × other)."""
        return Self(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    fn length_squared(self) -> Float64:
        """Squared length (avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    fn length(self) -> Float64:
        """Euclidean length."""
        return sqrt(self.length_squared())

    fn normalized(self) -> Self:
        """Return unit vector in same direction.

        Returns zero vector if length is zero.
        """
        var len = self.length()
        if len > 1e-10:
            return self / len
        return Self.zero()

    fn normalize(mut self):
        """Normalize in place."""
        var len = self.length()
        if len > 1e-10:
            self /= len

    fn distance_to(self, other: Self) -> Float64:
        """Distance to another point."""
        return (self - other).length()

    fn distance_squared_to(self, other: Self) -> Float64:
        """Squared distance to another point (avoids sqrt)."""
        return (self - other).length_squared()

    # =========================================================================
    # Component-wise Operations
    # =========================================================================

    fn hadamard(self, other: Self) -> Self:
        """Component-wise (Hadamard) product."""
        return Self(self.x * other.x, self.y * other.y, self.z * other.z)

    fn abs(self) -> Self:
        """Component-wise absolute value."""
        return Self(abs(self.x), abs(self.y), abs(self.z))

    fn min(self, other: Self) -> Self:
        """Component-wise minimum."""
        return Self(
            min(self.x, other.x),
            min(self.y, other.y),
            min(self.z, other.z),
        )

    fn max(self, other: Self) -> Self:
        """Component-wise maximum."""
        return Self(
            max(self.x, other.x),
            max(self.y, other.y),
            max(self.z, other.z),
        )

    fn clamp(self, min_val: Self, max_val: Self) -> Self:
        """Component-wise clamp between min and max."""
        return self.max(min_val).min(max_val)

    fn clamp_length(self, max_length: Float64) -> Self:
        """Clamp vector to maximum length."""
        var len_sq = self.length_squared()
        if len_sq > max_length * max_length:
            return self.normalized() * max_length
        return self

    # =========================================================================
    # Projection and Reflection
    # =========================================================================

    fn project_onto(self, onto: Self) -> Self:
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

    fn reject_from(self, from_vec: Self) -> Self:
        """Component of this vector perpendicular to another.

        Args:
            from_vec: The vector to reject from.

        Returns:
            The rejection of self from the given vector.
        """
        return self - self.project_onto(from_vec)

    fn reflect(self, normal: Self) -> Self:
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

    fn lerp(self, other: Self, t: Float64) -> Self:
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

    fn rotated_x(self, angle: Float64) -> Self:
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

    fn rotated_y(self, angle: Float64) -> Self:
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

    fn rotated_z(self, angle: Float64) -> Self:
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

    fn __getitem__(self, index: Int) -> Float64:
        """Get component by index (0=x, 1=y, 2=z)."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            return self.z

    fn __setitem__(mut self, index: Int, value: Float64):
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

    fn to_simd(self) -> SIMD[DType.float64, 4]:
        """Convert to SIMD vector (w component = 0)."""
        return SIMD[DType.float64, 4](self.x, self.y, self.z, 0.0)

    @staticmethod
    fn from_simd(v: SIMD[DType.float64, 4]) -> Self:
        """Create from SIMD vector (ignores w component)."""
        return Self(v[0], v[1], v[2])

    fn __str__(self) -> String:
        """String representation."""
        return "Vec3(" + String(self.x) + ", " + String(self.y) + ", " + String(self.z) + ")"


# =========================================================================
# Utility Functions
# =========================================================================


fn vec3(x: Float64, y: Float64, z: Float64) -> Vec3:
    """Convenience function to create a Vec3."""
    return Vec3(x, y, z)


fn dot(a: Vec3, b: Vec3) -> Float64:
    """Dot product of two vectors."""
    return a.dot(b)


fn cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two vectors."""
    return a.cross(b)


fn normalize(v: Vec3) -> Vec3:
    """Return normalized vector."""
    return v.normalized()


fn length(v: Vec3) -> Float64:
    """Return vector length."""
    return v.length()


fn distance(a: Vec3, b: Vec3) -> Float64:
    """Distance between two points."""
    return a.distance_to(b)


fn lerp(a: Vec3, b: Vec3, t: Float64) -> Vec3:
    """Linear interpolation."""
    return a.lerp(b, t)
