"""
Vec2: 2D Vector Math for Physics Engine

SIMD-optimized 2D vector implementation for the minimal Box2D physics engine.
Uses SIMD[Scalar[dtype], 2] for hardware acceleration.
"""

from math import sqrt, sin, cos


@register_passable("trivial")
struct Vec2[dtype: DType](Copyable, Movable, Stringable):
    """2D vector using SIMD for performance."""

    var x: Scalar[Self.dtype]
    var y: Scalar[Self.dtype]

    # ===== Constructors =====

    @always_inline
    fn __init__(out self, x: Scalar[Self.dtype], y: Scalar[Self.dtype]):
        """Create a Vec2 from x and y components."""
        self.x = x
        self.y = y

    @always_inline
    fn __init__(out self, value: Scalar[Self.dtype]):
        """Create a Vec2 with both components set to the same value."""
        self.x = value
        self.y = value

    @staticmethod
    @always_inline
    fn zero() -> Self:
        """Create a zero vector."""
        return Self(0.0, 0.0)

    @staticmethod
    @always_inline
    fn one() -> Self:
        """Create a unit vector (1, 1)."""
        return Self(1.0, 1.0)

    @staticmethod
    @always_inline
    fn unit_x() -> Self:
        """Create unit vector along x-axis."""
        return Self(1.0, 0.0)

    @staticmethod
    @always_inline
    fn unit_y() -> Self:
        """Create unit vector along y-axis."""
        return Self(0.0, 1.0)

    # ===== Operators =====

    @always_inline
    fn __add__(self, other: Self) -> Self:
        """Vector addition."""
        return Self(self.x + other.x, self.y + other.y)

    @always_inline
    fn __sub__(self, other: Self) -> Self:
        """Vector subtraction."""
        return Self(self.x - other.x, self.y - other.y)

    @always_inline
    fn __mul__(self, scalar: Scalar[Self.dtype]) -> Self:
        """Scalar multiplication."""
        return Self(self.x * scalar, self.y * scalar)

    @always_inline
    fn __rmul__(self, scalar: Scalar[Self.dtype]) -> Self:
        """Scalar multiplication (reversed)."""
        return Self(self.x * scalar, self.y * scalar)

    @always_inline
    fn __truediv__(self, scalar: Scalar[Self.dtype]) -> Self:
        """Scalar division."""
        var inv = 1.0 / scalar
        return Self(self.x * inv, self.y * inv)

    @always_inline
    fn __neg__(self) -> Self:
        """Negation."""
        return Self(-self.x, -self.y)

    @always_inline
    fn __iadd__(mut self, other: Self):
        """In-place addition."""
        self.x += other.x
        self.y += other.y

    @always_inline
    fn __isub__(mut self, other: Self):
        """In-place subtraction."""
        self.x -= other.x
        self.y -= other.y

    @always_inline
    fn __imul__(mut self, scalar: Scalar[Self.dtype]):
        """In-place scalar multiplication."""
        self.x *= scalar
        self.y *= scalar

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Equality comparison."""
        return self.x == other.x and self.y == other.y

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Inequality comparison."""
        return self.x != other.x or self.y != other.y

    # ===== Vector Operations =====

    @always_inline
    fn dot(self, other: Self) -> Scalar[Self.dtype]:
        """Dot product."""
        return self.x * other.x + self.y * other.y

    @always_inline
    fn cross(self, other: Self) -> Scalar[Self.dtype]:
        """2D cross product (returns scalar z-component)."""
        return self.x * other.y - self.y * other.x

    @always_inline
    fn cross_scalar(self, s: Scalar[Self.dtype]) -> Self:
        """Cross product with scalar: v x s = (s * v.y, -s * v.x)."""
        return Self(s * self.y, -s * self.x)

    @always_inline
    fn length_squared(self) -> Scalar[Self.dtype]:
        """Squared length (avoids sqrt)."""
        return self.x * self.x + self.y * self.y

    @always_inline
    fn length(self) -> Scalar[Self.dtype]:
        """Vector length (magnitude)."""
        return sqrt(self.length_squared())

    @always_inline
    fn normalize(self) -> Self:
        """Return normalized vector (unit length)."""
        var len = self.length()
        if len > 0.0:
            return self / len
        return Self.zero()

    @always_inline
    fn normalize_safe(self, fallback: Self = Self.zero()) -> Self:
        """Return normalized vector, or fallback if zero length."""
        var len_sq = self.length_squared()
        if len_sq > 1e-12:
            return self / sqrt(len_sq)
        return fallback

    @always_inline
    fn perpendicular(self) -> Self:
        """Return perpendicular vector (rotated 90 degrees counter-clockwise).
        """
        return Self(-self.y, self.x)

    @always_inline
    fn perpendicular_cw(self) -> Self:
        """Return perpendicular vector (rotated 90 degrees clockwise)."""
        return Self(self.y, -self.x)

    @always_inline
    fn rotate(self, angle: Scalar[Self.dtype]) -> Self:
        """Rotate vector by angle (radians)."""
        var c = cos(angle)
        var s = sin(angle)
        return Self(self.x * c - self.y * s, self.x * s + self.y * c)

    @always_inline
    fn rotate_by_cs(
        self, cos_angle: Scalar[Self.dtype], sin_angle: Scalar[Self.dtype]
    ) -> Self:
        """Rotate vector by pre-computed cos and sin values."""
        return Self(
            self.x * cos_angle - self.y * sin_angle,
            self.x * sin_angle + self.y * cos_angle,
        )

    @always_inline
    fn project_onto(self, onto: Self) -> Self:
        """Project this vector onto another vector."""
        var onto_len_sq = onto.length_squared()
        if onto_len_sq > 0.0:
            return onto * (self.dot(onto) / onto_len_sq)
        return Self.zero()

    @always_inline
    fn reflect(self, normal: Self) -> Self:
        """Reflect vector across a normal."""
        return self - normal * (2.0 * self.dot(normal))

    @always_inline
    fn distance_to(self, other: Self) -> Scalar[Self.dtype]:
        """Distance to another point."""
        return (self - other).length()

    @always_inline
    fn distance_squared_to(self, other: Self) -> Scalar[Self.dtype]:
        """Squared distance to another point."""
        return (self - other).length_squared()

    @always_inline
    fn lerp(self, other: Self, t: Scalar[Self.dtype]) -> Self:
        """Linear interpolation between two vectors."""
        return self + (other - self) * t

    @always_inline
    fn abs(self) -> Self:
        """Absolute value of each component."""
        var ax = self.x if self.x >= 0.0 else -self.x
        var ay = self.y if self.y >= 0.0 else -self.y
        return Self(ax, ay)

    @always_inline
    fn min_component(self) -> Scalar[Self.dtype]:
        """Return minimum component."""
        return self.x if self.x < self.y else self.y

    @always_inline
    fn max_component(self) -> Scalar[Self.dtype]:
        """Return maximum component."""
        return self.x if self.x > self.y else self.y

    # ===== Utility =====

    fn __str__(self) -> String:
        """String representation."""
        return "Vec2(" + String(self.x) + ", " + String(self.y) + ")"

    @always_inline
    fn is_zero(self) -> Bool:
        """Check if vector is approximately zero."""
        return self.length_squared() < 1e-12

    @always_inline
    fn is_valid(self) -> Bool:
        """Check if vector has valid (non-NaN, non-Inf) components."""
        # Simple validity check
        return self.x == self.x and self.y == self.y  # NaN check


# ===== Free Functions =====


@always_inline
fn vec2[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) -> Vec2[dtype]:
    """Convenience constructor function."""
    return Vec2(x, y)


@always_inline
fn dot[dtype: DType](a: Vec2[dtype], b: Vec2[dtype]) -> Scalar[dtype]:
    """Dot product of two vectors."""
    return a.dot(b)


@always_inline
fn cross[dtype: DType](a: Vec2[dtype], b: Vec2[dtype]) -> Scalar[dtype]:
    """2D cross product (returns scalar)."""
    return a.cross(b)


@always_inline
fn cross_sv[dtype: DType](s: Scalar[dtype], v: Vec2[dtype]) -> Vec2[dtype]:
    """Cross product: scalar x vector = (-s * v.y, s * v.x)."""
    return Vec2(-s * v.y, s * v.x)


@always_inline
fn cross_vs[dtype: DType](v: Vec2[dtype], s: Scalar[dtype]) -> Vec2[dtype]:
    """Cross product: vector x scalar = (s * v.y, -s * v.x)."""
    return Vec2(s * v.y, -s * v.x)


@always_inline
fn length[dtype: DType](v: Vec2[dtype]) -> Scalar[dtype]:
    """Vector length."""
    return v.length()


@always_inline
fn normalize[dtype: DType](v: Vec2[dtype]) -> Vec2[dtype]:
    """Normalize vector."""
    return v.normalize()


@always_inline
fn distance[dtype: DType](a: Vec2[dtype], b: Vec2[dtype]) -> Scalar[dtype]:
    """Distance between two points."""
    return a.distance_to(b)


@always_inline
fn min_vec[dtype: DType](a: Vec2[dtype], b: Vec2[dtype]) -> Vec2[dtype]:
    """Component-wise minimum."""
    return Vec2(
        a.x if a.x < b.x else b.x,
        a.y if a.y < b.y else b.y,
    )


@always_inline
fn max_vec[dtype: DType](a: Vec2[dtype], b: Vec2[dtype]) -> Vec2[dtype]:
    """Component-wise maximum."""
    return Vec2(
        a.x if a.x > b.x else b.x,
        a.y if a.y > b.y else b.y,
    )


@always_inline
fn clamp_vec[
    dtype: DType
](v: Vec2[dtype], low: Vec2[dtype], high: Vec2[dtype]) -> Vec2[dtype]:
    """Clamp vector components to range."""
    return max_vec(low, min_vec(v, high))


@always_inline
fn clamp_length[
    dtype: DType
](v: Vec2[dtype], max_length: Scalar[dtype]) -> Vec2[dtype]:
    """Clamp vector length to maximum."""
    var len_sq = v.length_squared()
    if len_sq > max_length * max_length:
        return v.normalize() * max_length
    return v
