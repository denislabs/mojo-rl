"""3x3 Matrix type for rotations and linear transformations.

Provides Mat3 struct for 3D rotation matrices and general 3x3 linear transforms.
Uses row-major storage: m[row, col] where row is first index.
"""

from math import sqrt, cos, sin

from .vec3 import Vec3
from .quat import Quat


struct Mat3(ImplicitlyCopyable, Movable, Stringable):
    """3x3 matrix for rotations and linear transformations.

    Row-major storage: elements are stored as [row0, row1, row2].
    Matrix-vector multiplication: v' = M * v (column vector convention).

    Memory layout:
        [m00, m01, m02]
        [m10, m11, m12]
        [m20, m21, m22]
    """

    var m00: Float64
    var m01: Float64
    var m02: Float64
    var m10: Float64
    var m11: Float64
    var m12: Float64
    var m20: Float64
    var m21: Float64
    var m22: Float64

    # =========================================================================
    # Constructors
    # =========================================================================

    fn __init__(out self):
        """Initialize to identity matrix."""
        self.m00 = 1.0
        self.m01 = 0.0
        self.m02 = 0.0
        self.m10 = 0.0
        self.m11 = 1.0
        self.m12 = 0.0
        self.m20 = 0.0
        self.m21 = 0.0
        self.m22 = 1.0

    fn __init__(
        out self,
        m00: Float64,
        m01: Float64,
        m02: Float64,
        m10: Float64,
        m11: Float64,
        m12: Float64,
        m20: Float64,
        m21: Float64,
        m22: Float64,
    ):
        """Initialize with explicit elements (row-major order)."""
        self.m00 = m00
        self.m01 = m01
        self.m02 = m02
        self.m10 = m10
        self.m11 = m11
        self.m12 = m12
        self.m20 = m20
        self.m21 = m21
        self.m22 = m22

    fn __init__(out self, row0: Vec3, row1: Vec3, row2: Vec3):
        """Initialize from row vectors."""
        self.m00 = row0.x
        self.m01 = row0.y
        self.m02 = row0.z
        self.m10 = row1.x
        self.m11 = row1.y
        self.m12 = row1.z
        self.m20 = row2.x
        self.m21 = row2.y
        self.m22 = row2.z

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @staticmethod
    fn identity() -> Self:
        """Return the identity matrix."""
        return Self()

    @staticmethod
    fn zero() -> Self:
        """Return the zero matrix."""
        return Self(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @staticmethod
    fn diagonal(d: Float64) -> Self:
        """Return a diagonal matrix with d on diagonal."""
        return Self(d, 0.0, 0.0, 0.0, d, 0.0, 0.0, 0.0, d)

    @staticmethod
    fn diagonal(d: Vec3) -> Self:
        """Return a diagonal matrix from a vector."""
        return Self(d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z)

    @staticmethod
    fn from_rows(row0: Vec3, row1: Vec3, row2: Vec3) -> Self:
        """Create matrix from row vectors."""
        return Self(row0, row1, row2)

    @staticmethod
    fn from_cols(col0: Vec3, col1: Vec3, col2: Vec3) -> Self:
        """Create matrix from column vectors."""
        return Self(
            col0.x,
            col1.x,
            col2.x,
            col0.y,
            col1.y,
            col2.y,
            col0.z,
            col1.z,
            col2.z,
        )

    @staticmethod
    fn from_scale(scale: Vec3) -> Self:
        """Create a scaling matrix."""
        return Self.diagonal(scale)

    @staticmethod
    fn from_scale(s: Float64) -> Self:
        """Create a uniform scaling matrix."""
        return Self.diagonal(s)

    # =========================================================================
    # Rotation Matrix Factories
    # =========================================================================

    @staticmethod
    fn rotation_x(angle: Float64) -> Self:
        """Create rotation matrix around X axis.

        Args:
            angle: Rotation angle in radians.
        """
        var c = cos(angle)
        var s = sin(angle)
        return Self(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)

    @staticmethod
    fn rotation_y(angle: Float64) -> Self:
        """Create rotation matrix around Y axis.

        Args:
            angle: Rotation angle in radians.
        """
        var c = cos(angle)
        var s = sin(angle)
        return Self(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)

    @staticmethod
    fn rotation_z(angle: Float64) -> Self:
        """Create rotation matrix around Z axis.

        Args:
            angle: Rotation angle in radians.
        """
        var c = cos(angle)
        var s = sin(angle)
        return Self(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)

    @staticmethod
    fn rotation_axis(axis: Vec3, angle: Float64) -> Self:
        """Create rotation matrix around arbitrary axis.

        Args:
            axis: Rotation axis (will be normalized).
            angle: Rotation angle in radians.
        """
        var n = axis.normalized()
        var c = cos(angle)
        var s = sin(angle)
        var t = 1.0 - c

        return Self(
            t * n.x * n.x + c,
            t * n.x * n.y - s * n.z,
            t * n.x * n.z + s * n.y,
            t * n.x * n.y + s * n.z,
            t * n.y * n.y + c,
            t * n.y * n.z - s * n.x,
            t * n.x * n.z - s * n.y,
            t * n.y * n.z + s * n.x,
            t * n.z * n.z + c,
        )

    @staticmethod
    fn from_quat(q: Quat) -> Self:
        """Create rotation matrix from quaternion.

        Args:
            q: Unit quaternion representing rotation.
        """
        var x2 = q.x + q.x
        var y2 = q.y + q.y
        var z2 = q.z + q.z

        var xx2 = q.x * x2
        var xy2 = q.x * y2
        var xz2 = q.x * z2
        var yy2 = q.y * y2
        var yz2 = q.y * z2
        var zz2 = q.z * z2
        var wx2 = q.w * x2
        var wy2 = q.w * y2
        var wz2 = q.w * z2

        return Self(
            1.0 - yy2 - zz2,
            xy2 - wz2,
            xz2 + wy2,
            xy2 + wz2,
            1.0 - xx2 - zz2,
            yz2 - wx2,
            xz2 - wy2,
            yz2 + wx2,
            1.0 - xx2 - yy2,
        )

    @staticmethod
    fn skew(v: Vec3) -> Self:
        """Create skew-symmetric matrix from vector (for cross product).

        skew(v) * u = v × u
        """
        return Self(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)

    @staticmethod
    fn outer(a: Vec3, b: Vec3) -> Self:
        """Create outer product matrix (a ⊗ b).

        Result: M[i,j] = a[i] * b[j]
        """
        return Self(
            a.x * b.x,
            a.x * b.y,
            a.x * b.z,
            a.y * b.x,
            a.y * b.y,
            a.y * b.z,
            a.z * b.x,
            a.z * b.y,
            a.z * b.z,
        )

    # =========================================================================
    # Row/Column Access
    # =========================================================================

    fn row(self, i: Int) -> Vec3:
        """Get row i as a vector."""
        if i == 0:
            return Vec3(self.m00, self.m01, self.m02)
        elif i == 1:
            return Vec3(self.m10, self.m11, self.m12)
        else:
            return Vec3(self.m20, self.m21, self.m22)

    fn col(self, j: Int) -> Vec3:
        """Get column j as a vector."""
        if j == 0:
            return Vec3(self.m00, self.m10, self.m20)
        elif j == 1:
            return Vec3(self.m01, self.m11, self.m21)
        else:
            return Vec3(self.m02, self.m12, self.m22)

    fn set_row(mut self, i: Int, v: Vec3):
        """Set row i from a vector."""
        if i == 0:
            self.m00 = v.x
            self.m01 = v.y
            self.m02 = v.z
        elif i == 1:
            self.m10 = v.x
            self.m11 = v.y
            self.m12 = v.z
        else:
            self.m20 = v.x
            self.m21 = v.y
            self.m22 = v.z

    fn set_col(mut self, j: Int, v: Vec3):
        """Set column j from a vector."""
        if j == 0:
            self.m00 = v.x
            self.m10 = v.y
            self.m20 = v.z
        elif j == 1:
            self.m01 = v.x
            self.m11 = v.y
            self.m21 = v.z
        else:
            self.m02 = v.x
            self.m12 = v.y
            self.m22 = v.z

    # =========================================================================
    # Matrix Operations
    # =========================================================================

    fn __add__(self, other: Self) -> Self:
        """Matrix addition."""
        return Self(
            self.m00 + other.m00,
            self.m01 + other.m01,
            self.m02 + other.m02,
            self.m10 + other.m10,
            self.m11 + other.m11,
            self.m12 + other.m12,
            self.m20 + other.m20,
            self.m21 + other.m21,
            self.m22 + other.m22,
        )

    fn __sub__(self, other: Self) -> Self:
        """Matrix subtraction."""
        return Self(
            self.m00 - other.m00,
            self.m01 - other.m01,
            self.m02 - other.m02,
            self.m10 - other.m10,
            self.m11 - other.m11,
            self.m12 - other.m12,
            self.m20 - other.m20,
            self.m21 - other.m21,
            self.m22 - other.m22,
        )

    fn __mul__(self, scalar: Float64) -> Self:
        """Scalar multiplication."""
        return Self(
            self.m00 * scalar,
            self.m01 * scalar,
            self.m02 * scalar,
            self.m10 * scalar,
            self.m11 * scalar,
            self.m12 * scalar,
            self.m20 * scalar,
            self.m21 * scalar,
            self.m22 * scalar,
        )

    fn __mul__(self, v: Vec3) -> Vec3:
        """Matrix-vector multiplication (M * v)."""
        return Vec3(
            self.m00 * v.x + self.m01 * v.y + self.m02 * v.z,
            self.m10 * v.x + self.m11 * v.y + self.m12 * v.z,
            self.m20 * v.x + self.m21 * v.y + self.m22 * v.z,
        )

    fn __matmul__(self, other: Self) -> Self:
        """Matrix-matrix multiplication (self @ other)."""
        return Self(
            self.m00 * other.m00 + self.m01 * other.m10 + self.m02 * other.m20,
            self.m00 * other.m01 + self.m01 * other.m11 + self.m02 * other.m21,
            self.m00 * other.m02 + self.m01 * other.m12 + self.m02 * other.m22,
            self.m10 * other.m00 + self.m11 * other.m10 + self.m12 * other.m20,
            self.m10 * other.m01 + self.m11 * other.m11 + self.m12 * other.m21,
            self.m10 * other.m02 + self.m11 * other.m12 + self.m12 * other.m22,
            self.m20 * other.m00 + self.m21 * other.m10 + self.m22 * other.m20,
            self.m20 * other.m01 + self.m21 * other.m11 + self.m22 * other.m21,
            self.m20 * other.m02 + self.m21 * other.m12 + self.m22 * other.m22,
        )

    fn __neg__(self) -> Self:
        """Negation."""
        return Self(
            -self.m00,
            -self.m01,
            -self.m02,
            -self.m10,
            -self.m11,
            -self.m12,
            -self.m20,
            -self.m21,
            -self.m22,
        )

    fn transpose(self) -> Self:
        """Return transposed matrix."""
        return Self(
            self.m00,
            self.m10,
            self.m20,
            self.m01,
            self.m11,
            self.m21,
            self.m02,
            self.m12,
            self.m22,
        )

    fn determinant(self) -> Float64:
        """Compute determinant."""
        return (
            self.m00 * (self.m11 * self.m22 - self.m12 * self.m21)
            - self.m01 * (self.m10 * self.m22 - self.m12 * self.m20)
            + self.m02 * (self.m10 * self.m21 - self.m11 * self.m20)
        )

    fn inverse(self) -> Self:
        """Return inverse matrix.

        Returns identity if matrix is singular.
        """
        var det = self.determinant()
        if abs(det) < 1e-10:
            return Self.identity()

        var inv_det = 1.0 / det
        return Self(
            (self.m11 * self.m22 - self.m12 * self.m21) * inv_det,
            (self.m02 * self.m21 - self.m01 * self.m22) * inv_det,
            (self.m01 * self.m12 - self.m02 * self.m11) * inv_det,
            (self.m12 * self.m20 - self.m10 * self.m22) * inv_det,
            (self.m00 * self.m22 - self.m02 * self.m20) * inv_det,
            (self.m02 * self.m10 - self.m00 * self.m12) * inv_det,
            (self.m10 * self.m21 - self.m11 * self.m20) * inv_det,
            (self.m01 * self.m20 - self.m00 * self.m21) * inv_det,
            (self.m00 * self.m11 - self.m01 * self.m10) * inv_det,
        )

    fn trace(self) -> Float64:
        """Return trace (sum of diagonal elements)."""
        return self.m00 + self.m11 + self.m22

    # =========================================================================
    # Conversion to Quaternion
    # =========================================================================

    fn to_quat(self) -> Quat:
        """Convert rotation matrix to quaternion.

        Assumes this is a valid rotation matrix (orthogonal with det = 1).
        """
        var tr = self.trace()

        if tr > 0:
            var s = sqrt(tr + 1.0) * 2.0
            return Quat(
                0.25 * s,
                (self.m21 - self.m12) / s,
                (self.m02 - self.m20) / s,
                (self.m10 - self.m01) / s,
            )
        elif self.m00 > self.m11 and self.m00 > self.m22:
            var s = sqrt(1.0 + self.m00 - self.m11 - self.m22) * 2.0
            return Quat(
                (self.m21 - self.m12) / s,
                0.25 * s,
                (self.m01 + self.m10) / s,
                (self.m02 + self.m20) / s,
            )
        elif self.m11 > self.m22:
            var s = sqrt(1.0 + self.m11 - self.m00 - self.m22) * 2.0
            return Quat(
                (self.m02 - self.m20) / s,
                (self.m01 + self.m10) / s,
                0.25 * s,
                (self.m12 + self.m21) / s,
            )
        else:
            var s = sqrt(1.0 + self.m22 - self.m00 - self.m11) * 2.0
            return Quat(
                (self.m10 - self.m01) / s,
                (self.m02 + self.m20) / s,
                (self.m12 + self.m21) / s,
                0.25 * s,
            )

    # =========================================================================
    # Comparison
    # =========================================================================

    fn __eq__(self, other: Self) -> Bool:
        """Equality check."""
        return (
            self.m00 == other.m00
            and self.m01 == other.m01
            and self.m02 == other.m02
            and self.m10 == other.m10
            and self.m11 == other.m11
            and self.m12 == other.m12
            and self.m20 == other.m20
            and self.m21 == other.m21
            and self.m22 == other.m22
        )

    fn __ne__(self, other: Self) -> Bool:
        """Inequality check."""
        return not (self == other)

    fn approx_eq(self, other: Self, tolerance: Float64 = 1e-10) -> Bool:
        """Approximate equality with tolerance."""
        return (
            abs(self.m00 - other.m00) < tolerance
            and abs(self.m01 - other.m01) < tolerance
            and abs(self.m02 - other.m02) < tolerance
            and abs(self.m10 - other.m10) < tolerance
            and abs(self.m11 - other.m11) < tolerance
            and abs(self.m12 - other.m12) < tolerance
            and abs(self.m20 - other.m20) < tolerance
            and abs(self.m21 - other.m21) < tolerance
            and abs(self.m22 - other.m22) < tolerance
        )

    # =========================================================================
    # String Conversion
    # =========================================================================

    fn __str__(self) -> String:
        """String representation."""
        return (
            "Mat3(\n  ["
            + String(self.m00)
            + ", "
            + String(self.m01)
            + ", "
            + String(self.m02)
            + "],\n  ["
            + String(self.m10)
            + ", "
            + String(self.m11)
            + ", "
            + String(self.m12)
            + "],\n  ["
            + String(self.m20)
            + ", "
            + String(self.m21)
            + ", "
            + String(self.m22)
            + "]\n)"
        )


# =========================================================================
# Utility Functions
# =========================================================================


fn mat3_identity() -> Mat3:
    """Return identity matrix."""
    return Mat3.identity()


fn mat3_rotation_x(angle: Float64) -> Mat3:
    """Create rotation matrix around X axis."""
    return Mat3.rotation_x(angle)


fn mat3_rotation_y(angle: Float64) -> Mat3:
    """Create rotation matrix around Y axis."""
    return Mat3.rotation_y(angle)


fn mat3_rotation_z(angle: Float64) -> Mat3:
    """Create rotation matrix around Z axis."""
    return Mat3.rotation_z(angle)
