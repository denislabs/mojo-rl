"""4x4 Matrix type for 3D transformations.

Provides Mat4 struct for affine transformations (rotation + translation + scale)
and perspective projection matrices for rendering.
Uses row-major storage: m[row, col] where row is first index.
"""

from math import sqrt, cos, sin, tan

from .vec3 import Vec3
from .mat3 import Mat3
from .quat import Quat


struct Mat4(ImplicitlyCopyable, Movable, Stringable):
    """4x4 matrix for 3D transformations.

    Row-major storage. Affine transforms store rotation/scale in top-left 3x3
    and translation in the last column:
        [r00, r01, r02, tx]
        [r10, r11, r12, ty]
        [r20, r21, r22, tz]
        [0,   0,   0,   1 ]

    Homogeneous coordinates: point (x,y,z) -> (x,y,z,1), vector (x,y,z) -> (x,y,z,0)
    """

    # Row 0
    var m00: Float64
    var m01: Float64
    var m02: Float64
    var m03: Float64
    # Row 1
    var m10: Float64
    var m11: Float64
    var m12: Float64
    var m13: Float64
    # Row 2
    var m20: Float64
    var m21: Float64
    var m22: Float64
    var m23: Float64
    # Row 3
    var m30: Float64
    var m31: Float64
    var m32: Float64
    var m33: Float64

    # =========================================================================
    # Constructors
    # =========================================================================

    fn __init__(out self):
        """Initialize to identity matrix."""
        self.m00 = 1.0
        self.m01 = 0.0
        self.m02 = 0.0
        self.m03 = 0.0
        self.m10 = 0.0
        self.m11 = 1.0
        self.m12 = 0.0
        self.m13 = 0.0
        self.m20 = 0.0
        self.m21 = 0.0
        self.m22 = 1.0
        self.m23 = 0.0
        self.m30 = 0.0
        self.m31 = 0.0
        self.m32 = 0.0
        self.m33 = 1.0

    fn __init__(
        out self,
        m00: Float64,
        m01: Float64,
        m02: Float64,
        m03: Float64,
        m10: Float64,
        m11: Float64,
        m12: Float64,
        m13: Float64,
        m20: Float64,
        m21: Float64,
        m22: Float64,
        m23: Float64,
        m30: Float64,
        m31: Float64,
        m32: Float64,
        m33: Float64,
    ):
        """Initialize with explicit elements (row-major order)."""
        self.m00 = m00
        self.m01 = m01
        self.m02 = m02
        self.m03 = m03
        self.m10 = m10
        self.m11 = m11
        self.m12 = m12
        self.m13 = m13
        self.m20 = m20
        self.m21 = m21
        self.m22 = m22
        self.m23 = m23
        self.m30 = m30
        self.m31 = m31
        self.m32 = m32
        self.m33 = m33

    fn __init__(out self, rotation: Mat3, translation: Vec3):
        """Initialize from rotation matrix and translation."""
        self.m00 = rotation.m00
        self.m01 = rotation.m01
        self.m02 = rotation.m02
        self.m03 = translation.x
        self.m10 = rotation.m10
        self.m11 = rotation.m11
        self.m12 = rotation.m12
        self.m13 = translation.y
        self.m20 = rotation.m20
        self.m21 = rotation.m21
        self.m22 = rotation.m22
        self.m23 = translation.z
        self.m30 = 0.0
        self.m31 = 0.0
        self.m32 = 0.0
        self.m33 = 1.0

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
        return Self(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    @staticmethod
    fn from_translation(t: Vec3) -> Self:
        """Create a translation matrix."""
        return Self(
            1.0,
            0.0,
            0.0,
            t.x,
            0.0,
            1.0,
            0.0,
            t.y,
            0.0,
            0.0,
            1.0,
            t.z,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @staticmethod
    fn from_scale(s: Vec3) -> Self:
        """Create a scaling matrix."""
        return Self(
            s.x,
            0.0,
            0.0,
            0.0,
            0.0,
            s.y,
            0.0,
            0.0,
            0.0,
            0.0,
            s.z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @staticmethod
    fn from_scale(s: Float64) -> Self:
        """Create a uniform scaling matrix."""
        return Self(
            s,
            0.0,
            0.0,
            0.0,
            0.0,
            s,
            0.0,
            0.0,
            0.0,
            0.0,
            s,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @staticmethod
    fn from_rotation(r: Mat3) -> Self:
        """Create transformation from rotation matrix."""
        return Self(r, Vec3.zero())

    @staticmethod
    fn from_quat(q: Quat) -> Self:
        """Create transformation from quaternion rotation."""
        return Self(Mat3.from_quat(q), Vec3.zero())

    @staticmethod
    fn from_quat(q: Quat, translation: Vec3) -> Self:
        """Create transformation from quaternion and translation."""
        return Self(Mat3.from_quat(q), translation)

    @staticmethod
    fn compose(translation: Vec3, rotation: Quat, scale: Vec3) -> Self:
        """Compose a transformation from TRS components.

        Order: scale, then rotate, then translate.
        """
        var r = Mat3.from_quat(rotation)

        return Self(
            r.m00 * scale.x,
            r.m01 * scale.y,
            r.m02 * scale.z,
            translation.x,
            r.m10 * scale.x,
            r.m11 * scale.y,
            r.m12 * scale.z,
            translation.y,
            r.m20 * scale.x,
            r.m21 * scale.y,
            r.m22 * scale.z,
            translation.z,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    # =========================================================================
    # Rotation Matrix Factories
    # =========================================================================

    @staticmethod
    fn rotation_x(angle: Float64) -> Self:
        """Create rotation matrix around X axis."""
        var c = cos(angle)
        var s = sin(angle)
        return Self(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            c,
            -s,
            0.0,
            0.0,
            s,
            c,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @staticmethod
    fn rotation_y(angle: Float64) -> Self:
        """Create rotation matrix around Y axis."""
        var c = cos(angle)
        var s = sin(angle)
        return Self(
            c,
            0.0,
            s,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            -s,
            0.0,
            c,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @staticmethod
    fn rotation_z(angle: Float64) -> Self:
        """Create rotation matrix around Z axis."""
        var c = cos(angle)
        var s = sin(angle)
        return Self(
            c,
            -s,
            0.0,
            0.0,
            s,
            c,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @staticmethod
    fn rotation_axis(axis: Vec3, angle: Float64) -> Self:
        """Create rotation matrix around arbitrary axis."""
        return Self.from_rotation(Mat3.rotation_axis(axis, angle))

    # =========================================================================
    # View and Projection Matrices (for Rendering)
    # =========================================================================

    @staticmethod
    fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self:
        """Create a view matrix looking from eye toward target.

        Args:
            eye: Camera position.
            target: Point to look at.
            up: Up direction (usually Y-up).

        Returns:
            View matrix that transforms world coordinates to camera space.
        """
        var f = (target - eye).normalized()  # Forward
        var r = f.cross(up).normalized()  # Right
        var u = r.cross(f)  # Up (recalculated)

        return Self(
            r.x,
            r.y,
            r.z,
            -r.dot(eye),
            u.x,
            u.y,
            u.z,
            -u.dot(eye),
            -f.x,
            -f.y,
            -f.z,
            f.dot(eye),
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @staticmethod
    fn perspective(fov_y: Float64, aspect: Float64, near: Float64, far: Float64) -> Self:
        """Create a perspective projection matrix.

        Args:
            fov_y: Vertical field of view in radians.
            aspect: Width/height aspect ratio.
            near: Near clipping plane distance.
            far: Far clipping plane distance.

        Returns:
            Perspective projection matrix.
        """
        var tan_half_fov = tan(fov_y * 0.5)
        var f = 1.0 / tan_half_fov
        var range_inv = 1.0 / (near - far)

        return Self(
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            (far + near) * range_inv,
            2.0 * far * near * range_inv,
            0.0,
            0.0,
            -1.0,
            0.0,
        )

    @staticmethod
    fn orthographic(
        left: Float64,
        right: Float64,
        bottom: Float64,
        top: Float64,
        near: Float64,
        far: Float64,
    ) -> Self:
        """Create an orthographic projection matrix.

        Args:
            left: Left clipping plane.
            right: Right clipping plane.
            bottom: Bottom clipping plane.
            top: Top clipping plane.
            near: Near clipping plane.
            far: Far clipping plane.

        Returns:
            Orthographic projection matrix.
        """
        var rl = 1.0 / (right - left)
        var tb = 1.0 / (top - bottom)
        var fn = 1.0 / (far - near)

        return Self(
            2.0 * rl,
            0.0,
            0.0,
            -(right + left) * rl,
            0.0,
            2.0 * tb,
            0.0,
            -(top + bottom) * tb,
            0.0,
            0.0,
            -2.0 * fn,
            -(far + near) * fn,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    # =========================================================================
    # Component Access
    # =========================================================================

    fn get_translation(self) -> Vec3:
        """Extract translation component."""
        return Vec3(self.m03, self.m13, self.m23)

    fn get_rotation(self) -> Mat3:
        """Extract rotation/scale component as 3x3 matrix."""
        return Mat3(
            self.m00,
            self.m01,
            self.m02,
            self.m10,
            self.m11,
            self.m12,
            self.m20,
            self.m21,
            self.m22,
        )

    fn get_scale(self) -> Vec3:
        """Extract scale from the matrix (assuming no shear)."""
        return Vec3(
            Vec3(self.m00, self.m10, self.m20).length(),
            Vec3(self.m01, self.m11, self.m21).length(),
            Vec3(self.m02, self.m12, self.m22).length(),
        )

    fn set_translation(mut self, t: Vec3):
        """Set translation component."""
        self.m03 = t.x
        self.m13 = t.y
        self.m23 = t.z

    # =========================================================================
    # Matrix Operations
    # =========================================================================

    fn __add__(self, other: Self) -> Self:
        """Matrix addition."""
        return Self(
            self.m00 + other.m00,
            self.m01 + other.m01,
            self.m02 + other.m02,
            self.m03 + other.m03,
            self.m10 + other.m10,
            self.m11 + other.m11,
            self.m12 + other.m12,
            self.m13 + other.m13,
            self.m20 + other.m20,
            self.m21 + other.m21,
            self.m22 + other.m22,
            self.m23 + other.m23,
            self.m30 + other.m30,
            self.m31 + other.m31,
            self.m32 + other.m32,
            self.m33 + other.m33,
        )

    fn __sub__(self, other: Self) -> Self:
        """Matrix subtraction."""
        return Self(
            self.m00 - other.m00,
            self.m01 - other.m01,
            self.m02 - other.m02,
            self.m03 - other.m03,
            self.m10 - other.m10,
            self.m11 - other.m11,
            self.m12 - other.m12,
            self.m13 - other.m13,
            self.m20 - other.m20,
            self.m21 - other.m21,
            self.m22 - other.m22,
            self.m23 - other.m23,
            self.m30 - other.m30,
            self.m31 - other.m31,
            self.m32 - other.m32,
            self.m33 - other.m33,
        )

    fn __mul__(self, scalar: Float64) -> Self:
        """Scalar multiplication."""
        return Self(
            self.m00 * scalar,
            self.m01 * scalar,
            self.m02 * scalar,
            self.m03 * scalar,
            self.m10 * scalar,
            self.m11 * scalar,
            self.m12 * scalar,
            self.m13 * scalar,
            self.m20 * scalar,
            self.m21 * scalar,
            self.m22 * scalar,
            self.m23 * scalar,
            self.m30 * scalar,
            self.m31 * scalar,
            self.m32 * scalar,
            self.m33 * scalar,
        )

    fn __matmul__(self, other: Self) -> Self:
        """Matrix-matrix multiplication (self @ other)."""
        return Self(
            self.m00 * other.m00
            + self.m01 * other.m10
            + self.m02 * other.m20
            + self.m03 * other.m30,
            self.m00 * other.m01
            + self.m01 * other.m11
            + self.m02 * other.m21
            + self.m03 * other.m31,
            self.m00 * other.m02
            + self.m01 * other.m12
            + self.m02 * other.m22
            + self.m03 * other.m32,
            self.m00 * other.m03
            + self.m01 * other.m13
            + self.m02 * other.m23
            + self.m03 * other.m33,
            self.m10 * other.m00
            + self.m11 * other.m10
            + self.m12 * other.m20
            + self.m13 * other.m30,
            self.m10 * other.m01
            + self.m11 * other.m11
            + self.m12 * other.m21
            + self.m13 * other.m31,
            self.m10 * other.m02
            + self.m11 * other.m12
            + self.m12 * other.m22
            + self.m13 * other.m32,
            self.m10 * other.m03
            + self.m11 * other.m13
            + self.m12 * other.m23
            + self.m13 * other.m33,
            self.m20 * other.m00
            + self.m21 * other.m10
            + self.m22 * other.m20
            + self.m23 * other.m30,
            self.m20 * other.m01
            + self.m21 * other.m11
            + self.m22 * other.m21
            + self.m23 * other.m31,
            self.m20 * other.m02
            + self.m21 * other.m12
            + self.m22 * other.m22
            + self.m23 * other.m32,
            self.m20 * other.m03
            + self.m21 * other.m13
            + self.m22 * other.m23
            + self.m23 * other.m33,
            self.m30 * other.m00
            + self.m31 * other.m10
            + self.m32 * other.m20
            + self.m33 * other.m30,
            self.m30 * other.m01
            + self.m31 * other.m11
            + self.m32 * other.m21
            + self.m33 * other.m31,
            self.m30 * other.m02
            + self.m31 * other.m12
            + self.m32 * other.m22
            + self.m33 * other.m32,
            self.m30 * other.m03
            + self.m31 * other.m13
            + self.m32 * other.m23
            + self.m33 * other.m33,
        )

    fn transform_point(self, p: Vec3) -> Vec3:
        """Transform a point (applies translation).

        Treats p as (x, y, z, 1) and returns (x', y', z').
        """
        return Vec3(
            self.m00 * p.x + self.m01 * p.y + self.m02 * p.z + self.m03,
            self.m10 * p.x + self.m11 * p.y + self.m12 * p.z + self.m13,
            self.m20 * p.x + self.m21 * p.y + self.m22 * p.z + self.m23,
        )

    fn transform_vector(self, v: Vec3) -> Vec3:
        """Transform a direction vector (no translation).

        Treats v as (x, y, z, 0) and returns (x', y', z').
        """
        return Vec3(
            self.m00 * v.x + self.m01 * v.y + self.m02 * v.z,
            self.m10 * v.x + self.m11 * v.y + self.m12 * v.z,
            self.m20 * v.x + self.m21 * v.y + self.m22 * v.z,
        )

    fn transform_normal(self, n: Vec3) -> Vec3:
        """Transform a normal vector.

        Normals transform by the inverse transpose of the upper 3x3.
        This assumes orthogonal transform (rotation only).
        For non-uniform scale, use inverse transpose explicitly.
        """
        # For orthogonal matrices, inverse transpose = the matrix itself
        return self.transform_vector(n).normalized()

    fn transpose(self) -> Self:
        """Return transposed matrix."""
        return Self(
            self.m00,
            self.m10,
            self.m20,
            self.m30,
            self.m01,
            self.m11,
            self.m21,
            self.m31,
            self.m02,
            self.m12,
            self.m22,
            self.m32,
            self.m03,
            self.m13,
            self.m23,
            self.m33,
        )

    fn inverse_affine(self) -> Self:
        """Fast inverse for affine transforms (rotation + translation).

        Assumes bottom row is (0, 0, 0, 1) and no scale/shear.
        For general 4x4 inverse, use inverse() instead.
        """
        # Inverse of rotation is transpose
        var inv_rot = Mat3(
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

        # Inverse translation: -R^T * t
        var t = Vec3(self.m03, self.m13, self.m23)
        var inv_t = inv_rot * (-t)

        return Self(inv_rot, inv_t)

    fn determinant(self) -> Float64:
        """Compute determinant of the 4x4 matrix."""
        var a = self.m00 * (
            self.m11 * (self.m22 * self.m33 - self.m23 * self.m32)
            - self.m12 * (self.m21 * self.m33 - self.m23 * self.m31)
            + self.m13 * (self.m21 * self.m32 - self.m22 * self.m31)
        )
        var b = self.m01 * (
            self.m10 * (self.m22 * self.m33 - self.m23 * self.m32)
            - self.m12 * (self.m20 * self.m33 - self.m23 * self.m30)
            + self.m13 * (self.m20 * self.m32 - self.m22 * self.m30)
        )
        var c = self.m02 * (
            self.m10 * (self.m21 * self.m33 - self.m23 * self.m31)
            - self.m11 * (self.m20 * self.m33 - self.m23 * self.m30)
            + self.m13 * (self.m20 * self.m31 - self.m21 * self.m30)
        )
        var d = self.m03 * (
            self.m10 * (self.m21 * self.m32 - self.m22 * self.m31)
            - self.m11 * (self.m20 * self.m32 - self.m22 * self.m30)
            + self.m12 * (self.m20 * self.m31 - self.m21 * self.m30)
        )
        return a - b + c - d

    fn inverse(self) -> Self:
        """General 4x4 matrix inverse.

        Returns identity if matrix is singular.
        """
        # Compute cofactors
        var c00 = self.m11 * (self.m22 * self.m33 - self.m23 * self.m32) - self.m12 * (
            self.m21 * self.m33 - self.m23 * self.m31
        ) + self.m13 * (self.m21 * self.m32 - self.m22 * self.m31)

        var c01 = -(
            self.m10 * (self.m22 * self.m33 - self.m23 * self.m32)
            - self.m12 * (self.m20 * self.m33 - self.m23 * self.m30)
            + self.m13 * (self.m20 * self.m32 - self.m22 * self.m30)
        )

        var c02 = self.m10 * (self.m21 * self.m33 - self.m23 * self.m31) - self.m11 * (
            self.m20 * self.m33 - self.m23 * self.m30
        ) + self.m13 * (self.m20 * self.m31 - self.m21 * self.m30)

        var c03 = -(
            self.m10 * (self.m21 * self.m32 - self.m22 * self.m31)
            - self.m11 * (self.m20 * self.m32 - self.m22 * self.m30)
            + self.m12 * (self.m20 * self.m31 - self.m21 * self.m30)
        )

        var det = self.m00 * c00 + self.m01 * c01 + self.m02 * c02 + self.m03 * c03
        if abs(det) < 1e-10:
            return Self.identity()

        var inv_det = 1.0 / det

        # Compute remaining cofactors
        var c10 = -(
            self.m01 * (self.m22 * self.m33 - self.m23 * self.m32)
            - self.m02 * (self.m21 * self.m33 - self.m23 * self.m31)
            + self.m03 * (self.m21 * self.m32 - self.m22 * self.m31)
        )

        var c11 = self.m00 * (self.m22 * self.m33 - self.m23 * self.m32) - self.m02 * (
            self.m20 * self.m33 - self.m23 * self.m30
        ) + self.m03 * (self.m20 * self.m32 - self.m22 * self.m30)

        var c12 = -(
            self.m00 * (self.m21 * self.m33 - self.m23 * self.m31)
            - self.m01 * (self.m20 * self.m33 - self.m23 * self.m30)
            + self.m03 * (self.m20 * self.m31 - self.m21 * self.m30)
        )

        var c13 = self.m00 * (self.m21 * self.m32 - self.m22 * self.m31) - self.m01 * (
            self.m20 * self.m32 - self.m22 * self.m30
        ) + self.m02 * (self.m20 * self.m31 - self.m21 * self.m30)

        var c20 = self.m01 * (self.m12 * self.m33 - self.m13 * self.m32) - self.m02 * (
            self.m11 * self.m33 - self.m13 * self.m31
        ) + self.m03 * (self.m11 * self.m32 - self.m12 * self.m31)

        var c21 = -(
            self.m00 * (self.m12 * self.m33 - self.m13 * self.m32)
            - self.m02 * (self.m10 * self.m33 - self.m13 * self.m30)
            + self.m03 * (self.m10 * self.m32 - self.m12 * self.m30)
        )

        var c22 = self.m00 * (self.m11 * self.m33 - self.m13 * self.m31) - self.m01 * (
            self.m10 * self.m33 - self.m13 * self.m30
        ) + self.m03 * (self.m10 * self.m31 - self.m11 * self.m30)

        var c23 = -(
            self.m00 * (self.m11 * self.m32 - self.m12 * self.m31)
            - self.m01 * (self.m10 * self.m32 - self.m12 * self.m30)
            + self.m02 * (self.m10 * self.m31 - self.m11 * self.m30)
        )

        var c30 = -(
            self.m01 * (self.m12 * self.m23 - self.m13 * self.m22)
            - self.m02 * (self.m11 * self.m23 - self.m13 * self.m21)
            + self.m03 * (self.m11 * self.m22 - self.m12 * self.m21)
        )

        var c31 = self.m00 * (self.m12 * self.m23 - self.m13 * self.m22) - self.m02 * (
            self.m10 * self.m23 - self.m13 * self.m20
        ) + self.m03 * (self.m10 * self.m22 - self.m12 * self.m20)

        var c32 = -(
            self.m00 * (self.m11 * self.m23 - self.m13 * self.m21)
            - self.m01 * (self.m10 * self.m23 - self.m13 * self.m20)
            + self.m03 * (self.m10 * self.m21 - self.m11 * self.m20)
        )

        var c33 = self.m00 * (self.m11 * self.m22 - self.m12 * self.m21) - self.m01 * (
            self.m10 * self.m22 - self.m12 * self.m20
        ) + self.m02 * (self.m10 * self.m21 - self.m11 * self.m20)

        # Return adjugate transposed * (1/det)
        return Self(
            c00 * inv_det,
            c10 * inv_det,
            c20 * inv_det,
            c30 * inv_det,
            c01 * inv_det,
            c11 * inv_det,
            c21 * inv_det,
            c31 * inv_det,
            c02 * inv_det,
            c12 * inv_det,
            c22 * inv_det,
            c32 * inv_det,
            c03 * inv_det,
            c13 * inv_det,
            c23 * inv_det,
            c33 * inv_det,
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
            and self.m03 == other.m03
            and self.m10 == other.m10
            and self.m11 == other.m11
            and self.m12 == other.m12
            and self.m13 == other.m13
            and self.m20 == other.m20
            and self.m21 == other.m21
            and self.m22 == other.m22
            and self.m23 == other.m23
            and self.m30 == other.m30
            and self.m31 == other.m31
            and self.m32 == other.m32
            and self.m33 == other.m33
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
            and abs(self.m03 - other.m03) < tolerance
            and abs(self.m10 - other.m10) < tolerance
            and abs(self.m11 - other.m11) < tolerance
            and abs(self.m12 - other.m12) < tolerance
            and abs(self.m13 - other.m13) < tolerance
            and abs(self.m20 - other.m20) < tolerance
            and abs(self.m21 - other.m21) < tolerance
            and abs(self.m22 - other.m22) < tolerance
            and abs(self.m23 - other.m23) < tolerance
            and abs(self.m30 - other.m30) < tolerance
            and abs(self.m31 - other.m31) < tolerance
            and abs(self.m32 - other.m32) < tolerance
            and abs(self.m33 - other.m33) < tolerance
        )

    # =========================================================================
    # String Conversion
    # =========================================================================

    fn __str__(self) -> String:
        """String representation."""
        return (
            "Mat4(\n  ["
            + String(self.m00)
            + ", "
            + String(self.m01)
            + ", "
            + String(self.m02)
            + ", "
            + String(self.m03)
            + "],\n  ["
            + String(self.m10)
            + ", "
            + String(self.m11)
            + ", "
            + String(self.m12)
            + ", "
            + String(self.m13)
            + "],\n  ["
            + String(self.m20)
            + ", "
            + String(self.m21)
            + ", "
            + String(self.m22)
            + ", "
            + String(self.m23)
            + "],\n  ["
            + String(self.m30)
            + ", "
            + String(self.m31)
            + ", "
            + String(self.m32)
            + ", "
            + String(self.m33)
            + "]\n)"
        )


# =========================================================================
# Utility Functions
# =========================================================================


fn mat4_identity() -> Mat4:
    """Return identity matrix."""
    return Mat4.identity()


fn mat4_translation(t: Vec3) -> Mat4:
    """Create translation matrix."""
    return Mat4.from_translation(t)


fn mat4_look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    """Create view matrix."""
    return Mat4.look_at(eye, target, up)


fn mat4_perspective(fov_y: Float64, aspect: Float64, near: Float64, far: Float64) -> Mat4:
    """Create perspective projection matrix."""
    return Mat4.perspective(fov_y, aspect, near, far)
