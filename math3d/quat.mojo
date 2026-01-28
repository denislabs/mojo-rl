"""Unit Quaternion type for 3D rotations.

Provides Quat struct for representing 3D rotations without gimbal lock.
Convention: (w, x, y, z) where w is the scalar part.
"""

from math import sqrt, cos, sin, acos, atan2, asin


from .vec3 import Vec3


@fieldwise_init
struct Quat[DTYPE: DType](ImplicitlyCopyable, Movable, Stringable):
    """Unit quaternion for 3D rotations.

    Stores rotation as (w, x, y, z) where:
    - w is the scalar (real) part
    - (x, y, z) is the vector (imaginary) part

    For unit quaternion representing rotation by angle θ around axis n:
    q = (cos(θ/2), sin(θ/2) * n)
    """

    var w: Scalar[Self.DTYPE]  # Scalar part
    var x: Scalar[Self.DTYPE]  # Vector part x
    var y: Scalar[Self.DTYPE]  # Vector part y
    var z: Scalar[Self.DTYPE]  # Vector part z

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @staticmethod
    fn identity() -> Self:
        """Return the identity quaternion (no rotation)."""
        return Self(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    fn from_axis_angle(
        axis: Vec3[Self.DTYPE], angle: Scalar[Self.DTYPE]
    ) -> Self:
        """Create quaternion from axis-angle representation.

        Args:
            axis: Rotation axis (will be normalized).
            angle: Rotation angle in radians.

        Returns:
            Unit quaternion representing the rotation.
        """
        var half_angle = angle * 0.5
        var s = sin(half_angle)
        var c = cos(half_angle)
        var n = axis.normalized()
        return Self(c, n.x * s, n.y * s, n.z * s)

    @staticmethod
    fn from_euler_xyz(
        x: Scalar[Self.DTYPE], y: Scalar[Self.DTYPE], z: Scalar[Self.DTYPE]
    ) -> Self:
        """Create quaternion from Euler angles (XYZ rotation order).

        Args:
            x: Rotation around X axis in radians.
            y: Rotation around Y axis in radians.
            z: Rotation around Z axis in radians.

        Returns:
            Unit quaternion representing the combined rotation.
        """
        var cx = cos(x * 0.5)
        var sx = sin(x * 0.5)
        var cy = cos(y * 0.5)
        var sy = sin(y * 0.5)
        var cz = cos(z * 0.5)
        var sz = sin(z * 0.5)

        return Self(
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        )

    @staticmethod
    fn from_euler_zyx(
        z: Scalar[Self.DTYPE], y: Scalar[Self.DTYPE], x: Scalar[Self.DTYPE]
    ) -> Self:
        """Create quaternion from Euler angles (ZYX rotation order).

        Common in robotics and aviation (yaw-pitch-roll).

        Args:
            z: Yaw (rotation around Z axis) in radians.
            y: Pitch (rotation around Y axis) in radians.
            x: Roll (rotation around X axis) in radians.

        Returns:
            Unit quaternion representing the combined rotation.
        """
        var cz = cos(z * 0.5)
        var sz = sin(z * 0.5)
        var cy = cos(y * 0.5)
        var sy = sin(y * 0.5)
        var cx = cos(x * 0.5)
        var sx = sin(x * 0.5)

        return Self(
            cz * cy * cx + sz * sy * sx,
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx,
        )

    @staticmethod
    fn from_two_vectors(
        from_vec: Vec3[Self.DTYPE], to_vec: Vec3[Self.DTYPE]
    ) -> Self:
        """Create quaternion that rotates from_vec to to_vec.

        Args:
            from_vec: Source vector (will be normalized).
            to_vec: Target vector (will be normalized).

        Returns:
            Unit quaternion that rotates from_vec to align with to_vec.
        """
        var f = from_vec.normalized()
        var t = to_vec.normalized()
        var d = f.dot(t)

        if d > 0.9999:
            # Vectors are parallel
            return Self.identity()
        elif d < -0.9999:
            # Vectors are opposite, find orthogonal axis
            var axis = Vec3[Self.DTYPE].unit_x().cross(f)
            if axis.length_squared() < 0.0001:
                axis = Vec3[Self.DTYPE].unit_y().cross(f)
            return Self.from_axis_angle(
                axis.normalized(), 3.14159265358979323846
            )
        else:
            var axis = f.cross(t)
            var w = 1.0 + d
            return Self(w, axis.x, axis.y, axis.z).normalized()

    # =========================================================================
    # Quaternion Operations
    # =========================================================================

    fn __mul__(self, other: Self) -> Self:
        """Quaternion multiplication (combines rotations).

        Note: Multiplication order is right-to-left:
        (q1 * q2) applies q2 first, then q1.
        """
        return Self(
            self.w * other.w
            - self.x * other.x
            - self.y * other.y
            - self.z * other.z,
            self.w * other.x
            + self.x * other.w
            + self.y * other.z
            - self.z * other.y,
            self.w * other.y
            - self.x * other.z
            + self.y * other.w
            + self.z * other.x,
            self.w * other.z
            + self.x * other.y
            - self.y * other.x
            + self.z * other.w,
        )

    fn __neg__(self) -> Self:
        """Negate quaternion (represents same rotation)."""
        return Self(-self.w, -self.x, -self.y, -self.z)

    fn conjugate(self) -> Self:
        """Return conjugate (inverse rotation for unit quaternion)."""
        return Self(self.w, -self.x, -self.y, -self.z)

    fn inverse(self) -> Self:
        """Return inverse quaternion.

        For unit quaternions, this is the same as conjugate.
        """
        var len_sq = self.length_squared()
        if len_sq < 1e-10:
            return Self.identity()
        var inv_len_sq = 1.0 / len_sq
        return Self(
            self.w * inv_len_sq,
            -self.x * inv_len_sq,
            -self.y * inv_len_sq,
            -self.z * inv_len_sq,
        )

    fn length_squared(self) -> Scalar[Self.DTYPE]:
        """Squared norm of quaternion."""
        return (
            self.w * self.w
            + self.x * self.x
            + self.y * self.y
            + self.z * self.z
        )

    fn length(self) -> Scalar[Self.DTYPE]:
        """Norm of quaternion (should be 1 for unit quaternion)."""
        return sqrt(self.length_squared())

    fn normalized(self) -> Self:
        """Return normalized quaternion (unit length)."""
        var len = self.length()
        if len > 1e-10:
            var inv = 1.0 / len
            return Self(self.w * inv, self.x * inv, self.y * inv, self.z * inv)
        return Self.identity()

    fn normalize(mut self):
        """Normalize in place."""
        var len = self.length()
        if len > 1e-10:
            var inv = 1.0 / len
            self.w *= inv
            self.x *= inv
            self.y *= inv
            self.z *= inv

    # =========================================================================
    # Rotation Operations
    # =========================================================================

    fn rotate_vec(self, v: Vec3[Self.DTYPE]) -> Vec3[Self.DTYPE]:
        """Rotate a vector by this quaternion.

        Uses optimized formula: v' = v + 2*w*(q_vec × v) + 2*(q_vec × (q_vec × v))

        Args:
            v: The vector to rotate.

        Returns:
            The rotated vector.
        """
        # q_vec = (x, y, z) imaginary part
        var qv = Vec3(self.x, self.y, self.z)

        # t = 2 * (q_vec × v)
        var t = qv.cross(v) * 2.0

        # v' = v + w*t + q_vec × t
        return v + t * self.w + qv.cross(t)

    fn rotate_vec_inverse(self, v: Vec3[Self.DTYPE]) -> Vec3[Self.DTYPE]:
        """Rotate a vector by inverse of this quaternion.

        Args:
            v: The vector to rotate.

        Returns:
            The inversely rotated vector.
        """
        return self.conjugate().rotate_vec(v)

    # =========================================================================
    # Conversion to Other Representations
    # =========================================================================

    fn to_axis_angle(self) -> Tuple[Vec3[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Convert to axis-angle representation.

        Returns:
            Tuple of (axis, angle) where axis is the rotation axis
            and angle is the rotation angle in radians.
        """
        var q = self.normalized()
        var angle = 2.0 * acos(min(max(q.w, -1.0), 1.0))

        var s = sqrt(1.0 - q.w * q.w)
        var axis: Vec3[Self.DTYPE]
        if s < 0.0001:
            axis = Vec3[Self.DTYPE].unit_x()
        else:
            axis = Vec3[Self.DTYPE](q.x / s, q.y / s, q.z / s)

        return (axis, angle)

    fn to_euler_xyz(self) -> Vec3[Self.DTYPE]:
        """Convert to Euler angles (XYZ rotation order).

        Returns:
            Vec3 containing (roll, pitch, yaw) in radians.
        """
        # Roll (x-axis rotation)
        var sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        var cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        var roll = atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        var sinp = 2.0 * (self.w * self.y - self.z * self.x)
        var pitch: Scalar[Self.DTYPE]
        if abs(sinp) >= 1.0:
            pitch = Scalar[Self.DTYPE](
                1.5707963267948966
            ) if sinp > 0 else Scalar[Self.DTYPE](
                -1.5707963267948966
            )  # +/- pi/2
        else:
            pitch = asin(sinp)

        # Yaw (z-axis rotation)
        var siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        var cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        var yaw = atan2(siny_cosp, cosy_cosp)

        return Vec3(roll, pitch, yaw)

    fn get_forward(self) -> Vec3[Self.DTYPE]:
        """Get the forward direction (positive Z after rotation)."""
        return self.rotate_vec(Vec3[Self.DTYPE].unit_z())

    fn get_right(self) -> Vec3[Self.DTYPE]:
        """Get the right direction (positive X after rotation)."""
        return self.rotate_vec(Vec3[Self.DTYPE].unit_x())

    fn get_up(self) -> Vec3[Self.DTYPE]:
        """Get the up direction (positive Y after rotation)."""
        return self.rotate_vec(Vec3[Self.DTYPE].unit_y())

    # =========================================================================
    # Interpolation
    # =========================================================================

    fn slerp(self, other: Self, t: Scalar[Self.DTYPE]) -> Self:
        """Spherical linear interpolation.

        Interpolates smoothly between two rotations.

        Args:
            other: Target quaternion.
            t: Interpolation factor (0 = self, 1 = other).

        Returns:
            Interpolated quaternion.
        """
        # Compute dot product
        var d = (
            self.w * other.w
            + self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )

        # Ensure shortest path
        var other_adj = other
        if d < 0.0:
            other_adj = -other
            d = -d

        # If quaternions are very close, use linear interpolation
        if d > 0.9995:
            var result = Self(
                self.w + (other_adj.w - self.w) * t,
                self.x + (other_adj.x - self.x) * t,
                self.y + (other_adj.y - self.y) * t,
                self.z + (other_adj.z - self.z) * t,
            )
            return result.normalized()

        # Perform slerp
        var theta = acos(d)
        var sin_theta = sin(theta)
        var s0 = sin((1.0 - t) * theta) / sin_theta
        var s1 = sin(t * theta) / sin_theta

        return Self(
            self.w * s0 + other_adj.w * s1,
            self.x * s0 + other_adj.x * s1,
            self.y * s0 + other_adj.y * s1,
            self.z * s0 + other_adj.z * s1,
        )

    fn nlerp(self, other: Self, t: Scalar[Self.DTYPE]) -> Self:
        """Normalized linear interpolation.

        Faster than slerp but not constant velocity.

        Args:
            other: Target quaternion.
            t: Interpolation factor (0 = self, 1 = other).

        Returns:
            Interpolated quaternion.
        """
        # Ensure shortest path
        var other_adj = other
        var d = (
            self.w * other.w
            + self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )
        if d < 0.0:
            other_adj = -other

        return Self(
            self.w + (other_adj.w - self.w) * t,
            self.x + (other_adj.x - self.x) * t,
            self.y + (other_adj.y - self.y) * t,
            self.z + (other_adj.z - self.z) * t,
        ).normalized()

    # =========================================================================
    # Comparison
    # =========================================================================

    fn __eq__(self, other: Self) -> Bool:
        """Equality check."""
        return (
            self.w == other.w
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    fn __ne__(self, other: Self) -> Bool:
        """Inequality check."""
        return not (self == other)

    fn approx_eq(
        self, other: Self, tolerance: Scalar[Self.DTYPE] = 1e-10
    ) -> Bool:
        """Approximate equality (accounting for q and -q being same rotation).
        """
        var d = abs(
            self.w * other.w
            + self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )
        return abs(d - 1.0) < tolerance

    # =========================================================================
    # String Conversion
    # =========================================================================

    fn __str__(self) -> String:
        """String representation."""
        return (
            "Quat("
            + String(self.w)
            + ", "
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


fn quat_identity[DTYPE: DType]() -> Quat[DTYPE]:
    """Return identity quaternion."""
    return Quat[DTYPE].identity()


fn quat_from_axis_angle[
    DTYPE: DType
](axis: Vec3[DTYPE], angle: Scalar[DTYPE]) -> Quat[DTYPE]:
    """Create quaternion from axis-angle."""
    return Quat.from_axis_angle(axis, angle)


fn slerp[
    DTYPE: DType
](a: Quat[DTYPE], b: Quat[DTYPE], t: Scalar[DTYPE]) -> Quat[DTYPE]:
    """Spherical linear interpolation."""
    return a.slerp(b, t)
