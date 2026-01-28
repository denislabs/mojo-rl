"""3D Camera for perspective projection.

Provides view and projection matrices for rendering 3D scenes.
"""

from math import sqrt, sin, cos, tan
from math3d import Vec3 as Vec3Generic, Mat4 as Mat4Generic

# Type alias for Vec3 with float64 precision (used throughout render3d)
comptime Vec3 = Vec3Generic[DType.float64]
comptime Mat4 = Mat4Generic[DType.float64]


struct Camera3D:
    """Perspective camera for 3D rendering.

    Uses a look-at model where the camera is positioned at `eye`,
    looking at `target`, with `up` defining the vertical direction.
    """

    var eye: Vec3  # Camera position
    var target: Vec3  # Look-at point
    var up: Vec3  # Up vector

    var fov: Float64  # Field of view in radians
    var aspect: Float64  # Aspect ratio (width/height)
    var near: Float64  # Near clipping plane
    var far: Float64  # Far clipping plane

    # Screen dimensions for projection
    var screen_width: Int
    var screen_height: Int

    fn __init__(
        out self,
        eye: Vec3 = Vec3(0.0, -5.0, 2.0),
        target: Vec3 = Vec3(0.0, 0.0, 1.0),
        up: Vec3 = Vec3(0.0, 0.0, 1.0),
        fov: Float64 = 60.0,  # degrees
        aspect: Float64 = 16.0 / 9.0,
        near: Float64 = 0.1,
        far: Float64 = 100.0,
        screen_width: Int = 800,
        screen_height: Int = 450,
    ):
        """Initialize camera.

        Args:
            eye: Camera position in world space.
            target: Point the camera looks at.
            up: Up direction (usually Z-up).
            fov: Field of view in degrees.
            aspect: Aspect ratio (width/height).
            near: Near clipping plane distance.
            far: Far clipping plane distance.
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.
        """
        self.eye = eye
        self.target = target
        self.up = up
        self.fov = fov * 3.14159265358979 / 180.0  # Convert to radians
        self.aspect = aspect
        self.near = near
        self.far = far
        self.screen_width = screen_width
        self.screen_height = screen_height

    fn get_view_matrix(self) -> Mat4:
        """Compute the view matrix (world to camera transform).

        Returns:
            4x4 view matrix.
        """
        return Mat4.look_at(self.eye, self.target, self.up)

    fn get_projection_matrix(self) -> Mat4:
        """Compute the perspective projection matrix.

        Returns:
            4x4 projection matrix.
        """
        return Mat4.perspective(self.fov, self.aspect, self.near, self.far)

    fn get_view_projection(self) -> Mat4:
        """Compute combined view-projection matrix.

        Returns:
            4x4 view-projection matrix (projection * view).
        """
        return self.get_projection_matrix() @ self.get_view_matrix()

    fn project_to_screen(self, world_point: Vec3) -> Tuple[Int, Int, Bool]:
        """Project a 3D world point to 2D screen coordinates.

        Args:
            world_point: Point in world space.

        Returns:
            Tuple of (screen_x, screen_y, is_visible).
            is_visible is False if the point is behind the camera.
        """
        # First transform to view space to check if behind camera
        var view = self.get_view_matrix()
        var view_point = view.transform_point(world_point)

        # Check if point is behind camera (in view space, camera looks down -Z)
        if view_point.z > -self.near:
            return (0, 0, False)

        # Get projection matrix and compute clip space coordinates
        var proj = self.get_projection_matrix()

        # Compute homogeneous clip coordinates (need full 4D transform)
        # For perspective matrix, w' = -view_z (from m32 = -1)
        var clip_x = proj.m00 * view_point.x + proj.m01 * view_point.y + proj.m02 * view_point.z + proj.m03
        var clip_y = proj.m10 * view_point.x + proj.m11 * view_point.y + proj.m12 * view_point.z + proj.m13
        var clip_z = proj.m20 * view_point.x + proj.m21 * view_point.y + proj.m22 * view_point.z + proj.m23
        var clip_w = proj.m30 * view_point.x + proj.m31 * view_point.y + proj.m32 * view_point.z + proj.m33

        # Perspective divide to get NDC
        if abs(clip_w) < 0.0001:
            return (0, 0, False)

        var ndc_x = clip_x / clip_w
        var ndc_y = clip_y / clip_w

        # Clamp NDC coordinates to [-1, 1] range (with some margin for edge cases)
        if ndc_x < -1.5 or ndc_x > 1.5 or ndc_y < -1.5 or ndc_y > 1.5:
            return (0, 0, False)

        # Convert NDC to screen coordinates
        # NDC (-1,-1) maps to (0, height), (1, 1) maps to (width, 0)
        var screen_x = Int((ndc_x + 1.0) * 0.5 * Float64(self.screen_width))
        var screen_y = Int((1.0 - ndc_y) * 0.5 * Float64(self.screen_height))

        return (screen_x, screen_y, True)

    fn orbit(mut self, delta_theta: Float64, delta_phi: Float64):
        """Orbit camera around target point.

        Args:
            delta_theta: Horizontal rotation (radians).
            delta_phi: Vertical rotation (radians).
        """
        # Vector from target to eye
        var offset = self.eye - self.target
        var r = offset.length()

        # Current spherical coordinates
        var theta = self._atan2(offset.y, offset.x)
        var phi = self._acos(offset.z / r) if r > 0.0 else 0.0

        # Apply deltas
        theta += delta_theta
        phi += delta_phi

        # Clamp phi to avoid gimbal lock
        phi = self._clamp(phi, 0.1, 3.04159)

        # Convert back to Cartesian
        self.eye = Vec3(
            self.target.x + r * sin(phi) * cos(theta),
            self.target.y + r * sin(phi) * sin(theta),
            self.target.z + r * cos(phi),
        )

    fn zoom(mut self, delta: Float64):
        """Zoom camera in/out.

        Args:
            delta: Zoom amount (negative = zoom in).
        """
        var direction = (self.eye - self.target).normalized()
        var distance = (self.eye - self.target).length()

        # Clamp distance
        distance = self._clamp(distance + delta, 1.0, 50.0)

        self.eye = self.target + direction * distance

    fn pan(mut self, delta_x: Float64, delta_y: Float64):
        """Pan camera (move target and eye together).

        Args:
            delta_x: Horizontal pan amount.
            delta_y: Vertical pan amount.
        """
        # Get camera right and up vectors
        var forward = (self.target - self.eye).normalized()
        var right = forward.cross(self.up).normalized()
        var cam_up = right.cross(forward).normalized()

        # Apply pan
        var offset = right * delta_x + cam_up * delta_y
        self.eye += offset
        self.target += offset

    fn set_screen_size(mut self, width: Int, height: Int):
        """Update screen dimensions.

        Args:
            width: New screen width.
            height: New screen height.
        """
        self.screen_width = width
        self.screen_height = height
        self.aspect = Float64(width) / Float64(height) if height > 0 else 1.0

    @staticmethod
    fn _atan2(y: Float64, x: Float64) -> Float64:
        """Compute atan2(y, x)."""
        from math import atan2

        return atan2(y, x)

    @staticmethod
    fn _acos(x: Float64) -> Float64:
        """Compute acos(x) with clamping."""
        from math import acos

        var clamped = x
        if clamped > 1.0:
            clamped = 1.0
        elif clamped < -1.0:
            clamped = -1.0
        return acos(clamped)

    @staticmethod
    fn _clamp(x: Float64, min_val: Float64, max_val: Float64) -> Float64:
        """Clamp value to range."""
        if x < min_val:
            return min_val
        if x > max_val:
            return max_val
        return x
