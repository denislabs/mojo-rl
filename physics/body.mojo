"""
Body: Rigid Body Dynamics for Physics Engine

Implements 2D rigid bodies with position, rotation, and velocity.
Supports static (terrain) and dynamic (lander, legs, particles) bodies.
"""

from math import cos, sin
from physics.vec2 import Vec2, vec2, cross_sv, cross_vs


# Body type constants
comptime BODY_STATIC: Int = 0
comptime BODY_DYNAMIC: Int = 1

# Sleep constants
comptime SLEEP_TIME_THRESHOLD: Float64 = 0.5  # Seconds of low velocity before sleep
comptime SLEEP_LINEAR_THRESHOLD: Float64 = 0.01  # Linear velocity threshold
comptime SLEEP_ANGULAR_THRESHOLD: Float64 = 0.01  # Angular velocity threshold


struct MassData(Copyable, Movable):
    """Mass properties of a body."""

    var mass: Float64
    var inv_mass: Float64
    var inertia: Float64  # Moment of inertia about center of mass
    var inv_inertia: Float64
    var center: Vec2  # Local center of mass

    fn __init__(out self):
        """Create default (zero) mass data."""
        self.mass = 0.0
        self.inv_mass = 0.0
        self.inertia = 0.0
        self.inv_inertia = 0.0
        self.center = Vec2.zero()

    fn __init__(
        out self,
        mass: Float64,
        inertia: Float64,
        center: Vec2 = Vec2.zero(),
    ):
        """Create mass data with given values."""
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0.0 else 0.0
        self.inertia = inertia
        self.inv_inertia = 1.0 / inertia if inertia > 0.0 else 0.0
        self.center = center

    fn set_mass(mut self, mass: Float64):
        """Set mass and update inverse."""
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0.0 else 0.0

    fn set_inertia(mut self, inertia: Float64):
        """Set inertia and update inverse."""
        self.inertia = inertia
        self.inv_inertia = 1.0 / inertia if inertia > 0.0 else 0.0


struct Transform(Copyable, Movable):
    """2D rigid body transform (position + rotation)."""

    var position: Vec2
    var cos_angle: Float64
    var sin_angle: Float64

    fn __init__(out self):
        """Create identity transform."""
        self.position = Vec2.zero()
        self.cos_angle = 1.0
        self.sin_angle = 0.0

    fn __init__(out self, position: Vec2, angle: Float64):
        """Create transform from position and angle."""
        self.position = position
        self.cos_angle = cos(angle)
        self.sin_angle = sin(angle)

    fn set_angle(mut self, angle: Float64):
        """Update rotation."""
        self.cos_angle = cos(angle)
        self.sin_angle = sin(angle)

    @always_inline
    fn apply(self, v: Vec2) -> Vec2:
        """Transform local point to world space."""
        return Vec2(
            self.position.x + v.x * self.cos_angle - v.y * self.sin_angle,
            self.position.y + v.x * self.sin_angle + v.y * self.cos_angle,
        )

    @always_inline
    fn apply_rotation(self, v: Vec2) -> Vec2:
        """Transform local vector to world space (rotation only)."""
        return Vec2(
            v.x * self.cos_angle - v.y * self.sin_angle,
            v.x * self.sin_angle + v.y * self.cos_angle,
        )

    @always_inline
    fn apply_inverse(self, v: Vec2) -> Vec2:
        """Transform world point to local space."""
        var p = v - self.position
        return Vec2(
            p.x * self.cos_angle + p.y * self.sin_angle,
            -p.x * self.sin_angle + p.y * self.cos_angle,
        )

    @always_inline
    fn apply_inverse_rotation(self, v: Vec2) -> Vec2:
        """Transform world vector to local space (rotation only)."""
        return Vec2(
            v.x * self.cos_angle + v.y * self.sin_angle,
            -v.x * self.sin_angle + v.y * self.cos_angle,
        )


struct Body(Copyable, Movable):
    """Rigid body with dynamics."""

    # Identity
    var body_type: Int
    var user_data: Int  # For game-specific data (e.g., entity ID)

    # Transform
    var position: Vec2
    var angle: Float64
    var transform: Transform  # Cached transform

    # Velocities
    var linear_velocity: Vec2
    var angular_velocity: Float64

    # Forces accumulated this step
    var force: Vec2
    var torque: Float64

    # Mass properties
    var mass_data: MassData

    # State
    var awake: Bool
    var sleep_time: Float64

    # Fixture indices (stored in World)
    var fixture_start: Int
    var fixture_count: Int

    # LunarLander-specific: leg ground contact
    var ground_contact: Bool

    fn __init__(out self, body_type: Int, position: Vec2, angle: Float64 = 0.0):
        """Create a new body."""
        self.body_type = body_type
        self.user_data = 0
        self.position = position
        self.angle = angle
        self.transform = Transform(position, angle)
        self.linear_velocity = Vec2.zero()
        self.angular_velocity = 0.0
        self.force = Vec2.zero()
        self.torque = 0.0
        self.mass_data = MassData()
        self.awake = True
        self.sleep_time = 0.0
        self.fixture_start = 0
        self.fixture_count = 0
        self.ground_contact = False

    fn __init__(out self):
        """Create default body."""
        self.body_type = BODY_STATIC
        self.user_data = 0
        self.position = Vec2.zero()
        self.angle = 0.0
        self.transform = Transform()
        self.linear_velocity = Vec2.zero()
        self.angular_velocity = 0.0
        self.force = Vec2.zero()
        self.torque = 0.0
        self.mass_data = MassData()
        self.awake = True
        self.sleep_time = 0.0
        self.fixture_start = 0
        self.fixture_count = 0
        self.ground_contact = False

    # ===== Force/Impulse Application =====

    @always_inline
    fn apply_force_to_center(mut self, force: Vec2):
        """Apply force at center of mass (no torque)."""
        if self.body_type != BODY_DYNAMIC:
            return
        self.force += force
        self.set_awake(True)

    @always_inline
    fn apply_force(mut self, force: Vec2, point: Vec2):
        """Apply force at world point, generating torque."""
        if self.body_type != BODY_DYNAMIC:
            return
        self.force += force
        # Torque = r x F (2D cross product gives scalar)
        var r = point - self.position
        self.torque += r.cross(force)
        self.set_awake(True)

    @always_inline
    fn apply_torque(mut self, torque: Float64):
        """Apply torque (angular force)."""
        if self.body_type != BODY_DYNAMIC:
            return
        self.torque += torque
        self.set_awake(True)

    @always_inline
    fn apply_linear_impulse(mut self, impulse: Vec2, point: Vec2):
        """Apply impulse at world point."""
        if self.body_type != BODY_DYNAMIC:
            return
        self.linear_velocity += impulse * self.mass_data.inv_mass
        # Angular impulse = r x J
        var r = point - self.position
        self.angular_velocity += self.mass_data.inv_inertia * r.cross(impulse)
        self.set_awake(True)

    @always_inline
    fn apply_linear_impulse_to_center(mut self, impulse: Vec2):
        """Apply impulse at center of mass (no angular effect)."""
        if self.body_type != BODY_DYNAMIC:
            return
        self.linear_velocity += impulse * self.mass_data.inv_mass
        self.set_awake(True)

    @always_inline
    fn apply_angular_impulse(mut self, impulse: Float64):
        """Apply angular impulse."""
        if self.body_type != BODY_DYNAMIC:
            return
        self.angular_velocity += self.mass_data.inv_inertia * impulse
        self.set_awake(True)

    # ===== Transform Helpers =====

    @always_inline
    fn get_world_point(self, local_point: Vec2) -> Vec2:
        """Transform local point to world coordinates."""
        return self.transform.apply(local_point)

    @always_inline
    fn get_world_vector(self, local_vec: Vec2) -> Vec2:
        """Transform local vector to world coordinates."""
        return self.transform.apply_rotation(local_vec)

    @always_inline
    fn get_local_point(self, world_point: Vec2) -> Vec2:
        """Transform world point to local coordinates."""
        return self.transform.apply_inverse(world_point)

    @always_inline
    fn get_local_vector(self, world_vec: Vec2) -> Vec2:
        """Transform world vector to local coordinates."""
        return self.transform.apply_inverse_rotation(world_vec)

    @always_inline
    fn get_linear_velocity_at_point(self, world_point: Vec2) -> Vec2:
        """Get linear velocity at a world point (includes angular contribution)."""
        var r = world_point - self.position
        return self.linear_velocity + cross_sv(self.angular_velocity, r)

    # ===== Integration =====

    fn integrate_velocities(mut self, gravity: Vec2, dt: Float64):
        """Update velocities from forces (semi-implicit Euler step 1)."""
        if self.body_type != BODY_DYNAMIC or not self.awake:
            return

        # v' = v + (F/m + g) * dt
        self.linear_velocity += (
            self.force * self.mass_data.inv_mass + gravity
        ) * dt

        # w' = w + (tau/I) * dt
        self.angular_velocity += self.torque * self.mass_data.inv_inertia * dt

        # Clear forces
        self.force = Vec2.zero()
        self.torque = 0.0

    fn integrate_positions(mut self, dt: Float64):
        """Update positions from velocities (semi-implicit Euler step 2)."""
        if self.body_type != BODY_DYNAMIC or not self.awake:
            return

        # x' = x + v * dt
        self.position += self.linear_velocity * dt

        # theta' = theta + w * dt
        self.angle += self.angular_velocity * dt

        # Update cached transform
        self.transform = Transform(self.position, self.angle)

    # ===== Sleep Management =====

    fn set_awake(mut self, awake: Bool):
        """Set awake state."""
        if awake:
            self.awake = True
            self.sleep_time = 0.0
        else:
            self.awake = False
            self.linear_velocity = Vec2.zero()
            self.angular_velocity = 0.0
            self.force = Vec2.zero()
            self.torque = 0.0

    fn update_sleep_state(mut self, dt: Float64):
        """Update sleep timer and potentially put body to sleep."""
        if self.body_type == BODY_STATIC:
            return

        # Check if velocity is below threshold
        var linear_speed_sq = self.linear_velocity.length_squared()
        var angular_speed = abs(self.angular_velocity)

        if (
            linear_speed_sq > SLEEP_LINEAR_THRESHOLD * SLEEP_LINEAR_THRESHOLD
            or angular_speed > SLEEP_ANGULAR_THRESHOLD
        ):
            self.sleep_time = 0.0
        else:
            self.sleep_time += dt
            if self.sleep_time >= SLEEP_TIME_THRESHOLD:
                self.set_awake(False)

    # ===== Mass =====

    fn set_mass_data(
        mut self, mass: Float64, inertia: Float64, center: Vec2 = Vec2.zero()
    ):
        """Set mass properties directly."""
        if self.body_type != BODY_DYNAMIC:
            self.mass_data = MassData()
            return
        self.mass_data = MassData(mass, inertia, center)

    fn is_static(self) -> Bool:
        """Check if body is static."""
        return self.body_type == BODY_STATIC

    fn is_dynamic(self) -> Bool:
        """Check if body is dynamic."""
        return self.body_type == BODY_DYNAMIC


# Helper function
fn abs(x: Float64) -> Float64:
    """Absolute value."""
    return x if x >= 0.0 else -x
