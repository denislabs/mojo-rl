"""
Joint: Revolute Joint for Physics Engine

Implements a hinge (revolute) joint connecting two bodies.
Features:
- Point constraint (keeps anchor points together)
- Optional motor with torque limit
- Optional angle limits

Used by LunarLander to attach legs to lander body.
"""

from math import cos, sin
from physics.vec2 import Vec2, vec2, cross, cross_sv, cross_vs
from physics.body import Body, Transform, BODY_DYNAMIC
from physics.utils import clamp


struct RevoluteJoint[dtype: DType](Copyable, Movable):
    """Revolute (hinge) joint with motor and limits."""

    # Connected bodies (indices into World.bodies)
    var body_a_idx: Int
    var body_b_idx: Int

    # Anchor points in local coordinates
    var local_anchor_a: Vec2[Self.dtype]
    var local_anchor_b: Vec2[Self.dtype]

    # Reference angle (initial angle difference: body_b.angle - body_a.angle)
    var reference_angle: Scalar[Self.dtype]

    # Motor
    var enable_motor: Bool
    var motor_speed: Scalar[Self.dtype]  # Target angular velocity (rad/s)
    var max_motor_torque: Scalar[Self.dtype]

    # Limits
    var enable_limit: Bool
    var lower_angle: Scalar[Self.dtype]
    var upper_angle: Scalar[Self.dtype]

    # Solver state
    var impulse: Vec2[Self.dtype]  # Accumulated point constraint impulse
    var motor_impulse: Scalar[Self.dtype]  # Accumulated motor impulse
    var lower_impulse: Scalar[Self.dtype]  # Accumulated lower limit impulse
    var upper_impulse: Scalar[Self.dtype]  # Accumulated upper limit impulse

    # Cached solver data
    var r_a: Vec2[Self.dtype]  # World anchor relative to body A center
    var r_b: Vec2[Self.dtype]  # World anchor relative to body B center
    var mass: Scalar[Self.dtype]  # Effective mass for point constraint
    var motor_mass: Scalar[Self.dtype]  # Effective mass for motor constraint

    fn __init__(out self):
        """Create default joint."""
        self.body_a_idx = -1
        self.body_b_idx = -1
        self.local_anchor_a = Vec2[Self.dtype].zero()
        self.local_anchor_b = Vec2[Self.dtype].zero()
        self.reference_angle = 0.0
        self.enable_motor = False
        self.motor_speed = 0.0
        self.max_motor_torque = 0.0
        self.enable_limit = False
        self.lower_angle = 0.0
        self.upper_angle = 0.0
        self.impulse = Vec2[Self.dtype].zero()
        self.motor_impulse = 0.0
        self.lower_impulse = 0.0
        self.upper_impulse = 0.0
        self.r_a = Vec2[Self.dtype].zero()
        self.r_b = Vec2[Self.dtype].zero()
        self.mass = 0.0
        self.motor_mass = 0.0

    fn __init__(
        out self,
        body_a_idx: Int,
        body_b_idx: Int,
        local_anchor_a: Vec2[Self.dtype],
        local_anchor_b: Vec2[Self.dtype],
        reference_angle: Scalar[Self.dtype] = 0.0,
        enable_motor: Bool = False,
        motor_speed: Scalar[Self.dtype] = 0.0,
        max_motor_torque: Scalar[Self.dtype] = 0.0,
        enable_limit: Bool = False,
        lower_angle: Scalar[Self.dtype] = 0.0,
        upper_angle: Scalar[Self.dtype] = 0.0,
    ):
        """Create revolute joint with configuration."""
        self.body_a_idx = body_a_idx
        self.body_b_idx = body_b_idx
        self.local_anchor_a = local_anchor_a
        self.local_anchor_b = local_anchor_b
        self.reference_angle = reference_angle
        self.enable_motor = enable_motor
        self.motor_speed = motor_speed
        self.max_motor_torque = max_motor_torque
        self.enable_limit = enable_limit
        self.lower_angle = lower_angle
        self.upper_angle = upper_angle
        self.impulse = Vec2[Self.dtype].zero()
        self.motor_impulse = 0.0
        self.lower_impulse = 0.0
        self.upper_impulse = 0.0
        self.r_a = Vec2[Self.dtype].zero()
        self.r_b = Vec2[Self.dtype].zero()
        self.mass = 0.0
        self.motor_mass = 0.0

    fn get_joint_angle(
        self, body_a: Body[Self.dtype], body_b: Body[Self.dtype]
    ) -> Scalar[Self.dtype]:
        """Get current joint angle relative to reference."""
        return body_b.angle - body_a.angle - self.reference_angle

    fn get_joint_speed(
        self, body_a: Body[Self.dtype], body_b: Body[Self.dtype]
    ) -> Scalar[Self.dtype]:
        """Get current joint angular velocity."""
        return body_b.angular_velocity - body_a.angular_velocity

    fn init_velocity_constraints(
        mut self,
        body_a: Body[Self.dtype],
        body_b: Body[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        """Initialize solver data for velocity constraints."""
        # Compute world anchor positions relative to body centers
        self.r_a = body_a.get_world_vector(self.local_anchor_a)
        self.r_b = body_b.get_world_vector(self.local_anchor_b)

        # Compute effective mass for point constraint
        # K = M_a^-1 + M_b^-1 + (r_a x n)^2 * I_a^-1 + (r_b x n)^2 * I_b^-1
        var inv_mass_a = body_a.mass_data.inv_mass
        var inv_mass_b = body_b.mass_data.inv_mass
        var inv_inertia_a = body_a.mass_data.inv_inertia
        var inv_inertia_b = body_b.mass_data.inv_inertia

        # For 2D revolute joint, use scalar effective mass
        var k: Scalar[Self.dtype] = inv_mass_a + inv_mass_b
        k += inv_inertia_a * (self.r_a.x * self.r_a.x + self.r_a.y * self.r_a.y)
        k += inv_inertia_b * (self.r_b.x * self.r_b.x + self.r_b.y * self.r_b.y)
        self.mass = 1.0 / k if k > 0.0 else 0.0

        # Motor effective mass
        self.motor_mass = (
            1.0
            / (inv_inertia_a + inv_inertia_b) if (inv_inertia_a + inv_inertia_b)
            > 0.0 else 0.0
        )

    fn solve_velocity_constraints(
        mut self,
        mut body_a: Body[Self.dtype],
        mut body_b: Body[Self.dtype],
    ):
        """Solve velocity constraints (called iteratively)."""
        var inv_mass_a = body_a.mass_data.inv_mass
        var inv_mass_b = body_b.mass_data.inv_mass
        var inv_inertia_a = body_a.mass_data.inv_inertia
        var inv_inertia_b = body_b.mass_data.inv_inertia

        # Solve motor constraint
        if self.enable_motor:
            var cdot = (
                body_b.angular_velocity
                - body_a.angular_velocity
                - self.motor_speed
            )
            var impulse = -self.motor_mass * cdot
            var old_impulse = self.motor_impulse
            var max_impulse = self.max_motor_torque * (
                1.0 / 60.0
            )  # dt normalized
            self.motor_impulse = clamp(
                old_impulse + impulse, -max_impulse, max_impulse
            )
            impulse = self.motor_impulse - old_impulse

            body_a.angular_velocity -= inv_inertia_a * impulse
            body_b.angular_velocity += inv_inertia_b * impulse

        # Solve limit constraints
        if self.enable_limit:
            var joint_angle = self.get_joint_angle(body_a, body_b)
            var angular_vel = body_b.angular_velocity - body_a.angular_velocity

            # Lower limit
            if joint_angle <= self.lower_angle:
                var cdot = angular_vel
                var impulse = -self.motor_mass * cdot
                var old_impulse = self.lower_impulse
                self.lower_impulse = max(old_impulse + impulse, 0.0)
                impulse = self.lower_impulse - old_impulse

                body_a.angular_velocity -= inv_inertia_a * impulse
                body_b.angular_velocity += inv_inertia_b * impulse

            # Upper limit
            if joint_angle >= self.upper_angle:
                var cdot = angular_vel
                var impulse = self.motor_mass * cdot
                var old_impulse = self.upper_impulse
                self.upper_impulse = max(old_impulse + impulse, 0.0)
                impulse = self.upper_impulse - old_impulse

                body_a.angular_velocity += inv_inertia_a * impulse
                body_b.angular_velocity -= inv_inertia_b * impulse

        # Solve point-to-point constraint
        var cdot = (
            body_b.linear_velocity
            + cross_sv(body_b.angular_velocity, self.r_b)
            - body_a.linear_velocity
            - cross_sv(body_a.angular_velocity, self.r_a)
        )

        var impulse = cdot * (-self.mass)
        self.impulse += impulse

        body_a.linear_velocity -= impulse * inv_mass_a
        body_a.angular_velocity -= inv_inertia_a * self.r_a.cross(impulse)

        body_b.linear_velocity += impulse * inv_mass_b
        body_b.angular_velocity += inv_inertia_b * self.r_b.cross(impulse)

    fn solve_position_constraints(
        mut self,
        mut body_a: Body[Self.dtype],
        mut body_b: Body[Self.dtype],
    ) -> Bool:
        """Solve position constraints (position correction).

        Returns True if constraint is satisfied.
        """
        comptime SLOP: Scalar[Self.dtype] = 0.005  # Linear slop
        comptime MAX_CORRECTION: Scalar[Self.dtype] = 0.2

        # Recompute anchor positions
        var r_a = body_a.get_world_vector(self.local_anchor_a)
        var r_b = body_b.get_world_vector(self.local_anchor_b)

        # Position error
        var c = body_b.position + r_b - body_a.position - r_a

        var position_error = c.length()
        var angular_error: Scalar[Self.dtype] = 0.0

        # Compute effective mass
        var inv_mass_a = body_a.mass_data.inv_mass
        var inv_mass_b = body_b.mass_data.inv_mass
        var inv_inertia_a = body_a.mass_data.inv_inertia
        var inv_inertia_b = body_b.mass_data.inv_inertia

        var k: Scalar[Self.dtype] = inv_mass_a + inv_mass_b
        k += inv_inertia_a * (r_a.x * r_a.x + r_a.y * r_a.y)
        k += inv_inertia_b * (r_b.x * r_b.x + r_b.y * r_b.y)
        var mass = 1.0 / k if k > 0.0 else 0.0

        # Compute impulse
        var impulse = c * (-mass)

        # Apply corrections
        body_a.position -= impulse * inv_mass_a
        body_a.angle -= inv_inertia_a * r_a.cross(impulse)

        body_b.position += impulse * inv_mass_b
        body_b.angle += inv_inertia_b * r_b.cross(impulse)

        # Update transforms
        body_a.transform = Transform(body_a.position, body_a.angle)
        body_b.transform = Transform(body_b.position, body_b.angle)

        # Solve angle limits
        if self.enable_limit:
            var joint_angle = self.get_joint_angle(body_a, body_b)

            var angle_error: Scalar[Self.dtype] = 0.0
            if joint_angle < self.lower_angle:
                angle_error = joint_angle - self.lower_angle
            elif joint_angle > self.upper_angle:
                angle_error = joint_angle - self.upper_angle

            if abs(angle_error) > 0.0:
                var motor_mass = (
                    1.0
                    / (inv_inertia_a + inv_inertia_b) if (
                        inv_inertia_a + inv_inertia_b
                    )
                    > 0.0 else 0.0
                )
                var angle_impulse = clamp(
                    -motor_mass * angle_error, -MAX_CORRECTION, MAX_CORRECTION
                )

                body_a.angle -= inv_inertia_a * angle_impulse
                body_b.angle += inv_inertia_b * angle_impulse

                body_a.transform = Transform(body_a.position, body_a.angle)
                body_b.transform = Transform(body_b.position, body_b.angle)

                angular_error = abs(angle_error)

        return position_error < 3.0 * SLOP and angular_error < 0.01
