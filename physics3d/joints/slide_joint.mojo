"""Slide Joint (Prismatic Joint) Definition.

A slide joint constrains two bodies to move along a single axis,
like a piston or linear actuator.

Constraints:
- 2 DOF position: Motion only allowed along slide axis
- 3 DOF angular: All rotation locked (relative orientation fixed)

Physics:
- Perpendicular position constraint (2 DOF)
- Angular constraint to lock all relative rotation (3 DOF)
- Optional force actuation along slide axis

Free DOF Mode (Phase 11f):
- When is_free_dof=True, the joint does NOT apply constraints
- Instead, it tracks the position along the axis from body positions (like MuJoCo)
- Used for root joints (world→body) to avoid constraint conflicts
- The qpos field stores the tracked joint position (meters along axis)
"""

from math import sqrt


@fieldwise_init
struct SlideJoint[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Slide (prismatic) joint constraint between two bodies.

    A slide joint allows translation along a single axis while constraining
    all other degrees of freedom. Used for pistons, sliders, linear actuators.

    Parameters:
        DTYPE: Data type for scalar values (float32 or float64).

    Attributes:
        parent_body: Index of parent body (-1 for world anchor).
        child_body: Index of child body.
        anchor_parent_*: Anchor point in parent's local frame (or world if parent=-1).
        anchor_child_*: Anchor point in child's local frame.
        axis_*: Slide axis in parent's local frame (or world if parent=-1).
        impulse_p1/p2: Accumulated impulses for perpendicular constraints (2 DOF).
        impulse_a*: Accumulated angular impulses (3 DOF for warm starting).
        target_force: Control input (N) along slide axis.
        force_limit: Maximum force magnitude.
    """

    var parent_body: Int  # -1 for world anchor
    var child_body: Int

    # Anchor points in local frames
    var anchor_parent_x: Scalar[Self.DTYPE]
    var anchor_parent_y: Scalar[Self.DTYPE]
    var anchor_parent_z: Scalar[Self.DTYPE]
    var anchor_child_x: Scalar[Self.DTYPE]
    var anchor_child_y: Scalar[Self.DTYPE]
    var anchor_child_z: Scalar[Self.DTYPE]

    # Slide axis in parent's local frame (or world if parent=-1)
    var axis_x: Scalar[Self.DTYPE]
    var axis_y: Scalar[Self.DTYPE]
    var axis_z: Scalar[Self.DTYPE]

    # Accumulated impulses for warm starting (5 DOF total)
    # Perpendicular linear impulses (2 DOF) - for motion constraint perpendicular to axis
    var impulse_p1: Scalar[Self.DTYPE]
    var impulse_p2: Scalar[Self.DTYPE]
    # Angular impulses (3 DOF) - for locking all rotation
    var impulse_ax: Scalar[Self.DTYPE]
    var impulse_ay: Scalar[Self.DTYPE]
    var impulse_az: Scalar[Self.DTYPE]

    # Actuation
    var target_force: Scalar[Self.DTYPE]  # Control input (N)
    var force_limit: Scalar[Self.DTYPE]  # Maximum force magnitude

    # Free DOF mode (Phase 11f)
    # When True, joint tracks state without applying constraints (MuJoCo-style)
    var is_free_dof: Bool
    var qpos: Scalar[Self.DTYPE]  # Tracked joint position (meters along axis)
    var qvel: Scalar[Self.DTYPE]  # Tracked joint velocity (m/s along axis)

    @staticmethod
    fn create(
        parent_body: Int,
        child_body: Int,
        anchor_parent: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        anchor_child: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
    ) -> Self:
        """Create a slide joint with specified parameters.

        Args:
            parent_body: Parent body index (-1 for world anchor).
            child_body: Child body index.
            anchor_parent: Anchor point in parent's local frame.
            anchor_child: Anchor point in child's local frame.
            axis: Slide axis (will be normalized).

        Returns:
            Configured SlideJoint.
        """
        # Normalize axis
        var ax = axis[0]
        var ay = axis[1]
        var az = axis[2]
        var length_sq = ax * ax + ay * ay + az * az
        var inv_length = Scalar[Self.DTYPE](1.0) / sqrt(
            length_sq + Scalar[Self.DTYPE](1e-10)
        )
        ax = ax * inv_length
        ay = ay * inv_length
        az = az * inv_length

        return Self(
            parent_body=parent_body,
            child_body=child_body,
            anchor_parent_x=anchor_parent[0],
            anchor_parent_y=anchor_parent[1],
            anchor_parent_z=anchor_parent[2],
            anchor_child_x=anchor_child[0],
            anchor_child_y=anchor_child[1],
            anchor_child_z=anchor_child[2],
            axis_x=ax,
            axis_y=ay,
            axis_z=az,
            impulse_p1=Scalar[Self.DTYPE](0),
            impulse_p2=Scalar[Self.DTYPE](0),
            impulse_ax=Scalar[Self.DTYPE](0),
            impulse_ay=Scalar[Self.DTYPE](0),
            impulse_az=Scalar[Self.DTYPE](0),
            target_force=Scalar[Self.DTYPE](0),
            force_limit=Scalar[Self.DTYPE](1000.0),  # Default 1000 N limit
            is_free_dof=False,  # Default: apply constraints
            qpos=Scalar[Self.DTYPE](0),
            qvel=Scalar[Self.DTYPE](0),
        )

    @staticmethod
    fn empty() -> Self:
        """Create an empty/uninitialized joint."""
        return Self(
            parent_body=-1,
            child_body=-1,
            anchor_parent_x=Scalar[Self.DTYPE](0),
            anchor_parent_y=Scalar[Self.DTYPE](0),
            anchor_parent_z=Scalar[Self.DTYPE](0),
            anchor_child_x=Scalar[Self.DTYPE](0),
            anchor_child_y=Scalar[Self.DTYPE](0),
            anchor_child_z=Scalar[Self.DTYPE](0),
            axis_x=Scalar[Self.DTYPE](1),  # Default X-axis
            axis_y=Scalar[Self.DTYPE](0),
            axis_z=Scalar[Self.DTYPE](0),
            impulse_p1=Scalar[Self.DTYPE](0),
            impulse_p2=Scalar[Self.DTYPE](0),
            impulse_ax=Scalar[Self.DTYPE](0),
            impulse_ay=Scalar[Self.DTYPE](0),
            impulse_az=Scalar[Self.DTYPE](0),
            target_force=Scalar[Self.DTYPE](0),
            force_limit=Scalar[Self.DTYPE](1000.0),
            is_free_dof=False,
            qpos=Scalar[Self.DTYPE](0),
            qvel=Scalar[Self.DTYPE](0),
        )

    @staticmethod
    fn create_free_dof(
        parent_body: Int,
        child_body: Int,
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
    ) -> Self:
        """Create a free DOF slide joint (MuJoCo-style root joint).

        A free DOF joint tracks the position along the axis but does NOT
        apply constraints. Used for root joints where the body should move
        freely while tracking the position for observations.

        Args:
            parent_body: Parent body index (-1 for world).
            child_body: Child body index.
            axis: Slide axis (will be normalized).

        Returns:
            Configured free DOF SlideJoint.
        """
        # Normalize axis
        var ax = axis[0]
        var ay = axis[1]
        var az = axis[2]
        var length_sq = ax * ax + ay * ay + az * az
        var inv_length = Scalar[Self.DTYPE](1.0) / sqrt(
            length_sq + Scalar[Self.DTYPE](1e-10)
        )
        ax = ax * inv_length
        ay = ay * inv_length
        az = az * inv_length

        return Self(
            parent_body=parent_body,
            child_body=child_body,
            # Anchors not used for free DOF
            anchor_parent_x=Scalar[Self.DTYPE](0),
            anchor_parent_y=Scalar[Self.DTYPE](0),
            anchor_parent_z=Scalar[Self.DTYPE](0),
            anchor_child_x=Scalar[Self.DTYPE](0),
            anchor_child_y=Scalar[Self.DTYPE](0),
            anchor_child_z=Scalar[Self.DTYPE](0),
            axis_x=ax,
            axis_y=ay,
            axis_z=az,
            impulse_p1=Scalar[Self.DTYPE](0),
            impulse_p2=Scalar[Self.DTYPE](0),
            impulse_ax=Scalar[Self.DTYPE](0),
            impulse_ay=Scalar[Self.DTYPE](0),
            impulse_az=Scalar[Self.DTYPE](0),
            target_force=Scalar[Self.DTYPE](0),
            force_limit=Scalar[Self.DTYPE](0),  # No actuation for root joint
            is_free_dof=True,  # Free DOF mode
            qpos=Scalar[Self.DTYPE](0),
            qvel=Scalar[Self.DTYPE](0),
        )

    fn reset_impulses(mut self):
        """Reset accumulated impulses to zero."""
        self.impulse_p1 = Scalar[Self.DTYPE](0)
        self.impulse_p2 = Scalar[Self.DTYPE](0)
        self.impulse_ax = Scalar[Self.DTYPE](0)
        self.impulse_ay = Scalar[Self.DTYPE](0)
        self.impulse_az = Scalar[Self.DTYPE](0)

    fn set_force(mut self, force: Scalar[Self.DTYPE]):
        """Set target force, clamped to force_limit.

        Args:
            force: Desired force in N (positive = push along axis direction).
        """
        # Clamp to limits
        if force > self.force_limit:
            self.target_force = self.force_limit
        elif force < -self.force_limit:
            self.target_force = -self.force_limit
        else:
            self.target_force = force

    fn set_force_limit(mut self, limit: Scalar[Self.DTYPE]):
        """Set the maximum force magnitude.

        Args:
            limit: Maximum force in N (must be positive).
        """
        if limit > Scalar[Self.DTYPE](0):
            self.force_limit = limit
