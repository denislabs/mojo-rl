"""Hinge Joint Definition.

A hinge joint constrains two bodies to rotate around a single axis,
like a door hinge or pendulum pivot.

Constraints:
- 3 DOF position: Anchor points must coincide
- 2 DOF angular: Rotation only allowed around hinge axis

Physics:
- Point-to-point constraint for anchors
- Angular constraint to lock rotation perpendicular to axis
"""

from math import sqrt


@fieldwise_init
struct HingeJoint[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Hinge joint constraint between two bodies.

    A hinge joint allows rotation around a single axis while constraining
    all other degrees of freedom. Used for pendulums, doors, robot joints.

    Parameters:
        DTYPE: Data type for scalar values (float32 or float64).

    Attributes:
        parent_body: Index of parent body (-1 for world anchor).
        child_body: Index of child body.
        anchor_parent_*: Anchor point in parent's local frame (or world if parent=-1).
        anchor_child_*: Anchor point in child's local frame.
        axis_*: Rotation axis in parent's local frame (or world if parent=-1).
        impulse_l*: Accumulated linear impulses (3 DOF for warm starting).
        impulse_a*: Accumulated angular impulses (2 DOF for warm starting).
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

    # Rotation axis in parent's local frame (or world if parent=-1)
    var axis_x: Scalar[Self.DTYPE]
    var axis_y: Scalar[Self.DTYPE]
    var axis_z: Scalar[Self.DTYPE]

    # Accumulated impulses for warm starting (5 DOF total)
    # Linear impulses (3 DOF) - for point-to-point constraint
    var impulse_lx: Scalar[Self.DTYPE]
    var impulse_ly: Scalar[Self.DTYPE]
    var impulse_lz: Scalar[Self.DTYPE]
    # Angular impulses (2 DOF) - for axis alignment constraint
    var impulse_ax: Scalar[Self.DTYPE]
    var impulse_ay: Scalar[Self.DTYPE]

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
        """Create a hinge joint with specified parameters.

        Args:
            parent_body: Parent body index (-1 for world anchor).
            child_body: Child body index.
            anchor_parent: Anchor point in parent's local frame.
            anchor_child: Anchor point in child's local frame.
            axis: Rotation axis (will be normalized).

        Returns:
            Configured HingeJoint.
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
            impulse_lx=Scalar[Self.DTYPE](0),
            impulse_ly=Scalar[Self.DTYPE](0),
            impulse_lz=Scalar[Self.DTYPE](0),
            impulse_ax=Scalar[Self.DTYPE](0),
            impulse_ay=Scalar[Self.DTYPE](0),
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
            axis_x=Scalar[Self.DTYPE](0),
            axis_y=Scalar[Self.DTYPE](1),  # Default Y-axis
            axis_z=Scalar[Self.DTYPE](0),
            impulse_lx=Scalar[Self.DTYPE](0),
            impulse_ly=Scalar[Self.DTYPE](0),
            impulse_lz=Scalar[Self.DTYPE](0),
            impulse_ax=Scalar[Self.DTYPE](0),
            impulse_ay=Scalar[Self.DTYPE](0),
        )

    fn reset_impulses(mut self):
        """Reset accumulated impulses to zero."""
        self.impulse_lx = Scalar[Self.DTYPE](0)
        self.impulse_ly = Scalar[Self.DTYPE](0)
        self.impulse_lz = Scalar[Self.DTYPE](0)
        self.impulse_ax = Scalar[Self.DTYPE](0)
        self.impulse_ay = Scalar[Self.DTYPE](0)
