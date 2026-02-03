"""Joint type definitions for MuJoCo-style Generalized Coordinates engine.

Joint types define how DOFs are added to the kinematic chain:
- FREE: 7 qpos (3 pos + 4 quat), 6 qvel (3 lin + 3 ang) - free floating body
- BALL: 4 qpos (quaternion), 3 qvel (angular) - ball-and-socket
- SLIDE: 1 qpos (position along axis), 1 qvel - prismatic
- HINGE: 1 qpos (angle around axis), 1 qvel - revolute

Reference: MuJoCo joint types (mjJNT_FREE, mjJNT_BALL, mjJNT_SLIDE, mjJNT_HINGE)
"""

from math import sqrt

# =============================================================================
# Joint Type Constants
# =============================================================================

comptime JNT_FREE: Int = 0  # Free floating (7 qpos, 6 qvel)
comptime JNT_BALL: Int = 1  # Ball-and-socket (4 qpos, 3 qvel)
comptime JNT_SLIDE: Int = 2  # Prismatic (1 qpos, 1 qvel)
comptime JNT_HINGE: Int = 3  # Revolute (1 qpos, 1 qvel)

# DOF sizes for each joint type
comptime FREE_QPOS_SIZE: Int = 7  # 3 position + 4 quaternion
comptime FREE_QVEL_SIZE: Int = 6  # 3 linear + 3 angular
comptime BALL_QPOS_SIZE: Int = 4  # quaternion
comptime BALL_QVEL_SIZE: Int = 3  # angular velocity
comptime SLIDE_QPOS_SIZE: Int = 1  # position along axis
comptime SLIDE_QVEL_SIZE: Int = 1  # velocity along axis
comptime HINGE_QPOS_SIZE: Int = 1  # angle
comptime HINGE_QVEL_SIZE: Int = 1  # angular velocity


# =============================================================================
# JointDef - Static Joint Definition
# =============================================================================


@fieldwise_init
struct JointDef[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Static joint definition for generalized coordinates engine.

    Each joint specifies:
    - Type (FREE, BALL, SLIDE, HINGE)
    - Which body it belongs to
    - Address in qpos/qvel arrays
    - Anchor position in parent body frame
    - Axis direction (for HINGE/SLIDE)
    - Torque/force limits
    """

    var jnt_type: Int  # JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
    var body_id: Int  # Body this joint belongs to
    var qpos_adr: Int  # Start index in qpos array
    var dof_adr: Int  # Start index in qvel array (dof = degree of freedom)

    # Joint anchor in parent body frame (or world if body has no parent)
    var pos_x: Scalar[Self.DTYPE]
    var pos_y: Scalar[Self.DTYPE]
    var pos_z: Scalar[Self.DTYPE]

    # Joint axis (for HINGE/SLIDE, ignored for FREE/BALL)
    var axis_x: Scalar[Self.DTYPE]
    var axis_y: Scalar[Self.DTYPE]
    var axis_z: Scalar[Self.DTYPE]

    # Torque/force limit
    var tau_limit: Scalar[Self.DTYPE]

    @staticmethod
    fn empty() -> Self:
        """Create an empty/default joint definition."""
        return Self(
            jnt_type=JNT_HINGE,
            body_id=0,
            qpos_adr=0,
            dof_adr=0,
            pos_x=Scalar[Self.DTYPE](0),
            pos_y=Scalar[Self.DTYPE](0),
            pos_z=Scalar[Self.DTYPE](0),
            axis_x=Scalar[Self.DTYPE](0),
            axis_y=Scalar[Self.DTYPE](1),
            axis_z=Scalar[Self.DTYPE](0),
            tau_limit=Scalar[Self.DTYPE](1000.0),
        )

    @staticmethod
    fn create_hinge(
        body_id: Int,
        qpos_adr: Int,
        dof_adr: Int,
        pos: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        tau_limit: Scalar[Self.DTYPE] = 1000.0,
    ) -> Self:
        """Create a hinge (revolute) joint.

        Args:
            body_id: Index of the body this joint is attached to.
            qpos_adr: Start index in qpos array.
            dof_adr: Start index in qvel array.
            pos: Joint anchor position in parent frame.
            axis: Rotation axis (will be normalized).
            tau_limit: Maximum torque magnitude.

        Returns:
            JointDef configured as a hinge joint.
        """
        # Normalize axis
        var ax = axis[0]
        var ay = axis[1]
        var az = axis[2]
        var length = sqrt(ax * ax + ay * ay + az * az)
        if length > Scalar[Self.DTYPE](1e-10):
            ax = ax / length
            ay = ay / length
            az = az / length

        return Self(
            jnt_type=JNT_HINGE,
            body_id=body_id,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
            pos_x=pos[0],
            pos_y=pos[1],
            pos_z=pos[2],
            axis_x=ax,
            axis_y=ay,
            axis_z=az,
            tau_limit=tau_limit,
        )

    @staticmethod
    fn create_slide(
        body_id: Int,
        qpos_adr: Int,
        dof_adr: Int,
        pos: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        force_limit: Scalar[Self.DTYPE] = 1000.0,
    ) -> Self:
        """Create a slide (prismatic) joint.

        Args:
            body_id: Index of the body this joint is attached to.
            qpos_adr: Start index in qpos array.
            dof_adr: Start index in qvel array.
            pos: Joint anchor position in parent frame.
            axis: Slide axis (will be normalized).
            force_limit: Maximum force magnitude.

        Returns:
            JointDef configured as a slide joint.
        """
        # Normalize axis
        var ax = axis[0]
        var ay = axis[1]
        var az = axis[2]
        var length = sqrt(ax * ax + ay * ay + az * az)
        if length > Scalar[Self.DTYPE](1e-10):
            ax = ax / length
            ay = ay / length
            az = az / length

        return Self(
            jnt_type=JNT_SLIDE,
            body_id=body_id,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
            pos_x=pos[0],
            pos_y=pos[1],
            pos_z=pos[2],
            axis_x=ax,
            axis_y=ay,
            axis_z=az,
            tau_limit=force_limit,
        )

    @staticmethod
    fn create_ball(
        body_id: Int,
        qpos_adr: Int,
        dof_adr: Int,
        pos: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        tau_limit: Scalar[Self.DTYPE] = 1000.0,
    ) -> Self:
        """Create a ball (spherical) joint.

        Args:
            body_id: Index of the body this joint is attached to.
            qpos_adr: Start index in qpos array.
            dof_adr: Start index in qvel array.
            pos: Joint anchor position in parent frame.
            tau_limit: Maximum torque magnitude.

        Returns:
            JointDef configured as a ball joint.
        """
        return Self(
            jnt_type=JNT_BALL,
            body_id=body_id,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
            pos_x=pos[0],
            pos_y=pos[1],
            pos_z=pos[2],
            axis_x=Scalar[Self.DTYPE](0),
            axis_y=Scalar[Self.DTYPE](0),
            axis_z=Scalar[Self.DTYPE](1),
            tau_limit=tau_limit,
        )

    @staticmethod
    fn create_free(
        body_id: Int,
        qpos_adr: Int,
        dof_adr: Int,
    ) -> Self:
        """Create a free joint (6 DOF).

        Args:
            body_id: Index of the body this joint is attached to.
            qpos_adr: Start index in qpos array.
            dof_adr: Start index in qvel array.

        Returns:
            JointDef configured as a free joint.
        """
        return Self(
            jnt_type=JNT_FREE,
            body_id=body_id,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
            pos_x=Scalar[Self.DTYPE](0),
            pos_y=Scalar[Self.DTYPE](0),
            pos_z=Scalar[Self.DTYPE](0),
            axis_x=Scalar[Self.DTYPE](0),
            axis_y=Scalar[Self.DTYPE](0),
            axis_z=Scalar[Self.DTYPE](1),
            tau_limit=Scalar[Self.DTYPE](0),  # No limit for free joint
        )

    fn qpos_size(self) -> Int:
        """Get the number of qpos elements for this joint type."""
        if self.jnt_type == JNT_FREE:
            return FREE_QPOS_SIZE
        elif self.jnt_type == JNT_BALL:
            return BALL_QPOS_SIZE
        elif self.jnt_type == JNT_SLIDE:
            return SLIDE_QPOS_SIZE
        else:  # JNT_HINGE
            return HINGE_QPOS_SIZE

    fn qvel_size(self) -> Int:
        """Get the number of qvel elements for this joint type."""
        if self.jnt_type == JNT_FREE:
            return FREE_QVEL_SIZE
        elif self.jnt_type == JNT_BALL:
            return BALL_QVEL_SIZE
        elif self.jnt_type == JNT_SLIDE:
            return SLIDE_QVEL_SIZE
        else:  # JNT_HINGE
            return HINGE_QVEL_SIZE


# =============================================================================
# Helper Functions
# =============================================================================


fn get_joint_qpos_size(jnt_type: Int) -> Int:
    """Get qpos size for a joint type."""
    if jnt_type == JNT_FREE:
        return FREE_QPOS_SIZE
    elif jnt_type == JNT_BALL:
        return BALL_QPOS_SIZE
    elif jnt_type == JNT_SLIDE:
        return SLIDE_QPOS_SIZE
    else:  # JNT_HINGE
        return HINGE_QPOS_SIZE


fn get_joint_qvel_size(jnt_type: Int) -> Int:
    """Get qvel size for a joint type."""
    if jnt_type == JNT_FREE:
        return FREE_QVEL_SIZE
    elif jnt_type == JNT_BALL:
        return BALL_QVEL_SIZE
    elif jnt_type == JNT_SLIDE:
        return SLIDE_QVEL_SIZE
    else:  # JNT_HINGE
        return HINGE_QVEL_SIZE
