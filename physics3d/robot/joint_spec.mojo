"""JointSpec trait and concrete joint types for compile-time robot definitions.

Defines joint type, DOF dimensions, anchor, axis, limits, and dynamics
as compile-time constants. Joint types reuse constants from
physics3d/joint_types.mojo: JNT_HINGE = 3, JNT_SLIDE = 2.
"""

from ..joint_types import JNT_HINGE, JNT_SLIDE


# =============================================================================
# JointSpec Trait
# =============================================================================


trait JointSpec(TrivialRegisterPassable):
    """Compile-time joint specification for physics3d robot definitions.

    Properties match JointDef fields and GPU buffer layout.
    """

    comptime JNT_TYPE: Int  # JNT_HINGE, JNT_SLIDE
    comptime NQ: Int  # Dimension in qpos (1 for hinge/slide)
    comptime NV: Int  # Dimension in qvel (1 for hinge/slide)
    comptime BODY_IDX: Int  # Which body this joint is on

    # Joint anchor in parent frame
    comptime POS_X: Float64
    comptime POS_Y: Float64
    comptime POS_Z: Float64

    # Joint axis
    comptime AXIS_X: Float64
    comptime AXIS_Y: Float64
    comptime AXIS_Z: Float64

    # Limits and dynamics
    comptime TAU_LIMIT: Float64  # Torque/force limit (gear ratio for actuated)
    comptime RANGE_MIN: Float64  # Position lower limit
    comptime RANGE_MAX: Float64  # Position upper limit
    comptime ARMATURE: Float64  # Rotor inertia
    comptime DAMPING: Float64  # Velocity-dependent force
    comptime STIFFNESS: Float64  # Position-dependent spring
    comptime SPRINGREF: Float64  # Spring reference position
    comptime FRICTIONLOSS: Float64  # Dry friction torque
    comptime INIT_QPOS: Float64  # Initial joint position (qpos0)

    # Observation/actuation flags (for generic env infrastructure)
    comptime EXCLUDE_OBS_QPOS: Bool  # Skip qpos from observation
    comptime EXCLUDE_OBS_QVEL: Bool  # Skip qvel from observation
    comptime IS_ACTUATED: Bool  # Has motor (for action mapping)
    comptime HAS_LIMITS: Bool  # Has meaningful position limits


# =============================================================================
# HingeJoint
# =============================================================================


@fieldwise_init
struct HingeJoint[
    body_idx: Int,
    axis_x: Float64 = 0.0,
    axis_y: Float64 = 1.0,
    axis_z: Float64 = 0.0,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    tau_limit: Float64 = 1000.0,
    range_min: Float64 = -3.14,
    range_max: Float64 = 3.14,
    armature: Float64 = 0.1,
    damping: Float64 = 0.0,
    stiffness: Float64 = 0.0,
    springref: Float64 = 0.0,
    frictionloss: Float64 = 0.0,
    init_qpos: Float64 = 0.0,
    exclude_obs_qpos: Bool = False,
    exclude_obs_qvel: Bool = False,
    is_actuated: Bool = True,
    has_limits: Bool = True,
](JointSpec):
    """Revolute (hinge) joint: 1 DOF rotation around axis.

    Default axis is Y (into-screen for 2D planar robots like HalfCheetah).
    """

    comptime JNT_TYPE: Int = JNT_HINGE
    comptime NQ: Int = 1
    comptime NV: Int = 1
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime AXIS_X: Float64 = Self.axis_x
    comptime AXIS_Y: Float64 = Self.axis_y
    comptime AXIS_Z: Float64 = Self.axis_z
    comptime TAU_LIMIT: Float64 = Self.tau_limit
    comptime RANGE_MIN: Float64 = Self.range_min
    comptime RANGE_MAX: Float64 = Self.range_max
    comptime ARMATURE: Float64 = Self.armature
    comptime DAMPING: Float64 = Self.damping
    comptime STIFFNESS: Float64 = Self.stiffness
    comptime SPRINGREF: Float64 = Self.springref
    comptime FRICTIONLOSS: Float64 = Self.frictionloss
    comptime INIT_QPOS: Float64 = Self.init_qpos
    comptime EXCLUDE_OBS_QPOS: Bool = Self.exclude_obs_qpos
    comptime EXCLUDE_OBS_QVEL: Bool = Self.exclude_obs_qvel
    comptime IS_ACTUATED: Bool = Self.is_actuated
    comptime HAS_LIMITS: Bool = Self.has_limits


# =============================================================================
# SlideJoint
# =============================================================================


@fieldwise_init
struct SlideJoint[
    body_idx: Int,
    axis_x: Float64 = 1.0,
    axis_y: Float64 = 0.0,
    axis_z: Float64 = 0.0,
    pos_x: Float64 = 0.0,
    pos_y: Float64 = 0.0,
    pos_z: Float64 = 0.0,
    tau_limit: Float64 = 0.0,
    range_min: Float64 = -1e6,
    range_max: Float64 = 1e6,
    armature: Float64 = 0.0,
    damping: Float64 = 0.0,
    stiffness: Float64 = 0.0,
    springref: Float64 = 0.0,
    frictionloss: Float64 = 0.0,
    init_qpos: Float64 = 0.0,
    exclude_obs_qpos: Bool = False,
    exclude_obs_qvel: Bool = False,
    is_actuated: Bool = False,
    has_limits: Bool = False,
](JointSpec):
    """Prismatic (slide) joint: 1 DOF translation along axis.

    Default axis is X. Typically used for root translation DOFs.
    """

    comptime JNT_TYPE: Int = JNT_SLIDE
    comptime NQ: Int = 1
    comptime NV: Int = 1
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = Self.pos_x
    comptime POS_Y: Float64 = Self.pos_y
    comptime POS_Z: Float64 = Self.pos_z
    comptime AXIS_X: Float64 = Self.axis_x
    comptime AXIS_Y: Float64 = Self.axis_y
    comptime AXIS_Z: Float64 = Self.axis_z
    comptime TAU_LIMIT: Float64 = Self.tau_limit
    comptime RANGE_MIN: Float64 = Self.range_min
    comptime RANGE_MAX: Float64 = Self.range_max
    comptime ARMATURE: Float64 = Self.armature
    comptime EXCLUDE_OBS_QPOS: Bool = Self.exclude_obs_qpos
    comptime EXCLUDE_OBS_QVEL: Bool = Self.exclude_obs_qvel
    comptime IS_ACTUATED: Bool = Self.is_actuated
    comptime HAS_LIMITS: Bool = Self.has_limits
    comptime DAMPING: Float64 = Self.damping
    comptime STIFFNESS: Float64 = Self.stiffness
    comptime SPRINGREF: Float64 = Self.springref
    comptime FRICTIONLOSS: Float64 = Self.frictionloss
    comptime INIT_QPOS: Float64 = Self.init_qpos
