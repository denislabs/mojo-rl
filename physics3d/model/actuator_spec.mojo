"""ActuatorSpec trait and concrete actuator types for compile-time model definitions.

MuJoCo actuator abstraction: separates actuators from joints.
Supports dynamics (none, integrator, filter), gain/bias functions,
force limits, and gear ratios.

Actuator types:
  - MotorActuator: Direct torque/force (gain=gear, no bias)
  - PositionActuator: PD servo (force = kp*(ctrl-qpos) - kd*qvel)
  - VelocityActuator: Velocity servo (force = kv*(ctrl-qvel))
  - GeneralActuator: Full control over all parameters
"""


# Dynamics type constants
comptime DYN_NONE: Int = 0
comptime DYN_INTEGRATOR: Int = 1
comptime DYN_FILTER: Int = 2
comptime DYN_FILTEREXACT: Int = 3

# Gain type constants
comptime GAIN_FIXED: Int = 0
comptime GAIN_AFFINE: Int = 1

# Bias type constants
comptime BIAS_NONE: Int = 0
comptime BIAS_AFFINE: Int = 1


# =============================================================================
# ActuatorSpec Trait
# =============================================================================


trait ActuatorSpec(TrivialRegisterPassable):
    """Compile-time actuator specification for physics3d model definitions.

    Maps to MuJoCo's <general> actuator element with full control over
    dynamics, gain, and bias functions.

    DOF_ADR and QPOS_ADR are the compile-time DOF and qpos addresses for
    the actuated joint, matching Joints._qvel_offset[joint_idx]() and
    Joints._qpos_offset[joint_idx]() respectively. These must be set by
    the user to match the joint layout.
    """

    comptime JOINT_IDX: Int        # Which joint this actuates
    comptime DOF_ADR: Int          # DOF address (qvel offset of the joint)
    comptime QPOS_ADR: Int         # Qpos address (qpos offset of the joint)
    comptime GEAR: Float64         # Force scaling (gear ratio)
    comptime DYNTYPE: Int          # DYN_NONE / DYN_INTEGRATOR / DYN_FILTER / DYN_FILTEREXACT
    comptime DYNPRM_0: Float64     # Time constant for filter (default 1.0)
    comptime GAINTYPE: Int         # GAIN_FIXED / GAIN_AFFINE
    comptime GAINPRM_0: Float64    # Gain coefficient 0 (fixed gain, or affine intercept)
    comptime GAINPRM_1: Float64    # Gain coefficient 1 (length-dependent)
    comptime GAINPRM_2: Float64    # Gain coefficient 2 (velocity-dependent)
    comptime BIASTYPE: Int         # BIAS_NONE / BIAS_AFFINE
    comptime BIASPRM_0: Float64    # Bias coefficient 0 (constant)
    comptime BIASPRM_1: Float64    # Bias coefficient 1 (length-dependent)
    comptime BIASPRM_2: Float64    # Bias coefficient 2 (velocity-dependent)
    comptime CTRL_MIN: Float64     # Control range min (default -1.0)
    comptime CTRL_MAX: Float64     # Control range max (default 1.0)
    comptime FORCE_MIN: Float64    # Force range min (default -inf)
    comptime FORCE_MAX: Float64    # Force range max (default +inf)
    comptime HAS_ACTIVATION: Bool  # Whether this actuator has internal state


# =============================================================================
# MotorActuator
# =============================================================================


@fieldwise_init
struct MotorActuator[
    joint_idx: Int,
    dof_adr: Int,
    qpos_adr: Int = -1,  # Defaults to dof_adr if not specified
    gear: Float64 = 1.0,
    ctrl_min: Float64 = -1.0,
    ctrl_max: Float64 = 1.0,
    force_min: Float64 = -1e10,
    force_max: Float64 = 1e10,
](ActuatorSpec):
    """Motor actuator: direct torque/force control.

    force = gear * clamp(ctrl, ctrl_min, ctrl_max)

    Equivalent to MuJoCo <motor> element.
    """

    comptime JOINT_IDX: Int = Self.joint_idx
    comptime DOF_ADR: Int = Self.dof_adr
    comptime QPOS_ADR: Int = Self.qpos_adr if Self.qpos_adr >= 0 else Self.dof_adr
    comptime GEAR: Float64 = Self.gear
    comptime DYNTYPE: Int = DYN_NONE
    comptime DYNPRM_0: Float64 = 1.0
    comptime GAINTYPE: Int = GAIN_FIXED
    comptime GAINPRM_0: Float64 = 1.0
    comptime GAINPRM_1: Float64 = 0.0
    comptime GAINPRM_2: Float64 = 0.0
    comptime BIASTYPE: Int = BIAS_NONE
    comptime BIASPRM_0: Float64 = 0.0
    comptime BIASPRM_1: Float64 = 0.0
    comptime BIASPRM_2: Float64 = 0.0
    comptime CTRL_MIN: Float64 = Self.ctrl_min
    comptime CTRL_MAX: Float64 = Self.ctrl_max
    comptime FORCE_MIN: Float64 = Self.force_min
    comptime FORCE_MAX: Float64 = Self.force_max
    comptime HAS_ACTIVATION: Bool = False


# =============================================================================
# PositionActuator
# =============================================================================


@fieldwise_init
struct PositionActuator[
    joint_idx: Int,
    dof_adr: Int,
    qpos_adr: Int = -1,
    kp: Float64 = 1.0,
    kd: Float64 = 0.0,
    ctrl_min: Float64 = -1.0,
    ctrl_max: Float64 = 1.0,
    force_min: Float64 = -1e10,
    force_max: Float64 = 1e10,
](ActuatorSpec):
    """Position servo actuator: PD controller.

    force = kp * (ctrl - qpos) - kd * qvel
          = kp * ctrl + (0 - kp*qpos - kd*qvel)

    Implemented as: gain=FIXED(kp), bias=AFFINE(0, -kp, -kd)

    Equivalent to MuJoCo <position> element.
    """

    comptime JOINT_IDX: Int = Self.joint_idx
    comptime DOF_ADR: Int = Self.dof_adr
    comptime QPOS_ADR: Int = Self.qpos_adr if Self.qpos_adr >= 0 else Self.dof_adr
    comptime GEAR: Float64 = 1.0
    comptime DYNTYPE: Int = DYN_NONE
    comptime DYNPRM_0: Float64 = 1.0
    comptime GAINTYPE: Int = GAIN_FIXED
    comptime GAINPRM_0: Float64 = Self.kp
    comptime GAINPRM_1: Float64 = 0.0
    comptime GAINPRM_2: Float64 = 0.0
    comptime BIASTYPE: Int = BIAS_AFFINE
    comptime BIASPRM_0: Float64 = 0.0
    comptime BIASPRM_1: Float64 = -Self.kp
    comptime BIASPRM_2: Float64 = -Self.kd
    comptime CTRL_MIN: Float64 = Self.ctrl_min
    comptime CTRL_MAX: Float64 = Self.ctrl_max
    comptime FORCE_MIN: Float64 = Self.force_min
    comptime FORCE_MAX: Float64 = Self.force_max
    comptime HAS_ACTIVATION: Bool = False


# =============================================================================
# VelocityActuator
# =============================================================================


@fieldwise_init
struct VelocityActuator[
    joint_idx: Int,
    dof_adr: Int,
    qpos_adr: Int = -1,
    kv: Float64 = 1.0,
    ctrl_min: Float64 = -1.0,
    ctrl_max: Float64 = 1.0,
    force_min: Float64 = -1e10,
    force_max: Float64 = 1e10,
](ActuatorSpec):
    """Velocity servo actuator.

    force = kv * (ctrl - qvel)
          = kv * ctrl + (0 + 0 - kv*qvel)

    Implemented as: gain=FIXED(kv), bias=AFFINE(0, 0, -kv)

    Equivalent to MuJoCo <velocity> element.
    """

    comptime JOINT_IDX: Int = Self.joint_idx
    comptime DOF_ADR: Int = Self.dof_adr
    comptime QPOS_ADR: Int = Self.qpos_adr if Self.qpos_adr >= 0 else Self.dof_adr
    comptime GEAR: Float64 = 1.0
    comptime DYNTYPE: Int = DYN_NONE
    comptime DYNPRM_0: Float64 = 1.0
    comptime GAINTYPE: Int = GAIN_FIXED
    comptime GAINPRM_0: Float64 = Self.kv
    comptime GAINPRM_1: Float64 = 0.0
    comptime GAINPRM_2: Float64 = 0.0
    comptime BIASTYPE: Int = BIAS_AFFINE
    comptime BIASPRM_0: Float64 = 0.0
    comptime BIASPRM_1: Float64 = 0.0
    comptime BIASPRM_2: Float64 = -Self.kv
    comptime CTRL_MIN: Float64 = Self.ctrl_min
    comptime CTRL_MAX: Float64 = Self.ctrl_max
    comptime FORCE_MIN: Float64 = Self.force_min
    comptime FORCE_MAX: Float64 = Self.force_max
    comptime HAS_ACTIVATION: Bool = False


# =============================================================================
# GeneralActuator
# =============================================================================


@fieldwise_init
struct GeneralActuator[
    joint_idx: Int,
    dof_adr: Int,
    qpos_adr: Int = -1,
    gear: Float64 = 1.0,
    dyntype: Int = DYN_NONE,
    dynprm_0: Float64 = 1.0,
    gaintype: Int = GAIN_FIXED,
    gainprm_0: Float64 = 1.0,
    gainprm_1: Float64 = 0.0,
    gainprm_2: Float64 = 0.0,
    biastype: Int = BIAS_NONE,
    biasprm_0: Float64 = 0.0,
    biasprm_1: Float64 = 0.0,
    biasprm_2: Float64 = 0.0,
    ctrl_min: Float64 = -1.0,
    ctrl_max: Float64 = 1.0,
    force_min: Float64 = -1e10,
    force_max: Float64 = 1e10,
    has_activation: Bool = False,
](ActuatorSpec):
    """General actuator with full control over all parameters.

    Equivalent to MuJoCo <general> element. Allows arbitrary combinations
    of dynamics, gain, and bias functions.
    """

    comptime JOINT_IDX: Int = Self.joint_idx
    comptime DOF_ADR: Int = Self.dof_adr
    comptime QPOS_ADR: Int = Self.qpos_adr if Self.qpos_adr >= 0 else Self.dof_adr
    comptime GEAR: Float64 = Self.gear
    comptime DYNTYPE: Int = Self.dyntype
    comptime DYNPRM_0: Float64 = Self.dynprm_0
    comptime GAINTYPE: Int = Self.gaintype
    comptime GAINPRM_0: Float64 = Self.gainprm_0
    comptime GAINPRM_1: Float64 = Self.gainprm_1
    comptime GAINPRM_2: Float64 = Self.gainprm_2
    comptime BIASTYPE: Int = Self.biastype
    comptime BIASPRM_0: Float64 = Self.biasprm_0
    comptime BIASPRM_1: Float64 = Self.biasprm_1
    comptime BIASPRM_2: Float64 = Self.biasprm_2
    comptime CTRL_MIN: Float64 = Self.ctrl_min
    comptime CTRL_MAX: Float64 = Self.ctrl_max
    comptime FORCE_MIN: Float64 = Self.force_min
    comptime FORCE_MAX: Float64 = Self.force_max
    comptime HAS_ACTIVATION: Bool = Self.has_activation
