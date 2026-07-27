"""Hopper environment-specific parameters.

The model definition has moved to hopper_xml.mojo (ModelDefFromXML).
This file retains HopperParams and body/joint index constants
needed by physics3d regression tests.
"""

from .hopper_xml import HopperModel


# =============================================================================
# HopperParams — Environment-Specific Parameters
# =============================================================================


struct HopperParams[DTYPE: DType = DType.float64]:
    """Environment-specific parameters not derivable from the model definition.

    Replaces the former HopperConstants struct. Everything about body
    geometry, joint limits, gear ratios, damping, and indices is
    now in the model definition (BodySpec/JointSpec).

    Type Parameters:
        DTYPE: The floating point type for physics constants.
    """

    # Physics
    comptime DT: Scalar[Self.DTYPE] = 0.002  # Physics timestep (500 Hz)
    comptime FRAME_SKIP: Int = 4  # Number of physics steps per env step
    comptime GRAVITY_Z: Scalar[Self.DTYPE] = -9.81
    comptime MAX_CONTACTS: Int = 20

    # Reward
    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.001
    comptime HEALTHY_REWARD: Scalar[Self.DTYPE] = 1.0

    # Termination
    comptime MIN_HEIGHT: Scalar[Self.DTYPE] = 0.7
    comptime MAX_PITCH: Scalar[Self.DTYPE] = 0.2  # ~11 deg
    comptime MAX_STEPS: Int = 1000

    # Curriculum
    comptime CURRICULUM_INITIAL_MIN_HEIGHT: Scalar[Self.DTYPE] = 0.3
    comptime CURRICULUM_INITIAL_MAX_PITCH: Scalar[Self.DTYPE] = 1.0
    comptime CURRICULUM_FINAL_MIN_HEIGHT: Scalar[Self.DTYPE] = 0.7
    comptime CURRICULUM_FINAL_MAX_PITCH: Scalar[Self.DTYPE] = 0.2

    # Reset
    comptime RESET_NOISE_SCALE: Scalar[Self.DTYPE] = 0.005

    # Dimensions (derived from model definition, for convenience)
    comptime NQ: Int = HopperModel.NQ
    comptime NV: Int = HopperModel.NV
    comptime NUM_BODIES: Int = HopperModel.NBODY
    comptime NUM_JOINTS: Int = HopperModel.NJOINT
    comptime NGEOM: Int = HopperModel.NGEOM
    comptime OBS_DIM: Int = 11
    comptime ACTION_DIM: Int = 3

    # Initial torso height (qpos[rootz] = 1.25 = qpos0, matching MuJoCo ref)
    comptime INITIAL_Z: Scalar[Self.DTYPE] = 1.25

    # Motor
    comptime TORQUE_LIMIT: Scalar[Self.DTYPE] = 200.0

comptime BODY_WORLDBODY: Int = 0
comptime BODY_TORSO: Int = 1
comptime BODY_THIGH: Int = 2
comptime BODY_LEG: Int = 3
comptime BODY_FOOT: Int = 4

comptime JOINT_ROOTX: Int = 0
comptime JOINT_ROOTZ: Int = 1
comptime JOINT_ROOTY: Int = 2
comptime JOINT_THIGH: Int = 3
comptime JOINT_LEG: Int = 4
comptime JOINT_FOOT: Int = 5

# Body geometry constants for backward compatibility
comptime TORSO_RADIUS: Float64 = 0.05
comptime TORSO_HALF_LENGTH: Float64 = 0.2
comptime THIGH_RADIUS: Float64 = 0.05
comptime THIGH_HALF_LENGTH: Float64 = 0.225
comptime LEG_RADIUS: Float64 = 0.04
comptime LEG_HALF_LENGTH: Float64 = 0.25
comptime FOOT_RADIUS: Float64 = 0.06
comptime FOOT_HALF_LENGTH: Float64 = 0.195

# Body mass constants for backward compatibility
comptime TORSO_MASS: Float64 = 3.665191
comptime THIGH_MASS: Float64 = 4.057891
comptime LEG_MASS: Float64 = 2.781357
comptime FOOT_MASS: Float64 = 5.315575

# Dimension constants for backward compatibility
comptime NQ: Int = HopperModel.NQ
comptime NV: Int = HopperModel.NV
comptime NBODY: Int = HopperModel.NBODY
comptime NJOINT: Int = HopperModel.NJOINT
comptime MAX_CONTACTS: Int = 20
comptime NGEOM: Int = HopperModel.NGEOM
comptime OBS_DIM: Int = 11
comptime ACTION_DIM: Int = 3
comptime NUM_BODIES: Int = 5

# Physics constants for backward compatibility
comptime DT: Float64 = 0.002
comptime FRAME_SKIP: Int = 4
comptime EFFECTIVE_DT: Float64 = DT * Float64(FRAME_SKIP)
comptime MAX_STEPS: Int = 1000
comptime INITIAL_Z: Float64 = 1.25
comptime FRICTION: Float64 = 1.0
comptime RESTITUTION: Float64 = 0.0
comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0
comptime CTRL_COST_WEIGHT: Float64 = 0.001
comptime HEALTHY_REWARD: Float64 = 1.0
comptime RESET_NOISE_SCALE: Float64 = 0.005
comptime TORQUE_LIMIT: Float64 = 200.0

# Joint limits for backward compatibility
comptime THIGH_JOINT_MIN: Float64 = -2.618
comptime THIGH_JOINT_MAX: Float64 = 0.0
comptime LEG_JOINT_MIN: Float64 = -2.618
comptime LEG_JOINT_MAX: Float64 = 0.0
comptime FOOT_JOINT_MIN: Float64 = -0.785
comptime FOOT_JOINT_MAX: Float64 = 0.785
