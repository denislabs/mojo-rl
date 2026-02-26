"""HalfCheetah environment-specific parameters.

The model definition has moved to half_cheetah_xml.mojo (ModelDefFromXML).
This file retains HalfCheetahParams and body/joint index constants
needed by physics3d regression tests.
"""

from physics3d.gpu.constants import (
    state_size,
    model_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    xquat_offset,
    metadata_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    model_curriculum_offset,
)

from .half_cheetah_xml import HalfCheetahModel


# =============================================================================
# HalfCheetahParams — Environment-Specific Parameters
# =============================================================================


struct HalfCheetahParams[DTYPE: DType = DType.float64]:
    """Environment-specific parameters not derivable from the model definition.

    Everything about body geometry, joint limits, gear ratios, damping, stiffness,
    and indices is now in the model definition (BodySpec/JointSpec).

    Type Parameters:
        DTYPE: The floating point type for physics constants.
    """

    # Physics
    comptime FRAME_SKIP: Int = 5  # Number of physics steps per env step (matching MuJoCo)
    comptime MAX_CONTACTS: Int = 20

    # Reward
    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.1
    comptime ANGLE_PENALTY_WEIGHT: Scalar[Self.DTYPE] = 0.5

    # Termination
    comptime MAX_PITCH: Scalar[Self.DTYPE] = 1.0  # ~57 deg
    comptime MAX_STEPS: Int = 1000

    # Curriculum
    comptime CURRICULUM_INITIAL_MAX_PITCH: Scalar[Self.DTYPE] = 3.0
    comptime CURRICULUM_FINAL_MAX_PITCH: Scalar[Self.DTYPE] = 1.0

    # Reset
    comptime RESET_NOISE_SCALE: Scalar[Self.DTYPE] = 0.1
    comptime MIN_ROOTZ: Scalar[Self.DTYPE] = -0.3

    # Dimensions (derived from model definition, for convenience)
    comptime NQ: Int = HalfCheetahModel.NQ
    comptime NV: Int = HalfCheetahModel.NV
    comptime NUM_BODIES: Int = HalfCheetahModel.NBODY
    comptime NUM_JOINTS: Int = HalfCheetahModel.NJOINT
    comptime NGEOM: Int = HalfCheetahModel.NGEOM
    comptime OBS_DIM: Int = 17
    comptime ACTION_DIM: Int = 6

    # Initial torso height (from body_pos_z; qpos[rootz] starts at 0)
    comptime INITIAL_Z: Scalar[Self.DTYPE] = 0.7

    # GPU layout sizes
    comptime STATE_SIZE: Int = state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()
    comptime MODEL_SIZE: Int = model_size[
        Self.NUM_BODIES, Self.NUM_JOINTS, Self.NGEOM
    ]()

    # GPU layout helper methods
    @staticmethod
    @always_inline
    fn get_qpos_offset() -> Int:
        return qpos_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qvel_offset() -> Int:
        return qvel_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qacc_offset() -> Int:
        return qacc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qfrc_offset() -> Int:
        return qfrc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_xpos_offset() -> Int:
        return xpos_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xquat_offset() -> Int:
        return xquat_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_metadata_offset() -> Int:
        return metadata_offset[
            Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
        ]()

    @staticmethod
    @always_inline
    fn get_model_body_offset(body_idx: Int) -> Int:
        return model_body_offset(body_idx)

    @staticmethod
    @always_inline
    fn get_model_joint_offset(joint_idx: Int) -> Int:
        return model_joint_offset[Self.NUM_BODIES](joint_idx)

    @staticmethod
    @always_inline
    fn get_model_metadata_offset() -> Int:
        return model_metadata_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()

    @staticmethod
    @always_inline
    fn get_model_curriculum_offset() -> Int:
        return model_curriculum_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()


# Convenience type aliases
comptime HalfCheetahParamsCPU = HalfCheetahParams[DType.float64]
comptime HalfCheetahParamsGPU = HalfCheetahParams[DType.float32]


# =============================================================================
# Body/Joint Index Constants (for backward compatibility with external consumers)
# =============================================================================

comptime BODY_WORLDBODY: Int = 0
comptime BODY_TORSO: Int = 1
comptime BODY_BTHIGH: Int = 2
comptime BODY_BSHIN: Int = 3
comptime BODY_BFOOT: Int = 4
comptime BODY_FTHIGH: Int = 5
comptime BODY_FSHIN: Int = 6
comptime BODY_FFOOT: Int = 7

comptime JOINT_ROOTX: Int = 0
comptime JOINT_ROOTZ: Int = 1
comptime JOINT_ROOTY: Int = 2
comptime JOINT_BTHIGH: Int = 3
comptime JOINT_BSHIN: Int = 4
comptime JOINT_BFOOT: Int = 5
comptime JOINT_FTHIGH: Int = 6
comptime JOINT_FSHIN: Int = 7
comptime JOINT_FFOOT: Int = 8
