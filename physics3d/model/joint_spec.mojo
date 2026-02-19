"""JointSpec trait and concrete joint types for compile-time model definitions.

Defines joint type, DOF dimensions, anchor, axis, limits, and dynamics
as compile-time constants. Joint types reuse constants from
physics3d/joint_types.mojo: JNT_HINGE = 3, JNT_SLIDE = 2.

Fields that use sentinel value -1.0 mean "use ModelDefaults".
Resolution happens at Joints.setup_model time.
"""
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim, barrier
from math import sqrt

from ..gpu.constants import (
    TPB,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    model_size_with_invweight,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TAU_LIMIT,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_QPOS0,
    model_joint_offset,
)
from gpu.host import HostBuffer
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_FREE, JointDef
from ..model.defaults_spec import ModelDefaults, _resolve_f64, _resolve_int
from random.philox import Random as PhiloxRandom

# Sentinel value for "use model default"
comptime _UNSET_F64: Float64 = -1.0


# =============================================================================
# JointSpec Trait
# =============================================================================


trait JointSpec(TrivialRegisterPassable):
    """Compile-time joint specification for physics3d model definitions.

    Properties match JointDef fields and GPU buffer layout.
    Fields with value -1.0 are "unset" and resolved from ModelDefaults.
    """

    comptime JNT_TYPE: Int  # JNT_HINGE, JNT_SLIDE, JNT_FREE
    comptime NQ: Int  # Dimension in qpos (1 for hinge/slide, 7 for free)
    comptime NV: Int  # Dimension in qvel (1 for hinge/slide, 6 for free)
    comptime BODY_IDX: Int  # Which body this joint is on
    comptime NUM_EXCLUDED_QPOS: Int  # Leading qpos elements to skip from obs

    # Joint anchor in parent frame
    comptime POS_X: Float64
    comptime POS_Y: Float64
    comptime POS_Z: Float64

    # Joint axis
    comptime AXIS_X: Float64
    comptime AXIS_Y: Float64
    comptime AXIS_Z: Float64

    # Limits and dynamics (-1.0 = use ModelDefaults for armature/damping/stiffness/frictionloss)
    comptime TAU_LIMIT: Float64  # Torque/force limit (gear ratio for actuated)
    comptime RANGE_MIN: Float64  # Position lower limit
    comptime RANGE_MAX: Float64  # Position upper limit
    comptime ARMATURE: Float64  # Rotor inertia (-1.0 = use default)
    comptime DAMPING: Float64  # Velocity-dependent force (-1.0 = use default)
    comptime STIFFNESS: Float64  # Position-dependent spring (-1.0 = use default)
    comptime SPRINGREF: Float64  # Spring reference position
    comptime FRICTIONLOSS: Float64  # Dry friction torque (-1.0 = use default)
    comptime INIT_QPOS: Float64  # Initial joint position (qpos0)

    # Free joint initial position (0.0 for hinge/slide)
    comptime INIT_POS_X: Float64
    comptime INIT_POS_Y: Float64
    comptime INIT_POS_Z: Float64

    # Observation/actuation flags (for generic env infrastructure)
    comptime EXCLUDE_OBS_QPOS: Bool  # Skip qpos from observation
    comptime EXCLUDE_OBS_QVEL: Bool  # Skip qvel from observation
    comptime IS_ACTUATED: Bool  # Has motor (for action mapping)
    comptime HAS_LIMITS: Bool  # Has meaningful position limits

    # Per-joint solref/solimp for limits (-1.0 = use model-level defaults)
    comptime SOLREF_LIMIT_0: Float64  # timeconst
    comptime SOLREF_LIMIT_1: Float64  # dampratio
    comptime SOLIMP_LIMIT_0: Float64  # dmin
    comptime SOLIMP_LIMIT_1: Float64  # dmax
    comptime SOLIMP_LIMIT_2: Float64  # width


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
    armature: Float64 = _UNSET_F64,
    damping: Float64 = _UNSET_F64,
    stiffness: Float64 = _UNSET_F64,
    springref: Float64 = 0.0,
    frictionloss: Float64 = _UNSET_F64,
    init_qpos: Float64 = 0.0,
    exclude_obs_qpos: Bool = False,
    exclude_obs_qvel: Bool = False,
    is_actuated: Bool = True,
    has_limits: Bool = True,
    solref_limit_0: Float64 = _UNSET_F64,
    solref_limit_1: Float64 = _UNSET_F64,
    solimp_limit_0: Float64 = _UNSET_F64,
    solimp_limit_1: Float64 = _UNSET_F64,
    solimp_limit_2: Float64 = _UNSET_F64,
](JointSpec):
    """Revolute (hinge) joint: 1 DOF rotation around axis.

    Default axis is Y (into-screen for 2D planar models like HalfCheetah).
    """

    comptime JNT_TYPE: Int = JNT_HINGE
    comptime NQ: Int = 1
    comptime NV: Int = 1
    comptime BODY_IDX: Int = Self.body_idx
    comptime NUM_EXCLUDED_QPOS: Int = 0
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
    comptime INIT_POS_X: Float64 = 0.0
    comptime INIT_POS_Y: Float64 = 0.0
    comptime INIT_POS_Z: Float64 = 0.0
    comptime EXCLUDE_OBS_QPOS: Bool = Self.exclude_obs_qpos
    comptime EXCLUDE_OBS_QVEL: Bool = Self.exclude_obs_qvel
    comptime IS_ACTUATED: Bool = Self.is_actuated
    comptime HAS_LIMITS: Bool = Self.has_limits
    comptime SOLREF_LIMIT_0: Float64 = Self.solref_limit_0
    comptime SOLREF_LIMIT_1: Float64 = Self.solref_limit_1
    comptime SOLIMP_LIMIT_0: Float64 = Self.solimp_limit_0
    comptime SOLIMP_LIMIT_1: Float64 = Self.solimp_limit_1
    comptime SOLIMP_LIMIT_2: Float64 = Self.solimp_limit_2


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
    armature: Float64 = _UNSET_F64,
    damping: Float64 = _UNSET_F64,
    stiffness: Float64 = _UNSET_F64,
    springref: Float64 = 0.0,
    frictionloss: Float64 = _UNSET_F64,
    init_qpos: Float64 = 0.0,
    exclude_obs_qpos: Bool = False,
    exclude_obs_qvel: Bool = False,
    is_actuated: Bool = False,
    has_limits: Bool = False,
    solref_limit_0: Float64 = _UNSET_F64,
    solref_limit_1: Float64 = _UNSET_F64,
    solimp_limit_0: Float64 = _UNSET_F64,
    solimp_limit_1: Float64 = _UNSET_F64,
    solimp_limit_2: Float64 = _UNSET_F64,
](JointSpec):
    """Prismatic (slide) joint: 1 DOF translation along axis.

    Default axis is X. Typically used for root translation DOFs.
    """

    comptime JNT_TYPE: Int = JNT_SLIDE
    comptime NQ: Int = 1
    comptime NV: Int = 1
    comptime BODY_IDX: Int = Self.body_idx
    comptime NUM_EXCLUDED_QPOS: Int = 0
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
    comptime INIT_POS_X: Float64 = 0.0
    comptime INIT_POS_Y: Float64 = 0.0
    comptime INIT_POS_Z: Float64 = 0.0
    comptime SOLREF_LIMIT_0: Float64 = Self.solref_limit_0
    comptime SOLREF_LIMIT_1: Float64 = Self.solref_limit_1
    comptime SOLIMP_LIMIT_0: Float64 = Self.solimp_limit_0
    comptime SOLIMP_LIMIT_1: Float64 = Self.solimp_limit_1
    comptime SOLIMP_LIMIT_2: Float64 = Self.solimp_limit_2


# =============================================================================
# FreeJoint
# =============================================================================


@fieldwise_init
struct FreeJoint[
    body_idx: Int,
    init_pos_x: Float64 = 0.0,
    init_pos_y: Float64 = 0.0,
    init_pos_z: Float64 = 0.0,
    armature: Float64 = 0.0,
    damping: Float64 = 0.0,
    stiffness: Float64 = 0.0,
    frictionloss: Float64 = 0.0,
    exclude_obs_qpos: Bool = False,
    exclude_obs_qvel: Bool = False,
    is_actuated: Bool = False,
    num_excluded_qpos: Int = 0,
](JointSpec):
    """Free joint: 6 DOF (3 translation + 3 rotation).

    qpos: [x, y, z, quat_x, quat_y, quat_z, quat_w] (7 elements)
    qvel: [vx, vy, vz, wx, wy, wz] (6 elements)

    num_excluded_qpos: Number of leading qpos elements to skip from obs
    (e.g., 2 to exclude x,y for Ant, 0 to include all).
    """

    comptime JNT_TYPE: Int = JNT_FREE
    comptime NQ: Int = 7
    comptime NV: Int = 6
    comptime BODY_IDX: Int = Self.body_idx
    comptime POS_X: Float64 = 0.0
    comptime POS_Y: Float64 = 0.0
    comptime POS_Z: Float64 = 0.0
    comptime AXIS_X: Float64 = 0.0
    comptime AXIS_Y: Float64 = 0.0
    comptime AXIS_Z: Float64 = 1.0
    comptime TAU_LIMIT: Float64 = 0.0
    comptime RANGE_MIN: Float64 = -1e10
    comptime RANGE_MAX: Float64 = 1e10
    comptime ARMATURE: Float64 = Self.armature
    comptime DAMPING: Float64 = Self.damping
    comptime STIFFNESS: Float64 = Self.stiffness
    comptime SPRINGREF: Float64 = 0.0
    comptime FRICTIONLOSS: Float64 = Self.frictionloss
    comptime INIT_QPOS: Float64 = 0.0  # Not used; free joint uses INIT_POS_X/Y/Z
    comptime EXCLUDE_OBS_QPOS: Bool = Self.exclude_obs_qpos
    comptime EXCLUDE_OBS_QVEL: Bool = Self.exclude_obs_qvel
    comptime IS_ACTUATED: Bool = Self.is_actuated
    comptime HAS_LIMITS: Bool = False
    comptime SOLREF_LIMIT_0: Float64 = _UNSET_F64
    comptime SOLREF_LIMIT_1: Float64 = _UNSET_F64
    comptime SOLIMP_LIMIT_0: Float64 = _UNSET_F64
    comptime SOLIMP_LIMIT_1: Float64 = _UNSET_F64
    comptime SOLIMP_LIMIT_2: Float64 = _UNSET_F64

    # Free joint-specific fields
    comptime INIT_POS_X: Float64 = Self.init_pos_x
    comptime INIT_POS_Y: Float64 = Self.init_pos_y
    comptime INIT_POS_Z: Float64 = Self.init_pos_z
    comptime NUM_EXCLUDED_QPOS: Int = Self.num_excluded_qpos


trait JointsLike:
    """Trait for compile-time joint container types."""

    comptime N: Int
    comptime NQ: Int
    comptime NV: Int
    comptime OBS_DIM: Int
    comptime ACTION_DIM: Int

    @staticmethod
    fn write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        Defaults: ModelDefaultsLike,
    ](buffer: HostBuffer[DTYPE]):
        ...

    # CPU methods
    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int,
        Defaults: ModelDefaultsLike,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            Self.N,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
        ]
    ):
        ...

    @staticmethod
    fn reset_data[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        ...

    @staticmethod
    fn extract_obs[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS],
        mut obs: List[Scalar[DTYPE]],
    ):
        ...

    @staticmethod
    fn enforce_limits[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        ...

    # GPU kernel launchers
    @staticmethod
    fn extract_obs_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[GDTYPE],
        mut obs_buf: DeviceBuffer[GDTYPE],
    ) raises:
        ...

    @staticmethod
    fn enforce_limits_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[GDTYPE]) raises:
        ...

    # GPU inline per-env methods
    @always_inline
    @staticmethod
    fn extract_obs_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        ...

    @always_inline
    @staticmethod
    fn reset_env_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        noise_scale: Scalar[GDTYPE],
        seed: Int,
    ):
        ...


@fieldwise_init
struct _EmptyJoints(JointsLike):
    comptime N: Int = 0
    comptime NQ: Int = 0
    comptime NV: Int = 0
    comptime OBS_DIM: Int = 0
    comptime ACTION_DIM: Int = 0

    @staticmethod
    fn write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        Defaults: ModelDefaultsLike,
    ](buffer: HostBuffer[DTYPE]):
        pass

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int,
        MAX_EQUALITY: Int,
        CONE_TYPE: Int,
        MAX_TENDON: Int,
        Defaults: ModelDefaultsLike,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            Self.N,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
        ]
    ):
        pass

    @staticmethod
    fn reset_data[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        pass

    @staticmethod
    fn extract_obs[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS],
        mut obs: List[Scalar[DTYPE]],
    ):
        pass

    @staticmethod
    fn enforce_limits[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        pass

    @staticmethod
    fn extract_obs_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[GDTYPE],
        mut obs_buf: DeviceBuffer[GDTYPE],
    ) raises:
        pass

    @staticmethod
    fn enforce_limits_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[GDTYPE]) raises:
        pass

    @always_inline
    @staticmethod
    fn extract_obs_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        pass

    @always_inline
    @staticmethod
    fn reset_env_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        noise_scale: Scalar[GDTYPE],
        seed: Int,
    ):
        pass


@fieldwise_init
struct Joints[*J: JointSpec](JointsLike):
    """Compile-time list of joint specifications.

    Provides N (joint count), sum helpers for total NQ/NV, and offset helpers
    for computing qpos/qvel addresses of each joint.
    """

    comptime joint_types = Variadic.types[T=JointSpec, *Self.J]
    comptime N: Int = Variadic.size(Self.joint_types)

    # Explicit trait members (required by JointsLike)
    comptime NQ: Int = Self._sum_nq()
    comptime NV: Int = Self._sum_nv()
    comptime OBS_DIM: Int = Self._obs_dim()
    comptime ACTION_DIM: Int = Self._action_dim()

    @staticmethod
    fn _sum_nq() -> Int:
        """Sum NQ across all joints (total qpos dimension)."""
        var total = 0

        @parameter
        for i in range(Self.N):
            total += Self.joint_types[i].NQ
        return total

    @staticmethod
    fn _sum_nv() -> Int:
        """Sum NV across all joints (total qvel dimension)."""
        var total = 0

        @parameter
        for i in range(Self.N):
            total += Self.joint_types[i].NV
        return total

    @staticmethod
    fn _qpos_offset[idx: Int]() -> Int:
        """Compute qpos address for joint idx (sum of NQ for joints 0..idx-1).
        """
        var total = 0

        @parameter
        for j in range(idx):
            total += Self.joint_types[j].NQ
        return total

    @staticmethod
    fn _qvel_offset[idx: Int]() -> Int:
        """Compute qvel/dof address for joint idx (sum of NV for joints 0..idx-1).
        """
        var total = 0

        @parameter
        for j in range(idx):
            total += Self.joint_types[j].NV
        return total

    @staticmethod
    fn reset_data[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        """Reset qpos to initial joint positions (qpos0), zero qvel/qacc/qfrc.

        Sets each joint's qpos to its INIT_QPOS value and zeros all velocity,
        acceleration, and force arrays. Does NOT run forward kinematics.
        """

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]
            comptime offset = Self._qpos_offset[i]()

            @parameter
            if J.JNT_TYPE == JNT_FREE:
                # Free joint: qpos = [x, y, z, qx, qy, qz, qw]
                data.qpos[offset + 0] = Scalar[DTYPE](J.INIT_POS_X)
                data.qpos[offset + 1] = Scalar[DTYPE](J.INIT_POS_Y)
                data.qpos[offset + 2] = Scalar[DTYPE](J.INIT_POS_Z)
                data.qpos[offset + 3] = Scalar[DTYPE](0)  # qx
                data.qpos[offset + 4] = Scalar[DTYPE](0)  # qy
                data.qpos[offset + 5] = Scalar[DTYPE](0)  # qz
                data.qpos[offset + 6] = Scalar[DTYPE](1)  # qw (identity)
            else:
                data.qpos[offset] = Scalar[DTYPE](J.INIT_QPOS)
        for i in range(NV):
            data.qvel[i] = Scalar[DTYPE](0)
            data.qacc[i] = Scalar[DTYPE](0)
            data.qfrc[i] = Scalar[DTYPE](0)

    # =========================================================================
    # Dimension Helpers (observation / action)
    # =========================================================================

    @staticmethod
    fn _obs_qpos_dim() -> Int:
        """Count of qpos elements included in observation.

        For joints with NUM_EXCLUDED_QPOS > 0 (e.g., FreeJoint excluding x,y),
        only (NQ - NUM_EXCLUDED_QPOS) elements are included.
        """
        var total = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                total += J.NQ - J.NUM_EXCLUDED_QPOS
        return total

    @staticmethod
    fn _obs_qvel_dim() -> Int:
        """Count of qvel elements included in observation."""
        var total = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QVEL:
                total += J.NV
        return total

    @staticmethod
    fn _obs_dim() -> Int:
        """Total observation dimension (included qpos + included qvel)."""
        return Self._obs_qpos_dim() + Self._obs_qvel_dim()

    @staticmethod
    fn _action_dim() -> Int:
        """Count of actuated DOFs (joints with IS_ACTUATED=True)."""
        var total = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.IS_ACTUATED:
                total += J.NV
        return total

    # =========================================================================
    # CPU Operations
    # =========================================================================

    @staticmethod
    fn extract_obs[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](
        data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS],
        mut obs: List[Scalar[DTYPE]],
    ):
        """Extract observation from physics data into a list.

        Appends included qpos then included qvel to the obs list.
        """

        # Included qpos (skip first NUM_EXCLUDED_QPOS elements per joint)
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                comptime offset = Self._qpos_offset[i]()

                @parameter
                for k in range(J.NUM_EXCLUDED_QPOS, J.NQ):
                    obs.append(data.qpos[offset + k])

        # Included qvel
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QVEL:
                comptime offset = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NV):
                    obs.append(data.qvel[offset + k])

    @staticmethod
    fn apply_actions[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](
        mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS],
        actions: List[Float64],
    ):
        """Apply normalized actions to actuated joints.

        Clamps each action to [-1, 1], scales by TAU_LIMIT, writes to qfrc.
        actions[k] corresponds to the k-th actuated joint in declaration order.
        """
        var act_idx = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.IS_ACTUATED:
                comptime offset = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NV):
                    var a = actions[act_idx] if act_idx < len(actions) else 0.0
                    # Clamp to [-1, 1]
                    if a > 1.0:
                        a = 1.0
                    elif a < -1.0:
                        a = -1.0
                    data.qfrc[offset + k] = Scalar[DTYPE](a * J.TAU_LIMIT)
                    act_idx += 1

    @staticmethod
    fn enforce_limits[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
    ](mut data: Data[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS]):
        """Enforce joint position limits. Zeros velocity at limits."""

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.HAS_LIMITS:
                comptime qp_off = Self._qpos_offset[i]()
                comptime qv_off = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NQ):
                    var qpos = data.qpos[qp_off + k]
                    var qvel = data.qvel[qv_off + k]
                    if qpos < Scalar[DTYPE](J.RANGE_MIN):
                        data.qpos[qp_off + k] = Scalar[DTYPE](J.RANGE_MIN)
                        if qvel < Scalar[DTYPE](0):
                            data.qvel[qv_off + k] = Scalar[DTYPE](0)
                    elif qpos > Scalar[DTYPE](J.RANGE_MAX):
                        data.qpos[qp_off + k] = Scalar[DTYPE](J.RANGE_MAX)
                        if qvel > Scalar[DTYPE](0):
                            data.qvel[qv_off + k] = Scalar[DTYPE](0)

    # =========================================================================
    # GPU Operations — inline per-env (called from inside kernels)
    # =========================================================================

    @always_inline
    @staticmethod
    fn extract_obs_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Extract observation for a single env on GPU."""
        comptime NQ_VAL = Self._sum_nq()
        comptime NV_VAL = Self._sum_nv()
        comptime QPOS_OFF = qpos_offset[NQ_VAL, NV_VAL]()
        comptime QVEL_OFF = qvel_offset[NQ_VAL, NV_VAL]()

        var obs_idx = 0

        # Included qpos (skip first NUM_EXCLUDED_QPOS elements per joint)
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                comptime offset = Self._qpos_offset[i]()

                @parameter
                for k in range(J.NUM_EXCLUDED_QPOS, J.NQ):
                    obs[env, obs_idx] = states[env, QPOS_OFF + offset + k]
                    obs_idx += 1

        # Included qvel
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QVEL:
                comptime offset = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NV):
                    obs[env, obs_idx] = states[env, QVEL_OFF + offset + k]
                    obs_idx += 1

    @always_inline
    @staticmethod
    fn apply_actions_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Apply actions for a single env on GPU."""
        comptime NQ_VAL = Self._sum_nq()
        comptime NV_VAL = Self._sum_nv()
        comptime QFRC_OFF = qfrc_offset[NQ_VAL, NV_VAL]()

        var act_idx = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.IS_ACTUATED:
                comptime offset = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NV):
                    var a = actions[env, act_idx]
                    if a > Scalar[GDTYPE](1.0):
                        a = Scalar[GDTYPE](1.0)
                    elif a < Scalar[GDTYPE](-1.0):
                        a = Scalar[GDTYPE](-1.0)
                    states[env, QFRC_OFF + offset + k] = a * Scalar[GDTYPE](
                        J.TAU_LIMIT
                    )
                    act_idx += 1

    @always_inline
    @staticmethod
    fn enforce_limits_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Enforce joint limits for a single env on GPU."""
        comptime NQ_VAL = Self._sum_nq()
        comptime NV_VAL = Self._sum_nv()
        comptime QPOS_OFF = qpos_offset[NQ_VAL, NV_VAL]()
        comptime QVEL_OFF = qvel_offset[NQ_VAL, NV_VAL]()

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if J.HAS_LIMITS:
                comptime qp_off = Self._qpos_offset[i]()
                comptime qv_off = Self._qvel_offset[i]()

                @parameter
                for k in range(J.NQ):
                    var qpos = states[env, QPOS_OFF + qp_off + k]
                    if qpos < Scalar[GDTYPE](J.RANGE_MIN):
                        states[env, QPOS_OFF + qp_off + k] = Scalar[GDTYPE](
                            J.RANGE_MIN
                        )
                        var qvel = states[env, QVEL_OFF + qv_off + k]
                        if qvel < Scalar[GDTYPE](0):
                            states[env, QVEL_OFF + qv_off + k] = Scalar[GDTYPE](
                                0
                            )
                    elif qpos > Scalar[GDTYPE](J.RANGE_MAX):
                        states[env, QPOS_OFF + qp_off + k] = Scalar[GDTYPE](
                            J.RANGE_MAX
                        )
                        var qvel = states[env, QVEL_OFF + qv_off + k]
                        if qvel > Scalar[GDTYPE](0):
                            states[env, QVEL_OFF + qv_off + k] = Scalar[GDTYPE](
                                0
                            )

    @always_inline
    @staticmethod
    fn reset_env_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        noise_scale: Scalar[GDTYPE],
        seed: Int,
    ):
        """Reset a single env on GPU with random noise.

        Sets qpos = INIT_QPOS + noise, qvel = noise, qacc/qfrc = 0.
        """
        comptime NQ_VAL = Self._sum_nq()
        comptime NV_VAL = Self._sum_nv()
        comptime QPOS_OFF = qpos_offset[NQ_VAL, NV_VAL]()
        comptime QVEL_OFF = qvel_offset[NQ_VAL, NV_VAL]()
        comptime QACC_OFF = qacc_offset[NQ_VAL, NV_VAL]()
        comptime QFRC_OFF = qfrc_offset[NQ_VAL, NV_VAL]()

        # Create RNG with unique seed per environment
        var rng = PhiloxRandom(seed=seed * 2654435761 + env * 12345, offset=0)

        # Generate noise batches (4 values at a time from Philox)
        # We need NQ values for qpos + NV values for qvel
        # Generate enough batches to cover all values
        comptime TOTAL_VALS = NQ_VAL + NV_VAL
        comptime NUM_BATCHES = (TOTAL_VALS + 3) // 4

        var rand_vals = InlineArray[Scalar[DType.float32], NUM_BATCHES * 4](
            fill=Scalar[DType.float32](0)
        )
        for b in range(NUM_BATCHES):
            var batch = rng.step_uniform()
            rand_vals[b * 4 + 0] = batch[0]
            rand_vals[b * 4 + 1] = batch[1]
            rand_vals[b * 4 + 2] = batch[2]
            rand_vals[b * 4 + 3] = batch[3]

        # Reset qpos with noise
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]
            comptime offset = Self._qpos_offset[i]()

            @parameter
            if J.JNT_TYPE == JNT_FREE:
                # Free joint: init position + identity quaternion (no noise for now)
                states[env, QPOS_OFF + offset + 0] = Scalar[GDTYPE](
                    J.INIT_POS_X
                )
                states[env, QPOS_OFF + offset + 1] = Scalar[GDTYPE](
                    J.INIT_POS_Y
                )
                states[env, QPOS_OFF + offset + 2] = Scalar[GDTYPE](
                    J.INIT_POS_Z
                )
                states[env, QPOS_OFF + offset + 3] = Scalar[GDTYPE](0)  # qx
                states[env, QPOS_OFF + offset + 4] = Scalar[GDTYPE](0)  # qy
                states[env, QPOS_OFF + offset + 5] = Scalar[GDTYPE](0)  # qz
                states[env, QPOS_OFF + offset + 6] = Scalar[GDTYPE](1)  # qw
            else:

                @parameter
                for k in range(J.NQ):
                    var noise = (
                        Scalar[GDTYPE](rand_vals[offset + k] * 2.0 - 1.0)
                        * noise_scale
                    )
                    states[env, QPOS_OFF + offset + k] = (
                        Scalar[GDTYPE](J.INIT_QPOS) + noise
                    )

        # Reset qvel with noise
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]
            comptime offset = Self._qvel_offset[i]()

            @parameter
            for k in range(J.NV):
                var noise = (
                    Scalar[GDTYPE](rand_vals[NQ_VAL + offset + k] * 2.0 - 1.0)
                    * noise_scale
                )
                states[env, QVEL_OFF + offset + k] = noise

        # Reset qacc, qfrc to zero
        for i in range(NV_VAL):
            states[env, QACC_OFF + i] = Scalar[GDTYPE](0.0)
            states[env, QFRC_OFF + i] = Scalar[GDTYPE](0.0)

    # =========================================================================
    # GPU Operations — kernel launchers
    # =========================================================================

    @staticmethod
    fn extract_obs_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[GDTYPE],
        mut obs_buf: DeviceBuffer[GDTYPE],
    ) raises:
        """Launch kernel to extract observations for all envs."""
        var states = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var obs = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn kernel(
            states: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            obs: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, OBS_DIM),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.extract_obs_gpu[GDTYPE, BATCH_SIZE, STATE_SIZE, OBS_DIM](
                states, obs, env
            )

        ctx.enqueue_function[kernel, kernel](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn apply_actions_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[GDTYPE],
        actions_buf: DeviceBuffer[GDTYPE],
    ) raises:
        """Launch kernel to apply actions for all envs."""
        var states = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn kernel(
            states: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, ACTION_DIM),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.apply_actions_gpu[GDTYPE, BATCH_SIZE, STATE_SIZE, ACTION_DIM](
                states, actions, env
            )

        ctx.enqueue_function[kernel, kernel](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn enforce_limits_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[GDTYPE],) raises:
        """Launch kernel to enforce joint limits for all envs."""
        var states = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn kernel(
            states: LayoutTensor[
                GDTYPE,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.enforce_limits_gpu[GDTYPE, BATCH_SIZE, STATE_SIZE](states, env)

        ctx.enqueue_function[kernel, kernel](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Model Setup
    # =========================================================================

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        Defaults: ModelDefaultsLike = ModelDefaults[],
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            Self.N,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
        ]
    ):
        """Populate model joints from compile-time JointSpec list.

        Resolves sentinel values (-1.0) from ModelDefaults.
        Also populates per-joint solref/solimp limit arrays.
        """

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            # Resolve dynamics fields from defaults
            comptime arm = _resolve_f64[J.ARMATURE, Defaults.JOINT_ARMATURE]()
            comptime damp = _resolve_f64[J.DAMPING, Defaults.JOINT_DAMPING]()
            comptime stiff = _resolve_f64[
                J.STIFFNESS, Defaults.JOINT_STIFFNESS
            ]()
            comptime frloss = _resolve_f64[
                J.FRICTIONLOSS, Defaults.JOINT_FRICTIONLOSS
            ]()

            @parameter
            if J.JNT_TYPE == JNT_HINGE:
                _ = model.add_hinge_joint(
                    body_id=J.BODY_IDX,
                    pos=(
                        Scalar[DTYPE](J.POS_X),
                        Scalar[DTYPE](J.POS_Y),
                        Scalar[DTYPE](J.POS_Z),
                    ),
                    axis=(
                        Scalar[DTYPE](J.AXIS_X),
                        Scalar[DTYPE](J.AXIS_Y),
                        Scalar[DTYPE](J.AXIS_Z),
                    ),
                    tau_limit=Scalar[DTYPE](J.TAU_LIMIT),
                    range_min=Scalar[DTYPE](J.RANGE_MIN),
                    range_max=Scalar[DTYPE](J.RANGE_MAX),
                    armature=Scalar[DTYPE](arm),
                    damping=Scalar[DTYPE](damp),
                    stiffness=Scalar[DTYPE](stiff),
                    springref=Scalar[DTYPE](J.SPRINGREF),
                    frictionloss=Scalar[DTYPE](frloss),
                )
            elif J.JNT_TYPE == JNT_SLIDE:
                _ = model.add_slide_joint(
                    body_id=J.BODY_IDX,
                    pos=(
                        Scalar[DTYPE](J.POS_X),
                        Scalar[DTYPE](J.POS_Y),
                        Scalar[DTYPE](J.POS_Z),
                    ),
                    axis=(
                        Scalar[DTYPE](J.AXIS_X),
                        Scalar[DTYPE](J.AXIS_Y),
                        Scalar[DTYPE](J.AXIS_Z),
                    ),
                    force_limit=Scalar[DTYPE](J.TAU_LIMIT),
                    range_min=Scalar[DTYPE](J.RANGE_MIN),
                    range_max=Scalar[DTYPE](J.RANGE_MAX),
                    armature=Scalar[DTYPE](arm),
                    damping=Scalar[DTYPE](damp),
                    stiffness=Scalar[DTYPE](stiff),
                    springref=Scalar[DTYPE](J.SPRINGREF),
                    frictionloss=Scalar[DTYPE](frloss),
                )
            elif J.JNT_TYPE == JNT_FREE:
                # Free joint: use JointDef.create_free() directly
                # Compute qpos/qvel addresses manually
                var qpos_adr = 0
                var dof_adr = 0
                for ji in range(model.num_joints):
                    qpos_adr += model.joints[ji].qpos_size()
                    dof_adr += model.joints[ji].qvel_size()

                var joint_idx = model.num_joints
                model.joints[joint_idx] = JointDef[DTYPE].create_free(
                    body_id=J.BODY_IDX,
                    qpos_adr=qpos_adr,
                    dof_adr=dof_adr,
                )
                # Set dynamics fields
                model.joints[joint_idx].armature = Scalar[DTYPE](arm)
                model.joints[joint_idx].damping = Scalar[DTYPE](damp)
                model.joints[joint_idx].stiffness = Scalar[DTYPE](stiff)
                model.joints[joint_idx].springref = Scalar[DTYPE](J.SPRINGREF)
                model.joints[joint_idx].frictionloss = Scalar[DTYPE](frloss)
                model.num_joints += 1

            # Per-joint solref/solimp for limits (resolved from defaults)
            model.joint_solref_limit[i * 2 + 0] = Scalar[DTYPE](
                _resolve_f64[J.SOLREF_LIMIT_0, Defaults.JOINT_SOLREF_LIMIT_0]()
            )
            model.joint_solref_limit[i * 2 + 1] = Scalar[DTYPE](
                _resolve_f64[J.SOLREF_LIMIT_1, Defaults.JOINT_SOLREF_LIMIT_1]()
            )
            model.joint_solimp_limit[i * 3 + 0] = Scalar[DTYPE](
                _resolve_f64[J.SOLIMP_LIMIT_0, Defaults.JOINT_SOLIMP_LIMIT_0]()
            )
            model.joint_solimp_limit[i * 3 + 1] = Scalar[DTYPE](
                _resolve_f64[J.SOLIMP_LIMIT_1, Defaults.JOINT_SOLIMP_LIMIT_1]()
            )
            model.joint_solimp_limit[i * 3 + 2] = Scalar[DTYPE](
                _resolve_f64[J.SOLIMP_LIMIT_2, Defaults.JOINT_SOLIMP_LIMIT_2]()
            )

            # Set qpos0 (MuJoCo ref / initial position)
            comptime qp_off = Self._qpos_offset[i]()

            @parameter
            if J.JNT_TYPE == JNT_FREE:
                model.qpos0[qp_off + 0] = Scalar[DTYPE](J.INIT_POS_X)
                model.qpos0[qp_off + 1] = Scalar[DTYPE](J.INIT_POS_Y)
                model.qpos0[qp_off + 2] = Scalar[DTYPE](J.INIT_POS_Z)
                model.qpos0[qp_off + 3] = Scalar[DTYPE](0)  # qx
                model.qpos0[qp_off + 4] = Scalar[DTYPE](0)  # qy
                model.qpos0[qp_off + 5] = Scalar[DTYPE](0)  # qz
                model.qpos0[qp_off + 6] = Scalar[DTYPE](1)  # qw
            else:
                model.qpos0[qp_off] = Scalar[DTYPE](J.INIT_QPOS)

    @staticmethod
    fn write_to_buffer[
        DTYPE: DType,
        NBODY: Int,
        Defaults: ModelDefaultsLike = ModelDefaults[],
    ](buffer: HostBuffer[DTYPE]):
        """Write joint data directly to GPU HostBuffer (no Model struct).

        Computes qpos_adr/dof_adr incrementally from joint NQ/NV.
        Resolves sentinel values (-1.0) from Defaults.
        """
        var qpos_adr = 0
        var dof_adr = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]
            var off = model_joint_offset[NBODY](i)

            # Resolve dynamics fields from defaults
            comptime arm = _resolve_f64[J.ARMATURE, Defaults.JOINT_ARMATURE]()
            comptime damp = _resolve_f64[J.DAMPING, Defaults.JOINT_DAMPING]()
            comptime stiff = _resolve_f64[
                J.STIFFNESS, Defaults.JOINT_STIFFNESS
            ]()
            comptime frloss = _resolve_f64[
                J.FRICTIONLOSS, Defaults.JOINT_FRICTIONLOSS
            ]()

            buffer[off + JOINT_IDX_TYPE] = Scalar[DTYPE](J.JNT_TYPE)
            buffer[off + JOINT_IDX_BODY_ID] = Scalar[DTYPE](J.BODY_IDX)
            buffer[off + JOINT_IDX_QPOS_ADR] = Scalar[DTYPE](qpos_adr)
            buffer[off + JOINT_IDX_DOF_ADR] = Scalar[DTYPE](dof_adr)
            buffer[off + JOINT_IDX_POS_X] = Scalar[DTYPE](J.POS_X)
            buffer[off + JOINT_IDX_POS_Y] = Scalar[DTYPE](J.POS_Y)
            buffer[off + JOINT_IDX_POS_Z] = Scalar[DTYPE](J.POS_Z)
            buffer[off + JOINT_IDX_AXIS_X] = Scalar[DTYPE](J.AXIS_X)
            buffer[off + JOINT_IDX_AXIS_Y] = Scalar[DTYPE](J.AXIS_Y)
            buffer[off + JOINT_IDX_AXIS_Z] = Scalar[DTYPE](J.AXIS_Z)
            buffer[off + JOINT_IDX_TAU_LIMIT] = Scalar[DTYPE](J.TAU_LIMIT)
            buffer[off + JOINT_IDX_RANGE_MIN] = Scalar[DTYPE](J.RANGE_MIN)
            buffer[off + JOINT_IDX_RANGE_MAX] = Scalar[DTYPE](J.RANGE_MAX)
            buffer[off + JOINT_IDX_ARMATURE] = Scalar[DTYPE](arm)
            buffer[off + JOINT_IDX_DAMPING] = Scalar[DTYPE](damp)
            buffer[off + JOINT_IDX_STIFFNESS] = Scalar[DTYPE](stiff)
            buffer[off + JOINT_IDX_SPRINGREF] = Scalar[DTYPE](J.SPRINGREF)
            buffer[off + JOINT_IDX_FRICTIONLOSS] = Scalar[DTYPE](frloss)
            buffer[off + JOINT_IDX_QPOS0] = Scalar[DTYPE](J.INIT_QPOS)

            # Per-joint solref/solimp for limits
            buffer[off + JOINT_IDX_SOLREF_LIMIT_0] = Scalar[DTYPE](
                _resolve_f64[
                    J.SOLREF_LIMIT_0, Defaults.JOINT_SOLREF_LIMIT_0
                ]()
            )
            buffer[off + JOINT_IDX_SOLREF_LIMIT_1] = Scalar[DTYPE](
                _resolve_f64[
                    J.SOLREF_LIMIT_1, Defaults.JOINT_SOLREF_LIMIT_1
                ]()
            )
            buffer[off + JOINT_IDX_SOLIMP_LIMIT_0] = Scalar[DTYPE](
                _resolve_f64[
                    J.SOLIMP_LIMIT_0, Defaults.JOINT_SOLIMP_LIMIT_0
                ]()
            )
            buffer[off + JOINT_IDX_SOLIMP_LIMIT_1] = Scalar[DTYPE](
                _resolve_f64[
                    J.SOLIMP_LIMIT_1, Defaults.JOINT_SOLIMP_LIMIT_1
                ]()
            )
            buffer[off + JOINT_IDX_SOLIMP_LIMIT_2] = Scalar[DTYPE](
                _resolve_f64[
                    J.SOLIMP_LIMIT_2, Defaults.JOINT_SOLIMP_LIMIT_2
                ]()
            )

            # Advance addresses
            qpos_adr += J.NQ
            dof_adr += J.NV
