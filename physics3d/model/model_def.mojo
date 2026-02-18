"""ModelDef compositor for compile-time model definitions.

Composes Bodies and Joints into a ModelDef with auto-computed dimensions.
Uses Variadic.types + @parameter for to iterate at compile time, following
the same pattern as Sequential[*LAYERS: Model] in deep_rl/model/sequential.mojo.

Note: Bodies and Joints are standalone variadic containers. ModelDef takes
concrete Int parameters because Mojo cannot resolve variadic type packs
through multiple levels of nesting (accessing ModelDef.NQ would fail with
"unbound parameter" if ModelDef contained Bodies/Joints directly).

Usage:
    comptime HalfCheetahBodies = Bodies[Torso, BThigh, ...]
    comptime HalfCheetahJoints = Joints[RootX, RootZ, ...]
    comptime HalfCheetahModel = ModelDef[
        HalfCheetahBodies.N,
        HalfCheetahJoints.N,
        HalfCheetahJoints._sum_nq(),
        HalfCheetahJoints._sum_nv(),
    ]
"""

from collections import InlineArray
from std.builtin.variadics import Variadic
from random.philox import Random as PhiloxRandom

from .body_spec import BodySpec
from .joint_spec import JointSpec
from .geom_spec import GeomSpec
from .equality_spec import EqualitySpec
from .camera_spec import CameraSpec
from .light_spec import LightSpec
from .texture_spec import TextureSpec
from .material_spec import MaterialSpec
from .actuator_spec import (
    ActuatorSpec,
    DYN_NONE,
    DYN_INTEGRATOR,
    DYN_FILTER,
    DYN_FILTEREXACT,
    GAIN_FIXED,
    GAIN_AFFINE,
    BIAS_NONE,
    BIAS_AFFINE,
)
from ..types import (
    Model,
    Data,
    ActuatorDef,
    EqualityConstraintDef,
    EQ_CONNECT,
    EQ_WELD,
    ConeType,
)
from ..joint_types import JNT_HINGE, JNT_SLIDE
from math import sqrt
from ..constants import GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX, GEOM_PLANE

# GPU imports
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from ..gpu.constants import (
    TPB,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    model_size_with_invweight,
)
from ..gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_invweight0_to_buffer,
)
from ..kinematics.forward_kinematics import forward_kinematics
from ..dynamics.mass_matrix import compute_body_invweight0
from .inertia_from_geom import geom_volume, compute_inertia_from_geoms
from gpu.host import HostBuffer


# =============================================================================
# Sentinel value for "use model default"
# =============================================================================

# Float64 fields that are always non-negative use -1.0 as "unset"
comptime UNSET_F64: Float64 = -1.0
# Int fields that are always non-negative use -1 as "unset"
comptime UNSET_INT: Int = -1


fn _resolve_f64[val: Float64, default: Float64]() -> Float64:
    """Resolve a compile-time Float64: use val if set (>= 0), else default."""

    @parameter
    if val >= 0.0:
        return val
    else:
        return default


fn _resolve_int[val: Int, default: Int]() -> Int:
    """Resolve a compile-time Int: use val if set (>= 0), else default."""

    @parameter
    if val >= 0:
        return val
    else:
        return default


# =============================================================================
# ModelDefaults — MuJoCo-style <default> block
# =============================================================================


trait ModelDefaultsLike(TrivialRegisterPassable):
    """Trait for compile-time model defaults (MuJoCo-style <default> + <option>).

    Allows different specializations of ModelDefaults to be passed as
    type parameters to setup_model functions.
    """

    comptime GEOM_FRICTION: Float64
    comptime GEOM_FRICTION_SPIN: Float64
    comptime GEOM_FRICTION_ROLL: Float64
    comptime GEOM_CONDIM: Int
    comptime GEOM_CONTYPE: Int
    comptime GEOM_CONAFFINITY: Int
    comptime GEOM_SOLREF_0: Float64
    comptime GEOM_SOLREF_1: Float64
    comptime GEOM_SOLIMP_0: Float64
    comptime GEOM_SOLIMP_1: Float64
    comptime GEOM_SOLIMP_2: Float64
    comptime GEOM_MARGIN: Float64
    comptime JOINT_ARMATURE: Float64
    comptime JOINT_DAMPING: Float64
    comptime JOINT_STIFFNESS: Float64
    comptime JOINT_FRICTIONLOSS: Float64
    comptime JOINT_SOLREF_LIMIT_0: Float64
    comptime JOINT_SOLREF_LIMIT_1: Float64
    comptime JOINT_SOLIMP_LIMIT_0: Float64
    comptime JOINT_SOLIMP_LIMIT_1: Float64
    comptime JOINT_SOLIMP_LIMIT_2: Float64
    comptime IMPRATIO: Float64
    # Geom density default (kg/m³)
    comptime GEOM_DENSITY: Float64
    # MuJoCo <compiler> inertiafromgeom
    comptime INERTIAFROMGEOM: Bool
    # MuJoCo <option> block
    comptime GRAVITY_X: Float64
    comptime GRAVITY_Y: Float64
    comptime GRAVITY_Z: Float64
    comptime TIMESTEP: Float64
    # MuJoCo <compiler> block
    comptime SETTOTALMASS: Float64


@fieldwise_init
struct ModelDefaults[
    # Geom defaults (MuJoCo <default><geom .../>)
    geom_friction: Float64 = 0.5,
    geom_friction_spin: Float64 = 0.005,
    geom_friction_roll: Float64 = 0.0001,
    geom_condim: Int = 3,
    geom_contype: Int = 1,
    geom_conaffinity: Int = 1,
    geom_solref_0: Float64 = 0.02,
    geom_solref_1: Float64 = 1.0,
    geom_solimp_0: Float64 = 0.0,
    geom_solimp_1: Float64 = 0.8,
    geom_solimp_2: Float64 = 0.01,
    geom_margin: Float64 = 0.0,
    # Joint defaults (MuJoCo <default><joint .../>)
    joint_armature: Float64 = 0.1,
    joint_damping: Float64 = 0.0,
    joint_stiffness: Float64 = 0.0,
    joint_frictionloss: Float64 = 0.0,
    joint_solref_limit_0: Float64 = 0.02,
    joint_solref_limit_1: Float64 = 1.0,
    joint_solimp_limit_0: Float64 = 0.0,
    joint_solimp_limit_1: Float64 = 0.8,
    joint_solimp_limit_2: Float64 = 0.03,
    # Motor defaults (MuJoCo <default><motor .../>)
    motor_ctrl_min: Float64 = -1.0,
    motor_ctrl_max: Float64 = 1.0,
    # Model-level (MuJoCo <option>)
    impratio: Float64 = 1.0,
    # Geom density default (MuJoCo default = 1000 kg/m³)
    geom_density: Float64 = 1000.0,
    # MuJoCo <compiler> inertiafromgeom (default True, matching MuJoCo compiler default)
    inertiafromgeom: Bool = True,
    gravity_x: Float64 = 0.0,
    gravity_y: Float64 = 0.0,
    gravity_z: Float64 = -9.81,
    timestep: Float64 = 0.01,
    # Compiler directive (MuJoCo <compiler>)
    settotalmass: Float64 = -1.0,
](ModelDefaultsLike):
    """MuJoCo-style model defaults block.

    Components that don't specify a value (sentinel = -1.0/-1) inherit
    from these defaults. Resolution happens at Geoms/Joints.setup_model time.

    Default values match MuJoCo's built-in defaults for geom/joint elements.
    """

    # Explicit trait member mapping (Mojo struct params don't auto-satisfy traits)
    comptime GEOM_FRICTION: Float64 = Self.geom_friction
    comptime GEOM_FRICTION_SPIN: Float64 = Self.geom_friction_spin
    comptime GEOM_FRICTION_ROLL: Float64 = Self.geom_friction_roll
    comptime GEOM_CONDIM: Int = Self.geom_condim
    comptime GEOM_CONTYPE: Int = Self.geom_contype
    comptime GEOM_CONAFFINITY: Int = Self.geom_conaffinity
    comptime GEOM_SOLREF_0: Float64 = Self.geom_solref_0
    comptime GEOM_SOLREF_1: Float64 = Self.geom_solref_1
    comptime GEOM_SOLIMP_0: Float64 = Self.geom_solimp_0
    comptime GEOM_SOLIMP_1: Float64 = Self.geom_solimp_1
    comptime GEOM_SOLIMP_2: Float64 = Self.geom_solimp_2
    comptime GEOM_MARGIN: Float64 = Self.geom_margin
    comptime JOINT_ARMATURE: Float64 = Self.joint_armature
    comptime JOINT_DAMPING: Float64 = Self.joint_damping
    comptime JOINT_STIFFNESS: Float64 = Self.joint_stiffness
    comptime JOINT_FRICTIONLOSS: Float64 = Self.joint_frictionloss
    comptime JOINT_SOLREF_LIMIT_0: Float64 = Self.joint_solref_limit_0
    comptime JOINT_SOLREF_LIMIT_1: Float64 = Self.joint_solref_limit_1
    comptime JOINT_SOLIMP_LIMIT_0: Float64 = Self.joint_solimp_limit_0
    comptime JOINT_SOLIMP_LIMIT_1: Float64 = Self.joint_solimp_limit_1
    comptime JOINT_SOLIMP_LIMIT_2: Float64 = Self.joint_solimp_limit_2
    comptime IMPRATIO: Float64 = Self.impratio
    comptime GEOM_DENSITY: Float64 = Self.geom_density
    comptime INERTIAFROMGEOM: Bool = Self.inertiafromgeom
    comptime GRAVITY_X: Float64 = Self.gravity_x
    comptime GRAVITY_Y: Float64 = Self.gravity_y
    comptime GRAVITY_Z: Float64 = Self.gravity_z
    comptime TIMESTEP: Float64 = Self.timestep
    comptime SETTOTALMASS: Float64 = Self.settotalmass


# =============================================================================
# Bodies — variadic body list
# =============================================================================


@fieldwise_init
struct Bodies[*B: BodySpec]:
    """Compile-time list of body specifications.

    Provides N (body count) and type-level access to each body via body_types[i].
    """

    comptime body_types = Variadic.types[T=BodySpec, *Self.B]
    comptime N: Int = Variadic.size(Self.body_types)

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            Self.N + 1,  # +1 for worldbody at index 0
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
        ]
    ):
        """Populate model body properties from compile-time BodySpec list.

        Iterates over all body specs and sets mass, inertia, geometry, parent,
        local frame, and collision filtering on the model. Body indices start
        at 1 (worldbody at index 0 is initialized by Model.__init__).
        """

        @parameter
        for i in range(Self.N):
            comptime B = Self.body_types[i]
            # Body index i+1: worldbody is at index 0 (reserved)
            comptime body_idx = i + 1

            # Mass, inertia
            model.set_body(
                body_idx,
                name=B.NAME,
                mass=Scalar[DTYPE](B.MASS),
                inertia=(
                    Scalar[DTYPE](B.ixx()),
                    Scalar[DTYPE](B.iyy()),
                    Scalar[DTYPE](B.izz()),
                ),
            )

            # Kinematic tree
            model.set_body_parent(body_idx, B.PARENT)

            # Local frame in parent
            model.set_body_local_frame(
                body_idx,
                pos=(
                    Scalar[DTYPE](B.POS_X),
                    Scalar[DTYPE](B.POS_Y),
                    Scalar[DTYPE](B.POS_Z),
                ),
                quat=(
                    Scalar[DTYPE](B.QUAT_X),
                    Scalar[DTYPE](B.QUAT_Y),
                    Scalar[DTYPE](B.QUAT_Z),
                    Scalar[DTYPE](B.QUAT_W),
                ),
            )

            # CoM offset and inertia frame
            model.set_body_ipos_iquat(
                body_idx,
                ipos=(
                    Scalar[DTYPE](B.IPOS_X),
                    Scalar[DTYPE](B.IPOS_Y),
                    Scalar[DTYPE](B.IPOS_Z),
                ),
                iquat=(
                    Scalar[DTYPE](B.IQUAT_X),
                    Scalar[DTYPE](B.IQUAT_Y),
                    Scalar[DTYPE](B.IQUAT_Z),
                    Scalar[DTYPE](B.IQUAT_W),
                ),
            )


# =============================================================================
# Joints — variadic joint list with sum helpers
# =============================================================================


@fieldwise_init
struct Joints[*J: JointSpec]:
    """Compile-time list of joint specifications.

    Provides N (joint count), sum helpers for total NQ/NV, and offset helpers
    for computing qpos/qvel addresses of each joint.
    """

    comptime joint_types = Variadic.types[T=JointSpec, *Self.J]
    comptime N: Int = Variadic.size(Self.joint_types)

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
        """Count of qpos elements included in observation."""
        var total = 0

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                total += J.NQ
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

        # Included qpos
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                comptime offset = Self._qpos_offset[i]()

                @parameter
                for k in range(J.NQ):
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

        # Included qpos
        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

            @parameter
            if not J.EXCLUDE_OBS_QPOS:
                comptime offset = Self._qpos_offset[i]()

                @parameter
                for k in range(J.NQ):
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


# =============================================================================
# ModelDef — full model compositor (concrete Int parameters)
# =============================================================================


# =============================================================================
# Equalities — variadic equality constraint list
# =============================================================================


@fieldwise_init
struct Equalities[*E: EqualitySpec]:
    """Compile-time list of equality constraint specifications.

    Provides N (constraint count) and _sum_rows() for total row count.
    """

    comptime eq_types = Variadic.types[T=EqualitySpec, *Self.E]
    comptime N: Int = Variadic.size(Self.eq_types)

    @staticmethod
    fn _sum_rows() -> Int:
        """Sum NUM_ROWS across all equality constraints (total constraint rows).
        """
        var total = 0

        @parameter
        for i in range(Self.N):
            total += Self.eq_types[i].NUM_ROWS
        return total

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            Self.N,
            MAX_EQUALITY,
            CONE_TYPE,
        ]
    ):
        """Populate model equality constraints from compile-time specs."""

        @parameter
        for i in range(Self.N):
            comptime E_item = Self.eq_types[i]

            model.equality_constraints[i] = EqualityConstraintDef[DTYPE](
                eq_type=E_item.EQ_TYPE,
                body_a=E_item.BODY_A,
                body_b=E_item.BODY_B,
                anchor_a_x=Scalar[DTYPE](E_item.ANCHOR_A_X),
                anchor_a_y=Scalar[DTYPE](E_item.ANCHOR_A_Y),
                anchor_a_z=Scalar[DTYPE](E_item.ANCHOR_A_Z),
                anchor_b_x=Scalar[DTYPE](E_item.ANCHOR_B_X),
                anchor_b_y=Scalar[DTYPE](E_item.ANCHOR_B_Y),
                anchor_b_z=Scalar[DTYPE](E_item.ANCHOR_B_Z),
                relpose_x=Scalar[DTYPE](E_item.RELPOSE_X),
                relpose_y=Scalar[DTYPE](E_item.RELPOSE_Y),
                relpose_z=Scalar[DTYPE](E_item.RELPOSE_Z),
                relpose_w=Scalar[DTYPE](E_item.RELPOSE_W),
                solref_0=Scalar[DTYPE](E_item.SOLREF_0),
                solref_1=Scalar[DTYPE](E_item.SOLREF_1),
                solimp_0=Scalar[DTYPE](E_item.SOLIMP_0),
                solimp_1=Scalar[DTYPE](E_item.SOLIMP_1),
                solimp_2=Scalar[DTYPE](E_item.SOLIMP_2),
            )
        model.num_equality = Self.N


@fieldwise_init
struct Geoms[*G: GeomSpec]:
    """Compile-time list of geom specifications (static + body-attached).

    Provides N (total geom count), type-level access via geom_types[i],
    and helper counts for static vs dynamic geoms.
    """

    comptime geom_types = Variadic.types[T=GeomSpec, *Self.G]
    comptime N: Int = Variadic.size(Self.geom_types)

    @staticmethod
    fn _count_static_geoms() -> Int:
        """Count of static (worldbody) geoms (BODY_IDX == 0)."""
        var total = 0

        @parameter
        for i in range(Self.N):

            @parameter
            if Self.geom_types[i].BODY_IDX == 0:
                total += 1
        return total

    @staticmethod
    fn _count_plane_geoms() -> Int:
        """Count of plane geoms (GEOM_TYPE == GEOM_PLANE)."""
        var total = 0

        @parameter
        for i in range(Self.N):

            @parameter
            if Self.geom_types[i].GEOM_TYPE == GEOM_PLANE:
                total += 1
        return total

    @staticmethod
    fn setup_model[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        Defaults: ModelDefaultsLike = ModelDefaults[],
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
        ]
    ):
        """Populate model geom arrays from compile-time GeomSpec list.

        Resolves sentinel values (-1.0/-1) from ModelDefaults.
        Sets geom type, body index, position, orientation, size, collision
        filtering, friction, and per-geom solref/solimp.
        """

        @parameter
        for i in range(Self.N):
            comptime G_item = Self.geom_types[i]

            # Geom arrays
            model.geom_type[i] = G_item.GEOM_TYPE
            model.geom_body[i] = G_item.BODY_IDX
            model.geom_pos[i * 3 + 0] = Scalar[DTYPE](G_item.POS_X)
            model.geom_pos[i * 3 + 1] = Scalar[DTYPE](G_item.POS_Y)
            model.geom_pos[i * 3 + 2] = Scalar[DTYPE](G_item.POS_Z)
            model.geom_quat[i * 4 + 0] = Scalar[DTYPE](G_item.QUAT_X)
            model.geom_quat[i * 4 + 1] = Scalar[DTYPE](G_item.QUAT_Y)
            model.geom_quat[i * 4 + 2] = Scalar[DTYPE](G_item.QUAT_Z)
            model.geom_quat[i * 4 + 3] = Scalar[DTYPE](G_item.QUAT_W)
            model.geom_radius[i] = Scalar[DTYPE](G_item.RADIUS)
            model.geom_half_length[i] = Scalar[DTYPE](G_item.HALF_LENGTH)
            model.geom_half_x[i] = Scalar[DTYPE](G_item.HALF_X)
            model.geom_half_y[i] = Scalar[DTYPE](G_item.HALF_Y)
            model.geom_half_z[i] = Scalar[DTYPE](G_item.HALF_Z)

            # Resolve physics fields via defaults
            model.geom_friction[i] = Scalar[DTYPE](
                _resolve_f64[G_item.FRICTION, Defaults.GEOM_FRICTION]()
            )
            model.geom_condim[i] = _resolve_int[
                G_item.CONDIM, Defaults.GEOM_CONDIM
            ]()
            model.geom_friction_spin[i] = Scalar[DTYPE](
                _resolve_f64[
                    G_item.FRICTION_SPIN, Defaults.GEOM_FRICTION_SPIN
                ]()
            )
            model.geom_friction_roll[i] = Scalar[DTYPE](
                _resolve_f64[
                    G_item.FRICTION_ROLL, Defaults.GEOM_FRICTION_ROLL
                ]()
            )
            model.geom_contype[i] = _resolve_int[
                G_item.CONTYPE, Defaults.GEOM_CONTYPE
            ]()
            model.geom_conaffinity[i] = _resolve_int[
                G_item.CONAFFINITY, Defaults.GEOM_CONAFFINITY
            ]()

            # Per-geom solref/solimp (resolved from defaults)
            model.geom_solref[i * 2 + 0] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLREF_0, Defaults.GEOM_SOLREF_0]()
            )
            model.geom_solref[i * 2 + 1] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLREF_1, Defaults.GEOM_SOLREF_1]()
            )
            model.geom_solimp[i * 3 + 0] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_0, Defaults.GEOM_SOLIMP_0]()
            )
            model.geom_solimp[i * 3 + 1] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_1, Defaults.GEOM_SOLIMP_1]()
            )
            model.geom_solimp[i * 3 + 2] = Scalar[DTYPE](
                _resolve_f64[G_item.SOLIMP_2, Defaults.GEOM_SOLIMP_2]()
            )

            # Contact margin (resolved from defaults)
            model.geom_margin[i] = Scalar[DTYPE](
                _resolve_f64[G_item.MARGIN, Defaults.GEOM_MARGIN]()
            )

            # Compute bounding sphere radius
            @parameter
            if G_item.GEOM_TYPE == GEOM_SPHERE:
                model.geom_rbound[i] = Scalar[DTYPE](G_item.RADIUS)
            elif G_item.GEOM_TYPE == GEOM_CAPSULE:
                model.geom_rbound[i] = Scalar[DTYPE](
                    G_item.HALF_LENGTH
                ) + Scalar[DTYPE](G_item.RADIUS)
            elif G_item.GEOM_TYPE == GEOM_BOX:
                model.geom_rbound[i] = sqrt(
                    Scalar[DTYPE](G_item.HALF_X) * Scalar[DTYPE](G_item.HALF_X)
                    + Scalar[DTYPE](G_item.HALF_Y)
                    * Scalar[DTYPE](G_item.HALF_Y)
                    + Scalar[DTYPE](G_item.HALF_Z)
                    * Scalar[DTYPE](G_item.HALF_Z)
                )
            elif G_item.GEOM_TYPE == GEOM_PLANE:
                model.geom_rbound[i] = Scalar[DTYPE](
                    1e10
                )  # Planes are infinite

            # Compute and store per-geom mass
            # Priority: explicit mass > explicit density > default density
            @parameter
            if G_item.GEOM_TYPE == GEOM_PLANE:
                model.geom_mass[i] = Scalar[DTYPE](0)
            elif G_item.GEOM_MASS >= 0.0:
                # Explicit mass on the geom
                model.geom_mass[i] = Scalar[DTYPE](G_item.GEOM_MASS)
            else:
                # Compute volume
                var vol = geom_volume[DTYPE](
                    G_item.GEOM_TYPE,
                    Scalar[DTYPE](G_item.RADIUS),
                    Scalar[DTYPE](G_item.HALF_LENGTH),
                    Scalar[DTYPE](G_item.HALF_X),
                    Scalar[DTYPE](G_item.HALF_Y),
                    Scalar[DTYPE](G_item.HALF_Z),
                )

                @parameter
                if G_item.DENSITY >= 0.0:
                    # Explicit density on the geom
                    model.geom_mass[i] = Scalar[DTYPE](
                        G_item.DENSITY
                    ) * vol
                else:
                    # Use default density
                    model.geom_mass[i] = Scalar[DTYPE](
                        Defaults.GEOM_DENSITY
                    ) * vol


# =============================================================================
# Actuators — variadic actuator list
# =============================================================================


@fieldwise_init
struct Actuators[*A: ActuatorSpec]:
    """Compile-time list of actuator specifications.

    Provides N (actuator count), force computation (CPU + GPU), and
    velocity derivative computation for implicit integration (qDeriv).

    Actuators replace the Joints IS_ACTUATED/TAU_LIMIT mechanism with
    MuJoCo-style gain/bias functions: force = gain*ctrl + bias(qpos, qvel).
    """

    comptime act_types = Variadic.types[T=ActuatorSpec, *Self.A]
    comptime N: Int = Variadic.size(Self.act_types)

    # =========================================================================
    # CPU Operations
    # =========================================================================

    @staticmethod
    fn apply_actions[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        actions: List[Float64],
    ):
        """Apply actions through actuators to produce joint forces.

        For each actuator:
        1. Clamp ctrl to [ctrl_min, ctrl_max]
        2. Compute: force = gain * ctrl + bias(qpos, qvel)
        3. Clamp force to [force_min, force_max]
        4. Write gear * force to qfrc[dof_adr]

        Uses compile-time DOF_ADR and QPOS_ADR from ActuatorSpec.
        """

        @parameter
        for i in range(Self.N):
            comptime A_item = Self.act_types[i]
            comptime dof_adr = A_item.DOF_ADR
            comptime qpos_adr = A_item.QPOS_ADR

            # Get action value
            var ctrl = Float64(0)
            if i < len(actions):
                ctrl = actions[i]

            # Clamp ctrl
            if ctrl > A_item.CTRL_MAX:
                ctrl = A_item.CTRL_MAX
            elif ctrl < A_item.CTRL_MIN:
                ctrl = A_item.CTRL_MIN

            # Compute gain
            var gain = Scalar[DTYPE](A_item.GAINPRM_0)

            @parameter
            if A_item.GAINTYPE == GAIN_AFFINE:
                var qpos_val = data.qpos[qpos_adr]
                var qvel_val = data.qvel[dof_adr]
                gain = (
                    Scalar[DTYPE](A_item.GAINPRM_0)
                    + Scalar[DTYPE](A_item.GAINPRM_1) * qpos_val
                    + Scalar[DTYPE](A_item.GAINPRM_2) * qvel_val
                )

            # Compute bias
            var bias = Scalar[DTYPE](0)

            @parameter
            if A_item.BIASTYPE == BIAS_AFFINE:
                var qpos_val = data.qpos[qpos_adr]
                var qvel_val = data.qvel[dof_adr]
                bias = (
                    Scalar[DTYPE](A_item.BIASPRM_0)
                    + Scalar[DTYPE](A_item.BIASPRM_1) * qpos_val
                    + Scalar[DTYPE](A_item.BIASPRM_2) * qvel_val
                )

            # Compute force
            var force = gain * Scalar[DTYPE](ctrl) + bias

            # Clamp force
            if force > Scalar[DTYPE](A_item.FORCE_MAX):
                force = Scalar[DTYPE](A_item.FORCE_MAX)
            elif force < Scalar[DTYPE](A_item.FORCE_MIN):
                force = Scalar[DTYPE](A_item.FORCE_MIN)

            # Write to qfrc (gear * force)
            data.qfrc[dof_adr] = Scalar[DTYPE](A_item.GEAR) * force

    # =========================================================================
    # GPU Operations — inline per-env (called from inside kernels)
    # =========================================================================

    @always_inline
    @staticmethod
    fn apply_actions_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
        NQ: Int,
        NV: Int,
    ](
        states: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Apply actions through actuators for a single env on GPU.

        Uses compile-time DOF_ADR/QPOS_ADR from ActuatorSpec for state
        buffer addressing. NQ/NV are compile-time template params matching
        the joint layout.
        """
        comptime QPOS_OFF = qpos_offset[NQ, NV]()
        comptime QVEL_OFF = qvel_offset[NQ, NV]()
        comptime QFRC_OFF = qfrc_offset[NQ, NV]()

        @parameter
        for i in range(Self.N):
            comptime A_item = Self.act_types[i]
            comptime dof_adr = A_item.DOF_ADR
            comptime qpos_adr = A_item.QPOS_ADR

            # Get action value
            var ctrl = Scalar[GDTYPE](0)
            if i < ACTION_DIM:
                ctrl = rebind[Scalar[GDTYPE]](actions[env, i])

            # Clamp ctrl
            if ctrl > Scalar[GDTYPE](A_item.CTRL_MAX):
                ctrl = Scalar[GDTYPE](A_item.CTRL_MAX)
            elif ctrl < Scalar[GDTYPE](A_item.CTRL_MIN):
                ctrl = Scalar[GDTYPE](A_item.CTRL_MIN)

            # Compute gain
            var gain = Scalar[GDTYPE](A_item.GAINPRM_0)

            @parameter
            if A_item.GAINTYPE == GAIN_AFFINE:
                var qpos_val = rebind[Scalar[GDTYPE]](
                    states[env, QPOS_OFF + qpos_adr]
                )
                var qvel_val = rebind[Scalar[GDTYPE]](
                    states[env, QVEL_OFF + dof_adr]
                )
                gain = (
                    Scalar[GDTYPE](A_item.GAINPRM_0)
                    + Scalar[GDTYPE](A_item.GAINPRM_1) * qpos_val
                    + Scalar[GDTYPE](A_item.GAINPRM_2) * qvel_val
                )

            # Compute bias
            var bias = Scalar[GDTYPE](0)

            @parameter
            if A_item.BIASTYPE == BIAS_AFFINE:
                var qpos_val = rebind[Scalar[GDTYPE]](
                    states[env, QPOS_OFF + qpos_adr]
                )
                var qvel_val = rebind[Scalar[GDTYPE]](
                    states[env, QVEL_OFF + dof_adr]
                )
                bias = (
                    Scalar[GDTYPE](A_item.BIASPRM_0)
                    + Scalar[GDTYPE](A_item.BIASPRM_1) * qpos_val
                    + Scalar[GDTYPE](A_item.BIASPRM_2) * qvel_val
                )

            # Compute force
            var force = gain * ctrl + bias

            # Clamp force
            if force > Scalar[GDTYPE](A_item.FORCE_MAX):
                force = Scalar[GDTYPE](A_item.FORCE_MAX)
            elif force < Scalar[GDTYPE](A_item.FORCE_MIN):
                force = Scalar[GDTYPE](A_item.FORCE_MIN)

            # Write to qfrc (gear * force) at compile-time DOF address
            states[env, QFRC_OFF + dof_adr] = (
                Scalar[GDTYPE](A_item.GEAR) * force
            )

    @staticmethod
    fn compute_qderiv_contribution[
        DTYPE: DType,
        NV: Int,
    ](mut qderiv: InlineArray[Scalar[DTYPE], NV * NV]):
        """Add actuator velocity derivative contributions to qDeriv.

        For each actuator with velocity-dependent gain or bias:
            qDeriv[dof, dof] += gear * (gainprm_2 + biasprm_2)

        This is used by ImplicitFastIntegrator: M_hat = M + arm - dt*qDeriv.
        Velocity-dependent terms contribute negative damping-like effects.
        """

        @parameter
        for i in range(Self.N):
            comptime A_item = Self.act_types[i]
            comptime dof = A_item.DOF_ADR
            # Velocity derivative: d(force)/d(qvel) = gear * (gainprm_2 + biasprm_2)
            comptime vel_deriv = A_item.GEAR * (
                A_item.GAINPRM_2 + A_item.BIASPRM_2
            )

            @parameter
            if vel_deriv != 0.0:
                qderiv[dof * NV + dof] += Scalar[DTYPE](vel_deriv)

    @always_inline
    @staticmethod
    fn compute_qderiv_contribution_gpu[
        GDTYPE: DType,
        NV: Int,
    ](
        workspace: LayoutTensor[GDTYPE, _, MutAnyOrigin],
        env: Int,
        qderiv_offset: Int,
    ):
        """Add actuator velocity derivative contributions to qDeriv in GPU workspace.
        """

        @parameter
        for i in range(Self.N):
            comptime A_item = Self.act_types[i]
            comptime dof = A_item.DOF_ADR
            comptime vel_deriv = A_item.GEAR * (
                A_item.GAINPRM_2 + A_item.BIASPRM_2
            )

            @parameter
            if vel_deriv != 0.0:
                var idx = qderiv_offset + dof * NV + dof
                var cur = rebind[Scalar[GDTYPE]](workspace[env, idx])
                workspace[env, idx] = cur + Scalar[GDTYPE](vel_deriv)

    @staticmethod
    fn apply_actions_kernel_gpu[
        GDTYPE: DType,
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
        NQ: Int,
        NV: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[GDTYPE],
        actions_buf: DeviceBuffer[GDTYPE],
    ) raises:
        """Launch kernel to apply actuator actions for all envs."""
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
            Self.apply_actions_gpu[
                GDTYPE, BATCH_SIZE, STATE_SIZE, ACTION_DIM, NQ, NV
            ](states, actions, env)

        ctx.enqueue_function[kernel, kernel](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


# =============================================================================
# Cameras — variadic camera list (purely visual, no setup_model)
# =============================================================================


@fieldwise_init
struct Cameras[*C: CameraSpec]:
    """Compile-time list of camera specifications.

    Provides N (camera count) and type-level access to each camera via cam_types[i].
    Cameras are purely visual — no setup_model needed.
    """

    comptime cam_types = Variadic.types[T=CameraSpec, *Self.C]
    comptime N: Int = Variadic.size(Self.cam_types)


# =============================================================================
# Lights — variadic light list (purely visual, no setup_model)
# =============================================================================


@fieldwise_init
struct Lights[*L: LightSpec]:
    """Compile-time list of light specifications.

    Provides N (light count) and type-level access to each light via light_types[i].
    Lights are purely visual — no setup_model needed.
    """

    comptime light_types = Variadic.types[T=LightSpec, *Self.L]
    comptime N: Int = Variadic.size(Self.light_types)


# =============================================================================
# Textures — variadic texture list (purely visual, no setup_model)
# =============================================================================


@fieldwise_init
struct Textures[*T: TextureSpec]:
    """Compile-time list of texture specifications.

    Provides N (texture count) and type-level access to each texture via tex_types[i].
    Textures are purely visual assets — no setup_model needed.
    """

    comptime tex_types = Variadic.types[T=TextureSpec, *Self.T]
    comptime N: Int = Variadic.size(Self.tex_types)


# =============================================================================
# Materials — variadic material list (purely visual, no setup_model)
# =============================================================================


@fieldwise_init
struct Materials[*M: MaterialSpec]:
    """Compile-time list of material specifications.

    Provides N (material count) and type-level access to each material via mat_types[i].
    Materials define surface appearance (shininess, specular, reflectance, emission)
    and can reference a texture by name. Geoms reference materials by MATERIAL_NAME.
    """

    comptime mat_types = Variadic.types[T=MaterialSpec, *Self.M]
    comptime N: Int = Variadic.size(Self.mat_types)


@fieldwise_init
struct ModelDef[
    nbody: Int,
    njoint: Int,
    nq: Int,
    nv: Int,
    ngeom: Int = 0,
    max_equality: Int = 0,
    cone_type: Int = ConeType.ELLIPTIC,
]:
    """Compile-time model definition with pre-computed dimensions.

    Takes concrete Int parameters rather than Bodies/Joints directly,
    because Mojo cannot resolve variadic type packs through nesting.

    Usage:
        comptime MyBodies = Bodies[...]
        comptime MyJoints = Joints[...]
        comptime MyGeoms = Geoms[...]
        comptime MyModel = ModelDef[
            MyBodies.N, MyJoints.N,
            MyJoints._sum_nq(), MyJoints._sum_nv(),
            MyGeoms.N,
        ]
    """

    comptime NBODY: Int = Self.nbody
    comptime NJOINT: Int = Self.njoint
    comptime NQ: Int = Self.nq
    comptime NV: Int = Self.nv
    comptime NGEOM: Int = Self.ngeom
    comptime MAX_EQUALITY: Int = Self.max_equality
    comptime CONE_TYPE: Int = Self.cone_type

    # =========================================================================
    # Model Creation Helpers
    # =========================================================================

    @staticmethod
    fn setup_solver_params[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        Defaults: ModelDefaultsLike = ModelDefaults[],
    ](
        mut model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
        ],
    ):
        """Set all solver impedance params on a Model from ModelDefaults.

        Sets model-level solref/solimp for contacts and limits, plus impratio.
        Per-geom and per-joint overrides are set in Geoms/Joints.setup_model.
        """
        model.solref_contact[0] = Scalar[DTYPE](Defaults.GEOM_SOLREF_0)
        model.solref_contact[1] = Scalar[DTYPE](Defaults.GEOM_SOLREF_1)
        model.solimp_contact[0] = Scalar[DTYPE](Defaults.GEOM_SOLIMP_0)
        model.solimp_contact[1] = Scalar[DTYPE](Defaults.GEOM_SOLIMP_1)
        model.solimp_contact[2] = Scalar[DTYPE](Defaults.GEOM_SOLIMP_2)
        model.solref_limit[0] = Scalar[DTYPE](Defaults.JOINT_SOLREF_LIMIT_0)
        model.solref_limit[1] = Scalar[DTYPE](Defaults.JOINT_SOLREF_LIMIT_1)
        model.solimp_limit[0] = Scalar[DTYPE](Defaults.JOINT_SOLIMP_LIMIT_0)
        model.solimp_limit[1] = Scalar[DTYPE](Defaults.JOINT_SOLIMP_LIMIT_1)
        model.solimp_limit[2] = Scalar[DTYPE](Defaults.JOINT_SOLIMP_LIMIT_2)
        model.impratio = Scalar[DTYPE](Defaults.IMPRATIO)
        model.gravity = SIMD[DTYPE, 4](
            Scalar[DTYPE](Defaults.GRAVITY_X),
            Scalar[DTYPE](Defaults.GRAVITY_Y),
            Scalar[DTYPE](Defaults.GRAVITY_Z),
            Scalar[DTYPE](0),
        )
        model.timestep = Scalar[DTYPE](Defaults.TIMESTEP)

    @staticmethod
    fn finalize[
        DTYPE: DType,
        MAX_CONTACTS: Int,
        Defaults: ModelDefaultsLike = ModelDefaults[],
    ](
        mut model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
        ],
        mut data: Data[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
        ],
    ) where DTYPE.is_floating_point():
        """Run FK + compute_body_invweight0 in the correct order.

        Must be called after Bodies/Joints/Geoms.setup_model and after
        Joints.reset_data (or manual qpos initialization).

        Order: inertiafromgeom → settotalmass → FK → invweight0

        If Defaults.INERTIAFROMGEOM is True, computes body mass/inertia/ipos/iquat
        from child geoms (overwriting any values set by Bodies.setup_model).

        If Defaults.SETTOTALMASS > 0, rescales all body masses and inertias
        so the total mass equals the target (MuJoCo <compiler settotalmass>).
        """
        # MuJoCo <compiler inertiafromgeom> — compute body inertia from geoms
        @parameter
        if Defaults.INERTIAFROMGEOM and Self.NGEOM > 0:
            compute_inertia_from_geoms(model)

        # MuJoCo <compiler settotalmass> — rescale body masses/inertias
        @parameter
        if Defaults.SETTOTALMASS > 0.0:
            var total_mass = Scalar[DTYPE](0)
            for i in range(1, Self.NBODY):
                total_mass += model.body_mass[i]
            if total_mass > 0:
                var scale = Scalar[DTYPE](Defaults.SETTOTALMASS) / total_mass
                for i in range(1, Self.NBODY):
                    model.body_mass[i] *= scale
                    model.body_inv_mass[i] /= scale
                    for k in range(3):
                        model.body_inertia[i * 3 + k] *= scale
                        model.body_inv_inertia[i * 3 + k] /= scale

        forward_kinematics(model, data)
        compute_body_invweight0(model, data)

    @staticmethod
    fn create_gpu_model_buffer[
        DTYPE: DType,
        MAX_CONTACTS: Int,
    ](
        ctx: DeviceContext,
        model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
            Self.NGEOM,
            Self.MAX_EQUALITY,
            Self.CONE_TYPE,
        ],
    ) raises -> HostBuffer[DTYPE]:
        """Create a GPU host buffer from a fully-configured model.

        Allocates buffer with model_size_with_invweight, copies model data,
        geoms, and invweight0 arrays. Returns host buffer ready for
        ctx.enqueue_copy to a DeviceBuffer.
        """
        comptime BUF_SIZE = model_size_with_invweight[
            Self.NBODY,
            Self.NJOINT,
            Self.NV,
            Self.NGEOM,
        ]()
        var host_buf = ctx.enqueue_create_host_buffer[DTYPE](BUF_SIZE)
        for i in range(BUF_SIZE):
            host_buf[i] = Scalar[DTYPE](0)
        copy_model_to_buffer(model, host_buf)
        copy_geoms_to_buffer(model, host_buf)
        copy_invweight0_to_buffer(model, host_buf)
        return host_buf^
