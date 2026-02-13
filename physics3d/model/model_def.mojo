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
from ..types import Model, Data
from ..joint_types import JNT_HINGE, JNT_SLIDE
from ..constants import GEOM_PLANE

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
)


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
    ](mut model: Model[DTYPE, NQ, NV, Self.N, NJOINT, MAX_CONTACTS, NGEOM]):
        """Populate model body properties from compile-time BodySpec list.

        Iterates over all body specs and sets mass, inertia, geometry, parent,
        local frame, and collision filtering on the model.
        """

        @parameter
        for i in range(Self.N):
            comptime B = Self.body_types[i]

            # Mass, inertia
            model.set_body(
                i,
                name=B.NAME,
                mass=Scalar[DTYPE](B.MASS),
                inertia=(
                    Scalar[DTYPE](B.ixx()),
                    Scalar[DTYPE](B.iyy()),
                    Scalar[DTYPE](B.izz()),
                ),
            )

            # Kinematic tree
            model.set_body_parent(i, B.PARENT)

            # Local frame in parent
            model.set_body_local_frame(
                i,
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
                i,
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
    ](mut model: Model[DTYPE, NQ, NV, NBODY, Self.N, MAX_CONTACTS, NGEOM]):
        """Populate model joints from compile-time JointSpec list.

        Iterates over all joint specs and calls add_hinge_joint or
        add_slide_joint with correct qpos/qvel offsets.
        """

        @parameter
        for i in range(Self.N):
            comptime J = Self.joint_types[i]

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
                    armature=Scalar[DTYPE](J.ARMATURE),
                    damping=Scalar[DTYPE](J.DAMPING),
                    stiffness=Scalar[DTYPE](J.STIFFNESS),
                    springref=Scalar[DTYPE](J.SPRINGREF),
                    frictionloss=Scalar[DTYPE](J.FRICTIONLOSS),
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
                    armature=Scalar[DTYPE](J.ARMATURE),
                    damping=Scalar[DTYPE](J.DAMPING),
                    stiffness=Scalar[DTYPE](J.STIFFNESS),
                    springref=Scalar[DTYPE](J.SPRINGREF),
                    frictionloss=Scalar[DTYPE](J.FRICTIONLOSS),
                )


# =============================================================================
# ModelDef — full model compositor (concrete Int parameters)
# =============================================================================


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
        """Count of static (worldbody) geoms (BODY_IDX == -1)."""
        var total = 0

        @parameter
        for i in range(Self.N):

            @parameter
            if Self.geom_types[i].BODY_IDX == -1:
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
    ](mut model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM]):
        """Populate model geom arrays from compile-time GeomSpec list.

        Sets geom type, body index, position, orientation, size, collision
        filtering, and friction for each geom. For plane geoms, also writes
        to ground_z and friction.
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
            model.geom_friction[i] = Scalar[DTYPE](G_item.FRICTION)
            model.geom_contype[i] = G_item.CONTYPE
            model.geom_conaffinity[i] = G_item.CONAFFINITY

            # For plane geoms, also write to legacy ground fields
            @parameter
            if G_item.GEOM_TYPE == GEOM_PLANE:
                model.ground_z = Scalar[DTYPE](G_item.POS_Z)
                model.friction = Scalar[DTYPE](G_item.FRICTION)


@fieldwise_init
struct ModelDef[nbody: Int, njoint: Int, nq: Int, nv: Int, ngeom: Int = 0]:
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
