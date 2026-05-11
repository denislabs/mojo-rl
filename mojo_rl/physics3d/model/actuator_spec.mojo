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

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim, barrier
from ..gpu.constants import (
    TPB,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    model_size_with_invweight,
)

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

    comptime JOINT_IDX: Int  # Which joint this actuates
    comptime DOF_ADR: Int  # DOF address (qvel offset of the joint)
    comptime QPOS_ADR: Int  # Qpos address (qpos offset of the joint)
    comptime GEAR: Float64  # Force scaling (gear ratio)
    comptime DYNTYPE: Int  # DYN_NONE / DYN_INTEGRATOR / DYN_FILTER / DYN_FILTEREXACT
    comptime DYNPRM_0: Float64  # Time constant for filter (default 1.0)
    comptime GAINTYPE: Int  # GAIN_FIXED / GAIN_AFFINE
    comptime GAINPRM_0: Float64  # Gain coefficient 0 (fixed gain, or affine intercept)
    comptime GAINPRM_1: Float64  # Gain coefficient 1 (length-dependent)
    comptime GAINPRM_2: Float64  # Gain coefficient 2 (velocity-dependent)
    comptime BIASTYPE: Int  # BIAS_NONE / BIAS_AFFINE
    comptime BIASPRM_0: Float64  # Bias coefficient 0 (constant)
    comptime BIASPRM_1: Float64  # Bias coefficient 1 (length-dependent)
    comptime BIASPRM_2: Float64  # Bias coefficient 2 (velocity-dependent)
    comptime CTRL_MIN: Float64  # Control range min (default -1.0)
    comptime CTRL_MAX: Float64  # Control range max (default 1.0)
    comptime FORCE_MIN: Float64  # Force range min (default -inf)
    comptime FORCE_MAX: Float64  # Force range max (default +inf)
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


# =============================================================================
# Actuators — variadic actuator list
# =============================================================================


trait ActuatorsLike:
    """Trait for compile-time actuator container types."""

    comptime N: Int

    @staticmethod
    def apply_actions[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        actions: List[Float64],
    ):
        ...

    @staticmethod
    def apply_actions_kernel_gpu[
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
        ...

    @staticmethod
    def compute_qderiv_contribution[
        DTYPE: DType,
        NV: Int,
    ](mut qderiv: InlineArray[Scalar[DTYPE], NV * NV]):
        ...

    @always_inline
    @staticmethod
    def compute_qderiv_contribution_gpu[
        GDTYPE: DType,
        NV: Int,
    ](
        workspace: LayoutTensor[GDTYPE, _, MutAnyOrigin, ...],
        env: Int,
        qderiv_offset: Int,
    ):
        ...

    @always_inline
    @staticmethod
    def apply_actions_gpu[
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
        ...


@fieldwise_init
struct Actuators[*A: ActuatorSpec](ActuatorsLike):
    """Compile-time list of actuator specifications.

    Provides N (actuator count), force computation (CPU + GPU), and
    velocity derivative computation for implicit integration (qDeriv).

    Actuators replace the Joints IS_ACTUATED/TAU_LIMIT mechanism with
    MuJoCo-style gain/bias functions: force = gain*ctrl + bias(qpos, qvel).
    """

    comptime act_types = Self.A
    comptime N: Int = Self.act_types.size

    # =========================================================================
    # CPU Operations
    # =========================================================================

    @staticmethod
    def apply_actions[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
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

        comptime for i in range(Self.N):
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

            comptime if A_item.GAINTYPE == GAIN_AFFINE:
                var qpos_val = data.qpos[qpos_adr]
                var qvel_val = data.qvel[dof_adr]
                gain = (
                    Scalar[DTYPE](A_item.GAINPRM_0)
                    + Scalar[DTYPE](A_item.GAINPRM_1) * qpos_val
                    + Scalar[DTYPE](A_item.GAINPRM_2) * qvel_val
                )

            # Compute bias
            var bias = Scalar[DTYPE](0)

            comptime if A_item.BIASTYPE == BIAS_AFFINE:
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
    def apply_actions_gpu[
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

        comptime for i in range(Self.N):
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

            comptime if A_item.GAINTYPE == GAIN_AFFINE:
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

            comptime if A_item.BIASTYPE == BIAS_AFFINE:
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
            var gear_force = Scalar[GDTYPE](A_item.GEAR) * force
            states[env, QFRC_OFF + dof_adr] = gear_force

            # Also capture to qfrc_actuator region (last NV elements of state buffer).
            # qfrc_actuator is always the last NV elements: offset = STATE_SIZE - NV.
            comptime QFRC_ACT_OFF = STATE_SIZE - NV
            states[env, QFRC_ACT_OFF + dof_adr] = gear_force

    @staticmethod
    def compute_qderiv_contribution[
        DTYPE: DType,
        NV: Int,
    ](mut qderiv: InlineArray[Scalar[DTYPE], NV * NV]):
        """Add actuator velocity derivative contributions to qDeriv.

        For each actuator with velocity-dependent gain or bias:
            qDeriv[dof, dof] += gear * (gainprm_2 + biasprm_2)

        This is used by ImplicitFastIntegrator: M_hat = M + arm - dt*qDeriv.
        Velocity-dependent terms contribute negative damping-like effects.
        """

        comptime for i in range(Self.N):
            comptime A_item = Self.act_types[i]
            comptime dof = A_item.DOF_ADR
            # Velocity derivative: d(force)/d(qvel) = gear * (gainprm_2 + biasprm_2)
            comptime vel_deriv = A_item.GEAR * (
                A_item.GAINPRM_2 + A_item.BIASPRM_2
            )

            comptime if vel_deriv != 0.0:
                qderiv[dof * NV + dof] += Scalar[DTYPE](vel_deriv)

    @always_inline
    @staticmethod
    def compute_qderiv_contribution_gpu[
        GDTYPE: DType,
        NV: Int,
    ](
        workspace: LayoutTensor[GDTYPE, _, MutAnyOrigin, ...],
        env: Int,
        qderiv_offset: Int,
    ):
        """Add actuator velocity derivative contributions to qDeriv in GPU workspace.
        """

        comptime for i in range(Self.N):
            comptime A_item = Self.act_types[i]
            comptime dof = A_item.DOF_ADR
            comptime vel_deriv = A_item.GEAR * (
                A_item.GAINPRM_2 + A_item.BIASPRM_2
            )

            comptime if vel_deriv != 0.0:
                var idx = qderiv_offset + dof * NV + dof
                var cur = rebind[Scalar[GDTYPE]](workspace[env, idx])
                workspace[env, idx] = cur + Scalar[GDTYPE](vel_deriv)

    @staticmethod
    def apply_actions_kernel_gpu[
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
        ](states_buf)
        var actions = LayoutTensor[
            GDTYPE, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
        ](actions_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @parameter
        @always_inline
        def kernel(
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

        ctx.enqueue_function[kernel](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


@fieldwise_init
struct _EmptyActuators(ActuatorsLike):
    comptime N: Int = 0

    @staticmethod
    def apply_actions[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        actions: List[Float64],
    ):
        pass

    @staticmethod
    def apply_actions_kernel_gpu[
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
        pass

    @staticmethod
    def compute_qderiv_contribution[
        DTYPE: DType,
        NV: Int,
    ](mut qderiv: InlineArray[Scalar[DTYPE], NV * NV]):
        pass

    @always_inline
    @staticmethod
    def compute_qderiv_contribution_gpu[
        GDTYPE: DType,
        NV: Int,
    ](
        workspace: LayoutTensor[GDTYPE, _, MutAnyOrigin, ...],
        env: Int,
        qderiv_offset: Int,
    ):
        pass

    @always_inline
    @staticmethod
    def apply_actions_gpu[
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
        pass
