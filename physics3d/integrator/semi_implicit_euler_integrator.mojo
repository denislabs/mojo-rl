"""Semi-implicit Euler integrator for Generalized Coordinates engine.

Implements the main simulation step:
1. Forward kinematics: qpos -> xpos, xquat
2. Collision detection (optional)
3. Compute dynamics: mass matrix M(q), bias forces b(q, qvel)
4. Solve: qacc = M^-1 * (qfrc - bias)
5. Integrate: qvel += qacc * dt, qpos += qvel * dt
6. Normalize quaternions in qpos

Semi-implicit Euler is symplectic and provides good energy conservation.
"""

from math import sqrt
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout

from ..types import ModelGC, DataGC, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from ..kinematics.quat_math import quat_normalize, quat_integrate, quat_rotate
from ..dynamics.mass_matrix import compute_mass_matrix, solve_linear_diagonal
from ..dynamics.bias_forces import compute_bias_forces
from ..solver.semi_implicit_euler_solver import (
    forward_kinematics,
    compute_body_velocities,
    detect_ground_contacts,
    compute_mass_matrix,
    solve_linear_diagonal,
    compute_bias_forces,
    compute_contact_forces,
    normalize_qpos_quaternions,
    enforce_joint_limits,
)
from ..traits.integrator import GcIntegrator
from ..gpu.constants import (
    TPB,
    gc_state_size,
    gc_model_size,
)
from ..gpu.gc_kernels import step_gc_kernel


struct SemiImplicitEulerIntegrator(GcIntegrator):
    """Generalized Coordinates integrator.

    Implements the main simulation step for the Generalized Coordinates engine.
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    @staticmethod
    fn step[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](
        model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    ):
        """Execute one simulation step with collision detection.

        Same as step_gc but includes ground plane collision detection.
        Contact forces are applied as additional joint-space forces.

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
        """
        var dt = model.timestep

        # 1. Forward kinematics
        forward_kinematics(model, data)
        compute_body_velocities(model, data)

        # 2. Collision detection with ground
        detect_ground_contacts(model, data)

        # 3. Compute dynamics
        comptime M_SIZE = _max_one[NV * NV]()
        comptime V_SIZE = _max_one[NV]()

        var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M[i] = Scalar[DTYPE](0)
        compute_mass_matrix[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, M_SIZE](
            model, data, M
        )

        var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            bias[i] = Scalar[DTYPE](0)
        compute_bias_forces[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE](
            model, data, bias
        )

        # 4. Compute contact forces in joint space
        var qfrc_contact = InlineArray[Scalar[DTYPE], V_SIZE](
            uninitialized=True
        )
        for i in range(V_SIZE):
            qfrc_contact[i] = Scalar[DTYPE](0)
        compute_contact_forces(model, data, qfrc_contact)

        # 5. Net force
        var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            f_net[i] = Scalar[DTYPE](0)
        for i in range(NV):
            f_net[i] = data.qfrc[i] + qfrc_contact[i] - bias[i]

        # 6. Solve M * qacc = f_net
        solve_linear_diagonal[DTYPE, NV, M_SIZE, V_SIZE](M, f_net, data.qacc)

        # 7. Integration
        for i in range(NV):
            data.qvel[i] = data.qvel[i] + data.qacc[i] * dt

        for i in range(NQ):
            data.qpos[i] = data.qpos[i] + data.qvel[i] * dt

        # 8. Normalize quaternions
        normalize_qpos_quaternions(model, data)

        # 9. Enforce joint limits
        enforce_joint_limits(model, data)

    @staticmethod
    fn simulate[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](
        model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        num_steps: Int,
    ):
        """Run simulation for multiple steps on CPU.

        Args:
            model: Static model configuration.
            data: Mutable simulation state (will be modified).
            num_steps: Number of simulation steps to run.
        """
        for _ in range(num_steps):
            Self.step(model, data)

    # =========================================================================
    # GPU Methods
    # =========================================================================

    @staticmethod
    fn step_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU.

        Runs physics for all BATCH environments in parallel.

        GPU state buffer layout for GC engine:
        - qpos: [NQ] joint positions
        - qvel: [NV] joint velocities
        - qacc: [NV] joint accelerations
        - qfrc: [NV] joint forces
        - xpos: [NBODY * 3] body world positions
        - xquat: [NBODY * 4] body world orientations
        - xvel: [NBODY * 3] body world linear velocities
        - xangvel: [NBODY * 3] body world angular velocities
        - contacts: [MAX_CONTACTS * 12] contact data
        - metadata: [4] (num_contacts, padding)

        Pipeline:
        1. Forward kinematics (qpos -> xpos, xquat)
        2. Compute body velocities (qvel -> xvel, xangvel)
        3. Ground contact detection
        4. Mass matrix computation (diagonal)
        5. Bias forces computation
        6. Contact forces computation
        7. Integration (solve M*qacc = f, integrate qvel, qpos)
        8. Quaternion normalization

        Args:
            ctx: GPU device context.
            state_buf: Device buffer containing joint-space state for all environments.
            model_buf: Device buffer containing per-body model data.
            dt: Timestep.
            gravity_z: Z-component of gravity.
            ground_z: Ground plane height.
        """
        comptime STATE_SIZE = gc_state_size[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime MODEL_SIZE = gc_model_size[NBODY, NJOINT]()
        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())

        var model = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            model: LayoutTensor[
                DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return

            step_gc_kernel[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
            ](env, state, model)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            model,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn simulate_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        num_steps: Int,
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """Run simulation for multiple steps on GPU.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer containing joint-space state for all environments.
            model_buf: Device buffer containing per-body model data.
            num_steps: Number of simulation steps to run.
            dt: Timestep.
            gravity_z: Z-component of gravity.
            ground_z: Ground plane height.
        """
        for _ in range(num_steps):
            Self.step_gpu[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH](
                ctx, state_buf, model_buf, dt, gravity_z, ground_z
            )
