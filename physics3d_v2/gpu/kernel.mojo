"""Physics3D v2 GPU Kernel - Fused physics step kernel.

Implements the complete Phase 1-2 physics step in a single kernel:
1. Update kinematics (xpos from qpos)
2. Collision detection (pre-step)
3. Contact constraint solving
4. Compute accelerations
5. Cancel gravity if resting
6. Integrate velocities and positions
7. Update kinematics (post-step)
8. Collision detection (post-step)
9. Position correction

Pattern: 1 environment = 1 thread
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from .constants import (
    STATE_SIZE,
    TPB,
    GEOM_SPHERE,
    # Field indices
    IDX_X,
    IDX_Y,
    IDX_Z,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_QW,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_AX,
    IDX_AY,
    IDX_AZ,
    IDX_ALPHA_X,
    IDX_ALPHA_Y,
    IDX_ALPHA_Z,
    IDX_FX,
    IDX_FY,
    IDX_FZ,
    IDX_TAU_X,
    IDX_TAU_Y,
    IDX_TAU_Z,
    IDX_XPOS_X,
    IDX_XPOS_Y,
    IDX_XPOS_Z,
    IDX_CONTACT_ACTIVE,
    IDX_CONTACT_DEPTH,
    IDX_CONTACT_NX,
    IDX_CONTACT_NY,
    IDX_CONTACT_NZ,
    IDX_CONTACT_PX,
    IDX_CONTACT_PY,
    IDX_CONTACT_PZ,
)


# =============================================================================
# Physics3D v2 Fused Kernel
# =============================================================================


struct Physics3DV2Kernel:
    """Fused physics step kernel for single-body simulation.

    This kernel performs the COMPLETE physics step in ONE GPU launch:
    1. Update kinematics (xpos from qpos)
    2. Pre-step collision detection
    3. Velocity constraint solving (contact impulse)
    4. Compute accelerations (gravity + applied forces)
    5. Cancel gravity if resting on ground
    6. Semi-implicit Euler integration
    7. Post-step collision detection
    8. Position correction (Baumgarte)

    Performance: All operations in one kernel = minimal memory traffic.
    """

    # =========================================================================
    # Core Physics Operations (GPU-compatible inline functions)
    # =========================================================================

    @always_inline
    @staticmethod
    fn _update_kinematics[
        DTYPE: DType, BATCH: Int
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Update world-frame position from generalized coordinates.

        For FREE joint (single body), xpos = qpos[0:3].
        """
        state[env, IDX_XPOS_X] = state[env, IDX_X]
        state[env, IDX_XPOS_Y] = state[env, IDX_Y]
        state[env, IDX_XPOS_Z] = state[env, IDX_Z]

    @always_inline
    @staticmethod
    fn _detect_sphere_plane[
        DTYPE: DType, BATCH: Int
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        radius: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        geom_type: Int,
    ):
        """Detect sphere-ground collision.

        Sets contact fields in state if penetrating.
        """
        # Only handle sphere geometry
        if geom_type != GEOM_SPHERE:
            state[env, IDX_CONTACT_ACTIVE] = Scalar[DTYPE](0)
            return

        var sphere_z = state[env, IDX_XPOS_Z]

        # Penetration depth: how far sphere bottom is below ground
        var depth = radius - (sphere_z - ground_z)

        if depth > Scalar[DTYPE](0):
            state[env, IDX_CONTACT_ACTIVE] = Scalar[DTYPE](1)
            state[env, IDX_CONTACT_DEPTH] = depth
            # Normal points up (from ground toward sphere)
            state[env, IDX_CONTACT_NX] = Scalar[DTYPE](0)
            state[env, IDX_CONTACT_NY] = Scalar[DTYPE](0)
            state[env, IDX_CONTACT_NZ] = Scalar[DTYPE](1)
            # Contact point on ground below sphere center
            state[env, IDX_CONTACT_PX] = state[env, IDX_XPOS_X]
            state[env, IDX_CONTACT_PY] = state[env, IDX_XPOS_Y]
            state[env, IDX_CONTACT_PZ] = ground_z
        else:
            state[env, IDX_CONTACT_ACTIVE] = Scalar[DTYPE](0)
            state[env, IDX_CONTACT_DEPTH] = Scalar[DTYPE](0)

    @always_inline
    @staticmethod
    fn _solve_contact[
        DTYPE: DType, BATCH: Int
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        mass: Scalar[DTYPE],
        restitution: Scalar[DTYPE],
        baumgarte: Scalar[DTYPE],
        slop: Scalar[DTYPE],
    ):
        """Apply contact impulse and position correction.

        Velocity constraint: if approaching ground, apply impulse.
        Position constraint: Baumgarte stabilization for residual penetration.
        """
        var contact_active = state[env, IDX_CONTACT_ACTIVE]
        if contact_active < Scalar[DTYPE](0.5):
            return

        # Velocity toward ground (negative = approaching)
        var vn = state[env, IDX_VZ]

        # Only apply impulse if approaching ground
        if vn < Scalar[DTYPE](0):
            # Impulse magnitude: j = -(1+e) * m * vn
            var j = -(Scalar[DTYPE](1) + restitution) * mass * vn
            # Apply impulse: Δv = j/m (only in z direction)
            state[env, IDX_VZ] = state[env, IDX_VZ] + j / mass

        # Position correction (Baumgarte stabilization)
        var depth = rebind[Scalar[DTYPE]](state[env, IDX_CONTACT_DEPTH])
        var correction_depth = depth - slop
        var correction: Scalar[DTYPE]
        if correction_depth > Scalar[DTYPE](0):
            correction = correction_depth * baumgarte
        else:
            correction = Scalar[DTYPE](0)

        # Apply correction directly to position
        state[env, IDX_Z] = rebind[Scalar[DTYPE]](state[env, IDX_Z]) + correction

    @always_inline
    @staticmethod
    fn _compute_acceleration[
        DTYPE: DType, BATCH: Int
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        gravity_z: Scalar[DTYPE],
        inv_mass: Scalar[DTYPE],
        inv_ixx: Scalar[DTYPE],
        inv_iyy: Scalar[DTYPE],
        inv_izz: Scalar[DTYPE],
    ):
        """Compute accelerations from forces (Newton's 2nd law).

        Linear: a = F/m + g
        Angular: α = I⁻¹·τ (diagonal inertia)
        """
        # Linear acceleration
        state[env, IDX_AX] = state[env, IDX_FX] * inv_mass
        state[env, IDX_AY] = state[env, IDX_FY] * inv_mass
        state[env, IDX_AZ] = state[env, IDX_FZ] * inv_mass + gravity_z

        # Angular acceleration
        state[env, IDX_ALPHA_X] = state[env, IDX_TAU_X] * inv_ixx
        state[env, IDX_ALPHA_Y] = state[env, IDX_TAU_Y] * inv_iyy
        state[env, IDX_ALPHA_Z] = state[env, IDX_TAU_Z] * inv_izz

    @always_inline
    @staticmethod
    fn _cancel_gravity_if_resting[
        DTYPE: DType, BATCH: Int
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Cancel gravity when resting on ground.

        If in contact and not moving up, clamp downward acceleration.
        """
        var contact_active = state[env, IDX_CONTACT_ACTIVE]
        var vz = state[env, IDX_VZ]
        var az = state[env, IDX_AZ]

        if contact_active >= Scalar[DTYPE](0.5) and vz <= Scalar[DTYPE](0):
            if az < Scalar[DTYPE](0):
                state[env, IDX_AZ] = Scalar[DTYPE](0)

    @always_inline
    @staticmethod
    fn _integrate[
        DTYPE: DType, BATCH: Int
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        dt: Scalar[DTYPE],
    ):
        """Semi-implicit Euler integration.

        1. Update velocities using current accelerations
        2. Update positions using NEW velocities
        3. Quaternion integration with normalization
        """
        # 1. Update velocities (using current accelerations)
        state[env, IDX_VX] = state[env, IDX_VX] + dt * state[env, IDX_AX]
        state[env, IDX_VY] = state[env, IDX_VY] + dt * state[env, IDX_AY]
        state[env, IDX_VZ] = state[env, IDX_VZ] + dt * state[env, IDX_AZ]
        state[env, IDX_WX] = state[env, IDX_WX] + dt * state[env, IDX_ALPHA_X]
        state[env, IDX_WY] = state[env, IDX_WY] + dt * state[env, IDX_ALPHA_Y]
        state[env, IDX_WZ] = state[env, IDX_WZ] + dt * state[env, IDX_ALPHA_Z]

        # 2. Update positions (using NEW velocities - semi-implicit)
        state[env, IDX_X] = state[env, IDX_X] + dt * state[env, IDX_VX]
        state[env, IDX_Y] = state[env, IDX_Y] + dt * state[env, IDX_VY]
        state[env, IDX_Z] = state[env, IDX_Z] + dt * state[env, IDX_VZ]

        # 3. Quaternion integration: q' = q + 0.5*dt*ω⊗q
        var half_dt = Scalar[DTYPE](0.5) * dt
        var wx = state[env, IDX_WX]
        var wy = state[env, IDX_WY]
        var wz = state[env, IDX_WZ]
        var qx = state[env, IDX_QX]
        var qy = state[env, IDX_QY]
        var qz = state[env, IDX_QZ]
        var qw = state[env, IDX_QW]

        # Quaternion derivative using Hamilton product
        state[env, IDX_QX] = qx + half_dt * (wx * qw + wy * qz - wz * qy)
        state[env, IDX_QY] = qy + half_dt * (-wx * qz + wy * qw + wz * qx)
        state[env, IDX_QZ] = qz + half_dt * (wx * qy - wy * qx + wz * qw)
        state[env, IDX_QW] = qw + half_dt * (-wx * qx - wy * qy - wz * qz)

        # 4. Normalize quaternion to prevent drift
        var qx_new = state[env, IDX_QX]
        var qy_new = state[env, IDX_QY]
        var qz_new = state[env, IDX_QZ]
        var qw_new = state[env, IDX_QW]
        var norm_sq = qx_new * qx_new + qy_new * qy_new + qz_new * qz_new + qw_new * qw_new

        if norm_sq > Scalar[DTYPE](1e-10):
            var inv_norm = Scalar[DTYPE](1.0) / sqrt(norm_sq)
            state[env, IDX_QX] = qx_new * inv_norm
            state[env, IDX_QY] = qy_new * inv_norm
            state[env, IDX_QZ] = qz_new * inv_norm
            state[env, IDX_QW] = qw_new * inv_norm

    # =========================================================================
    # Fused Physics Step - Single Environment
    # =========================================================================

    @always_inline
    @staticmethod
    fn _step_single_env[
        DTYPE: DType, BATCH: Int
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        mass: Scalar[DTYPE],
        inv_mass: Scalar[DTYPE],
        inv_ixx: Scalar[DTYPE],
        inv_iyy: Scalar[DTYPE],
        inv_izz: Scalar[DTYPE],
        geom_type: Int,
        radius: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        restitution: Scalar[DTYPE],
        baumgarte: Scalar[DTYPE],
        slop: Scalar[DTYPE],
    ):
        """Complete physics step for one environment.

        Pipeline (Phase 2 with collision):
        1. Update kinematics
        2. Pre-step collision detection
        3. Solve contact constraints
        4. Compute accelerations
        5. Cancel gravity if resting
        6. Integrate
        7. Post-step kinematics
        8. Post-step collision detection
        9. Position correction
        """
        # 1. Update world-frame positions
        Self._update_kinematics[DTYPE, BATCH](env, state)

        # 2. Pre-step collision detection
        Self._detect_sphere_plane[DTYPE, BATCH](
            env, state, radius, ground_z, geom_type
        )

        # 3. Solve contact constraints (velocity impulse)
        Self._solve_contact[DTYPE, BATCH](
            env, state, mass, restitution, baumgarte, slop
        )

        # 4. Compute accelerations
        Self._compute_acceleration[DTYPE, BATCH](
            env, state, gravity_z, inv_mass, inv_ixx, inv_iyy, inv_izz
        )

        # 5. Cancel gravity when resting on ground
        Self._cancel_gravity_if_resting[DTYPE, BATCH](env, state)

        # 6. Integrate velocities and positions
        Self._integrate[DTYPE, BATCH](env, state, dt)

        # 7. Post-integration kinematics
        Self._update_kinematics[DTYPE, BATCH](env, state)

        # 8. Post-step collision detection
        Self._detect_sphere_plane[DTYPE, BATCH](
            env, state, radius, ground_z, geom_type
        )

        # 9. Position correction if penetrating
        Self._solve_contact[DTYPE, BATCH](
            env, state, mass, restitution, baumgarte, slop
        )

    # =========================================================================
    # GPU Kernel Entry Point
    # =========================================================================

    @always_inline
    @staticmethod
    fn _step_kernel[
        DTYPE: DType, BATCH: Int
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        mass: Scalar[DTYPE],
        inv_mass: Scalar[DTYPE],
        inv_ixx: Scalar[DTYPE],
        inv_iyy: Scalar[DTYPE],
        inv_izz: Scalar[DTYPE],
        geom_type: Int,
        radius: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        restitution: Scalar[DTYPE],
        baumgarte: Scalar[DTYPE],
        slop: Scalar[DTYPE],
    ):
        """GPU kernel entry point - runs physics step for all environments."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        Physics3DV2Kernel._step_single_env[DTYPE, BATCH](
            env,
            state,
            dt,
            gravity_z,
            mass,
            inv_mass,
            inv_ixx,
            inv_iyy,
            inv_izz,
            geom_type,
            radius,
            ground_z,
            restitution,
            baumgarte,
            slop,
        )

    # =========================================================================
    # Public GPU API
    # =========================================================================

    @staticmethod
    fn step_gpu[
        DTYPE: DType, BATCH: Int
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        mass: Scalar[DTYPE],
        ixx: Scalar[DTYPE],
        iyy: Scalar[DTYPE],
        izz: Scalar[DTYPE],
        geom_type: Int,
        radius: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        restitution: Scalar[DTYPE] = 0.0,
        baumgarte: Scalar[DTYPE] = 0.2,
        slop: Scalar[DTYPE] = 0.001,
    ) raises:
        """Run the complete physics step on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            dt: Time step.
            gravity_z: Z-component of gravity.
            mass: Body mass.
            ixx, iyy, izz: Diagonal inertia components.
            geom_type: Geometry type (GEOM_SPHERE = 1).
            radius: Sphere radius.
            ground_z: Ground plane height.
            restitution: Coefficient of restitution.
            baumgarte: Position correction factor.
            slop: Penetration allowance.
        """
        # Precompute inverse values
        var inv_mass = Scalar[DTYPE](1.0) / mass
        var inv_ixx = Scalar[DTYPE](1.0) / ixx
        var inv_iyy = Scalar[DTYPE](1.0) / iyy
        var inv_izz = Scalar[DTYPE](1.0) / izz

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            dt: Scalar[DTYPE],
            gravity_z: Scalar[DTYPE],
            mass: Scalar[DTYPE],
            inv_mass: Scalar[DTYPE],
            inv_ixx: Scalar[DTYPE],
            inv_iyy: Scalar[DTYPE],
            inv_izz: Scalar[DTYPE],
            geom_type: Int,
            radius: Scalar[DTYPE],
            ground_z: Scalar[DTYPE],
            restitution: Scalar[DTYPE],
            baumgarte: Scalar[DTYPE],
            slop: Scalar[DTYPE],
        ):
            Physics3DV2Kernel._step_kernel[DTYPE, BATCH](
                state,
                dt,
                gravity_z,
                mass,
                inv_mass,
                inv_ixx,
                inv_iyy,
                inv_izz,
                geom_type,
                radius,
                ground_z,
                restitution,
                baumgarte,
                slop,
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            dt,
            gravity_z,
            mass,
            inv_mass,
            inv_ixx,
            inv_iyy,
            inv_izz,
            geom_type,
            radius,
            ground_z,
            restitution,
            baumgarte,
            slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


# =============================================================================
# Convenience Functions
# =============================================================================


fn step_gpu[
    DTYPE: DType, BATCH: Int
](
    ctx: DeviceContext,
    mut state_buf: DeviceBuffer[DTYPE],
    dt: Scalar[DTYPE],
    gravity_z: Scalar[DTYPE],
    mass: Scalar[DTYPE],
    ixx: Scalar[DTYPE],
    iyy: Scalar[DTYPE],
    izz: Scalar[DTYPE],
    geom_type: Int,
    radius: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
    restitution: Scalar[DTYPE] = 0.0,
    baumgarte: Scalar[DTYPE] = 0.2,
    slop: Scalar[DTYPE] = 0.001,
) raises:
    """Convenience wrapper for Physics3DV2Kernel.step_gpu."""
    Physics3DV2Kernel.step_gpu[DTYPE, BATCH](
        ctx,
        state_buf,
        dt,
        gravity_z,
        mass,
        ixx,
        iyy,
        izz,
        geom_type,
        radius,
        ground_z,
        restitution,
        baumgarte,
        slop,
    )


fn step_gpu_batched[
    DTYPE: DType, BATCH: Int
](
    ctx: DeviceContext,
    mut state_buf: DeviceBuffer[DTYPE],
    num_steps: Int,
    dt: Scalar[DTYPE],
    gravity_z: Scalar[DTYPE],
    mass: Scalar[DTYPE],
    ixx: Scalar[DTYPE],
    iyy: Scalar[DTYPE],
    izz: Scalar[DTYPE],
    geom_type: Int,
    radius: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
    restitution: Scalar[DTYPE] = 0.0,
    baumgarte: Scalar[DTYPE] = 0.2,
    slop: Scalar[DTYPE] = 0.001,
) raises:
    """Run multiple physics steps on GPU.

    Each step is a separate kernel launch, but all environments
    are processed in parallel within each step.
    """
    for _ in range(num_steps):
        step_gpu[DTYPE, BATCH](
            ctx,
            state_buf,
            dt,
            gravity_z,
            mass,
            ixx,
            iyy,
            izz,
            geom_type,
            radius,
            ground_z,
            restitution,
            baumgarte,
            slop,
        )
