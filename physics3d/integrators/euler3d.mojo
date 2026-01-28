"""Semi-Implicit Euler Integrator for 3D Physics.

Integrates velocities and positions for 3D rigid bodies, including
quaternion integration for rotations.

GPU support follows the three-method hierarchy pattern:
1. *_single_env - @always_inline @staticmethod, inline core logic
2. *_kernel - GPU kernel entry point, computes env from thread ID
3. *_gpu - Public API, creates LayoutTensor views, launches kernel
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from math3d import Vec3, Quat
from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_FX,
    IDX_FY,
    IDX_FZ,
    IDX_TX,
    IDX_TY,
    IDX_TZ,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    DEFAULT_GRAVITY_Z_3D,
)


fn integrate_quaternion[
    DTYPE: DType
](q: Quat[DTYPE], omega: Vec3[DTYPE], dt: Scalar[DTYPE]) -> Quat[DTYPE]:
    """Integrate quaternion with angular velocity.

    Uses the standard quaternion integration formula:
    q(t+dt) = q(t) + 0.5 * dt * omega * q(t)

    where omega * q is the quaternion product of (0, omega) and q.

    Args:
        q: Current orientation quaternion.
        omega: Angular velocity vector (world space).
        dt: Time step.

    Returns:
        New orientation quaternion (normalized).
    """
    # Convert angular velocity to quaternion: (0, omega_x, omega_y, omega_z)
    var omega_quat = Quat[DTYPE](0.0, omega.x, omega.y, omega.z)

    # q_dot = 0.5 * omega_quat * q
    var q_dot = omega_quat * q
    q_dot.w *= 0.5
    q_dot.x *= 0.5
    q_dot.y *= 0.5
    q_dot.z *= 0.5

    # Integrate: q_new = q + dt * q_dot
    var q_new = Quat(
        q.w + dt * q_dot.w,
        q.x + dt * q_dot.x,
        q.y + dt * q_dot.y,
        q.z + dt * q_dot.z,
    )

    # Normalize to maintain unit quaternion
    return q_new.normalized()


fn integrate_velocities_3d[
    DTYPE: DType
](
    mut state: List[Scalar[DTYPE]],
    body_idx: Int,
    gravity: Vec3[DTYPE],
    dt: Scalar[DTYPE],
    bodies_offset: Int = 0,
):
    """Integrate velocities with forces and gravity.

    Updates linear and angular velocities based on accumulated forces/torques.

    Args:
        state: Physics state array (modified in place).
        body_idx: Index of body in state array.
        gravity: Gravity acceleration vector.
        dt: Time step.
        bodies_offset: Offset to first body in state array.
    """
    var base = bodies_offset + body_idx * BODY_STATE_SIZE_3D

    # Check if body is dynamic
    var body_type = Int(state[base + IDX_BODY_TYPE])
    if body_type != BODY_DYNAMIC:
        return

    # Get inverse mass and inertia
    var inv_mass = state[base + IDX_INV_MASS]
    if inv_mass <= 0.0:
        return

    var inv_ixx = (
        1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
    )
    var inv_iyy = (
        1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
    )
    var inv_izz = (
        1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0
    )

    # Get current forces and torques
    var fx = state[base + IDX_FX]
    var fy = state[base + IDX_FY]
    var fz = state[base + IDX_FZ]
    var tx = state[base + IDX_TX]
    var ty = state[base + IDX_TY]
    var tz = state[base + IDX_TZ]

    # Add gravity force (F = m * g)
    var mass = state[base + IDX_MASS]
    fx += mass * gravity.x
    fy += mass * gravity.y
    fz += mass * gravity.z

    # Integrate linear velocity: v += dt * (F / m)
    state[base + IDX_VX] += dt * fx * inv_mass
    state[base + IDX_VY] += dt * fy * inv_mass
    state[base + IDX_VZ] += dt * fz * inv_mass

    # Integrate angular velocity: w += dt * (I^-1 * tau)
    # Note: Using diagonal inertia approximation
    state[base + IDX_WX] += dt * tx * inv_ixx
    state[base + IDX_WY] += dt * ty * inv_iyy
    state[base + IDX_WZ] += dt * tz * inv_izz

    # Clear force/torque accumulators
    state[base + IDX_FX] = 0.0
    state[base + IDX_FY] = 0.0
    state[base + IDX_FZ] = 0.0
    state[base + IDX_TX] = 0.0
    state[base + IDX_TY] = 0.0
    state[base + IDX_TZ] = 0.0


fn integrate_positions_3d[
    DTYPE: DType
](
    mut state: List[Scalar[DTYPE]],
    body_idx: Int,
    dt: Scalar[DTYPE],
    bodies_offset: Int = 0,
):
    """Integrate positions with velocities.

    Updates position and orientation based on linear and angular velocities.

    Args:
        state: Physics state array (modified in place).
        body_idx: Index of body in state array.
        dt: Time step.
        bodies_offset: Offset to first body in state array.
    """
    var base = bodies_offset + body_idx * BODY_STATE_SIZE_3D

    # Check if body is dynamic
    var body_type = Int(state[base + IDX_BODY_TYPE])
    if body_type != BODY_DYNAMIC:
        return

    # Integrate linear position: x += dt * v
    state[base + IDX_PX] += dt * state[base + IDX_VX]
    state[base + IDX_PY] += dt * state[base + IDX_VY]
    state[base + IDX_PZ] += dt * state[base + IDX_VZ]

    # Integrate orientation using quaternion
    var q = Quat(
        state[base + IDX_QW],
        state[base + IDX_QX],
        state[base + IDX_QY],
        state[base + IDX_QZ],
    )
    var omega = Vec3(
        state[base + IDX_WX],
        state[base + IDX_WY],
        state[base + IDX_WZ],
    )

    var q_new = integrate_quaternion(q, omega, dt)

    # Store normalized quaternion
    state[base + IDX_QW] = q_new.w
    state[base + IDX_QX] = q_new.x
    state[base + IDX_QY] = q_new.y
    state[base + IDX_QZ] = q_new.z


struct SemiImplicitEuler3D[DTYPE: DType]:
    """Semi-implicit Euler integrator for 3D rigid body dynamics.

    This integrator:
    1. Updates velocities with forces/gravity
    2. Updates positions with new velocities (semi-implicit)

    The semi-implicit scheme is more stable than explicit Euler
    while being simple and fast.
    """

    var gravity: Vec3[Self.DTYPE]
    var dt: Scalar[Self.DTYPE]
    var damping_linear: Scalar[Self.DTYPE]
    var damping_angular: Scalar[Self.DTYPE]
    var bodies_offset: Int

    fn __init__(
        out self,
        gravity: Vec3[Self.DTYPE] = Vec3[Self.DTYPE](
            0.0, 0.0, Scalar[Self.DTYPE](DEFAULT_GRAVITY_Z_3D)
        ),
        dt: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](1.0 / 60.0),
        damping_linear: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](0.01),
        damping_angular: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](0.01),
        bodies_offset: Int = 0,
    ):
        """Initialize the integrator.

        Args:
            gravity: Gravity acceleration vector.
            dt: Time step.
            damping_linear: Linear velocity damping (0-1).
            damping_angular: Angular velocity damping (0-1).
            bodies_offset: Offset to first body in state array.
        """
        self.gravity = gravity
        self.dt = dt
        self.damping_linear = damping_linear
        self.damping_angular = damping_angular
        self.bodies_offset = bodies_offset

    fn step(self, mut state: List[Scalar[Self.DTYPE]], num_bodies: Int):
        """Perform one integration step for all bodies.

        Args:
            state: Physics state array (modified in place).
            num_bodies: Number of bodies in state.
        """
        for i in range(num_bodies):
            self._step_body(state, i)

    fn _step_body(self, mut state: List[Scalar[Self.DTYPE]], body_idx: Int):
        """Integrate a single body.

        Args:
            state: Physics state array.
            body_idx: Index of body.
        """
        var base = self.bodies_offset + body_idx * BODY_STATE_SIZE_3D

        # Check if body is dynamic
        var body_type = Int(state[base + IDX_BODY_TYPE])
        if body_type != BODY_DYNAMIC:
            return

        # Step 1: Integrate velocities
        integrate_velocities_3d(state, body_idx, self.gravity, self.dt, self.bodies_offset)

        # Apply damping
        var damp_lin = 1.0 - self.damping_linear
        var damp_ang = 1.0 - self.damping_angular

        state[base + IDX_VX] *= damp_lin
        state[base + IDX_VY] *= damp_lin
        state[base + IDX_VZ] *= damp_lin
        state[base + IDX_WX] *= damp_ang
        state[base + IDX_WY] *= damp_ang
        state[base + IDX_WZ] *= damp_ang

        # Step 2: Integrate positions
        integrate_positions_3d(state, body_idx, self.dt, self.bodies_offset)

    fn apply_impulse(
        self,
        mut state: List[Scalar[Self.DTYPE]],
        body_idx: Int,
        impulse: Vec3[Self.DTYPE],
        point: Vec3[Self.DTYPE],
    ):
        """Apply an impulse to a body at a world-space point.

        Args:
            state: Physics state array.
            body_idx: Index of body.
            impulse: Impulse vector (world space).
            point: Application point (world space).
        """
        var base = self.bodies_offset + body_idx * BODY_STATE_SIZE_3D

        # Check if body is dynamic
        var body_type = Int(state[base + IDX_BODY_TYPE])
        if body_type != BODY_DYNAMIC:
            return

        var inv_mass = state[base + IDX_INV_MASS]
        if inv_mass <= 0.0:
            return

        # Get body position
        var pos = Vec3(
            state[base + IDX_PX],
            state[base + IDX_PY],
            state[base + IDX_PZ],
        )

        # Linear velocity change: dv = impulse / m
        state[base + IDX_VX] += impulse.x * inv_mass
        state[base + IDX_VY] += impulse.y * inv_mass
        state[base + IDX_VZ] += impulse.z * inv_mass

        # Angular velocity change: dw = I^-1 * (r x impulse)
        var r = point - pos
        var torque_impulse = r.cross(impulse)

        var inv_ixx = (
            1.0 / state[base + IDX_IXX] if state[base + IDX_IXX] > 0.0 else 0.0
        )
        var inv_iyy = (
            1.0 / state[base + IDX_IYY] if state[base + IDX_IYY] > 0.0 else 0.0
        )
        var inv_izz = (
            1.0 / state[base + IDX_IZZ] if state[base + IDX_IZZ] > 0.0 else 0.0
        )

        state[base + IDX_WX] += torque_impulse.x * inv_ixx
        state[base + IDX_WY] += torque_impulse.y * inv_iyy
        state[base + IDX_WZ] += torque_impulse.z * inv_izz

    fn get_velocity_at_point(
        self,
        state: List[Scalar[Self.DTYPE]],
        body_idx: Int,
        point: Vec3[Self.DTYPE],
    ) -> Vec3[Self.DTYPE]:
        """Get velocity of body at a world-space point.

        Args:
            state: Physics state array.
            body_idx: Index of body.
            point: Query point (world space).

        Returns:
            Velocity at point (includes angular contribution).
        """
        var base = self.bodies_offset + body_idx * BODY_STATE_SIZE_3D

        var pos = Vec3(
            state[base + IDX_PX],
            state[base + IDX_PY],
            state[base + IDX_PZ],
        )
        var vel = Vec3(
            state[base + IDX_VX],
            state[base + IDX_VY],
            state[base + IDX_VZ],
        )
        var omega = Vec3(
            state[base + IDX_WX],
            state[base + IDX_WY],
            state[base + IDX_WZ],
        )

        # v_point = v_cm + omega x r
        var r = point - pos
        return vel + omega.cross(r)


# =============================================================================
# GPU Implementation
# =============================================================================


struct SemiImplicitEuler3DGPU:
    """GPU-compatible Semi-Implicit Euler integrator for 3D rigid body dynamics.

    This integrator provides GPU kernel methods following the three-method hierarchy:
    1. *_single_env - Core logic callable from fused kernels
    2. *_kernel - GPU kernel entry point
    3. *_gpu - Public API for standalone kernel launches

    Integration order (matches Box2D):
    1. v' = v + (F/m + g) * dt   (velocity update)
    2. x' = x + v' * dt          (position update using NEW velocity)
    """

    # =========================================================================
    # Single-Environment Methods (can be called from fused kernels)
    # =========================================================================

    @always_inline
    @staticmethod
    fn integrate_velocities_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        gravity_z: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Integrate velocities for a single environment.

        Updates linear and angular velocities based on accumulated forces/torques.
        Clears force/torque accumulators after integration.

        Args:
            env: Environment index.
            state: State buffer [BATCH, STATE_SIZE].
            gravity_x, gravity_y, gravity_z: Gravity components.
            dt: Time step.
        """

        @parameter
        for body in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE_3D

            # Check if body is dynamic
            var body_type = Int(state[env, body_off + IDX_BODY_TYPE])
            if body_type != BODY_DYNAMIC:
                continue

            # Get inverse mass (0 = static body)
            var inv_mass = state[env, body_off + IDX_INV_MASS]
            if inv_mass <= Scalar[dtype](0):
                continue

            # Get inverse inertia (diagonal approximation)
            var ixx = state[env, body_off + IDX_IXX]
            var iyy = state[env, body_off + IDX_IYY]
            var izz = state[env, body_off + IDX_IZZ]

            var inv_ixx: state.element_type = 0
            var inv_iyy: state.element_type = 0
            var inv_izz: state.element_type = 0
            if ixx > Scalar[dtype](1e-10):
                inv_ixx = Scalar[dtype](1.0) / ixx
            if iyy > Scalar[dtype](1e-10):
                inv_iyy = Scalar[dtype](1.0) / iyy
            if izz > Scalar[dtype](1e-10):
                inv_izz = Scalar[dtype](1.0) / izz

            # Get current velocities
            var vx = state[env, body_off + IDX_VX]
            var vy = state[env, body_off + IDX_VY]
            var vz = state[env, body_off + IDX_VZ]
            var wx = state[env, body_off + IDX_WX]
            var wy = state[env, body_off + IDX_WY]
            var wz = state[env, body_off + IDX_WZ]

            # Get forces and torques
            var fx = state[env, body_off + IDX_FX]
            var fy = state[env, body_off + IDX_FY]
            var fz = state[env, body_off + IDX_FZ]
            var tx = state[env, body_off + IDX_TX]
            var ty = state[env, body_off + IDX_TY]
            var tz = state[env, body_off + IDX_TZ]

            # Add gravity force (F = m * g, but we use F/m = g * inv_mass * mass = g)
            # Linear velocity: v' = v + (F/m + g) * dt
            vx = vx + (fx * inv_mass + gravity_x) * dt
            vy = vy + (fy * inv_mass + gravity_y) * dt
            vz = vz + (fz * inv_mass + gravity_z) * dt

            # Angular velocity: w' = w + I^-1 * tau * dt (diagonal inertia)
            wx = wx + tx * inv_ixx * dt
            wy = wy + ty * inv_iyy * dt
            wz = wz + tz * inv_izz * dt

            # Write back velocities
            state[env, body_off + IDX_VX] = vx
            state[env, body_off + IDX_VY] = vy
            state[env, body_off + IDX_VZ] = vz
            state[env, body_off + IDX_WX] = wx
            state[env, body_off + IDX_WY] = wy
            state[env, body_off + IDX_WZ] = wz

            # Clear force/torque accumulators
            state[env, body_off + IDX_FX] = Scalar[dtype](0)
            state[env, body_off + IDX_FY] = Scalar[dtype](0)
            state[env, body_off + IDX_FZ] = Scalar[dtype](0)
            state[env, body_off + IDX_TX] = Scalar[dtype](0)
            state[env, body_off + IDX_TY] = Scalar[dtype](0)
            state[env, body_off + IDX_TZ] = Scalar[dtype](0)

    @always_inline
    @staticmethod
    fn integrate_positions_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """Integrate positions for a single environment.

        Updates position and orientation using the NEW velocities
        (after integrate_velocities_single_env).

        Args:
            env: Environment index.
            state: State buffer [BATCH, STATE_SIZE].
            dt: Time step.
        """

        @parameter
        for body in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE_3D

            # Check if body is dynamic
            var body_type = Int(state[env, body_off + IDX_BODY_TYPE])
            if body_type != BODY_DYNAMIC:
                continue

            var inv_mass = state[env, body_off + IDX_INV_MASS]
            if inv_mass <= Scalar[dtype](0):
                continue

            # Get current position
            var px = state[env, body_off + IDX_PX]
            var py = state[env, body_off + IDX_PY]
            var pz = state[env, body_off + IDX_PZ]

            # Get NEW velocities (after velocity integration)
            var vx = state[env, body_off + IDX_VX]
            var vy = state[env, body_off + IDX_VY]
            var vz = state[env, body_off + IDX_VZ]
            var wx = state[env, body_off + IDX_WX]
            var wy = state[env, body_off + IDX_WY]
            var wz = state[env, body_off + IDX_WZ]

            # Integrate linear position: x' = x + v * dt
            px = px + vx * dt
            py = py + vy * dt
            pz = pz + vz * dt

            # Write back position
            state[env, body_off + IDX_PX] = px
            state[env, body_off + IDX_PY] = py
            state[env, body_off + IDX_PZ] = pz

            # Integrate quaternion orientation
            # q' = q + 0.5 * dt * omega_quat * q
            # where omega_quat = (0, wx, wy, wz)
            var qw = state[env, body_off + IDX_QW]
            var qx = state[env, body_off + IDX_QX]
            var qy = state[env, body_off + IDX_QY]
            var qz = state[env, body_off + IDX_QZ]

            # Quaternion multiplication: omega_quat * q
            # (0, wx, wy, wz) * (qw, qx, qy, qz)
            var half_dt = Scalar[dtype](0.5) * dt
            var dqw = half_dt * (-wx * qx - wy * qy - wz * qz)
            var dqx = half_dt * (wx * qw + wy * qz - wz * qy)
            var dqy = half_dt * (wy * qw + wz * qx - wx * qz)
            var dqz = half_dt * (wz * qw + wx * qy - wy * qx)

            # Update quaternion
            qw = qw + dqw
            qx = qx + dqx
            qy = qy + dqy
            qz = qz + dqz

            # Normalize quaternion
            var len_sq = qw * qw + qx * qx + qy * qy + qz * qz
            if len_sq > Scalar[dtype](1e-10):
                var inv_len = Scalar[dtype](1.0) / sqrt(len_sq)
                qw = qw * inv_len
                qx = qx * inv_len
                qy = qy * inv_len
                qz = qz * inv_len

            # Write back normalized quaternion
            state[env, body_off + IDX_QW] = qw
            state[env, body_off + IDX_QX] = qx
            state[env, body_off + IDX_QY] = qy
            state[env, body_off + IDX_QZ] = qz

    # =========================================================================
    # GPU Kernel Entry Points
    # =========================================================================

    @always_inline
    @staticmethod
    fn integrate_velocities_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        gravity_z: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """GPU kernel for velocity integration with 2D strided layout."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        SemiImplicitEuler3DGPU.integrate_velocities_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](env, state, gravity_x, gravity_y, gravity_z, dt)

    @always_inline
    @staticmethod
    fn integrate_positions_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """GPU kernel for position integration with 2D strided layout."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        SemiImplicitEuler3DGPU.integrate_positions_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](env, state, dt)

    # =========================================================================
    # Public GPU API
    # =========================================================================

    @staticmethod
    fn integrate_velocities_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        gravity_z: Scalar[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Launch velocity integration kernel on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            gravity_x, gravity_y, gravity_z: Gravity components.
            dt: Time step.
        """
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ](state_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, STATE_SIZE),
                MutAnyOrigin,
            ],
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            gravity_z: Scalar[dtype],
            dt: Scalar[dtype],
        ):
            SemiImplicitEuler3DGPU.integrate_velocities_kernel[
                BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
            ](state, gravity_x, gravity_y, gravity_z, dt)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            gravity_x,
            gravity_y,
            gravity_z,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn integrate_positions_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Launch position integration kernel on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            dt: Time step.
        """
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ](state_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, STATE_SIZE),
                MutAnyOrigin,
            ],
            dt: Scalar[dtype],
        ):
            SemiImplicitEuler3DGPU.integrate_positions_kernel[
                BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
            ](state, dt)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
