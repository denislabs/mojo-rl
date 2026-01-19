"""Semi-Implicit Euler integrator implementation.

This integrator follows Box2D's integration order:
1. v(t+dt) = v(t) + a(t) * dt
2. x(t+dt) = x(t) + v(t+dt) * dt  <- uses NEW velocity

This order is crucial for energy conservation in constrained systems.
"""

from math import cos, sin, pi
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    PI,
    TWO_PI,
)
from ..traits.integrator import Integrator


@always_inline
fn normalize_angle_inplace(
    mut angle: Scalar[dtype],
    pi_val: Scalar[dtype],
    two_pi_val: Scalar[dtype],
    zero: Scalar[dtype],
    extreme: Scalar[dtype],
):
    """Normalize angle to [-pi, pi] range in place."""
    # Handle extreme values
    var neg_extreme = Scalar[dtype](0) - extreme
    if angle > extreme or angle < neg_extreme:
        angle = zero
        return

    var neg_pi_val = Scalar[dtype](0) - pi_val
    # Normalize (bounded iterations for GPU safety)
    if angle > pi_val:
        angle = angle - two_pi_val
    if angle > pi_val:
        angle = angle - two_pi_val
    if angle < neg_pi_val:
        angle = angle + two_pi_val
    if angle < neg_pi_val:
        angle = angle + two_pi_val


struct SemiImplicitEuler(Integrator):
    """Semi-implicit (symplectic) Euler integrator.

    This is the workhorse integrator for game physics, matching Box2D.
    It provides good stability and energy conservation for constrained systems.

    Integration order:
    1. v' = v + (F/m + g) * dt   (velocity update)
    2. x' = x + v' * dt          (position update using NEW velocity)
    """

    fn __init__(out self):
        """Initialize the integrator (stateless, nothing to store)."""
        pass

    # =========================================================================
    # CPU Implementation
    # =========================================================================

    fn integrate_velocities[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        forces: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, 3), MutAnyOrigin
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Integrate velocities: v' = v + (F/m + g) * dt."""
        for env in range(BATCH):
            for body in range(NUM_BODIES):
                # Get inverse mass (0 = static body)
                var inv_mass = bodies[env, body, IDX_INV_MASS]
                var inv_inertia = bodies[env, body, IDX_INV_INERTIA]

                # Skip static bodies
                if inv_mass == Scalar[dtype](0):
                    continue

                # Read current velocities
                var vx = bodies[env, body, IDX_VX]
                var vy = bodies[env, body, IDX_VY]
                var omega = bodies[env, body, IDX_OMEGA]

                # Read forces
                var fx = forces[env, body, 0]
                var fy = forces[env, body, 1]
                var tau = forces[env, body, 2]

                # Integrate: v' = v + a * dt where a = F/m + g
                vx = vx + (fx * inv_mass + gravity_x) * dt
                vy = vy + (fy * inv_mass + gravity_y) * dt
                omega = omega + tau * inv_inertia * dt

                # Write back
                bodies[env, body, IDX_VX] = vx
                bodies[env, body, IDX_VY] = vy
                bodies[env, body, IDX_OMEGA] = omega

    fn integrate_positions[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """Integrate positions: x' = x + v' * dt (using NEW velocity)."""
        for env in range(BATCH):
            for body in range(NUM_BODIES):
                var inv_mass = bodies[env, body, IDX_INV_MASS]

                # Skip static bodies
                if inv_mass == Scalar[dtype](0):
                    continue

                # Read positions and NEW velocities (after integrate_velocities)
                var x = bodies[env, body, IDX_X]
                var y = bodies[env, body, IDX_Y]
                var angle = bodies[env, body, IDX_ANGLE]
                var vx = bodies[env, body, IDX_VX]
                var vy = bodies[env, body, IDX_VY]
                var omega = bodies[env, body, IDX_OMEGA]

                # Integrate positions
                x = x + vx * dt
                y = y + vy * dt
                angle = angle + omega * dt

                # Normalize angle to [-pi, pi]
                var pi_val = Scalar[dtype](PI)
                var two_pi_val = Scalar[dtype](TWO_PI)
                var zero = Scalar[dtype](0.0)
                var extreme = Scalar[dtype](100.0)
                if angle > extreme or angle < -extreme:
                    angle = zero
                elif angle > pi_val:
                    angle = angle - two_pi_val
                    if angle > pi_val:
                        angle = angle - two_pi_val
                elif angle < -pi_val:
                    angle = angle + two_pi_val
                    if angle < -pi_val:
                        angle = angle + two_pi_val

                # Write back
                bodies[env, body, IDX_X] = x
                bodies[env, body, IDX_Y] = y
                bodies[env, body, IDX_ANGLE] = angle

    # =========================================================================
    # GPU Kernel Implementations
    # =========================================================================

    @always_inline
    @staticmethod
    fn integrate_velocities_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        forces: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, 3), MutAnyOrigin
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """GPU kernel: one thread per environment."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        # Process all bodies in this environment sequentially
        # (NUM_BODIES is small for RL, typically 3-10)
        @parameter
        for body in range(NUM_BODIES):
            var inv_mass = bodies[env, body, IDX_INV_MASS]
            var inv_inertia = bodies[env, body, IDX_INV_INERTIA]

            # Skip static bodies
            if inv_mass == Scalar[dtype](0):
                continue

            var vx = bodies[env, body, IDX_VX]
            var vy = bodies[env, body, IDX_VY]
            var omega = bodies[env, body, IDX_OMEGA]

            var fx = forces[env, body, 0]
            var fy = forces[env, body, 1]
            var tau = forces[env, body, 2]

            vx = vx + (fx * inv_mass + gravity_x) * dt
            vy = vy + (fy * inv_mass + gravity_y) * dt
            omega = omega + tau * inv_inertia * dt

            bodies[env, body, IDX_VX] = vx
            bodies[env, body, IDX_VY] = vy
            bodies[env, body, IDX_OMEGA] = omega

    @always_inline
    @staticmethod
    fn integrate_positions_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        dt: Scalar[dtype],
    ):
        """GPU kernel: one thread per environment."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        @parameter
        for body in range(NUM_BODIES):
            var inv_mass = bodies[env, body, IDX_INV_MASS]

            if inv_mass == Scalar[dtype](0):
                continue

            var x = bodies[env, body, IDX_X]
            var y = bodies[env, body, IDX_Y]
            var angle = bodies[env, body, IDX_ANGLE]
            var vx = bodies[env, body, IDX_VX]
            var vy = bodies[env, body, IDX_VY]
            var omega = bodies[env, body, IDX_OMEGA]

            x = x + vx * dt
            y = y + vy * dt
            angle = angle + omega * dt

            # Normalize angle
            var pi_val = Scalar[dtype](PI)
            var two_pi_val = Scalar[dtype](TWO_PI)
            if angle > Scalar[dtype](100.0) or angle < Scalar[dtype](-100.0):
                angle = Scalar[dtype](0.0)
            if angle > pi_val:
                angle = angle - two_pi_val
            if angle > pi_val:
                angle = angle - two_pi_val
            if angle < -pi_val:
                angle = angle + two_pi_val
            if angle < -pi_val:
                angle = angle + two_pi_val

            bodies[env, body, IDX_X] = x
            bodies[env, body, IDX_Y] = y
            bodies[env, body, IDX_ANGLE] = angle

    # =========================================================================
    # GPU Kernel Launchers
    # =========================================================================

    @staticmethod
    fn integrate_velocities_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        forces_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Launch velocity integration kernel on GPU."""
        var bodies = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ](bodies_buf.unsafe_ptr())
        var forces = LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, 3), MutAnyOrigin
        ](forces_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            bodies: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
                MutAnyOrigin,
            ],
            forces: LayoutTensor[
                dtype, Layout.row_major(BATCH, NUM_BODIES, 3), MutAnyOrigin
            ],
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            dt: Scalar[dtype],
        ):
            SemiImplicitEuler.integrate_velocities_kernel[BATCH, NUM_BODIES](
                bodies, forces, gravity_x, gravity_y, dt
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            bodies,
            forces,
            gravity_x,
            gravity_y,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn integrate_positions_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Launch position integration kernel on GPU."""
        var bodies = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ](bodies_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            bodies: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
                MutAnyOrigin,
            ],
            dt: Scalar[dtype],
        ):
            SemiImplicitEuler.integrate_positions_kernel[BATCH, NUM_BODIES](
                bodies, dt
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            bodies,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Strided GPU Kernels for 2D State Layout
    # =========================================================================
    #
    # These methods work with 2D [BATCH, STATE_SIZE] layout where physics
    # data is packed per-environment with offsets.
    #
    # Memory layout: state[env, OFFSET + body * BODY_STATE_SIZE + field]
    # This enables integration with GPUDiscreteEnv trait.
    # =========================================================================

    @always_inline
    @staticmethod
    fn integrate_velocities_kernel_strided[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """GPU kernel for velocity integration with 2D strided layout.

        Args:
            state: State tensor [BATCH, STATE_SIZE].
            gravity_x: Gravity X component.
            gravity_y: Gravity Y component.
            dt: Time step.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        @parameter
        for body in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE
            var force_off = FORCES_OFFSET + body * 3

            var inv_mass = state[env, body_off + IDX_INV_MASS]
            var inv_inertia = state[env, body_off + IDX_INV_INERTIA]

            # Skip static bodies
            if inv_mass == Scalar[dtype](0):
                continue

            var vx = state[env, body_off + IDX_VX]
            var vy = state[env, body_off + IDX_VY]
            var omega = state[env, body_off + IDX_OMEGA]

            var fx = state[env, force_off + 0]
            var fy = state[env, force_off + 1]
            var tau = state[env, force_off + 2]

            vx = vx + (fx * inv_mass + gravity_x) * dt
            vy = vy + (fy * inv_mass + gravity_y) * dt
            omega = omega + tau * inv_inertia * dt

            state[env, body_off + IDX_VX] = vx
            state[env, body_off + IDX_VY] = vy
            state[env, body_off + IDX_OMEGA] = omega

    @always_inline
    @staticmethod
    fn integrate_positions_kernel_strided[
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
        """GPU kernel for position integration with 2D strided layout.

        Args:
            state: State tensor [BATCH, STATE_SIZE].
            dt: Time step.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        @parameter
        for body in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + body * BODY_STATE_SIZE

            var inv_mass = state[env, body_off + IDX_INV_MASS]

            if inv_mass == Scalar[dtype](0):
                continue

            var x = state[env, body_off + IDX_X]
            var y = state[env, body_off + IDX_Y]
            var angle = state[env, body_off + IDX_ANGLE]
            var vx = state[env, body_off + IDX_VX]
            var vy = state[env, body_off + IDX_VY]
            var omega = state[env, body_off + IDX_OMEGA]

            x = x + vx * dt
            y = y + vy * dt
            angle = angle + omega * dt

            # Normalize angle
            var pi_val = Scalar[dtype](PI)
            var two_pi_val = Scalar[dtype](TWO_PI)
            if angle > Scalar[dtype](100.0) or angle < Scalar[dtype](-100.0):
                angle = Scalar[dtype](0.0)
            if angle > pi_val:
                angle = angle - two_pi_val
            if angle > pi_val:
                angle = angle - two_pi_val
            if angle < -pi_val:
                angle = angle + two_pi_val
            if angle < -pi_val:
                angle = angle + two_pi_val

            state[env, body_off + IDX_X] = x
            state[env, body_off + IDX_Y] = y
            state[env, body_off + IDX_ANGLE] = angle

    @staticmethod
    fn integrate_velocities_gpu_strided[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Launch strided velocity integration kernel on GPU.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            gravity_x: Gravity X component.
            gravity_y: Gravity Y component.
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
            dt: Scalar[dtype],
        ):
            SemiImplicitEuler.integrate_velocities_kernel_strided[
                BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
            ](state, gravity_x, gravity_y, dt)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            gravity_x,
            gravity_y,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn integrate_positions_gpu_strided[
        BATCH: Int,
        NUM_BODIES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Launch strided position integration kernel on GPU.

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
            SemiImplicitEuler.integrate_positions_kernel_strided[
                BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
            ](state, dt)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
