"""PhysicsStateStrided - Helper for accessing physics data in flat strided layout.

This module provides a helper struct that gives PhysicsState-like accessors
for physics data stored in a flat [BATCH, STATE_SIZE] layout (as required by GPUDiscreteEnv).

Unlike PhysicsState which uses structured [BATCH, NUM_BODIES, BODY_STATE_SIZE] tensors,
PhysicsStateStrided works with a flat buffer where each environment's data is contiguous:

    state[env * ENV_STRIDE + OFFSET + body * BODY_STATE_SIZE + field]

This is compatible with the strided GPU methods in:
- SemiImplicitEuler.integrate_velocities_gpu_strided
- SemiImplicitEuler.integrate_positions_gpu_strided
- ImpulseSolver.solve_velocity_gpu_strided
- ImpulseSolver.solve_position_gpu_strided
- RevoluteJointSolver.solve_velocity_gpu_strided
- RevoluteJointSolver.solve_position_gpu_strided
- EdgeTerrainCollision.detect_gpu_strided

Example:
    ```mojo
    from physics_gpu.state_strided import PhysicsStateStrided

    # Define layout constants
    alias NUM_BODIES = 3
    alias ENV_STRIDE = 192
    alias BODIES_OFFSET = 8

    # Create helper for a specific environment
    var physics = PhysicsStateStrided[NUM_BODIES, ENV_STRIDE, BODIES_OFFSET](states, env)

    # Use PhysicsState-like interface
    physics.set_body_position(0, 5.0, 10.0)
    var x = physics.get_body_x(0)
    ```
"""

from layout import LayoutTensor, Layout

from .constants import (
    dtype,
    BODY_STATE_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_STIFFNESS,
    JOINT_DAMPING,
    JOINT_FLAGS,
    JOINT_REVOLUTE,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_SPRING_ENABLED,
)


struct PhysicsStateStrided[
    NUM_BODIES: Int,
    ENV_STRIDE: Int,
    BODIES_OFFSET: Int,
    FORCES_OFFSET: Int = 0,
    JOINTS_OFFSET: Int = 0,
    JOINT_COUNT_OFFSET: Int = 0,
    EDGES_OFFSET: Int = 0,
    EDGE_COUNT_OFFSET: Int = 0,
    MAX_JOINTS: Int = 8,
    FORCES_PER_BODY: Int = 3,
]:
    """Helper for accessing physics data in flat strided layout.

    This provides PhysicsState-like accessors for data stored in a flat buffer
    where each environment's state is contiguous.

    Parameters:
        NUM_BODIES: Number of bodies per environment.
        ENV_STRIDE: Total size of each environment's state.
        BODIES_OFFSET: Offset to body data within environment state.
        FORCES_OFFSET: Offset to forces data within environment state.
        JOINTS_OFFSET: Offset to joints data within environment state.
        JOINT_COUNT_OFFSET: Offset to joint count within environment state.
        EDGES_OFFSET: Offset to edge terrain data within environment state.
        EDGE_COUNT_OFFSET: Offset to edge count within environment state.
        MAX_JOINTS: Maximum joints per environment.
        FORCES_PER_BODY: Number of force values per body (fx, fy, torque).
    """

    var env_base: Int

    fn __init__(out self, env: Int):
        """Create helper for accessing a specific environment's physics data.

        Args:
            env: Environment index.
        """
        self.env_base = env * Self.ENV_STRIDE

    # =========================================================================
    # Body Position Accessors
    # =========================================================================

    @always_inline
    fn body_offset(self, body: Int) -> Int:
        """Get offset for a body's data."""
        return self.env_base + Self.BODIES_OFFSET + body * BODY_STATE_SIZE

    @always_inline
    fn set_body_position[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
        x: Float64,
        y: Float64,
    ):
        """Set body position."""
        var offset = self.body_offset(body)
        states[offset + IDX_X] = Scalar[dtype](x)
        states[offset + IDX_Y] = Scalar[dtype](y)

    @always_inline
    fn set_body_angle[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
        angle: Float64,
    ):
        """Set body angle."""
        var offset = self.body_offset(body)
        states[offset + IDX_ANGLE] = Scalar[dtype](angle)

    @always_inline
    fn set_body_velocity[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
        vx: Float64,
        vy: Float64,
        omega: Float64,
    ):
        """Set body linear and angular velocity."""
        var offset = self.body_offset(body)
        states[offset + IDX_VX] = Scalar[dtype](vx)
        states[offset + IDX_VY] = Scalar[dtype](vy)
        states[offset + IDX_OMEGA] = Scalar[dtype](omega)

    @always_inline
    fn set_body_mass[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
        mass: Float64,
        inertia: Float64,
    ):
        """Set body mass properties. Use mass=0 for static bodies."""
        var offset = self.body_offset(body)
        states[offset + IDX_MASS] = Scalar[dtype](mass)
        if mass > 0:
            states[offset + IDX_INV_MASS] = Scalar[dtype](1.0 / mass)
            if inertia > 0:
                states[offset + IDX_INV_INERTIA] = Scalar[dtype](1.0 / inertia)
            else:
                states[offset + IDX_INV_INERTIA] = Scalar[dtype](0)
        else:
            states[offset + IDX_INV_MASS] = Scalar[dtype](0)
            states[offset + IDX_INV_INERTIA] = Scalar[dtype](0)

    @always_inline
    fn set_body_shape[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
        shape_idx: Int,
    ):
        """Set body shape reference."""
        var offset = self.body_offset(body)
        states[offset + IDX_SHAPE] = Scalar[dtype](shape_idx)

    @always_inline
    fn get_body_x[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body x position."""
        var offset = self.body_offset(body)
        return rebind[Scalar[dtype]](states[offset + IDX_X])

    @always_inline
    fn get_body_y[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body y position."""
        var offset = self.body_offset(body)
        return rebind[Scalar[dtype]](states[offset + IDX_Y])

    @always_inline
    fn get_body_angle[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body angle."""
        var offset = self.body_offset(body)
        return rebind[Scalar[dtype]](states[offset + IDX_ANGLE])

    @always_inline
    fn get_body_vx[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body x velocity."""
        var offset = self.body_offset(body)
        return rebind[Scalar[dtype]](states[offset + IDX_VX])

    @always_inline
    fn get_body_vy[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body y velocity."""
        var offset = self.body_offset(body)
        return rebind[Scalar[dtype]](states[offset + IDX_VY])

    @always_inline
    fn get_body_omega[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body angular velocity."""
        var offset = self.body_offset(body)
        return rebind[Scalar[dtype]](states[offset + IDX_OMEGA])

    # =========================================================================
    # Force Application
    # =========================================================================

    @always_inline
    fn force_offset(self, body: Int) -> Int:
        """Get offset for a body's forces."""
        return self.env_base + Self.FORCES_OFFSET + body * Self.FORCES_PER_BODY

    @always_inline
    fn clear_forces[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    ):
        """Clear all forces for this environment."""
        for body in range(Self.NUM_BODIES):
            var offset = self.force_offset(body)
            states[offset + 0] = Scalar[dtype](0)
            states[offset + 1] = Scalar[dtype](0)
            states[offset + 2] = Scalar[dtype](0)

    @always_inline
    fn apply_force[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
        fx: Float64,
        fy: Float64,
    ):
        """Apply a force to a body (accumulated until step)."""
        var offset = self.force_offset(body)
        states[offset + 0] = states[offset + 0] + Scalar[dtype](fx)
        states[offset + 1] = states[offset + 1] + Scalar[dtype](fy)

    @always_inline
    fn apply_torque[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
        torque: Float64,
    ):
        """Apply a torque to a body (accumulated until step)."""
        var offset = self.force_offset(body)
        states[offset + 2] = states[offset + 2] + Scalar[dtype](torque)

    @always_inline
    fn set_force[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body: Int,
        fx: Float64,
        fy: Float64,
        torque: Float64,
    ):
        """Set force directly (replaces existing)."""
        var offset = self.force_offset(body)
        states[offset + 0] = Scalar[dtype](fx)
        states[offset + 1] = Scalar[dtype](fy)
        states[offset + 2] = Scalar[dtype](torque)

    # =========================================================================
    # Joint Management
    # =========================================================================

    @always_inline
    fn joint_offset(self, joint: Int) -> Int:
        """Get offset for a joint's data."""
        return self.env_base + Self.JOINTS_OFFSET + joint * JOINT_DATA_SIZE

    @always_inline
    fn get_joint_count[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    ) -> Int:
        """Get number of active joints."""
        return Int(states[self.env_base + Self.JOINT_COUNT_OFFSET])

    @always_inline
    fn set_joint_count[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        count: Int,
    ):
        """Set joint count."""
        states[self.env_base + Self.JOINT_COUNT_OFFSET] = Scalar[dtype](count)

    @always_inline
    fn add_revolute_joint[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        body_a: Int,
        body_b: Int,
        anchor_ax: Float64,
        anchor_ay: Float64,
        anchor_bx: Float64,
        anchor_by: Float64,
        stiffness: Float64 = 0.0,
        damping: Float64 = 0.0,
        lower_limit: Float64 = 0.0,
        upper_limit: Float64 = 0.0,
        enable_limit: Bool = False,
    ) -> Int:
        """Add a revolute joint between two bodies.

        Returns:
            Joint index, or -1 if max joints reached.
        """
        var joint_idx = self.get_joint_count(states)
        if joint_idx >= Self.MAX_JOINTS:
            return -1

        var offset = self.joint_offset(joint_idx)

        # Set joint type
        states[offset + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)

        # Set body indices
        states[offset + JOINT_BODY_A] = Scalar[dtype](body_a)
        states[offset + JOINT_BODY_B] = Scalar[dtype](body_b)

        # Set local anchors
        states[offset + JOINT_ANCHOR_AX] = Scalar[dtype](anchor_ax)
        states[offset + JOINT_ANCHOR_AY] = Scalar[dtype](anchor_ay)
        states[offset + JOINT_ANCHOR_BX] = Scalar[dtype](anchor_bx)
        states[offset + JOINT_ANCHOR_BY] = Scalar[dtype](anchor_by)

        # Compute reference angle (angle_b - angle_a at creation)
        var angle_a = self.get_body_angle(states, body_a)
        var angle_b = self.get_body_angle(states, body_b)
        states[offset + JOINT_REF_ANGLE] = angle_b - angle_a

        # Set angle limits
        states[offset + JOINT_LOWER_LIMIT] = Scalar[dtype](lower_limit)
        states[offset + JOINT_UPPER_LIMIT] = Scalar[dtype](upper_limit)

        # Set spring properties
        states[offset + JOINT_STIFFNESS] = Scalar[dtype](stiffness)
        states[offset + JOINT_DAMPING] = Scalar[dtype](damping)

        # Set flags
        var flags = 0
        if stiffness > 0.0 or damping > 0.0:
            flags = flags | JOINT_FLAG_SPRING_ENABLED
        if enable_limit:
            flags = flags | JOINT_FLAG_LIMIT_ENABLED
        states[offset + JOINT_FLAGS] = Scalar[dtype](flags)

        # Increment count
        self.set_joint_count(states, joint_idx + 1)

        return joint_idx

    @always_inline
    fn clear_joints[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    ):
        """Clear all joints for this environment."""
        for j in range(Self.MAX_JOINTS):
            var offset = self.joint_offset(j)
            for i in range(JOINT_DATA_SIZE):
                states[offset + i] = Scalar[dtype](0)
        self.set_joint_count(states, 0)

    # =========================================================================
    # Edge Terrain
    # =========================================================================

    @always_inline
    fn edge_offset(self, edge: Int) -> Int:
        """Get offset for an edge's data (6 floats: x0, y0, x1, y1, nx, ny)."""
        return self.env_base + Self.EDGES_OFFSET + edge * 6

    @always_inline
    fn get_edge_count[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    ) -> Int:
        """Get number of active edges."""
        return Int(states[self.env_base + Self.EDGE_COUNT_OFFSET])

    @always_inline
    fn set_edge_count[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        count: Int,
    ):
        """Set edge count."""
        states[self.env_base + Self.EDGE_COUNT_OFFSET] = Scalar[dtype](count)

    @always_inline
    fn set_edge[
        TOTAL_SIZE: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        edge: Int,
        x0: Float64,
        y0: Float64,
        x1: Float64,
        y1: Float64,
        nx: Float64,
        ny: Float64,
    ):
        """Set edge data."""
        var offset = self.edge_offset(edge)
        states[offset + 0] = Scalar[dtype](x0)
        states[offset + 1] = Scalar[dtype](y0)
        states[offset + 2] = Scalar[dtype](x1)
        states[offset + 3] = Scalar[dtype](y1)
        states[offset + 4] = Scalar[dtype](nx)
        states[offset + 5] = Scalar[dtype](ny)
