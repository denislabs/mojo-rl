"""PhysicsStateStrided - Helper for accessing physics data in 2D strided layout.

This module provides a helper struct that gives PhysicsState-like accessors
for physics data stored in a 2D [BATCH, STATE_SIZE] layout (as required by GPUDiscreteEnv).

Unlike PhysicsState which uses structured [BATCH, NUM_BODIES, BODY_STATE_SIZE] tensors,
PhysicsStateStrided works with a 2D buffer where the second dimension is the state:

    state[env, OFFSET + body * BODY_STATE_SIZE + field]

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
    alias STATE_SIZE = 192
    alias BODIES_OFFSET = 8

    # Create helper for a specific environment
    var physics = PhysicsStateStrided[NUM_BODIES, STATE_SIZE, BODIES_OFFSET](env)

    # Use PhysicsState-like interface with 2D tensor
    physics.set_body_position(states, 0, 5.0, 10.0)
    var x = physics.get_body_x(states, 0)
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
    STATE_SIZE: Int,
    BODIES_OFFSET: Int,
    FORCES_OFFSET: Int = 0,
    JOINTS_OFFSET: Int = 0,
    JOINT_COUNT_OFFSET: Int = 0,
    EDGES_OFFSET: Int = 0,
    EDGE_COUNT_OFFSET: Int = 0,
    MAX_JOINTS: Int = 8,
    FORCES_PER_BODY: Int = 3,
]:
    """Helper for accessing physics data in 2D strided layout.

    This provides PhysicsState-like accessors for data stored in a 2D [BATCH, STATE_SIZE]
    tensor where each environment's state is a row.

    Parameters:
        NUM_BODIES: Number of bodies per environment.
        STATE_SIZE: Total size of each environment's state (row size).
        BODIES_OFFSET: Offset to body data within environment state.
        FORCES_OFFSET: Offset to forces data within environment state.
        JOINTS_OFFSET: Offset to joints data within environment state.
        JOINT_COUNT_OFFSET: Offset to joint count within environment state.
        EDGES_OFFSET: Offset to edge terrain data within environment state.
        EDGE_COUNT_OFFSET: Offset to edge count within environment state.
        MAX_JOINTS: Maximum joints per environment.
        FORCES_PER_BODY: Number of force values per body (fx, fy, torque).
    """

    var env: Int

    fn __init__(out self, env: Int):
        """Create helper for accessing a specific environment's physics data.

        Args:
            env: Environment index.
        """
        self.env = env

    # =========================================================================
    # Body Position Accessors
    # =========================================================================

    @always_inline
    fn body_offset(self, body: Int) -> Int:
        """Get offset for a body's data within the state row."""
        return Self.BODIES_OFFSET + body * BODY_STATE_SIZE

    @always_inline
    fn set_body_position[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
        x: Float64,
        y: Float64,
    ):
        """Set body position."""
        var off = self.body_offset(body)
        states[self.env, off + IDX_X] = Scalar[dtype](x)
        states[self.env, off + IDX_Y] = Scalar[dtype](y)

    @always_inline
    fn set_body_angle[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
        angle: Float64,
    ):
        """Set body angle."""
        var off = self.body_offset(body)
        states[self.env, off + IDX_ANGLE] = Scalar[dtype](angle)

    @always_inline
    fn set_body_velocity[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
        vx: Float64,
        vy: Float64,
        omega: Float64,
    ):
        """Set body linear and angular velocity."""
        var off = self.body_offset(body)
        states[self.env, off + IDX_VX] = Scalar[dtype](vx)
        states[self.env, off + IDX_VY] = Scalar[dtype](vy)
        states[self.env, off + IDX_OMEGA] = Scalar[dtype](omega)

    @always_inline
    fn set_body_mass[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
        mass: Float64,
        inertia: Float64,
    ):
        """Set body mass properties. Use mass=0 for static bodies."""
        var off = self.body_offset(body)
        states[self.env, off + IDX_MASS] = Scalar[dtype](mass)
        if mass > 0:
            states[self.env, off + IDX_INV_MASS] = Scalar[dtype](1.0 / mass)
            if inertia > 0:
                states[self.env, off + IDX_INV_INERTIA] = Scalar[dtype](1.0 / inertia)
            else:
                states[self.env, off + IDX_INV_INERTIA] = Scalar[dtype](0)
        else:
            states[self.env, off + IDX_INV_MASS] = Scalar[dtype](0)
            states[self.env, off + IDX_INV_INERTIA] = Scalar[dtype](0)

    @always_inline
    fn set_body_shape[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
        shape_idx: Int,
    ):
        """Set body shape reference."""
        var off = self.body_offset(body)
        states[self.env, off + IDX_SHAPE] = Scalar[dtype](shape_idx)

    @always_inline
    fn get_body_x[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body x position."""
        var off = self.body_offset(body)
        return rebind[Scalar[dtype]](states[self.env, off + IDX_X])

    @always_inline
    fn get_body_y[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body y position."""
        var off = self.body_offset(body)
        return rebind[Scalar[dtype]](states[self.env, off + IDX_Y])

    @always_inline
    fn get_body_angle[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body angle."""
        var off = self.body_offset(body)
        return rebind[Scalar[dtype]](states[self.env, off + IDX_ANGLE])

    @always_inline
    fn get_body_vx[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body x velocity."""
        var off = self.body_offset(body)
        return rebind[Scalar[dtype]](states[self.env, off + IDX_VX])

    @always_inline
    fn get_body_vy[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body y velocity."""
        var off = self.body_offset(body)
        return rebind[Scalar[dtype]](states[self.env, off + IDX_VY])

    @always_inline
    fn get_body_omega[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
    ) -> Scalar[dtype]:
        """Get body angular velocity."""
        var off = self.body_offset(body)
        return rebind[Scalar[dtype]](states[self.env, off + IDX_OMEGA])

    # =========================================================================
    # Force Application
    # =========================================================================

    @always_inline
    fn force_offset(self, body: Int) -> Int:
        """Get offset for a body's forces within the state row."""
        return Self.FORCES_OFFSET + body * Self.FORCES_PER_BODY

    @always_inline
    fn clear_forces[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
    ):
        """Clear all forces for this environment."""
        for body in range(Self.NUM_BODIES):
            var off = self.force_offset(body)
            states[self.env, off + 0] = Scalar[dtype](0)
            states[self.env, off + 1] = Scalar[dtype](0)
            states[self.env, off + 2] = Scalar[dtype](0)

    @always_inline
    fn apply_force[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
        fx: Float64,
        fy: Float64,
    ):
        """Apply a force to a body (accumulated until step)."""
        var off = self.force_offset(body)
        states[self.env, off + 0] = states[self.env, off + 0] + Scalar[dtype](fx)
        states[self.env, off + 1] = states[self.env, off + 1] + Scalar[dtype](fy)

    @always_inline
    fn apply_torque[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
        torque: Float64,
    ):
        """Apply a torque to a body (accumulated until step)."""
        var off = self.force_offset(body)
        states[self.env, off + 2] = states[self.env, off + 2] + Scalar[dtype](torque)

    @always_inline
    fn set_force[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        body: Int,
        fx: Float64,
        fy: Float64,
        torque: Float64,
    ):
        """Set force directly (replaces existing)."""
        var off = self.force_offset(body)
        states[self.env, off + 0] = Scalar[dtype](fx)
        states[self.env, off + 1] = Scalar[dtype](fy)
        states[self.env, off + 2] = Scalar[dtype](torque)

    # =========================================================================
    # Joint Management
    # =========================================================================

    @always_inline
    fn joint_offset(self, joint: Int) -> Int:
        """Get offset for a joint's data within the state row."""
        return Self.JOINTS_OFFSET + joint * JOINT_DATA_SIZE

    @always_inline
    fn get_joint_count[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
    ) -> Int:
        """Get number of active joints."""
        return Int(states[self.env, Self.JOINT_COUNT_OFFSET])

    @always_inline
    fn set_joint_count[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        count: Int,
    ):
        """Set joint count."""
        states[self.env, Self.JOINT_COUNT_OFFSET] = Scalar[dtype](count)

    @always_inline
    fn add_revolute_joint[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
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

        var off = self.joint_offset(joint_idx)

        # Set joint type
        states[self.env, off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)

        # Set body indices
        states[self.env, off + JOINT_BODY_A] = Scalar[dtype](body_a)
        states[self.env, off + JOINT_BODY_B] = Scalar[dtype](body_b)

        # Set local anchors
        states[self.env, off + JOINT_ANCHOR_AX] = Scalar[dtype](anchor_ax)
        states[self.env, off + JOINT_ANCHOR_AY] = Scalar[dtype](anchor_ay)
        states[self.env, off + JOINT_ANCHOR_BX] = Scalar[dtype](anchor_bx)
        states[self.env, off + JOINT_ANCHOR_BY] = Scalar[dtype](anchor_by)

        # Compute reference angle (angle_b - angle_a at creation)
        var angle_a = self.get_body_angle[BATCH](states, body_a)
        var angle_b = self.get_body_angle[BATCH](states, body_b)
        states[self.env, off + JOINT_REF_ANGLE] = angle_b - angle_a

        # Set angle limits
        states[self.env, off + JOINT_LOWER_LIMIT] = Scalar[dtype](lower_limit)
        states[self.env, off + JOINT_UPPER_LIMIT] = Scalar[dtype](upper_limit)

        # Set spring properties
        states[self.env, off + JOINT_STIFFNESS] = Scalar[dtype](stiffness)
        states[self.env, off + JOINT_DAMPING] = Scalar[dtype](damping)

        # Set flags
        var flags = 0
        if stiffness > 0.0 or damping > 0.0:
            flags = flags | JOINT_FLAG_SPRING_ENABLED
        if enable_limit:
            flags = flags | JOINT_FLAG_LIMIT_ENABLED
        states[self.env, off + JOINT_FLAGS] = Scalar[dtype](flags)

        # Increment count
        self.set_joint_count[BATCH](states, joint_idx + 1)

        return joint_idx

    @always_inline
    fn clear_joints[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
    ):
        """Clear all joints for this environment."""
        for j in range(Self.MAX_JOINTS):
            var off = self.joint_offset(j)
            for i in range(JOINT_DATA_SIZE):
                states[self.env, off + i] = Scalar[dtype](0)
        self.set_joint_count[BATCH](states, 0)

    # =========================================================================
    # Edge Terrain
    # =========================================================================

    @always_inline
    fn edge_offset(self, edge: Int) -> Int:
        """Get offset for an edge's data (6 floats: x0, y0, x1, y1, nx, ny)."""
        return Self.EDGES_OFFSET + edge * 6

    @always_inline
    fn get_edge_count[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
    ) -> Int:
        """Get number of active edges."""
        return Int(states[self.env, Self.EDGE_COUNT_OFFSET])

    @always_inline
    fn set_edge_count[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        count: Int,
    ):
        """Set edge count."""
        states[self.env, Self.EDGE_COUNT_OFFSET] = Scalar[dtype](count)

    @always_inline
    fn set_edge[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin],
        edge: Int,
        x0: Float64,
        y0: Float64,
        x1: Float64,
        y1: Float64,
        nx: Float64,
        ny: Float64,
    ):
        """Set edge data."""
        var off = self.edge_offset(edge)
        states[self.env, off + 0] = Scalar[dtype](x0)
        states[self.env, off + 1] = Scalar[dtype](y0)
        states[self.env, off + 2] = Scalar[dtype](x1)
        states[self.env, off + 3] = Scalar[dtype](y1)
        states[self.env, off + 4] = Scalar[dtype](nx)
        states[self.env, off + 5] = Scalar[dtype](ny)
