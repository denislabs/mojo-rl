"""PhysicsState - Helper for accessing physics data in 2D strided layout.

This module provides a helper struct that gives PhysicsState-like accessors
for physics data stored in a 2D [BATCH, STATE_SIZE] layout (as required by GPUDiscreteEnv).

PhysicsState works with a 2D buffer where the second dimension is the state:

    state[env, OFFSET + body * BODY_STATE_SIZE + field]

This is compatible with the strided GPU methods in:
- SemiImplicitEuler.integrate_velocities_gpu
- SemiImplicitEuler.integrate_positions_gpu
- ImpulseSolver.solve_velocity_gpu
- ImpulseSolver.solve_position_gpu
- RevoluteJointSolver.solve_velocity_gpu
- RevoluteJointSolver.solve_position_gpu
- EdgeTerrainCollision.detect_gpu

Example:
    ```mojo
    from physics_gpu.state import PhysicsState

    # Define layout constants
    comptime NUM_BODIES = 3
    comptime STATE_SIZE = 192
    comptime BODIES_OFFSET = 8

    # Create helper for a specific environment
    var physics = PhysicsState[NUM_BODIES, STATE_SIZE, BODIES_OFFSET](env)

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


struct PhysicsState[
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
                states[self.env, off + IDX_INV_INERTIA] = Scalar[dtype](
                    1.0 / inertia
                )
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
        body: Int,
        fx: Float64,
        fy: Float64,
    ):
        """Apply a force to a body (accumulated until step)."""
        var off = self.force_offset(body)
        states[self.env, off + 0] = states[self.env, off + 0] + Scalar[dtype](
            fx
        )
        states[self.env, off + 1] = states[self.env, off + 1] + Scalar[dtype](
            fy
        )

    @always_inline
    fn apply_torque[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
        body: Int,
        torque: Float64,
    ):
        """Apply a torque to a body (accumulated until step)."""
        var off = self.force_offset(body)
        states[self.env, off + 2] = states[self.env, off + 2] + Scalar[dtype](
            torque
        )

    @always_inline
    fn set_force[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
    ) -> Int:
        """Get number of active joints."""
        return Int(states[self.env, Self.JOINT_COUNT_OFFSET])

    @always_inline
    fn set_joint_count[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
        count: Int,
    ):
        """Set joint count."""
        states[self.env, Self.JOINT_COUNT_OFFSET] = Scalar[dtype](count)

    @always_inline
    fn add_revolute_joint[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
    ) -> Int:
        """Get number of active edges."""
        return Int(states[self.env, Self.EDGE_COUNT_OFFSET])

    @always_inline
    fn set_edge_count[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
        count: Int,
    ):
        """Set edge count."""
        states[self.env, Self.EDGE_COUNT_OFFSET] = Scalar[dtype](count)

    @always_inline
    fn set_edge[
        BATCH: Int
    ](
        self,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.STATE_SIZE), MutAnyOrigin
        ],
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


# =============================================================================
# PhysicsStateOwned - Memory-owning wrapper for single-env CPU mode
# =============================================================================


from .constants import SHAPE_POLYGON, SHAPE_MAX_SIZE, CONTACT_DATA_SIZE


struct PhysicsStateOwned[
    NUM_BODIES: Int,
    NUM_SHAPES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
    STATE_SIZE: Int,
    BODIES_OFFSET: Int,
    FORCES_OFFSET: Int,
    JOINTS_OFFSET: Int,
    JOINT_COUNT_OFFSET: Int,
    EDGES_OFFSET: Int,
    EDGE_COUNT_OFFSET: Int,
]:
    """Memory-owning physics state for single-env CPU operation.

    This struct owns the state buffer and provides PhysicsState-like accessors.
    It uses strided layout internally, compatible with GPU kernels.

    Parameters:
        NUM_BODIES: Number of bodies.
        NUM_SHAPES: Number of shape definitions.
        MAX_CONTACTS: Maximum contacts.
        MAX_JOINTS: Maximum joints.
        STATE_SIZE: Total state size per environment.
        BODIES_OFFSET: Offset to body data.
        FORCES_OFFSET: Offset to forces data.
        JOINTS_OFFSET: Offset to joints data.
        JOINT_COUNT_OFFSET: Offset to joint count.
        EDGES_OFFSET: Offset to edge terrain data.
        EDGE_COUNT_OFFSET: Offset to edge count.
    """

    # Type alias for the strided helper
    comptime Helper = PhysicsState[
        Self.NUM_BODIES,
        Self.STATE_SIZE,
        Self.BODIES_OFFSET,
        Self.FORCES_OFFSET,
        Self.JOINTS_OFFSET,
        Self.JOINT_COUNT_OFFSET,
        Self.EDGES_OFFSET,
        Self.EDGE_COUNT_OFFSET,
        Self.MAX_JOINTS,
    ]

    # Buffer sizes
    comptime SHAPES_SIZE: Int = Self.NUM_SHAPES * SHAPE_MAX_SIZE
    comptime CONTACTS_SIZE: Int = Self.MAX_CONTACTS * CONTACT_DATA_SIZE

    # Owned buffers
    var state: List[Scalar[dtype]]
    var shapes: List[Scalar[dtype]]
    var contacts: List[Scalar[dtype]]
    var contact_counts: List[
        Scalar[dtype]
    ]  # Single-element buffer for tensor view

    fn __init__(out self):
        """Initialize with zeroed buffers."""
        # Allocate state buffer
        self.state = List[Scalar[dtype]](capacity=Self.STATE_SIZE)
        for _ in range(Self.STATE_SIZE):
            self.state.append(Scalar[dtype](0))

        # Allocate shapes buffer
        self.shapes = List[Scalar[dtype]](capacity=Self.SHAPES_SIZE)
        for _ in range(Self.SHAPES_SIZE):
            self.shapes.append(Scalar[dtype](0))

        # Allocate contacts buffer
        self.contacts = List[Scalar[dtype]](capacity=Self.CONTACTS_SIZE)
        for _ in range(Self.CONTACTS_SIZE):
            self.contacts.append(Scalar[dtype](0))

        # Allocate contact counts buffer (single element for single env)
        self.contact_counts = List[Scalar[dtype]](capacity=1)
        self.contact_counts.append(Scalar[dtype](0))

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        # Explicitly copy Lists
        self.state = List[Scalar[dtype]](capacity=Self.STATE_SIZE)
        for i in range(len(existing.state)):
            self.state.append(existing.state[i])
        self.shapes = List[Scalar[dtype]](capacity=Self.SHAPES_SIZE)
        for i in range(len(existing.shapes)):
            self.shapes.append(existing.shapes[i])
        self.contacts = List[Scalar[dtype]](capacity=Self.CONTACTS_SIZE)
        for i in range(len(existing.contacts)):
            self.contacts.append(existing.contacts[i])
        self.contact_counts = List[Scalar[dtype]](capacity=1)
        self.contact_counts.append(existing.contact_counts[0])

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.state = existing.state^
        self.shapes = existing.shapes^
        self.contacts = existing.contacts^
        self.contact_counts = existing.contact_counts^

    # =========================================================================
    # Tensor Views
    # =========================================================================

    @always_inline
    fn get_state_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(1, Self.STATE_SIZE), MutAnyOrigin
    ]:
        """Get tensor view of state (1 x STATE_SIZE for single env)."""
        return LayoutTensor[
            dtype, Layout.row_major(1, Self.STATE_SIZE), MutAnyOrigin
        ](self.state.unsafe_ptr())

    @always_inline
    fn get_bodies_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(1, Self.NUM_BODIES, BODY_STATE_SIZE),
        MutAnyOrigin,
    ]:
        """Get tensor view of body state (for compatibility with old code)."""
        # Bodies are at BODIES_OFFSET in the state buffer
        return LayoutTensor[
            dtype,
            Layout.row_major(1, Self.NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ](self.state.unsafe_ptr() + Self.BODIES_OFFSET)

    @always_inline
    fn get_shapes_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
    ]:
        """Get tensor view of shapes."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ](self.shapes.unsafe_ptr())

    @always_inline
    fn get_forces_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(1, Self.NUM_BODIES, 3), MutAnyOrigin
    ]:
        """Get tensor view of forces."""
        return LayoutTensor[
            dtype, Layout.row_major(1, Self.NUM_BODIES, 3), MutAnyOrigin
        ](self.state.unsafe_ptr() + Self.FORCES_OFFSET)

    @always_inline
    fn get_contacts_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(1, Self.MAX_CONTACTS, CONTACT_DATA_SIZE),
        MutAnyOrigin,
    ]:
        """Get tensor view of contacts."""
        return LayoutTensor[
            dtype,
            Layout.row_major(1, Self.MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](self.contacts.unsafe_ptr())

    @always_inline
    fn get_contact_counts_tensor(
        mut self,
    ) -> LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin]:
        """Get tensor view of contact counts (single element for single env)."""
        return LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            self.contact_counts.unsafe_ptr()
        )

    @always_inline
    fn get_joints_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(1, Self.MAX_JOINTS, JOINT_DATA_SIZE),
        MutAnyOrigin,
    ]:
        """Get tensor view of joints."""
        return LayoutTensor[
            dtype,
            Layout.row_major(1, Self.MAX_JOINTS, JOINT_DATA_SIZE),
            MutAnyOrigin,
        ](self.state.unsafe_ptr() + Self.JOINTS_OFFSET)

    @always_inline
    fn get_joint_counts_tensor(
        mut self,
    ) -> LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin]:
        """Get tensor view of joint counts."""
        return LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            self.state.unsafe_ptr() + Self.JOINT_COUNT_OFFSET
        )

    # =========================================================================
    # Helper accessor
    # =========================================================================

    @always_inline
    fn helper(self) -> Self.Helper:
        """Get a PhysicsStateStrided helper for env 0."""
        return Self.Helper(0)

    # =========================================================================
    # Body State Accessors (env is always 0 for single-env mode)
    # =========================================================================

    fn set_body_position(mut self, env: Int, body: Int, x: Float64, y: Float64):
        """Set body position."""
        var states = self.get_state_tensor()
        self.helper().set_body_position[1](states, body, x, y)

    fn set_body_angle(mut self, env: Int, body: Int, angle: Float64):
        """Set body angle."""
        var states = self.get_state_tensor()
        self.helper().set_body_angle[1](states, body, angle)

    fn set_body_velocity(
        mut self, env: Int, body: Int, vx: Float64, vy: Float64, omega: Float64
    ):
        """Set body velocity."""
        var states = self.get_state_tensor()
        self.helper().set_body_velocity[1](states, body, vx, vy, omega)

    fn set_body_mass(
        mut self, env: Int, body: Int, mass: Float64, inertia: Float64
    ):
        """Set body mass properties."""
        var states = self.get_state_tensor()
        self.helper().set_body_mass[1](states, body, mass, inertia)

    fn set_body_shape(mut self, env: Int, body: Int, shape_idx: Int):
        """Set body shape reference."""
        var states = self.get_state_tensor()
        self.helper().set_body_shape[1](states, body, shape_idx)

    fn get_body_x(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body x position."""
        var states = self.get_state_tensor()
        return self.helper().get_body_x[1](states, body)

    fn get_body_y(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body y position."""
        var states = self.get_state_tensor()
        return self.helper().get_body_y[1](states, body)

    fn get_body_angle(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body angle."""
        var states = self.get_state_tensor()
        return self.helper().get_body_angle[1](states, body)

    fn get_body_vx(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body x velocity."""
        var states = self.get_state_tensor()
        return self.helper().get_body_vx[1](states, body)

    fn get_body_vy(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body y velocity."""
        var states = self.get_state_tensor()
        return self.helper().get_body_vy[1](states, body)

    fn get_body_omega(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body angular velocity."""
        var states = self.get_state_tensor()
        return self.helper().get_body_omega[1](states, body)

    # =========================================================================
    # Shape Definitions
    # =========================================================================

    fn define_polygon_shape(
        mut self,
        shape_idx: Int,
        vertices_x: List[Float64],
        vertices_y: List[Float64],
    ):
        """Define a polygon shape."""
        var shapes = self.get_shapes_tensor()
        var n_verts = min(len(vertices_x), 8)

        shapes[shape_idx, 0] = Scalar[dtype](SHAPE_POLYGON)
        shapes[shape_idx, 1] = Scalar[dtype](n_verts)

        for i in range(n_verts):
            shapes[shape_idx, 2 + i * 2] = Scalar[dtype](vertices_x[i])
            shapes[shape_idx, 3 + i * 2] = Scalar[dtype](vertices_y[i])

    # =========================================================================
    # Joint Management
    # =========================================================================

    fn add_revolute_joint(
        mut self,
        env: Int,
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
        """Add a revolute joint between two bodies."""
        var states = self.get_state_tensor()
        return self.helper().add_revolute_joint[1](
            states,
            body_a,
            body_b,
            anchor_ax,
            anchor_ay,
            anchor_bx,
            anchor_by,
            stiffness,
            damping,
            lower_limit,
            upper_limit,
            enable_limit,
        )

    fn clear_joints(mut self, env: Int):
        """Clear all joints."""
        var states = self.get_state_tensor()
        self.helper().clear_joints[1](states)

    fn get_joint_count(mut self, env: Int) -> Int:
        """Get number of active joints."""
        var states = self.get_state_tensor()
        return self.helper().get_joint_count[1](states)

    # =========================================================================
    # Contact Information
    # =========================================================================

    fn get_contact_count(mut self, env: Int) -> Int:
        """Get number of active contacts."""
        return Int(self.contact_counts[0])

    fn set_contact_count(mut self, count: Int):
        """Set contact count (called by collision system)."""
        self.contact_counts[0] = Scalar[dtype](count)
