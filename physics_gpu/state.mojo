"""PhysicsState - Memory management for GPU physics.

This module provides a memory management wrapper for physics simulation,
similar to how deep_rl's Network wraps a stateless Model.

PhysicsState:
- Owns the List buffers for physics state
- Provides LayoutTensor views for zero-copy access
- Handles CPU↔GPU buffer transfers
- Manages shape definitions (shared across environments)

Example:
    ```mojo
    from physics_gpu import PhysicsState, PhysicsConfig

    # Create physics state: 1024 envs, 3 bodies, 2 shapes, 16 max contacts
    var state = PhysicsState[1024, 3, 2, 16]()

    # Define shapes (once, shared across envs)
    state.define_polygon_shape(0, vertices_x, vertices_y)

    # Initialize bodies for all envs
    for env in range(1024):
        state.set_body_position(env, 0, 0.0, 10.0)
        state.set_body_mass(env, 0, 1.0, 0.1)

    # CPU step
    var config = PhysicsConfig(ground_y=0.0)
    state.step(config)

    # GPU step
    var ctx = DeviceContext()
    state.step_gpu(ctx, config)
    ```
"""

from layout import LayoutTensor, Layout
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from .constants import (
    dtype,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
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
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    # Joint constants
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
from .layout import PhysicsLayout
from .kernel import PhysicsKernel, PhysicsConfig


struct PhysicsState[
    BATCH: Int,
    NUM_BODIES: Int,
    NUM_SHAPES: Int,
    MAX_CONTACTS: Int = 16,
    MAX_JOINTS: Int = 8,
]:
    """Memory management wrapper for physics simulation.

    This struct owns the List buffers for physics state and provides
    LayoutTensor views for zero-copy access. Similar to how Network
    wraps a stateless Model.

    Parameters:
        BATCH: Number of parallel environments.
        NUM_BODIES: Number of bodies per environment.
        NUM_SHAPES: Total number of shape definitions.
        MAX_CONTACTS: Maximum contacts per environment.
        MAX_JOINTS: Maximum joints per environment.
    """

    # Layout for compile-time sizes
    comptime LAYOUT = PhysicsLayout[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS, Self.MAX_JOINTS]

    # Buffer sizes (from layout)
    comptime BODIES_SIZE: Int = Self.LAYOUT.BODIES_SIZE
    comptime SHAPES_SIZE: Int = Self.LAYOUT.SHAPES_SIZE
    comptime FORCES_SIZE: Int = Self.LAYOUT.FORCES_SIZE
    comptime CONTACTS_SIZE: Int = Self.LAYOUT.CONTACTS_SIZE
    comptime COUNTS_SIZE: Int = Self.LAYOUT.COUNTS_SIZE
    comptime JOINTS_SIZE: Int = Self.LAYOUT.JOINTS_SIZE
    comptime JOINT_COUNTS_SIZE: Int = Self.LAYOUT.JOINT_COUNTS_SIZE

    # Heap-allocated buffers
    var bodies: List[Scalar[dtype]]
    var shapes: List[Scalar[dtype]]
    var forces: List[Scalar[dtype]]
    var contacts: List[Scalar[dtype]]
    var contact_counts: List[Scalar[dtype]]
    var joints: List[Scalar[dtype]]
    var joint_counts: List[Scalar[dtype]]

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(out self):
        """Initialize physics state with zeroed buffers."""
        # Allocate and zero-fill bodies
        self.bodies = List[Scalar[dtype]](capacity=Self.BODIES_SIZE)
        for _ in range(Self.BODIES_SIZE):
            self.bodies.append(Scalar[dtype](0))

        # Allocate and zero-fill shapes
        self.shapes = List[Scalar[dtype]](capacity=Self.SHAPES_SIZE)
        for _ in range(Self.SHAPES_SIZE):
            self.shapes.append(Scalar[dtype](0))

        # Allocate and zero-fill forces
        self.forces = List[Scalar[dtype]](capacity=Self.FORCES_SIZE)
        for _ in range(Self.FORCES_SIZE):
            self.forces.append(Scalar[dtype](0))

        # Allocate and zero-fill contacts
        self.contacts = List[Scalar[dtype]](capacity=Self.CONTACTS_SIZE)
        for _ in range(Self.CONTACTS_SIZE):
            self.contacts.append(Scalar[dtype](0))

        # Allocate and zero-fill contact counts
        self.contact_counts = List[Scalar[dtype]](capacity=Self.COUNTS_SIZE)
        for _ in range(Self.COUNTS_SIZE):
            self.contact_counts.append(Scalar[dtype](0))

        # Allocate and zero-fill joints
        self.joints = List[Scalar[dtype]](capacity=Self.JOINTS_SIZE)
        for _ in range(Self.JOINTS_SIZE):
            self.joints.append(Scalar[dtype](0))

        # Allocate and zero-fill joint counts
        self.joint_counts = List[Scalar[dtype]](capacity=Self.JOINT_COUNTS_SIZE)
        for _ in range(Self.JOINT_COUNTS_SIZE):
            self.joint_counts.append(Scalar[dtype](0))

    # =========================================================================
    # Tensor Views (zero-copy)
    # =========================================================================

    @always_inline
    fn get_bodies_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.BATCH, Self.NUM_BODIES, BODY_STATE_SIZE),
        MutAnyOrigin,
    ]:
        """Get tensor view of body state."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ](self.bodies.unsafe_ptr())

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
        dtype, Layout.row_major(Self.BATCH, Self.NUM_BODIES, 3), MutAnyOrigin
    ]:
        """Get tensor view of forces."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.NUM_BODIES, 3), MutAnyOrigin
        ](self.forces.unsafe_ptr())

    @always_inline
    fn get_contacts_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.BATCH, Self.MAX_CONTACTS, CONTACT_DATA_SIZE),
        MutAnyOrigin,
    ]:
        """Get tensor view of contacts."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](self.contacts.unsafe_ptr())

    @always_inline
    fn get_contact_counts_tensor(
        mut self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.BATCH), MutAnyOrigin]:
        """Get tensor view of contact counts."""
        return LayoutTensor[dtype, Layout.row_major(Self.BATCH), MutAnyOrigin](
            self.contact_counts.unsafe_ptr()
        )

    @always_inline
    fn get_joints_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.BATCH, Self.MAX_JOINTS, JOINT_DATA_SIZE),
        MutAnyOrigin,
    ]:
        """Get tensor view of joints."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.MAX_JOINTS, JOINT_DATA_SIZE),
            MutAnyOrigin,
        ](self.joints.unsafe_ptr())

    @always_inline
    fn get_joint_counts_tensor(
        mut self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.BATCH), MutAnyOrigin]:
        """Get tensor view of joint counts."""
        return LayoutTensor[dtype, Layout.row_major(Self.BATCH), MutAnyOrigin](
            self.joint_counts.unsafe_ptr()
        )

    # =========================================================================
    # CPU Physics Step
    # =========================================================================

    fn step(mut self, config: PhysicsConfig):
        """Execute one physics step on CPU.

        Args:
            config: Physics configuration.
        """
        var bodies = self.get_bodies_tensor()
        var shapes = self.get_shapes_tensor()
        var forces = self.get_forces_tensor()
        var contacts = self.get_contacts_tensor()
        var contact_counts = self.get_contact_counts_tensor()

        PhysicsKernel.step[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS](
            bodies, shapes, forces, contacts, contact_counts, config
        )

    # =========================================================================
    # GPU Physics Step
    # =========================================================================

    fn step_gpu(mut self, ctx: DeviceContext, config: PhysicsConfig) raises:
        """Execute one physics step on GPU.

        Data is transferred to GPU, processed, and transferred back.

        Args:
            ctx: GPU device context.
            config: Physics configuration.
        """
        # Allocate GPU buffers
        var bodies_buf = ctx.enqueue_create_buffer[dtype](Self.BODIES_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](Self.SHAPES_SIZE)
        var forces_buf = ctx.enqueue_create_buffer[dtype](Self.FORCES_SIZE)
        var contacts_buf = ctx.enqueue_create_buffer[dtype](Self.CONTACTS_SIZE)
        var contact_counts_buf = ctx.enqueue_create_buffer[dtype](Self.COUNTS_SIZE)

        # Copy to GPU
        ctx.enqueue_copy(bodies_buf, self.bodies.unsafe_ptr())
        ctx.enqueue_copy(shapes_buf, self.shapes.unsafe_ptr())
        ctx.enqueue_copy(forces_buf, self.forces.unsafe_ptr())

        # Execute physics step
        PhysicsKernel.step_gpu[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS](
            ctx, bodies_buf, shapes_buf, forces_buf, contacts_buf, contact_counts_buf, config
        )

        # Copy back to CPU
        ctx.enqueue_copy(self.bodies.unsafe_ptr(), bodies_buf)
        ctx.enqueue_copy(self.contacts.unsafe_ptr(), contacts_buf)
        ctx.enqueue_copy(self.contact_counts.unsafe_ptr(), contact_counts_buf)

        # Synchronize
        ctx.synchronize()

        # Clear forces on CPU
        self._clear_forces()

    fn _clear_forces(mut self):
        """Clear accumulated forces after physics step."""
        var forces = self.get_forces_tensor()
        for env in range(Self.BATCH):
            for body in range(Self.NUM_BODIES):
                forces[env, body, 0] = Scalar[dtype](0)
                forces[env, body, 1] = Scalar[dtype](0)
                forces[env, body, 2] = Scalar[dtype](0)

    # =========================================================================
    # Body State Accessors
    # =========================================================================

    fn set_body_position(mut self, env: Int, body: Int, x: Float64, y: Float64):
        """Set body position."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, IDX_X] = Scalar[dtype](x)
        bodies[env, body, IDX_Y] = Scalar[dtype](y)

    fn set_body_angle(mut self, env: Int, body: Int, angle: Float64):
        """Set body angle."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, IDX_ANGLE] = Scalar[dtype](angle)

    fn set_body_velocity(
        mut self, env: Int, body: Int, vx: Float64, vy: Float64, omega: Float64
    ):
        """Set body linear and angular velocity."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, IDX_VX] = Scalar[dtype](vx)
        bodies[env, body, IDX_VY] = Scalar[dtype](vy)
        bodies[env, body, IDX_OMEGA] = Scalar[dtype](omega)

    fn set_body_mass(
        mut self, env: Int, body: Int, mass: Float64, inertia: Float64
    ):
        """Set body mass properties. Use mass=0 for static bodies."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, IDX_MASS] = Scalar[dtype](mass)
        if mass > 0:
            bodies[env, body, IDX_INV_MASS] = Scalar[dtype](1.0 / mass)
            if inertia > 0:
                bodies[env, body, IDX_INV_INERTIA] = Scalar[dtype](1.0 / inertia)
            else:
                bodies[env, body, IDX_INV_INERTIA] = Scalar[dtype](0)
        else:
            bodies[env, body, IDX_INV_MASS] = Scalar[dtype](0)
            bodies[env, body, IDX_INV_INERTIA] = Scalar[dtype](0)

    fn set_body_shape(mut self, env: Int, body: Int, shape_idx: Int):
        """Set body shape reference."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, IDX_SHAPE] = Scalar[dtype](shape_idx)

    fn get_body_x(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body x position."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, IDX_X])

    fn get_body_y(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body y position."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, IDX_Y])

    fn get_body_angle(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body angle."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, IDX_ANGLE])

    fn get_body_vx(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body x velocity."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, IDX_VX])

    fn get_body_vy(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body y velocity."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, IDX_VY])

    fn get_body_omega(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body angular velocity."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, IDX_OMEGA])

    # =========================================================================
    # Force Application
    # =========================================================================

    fn apply_force(mut self, env: Int, body: Int, fx: Float64, fy: Float64):
        """Apply a force to a body (accumulated until step)."""
        var forces = self.get_forces_tensor()
        forces[env, body, 0] = forces[env, body, 0] + Scalar[dtype](fx)
        forces[env, body, 1] = forces[env, body, 1] + Scalar[dtype](fy)

    fn apply_torque(mut self, env: Int, body: Int, torque: Float64):
        """Apply a torque to a body (accumulated until step)."""
        var forces = self.get_forces_tensor()
        forces[env, body, 2] = forces[env, body, 2] + Scalar[dtype](torque)

    # =========================================================================
    # Shape Definitions
    # =========================================================================

    fn define_polygon_shape(
        mut self,
        shape_idx: Int,
        vertices_x: List[Float64],
        vertices_y: List[Float64],
    ):
        """Define a polygon shape. Vertices should be in CCW order."""
        var shapes = self.get_shapes_tensor()
        var n_verts = min(len(vertices_x), 8)

        shapes[shape_idx, 0] = Scalar[dtype](SHAPE_POLYGON)
        shapes[shape_idx, 1] = Scalar[dtype](n_verts)

        for i in range(n_verts):
            shapes[shape_idx, 2 + i * 2] = Scalar[dtype](vertices_x[i])
            shapes[shape_idx, 3 + i * 2] = Scalar[dtype](vertices_y[i])

    fn define_circle_shape(
        mut self,
        shape_idx: Int,
        radius: Float64,
        center_x: Float64 = 0.0,
        center_y: Float64 = 0.0,
    ):
        """Define a circle shape."""
        var shapes = self.get_shapes_tensor()
        shapes[shape_idx, 0] = Scalar[dtype](SHAPE_CIRCLE)
        shapes[shape_idx, 1] = Scalar[dtype](radius)
        shapes[shape_idx, 2] = Scalar[dtype](center_x)
        shapes[shape_idx, 3] = Scalar[dtype](center_y)

    # =========================================================================
    # Contact Information
    # =========================================================================

    fn get_contact_count(mut self, env: Int) -> Int:
        """Get number of active contacts for an environment."""
        var contact_counts = self.get_contact_counts_tensor()
        return Int(contact_counts[env])

    # =========================================================================
    # Joint Definitions
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
        """Add a revolute joint between two bodies.

        Args:
            env: Environment index.
            body_a: First body index.
            body_b: Second body index.
            anchor_ax: Local anchor x on body A.
            anchor_ay: Local anchor y on body A.
            anchor_bx: Local anchor x on body B.
            anchor_by: Local anchor y on body B.
            stiffness: Spring stiffness (0 for rigid joint).
            damping: Spring damping.
            lower_limit: Lower angle limit in radians (relative to reference angle).
            upper_limit: Upper angle limit in radians (relative to reference angle).
            enable_limit: Whether to enable angle limits.

        Returns:
            Joint index, or -1 if max joints reached.
        """
        var joint_counts = self.get_joint_counts_tensor()
        var joints = self.get_joints_tensor()
        var bodies = self.get_bodies_tensor()

        var joint_idx = Int(joint_counts[env])
        if joint_idx >= Self.MAX_JOINTS:
            return -1

        # Set joint type
        joints[env, joint_idx, JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)

        # Set body indices
        joints[env, joint_idx, JOINT_BODY_A] = Scalar[dtype](body_a)
        joints[env, joint_idx, JOINT_BODY_B] = Scalar[dtype](body_b)

        # Set local anchors
        joints[env, joint_idx, JOINT_ANCHOR_AX] = Scalar[dtype](anchor_ax)
        joints[env, joint_idx, JOINT_ANCHOR_AY] = Scalar[dtype](anchor_ay)
        joints[env, joint_idx, JOINT_ANCHOR_BX] = Scalar[dtype](anchor_bx)
        joints[env, joint_idx, JOINT_ANCHOR_BY] = Scalar[dtype](anchor_by)

        # Compute reference angle (angle_b - angle_a at creation)
        var angle_a = rebind[Scalar[dtype]](bodies[env, body_a, IDX_ANGLE])
        var angle_b = rebind[Scalar[dtype]](bodies[env, body_b, IDX_ANGLE])
        joints[env, joint_idx, JOINT_REF_ANGLE] = angle_b - angle_a

        # Set angle limits
        joints[env, joint_idx, JOINT_LOWER_LIMIT] = Scalar[dtype](lower_limit)
        joints[env, joint_idx, JOINT_UPPER_LIMIT] = Scalar[dtype](upper_limit)

        # Set spring properties
        joints[env, joint_idx, JOINT_STIFFNESS] = Scalar[dtype](stiffness)
        joints[env, joint_idx, JOINT_DAMPING] = Scalar[dtype](damping)

        # Set flags
        var flags = 0
        if stiffness > 0.0 or damping > 0.0:
            flags = flags | JOINT_FLAG_SPRING_ENABLED
        if enable_limit:
            flags = flags | JOINT_FLAG_LIMIT_ENABLED
        joints[env, joint_idx, JOINT_FLAGS] = Scalar[dtype](flags)

        # Increment count
        joint_counts[env] = Scalar[dtype](joint_idx + 1)

        return joint_idx

    fn get_joint_count(mut self, env: Int) -> Int:
        """Get number of active joints for an environment."""
        var joint_counts = self.get_joint_counts_tensor()
        return Int(joint_counts[env])

    fn clear_joints(mut self, env: Int):
        """Clear all joints for an environment."""
        var joint_counts = self.get_joint_counts_tensor()
        var joints = self.get_joints_tensor()

        for j in range(Self.MAX_JOINTS):
            for i in range(JOINT_DATA_SIZE):
                joints[env, j, i] = Scalar[dtype](0)
        joint_counts[env] = Scalar[dtype](0)

    # =========================================================================
    # Environment Reset
    # =========================================================================

    fn reset_env(mut self, env: Int):
        """Reset a single environment to initial state (clears velocities, forces, contacts, joints)."""
        var bodies = self.get_bodies_tensor()
        var forces = self.get_forces_tensor()
        var contacts = self.get_contacts_tensor()
        var contact_counts = self.get_contact_counts_tensor()
        var joints = self.get_joints_tensor()
        var joint_counts = self.get_joint_counts_tensor()

        # Clear body velocities and forces
        for body in range(Self.NUM_BODIES):
            bodies[env, body, IDX_VX] = Scalar[dtype](0)
            bodies[env, body, IDX_VY] = Scalar[dtype](0)
            bodies[env, body, IDX_OMEGA] = Scalar[dtype](0)
            forces[env, body, 0] = Scalar[dtype](0)
            forces[env, body, 1] = Scalar[dtype](0)
            forces[env, body, 2] = Scalar[dtype](0)

        # Clear contacts
        for c in range(Self.MAX_CONTACTS):
            for i in range(CONTACT_DATA_SIZE):
                contacts[env, c, i] = Scalar[dtype](0)
        contact_counts[env] = Scalar[dtype](0)

        # Clear joints
        for j in range(Self.MAX_JOINTS):
            for i in range(JOINT_DATA_SIZE):
                joints[env, j, i] = Scalar[dtype](0)
        joint_counts[env] = Scalar[dtype](0)

    # =========================================================================
    # GPU Buffer Management (for persistent GPU execution)
    # =========================================================================

    fn copy_to_device(
        self,
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        mut shapes_buf: DeviceBuffer[dtype],
        mut forces_buf: DeviceBuffer[dtype],
    ) raises:
        """Copy CPU state to GPU buffers.

        Use this for persistent GPU execution where you want to minimize
        transfers by keeping state on GPU across multiple steps.

        Args:
            ctx: GPU device context.
            bodies_buf: Device buffer for bodies [BODIES_SIZE].
            shapes_buf: Device buffer for shapes [SHAPES_SIZE].
            forces_buf: Device buffer for forces [FORCES_SIZE].
        """
        ctx.enqueue_copy(bodies_buf, self.bodies.unsafe_ptr())
        ctx.enqueue_copy(shapes_buf, self.shapes.unsafe_ptr())
        ctx.enqueue_copy(forces_buf, self.forces.unsafe_ptr())

    fn copy_from_device(
        mut self,
        ctx: DeviceContext,
        bodies_buf: DeviceBuffer[dtype],
        contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Copy GPU state back to CPU buffers.

        Args:
            ctx: GPU device context.
            bodies_buf: Device buffer with body state.
            contacts_buf: Device buffer with contacts.
            contact_counts_buf: Device buffer with contact counts.
        """
        ctx.enqueue_copy(self.bodies.unsafe_ptr(), bodies_buf)
        ctx.enqueue_copy(self.contacts.unsafe_ptr(), contacts_buf)
        ctx.enqueue_copy(self.contact_counts.unsafe_ptr(), contact_counts_buf)
        ctx.synchronize()

    fn allocate_device_buffers(
        self,
        ctx: DeviceContext,
    ) raises -> Tuple[
        DeviceBuffer[dtype],
        DeviceBuffer[dtype],
        DeviceBuffer[dtype],
        DeviceBuffer[dtype],
        DeviceBuffer[dtype],
    ]:
        """Allocate GPU buffers for physics state.

        Returns a tuple of (bodies_buf, shapes_buf, forces_buf, contacts_buf, counts_buf).

        Args:
            ctx: GPU device context.

        Returns:
            Tuple of allocated device buffers.
        """
        var bodies_buf = ctx.enqueue_create_buffer[dtype](Self.BODIES_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](Self.SHAPES_SIZE)
        var forces_buf = ctx.enqueue_create_buffer[dtype](Self.FORCES_SIZE)
        var contacts_buf = ctx.enqueue_create_buffer[dtype](Self.CONTACTS_SIZE)
        var counts_buf = ctx.enqueue_create_buffer[dtype](Self.COUNTS_SIZE)
        return (bodies_buf, shapes_buf, forces_buf, contacts_buf, counts_buf)
