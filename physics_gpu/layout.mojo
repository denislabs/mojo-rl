"""PhysicsLayout - Compile-time buffer size computation for GPU physics.

This module provides compile-time computation of buffer sizes and offsets,
similar to how deep_rl models compute PARAM_SIZE and CACHE_SIZE.

The layout defines:
- BODIES_SIZE: Total floats for body state [BATCH, NUM_BODIES, BODY_STATE_SIZE]
- SHAPES_SIZE: Total floats for shapes [NUM_SHAPES, SHAPE_MAX_SIZE]
- FORCES_SIZE: Total floats for forces [BATCH, NUM_BODIES, 3]
- CONTACTS_SIZE: Total floats for contacts [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE]
- COUNTS_SIZE: Total floats for contact counts [BATCH]
- TOTAL_SIZE: Combined size for a unified buffer (optional)

Example:
    ```mojo
    from physics_gpu.layout import PhysicsLayout

    # Define layout for 1024 envs, 3 bodies, 2 shapes, up to 16 contacts
    alias Layout = PhysicsLayout[1024, 3, 2, 16]

    # Access sizes at compile time
    comptime bodies_size = Layout.BODIES_SIZE  # 1024 * 3 * 13
    comptime total_size = Layout.TOTAL_SIZE
    ```
"""

from .constants import (
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    JOINT_DATA_SIZE,
    MAX_JOINTS_PER_ENV,
)


struct PhysicsLayout[
    BATCH: Int,
    NUM_BODIES: Int,
    NUM_SHAPES: Int,
    MAX_CONTACTS: Int = 16,
    MAX_JOINTS: Int = MAX_JOINTS_PER_ENV,
]:
    """Compile-time buffer layout for GPU physics.

    This struct is purely for compile-time constants - it has no runtime state.
    Use it to compute buffer sizes and offsets for physics state.

    Parameters:
        BATCH: Number of parallel environments.
        NUM_BODIES: Number of bodies per environment.
        NUM_SHAPES: Total number of shape definitions (shared across envs).
        MAX_CONTACTS: Maximum contacts per environment.
        MAX_JOINTS: Maximum joints per environment.
    """

    # =========================================================================
    # Individual Buffer Sizes
    # =========================================================================

    # Bodies: [BATCH, NUM_BODIES, BODY_STATE_SIZE]
    # Each body stores: x, y, angle, vx, vy, omega, fx, fy, tau, mass, inv_mass, inv_inertia, shape_idx
    comptime BODIES_SIZE: Int = Self.BATCH * Self.NUM_BODIES * BODY_STATE_SIZE

    # Shapes: [NUM_SHAPES, SHAPE_MAX_SIZE]
    # Shape definitions are shared across all environments
    comptime SHAPES_SIZE: Int = Self.NUM_SHAPES * SHAPE_MAX_SIZE

    # Forces: [BATCH, NUM_BODIES, 3] (fx, fy, torque)
    # Accumulated forces cleared after each physics step
    comptime FORCES_SIZE: Int = Self.BATCH * Self.NUM_BODIES * 3

    # Contacts: [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE]
    # Contact manifold data for constraint solving
    comptime CONTACTS_SIZE: Int = Self.BATCH * Self.MAX_CONTACTS * CONTACT_DATA_SIZE

    # Contact counts: [BATCH]
    # Number of active contacts per environment
    comptime COUNTS_SIZE: Int = Self.BATCH

    # Joints: [BATCH, MAX_JOINTS, JOINT_DATA_SIZE]
    # Joint constraint data for each environment
    comptime JOINTS_SIZE: Int = Self.BATCH * Self.MAX_JOINTS * JOINT_DATA_SIZE

    # Joint counts: [BATCH]
    # Number of active joints per environment
    comptime JOINT_COUNTS_SIZE: Int = Self.BATCH

    # =========================================================================
    # Unified Buffer Layout (Option A)
    # =========================================================================
    # If using a single unified buffer, these are the offsets

    comptime BODIES_OFFSET: Int = 0
    comptime SHAPES_OFFSET: Int = Self.BODIES_OFFSET + Self.BODIES_SIZE
    comptime FORCES_OFFSET: Int = Self.SHAPES_OFFSET + Self.SHAPES_SIZE
    comptime CONTACTS_OFFSET: Int = Self.FORCES_OFFSET + Self.FORCES_SIZE
    comptime COUNTS_OFFSET: Int = Self.CONTACTS_OFFSET + Self.CONTACTS_SIZE
    comptime JOINTS_OFFSET: Int = Self.COUNTS_OFFSET + Self.COUNTS_SIZE
    comptime JOINT_COUNTS_OFFSET: Int = Self.JOINTS_OFFSET + Self.JOINTS_SIZE

    # Total size for unified buffer
    comptime TOTAL_SIZE: Int = Self.JOINT_COUNTS_OFFSET + Self.JOINT_COUNTS_SIZE

    # =========================================================================
    # Per-Environment Sizes (for partial state operations)
    # =========================================================================

    # Size of physics state for a single environment (useful for reset)
    comptime SINGLE_ENV_BODIES_SIZE: Int = Self.NUM_BODIES * BODY_STATE_SIZE
    comptime SINGLE_ENV_FORCES_SIZE: Int = Self.NUM_BODIES * 3
    comptime SINGLE_ENV_CONTACTS_SIZE: Int = Self.MAX_CONTACTS * CONTACT_DATA_SIZE
    comptime SINGLE_ENV_JOINTS_SIZE: Int = Self.MAX_JOINTS * JOINT_DATA_SIZE

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    @always_inline
    fn body_offset(env: Int, body: Int) -> Int:
        """Compute flat buffer offset for a specific body in an environment.

        Args:
            env: Environment index.
            body: Body index within environment.

        Returns:
            Offset in the bodies buffer.
        """
        return (env * Self.NUM_BODIES + body) * BODY_STATE_SIZE

    @staticmethod
    @always_inline
    fn contact_offset(env: Int, contact: Int) -> Int:
        """Compute flat buffer offset for a specific contact.

        Args:
            env: Environment index.
            contact: Contact index within environment.

        Returns:
            Offset in the contacts buffer.
        """
        return (env * Self.MAX_CONTACTS + contact) * CONTACT_DATA_SIZE

    @staticmethod
    @always_inline
    fn force_offset(env: Int, body: Int) -> Int:
        """Compute flat buffer offset for a body's forces.

        Args:
            env: Environment index.
            body: Body index within environment.

        Returns:
            Offset in the forces buffer.
        """
        return (env * Self.NUM_BODIES + body) * 3

    @staticmethod
    @always_inline
    fn joint_offset(env: Int, joint: Int) -> Int:
        """Compute flat buffer offset for a specific joint.

        Args:
            env: Environment index.
            joint: Joint index within environment.

        Returns:
            Offset in the joints buffer.
        """
        return (env * Self.MAX_JOINTS + joint) * JOINT_DATA_SIZE


# =========================================================================
# Common Layout Aliases
# =========================================================================

# LunarLander: 1 body (lander), 1 shape (hexagon), up to 4 contacts
comptime LunarLanderLayout = PhysicsLayout[1, 1, 1, 4]

# CartPole: 2 bodies (cart + pole), 2 shapes, up to 2 contacts
comptime CartPoleLayout = PhysicsLayout[1, 2, 2, 2]
