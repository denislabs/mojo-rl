"""PhysicsLayout3D - Compile-time layout for 3D physics buffers.

This module provides compile-time computation of buffer sizes and offsets
for the flat 2D [BATCH, STATE_SIZE] layout required by GPU environments.
"""

from .constants import (
    BODY_STATE_SIZE_3D,
    SHAPE_MAX_SIZE_3D,
    CONTACT_DATA_SIZE_3D,
    JOINT_DATA_SIZE_3D,
    MAX_JOINTS_PER_ENV_3D,
)


struct PhysicsLayout3D[
    NUM_BODIES: Int,
    MAX_CONTACTS: Int = 32,
    MAX_JOINTS: Int = MAX_JOINTS_PER_ENV_3D,
    OBS_DIM: Int = 27,
    METADATA_SIZE: Int = 8,
    NUM_SHAPES: Int = 8,
]:
    """Compile-time layout calculator for 3D physics buffers.

    Parameters:
        NUM_BODIES: Number of rigid bodies per environment.
        MAX_CONTACTS: Maximum contact points per environment.
        MAX_JOINTS: Maximum joints per environment.
        OBS_DIM: Observation dimension.
        METADATA_SIZE: Size of metadata (step_count, total_reward, etc.).
        NUM_SHAPES: Number of shape definitions.

    Layout (offsets are cumulative):
        [observation | bodies | joints | joint_count | contacts | contact_count | metadata]
    """

    # =========================================================================
    # Component Sizes
    # =========================================================================

    # Bodies: NUM_BODIES * 26 floats per body
    comptime BODIES_SIZE: Int = Self.NUM_BODIES * BODY_STATE_SIZE_3D

    # Joints: MAX_JOINTS * 32 floats per joint
    comptime JOINTS_SIZE: Int = Self.MAX_JOINTS * JOINT_DATA_SIZE_3D

    # Contacts: MAX_CONTACTS * 16 floats per contact
    comptime CONTACTS_SIZE: Int = Self.MAX_CONTACTS * CONTACT_DATA_SIZE_3D

    # Shapes: Shared across all environments (separate buffer)
    comptime SHAPES_SIZE: Int = Self.NUM_SHAPES * SHAPE_MAX_SIZE_3D

    # =========================================================================
    # Offsets within Environment State (cumulative)
    # =========================================================================

    # Observation at the start
    comptime OBS_OFFSET: Int = 0

    # Bodies follow observation
    comptime BODIES_OFFSET: Int = Self.OBS_DIM

    # Joints follow bodies
    comptime JOINTS_OFFSET: Int = Self.BODIES_OFFSET + Self.BODIES_SIZE

    # Joint count (single value)
    comptime JOINT_COUNT_OFFSET: Int = Self.JOINTS_OFFSET + Self.JOINTS_SIZE

    # Contacts follow joint count
    comptime CONTACTS_OFFSET: Int = Self.JOINT_COUNT_OFFSET + 1

    # Contact count (single value)
    comptime CONTACT_COUNT_OFFSET: Int = Self.CONTACTS_OFFSET + Self.CONTACTS_SIZE

    # Metadata at the end
    comptime METADATA_OFFSET: Int = Self.CONTACT_COUNT_OFFSET + 1

    # =========================================================================
    # Total State Size
    # =========================================================================

    comptime STATE_SIZE: Int = Self.METADATA_OFFSET + Self.METADATA_SIZE

    # =========================================================================
    # Utility Methods - Body Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn body_offset(body: Int) -> Int:
        """Compute offset for a specific body within environment state."""
        return Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D

    @staticmethod
    @always_inline
    fn body_field_offset(body: Int, field: Int) -> Int:
        """Compute offset for a specific field of a body."""
        return Self.BODIES_OFFSET + body * BODY_STATE_SIZE_3D + field

    # =========================================================================
    # Utility Methods - Joint Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn joint_offset(joint: Int) -> Int:
        """Compute offset for a specific joint within environment state."""
        return Self.JOINTS_OFFSET + joint * JOINT_DATA_SIZE_3D

    @staticmethod
    @always_inline
    fn joint_field_offset(joint: Int, field: Int) -> Int:
        """Compute offset for a specific field of a joint."""
        return Self.JOINTS_OFFSET + joint * JOINT_DATA_SIZE_3D + field

    # =========================================================================
    # Utility Methods - Contact Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn contact_offset(contact: Int) -> Int:
        """Compute offset for a specific contact within environment state."""
        return Self.CONTACTS_OFFSET + contact * CONTACT_DATA_SIZE_3D

    @staticmethod
    @always_inline
    fn contact_field_offset(contact: Int, field: Int) -> Int:
        """Compute offset for a specific field of a contact."""
        return Self.CONTACTS_OFFSET + contact * CONTACT_DATA_SIZE_3D + field

    # =========================================================================
    # Utility Methods - Metadata Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn metadata_offset(field: Int) -> Int:
        """Compute offset for a metadata field."""
        return Self.METADATA_OFFSET + field


# =============================================================================
# Common Layout Aliases for MuJoCo-style Environments
# =============================================================================

# Hopper: 4 bodies, 3 joints
comptime HopperLayout3D = PhysicsLayout3D[
    NUM_BODIES=4,
    MAX_CONTACTS=8,
    MAX_JOINTS=3,
    OBS_DIM=11,
    METADATA_SIZE=8,
    NUM_SHAPES=4,
]

# Walker2d: 7 bodies, 6 joints
comptime Walker2dLayout3D = PhysicsLayout3D[
    NUM_BODIES=7,
    MAX_CONTACTS=16,
    MAX_JOINTS=6,
    OBS_DIM=17,
    METADATA_SIZE=8,
    NUM_SHAPES=7,
]

# HalfCheetah: 7 bodies, 6 joints
comptime HalfCheetahLayout3D = PhysicsLayout3D[
    NUM_BODIES=7,
    MAX_CONTACTS=12,
    MAX_JOINTS=6,
    OBS_DIM=17,
    METADATA_SIZE=8,
    NUM_SHAPES=7,
]

# Ant: 13 bodies, 8 joints
comptime AntLayout3D = PhysicsLayout3D[
    NUM_BODIES=13,
    MAX_CONTACTS=32,
    MAX_JOINTS=8,
    OBS_DIM=27,
    METADATA_SIZE=8,
    NUM_SHAPES=13,
]

# Humanoid: 13 bodies, 17 joints
comptime HumanoidLayout3D = PhysicsLayout3D[
    NUM_BODIES=13,
    MAX_CONTACTS=32,
    MAX_JOINTS=17,
    OBS_DIM=376,
    METADATA_SIZE=8,
    NUM_SHAPES=13,
]
