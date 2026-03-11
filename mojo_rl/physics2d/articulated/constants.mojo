"""Constants for 2D articulated body chains.

Extends physics_gpu/constants.mojo with additional constants for
multi-body articulated chains (for Hopper, Walker2d, HalfCheetah planar).
"""

# =============================================================================
# Articulated Chain Layout
# =============================================================================
# An articulated chain is a tree of bodies connected by joints.
# For MuJoCo-style robots, we use a specific layout:
#
# Chain data: [body_count, joint_count, root_body_idx, <bodies>, <joints>, <parent_map>]

# Chain header
comptime CHAIN_BODY_COUNT: Int = 0
comptime CHAIN_JOINT_COUNT: Int = 1
comptime CHAIN_ROOT_IDX: Int = 2
comptime CHAIN_HEADER_SIZE: Int = 3

# =============================================================================
# Link Layout (for articulated bodies)
# =============================================================================
# Each link in the chain has additional properties beyond basic body state

comptime LINK_DATA_SIZE: Int = 8

# Link indices within articulated chain
comptime LINK_PARENT_IDX: Int = 0     # Index of parent link (-1 for root)
comptime LINK_JOINT_IDX: Int = 1      # Index of joint connecting to parent
comptime LINK_CHILD_START: Int = 2    # Start index of children
comptime LINK_CHILD_COUNT: Int = 3    # Number of children
comptime LINK_LENGTH: Int = 4         # Link length (for rendering)
comptime LINK_WIDTH: Int = 5          # Link width (for rendering/collision)
comptime LINK_DENSITY: Int = 6        # Material density
comptime LINK_FLAGS: Int = 7          # Link flags

# Link flags
comptime LINK_FLAG_FOOT: Int = 1      # This link is a foot (can touch ground)
comptime LINK_FLAG_FIXED: Int = 2     # This link is fixed (no dynamics)

# =============================================================================
# Planar Locomotion Constants
# =============================================================================

# Hopper (4 bodies, 3 joints)
# Bodies: torso, thigh, leg, foot
# Joints: hip, knee, ankle
comptime HOPPER_NUM_BODIES: Int = 4
comptime HOPPER_NUM_JOINTS: Int = 3
comptime HOPPER_OBS_DIM: Int = 11
comptime HOPPER_ACTION_DIM: Int = 3

# Walker2d (7 bodies, 6 joints)
# Bodies: torso, thigh_r, leg_r, foot_r, thigh_l, leg_l, foot_l
# Joints: hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l
comptime WALKER_NUM_BODIES: Int = 7
comptime WALKER_NUM_JOINTS: Int = 6
comptime WALKER_OBS_DIM: Int = 17
comptime WALKER_ACTION_DIM: Int = 6

# HalfCheetah (7 bodies, 6 joints)
# Bodies: torso, bthigh, bshin, bfoot, fthigh, fshin, ffoot
# Joints: bhip, bknee, bfoot, fhip, fknee, ffoot
comptime CHEETAH_NUM_BODIES: Int = 7
comptime CHEETAH_NUM_JOINTS: Int = 6
comptime CHEETAH_OBS_DIM: Int = 17
comptime CHEETAH_ACTION_DIM: Int = 6

# =============================================================================
# Motor Control Constants
# =============================================================================

# Default PD controller gains
comptime DEFAULT_KP: Float64 = 100.0
comptime DEFAULT_KD: Float64 = 10.0
comptime DEFAULT_MAX_TORQUE: Float64 = 100.0

# Joint limits (radians)
comptime DEFAULT_JOINT_LOWER: Float64 = -2.356  # -135 degrees
comptime DEFAULT_JOINT_UPPER: Float64 = 0.785   # +45 degrees

# =============================================================================
# Reward Constants
# =============================================================================

# Forward velocity reward coefficient
comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0

# Control cost coefficient
comptime CTRL_COST_WEIGHT: Float64 = 0.001

# Healthy reward (alive bonus)
comptime HEALTHY_REWARD: Float64 = 1.0

# Health bounds (for termination)
comptime HEALTHY_Z_MIN: Float64 = 0.7    # Minimum torso height
comptime HEALTHY_Z_MAX: Float64 = 1000.0  # Maximum torso height
comptime HEALTHY_ANGLE_MIN: Float64 = -1.0  # Min torso angle
comptime HEALTHY_ANGLE_MAX: Float64 = 1.0   # Max torso angle
