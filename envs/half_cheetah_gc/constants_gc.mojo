"""Half Cheetah GC environment constants.

Physics constants and body/joint parameters matching MuJoCo Half Cheetah.
Based on Gymnasium's half_cheetah.xml.

The Half Cheetah is a 2D planar robot (movement in XZ plane, rotation around Y axis)
with 7 bodies and 9 joints (3 root DOFs + 6 actuated joints).
"""


# ============================================================================
# Physics Parameters
# ============================================================================

comptime DT: Float64 = 0.002  # Physics timestep (500 Hz)
comptime FRAME_SKIP: Int = 5  # Number of physics steps per environment step
comptime EFFECTIVE_DT: Float64 = DT * FRAME_SKIP  # 0.01s per env step

comptime GRAVITY_Z: Float64 = -9.81  # Gravity acceleration (m/s²)
comptime GROUND_Z: Float64 = 0.0  # Ground plane height

comptime MAX_STEPS: Int = 1000  # Maximum episode length
comptime INIT_HEIGHT: Float64 = 0.7  # Initial torso z-position (matching MuJoCo)

# Contact physics
comptime FRICTION: Float64 = 0.9
comptime RESTITUTION: Float64 = 0.0


# ============================================================================
# Generalized Coordinates Dimensions
# ============================================================================

# Joint configuration:
#   rootx (slide), rootz (slide), rooty (hinge),
#   bthigh (hinge), bshin (hinge), bfoot (hinge),
#   fthigh (hinge), fshin (hinge), ffoot (hinge)

comptime NQ: Int = 9  # Position DOFs (all joints have 1 qpos each)
comptime NV: Int = 9  # Velocity DOFs (all joints have 1 qvel each)
comptime NBODY: Int = 7  # Number of rigid bodies
comptime NJOINT: Int = 9  # Number of joints
comptime MAX_CONTACTS: Int = 20  # Maximum contact points


# ============================================================================
# Body Indices
# ============================================================================

comptime BODY_TORSO: Int = 0
comptime BODY_BTHIGH: Int = 1  # Back thigh
comptime BODY_BSHIN: Int = 2  # Back shin
comptime BODY_BFOOT: Int = 3  # Back foot
comptime BODY_FTHIGH: Int = 4  # Front thigh
comptime BODY_FSHIN: Int = 5  # Front shin
comptime BODY_FFOOT: Int = 6  # Front foot


# ============================================================================
# Joint Indices (qpos/qvel addresses)
# ============================================================================

comptime JOINT_ROOTX: Int = 0  # Root x slide (unactuated)
comptime JOINT_ROOTZ: Int = 1  # Root z slide (unactuated)
comptime JOINT_ROOTY: Int = 2  # Root y hinge (unactuated)
comptime JOINT_BTHIGH: Int = 3  # Back thigh hinge (actuated)
comptime JOINT_BSHIN: Int = 4  # Back shin hinge (actuated)
comptime JOINT_BFOOT: Int = 5  # Back foot hinge (actuated)
comptime JOINT_FTHIGH: Int = 6  # Front thigh hinge (actuated)
comptime JOINT_FSHIN: Int = 7  # Front shin hinge (actuated)
comptime JOINT_FFOOT: Int = 8  # Front foot hinge (actuated)


# ============================================================================
# Body Geometry (Capsules)
# All bodies are capsules with radius 0.046m (matching MuJoCo)
# Values taken directly from MuJoCo XML size attributes (radius, half_length)
# ============================================================================

comptime CAPSULE_RADIUS: Float64 = 0.046

# Half-lengths for each body capsule (from MuJoCo XML size attribute)
comptime TORSO_HALF_LENGTH: Float64 = 0.5  # Torso extends from -0.5 to 0.5 along X
comptime HEAD_HALF_LENGTH: Float64 = 0.15  # Head capsule half-length
comptime BTHIGH_HALF_LENGTH: Float64 = 0.145  # Back thigh half-length
comptime BSHIN_HALF_LENGTH: Float64 = 0.15  # Back shin half-length
comptime BFOOT_HALF_LENGTH: Float64 = 0.094  # Back foot half-length
comptime FTHIGH_HALF_LENGTH: Float64 = 0.133  # Front thigh half-length
comptime FSHIN_HALF_LENGTH: Float64 = 0.106  # Front shin half-length
comptime FFOOT_HALF_LENGTH: Float64 = 0.07  # Front foot half-length

# Head position relative to torso (from MuJoCo XML: pos=".6 0 .1")
comptime HEAD_POS_X: Float64 = 0.6  # Forward of torso center
comptime HEAD_POS_Y: Float64 = 0.0
comptime HEAD_POS_Z: Float64 = 0.1  # Slightly above torso

# Head orientation (from MuJoCo XML: axisangle="0 1 0 .87")
# Tilted upward by 0.87 radians (~50 degrees) around Y axis
comptime HEAD_AXIS_ANGLE: Float64 = 0.87


# ============================================================================
# Body Masses
# Total mass is 14 kg, distributed across bodies
# Based on MuJoCo density calculations for capsules
# ============================================================================

comptime TORSO_MASS: Float64 = 6.25  # Heavier torso
comptime BTHIGH_MASS: Float64 = 1.54
comptime BSHIN_MASS: Float64 = 1.58
comptime BFOOT_MASS: Float64 = 1.10
comptime FTHIGH_MASS: Float64 = 1.43
comptime FSHIN_MASS: Float64 = 1.17
comptime FFOOT_MASS: Float64 = 0.93


# ============================================================================
# Joint Limits (radians)
# From MuJoCo XML joint range specifications
# ============================================================================

# Back thigh joint limits
comptime BTHIGH_LOWER: Float64 = -0.52
comptime BTHIGH_UPPER: Float64 = 1.05

# Back shin joint limits
comptime BSHIN_LOWER: Float64 = -0.785
comptime BSHIN_UPPER: Float64 = 0.785

# Back foot joint limits
comptime BFOOT_LOWER: Float64 = -0.4
comptime BFOOT_UPPER: Float64 = 0.785

# Front thigh joint limits
comptime FTHIGH_LOWER: Float64 = -1.0
comptime FTHIGH_UPPER: Float64 = 0.7

# Front shin joint limits
comptime FSHIN_LOWER: Float64 = -1.2
comptime FSHIN_UPPER: Float64 = 0.87

# Front foot joint limits
comptime FFOOT_LOWER: Float64 = -0.5
comptime FFOOT_UPPER: Float64 = 0.5


# ============================================================================
# Joint Gear Ratios (for torque scaling)
# From MuJoCo XML actuator gear specifications
# ============================================================================

comptime BTHIGH_GEAR: Float64 = 120.0
comptime BSHIN_GEAR: Float64 = 90.0
comptime BFOOT_GEAR: Float64 = 60.0
comptime FTHIGH_GEAR: Float64 = 120.0
comptime FSHIN_GEAR: Float64 = 60.0
comptime FFOOT_GEAR: Float64 = 30.0


# ============================================================================
# Joint Stiffness and Damping
# From MuJoCo XML joint specifications
# ============================================================================

comptime BTHIGH_STIFFNESS: Float64 = 240.0
comptime BTHIGH_DAMPING: Float64 = 6.0

comptime BSHIN_STIFFNESS: Float64 = 180.0
comptime BSHIN_DAMPING: Float64 = 4.5

comptime BFOOT_STIFFNESS: Float64 = 120.0
comptime BFOOT_DAMPING: Float64 = 3.0

comptime FTHIGH_STIFFNESS: Float64 = 180.0
comptime FTHIGH_DAMPING: Float64 = 4.5

comptime FSHIN_STIFFNESS: Float64 = 120.0
comptime FSHIN_DAMPING: Float64 = 3.0

comptime FFOOT_STIFFNESS: Float64 = 60.0
comptime FFOOT_DAMPING: Float64 = 1.5


# ============================================================================
# Joint Armature (rotor inertia for stability)
# ============================================================================

comptime JOINT_ARMATURE: Float64 = 0.1


# ============================================================================
# Reward Parameters
# ============================================================================

comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0
comptime CTRL_COST_WEIGHT: Float64 = 0.1


# ============================================================================
# Reset Noise Scale
# ============================================================================

comptime RESET_NOISE_SCALE: Float64 = 0.1


# ============================================================================
# Observation and Action Dimensions
# ============================================================================

comptime OBS_DIM: Int = 17  # 8 qpos (excluding rootx) + 9 qvel
comptime ACTION_DIM: Int = 6  # 6 actuated joints
