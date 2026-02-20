"""Ant as a compile-time model definition.

Defines all 13 bodies and 9 joints as type aliases using BodySpec/JointSpec,
composed into AntModel via ModelDef. MuJoCo Ant-v5 quadruped with free joint root.

Body/joint values match MuJoCo ant.xml and Gymnasium Ant-v5.
Uses inertiafromgeom=true with density=5.0 for all geoms.

Dimensions: NQ=15, NV=14, OBS_DIM=27, ACTION_DIM=8.
"""

from physics3d.model.body_spec import CapsuleBody, SphereBody
from physics3d.model.joint_spec import HingeJoint, SlideJoint, FreeJoint
from physics3d.model import (
    Bodies,
    Joints,
    Geoms,
    Textures,
    Materials,
    Actuators,
    ModelDef,
    ModelDefaults,
    Lights,
    Cameras,
)
from physics3d.model.actuator_spec import MotorActuator
from physics3d.model.geom_spec import Plane, Capsule, Sphere
from physics3d.model.texture_spec import CheckerTexture, GradientTexture
from physics3d.model.material_spec import Material
from physics3d.model.camera_spec import TrackCamera
from physics3d.model.light_spec import DirectionalLight
from render import Color
from physics3d.gpu.constants import (
    state_size,
    model_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    contacts_offset,
    metadata_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    model_curriculum_offset,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_CURRICULUM_SIZE,
    CONTACT_SIZE,
)


# =============================================================================
# Body Type Aliases
#
# Ant XML: 13 bodies + worldbody (14 total).
# inertiafromgeom=true → body mass/inertia computed from child geoms in finalize.
# Bodies 2, 5, 8, 11 (front_left_leg, front_right_leg, back_leg, right_back_leg)
# have NO joints — welded to torso.
# =============================================================================

# Ant default color from XML: rgba="0.8 0.6 0.4 1"
comptime ANT_COLOR = Color(204, 153, 102, 255)

# Body 1: Torso — sphere, root of kinematic tree
# MuJoCo: <body name="torso" pos="0 0 0.75">
# Free joint starts at init_qpos z=0.55
comptime AntTorso = SphereBody[
    parent=0,
    mass=1.0,  # Overridden by inertiafromgeom
    name="torso",
    radius=0.25,
    pos_z=0.75,
    conaffinity=0,  # XML default geom: conaffinity=0
    color=ANT_COLOR,
]

# Body 2: front_left_leg — welded to torso (no joint), at same position
# MuJoCo: <body name="front_left_leg" pos="0 0 0">
comptime AntFrontLeftLeg = CapsuleBody[
    parent=1,
    mass=1.0,  # Overridden
    name="front_left_leg",
    radius=0.08,
    half_length=0.141421,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 3: aux_1 — hip_1 joint, child of front_left_leg
# MuJoCo: <body name="aux_1" pos="0.2 0.2 0">
comptime AntAux1 = CapsuleBody[
    parent=2,
    mass=1.0,
    name="aux_1",
    radius=0.08,
    half_length=0.141421,
    pos_x=0.2,
    pos_y=0.2,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 4: ankle_1_body — ankle_1 joint
# MuJoCo: <body pos="0.2 0.2 0"> (child of aux_1)
comptime AntAnkle1Body = CapsuleBody[
    parent=3,
    mass=1.0,
    name="ankle_1_body",
    radius=0.08,
    half_length=0.282843,
    pos_x=0.2,
    pos_y=0.2,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 5: front_right_leg — welded to torso (no joint)
# MuJoCo: <body name="front_right_leg" pos="0 0 0">
comptime AntFrontRightLeg = CapsuleBody[
    parent=1,
    mass=1.0,
    name="front_right_leg",
    radius=0.08,
    half_length=0.141421,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 6: aux_2 — hip_2 joint
# MuJoCo: <body name="aux_2" pos="-0.2 0.2 0">
comptime AntAux2 = CapsuleBody[
    parent=5,
    mass=1.0,
    name="aux_2",
    radius=0.08,
    half_length=0.141421,
    pos_x=-0.2,
    pos_y=0.2,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 7: ankle_2_body — ankle_2 joint
# MuJoCo: <body pos="-0.2 0.2 0">
comptime AntAnkle2Body = CapsuleBody[
    parent=6,
    mass=1.0,
    name="ankle_2_body",
    radius=0.08,
    half_length=0.282843,
    pos_x=-0.2,
    pos_y=0.2,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 8: back_leg — welded to torso (no joint)
# MuJoCo: <body name="back_leg" pos="0 0 0">
comptime AntBackLeg = CapsuleBody[
    parent=1,
    mass=1.0,
    name="back_leg",
    radius=0.08,
    half_length=0.141421,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 9: aux_3 — hip_3 joint
# MuJoCo: <body name="aux_3" pos="-0.2 -0.2 0">
comptime AntAux3 = CapsuleBody[
    parent=8,
    mass=1.0,
    name="aux_3",
    radius=0.08,
    half_length=0.141421,
    pos_x=-0.2,
    pos_y=-0.2,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 10: ankle_3_body — ankle_3 joint
# MuJoCo: <body pos="-0.2 -0.2 0">
comptime AntAnkle3Body = CapsuleBody[
    parent=9,
    mass=1.0,
    name="ankle_3_body",
    radius=0.08,
    half_length=0.282843,
    pos_x=-0.2,
    pos_y=-0.2,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 11: right_back_leg — welded to torso (no joint)
# MuJoCo: <body name="right_back_leg" pos="0 0 0">
comptime AntRightBackLeg = CapsuleBody[
    parent=1,
    mass=1.0,
    name="right_back_leg",
    radius=0.08,
    half_length=0.141421,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 12: aux_4 — hip_4 joint
# MuJoCo: <body name="aux_4" pos="0.2 -0.2 0">
comptime AntAux4 = CapsuleBody[
    parent=11,
    mass=1.0,
    name="aux_4",
    radius=0.08,
    half_length=0.141421,
    pos_x=0.2,
    pos_y=-0.2,
    conaffinity=0,
    color=ANT_COLOR,
]

# Body 13: ankle_4_body — ankle_4 joint
# MuJoCo: <body pos="0.2 -0.2 0">
comptime AntAnkle4Body = CapsuleBody[
    parent=12,
    mass=1.0,
    name="ankle_4_body",
    radius=0.08,
    half_length=0.282843,
    pos_x=0.2,
    pos_y=-0.2,
    conaffinity=0,
    color=ANT_COLOR,
]


# =============================================================================
# Joint Type Aliases
#
# Joint 0: root (free) — body 1 (torso), 7 qpos, 6 qvel
# Joints 1-8: hinges for hip/ankle — 1 qpos, 1 qvel each
# Total: NQ=15, NV=14
#
# All hinges: armature=1, damping=1 (from XML defaults)
# Angle ranges in degrees in XML, converted to radians here.
# init_qpos from XML custom numeric (MuJoCo: scalar-first quat, we use scalar-last)
# =============================================================================

# Joint 0: root (free joint on torso)
# init_qpos: pos=(0,0,0.55), identity quat
# Exclude x,y from obs (num_excluded_qpos=2) → 5 qpos in obs
comptime AntRootJ = FreeJoint[
    body_idx=1,
    init_pos_x=0.0,
    init_pos_y=0.0,
    init_pos_z=0.55,
    num_excluded_qpos=2,
]

# Joint 1: hip_1 — hinge Z-axis on aux_1 (body 3)
# XML: axis="0 0 1", range="-30 30" deg = [-0.5236, 0.5236] rad
comptime AntHip1J = HingeJoint[
    body_idx=3,
    axis_x=0.0,
    axis_y=0.0,
    axis_z=1.0,
    range_min=-0.5236,
    range_max=0.5236,
    armature=1.0,
    damping=1.0,
    init_qpos=0.0,
    tau_limit=150.0,
]

# Joint 2: ankle_1 — hinge on body 4
# XML: axis="-1 1 0", range="30 70" deg = [0.5236, 1.2217] rad
comptime AntAnkle1J = HingeJoint[
    body_idx=4,
    axis_x=-1.0,
    axis_y=1.0,
    axis_z=0.0,
    range_min=0.5236,
    range_max=1.2217,
    armature=1.0,
    damping=1.0,
    init_qpos=1.0,
    tau_limit=150.0,
]

# Joint 3: hip_2 — hinge Z-axis on aux_2 (body 6)
comptime AntHip2J = HingeJoint[
    body_idx=6,
    axis_x=0.0,
    axis_y=0.0,
    axis_z=1.0,
    range_min=-0.5236,
    range_max=0.5236,
    armature=1.0,
    damping=1.0,
    init_qpos=0.0,
    tau_limit=150.0,
]

# Joint 4: ankle_2 — hinge on body 7
# XML: axis="1 1 0", range="-70 -30" deg = [-1.2217, -0.5236] rad
comptime AntAnkle2J = HingeJoint[
    body_idx=7,
    axis_x=1.0,
    axis_y=1.0,
    axis_z=0.0,
    range_min=-1.2217,
    range_max=-0.5236,
    armature=1.0,
    damping=1.0,
    init_qpos=-1.0,
    tau_limit=150.0,
]

# Joint 5: hip_3 — hinge Z-axis on aux_3 (body 9)
comptime AntHip3J = HingeJoint[
    body_idx=9,
    axis_x=0.0,
    axis_y=0.0,
    axis_z=1.0,
    range_min=-0.5236,
    range_max=0.5236,
    armature=1.0,
    damping=1.0,
    init_qpos=0.0,
    tau_limit=150.0,
]

# Joint 6: ankle_3 — hinge on body 10
# XML: axis="-1 1 0", range="-70 -30" deg
comptime AntAnkle3J = HingeJoint[
    body_idx=10,
    axis_x=-1.0,
    axis_y=1.0,
    axis_z=0.0,
    range_min=-1.2217,
    range_max=-0.5236,
    armature=1.0,
    damping=1.0,
    init_qpos=-1.0,
    tau_limit=150.0,
]

# Joint 7: hip_4 — hinge Z-axis on aux_4 (body 12)
comptime AntHip4J = HingeJoint[
    body_idx=12,
    axis_x=0.0,
    axis_y=0.0,
    axis_z=1.0,
    range_min=-0.5236,
    range_max=0.5236,
    armature=1.0,
    damping=1.0,
    init_qpos=0.0,
    tau_limit=150.0,
]

# Joint 8: ankle_4 — hinge on body 13
# XML: axis="1 1 0", range="30 70" deg
comptime AntAnkle4J = HingeJoint[
    body_idx=13,
    axis_x=1.0,
    axis_y=1.0,
    axis_z=0.0,
    range_min=0.5236,
    range_max=1.2217,
    armature=1.0,
    damping=1.0,
    init_qpos=1.0,
    tau_limit=150.0,
]


# =============================================================================
# Textures
# =============================================================================
comptime AntSkyboxTex = GradientTexture[
    name="skybox",
    rgb1_r=1.0, rgb1_g=1.0, rgb1_b=1.0,
    rgb2_r=0.0, rgb2_g=0.0, rgb2_b=0.0,
]
comptime AntCheckerTex = CheckerTexture[
    name="texplane",
    rgb1_r=0.0, rgb1_g=0.0, rgb1_b=0.0,
    rgb2_r=0.8, rgb2_g=0.8, rgb2_b=0.8,
    repeat_x=60.0, repeat_y=60.0,
]
comptime AntTextures = Textures[AntSkyboxTex, AntCheckerTex]


# =============================================================================
# Materials
# =============================================================================
comptime AntMatPlane = Material[
    name="MatPlane",
    shininess=1.0, specular=1.0, reflectance=0.5,
    has_texture=True, texture_name="texplane",
]
comptime AntMatGeom = Material[
    name="geom",
    shininess=0.5, specular=0.5, reflectance=0.0,
]
comptime AntMaterials = Materials[AntMatPlane, AntMatGeom]


# =============================================================================
# Geom Definitions
#
# Ground plane + torso sphere + 12 capsule geoms (fromto converted to pos/quat).
# All capsules: radius=0.08, density=5.0, conaffinity=0, condim=3,
#   friction=1.0, friction_spin=0.5, friction_roll=0.5, margin=0.01
# =============================================================================

comptime AntGroundGeom = Plane[
    z=0.0,
    friction=1.0,
    conaffinity=1,  # XML: ground has conaffinity=1
    condim=3,
    size_x=40.0,
    size_y=40.0,
    material_name="MatPlane",
    shininess=AntMatPlane.SHININESS,
    specular=AntMatPlane.SPECULAR,
    reflectance=AntMatPlane.REFLECTANCE,
]

# Torso sphere (body 1)
comptime AntTorsoGeom = Sphere[
    body_idx=1,
    radius=0.25,
    color=ANT_COLOR,
    material_name="geom",
]

# Capsule geoms — fromto converted to (center_pos, quat, half_length)
# Leg 1: front-left
comptime AntAux1Geom = Capsule[
    body_idx=2,
    radius=0.08,
    half_length=0.141421,
    pos_x=0.1, pos_y=0.1,
    quat_x=-0.5, quat_y=0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]
comptime AntLeftLegGeom = Capsule[
    body_idx=3,
    radius=0.08,
    half_length=0.141421,
    pos_x=0.1, pos_y=0.1,
    quat_x=-0.5, quat_y=0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]
comptime AntLeftAnkleGeom = Capsule[
    body_idx=4,
    radius=0.08,
    half_length=0.282843,
    pos_x=0.2, pos_y=0.2,
    quat_x=-0.5, quat_y=0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]

# Leg 2: front-right
comptime AntAux2Geom = Capsule[
    body_idx=5,
    radius=0.08,
    half_length=0.141421,
    pos_x=-0.1, pos_y=0.1,
    quat_x=-0.5, quat_y=-0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]
comptime AntRightLegGeom = Capsule[
    body_idx=6,
    radius=0.08,
    half_length=0.141421,
    pos_x=-0.1, pos_y=0.1,
    quat_x=-0.5, quat_y=-0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]
comptime AntRightAnkleGeom = Capsule[
    body_idx=7,
    radius=0.08,
    half_length=0.282843,
    pos_x=-0.2, pos_y=0.2,
    quat_x=-0.5, quat_y=-0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]

# Leg 3: back-left
comptime AntAux3Geom = Capsule[
    body_idx=8,
    radius=0.08,
    half_length=0.141421,
    pos_x=-0.1, pos_y=-0.1,
    quat_x=0.5, quat_y=-0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]
comptime AntBackLegGeom = Capsule[
    body_idx=9,
    radius=0.08,
    half_length=0.141421,
    pos_x=-0.1, pos_y=-0.1,
    quat_x=0.5, quat_y=-0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]
comptime AntThirdAnkleGeom = Capsule[
    body_idx=10,
    radius=0.08,
    half_length=0.282843,
    pos_x=-0.2, pos_y=-0.2,
    quat_x=0.5, quat_y=-0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]

# Leg 4: back-right
comptime AntAux4Geom = Capsule[
    body_idx=11,
    radius=0.08,
    half_length=0.141421,
    pos_x=0.1, pos_y=-0.1,
    quat_x=0.5, quat_y=0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]
comptime AntRightBackLegGeom = Capsule[
    body_idx=12,
    radius=0.08,
    half_length=0.141421,
    pos_x=0.1, pos_y=-0.1,
    quat_x=0.5, quat_y=0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]
comptime AntFourthAnkleGeom = Capsule[
    body_idx=13,
    radius=0.08,
    half_length=0.282843,
    pos_x=0.2, pos_y=-0.2,
    quat_x=0.5, quat_y=0.5, quat_w=0.7071,
    color=ANT_COLOR, material_name="geom",
]


# =============================================================================
# Composed Model
# =============================================================================

comptime AntBodies = Bodies[
    AntTorso,
    AntFrontLeftLeg, AntAux1, AntAnkle1Body,
    AntFrontRightLeg, AntAux2, AntAnkle2Body,
    AntBackLeg, AntAux3, AntAnkle3Body,
    AntRightBackLeg, AntAux4, AntAnkle4Body,
]

comptime AntJoints = Joints[
    AntRootJ,
    AntHip1J, AntAnkle1J,
    AntHip2J, AntAnkle2J,
    AntHip3J, AntAnkle3J,
    AntHip4J, AntAnkle4J,
]

comptime AntGeoms = Geoms[
    AntGroundGeom,
    AntTorsoGeom,
    AntAux1Geom, AntLeftLegGeom, AntLeftAnkleGeom,
    AntAux2Geom, AntRightLegGeom, AntRightAnkleGeom,
    AntAux3Geom, AntBackLegGeom, AntThirdAnkleGeom,
    AntAux4Geom, AntRightBackLegGeom, AntFourthAnkleGeom,
]


# =============================================================================
# Actuators
#
# XML actuator order: hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3
# All: gear=150, ctrllimited, ctrlrange=[-1,1]
# joint_idx → position in AntJoints (0-indexed), dof_adr → dof offset
# =============================================================================

# Actuator order matching XML:
# hip_4 (joint_idx=7, dof=12), ankle_4 (joint_idx=8, dof=13)
# hip_1 (joint_idx=1, dof=6),  ankle_1 (joint_idx=2, dof=7)
# hip_2 (joint_idx=3, dof=8),  ankle_2 (joint_idx=4, dof=9)
# hip_3 (joint_idx=5, dof=10), ankle_3 (joint_idx=6, dof=11)
comptime AntHip4Motor = MotorActuator[joint_idx=7, dof_adr=12, gear=150.0]
comptime AntAnkle4Motor = MotorActuator[joint_idx=8, dof_adr=13, gear=150.0]
comptime AntHip1Motor = MotorActuator[joint_idx=1, dof_adr=6, gear=150.0]
comptime AntAnkle1Motor = MotorActuator[joint_idx=2, dof_adr=7, gear=150.0]
comptime AntHip2Motor = MotorActuator[joint_idx=3, dof_adr=8, gear=150.0]
comptime AntAnkle2Motor = MotorActuator[joint_idx=4, dof_adr=9, gear=150.0]
comptime AntHip3Motor = MotorActuator[joint_idx=5, dof_adr=10, gear=150.0]
comptime AntAnkle3Motor = MotorActuator[joint_idx=6, dof_adr=11, gear=150.0]

comptime AntActuators = Actuators[
    AntHip4Motor, AntAnkle4Motor,
    AntHip1Motor, AntAnkle1Motor,
    AntHip2Motor, AntAnkle2Motor,
    AntHip3Motor, AntAnkle3Motor,
]


# =============================================================================
# Camera + Light
# =============================================================================
# MuJoCo: <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
comptime AntCamera = TrackCamera[pos_y=-3.0, pos_z=0.3]
comptime AntLight = DirectionalLight[]
comptime AntLights = Lights[AntLight]
comptime AntCameras = Cameras[AntCamera]


# =============================================================================
# Defaults
# =============================================================================
comptime AntDefaults = ModelDefaults[
    geom_friction=1.0,
    geom_friction_spin=0.5,
    geom_friction_roll=0.5,
    geom_condim=3,
    geom_margin=0.01,
    geom_density=5.0,
    geom_conaffinity=0,
    geom_solref_0=0.02,
    geom_solref_1=1.0,
    geom_solimp_0=0.9,
    geom_solimp_1=0.95,
    geom_solimp_2=0.001,
    joint_armature=1.0,
    joint_damping=1.0,
    joint_solref_limit_0=0.02,
    joint_solref_limit_1=1.0,
    joint_solimp_limit_0=0.0,
    joint_solimp_limit_1=0.8,
    joint_solimp_limit_2=0.03,
    gravity_z=-9.81,
    timestep=0.01,
    inertiafromgeom=True,
]


comptime AntModel = ModelDef[
    AntBodies,
    AntJoints,
    AntGeoms,
    AntActuators,
    AntDefaults,
    AntLights,
    AntTextures,
    AntMaterials,
    AntCameras,
    max_equality=0,
    max_contacts=40,
]


# =============================================================================
# AntParams — Environment-Specific Parameters
# =============================================================================

struct AntParams[DTYPE: DType = DType.float64]:
    """Environment-specific parameters for Ant-v5."""

    # Physics
    comptime DT: Scalar[Self.DTYPE] = 0.01
    comptime FRAME_SKIP: Int = 5
    comptime GRAVITY_Z: Scalar[Self.DTYPE] = -9.81
    comptime MAX_CONTACTS: Int = 40

    # Reward
    comptime FORWARD_REWARD_WEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime CTRL_COST_WEIGHT: Scalar[Self.DTYPE] = 0.5
    comptime HEALTHY_REWARD: Scalar[Self.DTYPE] = 1.0
    comptime CONTACT_COST_WEIGHT: Scalar[Self.DTYPE] = 5e-4

    # Termination
    comptime MIN_HEIGHT: Scalar[Self.DTYPE] = 0.2
    comptime MAX_HEIGHT: Scalar[Self.DTYPE] = 1.0
    comptime MAX_STEPS: Int = 1000

    # Curriculum
    comptime CURRICULUM_INITIAL_MIN_HEIGHT: Scalar[Self.DTYPE] = 0.1
    comptime CURRICULUM_INITIAL_MAX_HEIGHT: Scalar[Self.DTYPE] = 1.5
    comptime CURRICULUM_FINAL_MIN_HEIGHT: Scalar[Self.DTYPE] = 0.2
    comptime CURRICULUM_FINAL_MAX_HEIGHT: Scalar[Self.DTYPE] = 1.0

    # Reset
    comptime RESET_NOISE_SCALE: Scalar[Self.DTYPE] = 0.1

    # Dimensions
    comptime NQ: Int = AntModel.NQ  # 15
    comptime NV: Int = AntModel.NV  # 14
    comptime NUM_BODIES: Int = AntModel.NBODY  # 14 (13 + worldbody)
    comptime NUM_JOINTS: Int = AntModel.NJOINT  # 9
    comptime NGEOM: Int = AntModel.NGEOM  # 14
    comptime OBS_DIM: Int = 27  # 13 qpos (15-2) + 14 qvel
    comptime ACTION_DIM: Int = 8

    # Initial torso height (from body_pos_z; free joint starts at z=0.55)
    comptime INITIAL_Z: Scalar[Self.DTYPE] = 0.55

    # Motor
    comptime TORQUE_LIMIT: Scalar[Self.DTYPE] = 150.0

    # GPU layout sizes
    comptime STATE_SIZE: Int = state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()
    comptime MODEL_SIZE: Int = model_size[
        Self.NUM_BODIES, Self.NUM_JOINTS, Self.NGEOM
    ]()

    # GPU layout helper methods
    @staticmethod
    @always_inline
    fn get_qpos_offset() -> Int:
        return qpos_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qvel_offset() -> Int:
        return qvel_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qacc_offset() -> Int:
        return qacc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_qfrc_offset() -> Int:
        return qfrc_offset[Self.NQ, Self.NV]()

    @staticmethod
    @always_inline
    fn get_xpos_offset() -> Int:
        return xpos_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_xquat_offset() -> Int:
        return xquat_offset[Self.NQ, Self.NV, Self.NUM_BODIES]()

    @staticmethod
    @always_inline
    fn get_metadata_offset() -> Int:
        return metadata_offset[
            Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
        ]()

    @staticmethod
    @always_inline
    fn get_model_body_offset(body_idx: Int) -> Int:
        return model_body_offset(body_idx)

    @staticmethod
    @always_inline
    fn get_model_joint_offset(joint_idx: Int) -> Int:
        return model_joint_offset[Self.NUM_BODIES](joint_idx)

    @staticmethod
    @always_inline
    fn get_model_metadata_offset() -> Int:
        return model_metadata_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()

    @staticmethod
    @always_inline
    fn get_model_curriculum_offset() -> Int:
        return model_curriculum_offset[Self.NUM_BODIES, Self.NUM_JOINTS]()


comptime AntParamsCPU = AntParams[DType.float64]
comptime AntParamsGPU = AntParams[DType.float32]


# =============================================================================
# Body/Joint Index Constants
# =============================================================================

comptime BODY_WORLDBODY: Int = 0
comptime BODY_TORSO: Int = 1
comptime BODY_FRONT_LEFT_LEG: Int = 2
comptime BODY_AUX_1: Int = 3
comptime BODY_ANKLE_1: Int = 4
comptime BODY_FRONT_RIGHT_LEG: Int = 5
comptime BODY_AUX_2: Int = 6
comptime BODY_ANKLE_2: Int = 7
comptime BODY_BACK_LEG: Int = 8
comptime BODY_AUX_3: Int = 9
comptime BODY_ANKLE_3: Int = 10
comptime BODY_RIGHT_BACK_LEG: Int = 11
comptime BODY_AUX_4: Int = 12
comptime BODY_ANKLE_4: Int = 13

comptime JOINT_ROOT: Int = 0
comptime JOINT_HIP_1: Int = 1
comptime JOINT_ANKLE_1: Int = 2
comptime JOINT_HIP_2: Int = 3
comptime JOINT_ANKLE_2: Int = 4
comptime JOINT_HIP_3: Int = 5
comptime JOINT_ANKLE_3: Int = 6
comptime JOINT_HIP_4: Int = 7
comptime JOINT_ANKLE_4: Int = 8

# Dimension constants
comptime NQ: Int = AntModel.NQ  # 15
comptime NV: Int = AntModel.NV  # 14
comptime NBODY: Int = AntModel.NBODY  # 14
comptime NJOINT: Int = AntModel.NJOINT  # 9
comptime NGEOM: Int = AntModel.NGEOM  # 14
comptime MAX_CONTACTS: Int = 40
comptime OBS_DIM: Int = 27
comptime ACTION_DIM: Int = 8
