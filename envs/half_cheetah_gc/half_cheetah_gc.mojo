"""HalfCheetahGC Environment - MuJoCo-style Half Cheetah using Generalized Coordinates engine.

This implementation uses the physics3d Generalized Coordinates (GC) engine:
- Model/Data for joint-space physics (MuJoCo-style)
- DefaultIntegrator for constraint-based contact solving
- Joint-space state: qpos (positions), qvel (velocities)
- Forward kinematics computes body positions (xpos, xquat)

The Half Cheetah is a 2D planar robot (movement in XZ plane, rotation around Y axis)
consisting of a torso with two leg chains (front and back) and a head, totaling:
- 8 bodies: torso, bthigh, bshin, bfoot, fthigh, fshin, ffoot, head
- 10 joints: 3 root DOFs (unactuated) + 6 leg joints (actuated) + 1 head (fixed)
- 17D observation: 8 qpos (excluding rootx and head) + 9 qvel (excluding head)
- 6D action: torques for the 6 actuated leg joints
"""

from math import sqrt, sin, cos
from collections import InlineArray
from random.philox import Random as PhiloxRandom

from core import (
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
    State,
    Action,
)
from render import RendererBase
from deep_rl import dtype as gpu_dtype

# GPU imports
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

# Import GC physics engine
from physics3d.constants import GEOM_CAPSULE
from physics3d.types import Model, Data
from physics3d.integrator import EulerIntegrator
from physics3d.solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from physics3d.gpu.constants import (
    TPB,
    state_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    metadata_offset,
    model_size,
    META_IDX_NUM_CONTACTS,
    META_IDX_STEP_COUNT,
    META_IDX_PREV_X,
    model_curriculum_offset,
    CURRICULUM_IDX_MIN_HEIGHT,
    CURRICULUM_IDX_MAX_PITCH,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_GEOM_TYPE,
    BODY_IDX_RADIUS,
    BODY_IDX_HALF_LENGTH,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TAU_LIMIT,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_GROUND_Z,
    MODEL_META_IDX_FRICTION,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
)


from .constants_gc import (
    # Physics
    DT,
    FRAME_SKIP,
    GRAVITY_Z,
    GROUND_Z,
    MAX_STEPS,
    INIT_HEIGHT,
    FRICTION,
    # Dimensions
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    OBS_DIM,
    ACTION_DIM,
    # Body indices
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    BODY_HEAD,
    # Joint indices
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    JOINT_BTHIGH,
    JOINT_BSHIN,
    JOINT_BFOOT,
    JOINT_FTHIGH,
    JOINT_FSHIN,
    JOINT_FFOOT,
    JOINT_HEAD,
    # Body geometry
    CAPSULE_RADIUS,
    TORSO_HALF_LENGTH,
    HEAD_HALF_LENGTH,
    HEAD_POS_X,
    HEAD_POS_Y,
    HEAD_POS_Z,
    HEAD_AXIS_ANGLE,
    BTHIGH_HALF_LENGTH,
    BSHIN_HALF_LENGTH,
    BFOOT_HALF_LENGTH,
    FTHIGH_HALF_LENGTH,
    FSHIN_HALF_LENGTH,
    FFOOT_HALF_LENGTH,
    # Body masses
    TORSO_MASS,
    HEAD_MASS,
    BTHIGH_MASS,
    BSHIN_MASS,
    BFOOT_MASS,
    FTHIGH_MASS,
    FSHIN_MASS,
    FFOOT_MASS,
    # Gear ratios
    BTHIGH_GEAR,
    BSHIN_GEAR,
    BFOOT_GEAR,
    FTHIGH_GEAR,
    FSHIN_GEAR,
    FFOOT_GEAR,
    # Joint damping
    BTHIGH_DAMPING,
    BSHIN_DAMPING,
    BFOOT_DAMPING,
    FTHIGH_DAMPING,
    FSHIN_DAMPING,
    FFOOT_DAMPING,
    # Joint stiffness
    BTHIGH_STIFFNESS,
    BSHIN_STIFFNESS,
    BFOOT_STIFFNESS,
    FTHIGH_STIFFNESS,
    FSHIN_STIFFNESS,
    FFOOT_STIFFNESS,
    # Reward
    FORWARD_REWARD_WEIGHT,
    CTRL_COST_WEIGHT,
    # Reset
    RESET_NOISE_SCALE,
    # Joint limits
    BTHIGH_LOWER,
    BTHIGH_UPPER,
    BSHIN_LOWER,
    BSHIN_UPPER,
    BFOOT_LOWER,
    BFOOT_UPPER,
    FTHIGH_LOWER,
    FTHIGH_UPPER,
    FSHIN_LOWER,
    FSHIN_UPPER,
    FFOOT_LOWER,
    FFOOT_UPPER,
    HEAD_LOWER,
    HEAD_UPPER,
    # GPU constants struct
    HalfCheetahGCConstants,
)
from .state import HalfCheetahGCState
from .action import HalfCheetahGCAction
from .renderer import HalfCheetahGCRenderer

# Math types for renderer
from math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


# =============================================================================
# HalfCheetahGC Environment
# =============================================================================


struct HalfCheetahGC[
    DTYPE: DType = DType.float64, TERMINATE_ON_UNHEALTHY: Bool = False
](
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
):
    """HalfCheetahGC environment using Generalized Coordinates physics.

    Physical Configuration (matching MuJoCo Half Cheetah):
        - Body 0 (Torso): Horizontal capsule along X-axis
        - Body 1 (BThigh): Back thigh, vertical capsule
        - Body 2 (BShin): Back shin, vertical capsule
        - Body 3 (BFoot): Back foot, horizontal capsule
        - Body 4 (FThigh): Front thigh, vertical capsule
        - Body 5 (FShin): Front shin, vertical capsule
        - Body 6 (FFoot): Front foot, horizontal capsule
        - Body 7 (Head): Tilted capsule at front of torso

    Joint Configuration (MuJoCo style):
        - Joint 0 (rootx): Slide joint, X-axis translation (body 0)
        - Joint 1 (rootz): Slide joint, Z-axis translation (body 0)
        - Joint 2 (rooty): Hinge joint, Y-axis rotation (body 0)
        - Joint 3 (bthigh): Hinge joint, Y-axis rotation (body 1), range [-0.52, 1.05]
        - Joint 4 (bshin): Hinge joint, Y-axis rotation (body 2), range [-0.785, 0.785]
        - Joint 5 (bfoot): Hinge joint, Y-axis rotation (body 3), range [-0.4, 0.785]
        - Joint 6 (fthigh): Hinge joint, Y-axis rotation (body 4), range [-1.0, 0.7]
        - Joint 7 (fshin): Hinge joint, Y-axis rotation (body 5), range [-1.2, 0.87]
        - Joint 8 (ffoot): Hinge joint, Y-axis rotation (body 6), range [-0.5, 0.5]
        - Joint 9 (head): Hinge joint, Y-axis rotation (body 7), range [0, 0] (fixed)

    State (qpos, qvel):
        - qpos[0]: rootx (x position)
        - qpos[1]: rootz (z position / height)
        - qpos[2]: rooty (pitch angle)
        - qpos[3-5]: back leg joint angles (bthigh, bshin, bfoot)
        - qpos[6-8]: front leg joint angles (fthigh, fshin, ffoot)
        - qpos[9]: head angle (fixed at 0)
        - qvel[0:10]: corresponding velocities

    Observation Space (17 dimensions):
        Excludes qpos[0] (rootx) and head for translation invariance.
        [0:8]: qpos[1:9] (z, rooty, 6 joint angles)
        [8:17]: qvel[0:9] (all velocities excluding head)

    Action Space (6 dimensions):
        [0] bthigh torque (scaled by gear ratio 120)
        [1] bshin torque (scaled by gear ratio 90)
        [2] bfoot torque (scaled by gear ratio 60)
        [3] fthigh torque (scaled by gear ratio 120)
        [4] fshin torque (scaled by gear ratio 60)
        [5] ffoot torque (scaled by gear ratio 30)
    """

    # Trait type aliases
    comptime dtype = Self.DTYPE
    comptime StateType = HalfCheetahGCState
    comptime ActionType = HalfCheetahGCAction

    # Layout constants
    comptime OBS_DIM: Int = OBS_DIM
    comptime ACTION_DIM: Int = ACTION_DIM

    # GC physics layout constants
    comptime NQ: Int = NQ
    comptime NV: Int = NV
    comptime NUM_BODIES: Int = NBODY
    comptime NUM_JOINTS: Int = NJOINT
    comptime MAX_CONTACTS: Int = MAX_CONTACTS

    # GPU state size
    comptime STATE_SIZE: Int = state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()

    # Physics model and data
    var model: Model[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NUM_BODIES,
        Self.NUM_JOINTS,
        Self.MAX_CONTACTS,
    ]
    var data: Data[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NUM_BODIES,
        Self.NUM_JOINTS,
        Self.MAX_CONTACTS,
    ]

    # Environment parameters
    var max_steps: Int
    var current_step: Int
    var frame_skip: Int

    # Previous x position for velocity calculation
    var prev_x_position: Scalar[Self.DTYPE]

    # Cached observation state
    var cached_state: HalfCheetahGCState

    # Renderer (optional)
    var _renderer: UnsafePointer[HalfCheetahGCRenderer, MutAnyOrigin]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        max_steps: Int = 1000,
        frame_skip: Int = 5,
        timestep: Scalar[Self.DTYPE] = 0.002,
        friction: Scalar[Self.DTYPE] = 0.9,
    ):
        """Initialize the HalfCheetahGC environment.

        Args:
            max_steps: Maximum episode length (default 1000).
            frame_skip: Number of physics steps per environment step (default 5).
            timestep: Physics timestep in seconds (default 0.002).
            friction: Ground friction coefficient (default 0.9).
        """
        self.max_steps = max_steps
        self.current_step = 0
        self.frame_skip = frame_skip
        self.prev_x_position = Scalar[Self.DTYPE](0.0)
        self._renderer = UnsafePointer[HalfCheetahGCRenderer, MutAnyOrigin]()
        self._renderer_initialized = False

        # Initialize GC model
        self.model = Model[
            Self.DTYPE,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
        ](
            gravity_z=Scalar[Self.DTYPE](GRAVITY_Z),
            timestep=timestep,
            ground_z=Scalar[Self.DTYPE](GROUND_Z),
            friction=friction,
        )

        # Set solref/solimp from MuJoCo half_cheetah.xml
        comptime CC = HalfCheetahGCConstants[Self.DTYPE]
        self.model.solref_contact[0] = CC.SOLREF_CONTACT_0
        self.model.solref_contact[1] = CC.SOLREF_CONTACT_1
        self.model.solimp_contact[0] = CC.SOLIMP_CONTACT_0
        self.model.solimp_contact[1] = CC.SOLIMP_CONTACT_1
        self.model.solimp_contact[2] = CC.SOLIMP_CONTACT_2
        self.model.solref_limit[0] = CC.SOLREF_LIMIT_0
        self.model.solref_limit[1] = CC.SOLREF_LIMIT_1
        self.model.solimp_limit[0] = CC.SOLIMP_LIMIT_0
        self.model.solimp_limit[1] = CC.SOLIMP_LIMIT_1
        self.model.solimp_limit[2] = CC.SOLIMP_LIMIT_2

        # Initialize data (must be done before any method calls)
        self.data = Data[
            Self.DTYPE,
            Self.NQ,
            Self.NV,
            Self.NUM_BODIES,
            Self.NUM_JOINTS,
            Self.MAX_CONTACTS,
        ]()

        # Initialize cached state
        self.cached_state = HalfCheetahGCState()

        # Configure bodies
        self._setup_bodies()

        # Configure joints
        self._setup_joints()

        # Reset to initial state
        self._reset_state()
        self._update_cached_state()

    fn _setup_bodies(mut self):
        """Configure all body properties (mass, inertia, geometry).

        The Half Cheetah is a 2D planar robot in the XZ plane:
        - Torso is horizontal (extends along X)
        - Two leg chains (back and front) hang down from the torso
        - All rotations are around Y axis (into the screen)

        Key insight: The forward kinematics uses (0, 0, -parent_half) to find
        the joint pivot. With a 90° Y rotation on the torso:
        - local Z -> world X, so (0,0,-half) -> (-half, 0, 0) in world
        - This means the joint pivot is at the BACK of the torso (world -X)

        For the legs to hang DOWN (world -Z), we need:
        1. Body pos offset that gives (0, 0, -child_half) after rotation
        2. Body quat that counter-rotates the parent so legs are vertical

        Math for body attached to torso (90° Y rot transforms (x,y,z) -> (z,y,-x)):
        - We need: rotate(torso_quat, (px, py, pz + torso_half)) = (0, 0, -thigh_half)
        - Expanding: (pz + torso_half, py, -px) = (0, 0, -thigh_half)
        - Solving: pz = -torso_half, py = 0, px = thigh_half
        """

        # Helper to compute capsule inertia
        fn compute_capsule_inertia(
            mass: Scalar[Self.DTYPE],
            radius: Scalar[Self.DTYPE],
            half_length: Scalar[Self.DTYPE],
        ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
            var r2 = radius * radius
            var L = (
                Scalar[Self.DTYPE](2.0) * half_length
                + Scalar[Self.DTYPE](2.0) * radius
            )
            var L2 = L * L
            var I_trans = (
                mass
                * (Scalar[Self.DTYPE](3.0) * r2 + L2)
                / Scalar[Self.DTYPE](12.0)
            )
            var I_axial = Scalar[Self.DTYPE](0.5) * mass * r2
            return (I_trans, I_trans, I_axial)

        var radius = Scalar[Self.DTYPE](CAPSULE_RADIUS)
        var torso_half = Scalar[Self.DTYPE](TORSO_HALF_LENGTH)
        var bthigh_half = Scalar[Self.DTYPE](BTHIGH_HALF_LENGTH)
        var bshin_half = Scalar[Self.DTYPE](BSHIN_HALF_LENGTH)
        var bfoot_half = Scalar[Self.DTYPE](BFOOT_HALF_LENGTH)
        var fthigh_half = Scalar[Self.DTYPE](FTHIGH_HALF_LENGTH)
        var fshin_half = Scalar[Self.DTYPE](FSHIN_HALF_LENGTH)
        var ffoot_half = Scalar[Self.DTYPE](FFOOT_HALF_LENGTH)

        # Quaternion for 90° Y rotation (makes capsule horizontal)
        var quat_90y_x = Scalar[Self.DTYPE](0.0)
        var quat_90y_y = Scalar[Self.DTYPE](0.70710678)
        var quat_90y_z = Scalar[Self.DTYPE](0.0)
        var quat_90y_w = Scalar[Self.DTYPE](0.70710678)

        # Quaternion for -90° Y rotation (inverse, to counter-rotate)
        var quat_neg90y_x = Scalar[Self.DTYPE](0.0)
        var quat_neg90y_y = Scalar[Self.DTYPE](-0.70710678)
        var quat_neg90y_z = Scalar[Self.DTYPE](0.0)
        var quat_neg90y_w = Scalar[Self.DTYPE](0.70710678)

        # =====================================================================
        # Body 0: Torso (horizontal, 90° Y rotation)
        # Position controlled by rootx/rootz/rooty joints
        # =====================================================================
        var torso_mass = Scalar[Self.DTYPE](TORSO_MASS)
        var torso_inertia = compute_capsule_inertia(
            torso_mass, radius, torso_half
        )
        self.model.set_body(
            BODY_TORSO, mass=torso_mass, inertia=torso_inertia, radius=radius
        )
        self.model.set_body_parent(BODY_TORSO, -1)
        self.model.body_geom_type[BODY_TORSO] = GEOM_CAPSULE
        self.model.body_half_length[BODY_TORSO] = torso_half
        # 90° rotation around Y makes capsule horizontal (along X)
        self.model.set_body_local_frame(
            BODY_TORSO,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            quat=(quat_90y_x, quat_90y_y, quat_90y_z, quat_90y_w),
        )

        # =====================================================================
        # Body 1: Back Thigh (bthigh) - vertical, at back of torso
        # Joint pivot is at back of torso (world -X direction from torso center)
        # Body should be vertical (counter-rotate the torso's rotation)
        # =====================================================================
        var bthigh_mass = Scalar[Self.DTYPE](BTHIGH_MASS)
        var bthigh_inertia = compute_capsule_inertia(
            bthigh_mass, radius, bthigh_half
        )
        self.model.set_body(
            BODY_BTHIGH, mass=bthigh_mass, inertia=bthigh_inertia, radius=radius
        )
        self.model.set_body_parent(BODY_BTHIGH, BODY_TORSO)
        self.model.body_geom_type[BODY_BTHIGH] = GEOM_CAPSULE
        self.model.body_half_length[BODY_BTHIGH] = bthigh_half
        # body_pos: Need rotate(torso_quat, (px, py, pz + torso_half)) = (0, 0, -bthigh_half)
        # With 90° Y: (pz + torso_half, py, -px) = (0, 0, -bthigh_half)
        # Solution: px = bthigh_half, py = 0, pz = -torso_half
        # body_quat: -90° Y to counter-rotate torso's rotation (legs should be vertical)
        self.model.set_body_local_frame(
            BODY_BTHIGH,
            pos=(bthigh_half, Scalar[Self.DTYPE](0.0), -torso_half),
            quat=(quat_neg90y_x, quat_neg90y_y, quat_neg90y_z, quat_neg90y_w),
        )

        # =====================================================================
        # Body 2: Back Shin (bshin) - vertical, below bthigh
        # Parent (bthigh) is vertical with identity world orientation
        # =====================================================================
        var bshin_mass = Scalar[Self.DTYPE](BSHIN_MASS)
        var bshin_inertia = compute_capsule_inertia(
            bshin_mass, radius, bshin_half
        )
        self.model.set_body(
            BODY_BSHIN, mass=bshin_mass, inertia=bshin_inertia, radius=radius
        )
        self.model.set_body_parent(BODY_BSHIN, BODY_BTHIGH)
        self.model.body_geom_type[BODY_BSHIN] = GEOM_CAPSULE
        self.model.body_half_length[BODY_BSHIN] = bshin_half
        # Parent is vertical (identity orientation), so standard offset works
        self.model.set_body_local_frame(
            BODY_BSHIN,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -(bthigh_half + bshin_half),
            ),
        )

        # =====================================================================
        # Body 3: Back Foot (bfoot) - horizontal
        # Parent (bshin) is vertical, foot should be horizontal
        # =====================================================================
        var bfoot_mass = Scalar[Self.DTYPE](BFOOT_MASS)
        var bfoot_inertia = compute_capsule_inertia(
            bfoot_mass, radius, bfoot_half
        )
        self.model.set_body(
            BODY_BFOOT, mass=bfoot_mass, inertia=bfoot_inertia, radius=radius
        )
        self.model.set_body_parent(BODY_BFOOT, BODY_BSHIN)
        self.model.body_geom_type[BODY_BFOOT] = GEOM_CAPSULE
        self.model.body_half_length[BODY_BFOOT] = bfoot_half
        # Offset below bshin, with 90° Y rotation for horizontal foot
        self.model.set_body_local_frame(
            BODY_BFOOT,
            pos=(Scalar[Self.DTYPE](0.0), Scalar[Self.DTYPE](0.0), -bshin_half),
            quat=(quat_90y_x, quat_90y_y, quat_90y_z, quat_90y_w),
        )

        # =====================================================================
        # Body 4: Front Thigh (fthigh) - vertical, at front of torso
        # Joint pivot is at front of torso (world +X direction from torso center)
        # FK moves to (0, 0, -torso_half) in torso frame = (-torso_half, 0, 0) world
        # But we want FRONT, so we need different body_pos
        # =====================================================================
        var fthigh_mass = Scalar[Self.DTYPE](FTHIGH_MASS)
        var fthigh_inertia = compute_capsule_inertia(
            fthigh_mass, radius, fthigh_half
        )
        self.model.set_body(
            BODY_FTHIGH, mass=fthigh_mass, inertia=fthigh_inertia, radius=radius
        )
        self.model.set_body_parent(BODY_FTHIGH, BODY_TORSO)
        self.model.body_geom_type[BODY_FTHIGH] = GEOM_CAPSULE
        self.model.body_half_length[BODY_FTHIGH] = fthigh_half
        # FK algorithm moves to pivot at (0,0,-torso_half) in torso frame = (-torso_half, 0, 0) world (BACK)
        # But we want fthigh at FRONT. Need body_pos such that:
        # pivot + rotate(torso_quat, (px, py, pz + torso_half)) = torso_center + (torso_half, 0, -fthigh_half)
        # pivot = torso_center + (-torso_half, 0, 0)
        # So: rotate(torso_quat, (px, py, pz + torso_half)) = (2*torso_half, 0, -fthigh_half)
        # (pz + torso_half, py, -px) = (2*torso_half, 0, -fthigh_half)
        # Solution: pz = torso_half, py = 0, px = fthigh_half
        self.model.set_body_local_frame(
            BODY_FTHIGH,
            pos=(fthigh_half, Scalar[Self.DTYPE](0.0), torso_half),
            quat=(quat_neg90y_x, quat_neg90y_y, quat_neg90y_z, quat_neg90y_w),
        )

        # =====================================================================
        # Body 5: Front Shin (fshin) - vertical, below fthigh
        # =====================================================================
        var fshin_mass = Scalar[Self.DTYPE](FSHIN_MASS)
        var fshin_inertia = compute_capsule_inertia(
            fshin_mass, radius, fshin_half
        )
        self.model.set_body(
            BODY_FSHIN, mass=fshin_mass, inertia=fshin_inertia, radius=radius
        )
        self.model.set_body_parent(BODY_FSHIN, BODY_FTHIGH)
        self.model.body_geom_type[BODY_FSHIN] = GEOM_CAPSULE
        self.model.body_half_length[BODY_FSHIN] = fshin_half
        # Parent is vertical, standard offset
        self.model.set_body_local_frame(
            BODY_FSHIN,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -(fthigh_half + fshin_half),
            ),
        )

        # =====================================================================
        # Body 6: Front Foot (ffoot) - horizontal
        # =====================================================================
        var ffoot_mass = Scalar[Self.DTYPE](FFOOT_MASS)
        var ffoot_inertia = compute_capsule_inertia(
            ffoot_mass, radius, ffoot_half
        )
        self.model.set_body(
            BODY_FFOOT, mass=ffoot_mass, inertia=ffoot_inertia, radius=radius
        )
        self.model.set_body_parent(BODY_FFOOT, BODY_FSHIN)
        self.model.body_geom_type[BODY_FFOOT] = GEOM_CAPSULE
        self.model.body_half_length[BODY_FFOOT] = ffoot_half
        # Offset below fshin, with 90° Y rotation for horizontal foot
        self.model.set_body_local_frame(
            BODY_FFOOT,
            pos=(Scalar[Self.DTYPE](0.0), Scalar[Self.DTYPE](0.0), -fshin_half),
            quat=(quat_90y_x, quat_90y_y, quat_90y_z, quat_90y_w),
        )

        # =====================================================================
        # Body 7: Head - tilted capsule at front of torso
        # MuJoCo XML: pos=".6 0 .1" axisangle="0 1 0 .87"
        # Position relative to torso center, with 0.87 rad Y rotation
        # =====================================================================
        var head_half = Scalar[Self.DTYPE](HEAD_HALF_LENGTH)
        var head_mass = Scalar[Self.DTYPE](HEAD_MASS)
        var head_inertia = compute_capsule_inertia(head_mass, radius, head_half)
        self.model.set_body(
            BODY_HEAD, mass=head_mass, inertia=head_inertia, radius=radius
        )
        self.model.set_body_parent(BODY_HEAD, BODY_TORSO)
        self.model.body_geom_type[BODY_HEAD] = GEOM_CAPSULE
        self.model.body_half_length[BODY_HEAD] = head_half

        # Head position: MuJoCo pos=".6 0 .1" relative to torso center
        # Torso has 90° Y rotation, so world coords map:
        #   world_x -> torso_local_z, world_z -> -torso_local_x
        # FK formula: child_center = pivot + rotate(parent_quat, body_pos + (0,0,parent_half))
        # pivot = torso_center + (-0.5, 0, 0) [world]
        # We want head at torso_center + (0.6, 0, 0.1) [world]
        # So: rotate(quat_90y, (px, py, pz + 0.5)) = (1.1, 0, 0.1)
        # With 90°Y: (pz+0.5, py, -px) = (1.1, 0, 0.1)
        # Solution: pz = 0.6, py = 0, px = -0.1
        var head_pos_x = Scalar[Self.DTYPE](-HEAD_POS_Z)  # -0.1
        var head_pos_y = Scalar[Self.DTYPE](HEAD_POS_Y)  # 0.0
        var head_pos_z = Scalar[Self.DTYPE](HEAD_POS_X)  # 0.6

        # Head rotation: MuJoCo has 0.87 rad Y in world frame
        # But torso already has 90° Y rotation, so head inherits that.
        # To get final world rotation of 0.87 rad Y, we need:
        # head_body_quat = inverse(torso_90Y) * target_0.87Y
        # = (0.87 - π/2) rad Y ≈ -0.7 rad Y
        # quat for -0.7 rad Y: (0, sin(-0.35), 0, cos(-0.35)) = (0, -0.343, 0, 0.939)
        var head_angle = Scalar[Self.DTYPE](
            HEAD_AXIS_ANGLE - 1.5707963268
        )  # 0.87 - π/2 ≈ -0.7
        var head_sin = sin(head_angle / Scalar[Self.DTYPE](2.0))
        var head_cos = cos(head_angle / Scalar[Self.DTYPE](2.0))
        self.model.set_body_local_frame(
            BODY_HEAD,
            pos=(head_pos_x, head_pos_y, head_pos_z),
            quat=(
                Scalar[Self.DTYPE](0.0),
                head_sin,
                Scalar[Self.DTYPE](0.0),
                head_cos,
            ),
        )

    fn _setup_joints(mut self):
        """Configure all joints (root DOFs and actuated joints)."""
        # =====================================================================
        # Joint 0: rootx - Slide joint, X-axis translation (body 0)
        # =====================================================================
        _ = self.model.add_slide_joint(
            body_id=BODY_TORSO,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            force_limit=Scalar[Self.DTYPE](0.0),  # Not actuated
        )

        # =====================================================================
        # Joint 1: rootz - Slide joint, Z-axis translation (body 0)
        # =====================================================================
        _ = self.model.add_slide_joint(
            body_id=BODY_TORSO,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
            ),
            force_limit=Scalar[Self.DTYPE](0.0),  # Not actuated
        )

        # =====================================================================
        # Joint 2: rooty - Hinge joint, Y-axis rotation (body 0)
        # =====================================================================
        _ = self.model.add_hinge_joint(
            body_id=BODY_TORSO,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](0.0),  # Not actuated
        )

        # =====================================================================
        # Joint 3: bthigh - Back thigh hinge (body 1)
        # Joint attaches at back of torso: (0, 0, -torso_half) in torso frame
        # =====================================================================
        _ = self.model.add_hinge_joint(
            body_id=BODY_BTHIGH,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](-TORSO_HALF_LENGTH),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](BTHIGH_GEAR),
            range_min=Scalar[Self.DTYPE](BTHIGH_LOWER),
            range_max=Scalar[Self.DTYPE](BTHIGH_UPPER),
            armature=Scalar[Self.DTYPE](0.1),
            damping=Scalar[Self.DTYPE](BTHIGH_DAMPING),
            stiffness=Scalar[Self.DTYPE](BTHIGH_STIFFNESS),
        )

        # =====================================================================
        # Joint 4: bshin - Back shin hinge (body 2)
        # Joint attaches at bottom of bthigh: (0, 0, -bthigh_half) in bthigh frame
        # =====================================================================
        _ = self.model.add_hinge_joint(
            body_id=BODY_BSHIN,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](-BTHIGH_HALF_LENGTH),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](BSHIN_GEAR),
            range_min=Scalar[Self.DTYPE](BSHIN_LOWER),
            range_max=Scalar[Self.DTYPE](BSHIN_UPPER),
            armature=Scalar[Self.DTYPE](0.1),
            damping=Scalar[Self.DTYPE](BSHIN_DAMPING),
            stiffness=Scalar[Self.DTYPE](BSHIN_STIFFNESS),
        )

        # =====================================================================
        # Joint 5: bfoot - Back foot hinge (body 3)
        # Joint attaches at bottom of bshin: (0, 0, -bshin_half) in bshin frame
        # =====================================================================
        _ = self.model.add_hinge_joint(
            body_id=BODY_BFOOT,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](-BSHIN_HALF_LENGTH),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](BFOOT_GEAR),
            range_min=Scalar[Self.DTYPE](BFOOT_LOWER),
            range_max=Scalar[Self.DTYPE](BFOOT_UPPER),
            armature=Scalar[Self.DTYPE](0.1),
            damping=Scalar[Self.DTYPE](BFOOT_DAMPING),
            stiffness=Scalar[Self.DTYPE](BFOOT_STIFFNESS),
        )

        # =====================================================================
        # Joint 6: fthigh - Front thigh hinge (body 4)
        # Joint attaches at FRONT of torso: (0, 0, +torso_half) in torso frame
        # =====================================================================
        _ = self.model.add_hinge_joint(
            body_id=BODY_FTHIGH,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](TORSO_HALF_LENGTH),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](FTHIGH_GEAR),
            range_min=Scalar[Self.DTYPE](FTHIGH_LOWER),
            range_max=Scalar[Self.DTYPE](FTHIGH_UPPER),
            armature=Scalar[Self.DTYPE](0.1),
            damping=Scalar[Self.DTYPE](FTHIGH_DAMPING),
            stiffness=Scalar[Self.DTYPE](FTHIGH_STIFFNESS),
        )

        # =====================================================================
        # Joint 7: fshin - Front shin hinge (body 5)
        # Joint attaches at bottom of fthigh: (0, 0, -fthigh_half) in fthigh frame
        # =====================================================================
        _ = self.model.add_hinge_joint(
            body_id=BODY_FSHIN,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](-FTHIGH_HALF_LENGTH),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](FSHIN_GEAR),
            range_min=Scalar[Self.DTYPE](FSHIN_LOWER),
            range_max=Scalar[Self.DTYPE](FSHIN_UPPER),
            armature=Scalar[Self.DTYPE](0.1),
            damping=Scalar[Self.DTYPE](FSHIN_DAMPING),
            stiffness=Scalar[Self.DTYPE](FSHIN_STIFFNESS),
        )

        # =====================================================================
        # Joint 8: ffoot - Front foot hinge (body 6)
        # Joint attaches at bottom of fshin: (0, 0, -fshin_half) in fshin frame
        # =====================================================================
        _ = self.model.add_hinge_joint(
            body_id=BODY_FFOOT,
            pos=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](-FSHIN_HALF_LENGTH),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](FFOOT_GEAR),
            range_min=Scalar[Self.DTYPE](FFOOT_LOWER),
            range_max=Scalar[Self.DTYPE](FFOOT_UPPER),
            armature=Scalar[Self.DTYPE](0.1),
            damping=Scalar[Self.DTYPE](FFOOT_DAMPING),
            stiffness=Scalar[Self.DTYPE](FFOOT_STIFFNESS),
        )

        # =====================================================================
        # Joint 9: head - Head hinge (body 7) - FIXED joint (zero range)
        # Attaches at head position on torso, zero range makes it rigid
        # Joint pos is in parent (torso) frame. Torso has 90°Y rotation.
        # World pos (0.6, 0, 0.1) -> torso local (-0.1, 0, 0.6)
        # =====================================================================
        _ = self.model.add_hinge_joint(
            body_id=BODY_HEAD,
            pos=(
                Scalar[Self.DTYPE](-HEAD_POS_Z),  # -0.1 (world z -> -local x)
                Scalar[Self.DTYPE](HEAD_POS_Y),  # 0.0
                Scalar[Self.DTYPE](HEAD_POS_X),  # 0.6 (world x -> local z)
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),
                Scalar[Self.DTYPE](0.0),
            ),
            tau_limit=Scalar[Self.DTYPE](0.0),  # Not actuated
            range_min=Scalar[Self.DTYPE](HEAD_LOWER),
            range_max=Scalar[Self.DTYPE](HEAD_UPPER),
            armature=Scalar[Self.DTYPE](0.1),
            damping=Scalar[Self.DTYPE](0.01),  # MuJoCo default
            stiffness=Scalar[Self.DTYPE](8.0),  # MuJoCo default joint stiffness
        )

    # =========================================================================
    # Physics State Management
    # =========================================================================

    fn _reset_state(mut self):
        """Reset to initial standing position."""
        # Reset qpos
        self.data.qpos[JOINT_ROOTX] = Scalar[Self.DTYPE](0.0)  # rootx
        self.data.qpos[JOINT_ROOTZ] = Scalar[Self.DTYPE](INIT_HEIGHT)  # rootz
        self.data.qpos[JOINT_ROOTY] = Scalar[Self.DTYPE](0.0)  # rooty
        self.data.qpos[JOINT_BTHIGH] = Scalar[Self.DTYPE](0.0)  # bthigh
        self.data.qpos[JOINT_BSHIN] = Scalar[Self.DTYPE](0.0)  # bshin
        self.data.qpos[JOINT_BFOOT] = Scalar[Self.DTYPE](0.0)  # bfoot
        self.data.qpos[JOINT_FTHIGH] = Scalar[Self.DTYPE](0.0)  # fthigh
        self.data.qpos[JOINT_FSHIN] = Scalar[Self.DTYPE](0.0)  # fshin
        self.data.qpos[JOINT_FFOOT] = Scalar[Self.DTYPE](0.0)  # ffoot
        self.data.qpos[JOINT_HEAD] = Scalar[Self.DTYPE](0.0)  # head (fixed)

        # Reset qvel
        for i in range(Self.NV):
            self.data.qvel[i] = Scalar[Self.DTYPE](0.0)

        # Reset qacc and qfrc
        for i in range(Self.NV):
            self.data.qacc[i] = Scalar[Self.DTYPE](0.0)
            self.data.qfrc[i] = Scalar[Self.DTYPE](0.0)

        # Run forward kinematics to compute xpos/xquat
        forward_kinematics(self.model, self.data)

        # Reset step counter and previous position
        self.current_step = 0
        self.prev_x_position = self.data.qpos[JOINT_ROOTX]

    fn _update_cached_state(mut self):
        """Update cached state from physics data."""
        # Position observations (exclude qpos[0] = rootx)
        self.cached_state.z_position = Float64(self.data.qpos[JOINT_ROOTZ])
        self.cached_state.y_angle = Float64(self.data.qpos[JOINT_ROOTY])
        self.cached_state.bthigh_angle = Float64(self.data.qpos[JOINT_BTHIGH])
        self.cached_state.bshin_angle = Float64(self.data.qpos[JOINT_BSHIN])
        self.cached_state.bfoot_angle = Float64(self.data.qpos[JOINT_BFOOT])
        self.cached_state.fthigh_angle = Float64(self.data.qpos[JOINT_FTHIGH])
        self.cached_state.fshin_angle = Float64(self.data.qpos[JOINT_FSHIN])
        self.cached_state.ffoot_angle = Float64(self.data.qpos[JOINT_FFOOT])

        # Velocity observations (include all)
        self.cached_state.x_velocity = Float64(self.data.qvel[JOINT_ROOTX])
        self.cached_state.z_velocity = Float64(self.data.qvel[JOINT_ROOTZ])
        self.cached_state.y_angular_velocity = Float64(
            self.data.qvel[JOINT_ROOTY]
        )
        self.cached_state.bthigh_velocity = Float64(
            self.data.qvel[JOINT_BTHIGH]
        )
        self.cached_state.bshin_velocity = Float64(self.data.qvel[JOINT_BSHIN])
        self.cached_state.bfoot_velocity = Float64(self.data.qvel[JOINT_BFOOT])
        self.cached_state.fthigh_velocity = Float64(
            self.data.qvel[JOINT_FTHIGH]
        )
        self.cached_state.fshin_velocity = Float64(self.data.qvel[JOINT_FSHIN])
        self.cached_state.ffoot_velocity = Float64(self.data.qvel[JOINT_FFOOT])

    @always_inline
    fn _clamp_action(self, action: Float64) -> Float64:
        """Clamp action to [-1, 1]."""
        if action > 1.0:
            return 1.0
        elif action < -1.0:
            return -1.0
        return action

    fn _enforce_joint_limits(mut self):
        """Enforce joint position limits to prevent unrealistic poses.

        Clamps actuated joint positions to their defined limits.
        Also zeros velocity when hitting a limit (simple contact model).
        """

        # Helper to clamp and handle limit contact
        @always_inline
        fn clamp_joint(
            mut qpos: Scalar[Self.DTYPE],
            mut qvel: Scalar[Self.DTYPE],
            lower: Float64,
            upper: Float64,
        ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
            if qpos < Scalar[Self.DTYPE](lower):
                qpos = Scalar[Self.DTYPE](lower)
                if qvel < Scalar[Self.DTYPE](0):
                    qvel = Scalar[Self.DTYPE](0)  # Stop at limit
            elif qpos > Scalar[Self.DTYPE](upper):
                qpos = Scalar[Self.DTYPE](upper)
                if qvel > Scalar[Self.DTYPE](0):
                    qvel = Scalar[Self.DTYPE](0)  # Stop at limit
            return (qpos, qvel)

        # Clamp back leg joints
        var bthigh = clamp_joint(
            self.data.qpos[JOINT_BTHIGH],
            self.data.qvel[JOINT_BTHIGH],
            BTHIGH_LOWER,
            BTHIGH_UPPER,
        )
        self.data.qpos[JOINT_BTHIGH] = bthigh[0]
        self.data.qvel[JOINT_BTHIGH] = bthigh[1]

        var bshin = clamp_joint(
            self.data.qpos[JOINT_BSHIN],
            self.data.qvel[JOINT_BSHIN],
            BSHIN_LOWER,
            BSHIN_UPPER,
        )
        self.data.qpos[JOINT_BSHIN] = bshin[0]
        self.data.qvel[JOINT_BSHIN] = bshin[1]

        var bfoot = clamp_joint(
            self.data.qpos[JOINT_BFOOT],
            self.data.qvel[JOINT_BFOOT],
            BFOOT_LOWER,
            BFOOT_UPPER,
        )
        self.data.qpos[JOINT_BFOOT] = bfoot[0]
        self.data.qvel[JOINT_BFOOT] = bfoot[1]

        # Clamp front leg joints
        var fthigh = clamp_joint(
            self.data.qpos[JOINT_FTHIGH],
            self.data.qvel[JOINT_FTHIGH],
            FTHIGH_LOWER,
            FTHIGH_UPPER,
        )
        self.data.qpos[JOINT_FTHIGH] = fthigh[0]
        self.data.qvel[JOINT_FTHIGH] = fthigh[1]

        var fshin = clamp_joint(
            self.data.qpos[JOINT_FSHIN],
            self.data.qvel[JOINT_FSHIN],
            FSHIN_LOWER,
            FSHIN_UPPER,
        )
        self.data.qpos[JOINT_FSHIN] = fshin[0]
        self.data.qvel[JOINT_FSHIN] = fshin[1]

        var ffoot = clamp_joint(
            self.data.qpos[JOINT_FFOOT],
            self.data.qvel[JOINT_FFOOT],
            FFOOT_LOWER,
            FFOOT_UPPER,
        )
        self.data.qpos[JOINT_FFOOT] = ffoot[0]
        self.data.qvel[JOINT_FFOOT] = ffoot[1]

    fn _compute_reward(
        self,
        x_velocity: Float64,
        action: HalfCheetahGCAction,
        y_angle: Float64,
    ) -> Float64:
        """Compute reward for current state.

        Reward = forward_reward - ctrl_cost - angle_penalty
        - forward_reward = forward_reward_weight * x_velocity
        - ctrl_cost = ctrl_cost_weight * sum(action^2)
        - angle_penalty = angle_penalty_weight * abs(y_angle)
        """
        # Forward velocity reward
        var forward_reward = FORWARD_REWARD_WEIGHT * x_velocity

        # Control cost (penalize large actions)
        var ctrl_cost = CTRL_COST_WEIGHT * action.squared_sum()

        # Angle penalty (discourages flipping)
        comptime C = HalfCheetahGCConstants[DType.float64]
        var abs_angle = y_angle if y_angle >= 0.0 else -y_angle
        var angle_penalty = Float64(C.ANGLE_PENALTY_WEIGHT) * abs_angle

        return forward_reward - ctrl_cost - angle_penalty

    # =========================================================================
    # BoxContinuousActionEnv Interface
    # =========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a list."""
        var obs = List[Scalar[Self.dtype]](capacity=Self.OBS_DIM)
        # Position observations (8D)
        obs.append(Scalar[Self.dtype](self.cached_state.z_position))
        obs.append(Scalar[Self.dtype](self.cached_state.y_angle))
        obs.append(Scalar[Self.dtype](self.cached_state.bthigh_angle))
        obs.append(Scalar[Self.dtype](self.cached_state.bshin_angle))
        obs.append(Scalar[Self.dtype](self.cached_state.bfoot_angle))
        obs.append(Scalar[Self.dtype](self.cached_state.fthigh_angle))
        obs.append(Scalar[Self.dtype](self.cached_state.fshin_angle))
        obs.append(Scalar[Self.dtype](self.cached_state.ffoot_angle))
        # Velocity observations (9D)
        obs.append(Scalar[Self.dtype](self.cached_state.x_velocity))
        obs.append(Scalar[Self.dtype](self.cached_state.z_velocity))
        obs.append(Scalar[Self.dtype](self.cached_state.y_angular_velocity))
        obs.append(Scalar[Self.dtype](self.cached_state.bthigh_velocity))
        obs.append(Scalar[Self.dtype](self.cached_state.bshin_velocity))
        obs.append(Scalar[Self.dtype](self.cached_state.bfoot_velocity))
        obs.append(Scalar[Self.dtype](self.cached_state.fthigh_velocity))
        obs.append(Scalar[Self.dtype](self.cached_state.fshin_velocity))
        obs.append(Scalar[Self.dtype](self.cached_state.ffoot_velocity))
        return obs^

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial continuous observation."""
        self._reset_state()
        self._update_cached_state()
        return self.get_obs_list()

    fn obs_dim(self) -> Int:
        """Return the dimension of the observation vector."""
        return Self.OBS_DIM

    fn action_dim(self) -> Int:
        """Return the dimension of the action vector."""
        return Self.ACTION_DIM

    fn action_low(self) -> Scalar[Self.dtype]:
        """Return the lower bound for action values."""
        return Scalar[Self.dtype](-1.0)

    fn action_high(self) -> Scalar[Self.dtype]:
        """Return the upper bound for action values."""
        return Scalar[Self.dtype](1.0)

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take 1D continuous action (broadcasts to all joints)."""
        var actions = List[Scalar[Self.dtype]]()
        for _ in range(Self.ACTION_DIM):
            actions.append(action)
        return self.step_continuous_vec(actions)

    fn step_continuous_vec[
        DTYPE2: DType
    ](mut self, action: List[Scalar[DTYPE2]]) -> Tuple[
        List[Scalar[DTYPE2]], Scalar[DTYPE2], Bool
    ]:
        """Take multi-dimensional continuous action and return (obs, reward, done).

        Args:
            action: List of 6 action values (joint torques normalized to [-1, 1]).

        Returns:
            Tuple of (observation_list, reward, done).
        """
        # Convert to HalfCheetahGCAction
        var act = HalfCheetahGCAction(
            bthigh=Float64(action[0] if len(action) > 0 else 0),
            bshin=Float64(action[1] if len(action) > 1 else 0),
            bfoot=Float64(action[2] if len(action) > 2 else 0),
            fthigh=Float64(action[3] if len(action) > 3 else 0),
            fshin=Float64(action[4] if len(action) > 4 else 0),
            ffoot=Float64(action[5] if len(action) > 5 else 0),
        )

        # Take step
        var result = self.step(act)

        # Build observation list
        var obs = List[Scalar[DTYPE2]](capacity=Self.OBS_DIM)
        obs.append(Scalar[DTYPE2](self.cached_state.z_position))
        obs.append(Scalar[DTYPE2](self.cached_state.y_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.bthigh_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.bshin_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.bfoot_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.fthigh_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.fshin_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.ffoot_angle))
        obs.append(Scalar[DTYPE2](self.cached_state.x_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.z_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.y_angular_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.bthigh_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.bshin_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.bfoot_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.fthigh_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.fshin_velocity))
        obs.append(Scalar[DTYPE2](self.cached_state.ffoot_velocity))

        return (obs^, Scalar[DTYPE2](result[1]), result[2])

    # =========================================================================
    # Env Interface
    # =========================================================================

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done).

        When TERMINATE_ON_UNHEALTHY is True, terminates if |y_angle| > MAX_PITCH.
        """
        # Store previous x position for velocity calculation
        self.prev_x_position = self.data.qpos[JOINT_ROOTX]

        # Clamp actions and scale by gear ratios
        var bthigh_torque = Scalar[Self.DTYPE](
            self._clamp_action(action.bthigh) * BTHIGH_GEAR
        )
        var bshin_torque = Scalar[Self.DTYPE](
            self._clamp_action(action.bshin) * BSHIN_GEAR
        )
        var bfoot_torque = Scalar[Self.DTYPE](
            self._clamp_action(action.bfoot) * BFOOT_GEAR
        )
        var fthigh_torque = Scalar[Self.DTYPE](
            self._clamp_action(action.fthigh) * FTHIGH_GEAR
        )
        var fshin_torque = Scalar[Self.DTYPE](
            self._clamp_action(action.fshin) * FSHIN_GEAR
        )
        var ffoot_torque = Scalar[Self.DTYPE](
            self._clamp_action(action.ffoot) * FFOOT_GEAR
        )

        # Apply torques to actuated joints (joints 3-8)
        self.data.qfrc[JOINT_BTHIGH] = bthigh_torque
        self.data.qfrc[JOINT_BSHIN] = bshin_torque
        self.data.qfrc[JOINT_BFOOT] = bfoot_torque
        self.data.qfrc[JOINT_FTHIGH] = fthigh_torque
        self.data.qfrc[JOINT_FSHIN] = fshin_torque
        self.data.qfrc[JOINT_FFOOT] = ffoot_torque

        # Physics step (with frame skip)
        for _ in range(self.frame_skip):
            EulerIntegrator[SOLVER=NewtonSolver].step(self.model, self.data)
            # Enforce joint limits after each physics step
            self._enforce_joint_limits()

        self.current_step += 1

        # Update cached state
        self._update_cached_state()

        # Compute velocity from position change (more accurate than qvel)
        var x_position_after = Float64(self.data.qpos[JOINT_ROOTX])
        var dt = Float64(DT * self.frame_skip)
        var x_velocity = (x_position_after - Float64(self.prev_x_position)) / dt

        # Compute reward (using clamped action for ctrl_cost)
        var clamped_action = action.clamp()
        var y_angle = Float64(self.data.qpos[JOINT_ROOTY])
        var reward = self._compute_reward(x_velocity, clamped_action, y_angle)

        # Health check and termination
        var terminated = False

        @parameter
        if Self.TERMINATE_ON_UNHEALTHY:
            comptime C = HalfCheetahGCConstants[DType.float64]
            var abs_angle = y_angle if y_angle >= 0.0 else -y_angle
            terminated = abs_angle > Float64(C.MAX_PITCH)
        var truncated = self.current_step >= self.max_steps
        var done = terminated or truncated

        return (self.cached_state, Scalar[Self.dtype](reward), done)

    fn get_state(self) -> Self.StateType:
        """Get current state."""
        return self.cached_state

    fn reset(mut self) -> Self.StateType:
        """Reset and return initial state."""
        self._reset_state()
        self._update_cached_state()
        return self.cached_state

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment using 2D renderer (not supported for 3D)."""
        pass

    fn close(mut self):
        """Close the environment and release resources."""
        if self._renderer_initialized:
            try:
                self._renderer[].close()
            except:
                pass
            self._renderer.free()
            self._renderer_initialized = False

    # =========================================================================
    # Position/State Accessors
    # =========================================================================

    fn get_qpos(self) -> InlineArray[Scalar[Self.DTYPE], 10]:
        """Get full qpos array (10 DOFs including head)."""
        var qpos = InlineArray[Scalar[Self.DTYPE], 10](uninitialized=True)
        for i in range(10):
            qpos[i] = self.data.qpos[i]
        return qpos^

    fn get_qvel(self) -> InlineArray[Scalar[Self.DTYPE], 10]:
        """Get full qvel array (10 DOFs including head)."""
        var qvel = InlineArray[Scalar[Self.DTYPE], 10](uninitialized=True)
        for i in range(10):
            qvel[i] = self.data.qvel[i]
        return qvel^

    fn get_x_position(self) -> Scalar[Self.DTYPE]:
        """Get current x position (rootx qpos)."""
        return self.data.qpos[JOINT_ROOTX]

    fn get_x_velocity(self) -> Scalar[Self.DTYPE]:
        """Get current x velocity (rootx qvel)."""
        return self.data.qvel[JOINT_ROOTX]

    fn get_torso_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get torso world position from xpos."""
        return (
            self.data.xpos[BODY_TORSO * 3 + 0],
            self.data.xpos[BODY_TORSO * 3 + 1],
            self.data.xpos[BODY_TORSO * 3 + 2],
        )

    fn get_body_position(
        self, body_id: Int
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get body world position from xpos."""
        return (
            self.data.xpos[body_id * 3 + 0],
            self.data.xpos[body_id * 3 + 1],
            self.data.xpos[body_id * 3 + 2],
        )

    fn get_body_quaternion(
        self, body_id: Int
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        """Get body world orientation quaternion (x, y, z, w) from xquat."""
        return (
            self.data.xquat[body_id * 4 + 0],
            self.data.xquat[body_id * 4 + 1],
            self.data.xquat[body_id * 4 + 2],
            self.data.xquat[body_id * 4 + 3],
        )

    fn get_current_step(self) -> Int:
        """Get current step count."""
        return self.current_step

    fn get_max_steps(self) -> Int:
        """Get maximum steps per episode."""
        return self.max_steps

    fn is_done(self) -> Bool:
        """Check if episode is finished."""
        var truncated = self.current_step >= self.max_steps

        @parameter
        if Self.TERMINATE_ON_UNHEALTHY:
            comptime C = HalfCheetahGCConstants[DType.float64]
            var y_angle = Float64(self.data.qpos[JOINT_ROOTY])
            var abs_angle = y_angle if y_angle >= 0.0 else -y_angle
            return truncated or abs_angle > Float64(C.MAX_PITCH)
        else:
            return truncated

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    fn init_renderer(mut self) raises -> Bool:
        """Initialize the internal 3D renderer."""
        if self._renderer_initialized:
            return True

        from memory import alloc

        self._renderer = alloc[HalfCheetahGCRenderer](1)

        var renderer = HalfCheetahGCRenderer(
            width=1280,
            height=720,
            follow_cheetah=True,
            show_velocity=True,
        )
        renderer.init()

        self._renderer.init_pointee_move(renderer^)
        self._renderer_initialized = True
        return True

    fn render_frame(mut self) raises -> None:
        """Render the current state using the internal 3D renderer."""
        if not self._renderer_initialized:
            return

        if not self._renderer[].is_open():
            return

        # Extract body positions and quaternions
        var positions = List[Vec3](capacity=Self.NUM_BODIES)
        var quaternions = List[Quat](capacity=Self.NUM_BODIES)

        for i in range(Self.NUM_BODIES):
            var pos = self.get_body_position(i)
            positions.append(
                Vec3(Float64(pos[0]), Float64(pos[1]), Float64(pos[2]))
            )

            var quat = self.get_body_quaternion(i)
            # xyzw -> wxyz for math3d Quat
            quaternions.append(
                Quat(
                    Float64(quat[3]),
                    Float64(quat[0]),
                    Float64(quat[1]),
                    Float64(quat[2]),
                )
            )

        # Get velocity
        var vel_x = Float64(self.get_x_velocity())

        # Render
        self._renderer[].render(positions, quaternions, vel_x)

    fn close_renderer(mut self) raises -> None:
        """Close the internal renderer."""
        if not self._renderer_initialized:
            return

        self._renderer[].close()
        self._renderer.free()
        self._renderer_initialized = False

    fn is_renderer_open(self) -> Bool:
        """Check if the internal renderer is open."""
        if not self._renderer_initialized:
            return False
        return self._renderer[].is_open()

    fn check_renderer_quit(mut self) -> Bool:
        """Check if user requested to close the renderer window."""
        if not self._renderer_initialized:
            return False
        return self._renderer[].check_quit()

    fn renderer_delay(self, ms: Int) -> None:
        """Delay for specified milliseconds."""
        if not self._renderer_initialized:
            return
        self._renderer[].delay(ms)

    # =========================================================================
    # GPUContinuousEnv Interface (Static GPU Kernels)
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
        ACTION_DIM_VAL: Int,
        MAX_STEPS_VAL: Int = 1000,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[gpu_dtype]] = [],
    ) raises:
        """Batched GPU step function using GC physics engine.

        Uses DefaultIntegrator.step_gpu for physics.
        Runs FRAME_SKIP=5 sub-steps per env step with joint limit
        enforcement after each sub-step, matching CPU behavior.
        """

        # Create model buffer on GPU with curriculum values set directly
        comptime MODEL_SIZE = model_size[
            HalfCheetahGC.NUM_BODIES, HalfCheetahGC.NUM_JOINTS
        ]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)

        var max_pitch = (
            curriculum_values[1] if len(curriculum_values)
            > 1 else HalfCheetahGCConstants[gpu_dtype].MAX_PITCH
        )
        Self._init_model_gpu(ctx, model_buf, max_pitch)

        # Store prev_x_position before physics (for position-based velocity)
        Self._store_prev_x_gpu[BATCH_SIZE, STATE_SIZE_VAL](ctx, states_buf)

        # Apply actions to qfrc in state buffer
        Self._apply_actions_gpu[BATCH_SIZE, STATE_SIZE_VAL, ACTION_DIM_VAL](
            ctx, states_buf, actions_buf
        )

        # Run FRAME_SKIP physics sub-steps with joint limit enforcement
        comptime C = HalfCheetahGCConstants[gpu_dtype]
        for _ in range(C.FRAME_SKIP):
            EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
                gpu_dtype,
                Self.NQ,
                Self.NV,
                Self.NUM_BODIES,
                Self.NUM_JOINTS,
                Self.MAX_CONTACTS,
                BATCH_SIZE,
            ](
                ctx,
                states_buf,
                model_buf,
                dt=Scalar[gpu_dtype](C.DT),
                gravity_z=Scalar[gpu_dtype](-9.81),
                ground_z=Scalar[gpu_dtype](0.0),
            )
            # Enforce joint limits after each sub-step (matching CPU)
            Self._enforce_joint_limits_gpu[BATCH_SIZE, STATE_SIZE_VAL](
                ctx, states_buf
            )

        # Extract observations, compute rewards, check termination
        Self._extract_obs_rewards_dones_gpu[
            BATCH_SIZE,
            STATE_SIZE_VAL,
            MODEL_SIZE,
            OBS_DIM_VAL,
            MAX_STEPS_VAL,
        ](
            ctx,
            states_buf,
            model_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            obs_buf,
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments on GPU.

        Also runs forward kinematics to compute xpos/xquat, matching CPU behavior.
        """
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            seed: Int,
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](states, i, seed)

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            Int(rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

        # Run forward kinematics to compute xpos/xquat (matching CPU behavior)
        comptime MODEL_SIZE = model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        Self._init_model_gpu(ctx, model_buf)

        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn fk_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            forward_kinematics_gpu[
                gpu_dtype,
                Self.NQ,
                Self.NV,
                Self.NUM_BODIES,
                Self.NUM_JOINTS,
                Self.MAX_CONTACTS,
                STATE_SIZE_VAL,
                MODEL_SIZE,
                BATCH_SIZE,
            ](i, states, model)

        ctx.enqueue_function[fk_wrapper, fk_wrapper](
            states,
            model,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
    ) raises:
        """Reset only done environments on GPU.

        Also runs forward kinematics for reset environments, matching CPU behavior.
        """
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        # Create model buffer for FK
        comptime MODEL_SIZE = model_size[Self.NUM_BODIES, Self.NUM_JOINTS]()
        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        Self._init_model_gpu(ctx, model_buf)

        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn selective_reset_with_fk_wrapper(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            seed: Int,
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            if dones[i] > Scalar[gpu_dtype](0.5):
                Self._reset_env_gpu[BATCH_SIZE, STATE_SIZE_VAL](states, i, seed)
                # Run FK for this environment to compute xpos/xquat
                forward_kinematics_gpu[
                    gpu_dtype,
                    Self.NQ,
                    Self.NV,
                    Self.NUM_BODIES,
                    Self.NUM_JOINTS,
                    Self.MAX_CONTACTS,
                    STATE_SIZE_VAL,
                    MODEL_SIZE,
                    BATCH_SIZE,
                ](i, states, model)
                dones[i] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[
            selective_reset_with_fk_wrapper, selective_reset_with_fk_wrapper
        ](
            states,
            dones,
            model,
            Int(rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Extract correct observations from GC state buffer.

        For GC environments, observations are NOT at state[0:OBS_DIM].
        The correct 17D observation is: qpos[1:9] (8 positions) + qvel[0:9] (9 velocities).
        This excludes rootx (qpos[0]) and head (qpos[9]) from observations.
        """
        var states = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var obs = LayoutTensor[
            gpu_dtype,
            Layout.row_major(BATCH_SIZE, OBS_DIM_VAL),
            MutAnyOrigin,
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime QPOS_OFF = qpos_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()
        comptime QVEL_OFF = qvel_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()

        @always_inline
        fn extract_gc_obs(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            obs: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, OBS_DIM_VAL),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Position observations (8D): qpos[1:9] (excluding rootx and head)
            obs[env, 0] = states[env, QPOS_OFF + 1]  # rootz
            obs[env, 1] = states[env, QPOS_OFF + 2]  # rooty
            obs[env, 2] = states[env, QPOS_OFF + 3]  # bthigh
            obs[env, 3] = states[env, QPOS_OFF + 4]  # bshin
            obs[env, 4] = states[env, QPOS_OFF + 5]  # bfoot
            obs[env, 5] = states[env, QPOS_OFF + 6]  # fthigh
            obs[env, 6] = states[env, QPOS_OFF + 7]  # fshin
            obs[env, 7] = states[env, QPOS_OFF + 8]  # ffoot

            # Velocity observations (9D): qvel[0:9] (excluding head)
            obs[env, 8] = states[env, QVEL_OFF + 0]  # x_vel
            obs[env, 9] = states[env, QVEL_OFF + 1]  # z_vel
            obs[env, 10] = states[env, QVEL_OFF + 2]  # y_angvel
            obs[env, 11] = states[env, QVEL_OFF + 3]  # bthigh_vel
            obs[env, 12] = states[env, QVEL_OFF + 4]  # bshin_vel
            obs[env, 13] = states[env, QVEL_OFF + 5]  # bfoot_vel
            obs[env, 14] = states[env, QVEL_OFF + 6]  # fthigh_vel
            obs[env, 15] = states[env, QVEL_OFF + 7]  # fshin_vel
            obs[env, 16] = states[env, QVEL_OFF + 8]  # ffoot_vel

        ctx.enqueue_function[extract_gc_obs, extract_gc_obs](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Helper Functions
    # =========================================================================

    @staticmethod
    fn _init_model_gpu(
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[gpu_dtype],
        max_pitch: Scalar[gpu_dtype] = HalfCheetahGCConstants[gpu_dtype].MAX_PITCH,
    ) raises:
        """Initialize model buffer with HalfCheetahGC parameters for GC physics engine.

        Uses HalfCheetahGCConstants for all body dimensions and joint limits.
        Curriculum parameters can be set directly to avoid GPU↔CPU round-trips.

        Args:
            ctx: GPU device context.
            model_buf: Model buffer to initialize.
            max_pitch: Maximum torso pitch angle for health check.
        """
        comptime C = HalfCheetahGCConstants[gpu_dtype]

        comptime MODEL_SIZE = model_size[
            HalfCheetahGC.NUM_BODIES, HalfCheetahGC.NUM_JOINTS
        ]()

        var model_host = List[Scalar[gpu_dtype]](capacity=MODEL_SIZE)
        for _ in range(MODEL_SIZE):
            model_host.append(Scalar[gpu_dtype](0.0))

        # Body dimensions from constants
        var capsule_radius = C.CAPSULE_RADIUS
        var torso_half = C.TORSO_HALF_LENGTH
        var head_half = C.HEAD_HALF_LENGTH
        var bthigh_half = C.BTHIGH_HALF_LENGTH
        var bshin_half = C.BSHIN_HALF_LENGTH
        var bfoot_half = C.BFOOT_HALF_LENGTH
        var fthigh_half = C.FTHIGH_HALF_LENGTH
        var fshin_half = C.FSHIN_HALF_LENGTH
        var ffoot_half = C.FFOOT_HALF_LENGTH

        # Helper to compute capsule inertia
        fn compute_capsule_inertia(
            mass: Scalar[gpu_dtype],
            radius: Scalar[gpu_dtype],
            half_length: Scalar[gpu_dtype],
        ) -> Tuple[Scalar[gpu_dtype], Scalar[gpu_dtype], Scalar[gpu_dtype]]:
            var r2 = radius * radius
            var L = (
                Scalar[gpu_dtype](2.0) * half_length
                + Scalar[gpu_dtype](2.0) * radius
            )
            var L2 = L * L
            var I_trans = (
                mass
                * (Scalar[gpu_dtype](3.0) * r2 + L2)
                / Scalar[gpu_dtype](12.0)
            )
            var I_axial = Scalar[gpu_dtype](0.5) * mass * r2
            return (I_trans, I_trans, I_axial)

        # Quaternion for 90° Y rotation (makes capsule horizontal)
        var quat_90y_x = Scalar[gpu_dtype](0.0)
        var quat_90y_y = Scalar[gpu_dtype](0.70710678)
        var quat_90y_z = Scalar[gpu_dtype](0.0)
        var quat_90y_w = Scalar[gpu_dtype](0.70710678)

        # Quaternion for -90° Y rotation
        var quat_neg90y_x = Scalar[gpu_dtype](0.0)
        var quat_neg90y_y = Scalar[gpu_dtype](-0.70710678)
        var quat_neg90y_z = Scalar[gpu_dtype](0.0)
        var quat_neg90y_w = Scalar[gpu_dtype](0.70710678)

        # =================================================================
        # Body 0: Torso (horizontal, 90° Y rotation)
        # =================================================================
        var b0 = model_body_offset(0)
        var torso_mass = C.TORSO_MASS
        var torso_inertia = compute_capsule_inertia(
            torso_mass, capsule_radius, torso_half
        )

        model_host[b0 + BODY_IDX_MASS] = torso_mass
        model_host[b0 + BODY_IDX_INV_MASS] = Scalar[gpu_dtype](1.0) / torso_mass
        model_host[b0 + BODY_IDX_IXX] = torso_inertia[0]
        model_host[b0 + BODY_IDX_IYY] = torso_inertia[1]
        model_host[b0 + BODY_IDX_IZZ] = torso_inertia[2]
        model_host[b0 + BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / torso_inertia[0]
        )
        model_host[b0 + BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / torso_inertia[1]
        )
        model_host[b0 + BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / torso_inertia[2]
        )
        model_host[b0 + BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b0 + BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b0 + BODY_IDX_POS_Z] = Scalar[gpu_dtype](0.0)
        model_host[b0 + BODY_IDX_QUAT_X] = quat_90y_x
        model_host[b0 + BODY_IDX_QUAT_Y] = quat_90y_y
        model_host[b0 + BODY_IDX_QUAT_Z] = quat_90y_z
        model_host[b0 + BODY_IDX_QUAT_W] = quat_90y_w
        model_host[b0 + BODY_IDX_PARENT] = Scalar[gpu_dtype](-1)  # World
        model_host[b0 + BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](GEOM_CAPSULE)
        model_host[b0 + BODY_IDX_RADIUS] = capsule_radius
        model_host[b0 + BODY_IDX_HALF_LENGTH] = torso_half

        # =================================================================
        # Body 1: Back Thigh (vertical, at back of torso)
        # =================================================================
        var b1 = model_body_offset(1)
        var bthigh_mass = C.BTHIGH_MASS
        var bthigh_inertia = compute_capsule_inertia(
            bthigh_mass, capsule_radius, bthigh_half
        )

        model_host[b1 + BODY_IDX_MASS] = bthigh_mass
        model_host[b1 + BODY_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / bthigh_mass
        )
        model_host[b1 + BODY_IDX_IXX] = bthigh_inertia[0]
        model_host[b1 + BODY_IDX_IYY] = bthigh_inertia[1]
        model_host[b1 + BODY_IDX_IZZ] = bthigh_inertia[2]
        model_host[b1 + BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / bthigh_inertia[0]
        )
        model_host[b1 + BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / bthigh_inertia[1]
        )
        model_host[b1 + BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / bthigh_inertia[2]
        )
        # body_pos in torso frame: (bthigh_half, 0, -torso_half)
        model_host[b1 + BODY_IDX_POS_X] = bthigh_half
        model_host[b1 + BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b1 + BODY_IDX_POS_Z] = -torso_half
        model_host[b1 + BODY_IDX_QUAT_X] = quat_neg90y_x
        model_host[b1 + BODY_IDX_QUAT_Y] = quat_neg90y_y
        model_host[b1 + BODY_IDX_QUAT_Z] = quat_neg90y_z
        model_host[b1 + BODY_IDX_QUAT_W] = quat_neg90y_w
        model_host[b1 + BODY_IDX_PARENT] = Scalar[gpu_dtype](0)  # Torso
        model_host[b1 + BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](GEOM_CAPSULE)
        model_host[b1 + BODY_IDX_RADIUS] = capsule_radius
        model_host[b1 + BODY_IDX_HALF_LENGTH] = bthigh_half

        # =================================================================
        # Body 2: Back Shin (vertical, below bthigh)
        # =================================================================
        var b2 = model_body_offset(2)
        var bshin_mass = C.BSHIN_MASS
        var bshin_inertia = compute_capsule_inertia(
            bshin_mass, capsule_radius, bshin_half
        )

        model_host[b2 + BODY_IDX_MASS] = bshin_mass
        model_host[b2 + BODY_IDX_INV_MASS] = Scalar[gpu_dtype](1.0) / bshin_mass
        model_host[b2 + BODY_IDX_IXX] = bshin_inertia[0]
        model_host[b2 + BODY_IDX_IYY] = bshin_inertia[1]
        model_host[b2 + BODY_IDX_IZZ] = bshin_inertia[2]
        model_host[b2 + BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / bshin_inertia[0]
        )
        model_host[b2 + BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / bshin_inertia[1]
        )
        model_host[b2 + BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / bshin_inertia[2]
        )
        model_host[b2 + BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b2 + BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b2 + BODY_IDX_POS_Z] = -(bthigh_half + bshin_half)
        model_host[b2 + BODY_IDX_QUAT_X] = Scalar[gpu_dtype](0.0)
        model_host[b2 + BODY_IDX_QUAT_Y] = Scalar[gpu_dtype](0.0)
        model_host[b2 + BODY_IDX_QUAT_Z] = Scalar[gpu_dtype](0.0)
        model_host[b2 + BODY_IDX_QUAT_W] = Scalar[gpu_dtype](1.0)
        model_host[b2 + BODY_IDX_PARENT] = Scalar[gpu_dtype](1)  # BThigh
        model_host[b2 + BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](GEOM_CAPSULE)
        model_host[b2 + BODY_IDX_RADIUS] = capsule_radius
        model_host[b2 + BODY_IDX_HALF_LENGTH] = bshin_half

        # =================================================================
        # Body 3: Back Foot (horizontal)
        # =================================================================
        var b3 = model_body_offset(3)
        var bfoot_mass = C.BFOOT_MASS
        var bfoot_inertia = compute_capsule_inertia(
            bfoot_mass, capsule_radius, bfoot_half
        )

        model_host[b3 + BODY_IDX_MASS] = bfoot_mass
        model_host[b3 + BODY_IDX_INV_MASS] = Scalar[gpu_dtype](1.0) / bfoot_mass
        model_host[b3 + BODY_IDX_IXX] = bfoot_inertia[0]
        model_host[b3 + BODY_IDX_IYY] = bfoot_inertia[1]
        model_host[b3 + BODY_IDX_IZZ] = bfoot_inertia[2]
        model_host[b3 + BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / bfoot_inertia[0]
        )
        model_host[b3 + BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / bfoot_inertia[1]
        )
        model_host[b3 + BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / bfoot_inertia[2]
        )
        model_host[b3 + BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b3 + BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b3 + BODY_IDX_POS_Z] = -bshin_half
        model_host[b3 + BODY_IDX_QUAT_X] = quat_90y_x
        model_host[b3 + BODY_IDX_QUAT_Y] = quat_90y_y
        model_host[b3 + BODY_IDX_QUAT_Z] = quat_90y_z
        model_host[b3 + BODY_IDX_QUAT_W] = quat_90y_w
        model_host[b3 + BODY_IDX_PARENT] = Scalar[gpu_dtype](2)  # BShin
        model_host[b3 + BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](GEOM_CAPSULE)
        model_host[b3 + BODY_IDX_RADIUS] = capsule_radius
        model_host[b3 + BODY_IDX_HALF_LENGTH] = bfoot_half

        # =================================================================
        # Body 4: Front Thigh (vertical, at front of torso)
        # =================================================================
        var b4 = model_body_offset(4)
        var fthigh_mass = C.FTHIGH_MASS
        var fthigh_inertia = compute_capsule_inertia(
            fthigh_mass, capsule_radius, fthigh_half
        )

        model_host[b4 + BODY_IDX_MASS] = fthigh_mass
        model_host[b4 + BODY_IDX_INV_MASS] = (
            Scalar[gpu_dtype](1.0) / fthigh_mass
        )
        model_host[b4 + BODY_IDX_IXX] = fthigh_inertia[0]
        model_host[b4 + BODY_IDX_IYY] = fthigh_inertia[1]
        model_host[b4 + BODY_IDX_IZZ] = fthigh_inertia[2]
        model_host[b4 + BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / fthigh_inertia[0]
        )
        model_host[b4 + BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / fthigh_inertia[1]
        )
        model_host[b4 + BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / fthigh_inertia[2]
        )
        # body_pos in torso frame: (fthigh_half, 0, torso_half)
        model_host[b4 + BODY_IDX_POS_X] = fthigh_half
        model_host[b4 + BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b4 + BODY_IDX_POS_Z] = torso_half
        model_host[b4 + BODY_IDX_QUAT_X] = quat_neg90y_x
        model_host[b4 + BODY_IDX_QUAT_Y] = quat_neg90y_y
        model_host[b4 + BODY_IDX_QUAT_Z] = quat_neg90y_z
        model_host[b4 + BODY_IDX_QUAT_W] = quat_neg90y_w
        model_host[b4 + BODY_IDX_PARENT] = Scalar[gpu_dtype](0)  # Torso
        model_host[b4 + BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](GEOM_CAPSULE)
        model_host[b4 + BODY_IDX_RADIUS] = capsule_radius
        model_host[b4 + BODY_IDX_HALF_LENGTH] = fthigh_half

        # =================================================================
        # Body 5: Front Shin (vertical, below fthigh)
        # =================================================================
        var b5 = model_body_offset(5)
        var fshin_mass = C.FSHIN_MASS
        var fshin_inertia = compute_capsule_inertia(
            fshin_mass, capsule_radius, fshin_half
        )

        model_host[b5 + BODY_IDX_MASS] = fshin_mass
        model_host[b5 + BODY_IDX_INV_MASS] = Scalar[gpu_dtype](1.0) / fshin_mass
        model_host[b5 + BODY_IDX_IXX] = fshin_inertia[0]
        model_host[b5 + BODY_IDX_IYY] = fshin_inertia[1]
        model_host[b5 + BODY_IDX_IZZ] = fshin_inertia[2]
        model_host[b5 + BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / fshin_inertia[0]
        )
        model_host[b5 + BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / fshin_inertia[1]
        )
        model_host[b5 + BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / fshin_inertia[2]
        )
        model_host[b5 + BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b5 + BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b5 + BODY_IDX_POS_Z] = -(fthigh_half + fshin_half)
        model_host[b5 + BODY_IDX_QUAT_X] = Scalar[gpu_dtype](0.0)
        model_host[b5 + BODY_IDX_QUAT_Y] = Scalar[gpu_dtype](0.0)
        model_host[b5 + BODY_IDX_QUAT_Z] = Scalar[gpu_dtype](0.0)
        model_host[b5 + BODY_IDX_QUAT_W] = Scalar[gpu_dtype](1.0)
        model_host[b5 + BODY_IDX_PARENT] = Scalar[gpu_dtype](4)  # FThigh
        model_host[b5 + BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](GEOM_CAPSULE)
        model_host[b5 + BODY_IDX_RADIUS] = capsule_radius
        model_host[b5 + BODY_IDX_HALF_LENGTH] = fshin_half

        # =================================================================
        # Body 6: Front Foot (horizontal)
        # =================================================================
        var b6 = model_body_offset(6)
        var ffoot_mass = C.FFOOT_MASS
        var ffoot_inertia = compute_capsule_inertia(
            ffoot_mass, capsule_radius, ffoot_half
        )

        model_host[b6 + BODY_IDX_MASS] = ffoot_mass
        model_host[b6 + BODY_IDX_INV_MASS] = Scalar[gpu_dtype](1.0) / ffoot_mass
        model_host[b6 + BODY_IDX_IXX] = ffoot_inertia[0]
        model_host[b6 + BODY_IDX_IYY] = ffoot_inertia[1]
        model_host[b6 + BODY_IDX_IZZ] = ffoot_inertia[2]
        model_host[b6 + BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / ffoot_inertia[0]
        )
        model_host[b6 + BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / ffoot_inertia[1]
        )
        model_host[b6 + BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / ffoot_inertia[2]
        )
        model_host[b6 + BODY_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[b6 + BODY_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[b6 + BODY_IDX_POS_Z] = -fshin_half
        model_host[b6 + BODY_IDX_QUAT_X] = quat_90y_x
        model_host[b6 + BODY_IDX_QUAT_Y] = quat_90y_y
        model_host[b6 + BODY_IDX_QUAT_Z] = quat_90y_z
        model_host[b6 + BODY_IDX_QUAT_W] = quat_90y_w
        model_host[b6 + BODY_IDX_PARENT] = Scalar[gpu_dtype](5)  # FShin
        model_host[b6 + BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](GEOM_CAPSULE)
        model_host[b6 + BODY_IDX_RADIUS] = capsule_radius
        model_host[b6 + BODY_IDX_HALF_LENGTH] = ffoot_half

        # =================================================================
        # Body 7: Head (tilted capsule at front of torso)
        # =================================================================
        var b7 = model_body_offset(7)
        var head_mass = C.HEAD_MASS
        var head_inertia = compute_capsule_inertia(
            head_mass, capsule_radius, head_half
        )

        model_host[b7 + BODY_IDX_MASS] = head_mass
        model_host[b7 + BODY_IDX_INV_MASS] = Scalar[gpu_dtype](1.0) / head_mass
        model_host[b7 + BODY_IDX_IXX] = head_inertia[0]
        model_host[b7 + BODY_IDX_IYY] = head_inertia[1]
        model_host[b7 + BODY_IDX_IZZ] = head_inertia[2]
        model_host[b7 + BODY_IDX_INV_IXX] = (
            Scalar[gpu_dtype](1.0) / head_inertia[0]
        )
        model_host[b7 + BODY_IDX_INV_IYY] = (
            Scalar[gpu_dtype](1.0) / head_inertia[1]
        )
        model_host[b7 + BODY_IDX_INV_IZZ] = (
            Scalar[gpu_dtype](1.0) / head_inertia[2]
        )
        # Head position in torso frame
        var head_pos_x = -C.HEAD_POS_Z  # -0.1
        var head_pos_y = C.HEAD_POS_Y  # 0.0
        var head_pos_z = C.HEAD_POS_X  # 0.6
        model_host[b7 + BODY_IDX_POS_X] = head_pos_x
        model_host[b7 + BODY_IDX_POS_Y] = head_pos_y
        model_host[b7 + BODY_IDX_POS_Z] = head_pos_z
        # Head rotation: 0.87 - π/2 ≈ -0.7 rad Y
        var head_angle = C.HEAD_AXIS_ANGLE - Scalar[gpu_dtype](1.5707963268)
        var head_sin = sin(head_angle / Scalar[gpu_dtype](2.0))
        var head_cos = cos(head_angle / Scalar[gpu_dtype](2.0))
        model_host[b7 + BODY_IDX_QUAT_X] = Scalar[gpu_dtype](0.0)
        model_host[b7 + BODY_IDX_QUAT_Y] = head_sin
        model_host[b7 + BODY_IDX_QUAT_Z] = Scalar[gpu_dtype](0.0)
        model_host[b7 + BODY_IDX_QUAT_W] = head_cos
        model_host[b7 + BODY_IDX_PARENT] = Scalar[gpu_dtype](0)  # Torso
        model_host[b7 + BODY_IDX_GEOM_TYPE] = Scalar[gpu_dtype](GEOM_CAPSULE)
        model_host[b7 + BODY_IDX_RADIUS] = capsule_radius
        model_host[b7 + BODY_IDX_HALF_LENGTH] = head_half

        # =================================================================
        # Joint 0: RootX - Slide joint, X-axis translation (body 0)
        # =================================================================
        var j0 = model_joint_offset[HalfCheetahGC.NUM_BODIES](0)
        model_host[j0 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_SLIDE)
        model_host[j0 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](0)
        model_host[j0 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](0)
        model_host[j0 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](0)
        model_host[j0 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j0 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j0 + JOINT_IDX_POS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j0 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](1.0)
        model_host[j0 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j0 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j0 + JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](0.0)
        model_host[j0 + JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](-1e10)
        model_host[j0 + JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](1e10)
        model_host[j0 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.0)
        model_host[j0 + JOINT_IDX_DAMPING] = Scalar[gpu_dtype](0.0)
        model_host[j0 + JOINT_IDX_STIFFNESS] = Scalar[gpu_dtype](0.0)

        # =================================================================
        # Joint 1: RootZ - Slide joint, Z-axis translation (body 0)
        # =================================================================
        var j1 = model_joint_offset[HalfCheetahGC.NUM_BODIES](1)
        model_host[j1 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_SLIDE)
        model_host[j1 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](0)
        model_host[j1 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](1)
        model_host[j1 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](1)
        model_host[j1 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j1 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j1 + JOINT_IDX_POS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j1 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j1 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j1 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](1.0)
        model_host[j1 + JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](0.0)
        model_host[j1 + JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](-1e10)
        model_host[j1 + JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](1e10)
        model_host[j1 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.0)
        model_host[j1 + JOINT_IDX_DAMPING] = Scalar[gpu_dtype](0.0)
        model_host[j1 + JOINT_IDX_STIFFNESS] = Scalar[gpu_dtype](0.0)

        # =================================================================
        # Joint 2: RootY - Hinge joint, Y-axis rotation (body 0)
        # =================================================================
        var j2 = model_joint_offset[HalfCheetahGC.NUM_BODIES](2)
        model_host[j2 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_HINGE)
        model_host[j2 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](0)
        model_host[j2 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](2)
        model_host[j2 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](2)
        model_host[j2 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j2 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j2 + JOINT_IDX_POS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j2 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j2 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j2 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j2 + JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](0.0)
        model_host[j2 + JOINT_IDX_RANGE_MIN] = Scalar[gpu_dtype](-1e10)
        model_host[j2 + JOINT_IDX_RANGE_MAX] = Scalar[gpu_dtype](1e10)
        model_host[j2 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.0)
        model_host[j2 + JOINT_IDX_DAMPING] = Scalar[gpu_dtype](0.0)
        model_host[j2 + JOINT_IDX_STIFFNESS] = Scalar[gpu_dtype](0.0)

        # =================================================================
        # Joint 3: BThigh - Hinge joint, Y-axis rotation (body 1)
        # =================================================================
        var j3 = model_joint_offset[HalfCheetahGC.NUM_BODIES](3)
        model_host[j3 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_HINGE)
        model_host[j3 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](1)
        model_host[j3 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](3)
        model_host[j3 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](3)
        model_host[j3 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j3 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j3 + JOINT_IDX_POS_Z] = -torso_half
        model_host[j3 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j3 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j3 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j3 + JOINT_IDX_TAU_LIMIT] = C.BTHIGH_GEAR
        model_host[j3 + JOINT_IDX_RANGE_MIN] = C.BTHIGH_JOINT_MIN
        model_host[j3 + JOINT_IDX_RANGE_MAX] = C.BTHIGH_JOINT_MAX
        model_host[j3 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.1)
        model_host[j3 + JOINT_IDX_DAMPING] = C.BTHIGH_DAMPING
        model_host[j3 + JOINT_IDX_STIFFNESS] = C.BTHIGH_STIFFNESS

        # =================================================================
        # Joint 4: BShin - Hinge joint, Y-axis rotation (body 2)
        # =================================================================
        var j4 = model_joint_offset[HalfCheetahGC.NUM_BODIES](4)
        model_host[j4 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_HINGE)
        model_host[j4 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](2)
        model_host[j4 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](4)
        model_host[j4 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](4)
        model_host[j4 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j4 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j4 + JOINT_IDX_POS_Z] = -bthigh_half
        model_host[j4 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j4 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j4 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j4 + JOINT_IDX_TAU_LIMIT] = C.BSHIN_GEAR
        model_host[j4 + JOINT_IDX_RANGE_MIN] = C.BSHIN_JOINT_MIN
        model_host[j4 + JOINT_IDX_RANGE_MAX] = C.BSHIN_JOINT_MAX
        model_host[j4 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.1)
        model_host[j4 + JOINT_IDX_DAMPING] = C.BSHIN_DAMPING
        model_host[j4 + JOINT_IDX_STIFFNESS] = C.BSHIN_STIFFNESS

        # =================================================================
        # Joint 5: BFoot - Hinge joint, Y-axis rotation (body 3)
        # =================================================================
        var j5 = model_joint_offset[HalfCheetahGC.NUM_BODIES](5)
        model_host[j5 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_HINGE)
        model_host[j5 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](3)
        model_host[j5 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](5)
        model_host[j5 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](5)
        model_host[j5 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j5 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j5 + JOINT_IDX_POS_Z] = -bshin_half
        model_host[j5 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j5 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j5 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j5 + JOINT_IDX_TAU_LIMIT] = C.BFOOT_GEAR
        model_host[j5 + JOINT_IDX_RANGE_MIN] = C.BFOOT_JOINT_MIN
        model_host[j5 + JOINT_IDX_RANGE_MAX] = C.BFOOT_JOINT_MAX
        model_host[j5 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.1)
        model_host[j5 + JOINT_IDX_DAMPING] = C.BFOOT_DAMPING
        model_host[j5 + JOINT_IDX_STIFFNESS] = C.BFOOT_STIFFNESS

        # =================================================================
        # Joint 6: FThigh - Hinge joint, Y-axis rotation (body 4)
        # =================================================================
        var j6 = model_joint_offset[HalfCheetahGC.NUM_BODIES](6)
        model_host[j6 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_HINGE)
        model_host[j6 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](4)
        model_host[j6 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](6)
        model_host[j6 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](6)
        model_host[j6 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j6 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j6 + JOINT_IDX_POS_Z] = torso_half
        model_host[j6 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j6 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j6 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j6 + JOINT_IDX_TAU_LIMIT] = C.FTHIGH_GEAR
        model_host[j6 + JOINT_IDX_RANGE_MIN] = C.FTHIGH_JOINT_MIN
        model_host[j6 + JOINT_IDX_RANGE_MAX] = C.FTHIGH_JOINT_MAX
        model_host[j6 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.1)
        model_host[j6 + JOINT_IDX_DAMPING] = C.FTHIGH_DAMPING
        model_host[j6 + JOINT_IDX_STIFFNESS] = C.FTHIGH_STIFFNESS

        # =================================================================
        # Joint 7: FShin - Hinge joint, Y-axis rotation (body 5)
        # =================================================================
        var j7 = model_joint_offset[HalfCheetahGC.NUM_BODIES](7)
        model_host[j7 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_HINGE)
        model_host[j7 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](5)
        model_host[j7 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](7)
        model_host[j7 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](7)
        model_host[j7 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j7 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j7 + JOINT_IDX_POS_Z] = -fthigh_half
        model_host[j7 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j7 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j7 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j7 + JOINT_IDX_TAU_LIMIT] = C.FSHIN_GEAR
        model_host[j7 + JOINT_IDX_RANGE_MIN] = C.FSHIN_JOINT_MIN
        model_host[j7 + JOINT_IDX_RANGE_MAX] = C.FSHIN_JOINT_MAX
        model_host[j7 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.1)
        model_host[j7 + JOINT_IDX_DAMPING] = C.FSHIN_DAMPING
        model_host[j7 + JOINT_IDX_STIFFNESS] = C.FSHIN_STIFFNESS

        # =================================================================
        # Joint 8: FFoot - Hinge joint, Y-axis rotation (body 6)
        # =================================================================
        var j8 = model_joint_offset[HalfCheetahGC.NUM_BODIES](8)
        model_host[j8 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_HINGE)
        model_host[j8 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](6)
        model_host[j8 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](8)
        model_host[j8 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](8)
        model_host[j8 + JOINT_IDX_POS_X] = Scalar[gpu_dtype](0.0)
        model_host[j8 + JOINT_IDX_POS_Y] = Scalar[gpu_dtype](0.0)
        model_host[j8 + JOINT_IDX_POS_Z] = -fshin_half
        model_host[j8 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j8 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j8 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j8 + JOINT_IDX_TAU_LIMIT] = C.FFOOT_GEAR
        model_host[j8 + JOINT_IDX_RANGE_MIN] = C.FFOOT_JOINT_MIN
        model_host[j8 + JOINT_IDX_RANGE_MAX] = C.FFOOT_JOINT_MAX
        model_host[j8 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.1)
        model_host[j8 + JOINT_IDX_DAMPING] = C.FFOOT_DAMPING
        model_host[j8 + JOINT_IDX_STIFFNESS] = C.FFOOT_STIFFNESS

        # =================================================================
        # Joint 9: Head - Hinge joint, Y-axis rotation (body 7, fixed)
        # =================================================================
        var j9 = model_joint_offset[HalfCheetahGC.NUM_BODIES](9)
        model_host[j9 + JOINT_IDX_TYPE] = Scalar[gpu_dtype](JNT_HINGE)
        model_host[j9 + JOINT_IDX_BODY_ID] = Scalar[gpu_dtype](7)
        model_host[j9 + JOINT_IDX_QPOS_ADR] = Scalar[gpu_dtype](9)
        model_host[j9 + JOINT_IDX_DOF_ADR] = Scalar[gpu_dtype](9)
        model_host[j9 + JOINT_IDX_POS_X] = head_pos_x
        model_host[j9 + JOINT_IDX_POS_Y] = head_pos_y
        model_host[j9 + JOINT_IDX_POS_Z] = head_pos_z
        model_host[j9 + JOINT_IDX_AXIS_X] = Scalar[gpu_dtype](0.0)
        model_host[j9 + JOINT_IDX_AXIS_Y] = Scalar[gpu_dtype](1.0)
        model_host[j9 + JOINT_IDX_AXIS_Z] = Scalar[gpu_dtype](0.0)
        model_host[j9 + JOINT_IDX_TAU_LIMIT] = Scalar[gpu_dtype](0.0)
        model_host[j9 + JOINT_IDX_RANGE_MIN] = C.HEAD_JOINT_MIN
        model_host[j9 + JOINT_IDX_RANGE_MAX] = C.HEAD_JOINT_MAX
        model_host[j9 + JOINT_IDX_ARMATURE] = Scalar[gpu_dtype](0.1)
        model_host[j9 + JOINT_IDX_DAMPING] = Scalar[gpu_dtype](0.01)
        model_host[j9 + JOINT_IDX_STIFFNESS] = Scalar[gpu_dtype](8.0)

        # =================================================================
        # Model Metadata
        # =================================================================
        var meta = model_metadata_offset[
            HalfCheetahGC.NUM_BODIES, HalfCheetahGC.NUM_JOINTS
        ]()
        model_host[meta + MODEL_META_IDX_NBODY] = Scalar[gpu_dtype](
            C.NUM_BODIES
        )
        model_host[meta + MODEL_META_IDX_NJOINT] = Scalar[gpu_dtype](
            C.NUM_JOINTS
        )
        model_host[meta + MODEL_META_IDX_GRAVITY_X] = Scalar[gpu_dtype](0.0)
        model_host[meta + MODEL_META_IDX_GRAVITY_Y] = Scalar[gpu_dtype](0.0)
        model_host[meta + MODEL_META_IDX_GRAVITY_Z] = C.GRAVITY_Z
        model_host[meta + MODEL_META_IDX_TIMESTEP] = C.DT
        model_host[meta + MODEL_META_IDX_GROUND_Z] = Scalar[gpu_dtype](0.0)
        model_host[meta + MODEL_META_IDX_FRICTION] = C.FRICTION
        # solref/solimp contact
        model_host[meta + MODEL_META_IDX_SOLREF_CONTACT_0] = C.SOLREF_CONTACT_0
        model_host[meta + MODEL_META_IDX_SOLREF_CONTACT_1] = C.SOLREF_CONTACT_1
        model_host[meta + MODEL_META_IDX_SOLIMP_CONTACT_0] = C.SOLIMP_CONTACT_0
        model_host[meta + MODEL_META_IDX_SOLIMP_CONTACT_1] = C.SOLIMP_CONTACT_1
        model_host[meta + MODEL_META_IDX_SOLIMP_CONTACT_2] = C.SOLIMP_CONTACT_2
        # solref/solimp limit
        model_host[meta + MODEL_META_IDX_SOLREF_LIMIT_0] = C.SOLREF_LIMIT_0
        model_host[meta + MODEL_META_IDX_SOLREF_LIMIT_1] = C.SOLREF_LIMIT_1
        model_host[meta + MODEL_META_IDX_SOLIMP_LIMIT_0] = C.SOLIMP_LIMIT_0
        model_host[meta + MODEL_META_IDX_SOLIMP_LIMIT_1] = C.SOLIMP_LIMIT_1
        model_host[meta + MODEL_META_IDX_SOLIMP_LIMIT_2] = C.SOLIMP_LIMIT_2

        # =================================================================
        # Curriculum Parameters (initialize to defaults)
        # =================================================================
        var curr = model_curriculum_offset[
            HalfCheetahGC.NUM_BODIES, HalfCheetahGC.NUM_JOINTS
        ]()
        model_host[curr + CURRICULUM_IDX_MIN_HEIGHT] = Scalar[gpu_dtype](
            0.0
        )  # Not used for HalfCheetah
        model_host[curr + CURRICULUM_IDX_MAX_PITCH] = max_pitch

        # Copy to GPU
        ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())

    @staticmethod
    fn init_model_gpu_with_curriculum(
        ctx: DeviceContext,
        mut model_buf: DeviceBuffer[gpu_dtype],
        max_pitch: Scalar[gpu_dtype],
    ) raises:
        """Initialize model buffer with specified curriculum parameters.

        Args:
            ctx: GPU device context.
            model_buf: Model buffer to initialize.
            max_pitch: Maximum torso pitch angle for health check.
        """
        Self._init_model_gpu(ctx, model_buf, max_pitch)

    @staticmethod
    fn _store_prev_x_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[gpu_dtype],) raises:
        """Store current rootx position into metadata for velocity computation.
        """
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime QPOS_OFF = qpos_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()
        comptime META_OFF = metadata_offset[
            HalfCheetahGC.NQ,
            HalfCheetahGC.NV,
            HalfCheetahGC.NUM_BODIES,
            HalfCheetahGC.MAX_CONTACTS,
        ]()

        @always_inline
        fn store_prev_x_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            # Store current rootx into metadata prev_x slot
            states[env, META_OFF + META_IDX_PREV_X] = states[env, QPOS_OFF + 0]

        ctx.enqueue_function[store_prev_x_kernel, store_prev_x_kernel](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _enforce_joint_limits_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[gpu_dtype],) raises:
        """Enforce joint position limits on GPU, matching CPU behavior.

        Clamps actuated joint positions to their defined limits.
        Zeros velocity when hitting a limit (simple contact model).
        """
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime C = HalfCheetahGCConstants[gpu_dtype]
        comptime QPOS_OFF = qpos_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()
        comptime QVEL_OFF = qvel_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()

        @always_inline
        fn enforce_limits_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Joint indices 3-8 are actuated (bthigh, bshin, bfoot, fthigh, fshin, ffoot)
            # Clamp position and zero velocity at limits

            # bthigh (joint 3)
            var bthigh_pos = states[env, QPOS_OFF + 3]
            var bthigh_vel = states[env, QVEL_OFF + 3]
            if bthigh_pos < C.BTHIGH_JOINT_MIN:
                states[env, QPOS_OFF + 3] = C.BTHIGH_JOINT_MIN
                if bthigh_vel < Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 3] = Scalar[gpu_dtype](0)
            elif bthigh_pos > C.BTHIGH_JOINT_MAX:
                states[env, QPOS_OFF + 3] = C.BTHIGH_JOINT_MAX
                if bthigh_vel > Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 3] = Scalar[gpu_dtype](0)

            # bshin (joint 4)
            var bshin_pos = states[env, QPOS_OFF + 4]
            var bshin_vel = states[env, QVEL_OFF + 4]
            if bshin_pos < C.BSHIN_JOINT_MIN:
                states[env, QPOS_OFF + 4] = C.BSHIN_JOINT_MIN
                if bshin_vel < Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 4] = Scalar[gpu_dtype](0)
            elif bshin_pos > C.BSHIN_JOINT_MAX:
                states[env, QPOS_OFF + 4] = C.BSHIN_JOINT_MAX
                if bshin_vel > Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 4] = Scalar[gpu_dtype](0)

            # bfoot (joint 5)
            var bfoot_pos = states[env, QPOS_OFF + 5]
            var bfoot_vel = states[env, QVEL_OFF + 5]
            if bfoot_pos < C.BFOOT_JOINT_MIN:
                states[env, QPOS_OFF + 5] = C.BFOOT_JOINT_MIN
                if bfoot_vel < Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 5] = Scalar[gpu_dtype](0)
            elif bfoot_pos > C.BFOOT_JOINT_MAX:
                states[env, QPOS_OFF + 5] = C.BFOOT_JOINT_MAX
                if bfoot_vel > Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 5] = Scalar[gpu_dtype](0)

            # fthigh (joint 6)
            var fthigh_pos = states[env, QPOS_OFF + 6]
            var fthigh_vel = states[env, QVEL_OFF + 6]
            if fthigh_pos < C.FTHIGH_JOINT_MIN:
                states[env, QPOS_OFF + 6] = C.FTHIGH_JOINT_MIN
                if fthigh_vel < Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 6] = Scalar[gpu_dtype](0)
            elif fthigh_pos > C.FTHIGH_JOINT_MAX:
                states[env, QPOS_OFF + 6] = C.FTHIGH_JOINT_MAX
                if fthigh_vel > Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 6] = Scalar[gpu_dtype](0)

            # fshin (joint 7)
            var fshin_pos = states[env, QPOS_OFF + 7]
            var fshin_vel = states[env, QVEL_OFF + 7]
            if fshin_pos < C.FSHIN_JOINT_MIN:
                states[env, QPOS_OFF + 7] = C.FSHIN_JOINT_MIN
                if fshin_vel < Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 7] = Scalar[gpu_dtype](0)
            elif fshin_pos > C.FSHIN_JOINT_MAX:
                states[env, QPOS_OFF + 7] = C.FSHIN_JOINT_MAX
                if fshin_vel > Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 7] = Scalar[gpu_dtype](0)

            # ffoot (joint 8)
            var ffoot_pos = states[env, QPOS_OFF + 8]
            var ffoot_vel = states[env, QVEL_OFF + 8]
            if ffoot_pos < C.FFOOT_JOINT_MIN:
                states[env, QPOS_OFF + 8] = C.FFOOT_JOINT_MIN
                if ffoot_vel < Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 8] = Scalar[gpu_dtype](0)
            elif ffoot_pos > C.FFOOT_JOINT_MAX:
                states[env, QPOS_OFF + 8] = C.FFOOT_JOINT_MAX
                if ffoot_vel > Scalar[gpu_dtype](0):
                    states[env, QVEL_OFF + 8] = Scalar[gpu_dtype](0)

        ctx.enqueue_function[enforce_limits_kernel, enforce_limits_kernel](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _apply_actions_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Apply actions as joint torques to qfrc in state buffer."""
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime C = HalfCheetahGCConstants[gpu_dtype]
        comptime QFRC_OFF = qfrc_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()

        @always_inline
        fn apply_actions_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, ACTION_DIM),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Clamp and scale actions by gear ratios
            var bthigh_action = actions[env, 0]
            var bshin_action = actions[env, 1]
            var bfoot_action = actions[env, 2]
            var fthigh_action = actions[env, 3]
            var fshin_action = actions[env, 4]
            var ffoot_action = actions[env, 5]

            # Clamp to [-1, 1]
            if bthigh_action > Scalar[gpu_dtype](1.0):
                bthigh_action = Scalar[gpu_dtype](1.0)
            elif bthigh_action < Scalar[gpu_dtype](-1.0):
                bthigh_action = Scalar[gpu_dtype](-1.0)

            if bshin_action > Scalar[gpu_dtype](1.0):
                bshin_action = Scalar[gpu_dtype](1.0)
            elif bshin_action < Scalar[gpu_dtype](-1.0):
                bshin_action = Scalar[gpu_dtype](-1.0)

            if bfoot_action > Scalar[gpu_dtype](1.0):
                bfoot_action = Scalar[gpu_dtype](1.0)
            elif bfoot_action < Scalar[gpu_dtype](-1.0):
                bfoot_action = Scalar[gpu_dtype](-1.0)

            if fthigh_action > Scalar[gpu_dtype](1.0):
                fthigh_action = Scalar[gpu_dtype](1.0)
            elif fthigh_action < Scalar[gpu_dtype](-1.0):
                fthigh_action = Scalar[gpu_dtype](-1.0)

            if fshin_action > Scalar[gpu_dtype](1.0):
                fshin_action = Scalar[gpu_dtype](1.0)
            elif fshin_action < Scalar[gpu_dtype](-1.0):
                fshin_action = Scalar[gpu_dtype](-1.0)

            if ffoot_action > Scalar[gpu_dtype](1.0):
                ffoot_action = Scalar[gpu_dtype](1.0)
            elif ffoot_action < Scalar[gpu_dtype](-1.0):
                ffoot_action = Scalar[gpu_dtype](-1.0)

            # Apply torques to joints 3-8 (actuated joints)
            states[env, QFRC_OFF + 3] = bthigh_action * C.BTHIGH_GEAR
            states[env, QFRC_OFF + 4] = bshin_action * C.BSHIN_GEAR
            states[env, QFRC_OFF + 5] = bfoot_action * C.BFOOT_GEAR
            states[env, QFRC_OFF + 6] = fthigh_action * C.FTHIGH_GEAR
            states[env, QFRC_OFF + 7] = fshin_action * C.FSHIN_GEAR
            states[env, QFRC_OFF + 8] = ffoot_action * C.FFOOT_GEAR

        ctx.enqueue_function[apply_actions_kernel, apply_actions_kernel](
            states,
            actions,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn _extract_obs_rewards_dones_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        OBS_DIM: Int,
        MAX_STEPS_VAL: Int = 1000,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        model_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Extract observations, compute rewards, check termination.

        Includes angle penalty in reward and health-based termination
        when TERMINATE_ON_UNHEALTHY is True.
        """
        var states = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var model = LayoutTensor[
            gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())
        var actions = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, 6), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var obs = LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime C = HalfCheetahGCConstants[gpu_dtype]
        comptime QPOS_OFF = qpos_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()
        comptime QVEL_OFF = qvel_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()
        comptime META_OFF = metadata_offset[
            HalfCheetahGC.NQ,
            HalfCheetahGC.NV,
            HalfCheetahGC.NUM_BODIES,
            HalfCheetahGC.MAX_CONTACTS,
        ]()
        comptime CURRICULUM_OFF = model_curriculum_offset[
            HalfCheetahGC.NUM_BODIES, HalfCheetahGC.NUM_JOINTS
        ]()

        @always_inline
        fn extract_kernel(
            states: LayoutTensor[
                gpu_dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE),
                MutAnyOrigin,
            ],
            model: LayoutTensor[
                gpu_dtype, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE, 6), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                gpu_dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Increment step counter
            var step_count = Int(
                rebind[Scalar[gpu_dtype]](
                    states[env, META_OFF + META_IDX_STEP_COUNT]
                )
            )
            step_count += 1
            states[env, META_OFF + META_IDX_STEP_COUNT] = Scalar[gpu_dtype](
                step_count
            )

            # Extract qpos (skip qpos[0] = rootx and qpos[9] = head for observation)
            var z_pos = states[env, QPOS_OFF + 1]  # rootz
            var y_angle = states[env, QPOS_OFF + 2]  # rooty
            var bthigh_angle = states[env, QPOS_OFF + 3]
            var bshin_angle = states[env, QPOS_OFF + 4]
            var bfoot_angle = states[env, QPOS_OFF + 5]
            var fthigh_angle = states[env, QPOS_OFF + 6]
            var fshin_angle = states[env, QPOS_OFF + 7]
            var ffoot_angle = states[env, QPOS_OFF + 8]

            # Extract qvel (skip qvel[9] = head for observation)
            var x_vel = states[env, QVEL_OFF + 0]  # rootx vel
            var z_vel = states[env, QVEL_OFF + 1]  # rootz vel
            var y_angvel = states[env, QVEL_OFF + 2]  # rooty vel
            var bthigh_angvel = states[env, QVEL_OFF + 3]
            var bshin_angvel = states[env, QVEL_OFF + 4]
            var bfoot_angvel = states[env, QVEL_OFF + 5]
            var fthigh_angvel = states[env, QVEL_OFF + 6]
            var fshin_angvel = states[env, QVEL_OFF + 7]
            var ffoot_angvel = states[env, QVEL_OFF + 8]

            # Build observation (17D)
            # Position observations (8D): qpos[1:9] (excluding rootx and head)
            obs[env, 0] = z_pos
            obs[env, 1] = y_angle
            obs[env, 2] = bthigh_angle
            obs[env, 3] = bshin_angle
            obs[env, 4] = bfoot_angle
            obs[env, 5] = fthigh_angle
            obs[env, 6] = fshin_angle
            obs[env, 7] = ffoot_angle
            # Velocity observations (9D): qvel[0:9] (excluding head)
            obs[env, 8] = x_vel
            obs[env, 9] = z_vel
            obs[env, 10] = y_angvel
            obs[env, 11] = bthigh_angvel
            obs[env, 12] = bshin_angvel
            obs[env, 13] = bfoot_angvel
            obs[env, 14] = fthigh_angvel
            obs[env, 15] = fshin_angvel
            obs[env, 16] = ffoot_angvel

            # Clamp actions for reward computation
            var bthigh_action = actions[env, 0]
            var bshin_action = actions[env, 1]
            var bfoot_action = actions[env, 2]
            var fthigh_action = actions[env, 3]
            var fshin_action = actions[env, 4]
            var ffoot_action = actions[env, 5]

            if bthigh_action > Scalar[gpu_dtype](1.0):
                bthigh_action = Scalar[gpu_dtype](1.0)
            elif bthigh_action < Scalar[gpu_dtype](-1.0):
                bthigh_action = Scalar[gpu_dtype](-1.0)
            if bshin_action > Scalar[gpu_dtype](1.0):
                bshin_action = Scalar[gpu_dtype](1.0)
            elif bshin_action < Scalar[gpu_dtype](-1.0):
                bshin_action = Scalar[gpu_dtype](-1.0)
            if bfoot_action > Scalar[gpu_dtype](1.0):
                bfoot_action = Scalar[gpu_dtype](1.0)
            elif bfoot_action < Scalar[gpu_dtype](-1.0):
                bfoot_action = Scalar[gpu_dtype](-1.0)
            if fthigh_action > Scalar[gpu_dtype](1.0):
                fthigh_action = Scalar[gpu_dtype](1.0)
            elif fthigh_action < Scalar[gpu_dtype](-1.0):
                fthigh_action = Scalar[gpu_dtype](-1.0)
            if fshin_action > Scalar[gpu_dtype](1.0):
                fshin_action = Scalar[gpu_dtype](1.0)
            elif fshin_action < Scalar[gpu_dtype](-1.0):
                fshin_action = Scalar[gpu_dtype](-1.0)
            if ffoot_action > Scalar[gpu_dtype](1.0):
                ffoot_action = Scalar[gpu_dtype](1.0)
            elif ffoot_action < Scalar[gpu_dtype](-1.0):
                ffoot_action = Scalar[gpu_dtype](-1.0)

            # Read curriculum parameters from model buffer
            var max_pitch = model[0, CURRICULUM_OFF + CURRICULUM_IDX_MAX_PITCH]

            # Compute velocity from position change (matching CPU)
            var x_position_after = states[env, QPOS_OFF + 0]  # rootx
            var prev_x = states[env, META_OFF + META_IDX_PREV_X]
            # DT=0.002 * FRAME_SKIP=5 = 0.01
            var effective_dt = C.DT * Scalar[gpu_dtype](C.FRAME_SKIP)
            var x_velocity = (x_position_after - prev_x) / effective_dt

            # Compute reward
            # Forward reward = forward_reward_weight * x_velocity
            var forward_reward = C.FORWARD_REWARD_WEIGHT * x_velocity

            # Control cost = ctrl_cost_weight * sum(action^2)
            var ctrl_cost = C.CTRL_COST_WEIGHT * (
                bthigh_action * bthigh_action
                + bshin_action * bshin_action
                + bfoot_action * bfoot_action
                + fthigh_action * fthigh_action
                + fshin_action * fshin_action
                + ffoot_action * ffoot_action
            )

            # Angle penalty (discourages flipping)
            var abs_y_angle = y_angle
            if abs_y_angle < Scalar[gpu_dtype](0.0):
                abs_y_angle = -abs_y_angle
            var angle_penalty = C.ANGLE_PENALTY_WEIGHT * abs_y_angle

            var reward = forward_reward - ctrl_cost - angle_penalty
            rewards[env] = reward

            # Health check using curriculum bounds
            var is_healthy = True
            if y_angle > max_pitch or y_angle < -max_pitch:
                is_healthy = False

            # Determine termination
            var terminated = False
            var truncated = step_count >= MAX_STEPS_VAL

            @parameter
            if Self.TERMINATE_ON_UNHEALTHY:
                terminated = not is_healthy

            # Set done flag (terminated OR truncated)
            if terminated or truncated:
                dones[env] = Scalar[gpu_dtype](1.0)
            else:
                dones[env] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[extract_kernel, extract_kernel](
            states,
            model,
            actions,
            rewards,
            dones,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    fn _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int = 0,
    ):
        """Reset a single environment on GPU with random noise.

        Adds random perturbations to initial qpos and qvel.
        """
        comptime C = HalfCheetahGCConstants[gpu_dtype]
        comptime QPOS_OFF = qpos_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()
        comptime QVEL_OFF = qvel_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()
        comptime QACC_OFF = qacc_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()
        comptime QFRC_OFF = qfrc_offset[HalfCheetahGC.NQ, HalfCheetahGC.NV]()

        # Reset noise scale
        comptime RESET_NOISE_SCALE: Scalar[gpu_dtype] = 0.1

        # Create RNG with unique seed per environment
        var rng = PhiloxRandom(seed=seed * 2654435761 + env * 12345, offset=0)

        # Generate random noise for qpos and qvel
        var rand_qpos1 = rng.step_uniform()  # 4 values
        var rand_qpos2 = rng.step_uniform()  # 4 values
        var rand_qpos3 = rng.step_uniform()  # 2 more values
        var rand_qvel1 = rng.step_uniform()  # 4 values
        var rand_qvel2 = rng.step_uniform()  # 4 values
        var rand_qvel3 = rng.step_uniform()  # 2 more values

        # Helper to convert uniform [0,1) to [-scale, scale)
        @always_inline
        fn to_noise(val: Scalar[DType.float32]) -> Scalar[gpu_dtype]:
            return Scalar[gpu_dtype](val * 2.0 - 1.0) * RESET_NOISE_SCALE

        # Reset qpos with noise (10 joints)
        states[env, QPOS_OFF + 0] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos1[0]
        )  # rootx
        states[env, QPOS_OFF + 1] = C.INITIAL_Z + to_noise(
            rand_qpos1[1]
        )  # rootz
        states[env, QPOS_OFF + 2] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos1[2]
        )  # rooty
        states[env, QPOS_OFF + 3] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos1[3]
        )  # bthigh
        states[env, QPOS_OFF + 4] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos2[0]
        )  # bshin
        states[env, QPOS_OFF + 5] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos2[1]
        )  # bfoot
        states[env, QPOS_OFF + 6] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos2[2]
        )  # fthigh
        states[env, QPOS_OFF + 7] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos2[3]
        )  # fshin
        states[env, QPOS_OFF + 8] = Scalar[gpu_dtype](0.0) + to_noise(
            rand_qpos3[0]
        )  # ffoot
        states[env, QPOS_OFF + 9] = Scalar[gpu_dtype](
            0.0
        )  # head (fixed, no noise)

        # Reset qvel with noise (10 joints)
        states[env, QVEL_OFF + 0] = to_noise(rand_qvel1[0])  # rootx vel
        states[env, QVEL_OFF + 1] = to_noise(rand_qvel1[1])  # rootz vel
        states[env, QVEL_OFF + 2] = to_noise(rand_qvel1[2])  # rooty vel
        states[env, QVEL_OFF + 3] = to_noise(rand_qvel1[3])  # bthigh vel
        states[env, QVEL_OFF + 4] = to_noise(rand_qvel2[0])  # bshin vel
        states[env, QVEL_OFF + 5] = to_noise(rand_qvel2[1])  # bfoot vel
        states[env, QVEL_OFF + 6] = to_noise(rand_qvel2[2])  # fthigh vel
        states[env, QVEL_OFF + 7] = to_noise(rand_qvel2[3])  # fshin vel
        states[env, QVEL_OFF + 8] = to_noise(rand_qvel3[0])  # ffoot vel
        states[env, QVEL_OFF + 9] = Scalar[gpu_dtype](0.0)  # head vel (fixed)

        # Reset qacc, qfrc to zero
        for i in range(HalfCheetahGC.NV):
            states[env, QACC_OFF + i] = Scalar[gpu_dtype](0.0)
            states[env, QFRC_OFF + i] = Scalar[gpu_dtype](0.0)

        # Reset step counter to 0
        comptime META_OFF = metadata_offset[
            HalfCheetahGC.NQ,
            HalfCheetahGC.NV,
            HalfCheetahGC.NUM_BODIES,
            HalfCheetahGC.MAX_CONTACTS,
        ]()
        states[env, META_OFF + META_IDX_STEP_COUNT] = Scalar[gpu_dtype](0.0)
        # Initialize prev_x to current rootx position
        states[env, META_OFF + META_IDX_PREV_X] = states[env, QPOS_OFF + 0]
