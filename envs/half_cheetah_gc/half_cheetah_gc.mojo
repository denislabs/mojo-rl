"""HalfCheetahGC Environment - MuJoCo-style Half Cheetah using Generalized Coordinates engine.

This implementation uses the physics3d_v2 Generalized Coordinates (GC) engine:
- ModelGC/DataGC for joint-space physics (MuJoCo-style)
- SemiImplicitEulerIntegrator for symplectic integration
- Joint-space state: qpos (positions), qvel (velocities)
- Forward kinematics computes body positions (xpos, xquat)

The Half Cheetah is a 2D planar robot (movement in XZ plane, rotation around Y axis)
consisting of a torso with two leg chains (front and back), totaling:
- 7 bodies: torso, bthigh, bshin, bfoot, fthigh, fshin, ffoot
- 9 joints: 3 root DOFs (unactuated) + 6 leg joints (actuated)
- 17D observation: 8 qpos (excluding rootx) + 9 qvel
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
from physics3d_v2.types import ModelGC, DataGC
from physics3d_v2.integrator import SemiImplicitEulerIntegrator
from physics3d_v2.kinematics.forward_kinematics import forward_kinematics
from physics3d_v2.joint_types import JNT_HINGE, JNT_SLIDE
from physics3d_v2.gpu.constants import (
    TPB,
    gc_state_size,
    gc_qpos_offset,
    gc_qvel_offset,
    gc_qacc_offset,
    gc_qfrc_offset,
    gc_xpos_offset,
    gc_metadata_offset,
    gc_model_size,
    GC_GEOM_CAPSULE,
    GC_META_IDX_NUM_CONTACTS,
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
    # Body geometry
    CAPSULE_RADIUS,
    TORSO_HALF_LENGTH,
    BTHIGH_HALF_LENGTH,
    BSHIN_HALF_LENGTH,
    BFOOT_HALF_LENGTH,
    FTHIGH_HALF_LENGTH,
    FSHIN_HALF_LENGTH,
    FFOOT_HALF_LENGTH,
    # Body masses
    TORSO_MASS,
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


struct HalfCheetahGC[DTYPE: DType = DType.float64](
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

    Joint Configuration (MuJoCo style):
        - Joint 0 (rootx): Slide joint, X-axis translation (body 0)
        - Joint 1 (rootz): Slide joint, Z-axis translation (body 0)
        - Joint 2 (rooty): Hinge joint, Y-axis rotation (body 0)
        - Joint 3 (bthigh): Hinge joint, Y-axis rotation (body 1)
        - Joint 4 (bshin): Hinge joint, Y-axis rotation (body 2)
        - Joint 5 (bfoot): Hinge joint, Y-axis rotation (body 3)
        - Joint 6 (fthigh): Hinge joint, Y-axis rotation (body 4)
        - Joint 7 (fshin): Hinge joint, Y-axis rotation (body 5)
        - Joint 8 (ffoot): Hinge joint, Y-axis rotation (body 6)

    State (qpos, qvel):
        - qpos[0]: rootx (x position)
        - qpos[1]: rootz (z position / height)
        - qpos[2]: rooty (pitch angle)
        - qpos[3-5]: back leg joint angles (bthigh, bshin, bfoot)
        - qpos[6-8]: front leg joint angles (fthigh, fshin, ffoot)
        - qvel[0:9]: corresponding velocities

    Observation Space (17 dimensions):
        Excludes qpos[0] (rootx) for translation invariance.
        [0:8]: qpos[1:9] (z, rooty, 6 joint angles)
        [8:17]: qvel[0:9] (all velocities including rootx velocity)

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
    comptime STATE_SIZE: Int = gc_state_size[
        Self.NQ, Self.NV, Self.NUM_BODIES, Self.MAX_CONTACTS
    ]()

    # Physics model and data
    var model: ModelGC[
        Self.DTYPE,
        Self.NQ,
        Self.NV,
        Self.NUM_BODIES,
        Self.NUM_JOINTS,
        Self.MAX_CONTACTS,
    ]
    var data: DataGC[
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
        self.model = ModelGC[
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

        # Initialize data (must be done before any method calls)
        self.data = DataGC[
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
        self.model.body_geom_type[BODY_TORSO] = GC_GEOM_CAPSULE
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
        self.model.body_geom_type[BODY_BTHIGH] = GC_GEOM_CAPSULE
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
        self.model.body_geom_type[BODY_BSHIN] = GC_GEOM_CAPSULE
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
        self.model.body_geom_type[BODY_BFOOT] = GC_GEOM_CAPSULE
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
        self.model.body_geom_type[BODY_FTHIGH] = GC_GEOM_CAPSULE
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
        self.model.body_geom_type[BODY_FSHIN] = GC_GEOM_CAPSULE
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
        self.model.body_geom_type[BODY_FFOOT] = GC_GEOM_CAPSULE
        self.model.body_half_length[BODY_FFOOT] = ffoot_half
        # Offset below fshin, with 90° Y rotation for horizontal foot
        self.model.set_body_local_frame(
            BODY_FFOOT,
            pos=(Scalar[Self.DTYPE](0.0), Scalar[Self.DTYPE](0.0), -fshin_half),
            quat=(quat_90y_x, quat_90y_y, quat_90y_z, quat_90y_w),
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
    ) -> Float64:
        """Compute reward for current state.

        Reward = forward_reward - ctrl_cost
        - forward_reward = forward_reward_weight * x_velocity
        - ctrl_cost = ctrl_cost_weight * sum(action^2)
        """
        # Forward velocity reward
        var forward_reward = FORWARD_REWARD_WEIGHT * x_velocity

        # Control cost (penalize large actions)
        var ctrl_cost = CTRL_COST_WEIGHT * action.squared_sum()

        return forward_reward - ctrl_cost

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

        Note: Half Cheetah never terminates early - only truncates at max_steps.
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
            SemiImplicitEulerIntegrator.step(self.model, self.data)
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
        var reward = self._compute_reward(x_velocity, clamped_action)

        # Half Cheetah never terminates, only truncates
        var truncated = self.current_step >= self.max_steps

        return (self.cached_state, Scalar[Self.dtype](reward), truncated)

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

    fn get_qpos(self) -> InlineArray[Scalar[Self.DTYPE], 9]:
        """Get full qpos array."""
        var qpos = InlineArray[Scalar[Self.DTYPE], 9](uninitialized=True)
        for i in range(9):
            qpos[i] = self.data.qpos[i]
        return qpos^

    fn get_qvel(self) -> InlineArray[Scalar[Self.DTYPE], 9]:
        """Get full qvel array."""
        var qvel = InlineArray[Scalar[Self.DTYPE], 9](uninitialized=True)
        for i in range(9):
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
        """Check if episode is finished (truncated only, never terminates)."""
        return self.current_step >= self.max_steps

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
            show_shadows=True,
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
    # Note: GPU support can be added following the HopperGC pattern
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
        ACTION_DIM_VAL: Int,
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
        """Batched GPU step function - placeholder for future implementation."""
        # TODO: Implement GPU physics step following HopperGC pattern
        pass

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments on GPU - placeholder for future implementation.
        """
        # TODO: Implement GPU reset following HopperGC pattern
        pass

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
        """Reset only done environments on GPU - placeholder for future implementation.
        """
        # TODO: Implement GPU selective reset following HopperGC pattern
        pass
