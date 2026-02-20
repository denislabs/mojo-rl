"""Physics3D types - Model/Data separation following MuJoCo.

Model contains static simulation configuration (kinematic tree, masses, etc.).
Data contains mutable simulation state (qpos, qvel, computed xpos/xquat).

State is joint positions (qpos) and velocities (qvel).
Body positions (xpos, xquat) are COMPUTED from qpos via forward kinematics.
Joints ADD DOFs. Dynamics computed in joint space (mass matrix, Coriolis, gravity).

Example usage:
    from physics3d.types import Model, Data

    # Create a single pendulum (1 body, 1 hinge joint)
    # NQ=1 (1 angle), NV=1 (1 angular velocity)
    var model = Model[DType.float64, 1, 1, 1, 1, 5]()
    model.set_body(0, mass=1.0, inertia=(0.1, 0.1, 0.1))
    model.set_body_parent(1, 0)  # Parent is worldbody

    var data = Data[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = 0.5  # Initial angle (radians)
    data.qvel[0] = 0.0  # Initial angular velocity
"""


from .joint_types import JointDef, JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from .joint_types import get_joint_qpos_size, get_joint_qvel_size

# Actuator dynamics type constants
comptime DYN_NONE: Int = 0
comptime DYN_INTEGRATOR: Int = 1
comptime DYN_FILTER: Int = 2
comptime DYN_FILTEREXACT: Int = 3

# Actuator gain type constants
comptime GAIN_FIXED: Int = 0
comptime GAIN_AFFINE: Int = 1

# Actuator bias type constants
comptime BIAS_NONE: Int = 0
comptime BIAS_AFFINE: Int = 1


# Helper to compute max(1, n) at compile time for array sizing
@always_inline
fn _max_one[n: Int]() -> Int:
    if n > 0:
        return n
    return 1


# Equality constraint type constants
comptime EQ_CONNECT: Int = 0  # Point-to-point ball joint (3 position rows)
comptime EQ_WELD: Int = 1  # Rigid attachment (3 position + 3 orientation rows)
comptime EQ_TENDON: Int = 2  # Fixed tendon (1 bilateral row)

# Maximum joints per tendon
comptime MAX_TENDON_JOINTS: Int = 4


@fieldwise_init
struct EqualityConstraintDef[DTYPE: DType](
    Copyable, ImplicitlyCopyable, Movable
):
    """Definition of an equality constraint (connect or weld).

    Connect: 3 rows enforcing world_anchor_a == world_anchor_b.
    Weld: 6 rows enforcing position + orientation match.
    """

    var eq_type: Int  # EQ_CONNECT or EQ_WELD
    var body_a: Int  # First body index
    var body_b: Int  # Second body index (0 for worldbody)
    var anchor_a_x: Scalar[Self.DTYPE]  # Anchor point in body_a frame
    var anchor_a_y: Scalar[Self.DTYPE]
    var anchor_a_z: Scalar[Self.DTYPE]
    var anchor_b_x: Scalar[
        Self.DTYPE
    ]  # Anchor point in body_b frame (or world)
    var anchor_b_y: Scalar[Self.DTYPE]
    var anchor_b_z: Scalar[Self.DTYPE]
    var relpose_x: Scalar[Self.DTYPE]  # Relative orientation quat (weld only)
    var relpose_y: Scalar[Self.DTYPE]
    var relpose_z: Scalar[Self.DTYPE]
    var relpose_w: Scalar[Self.DTYPE]
    var solref_0: Scalar[Self.DTYPE]  # timeconst
    var solref_1: Scalar[Self.DTYPE]  # dampratio
    var solimp_0: Scalar[Self.DTYPE]  # dmin
    var solimp_1: Scalar[Self.DTYPE]  # dmax
    var solimp_2: Scalar[Self.DTYPE]  # width

    @staticmethod
    fn empty() -> Self:
        """Create empty equality constraint."""
        return Self(
            eq_type=EQ_CONNECT,
            body_a=0,
            body_b=0,
            anchor_a_x=Scalar[Self.DTYPE](0),
            anchor_a_y=Scalar[Self.DTYPE](0),
            anchor_a_z=Scalar[Self.DTYPE](0),
            anchor_b_x=Scalar[Self.DTYPE](0),
            anchor_b_y=Scalar[Self.DTYPE](0),
            anchor_b_z=Scalar[Self.DTYPE](0),
            relpose_x=Scalar[Self.DTYPE](0),
            relpose_y=Scalar[Self.DTYPE](0),
            relpose_z=Scalar[Self.DTYPE](0),
            relpose_w=Scalar[Self.DTYPE](1),
            solref_0=Scalar[Self.DTYPE](0.02),
            solref_1=Scalar[Self.DTYPE](1.0),
            solimp_0=Scalar[Self.DTYPE](0.9),
            solimp_1=Scalar[Self.DTYPE](0.95),
            solimp_2=Scalar[Self.DTYPE](0.001),
        )


# =============================================================================
# TendonDef - Fixed tendon definition
# =============================================================================


@fieldwise_init
struct TendonDef[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """Definition of a fixed tendon (linear combination of joint positions).

    A fixed tendon computes: ten_length = Σ(coef_i * qpos[joint_qposadr_i])
    An equality constraint enforces: ten_length - length_ref = 0
    This produces 1 bilateral constraint row with trivial Jacobian: J[dof_adr_i] = coef_i.

    Uses flat fields (joint_idx_0..3, coef_0..3) instead of InlineArray
    to avoid Mojo copyability issues and simplify GPU buffer layout.
    """

    var num_joints: Int  # Number of joints in this tendon (1..4)
    var joint_idx_0: Int  # Joint indices (unused slots = -1)
    var joint_idx_1: Int
    var joint_idx_2: Int
    var joint_idx_3: Int
    var coef_0: Scalar[Self.DTYPE]  # Coefficients per joint
    var coef_1: Scalar[Self.DTYPE]
    var coef_2: Scalar[Self.DTYPE]
    var coef_3: Scalar[Self.DTYPE]
    var length_ref: Scalar[Self.DTYPE]  # Reference length (from initial qpos)
    var solref_0: Scalar[Self.DTYPE]  # timeconst
    var solref_1: Scalar[Self.DTYPE]  # dampratio
    var solimp_0: Scalar[Self.DTYPE]  # dmin
    var solimp_1: Scalar[Self.DTYPE]  # dmax
    var solimp_2: Scalar[Self.DTYPE]  # width

    @staticmethod
    fn empty() -> Self:
        """Create empty tendon definition."""
        return Self(
            num_joints=0,
            joint_idx_0=-1,
            joint_idx_1=-1,
            joint_idx_2=-1,
            joint_idx_3=-1,
            coef_0=Scalar[Self.DTYPE](0),
            coef_1=Scalar[Self.DTYPE](0),
            coef_2=Scalar[Self.DTYPE](0),
            coef_3=Scalar[Self.DTYPE](0),
            length_ref=Scalar[Self.DTYPE](0),
            solref_0=Scalar[Self.DTYPE](0.02),
            solref_1=Scalar[Self.DTYPE](1.0),
            solimp_0=Scalar[Self.DTYPE](0.9),
            solimp_1=Scalar[Self.DTYPE](0.95),
            solimp_2=Scalar[Self.DTYPE](0.001),
        )


# =============================================================================
# ActuatorDef - Runtime actuator definition
# =============================================================================


@fieldwise_init
struct ActuatorDef[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """Runtime representation of an actuator (populated from ActuatorSpec).

    Stores all parameters needed to compute actuator forces at runtime.
    Used by Actuators container on CPU; GPU uses flat buffer layout.
    """

    var joint_idx: Int  # Which joint this actuates
    var dof_adr: Int  # DOF address (computed from joint's dof_adr)
    var qpos_adr: Int  # Qpos address (computed from joint's qpos_adr)
    var gear: Scalar[Self.DTYPE]
    var dyntype: Int
    var dynprm_0: Scalar[Self.DTYPE]  # Time constant for filter
    var gaintype: Int
    var gainprm_0: Scalar[Self.DTYPE]
    var gainprm_1: Scalar[Self.DTYPE]
    var gainprm_2: Scalar[Self.DTYPE]
    var biastype: Int
    var biasprm_0: Scalar[Self.DTYPE]
    var biasprm_1: Scalar[Self.DTYPE]
    var biasprm_2: Scalar[Self.DTYPE]
    var ctrl_min: Scalar[Self.DTYPE]
    var ctrl_max: Scalar[Self.DTYPE]
    var force_min: Scalar[Self.DTYPE]
    var force_max: Scalar[Self.DTYPE]
    var has_activation: Bool

    @staticmethod
    fn empty() -> Self:
        """Create empty actuator definition."""
        return Self(
            joint_idx=-1,
            dof_adr=-1,
            qpos_adr=-1,
            gear=Scalar[Self.DTYPE](1),
            dyntype=DYN_NONE,
            dynprm_0=Scalar[Self.DTYPE](1),
            gaintype=GAIN_FIXED,
            gainprm_0=Scalar[Self.DTYPE](1),
            gainprm_1=Scalar[Self.DTYPE](0),
            gainprm_2=Scalar[Self.DTYPE](0),
            biastype=BIAS_NONE,
            biasprm_0=Scalar[Self.DTYPE](0),
            biasprm_1=Scalar[Self.DTYPE](0),
            biasprm_2=Scalar[Self.DTYPE](0),
            ctrl_min=Scalar[Self.DTYPE](-1),
            ctrl_max=Scalar[Self.DTYPE](1),
            force_min=Scalar[Self.DTYPE](-1e10),
            force_max=Scalar[Self.DTYPE](1e10),
            has_activation=False,
        )


# =============================================================================
# ContactInfo - Contact information for GC engine
# =============================================================================


@fieldwise_init
struct ContactInfo[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Contact information for generalized coordinates system.

    Similar to ContactInfo but designed for GC engine's needs.
    """

    var body_a: Int  # Index of first body
    var body_b: Int  # Index of second body (0 for ground/worldbody)
    var pos_x: Scalar[Self.DTYPE]  # Contact point (world)
    var pos_y: Scalar[Self.DTYPE]
    var pos_z: Scalar[Self.DTYPE]
    var normal_x: Scalar[Self.DTYPE]  # Normal (from A to B)
    var normal_y: Scalar[Self.DTYPE]
    var normal_z: Scalar[Self.DTYPE]
    var dist: Scalar[Self.DTYPE]  # Signed distance (negative = penetration)
    var force_n: Scalar[Self.DTYPE]  # Normal constraint force
    var force_t1: Scalar[Self.DTYPE]  # Tangent constraint force 1
    var force_t2: Scalar[Self.DTYPE]  # Tangent constraint force 2
    var friction: Scalar[Self.DTYPE]  # Per-contact slide friction coefficient
    var friction_spin: Scalar[Self.DTYPE]  # Torsional friction coefficient
    var friction_roll: Scalar[Self.DTYPE]  # Rolling friction coefficient
    var condim: Int  # Contact dimensionality (1, 3, 4, or 6)
    var force_torsion: Scalar[
        Self.DTYPE
    ]  # Torsional friction force (warm-start)
    var force_roll1: Scalar[Self.DTYPE]  # Rolling friction force 1 (warm-start)
    var force_roll2: Scalar[Self.DTYPE]  # Rolling friction force 2 (warm-start)
    var frame_t1_x: Scalar[
        Self.DTYPE
    ]  # T1 hint for tangent frame (capsule axis)
    var frame_t1_y: Scalar[Self.DTYPE]
    var frame_t1_z: Scalar[Self.DTYPE]

    @staticmethod
    fn empty() -> Self:
        """Create empty contact."""
        return Self(
            body_a=0,
            body_b=0,
            pos_x=Scalar[Self.DTYPE](0),
            pos_y=Scalar[Self.DTYPE](0),
            pos_z=Scalar[Self.DTYPE](0),
            normal_x=Scalar[Self.DTYPE](0),
            normal_y=Scalar[Self.DTYPE](0),
            normal_z=Scalar[Self.DTYPE](1),
            dist=Scalar[Self.DTYPE](0),
            force_n=Scalar[Self.DTYPE](0),
            force_t1=Scalar[Self.DTYPE](0),
            force_t2=Scalar[Self.DTYPE](0),
            friction=Scalar[Self.DTYPE](0),
            friction_spin=Scalar[Self.DTYPE](0),
            friction_roll=Scalar[Self.DTYPE](0),
            condim=3,
            force_torsion=Scalar[Self.DTYPE](0),
            force_roll1=Scalar[Self.DTYPE](0),
            force_roll2=Scalar[Self.DTYPE](0),
            frame_t1_x=Scalar[Self.DTYPE](0),
            frame_t1_y=Scalar[Self.DTYPE](0),
            frame_t1_z=Scalar[Self.DTYPE](0),
        )


# =============================================================================
# Model - Static Configuration for GC Engine
# =============================================================================


struct ConeType:
    comptime PYRAMIDAL: Int = 0
    comptime ELLIPTIC: Int = 1


struct Model[
    DTYPE: DType,
    NQ: Int,  # Total qpos size (sum of all joint qpos sizes)
    NV: Int,  # Total qvel size (sum of all joint qvel sizes)
    NBODY: Int,  # Number of bodies
    NJOINT: Int,  # Number of joints
    MAX_CONTACTS: Int,  # Maximum number of contacts
    NGEOM: Int = 0,  # Number of geoms (0 = legacy mode, uses body geometry)
    MAX_EQUALITY: Int = 0,  # Maximum number of equality constraints
    CONE_TYPE: Int = ConeType.ELLIPTIC,  # Cone type (0=pyramidal, 1=elliptic)
    MAX_TENDON: Int = 0,  # Maximum number of fixed tendons
]:
    """Static configuration for MuJoCo-style generalized coordinates simulation.

    Parameters:
        DTYPE: Data type for scalars (float32 or float64).
        NQ: Total qpos dimension (sum of all joints' qpos sizes).
        NV: Total qvel dimension (sum of all joints' qvel sizes).
        NBODY: Number of rigid bodies.
        NJOINT: Number of joints.
        MAX_CONTACTS: Maximum number of simultaneous contacts.
        NGEOM: Number of geoms (0 = legacy mode, uses body geometry).
        MAX_EQUALITY: Maximum number of equality constraints (0 = none).
        CONE_TYPE: Cone type (0=pyramidal, 1=elliptic).
        MAX_TENDON: Maximum number of fixed tendons (0 = none).

    The kinematic tree is defined by body_parent array:
    - body_parent[0] = 0 (worldbody, self-referencing)
    - body_parent[i] = index of parent body (0 for worldbody parent)
    - Real bodies start at index 1, worldbody is always index 0
    - Bodies must be added in topological order (parent before child)
    """

    # Global physics parameters
    var gravity: SIMD[Self.DTYPE, 4]  # (gx, gy, gz, 0)
    var timestep: Scalar[Self.DTYPE]

    # MuJoCo solref/solimp impedance parameters (contact)
    var solref_contact: InlineArray[
        Scalar[Self.DTYPE], 2
    ]  # [timeconst, dampratio]
    var solimp_contact: InlineArray[
        Scalar[Self.DTYPE], 3
    ]  # [dmin, dmax, width]
    # MuJoCo solref/solimp impedance parameters (joint limits)
    var solref_limit: InlineArray[
        Scalar[Self.DTYPE], 2
    ]  # [timeconst, dampratio]
    var solimp_limit: InlineArray[Scalar[Self.DTYPE], 3]  # [dmin, dmax, width]

    # Per-body properties
    var body_mass: InlineArray[Scalar[Self.DTYPE], Self.NBODY]
    var body_name: InlineArray[String, Self.NBODY]
    var body_inv_mass: InlineArray[Scalar[Self.DTYPE], Self.NBODY]
    # Diagonal inertia tensor (Ixx, Iyy, Izz) per body
    var body_inertia: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]
    var body_inv_inertia: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]

    # Body local frame (position and orientation relative to parent)
    var body_pos: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]
    var body_quat: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 4]

    # CoM offset from body origin (body frame) and inertia frame orientation
    var body_ipos: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]
    var body_iquat: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 4]

    # Kinematic tree structure
    var body_parent: InlineArray[Int, Self.NBODY]  # 0 for worldbody

    # Body inverse weights for primal solver (MuJoCo-style diagApprox)
    # [2*i] = translation, [2*i+1] = rotation  (precomputed model constant)
    var body_invweight0: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 2]

    # DOF inverse weights: dof_invweight0[d] = M_inv[d,d] (diagonal of inverse mass matrix)
    # Used for joint limit diagApprox (MuJoCo uses this instead of body_invweight0 for limits)
    var dof_invweight0: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()]

    # Joint reference positions (MuJoCo qpos0): FK uses qpos - qpos0 for slide/hinge joints
    # For slide joints with ref="X" in XML, qpos0 = X. For hinge joints, typically 0.
    var qpos0: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NQ]()]

    # Unified geom arrays (NGEOM > 0)
    var geom_type: InlineArray[Int, _max_one[Self.NGEOM]()]
    var geom_body: InlineArray[
        Int, _max_one[Self.NGEOM]()
    ]  # 0 for worldbody/static
    var geom_pos: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM * 3]()]
    var geom_quat: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM * 4]()]
    var geom_radius: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM]()]
    var geom_half_length: InlineArray[
        Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
    ]
    var geom_half_x: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM]()]
    var geom_half_y: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM]()]
    var geom_half_z: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM]()]
    var geom_friction: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM]()]
    var geom_condim: InlineArray[Int, _max_one[Self.NGEOM]()]
    var geom_friction_spin: InlineArray[
        Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
    ]
    var geom_friction_roll: InlineArray[
        Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
    ]
    var geom_contype: InlineArray[Int, _max_one[Self.NGEOM]()]
    var geom_conaffinity: InlineArray[Int, _max_one[Self.NGEOM]()]
    var geom_rbound: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM]()]

    # Per-geom solref/solimp (MuJoCo-style per-geom impedance overrides)
    var geom_solref: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM * 2]()]
    var geom_solimp: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM * 3]()]

    # Per-geom contact margin (MuJoCo-style: contacts activate when dist < margin)
    var geom_margin: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM]()]

    # Per-geom mass (computed from density * volume or explicit mass)
    var geom_mass: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NGEOM]()]

    # Per-joint solref/solimp for limits (MuJoCo-style per-joint impedance overrides)
    var joint_solref_limit: InlineArray[
        Scalar[Self.DTYPE], _max_one[Self.NJOINT * 2]()
    ]
    var joint_solimp_limit: InlineArray[
        Scalar[Self.DTYPE], _max_one[Self.NJOINT * 3]()
    ]

    # Friction cone model
    var impratio: Scalar[Self.DTYPE]  # MuJoCo impratio (default 1.0)

    # Joint definitions
    var joints: InlineArray[JointDef[Self.DTYPE], _max_one[Self.NJOINT]()]
    var num_joints: Int

    # Equality constraints (connect/weld)
    var equality_constraints: InlineArray[
        EqualityConstraintDef[Self.DTYPE], _max_one[Self.MAX_EQUALITY]()
    ]
    var num_equality: Int

    # Fixed tendons
    var tendons: InlineArray[
        TendonDef[Self.DTYPE], _max_one[Self.MAX_TENDON]()
    ]
    var num_tendons: Int

    fn __init__(out self):
        """Initialize model with default values."""
        self.gravity = SIMD[Self.DTYPE, 4](0, 0, -9.81, 0)
        self.timestep = Scalar[Self.DTYPE](0.01)

        # MuJoCo geom defaults: solref=[0.02, 1.0], solimp=[0.0, 0.8, 0.01]
        # Note: solimp defaults are MuJoCo's built-in geom defaults, NOT the
        # solver-level defaults [0.9, 0.95, 0.001]. Contacts use per-geom solimp
        # combined via max(), and geom defaults are [0.0, 0.8, 0.01, 0.5, 2.0].
        self.solref_contact = InlineArray[Scalar[Self.DTYPE], 2](
            uninitialized=True
        )
        self.solref_contact[0] = Scalar[Self.DTYPE](0.02)
        self.solref_contact[1] = Scalar[Self.DTYPE](1.0)
        self.solimp_contact = InlineArray[Scalar[Self.DTYPE], 3](
            uninitialized=True
        )
        self.solimp_contact[0] = Scalar[Self.DTYPE](0.0)
        self.solimp_contact[1] = Scalar[Self.DTYPE](0.8)
        self.solimp_contact[2] = Scalar[Self.DTYPE](0.01)
        self.solref_limit = InlineArray[Scalar[Self.DTYPE], 2](
            uninitialized=True
        )
        self.solref_limit[0] = Scalar[Self.DTYPE](0.02)
        self.solref_limit[1] = Scalar[Self.DTYPE](1.0)
        self.solimp_limit = InlineArray[Scalar[Self.DTYPE], 3](
            uninitialized=True
        )
        self.solimp_limit[0] = Scalar[Self.DTYPE](0.0)
        self.solimp_limit[1] = Scalar[Self.DTYPE](0.8)
        self.solimp_limit[2] = Scalar[Self.DTYPE](0.01)

        # Initialize body arrays
        self.body_mass = InlineArray[Scalar[Self.DTYPE], Self.NBODY](
            uninitialized=True
        )
        self.body_name = InlineArray[String, Self.NBODY](uninitialized=True)
        self.body_inv_mass = InlineArray[Scalar[Self.DTYPE], Self.NBODY](
            uninitialized=True
        )
        self.body_inertia = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3](
            uninitialized=True
        )
        self.body_inv_inertia = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3](
            uninitialized=True
        )
        self.body_pos = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3](
            uninitialized=True
        )
        self.body_quat = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 4](
            uninitialized=True
        )
        self.body_ipos = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3](
            uninitialized=True
        )
        self.body_iquat = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 4](
            uninitialized=True
        )
        self.body_parent = InlineArray[Int, Self.NBODY](uninitialized=True)
        self.body_invweight0 = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 2](
            fill=Scalar[Self.DTYPE](0)
        )
        self.dof_invweight0 = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NV]()
        ](fill=Scalar[Self.DTYPE](0))
        self.qpos0 = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NQ]()
        ](fill=Scalar[Self.DTYPE](0))

        # Initialize geom arrays
        self.geom_type = InlineArray[Int, _max_one[Self.NGEOM]()](
            uninitialized=True
        )
        self.geom_body = InlineArray[Int, _max_one[Self.NGEOM]()](
            uninitialized=True
        )
        self.geom_pos = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM * 3]()
        ](uninitialized=True)
        self.geom_quat = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM * 4]()
        ](uninitialized=True)
        self.geom_radius = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_half_length = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_half_x = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_half_y = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_half_z = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_friction = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_condim = InlineArray[Int, _max_one[Self.NGEOM]()](
            uninitialized=True
        )
        self.geom_friction_spin = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_friction_roll = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_contype = InlineArray[Int, _max_one[Self.NGEOM]()](
            uninitialized=True
        )
        self.geom_conaffinity = InlineArray[Int, _max_one[Self.NGEOM]()](
            uninitialized=True
        )
        self.geom_rbound = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](uninitialized=True)
        self.geom_solref = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM * 2]()
        ](fill=Scalar[Self.DTYPE](0))
        self.geom_solimp = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM * 3]()
        ](fill=Scalar[Self.DTYPE](0))
        self.geom_margin = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](fill=Scalar[Self.DTYPE](0))
        self.geom_mass = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NGEOM]()
        ](fill=Scalar[Self.DTYPE](0))
        self.joint_solref_limit = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NJOINT * 2]()
        ](fill=Scalar[Self.DTYPE](0))
        self.joint_solimp_limit = InlineArray[
            Scalar[Self.DTYPE], _max_one[Self.NJOINT * 3]()
        ](fill=Scalar[Self.DTYPE](0))
        self.impratio = Scalar[Self.DTYPE](1.0)
        for i in range(_max_one[Self.NGEOM]()):
            self.geom_type[i] = 0
            self.geom_body[i] = 0
            self.geom_radius[i] = Scalar[Self.DTYPE](0)
            self.geom_half_length[i] = Scalar[Self.DTYPE](0)
            self.geom_half_x[i] = Scalar[Self.DTYPE](0)
            self.geom_half_y[i] = Scalar[Self.DTYPE](0)
            self.geom_half_z[i] = Scalar[Self.DTYPE](0)
            self.geom_friction[i] = Scalar[Self.DTYPE](0.5)
            self.geom_condim[i] = 3
            self.geom_friction_spin[i] = Scalar[Self.DTYPE](0.005)
            self.geom_friction_roll[i] = Scalar[Self.DTYPE](0.0001)
            self.geom_contype[i] = 1
            self.geom_conaffinity[i] = 1
            self.geom_rbound[i] = Scalar[Self.DTYPE](0)
        for i in range(_max_one[Self.NGEOM * 3]()):
            self.geom_pos[i] = Scalar[Self.DTYPE](0)
        for i in range(_max_one[Self.NGEOM * 4]()):
            self.geom_quat[i] = Scalar[Self.DTYPE](0)

        # Initialize with defaults
        for i in range(Self.NBODY):
            self.body_mass[i] = Scalar[Self.DTYPE](1.0)
            self.body_inv_mass[i] = Scalar[Self.DTYPE](1.0)
            self.body_parent[
                i
            ] = 0  # Default: all bodies have worldbody as parent
            # Default body position: origin in parent frame
            self.body_pos[i * 3 + 0] = Scalar[Self.DTYPE](0)
            self.body_pos[i * 3 + 1] = Scalar[Self.DTYPE](0)
            self.body_pos[i * 3 + 2] = Scalar[Self.DTYPE](0)

            # Default body orientation: identity quaternion [x, y, z, w]
            self.body_quat[i * 4 + 0] = Scalar[Self.DTYPE](0)
            self.body_quat[i * 4 + 1] = Scalar[Self.DTYPE](0)
            self.body_quat[i * 4 + 2] = Scalar[Self.DTYPE](0)
            self.body_quat[i * 4 + 3] = Scalar[Self.DTYPE](1)

            # Default body ipos: zero (CoM at body origin)
            self.body_ipos[i * 3 + 0] = Scalar[Self.DTYPE](0)
            self.body_ipos[i * 3 + 1] = Scalar[Self.DTYPE](0)
            self.body_ipos[i * 3 + 2] = Scalar[Self.DTYPE](0)

            # Default body iquat: identity (inertia aligned with body frame)
            self.body_iquat[i * 4 + 0] = Scalar[Self.DTYPE](0)
            self.body_iquat[i * 4 + 1] = Scalar[Self.DTYPE](0)
            self.body_iquat[i * 4 + 2] = Scalar[Self.DTYPE](0)
            self.body_iquat[i * 4 + 3] = Scalar[Self.DTYPE](1)

        # Initialize inertia
        for i in range(Self.NBODY * 3):
            self.body_inertia[i] = Scalar[Self.DTYPE](
                0.004
            )  # Default sphere inertia
            self.body_inv_inertia[i] = Scalar[Self.DTYPE](250.0)

        # Initialize worldbody at index 0 (MuJoCo convention)
        self.body_mass[0] = Scalar[Self.DTYPE](0)
        self.body_inv_mass[0] = Scalar[Self.DTYPE](0)
        self.body_parent[0] = 0  # Self-referencing (MuJoCo convention)
        for k in range(3):
            self.body_inertia[k] = Scalar[Self.DTYPE](0)
            self.body_inv_inertia[k] = Scalar[Self.DTYPE](0)

        # Initialize joints
        self.joints = InlineArray[
            JointDef[Self.DTYPE], _max_one[Self.NJOINT]()
        ](uninitialized=True)
        for i in range(_max_one[Self.NJOINT]()):
            self.joints[i] = JointDef[Self.DTYPE].empty()
        self.num_joints = 0

        # Initialize equality constraints
        self.equality_constraints = InlineArray[
            EqualityConstraintDef[Self.DTYPE], _max_one[Self.MAX_EQUALITY]()
        ](uninitialized=True)
        for i in range(_max_one[Self.MAX_EQUALITY]()):
            self.equality_constraints[i] = EqualityConstraintDef[
                Self.DTYPE
            ].empty()
        self.num_equality = 0

        # Initialize tendons
        self.tendons = InlineArray[
            TendonDef[Self.DTYPE], _max_one[Self.MAX_TENDON]()
        ](uninitialized=True)
        for i in range(_max_one[Self.MAX_TENDON]()):
            self.tendons[i] = TendonDef[Self.DTYPE].empty()
        self.num_tendons = 0

    fn set_body(
        mut self,
        body_id: Int,
        name: String,
        mass: Scalar[Self.DTYPE],
        inertia: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
    ):
        """Set body properties.

        Args:
            body_id: Body index.
            name: Body name.
            mass: Body mass.
            inertia: Diagonal inertia tensor (Ixx, Iyy, Izz).
        """
        self.body_mass[body_id] = mass
        self.body_name[body_id] = name
        self.body_inv_mass[body_id] = Scalar[Self.DTYPE](1.0) / mass

        self.body_inertia[body_id * 3 + 0] = inertia[0]
        self.body_inertia[body_id * 3 + 1] = inertia[1]
        self.body_inertia[body_id * 3 + 2] = inertia[2]
        self.body_inv_inertia[body_id * 3 + 0] = (
            Scalar[Self.DTYPE](1.0) / inertia[0]
        )
        self.body_inv_inertia[body_id * 3 + 1] = (
            Scalar[Self.DTYPE](1.0) / inertia[1]
        )
        self.body_inv_inertia[body_id * 3 + 2] = (
            Scalar[Self.DTYPE](1.0) / inertia[2]
        )

    fn get_body_name(self, body_id: Int) -> String:
        """Get body name."""
        if body_id >= Self.NBODY:
            return ""
        if body_id == 0:
            return "world"
        return self.body_name[body_id]

    fn set_body_parent(mut self, body_id: Int, parent_id: Int):
        """Set parent body for kinematic tree.

        Args:
            body_id: Child body index.
            parent_id: Parent body index (0 for worldbody).
        """
        self.body_parent[body_id] = parent_id

    fn set_body_local_frame(
        mut self,
        body_id: Int,
        pos: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        quat: Tuple[
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
        ] = (
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](1),
        ),
    ):
        """Set body's local frame (position and orientation in parent frame).

        Args:
            body_id: Body index.
            pos: Position in parent frame.
            quat: Orientation quaternion [x, y, z, w] in parent frame.
        """
        self.body_pos[body_id * 3 + 0] = pos[0]
        self.body_pos[body_id * 3 + 1] = pos[1]
        self.body_pos[body_id * 3 + 2] = pos[2]

        self.body_quat[body_id * 4 + 0] = quat[0]
        self.body_quat[body_id * 4 + 1] = quat[1]
        self.body_quat[body_id * 4 + 2] = quat[2]
        self.body_quat[body_id * 4 + 3] = quat[3]

    fn set_body_ipos_iquat(
        mut self,
        body_id: Int,
        ipos: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        iquat: Tuple[
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
        ] = (
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](1),
        ),
    ):
        """Set body's CoM offset and inertia frame orientation.

        Args:
            body_id: Body index.
            ipos: CoM offset from body origin in body frame.
            iquat: Inertia frame quaternion [x, y, z, w] in body frame.
        """
        self.body_ipos[body_id * 3 + 0] = ipos[0]
        self.body_ipos[body_id * 3 + 1] = ipos[1]
        self.body_ipos[body_id * 3 + 2] = ipos[2]

        self.body_iquat[body_id * 4 + 0] = iquat[0]
        self.body_iquat[body_id * 4 + 1] = iquat[1]
        self.body_iquat[body_id * 4 + 2] = iquat[2]
        self.body_iquat[body_id * 4 + 3] = iquat[3]

    fn add_hinge_joint(
        mut self,
        body_id: Int,
        pos: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        tau_limit: Scalar[Self.DTYPE] = 1000.0,
        range_min: Scalar[Self.DTYPE] = -1e10,
        range_max: Scalar[Self.DTYPE] = 1e10,
        armature: Scalar[Self.DTYPE] = 0.0,
        damping: Scalar[Self.DTYPE] = 0.0,
        stiffness: Scalar[Self.DTYPE] = 0.0,
        springref: Scalar[Self.DTYPE] = 0.0,
        frictionloss: Scalar[Self.DTYPE] = 0.0,
    ) -> Int:
        """Add a hinge joint to a body.

        Args:
            body_id: Body this joint controls.
            pos: Joint position in parent frame.
            axis: Rotation axis in parent frame.
            tau_limit: Maximum torque.
            range_min: Minimum angle in radians (default: unlimited).
            range_max: Maximum angle in radians (default: unlimited).
            armature: Armature (default: 0.0).
            damping: Damping (default: 0.0).
            stiffness: Stiffness (default: 0.0).
            springref: Spring reference (default: 0.0).
            frictionloss: Friction loss (default: 0.0).
        Returns:
            Joint index, or -1 if max joints exceeded.
        """
        if self.num_joints >= Self.NJOINT:
            return -1

        # Compute qpos/qvel addresses
        var qpos_adr = 0
        var dof_adr = 0
        for i in range(self.num_joints):
            qpos_adr += self.joints[i].qpos_size()
            dof_adr += self.joints[i].qvel_size()

        var joint_idx = self.num_joints
        self.joints[joint_idx] = JointDef[Self.DTYPE].create_hinge(
            body_id,
            qpos_adr,
            dof_adr,
            pos,
            axis,
            tau_limit,
            range_min,
            range_max,
            armature,
            damping,
            stiffness,
            springref,
            frictionloss,
        )
        self.num_joints += 1
        return joint_idx

    fn add_slide_joint(
        mut self,
        body_id: Int,
        pos: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        force_limit: Scalar[Self.DTYPE] = 1000.0,
        range_min: Scalar[Self.DTYPE] = -1e10,
        range_max: Scalar[Self.DTYPE] = 1e10,
        armature: Scalar[Self.DTYPE] = 0.0,
        damping: Scalar[Self.DTYPE] = 0.0,
        stiffness: Scalar[Self.DTYPE] = 0.0,
        springref: Scalar[Self.DTYPE] = 0.0,
        frictionloss: Scalar[Self.DTYPE] = 0.0,
    ) -> Int:
        """Add a slide joint to a body.

        Args:
            body_id: Body this joint controls.
            pos: Joint position in parent frame.
            axis: Slide axis in parent frame.
            force_limit: Maximum force.
            range_min: Minimum position in meters (default: unlimited).
            range_max: Maximum position in meters (default: unlimited).
            armature: Armature (default: 0.0).
            damping: Damping (default: 0.0).
            stiffness: Stiffness (default: 0.0).
            springref: Spring reference (default: 0.0).
            frictionloss: Friction loss (default: 0.0).

        Returns:
            Joint index, or -1 if max joints exceeded.
        """
        if self.num_joints >= Self.NJOINT:
            return -1

        # Compute qpos/qvel addresses
        var qpos_adr = 0
        var dof_adr = 0
        for i in range(self.num_joints):
            qpos_adr += self.joints[i].qpos_size()
            dof_adr += self.joints[i].qvel_size()

        var joint_idx = self.num_joints
        self.joints[joint_idx] = JointDef[Self.DTYPE].create_slide(
            body_id,
            qpos_adr,
            dof_adr,
            pos,
            axis,
            force_limit,
            range_min,
            range_max,
            armature,
            damping,
            stiffness,
            springref,
            frictionloss,
        )
        self.num_joints += 1
        return joint_idx

    fn add_connect_constraint(
        mut self,
        body_a: Int,
        body_b: Int,
        anchor_a: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        anchor_b: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        solref: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE]] = (
            Scalar[Self.DTYPE](0.02),
            Scalar[Self.DTYPE](1.0),
        ),
        solimp: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ] = (
            Scalar[Self.DTYPE](0.9),
            Scalar[Self.DTYPE](0.95),
            Scalar[Self.DTYPE](0.001),
        ),
    ) -> Int:
        """Add a connect (ball joint) equality constraint.

        Args:
            body_a: First body index.
            body_b: Second body index (0 for worldbody).
            anchor_a: Anchor point in body_a frame.
            anchor_b: Anchor point in body_b frame (or world frame if body_b=0).
            solref: Impedance parameters [timeconst, dampratio].
            solimp: Impedance parameters [dmin, dmax, width].

        Returns:
            Constraint index, or -1 if max constraints exceeded.
        """
        if self.num_equality >= Self.MAX_EQUALITY:
            return -1
        var idx = self.num_equality
        self.equality_constraints[idx] = EqualityConstraintDef[Self.DTYPE](
            eq_type=EQ_CONNECT,
            body_a=body_a,
            body_b=body_b,
            anchor_a_x=anchor_a[0],
            anchor_a_y=anchor_a[1],
            anchor_a_z=anchor_a[2],
            anchor_b_x=anchor_b[0],
            anchor_b_y=anchor_b[1],
            anchor_b_z=anchor_b[2],
            relpose_x=Scalar[Self.DTYPE](0),
            relpose_y=Scalar[Self.DTYPE](0),
            relpose_z=Scalar[Self.DTYPE](0),
            relpose_w=Scalar[Self.DTYPE](1),
            solref_0=solref[0],
            solref_1=solref[1],
            solimp_0=solimp[0],
            solimp_1=solimp[1],
            solimp_2=solimp[2],
        )
        self.num_equality += 1
        return idx

    fn add_weld_constraint(
        mut self,
        body_a: Int,
        body_b: Int,
        anchor_a: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        anchor_b: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        relpose: Tuple[
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
            Scalar[Self.DTYPE],
        ] = (
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](1),
        ),
        solref: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE]] = (
            Scalar[Self.DTYPE](0.02),
            Scalar[Self.DTYPE](1.0),
        ),
        solimp: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ] = (
            Scalar[Self.DTYPE](0.9),
            Scalar[Self.DTYPE](0.95),
            Scalar[Self.DTYPE](0.001),
        ),
    ) -> Int:
        """Add a weld (rigid attachment) equality constraint.

        Args:
            body_a: First body index.
            body_b: Second body index (0 for worldbody).
            anchor_a: Anchor point in body_a frame.
            anchor_b: Anchor point in body_b frame (or world frame if body_b=0).
            relpose: Relative orientation quaternion [x, y, z, w].
            solref: Impedance parameters [timeconst, dampratio].
            solimp: Impedance parameters [dmin, dmax, width].

        Returns:
            Constraint index, or -1 if max constraints exceeded.
        """
        if self.num_equality >= Self.MAX_EQUALITY:
            return -1
        var idx = self.num_equality
        self.equality_constraints[idx] = EqualityConstraintDef[Self.DTYPE](
            eq_type=EQ_WELD,
            body_a=body_a,
            body_b=body_b,
            anchor_a_x=anchor_a[0],
            anchor_a_y=anchor_a[1],
            anchor_a_z=anchor_a[2],
            anchor_b_x=anchor_b[0],
            anchor_b_y=anchor_b[1],
            anchor_b_z=anchor_b[2],
            relpose_x=relpose[0],
            relpose_y=relpose[1],
            relpose_z=relpose[2],
            relpose_w=relpose[3],
            solref_0=solref[0],
            solref_1=solref[1],
            solimp_0=solimp[0],
            solimp_1=solimp[1],
            solimp_2=solimp[2],
        )
        self.num_equality += 1
        return idx

    fn add_tendon(
        mut self,
        num_joints: Int,
        joint_indices: InlineArray[Int, 4],
        coefs: InlineArray[Scalar[Self.DTYPE], 4],
        length_ref: Scalar[Self.DTYPE] = 0.0,
        solref: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE]] = (
            Scalar[Self.DTYPE](0.02),
            Scalar[Self.DTYPE](1.0),
        ),
        solimp: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ] = (
            Scalar[Self.DTYPE](0.9),
            Scalar[Self.DTYPE](0.95),
            Scalar[Self.DTYPE](0.001),
        ),
    ) -> Int:
        """Add a fixed tendon (linear combination of joint positions).

        Args:
            num_joints: Number of joints in the tendon (1..4).
            joint_indices: Joint indices (unused slots should be -1).
            coefs: Coefficients per joint.
            length_ref: Reference length (tendon length at rest).
            solref: Impedance parameters [timeconst, dampratio].
            solimp: Impedance parameters [dmin, dmax, width].

        Returns:
            Tendon index, or -1 if max tendons exceeded.
        """
        if self.num_tendons >= Self.MAX_TENDON:
            return -1
        var idx = self.num_tendons
        self.tendons[idx] = TendonDef[Self.DTYPE](
            num_joints=num_joints,
            joint_idx_0=joint_indices[0],
            joint_idx_1=joint_indices[1],
            joint_idx_2=joint_indices[2],
            joint_idx_3=joint_indices[3],
            coef_0=coefs[0],
            coef_1=coefs[1],
            coef_2=coefs[2],
            coef_3=coefs[3],
            length_ref=length_ref,
            solref_0=solref[0],
            solref_1=solref[1],
            solimp_0=solimp[0],
            solimp_1=solimp[1],
            solimp_2=solimp[2],
        )
        self.num_tendons += 1
        return idx

    fn get_joint(self, joint_idx: Int) -> JointDef[Self.DTYPE]:
        """Get joint definition by index."""
        return self.joints[joint_idx]


# =============================================================================
# Data - Mutable State for GC Engine
# =============================================================================


struct Data[
    DTYPE: DType,
    NQ: Int,  # Total qpos size
    NV: Int,  # Total qvel size
    NBODY: Int,  # Number of bodies
    NJOINT: Int,  # Number of joints
    MAX_CONTACTS: Int,  # Maximum number of contacts
]:
    """Mutable simulation state for MuJoCo-style generalized coordinates.

    Parameters:
        DTYPE: Data type for scalars.
        NQ: Total qpos dimension.
        NV: Total qvel dimension.
        NBODY: Number of rigid bodies.
        NJOINT: Number of joints.
        MAX_CONTACTS: Maximum number of contacts.

    State representation:
    - qpos: Joint positions (angles, quaternions, displacements)
    - qvel: Joint velocities
    - qacc: Joint accelerations (computed during step)
    - qfrc: Applied joint forces/torques (user input)

    Computed from qpos via forward kinematics:
    - xpos: World positions of bodies
    - xquat: World orientations of bodies
    """

    # Primary state (joint space)
    var qpos: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NQ]()]
    var qvel: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()]
    var qacc: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()]
    var qfrc: InlineArray[
        Scalar[Self.DTYPE], _max_one[Self.NV]()
    ]  # Applied forces
    # MuJoCo-style Newton warm-start: solved qacc saved at end of each step.
    # Used to initialize the Newton solver in the next step (like qacc_warmstart).
    var qacc_warmstart: InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()]

    # Computed world-space state (via forward kinematics)
    var xpos: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]
    var xquat: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 4]
    var xipos: InlineArray[
        Scalar[Self.DTYPE], Self.NBODY * 3
    ]  # CoM world position

    # Computed world-space velocities (for collision response)
    var xvel: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]  # Linear
    var xangvel: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]  # Angular

    # Contacts — heap-allocated to avoid stack overflow for large MAX_CONTACTS
    var contacts: List[ContactInfo[Self.DTYPE]]
    var num_contacts: Int

    fn __init__(out self):
        """Initialize with zero state."""
        # Initialize qpos to zero (neutral position for all joints)
        self.qpos = InlineArray[Scalar[Self.DTYPE], _max_one[Self.NQ]()](
            uninitialized=True
        )
        for i in range(_max_one[Self.NQ]()):
            self.qpos[i] = Scalar[Self.DTYPE](0)

        # Initialize qvel to zero
        self.qvel = InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()](
            uninitialized=True
        )
        for i in range(_max_one[Self.NV]()):
            self.qvel[i] = Scalar[Self.DTYPE](0)

        # Initialize qacc to zero
        self.qacc = InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()](
            uninitialized=True
        )
        for i in range(_max_one[Self.NV]()):
            self.qacc[i] = Scalar[Self.DTYPE](0)

        # Initialize qacc_warmstart to zero
        self.qacc_warmstart = InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()](
            uninitialized=True
        )
        for i in range(_max_one[Self.NV]()):
            self.qacc_warmstart[i] = Scalar[Self.DTYPE](0)

        # Initialize qfrc to zero
        self.qfrc = InlineArray[Scalar[Self.DTYPE], _max_one[Self.NV]()](
            uninitialized=True
        )
        for i in range(_max_one[Self.NV]()):
            self.qfrc[i] = Scalar[Self.DTYPE](0)

        # Initialize xpos to zero
        self.xpos = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3](
            uninitialized=True
        )
        for i in range(Self.NBODY * 3):
            self.xpos[i] = Scalar[Self.DTYPE](0)

        # Initialize xquat to identity
        self.xquat = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 4](
            uninitialized=True
        )
        for i in range(Self.NBODY):
            self.xquat[i * 4 + 0] = Scalar[Self.DTYPE](0)
            self.xquat[i * 4 + 1] = Scalar[Self.DTYPE](0)
            self.xquat[i * 4 + 2] = Scalar[Self.DTYPE](0)
            self.xquat[i * 4 + 3] = Scalar[Self.DTYPE](1)

        # Initialize xipos to zero
        self.xipos = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3](
            uninitialized=True
        )
        for i in range(Self.NBODY * 3):
            self.xipos[i] = Scalar[Self.DTYPE](0)

        # Initialize xvel to zero
        self.xvel = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3](
            uninitialized=True
        )
        for i in range(Self.NBODY * 3):
            self.xvel[i] = Scalar[Self.DTYPE](0)

        # Initialize xangvel to zero
        self.xangvel = InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3](
            uninitialized=True
        )
        for i in range(Self.NBODY * 3):
            self.xangvel[i] = Scalar[Self.DTYPE](0)

        # Initialize contacts (heap-allocated, pre-filled to MAX_CONTACTS)
        self.contacts = List[ContactInfo[Self.DTYPE]](
            capacity=_max_one[Self.MAX_CONTACTS]()
        )
        for _ in range(_max_one[Self.MAX_CONTACTS]()):
            self.contacts.append(ContactInfo[Self.DTYPE].empty())
        self.num_contacts = 0

    fn get_body_position(
        self, body_id: Int
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get world position of a body."""
        return (
            self.xpos[body_id * 3 + 0],
            self.xpos[body_id * 3 + 1],
            self.xpos[body_id * 3 + 2],
        )

    fn get_body_quaternion(
        self, body_id: Int
    ) -> Tuple[
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
        Scalar[Self.DTYPE],
    ]:
        """Get world orientation quaternion [x, y, z, w] of a body."""
        return (
            self.xquat[body_id * 4 + 0],
            self.xquat[body_id * 4 + 1],
            self.xquat[body_id * 4 + 2],
            self.xquat[body_id * 4 + 3],
        )

    fn get_body_z(self, body_id: Int) -> Scalar[Self.DTYPE]:
        """Get z position of a body."""
        return self.xpos[body_id * 3 + 2]

    fn set_qpos(mut self, idx: Int, value: Scalar[Self.DTYPE]):
        """Set a qpos element."""
        self.qpos[idx] = value

    fn set_qvel(mut self, idx: Int, value: Scalar[Self.DTYPE]):
        """Set a qvel element."""
        self.qvel[idx] = value

    fn set_qfrc(mut self, idx: Int, value: Scalar[Self.DTYPE]):
        """Set an applied force/torque."""
        self.qfrc[idx] = value

    fn clear_forces(mut self):
        """Clear all applied forces."""
        for i in range(_max_one[Self.NV]()):
            self.qfrc[i] = Scalar[Self.DTYPE](0)


@always_inline
fn compute_capsule_inertia[
    DTYPE: DType
](
    mass: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    var r2 = radius * radius
    var L = Scalar[DTYPE](2.0) * half_length + Scalar[DTYPE](2.0) * radius
    var L2 = L * L
    var I_trans = mass * (Scalar[DTYPE](3.0) * r2 + L2) / Scalar[DTYPE](12.0)
    var I_axial = Scalar[DTYPE](0.5) * mass * r2
    return (I_trans, I_trans, I_axial)
