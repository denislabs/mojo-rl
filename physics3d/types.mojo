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
    model.set_body_parent(0, -1)  # Parent is world

    var data = Data[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = 0.5  # Initial angle (radians)
    data.qvel[0] = 0.0  # Initial angular velocity
"""

from .gpu.constants import GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX
from .joint_types import JointDef, JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from .joint_types import get_joint_qpos_size, get_joint_qvel_size


# Helper to compute max(1, n) at compile time for array sizing
fn _max_one[n: Int]() -> Int:
    if n > 0:
        return n
    return 1




# =============================================================================
# ContactInfo - Contact information for GC engine
# =============================================================================


@fieldwise_init
struct ContactInfo[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Contact information for generalized coordinates system.

    Similar to ContactInfo but designed for GC engine's needs.
    """

    var body_a: Int  # Index of first body
    var body_b: Int  # Index of second body (-1 for ground)
    var pos_x: Scalar[Self.DTYPE]  # Contact point (world)
    var pos_y: Scalar[Self.DTYPE]
    var pos_z: Scalar[Self.DTYPE]
    var normal_x: Scalar[Self.DTYPE]  # Normal (from A to B)
    var normal_y: Scalar[Self.DTYPE]
    var normal_z: Scalar[Self.DTYPE]
    var dist: Scalar[Self.DTYPE]  # Signed distance (negative = penetration)
    var impulse_n: Scalar[Self.DTYPE]  # Normal impulse
    var impulse_t1: Scalar[Self.DTYPE]  # Tangent impulse 1
    var impulse_t2: Scalar[Self.DTYPE]  # Tangent impulse 2

    @staticmethod
    fn empty() -> Self:
        """Create empty contact."""
        return Self(
            body_a=-1,
            body_b=-1,
            pos_x=Scalar[Self.DTYPE](0),
            pos_y=Scalar[Self.DTYPE](0),
            pos_z=Scalar[Self.DTYPE](0),
            normal_x=Scalar[Self.DTYPE](0),
            normal_y=Scalar[Self.DTYPE](0),
            normal_z=Scalar[Self.DTYPE](1),
            dist=Scalar[Self.DTYPE](0),
            impulse_n=Scalar[Self.DTYPE](0),
            impulse_t1=Scalar[Self.DTYPE](0),
            impulse_t2=Scalar[Self.DTYPE](0),
        )


# =============================================================================
# Model - Static Configuration for GC Engine
# =============================================================================


struct Model[
    DTYPE: DType,
    NQ: Int,  # Total qpos size (sum of all joint qpos sizes)
    NV: Int,  # Total qvel size (sum of all joint qvel sizes)
    NBODY: Int,  # Number of bodies
    NJOINT: Int,  # Number of joints
    MAX_CONTACTS: Int,  # Maximum number of contacts
]:
    """Static configuration for MuJoCo-style generalized coordinates simulation.

    Parameters:
        DTYPE: Data type for scalars (float32 or float64).
        NQ: Total qpos dimension (sum of all joints' qpos sizes).
        NV: Total qvel dimension (sum of all joints' qvel sizes).
        NBODY: Number of rigid bodies.
        NJOINT: Number of joints.
        MAX_CONTACTS: Maximum number of simultaneous contacts.

    The kinematic tree is defined by body_parent array:
    - body_parent[i] = index of parent body (-1 for world)
    - Bodies must be added in topological order (parent before child)
    """

    # Global physics parameters
    var gravity: SIMD[Self.DTYPE, 4]  # (gx, gy, gz, 0)
    var timestep: Scalar[Self.DTYPE]
    var ground_z: Scalar[Self.DTYPE]
    var friction: Scalar[Self.DTYPE]

    # Per-body properties
    var body_mass: InlineArray[Scalar[Self.DTYPE], Self.NBODY]
    var body_inv_mass: InlineArray[Scalar[Self.DTYPE], Self.NBODY]
    # Diagonal inertia tensor (Ixx, Iyy, Izz) per body
    var body_inertia: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]
    var body_inv_inertia: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]

    # Body local frame (position and orientation relative to parent)
    var body_pos: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]
    var body_quat: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 4]

    # Kinematic tree structure
    var body_parent: InlineArray[Int, Self.NBODY]  # -1 for world

    # Geometry for collision
    var body_geom_type: InlineArray[Int, Self.NBODY]
    var body_radius: InlineArray[Scalar[Self.DTYPE], Self.NBODY]
    var body_half_length: InlineArray[
        Scalar[Self.DTYPE], Self.NBODY
    ]  # For capsules
    # Box half-extents
    var body_half_x: InlineArray[Scalar[Self.DTYPE], Self.NBODY]
    var body_half_y: InlineArray[Scalar[Self.DTYPE], Self.NBODY]
    var body_half_z: InlineArray[Scalar[Self.DTYPE], Self.NBODY]

    # Joint definitions
    var joints: InlineArray[JointDef[Self.DTYPE], _max_one[Self.NJOINT]()]
    var num_joints: Int

    fn __init__(
        out self,
        gravity_z: Scalar[Self.DTYPE] = -9.81,
        timestep: Scalar[Self.DTYPE] = 0.01,
        ground_z: Scalar[Self.DTYPE] = 0.0,
        friction: Scalar[Self.DTYPE] = 0.5,
    ):
        """Initialize model with default values."""
        self.gravity = SIMD[Self.DTYPE, 4](0, 0, gravity_z, 0)
        self.timestep = timestep
        self.ground_z = ground_z
        self.friction = friction

        # Initialize body arrays
        self.body_mass = InlineArray[Scalar[Self.DTYPE], Self.NBODY](
            uninitialized=True
        )
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
        self.body_parent = InlineArray[Int, Self.NBODY](uninitialized=True)

        # Initialize geometry arrays
        self.body_geom_type = InlineArray[Int, Self.NBODY](uninitialized=True)
        self.body_radius = InlineArray[Scalar[Self.DTYPE], Self.NBODY](
            uninitialized=True
        )
        self.body_half_length = InlineArray[Scalar[Self.DTYPE], Self.NBODY](
            uninitialized=True
        )
        self.body_half_x = InlineArray[Scalar[Self.DTYPE], Self.NBODY](
            uninitialized=True
        )
        self.body_half_y = InlineArray[Scalar[Self.DTYPE], Self.NBODY](
            uninitialized=True
        )
        self.body_half_z = InlineArray[Scalar[Self.DTYPE], Self.NBODY](
            uninitialized=True
        )

        # Initialize with defaults
        for i in range(Self.NBODY):
            self.body_mass[i] = Scalar[Self.DTYPE](1.0)
            self.body_inv_mass[i] = Scalar[Self.DTYPE](1.0)
            self.body_parent[i] = -1  # Default: all bodies have world as parent
            self.body_geom_type[i] = GEOM_SPHERE
            self.body_radius[i] = Scalar[Self.DTYPE](0.1)
            self.body_half_length[i] = Scalar[Self.DTYPE](0)
            self.body_half_x[i] = Scalar[Self.DTYPE](0)
            self.body_half_y[i] = Scalar[Self.DTYPE](0)
            self.body_half_z[i] = Scalar[Self.DTYPE](0)

            # Default body position: origin in parent frame
            self.body_pos[i * 3 + 0] = Scalar[Self.DTYPE](0)
            self.body_pos[i * 3 + 1] = Scalar[Self.DTYPE](0)
            self.body_pos[i * 3 + 2] = Scalar[Self.DTYPE](0)

            # Default body orientation: identity quaternion [x, y, z, w]
            self.body_quat[i * 4 + 0] = Scalar[Self.DTYPE](0)
            self.body_quat[i * 4 + 1] = Scalar[Self.DTYPE](0)
            self.body_quat[i * 4 + 2] = Scalar[Self.DTYPE](0)
            self.body_quat[i * 4 + 3] = Scalar[Self.DTYPE](1)

        # Initialize inertia
        for i in range(Self.NBODY * 3):
            self.body_inertia[i] = Scalar[Self.DTYPE](
                0.004
            )  # Default sphere inertia
            self.body_inv_inertia[i] = Scalar[Self.DTYPE](250.0)

        # Initialize joints
        self.joints = InlineArray[
            JointDef[Self.DTYPE], _max_one[Self.NJOINT]()
        ](uninitialized=True)
        for i in range(_max_one[Self.NJOINT]()):
            self.joints[i] = JointDef[Self.DTYPE].empty()
        self.num_joints = 0

    fn set_body(
        mut self,
        body_id: Int,
        mass: Scalar[Self.DTYPE],
        inertia: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        radius: Scalar[Self.DTYPE] = 0.1,
    ):
        """Set body properties.

        Args:
            body_id: Body index.
            mass: Body mass.
            inertia: Diagonal inertia tensor (Ixx, Iyy, Izz).
            radius: Collision radius (default sphere).
        """
        self.body_mass[body_id] = mass
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

        self.body_radius[body_id] = radius
        self.body_geom_type[body_id] = GEOM_SPHERE

    fn set_body_parent(mut self, body_id: Int, parent_id: Int):
        """Set parent body for kinematic tree.

        Args:
            body_id: Child body index.
            parent_id: Parent body index (-1 for world).
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
    ) -> Int:
        """Add a hinge joint to a body.

        Args:
            body_id: Body this joint controls.
            pos: Joint position in parent frame.
            axis: Rotation axis in parent frame.
            tau_limit: Maximum torque.
            range_min: Minimum angle in radians (default: unlimited).
            range_max: Maximum angle in radians (default: unlimited).

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
    ) -> Int:
        """Add a slide joint to a body.

        Args:
            body_id: Body this joint controls.
            pos: Joint position in parent frame.
            axis: Slide axis in parent frame.
            force_limit: Maximum force.
            range_min: Minimum position in meters (default: unlimited).
            range_max: Maximum position in meters (default: unlimited).

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
        )
        self.num_joints += 1
        return joint_idx

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

    # Computed world-space state (via forward kinematics)
    var xpos: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]
    var xquat: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 4]

    # Computed world-space velocities (for collision response)
    var xvel: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]  # Linear
    var xangvel: InlineArray[Scalar[Self.DTYPE], Self.NBODY * 3]  # Angular

    # Contacts
    var contacts: InlineArray[
        ContactInfo[Self.DTYPE], _max_one[Self.MAX_CONTACTS]()
    ]
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

        # Initialize contacts
        self.contacts = InlineArray[
            ContactInfo[Self.DTYPE], _max_one[Self.MAX_CONTACTS]()
        ](uninitialized=True)
        for i in range(_max_one[Self.MAX_CONTACTS]()):
            self.contacts[i] = ContactInfo[Self.DTYPE].empty()
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
