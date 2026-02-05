"""Physics3D v2 types - Model/Data separation following MuJoCo.

Model contains static simulation configuration.
Data contains mutable simulation state.

Example usage:
    from physics3d_v2 import Model, Data, step_multi_body

    # Create a 2-body system with max 10 contacts
    var model = Model[DType.float64, 2, 10](
        gravity_z=-9.81, restitution=0.6
    )
    model.set_body(0, mass=1.0, radius=0.1)
    model.set_body(1, mass=1.0, radius=0.1)

    var data = Data[DType.float64, 2, 10]()
    data.set_body_position(0, 0, 0, 1.0)  # Body 0 at height 1m
    data.set_body_position(1, 0, 0, 0.3)  # Body 1 at height 0.3m

    # Simulate
    for i in range(100):
        step_multi_body(model, data)
        print("body0 z =", data.get_body_z(0))

Single body is just Model[DTYPE, 1, MAX_CONTACTS]:
    var model = Model[DType.float64, 1, 5](gravity_z=-9.81)
    model.set_body(0, mass=1.0, radius=0.1)

With joints (pendulum example):
    var model = Model[DType.float64, 1, 5, 1](gravity_z=-9.81)  # MAX_JOINTS=1
    model.set_body(0, mass=1.0, radius=0.1)
    model.add_hinge_joint(
        parent=-1, child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )

ModelGC contains static simulation configuration (kinematic tree, masses, etc.).
DataGC contains mutable simulation state (qpos, qvel, computed xpos/xquat).

Key differences from Cartesian engine:
- State is joint positions (qpos) and velocities (qvel) instead of Cartesian
- Body positions (xpos, xquat) are COMPUTED from qpos via forward kinematics
- Joints ADD DOFs instead of constraining them
- Dynamics computed in joint space (mass matrix, Coriolis, gravity)

Example usage:
    from physics3d_v2.generalized import ModelGC, DataGC
    from physics3d_v2.generalized.integrator import step_gc

    # Create a single pendulum (1 body, 1 hinge joint)
    # NQ=1 (1 angle), NV=1 (1 angular velocity)
    var model = ModelGC[DType.float64, 1, 1, 1, 1, 5]()
    model.set_body(0, mass=1.0, inertia=(0.1, 0.1, 0.1))
    model.set_body_parent(0, -1)  # Parent is world
    model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 1.0),  # Pivot at height 1
        axis=(0.0, 1.0, 0.0),  # Rotate around Y
    )

    var data = DataGC[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = 0.5  # Initial angle (radians)
    data.qvel[0] = 0.0  # Initial angular velocity

    # Simulate
    for i in range(1000):
        step_gc(model, data)
        print("angle =", data.qpos[0], "xpos_z =", data.xpos[2])
"""

from .constants import CONTACT_SIZE
from .joints.hinge_joint import HingeJoint
from .joints.slide_joint import SlideJoint
from .gpu.constants import GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX
from .joint_types import JointDef, JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from .joint_types import get_joint_qpos_size, get_joint_qvel_size


@fieldwise_init
struct ContactInfo[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Contact information for multi-body system (MuJoCo-style).

    Stores contact geometry and involved bodies.
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
    var impulse_n: Scalar[Self.DTYPE]  # Normal impulse (warm start)
    var impulse_t1: Scalar[Self.DTYPE]  # Tangent impulse 1
    var impulse_t2: Scalar[Self.DTYPE]  # Tangent impulse 2

    @staticmethod
    fn empty() -> Self:
        """Create empty contact."""
        return Self(
            -1,
            -1,
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](1),  # Default normal up
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
        )

    fn set(
        mut self,
        body_a: Int,
        body_b: Int,
        pos_x: Scalar[Self.DTYPE],
        pos_y: Scalar[Self.DTYPE],
        pos_z: Scalar[Self.DTYPE],
        normal_x: Scalar[Self.DTYPE],
        normal_y: Scalar[Self.DTYPE],
        normal_z: Scalar[Self.DTYPE],
        dist: Scalar[Self.DTYPE],
    ):
        """Set contact data."""
        self.body_a = body_a
        self.body_b = body_b
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.normal_x = normal_x
        self.normal_y = normal_y
        self.normal_z = normal_z
        self.dist = dist
        self.impulse_n = Scalar[Self.DTYPE](0)
        self.impulse_t1 = Scalar[Self.DTYPE](0)
        self.impulse_t2 = Scalar[Self.DTYPE](0)


# Helper to compute max(1, n) at compile time for array sizing
fn _max_one[n: Int]() -> Int:
    if n > 0:
        return n
    return 1


struct Model[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int = 0,
    MAX_SLIDE_JOINTS: Int = 0,
]:
    """Static configuration for physics simulation.

    Parameters:
        DTYPE: Data type for scalars (float32 or float64).
        NUM_BODIES: Number of bodies (compile-time constant).
        MAX_CONTACTS: Maximum number of contacts (compile-time constant).
        MAX_JOINTS: Maximum number of hinge joints (compile-time constant, default 0).
        MAX_SLIDE_JOINTS: Maximum number of slide joints (compile-time constant, default 0).

    Example:
        # Single body simulation
        var model = Model[DType.float64, 1, 5](gravity_z=-9.81)
        model.set_body(0, mass=1.0, radius=0.1)

        # Multi-body simulation
        var model = Model[DType.float64, 5, 20](gravity_z=-9.81, restitution=0.6)
        for i in range(5):
            model.set_body(i, mass=1.0, radius=0.1)

        # With hinge joints (pendulum)
        var model = Model[DType.float64, 1, 5, 1](gravity_z=-9.81)
        model.set_body(0, mass=1.0, radius=0.1)
        model.add_hinge_joint(...)

        # With slide joints (constrained to plane)
        var model = Model[DType.float64, 1, 5, 0, 2](gravity_z=-9.81)
        model.add_slide_joint(...)
    """

    var gravity_z: Scalar[Self.DTYPE]
    var timestep: Scalar[Self.DTYPE]
    var ground_z: Scalar[Self.DTYPE]
    var restitution: Scalar[Self.DTYPE]
    var friction: Scalar[Self.DTYPE]

    # Per-body properties (compile-time sized arrays)
    var masses: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES]
    var inv_masses: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES]
    var radii: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES]
    # Diagonal inertia: 3 values per body
    var inertias: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3]
    var inv_inertias: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3]

    # Geometry type per body (GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX)
    var geom_types: InlineArray[Int, Self.NUM_BODIES]
    # Half-length for capsules (0 for spheres)
    var half_lengths: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES]
    # Box half-extents (Phase 9: 0 for spheres/capsules)
    var half_x: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES]
    var half_y: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES]
    var half_z: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES]

    # Hinge joint storage (sized to MAX_JOINTS, or 1 if MAX_JOINTS=0 to avoid zero-size array)
    var joints: InlineArray[HingeJoint[Self.DTYPE], _max_one[Self.MAX_JOINTS]()]
    var num_joints: Int

    # Slide joint storage (sized to MAX_SLIDE_JOINTS, or 1 if MAX_SLIDE_JOINTS=0)
    var slide_joints: InlineArray[
        SlideJoint[Self.DTYPE], _max_one[Self.MAX_SLIDE_JOINTS]()
    ]
    var num_slide_joints: Int

    fn __init__(
        out self,
        gravity_z: Scalar[Self.DTYPE] = -9.81,
        timestep: Scalar[Self.DTYPE] = 0.01,
        ground_z: Scalar[Self.DTYPE] = 0.0,
        restitution: Scalar[Self.DTYPE] = 0.0,
        friction: Scalar[Self.DTYPE] = 0.5,
    ):
        """Initialize model with default values."""
        self.gravity_z = gravity_z
        self.timestep = timestep
        self.ground_z = ground_z
        self.restitution = restitution
        self.friction = friction

        # Initialize arrays with zeros
        self.masses = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES](
            uninitialized=True
        )
        self.inv_masses = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES](
            uninitialized=True
        )
        self.radii = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES](
            uninitialized=True
        )
        self.inertias = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3](
            uninitialized=True
        )
        self.inv_inertias = InlineArray[
            Scalar[Self.DTYPE], Self.NUM_BODIES * 3
        ](uninitialized=True)

        for i in range(Self.NUM_BODIES):
            self.masses[i] = Scalar[Self.DTYPE](1.0)
            self.inv_masses[i] = Scalar[Self.DTYPE](1.0)
            self.radii[i] = Scalar[Self.DTYPE](0.1)

        for i in range(Self.NUM_BODIES * 3):
            self.inertias[i] = Scalar[Self.DTYPE](
                0.004
            )  # 2/5 * m * r^2 for unit sphere
            self.inv_inertias[i] = Scalar[Self.DTYPE](250.0)

        # Initialize geometry types (default: sphere)
        self.geom_types = InlineArray[Int, Self.NUM_BODIES](uninitialized=True)
        self.half_lengths = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES](
            uninitialized=True
        )
        self.half_x = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES](
            uninitialized=True
        )
        self.half_y = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES](
            uninitialized=True
        )
        self.half_z = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES](
            uninitialized=True
        )
        for i in range(Self.NUM_BODIES):
            self.geom_types[i] = GEOM_SPHERE
            self.half_lengths[i] = Scalar[Self.DTYPE](0.0)
            self.half_x[i] = Scalar[Self.DTYPE](0.0)
            self.half_y[i] = Scalar[Self.DTYPE](0.0)
            self.half_z[i] = Scalar[Self.DTYPE](0.0)

        # Initialize hinge joints
        self.joints = InlineArray[
            HingeJoint[Self.DTYPE], _max_one[Self.MAX_JOINTS]()
        ](uninitialized=True)
        for i in range(_max_one[Self.MAX_JOINTS]()):
            self.joints[i] = HingeJoint[Self.DTYPE].empty()
        self.num_joints = 0

        # Initialize slide joints
        self.slide_joints = InlineArray[
            SlideJoint[Self.DTYPE], _max_one[Self.MAX_SLIDE_JOINTS]()
        ](uninitialized=True)
        for i in range(_max_one[Self.MAX_SLIDE_JOINTS]()):
            self.slide_joints[i] = SlideJoint[Self.DTYPE].empty()
        self.num_slide_joints = 0

    fn set_body(
        mut self,
        index: Int,
        mass: Scalar[Self.DTYPE],
        radius: Scalar[Self.DTYPE],
    ):
        """Configure a body as a sphere with given mass and radius."""
        self.masses[index] = mass
        self.inv_masses[index] = Scalar[Self.DTYPE](1.0) / mass
        self.radii[index] = radius
        self.geom_types[index] = GEOM_SPHERE
        self.half_lengths[index] = Scalar[Self.DTYPE](0.0)
        self.half_x[index] = Scalar[Self.DTYPE](0.0)
        self.half_y[index] = Scalar[Self.DTYPE](0.0)
        self.half_z[index] = Scalar[Self.DTYPE](0.0)

        # Sphere inertia: I = 2/5 * m * r^2
        var inertia = Scalar[Self.DTYPE](0.4) * mass * radius * radius
        var inv_inertia = Scalar[Self.DTYPE](1.0) / inertia

        self.inertias[index * 3 + 0] = inertia
        self.inertias[index * 3 + 1] = inertia
        self.inertias[index * 3 + 2] = inertia
        self.inv_inertias[index * 3 + 0] = inv_inertia
        self.inv_inertias[index * 3 + 1] = inv_inertia
        self.inv_inertias[index * 3 + 2] = inv_inertia

    fn set_body_capsule(
        mut self,
        index: Int,
        mass: Scalar[Self.DTYPE],
        radius: Scalar[Self.DTYPE],
        half_length: Scalar[Self.DTYPE],
    ):
        """Configure a body as a capsule with given mass, radius, and half-length.

        A capsule is defined by a cylinder of length 2*half_length with hemispherical
        caps of the given radius at each end. The capsule's local Z-axis is along
        the cylinder's axis.

        Args:
            index: Body index.
            mass: Total mass of the capsule.
            radius: Radius of the cylinder and hemispherical caps.
            half_length: Half-length of the cylindrical part (total length = 2*half_length + 2*radius).
        """
        self.masses[index] = mass
        self.inv_masses[index] = Scalar[Self.DTYPE](1.0) / mass
        self.radii[index] = radius
        self.geom_types[index] = GEOM_CAPSULE
        self.half_lengths[index] = half_length
        self.half_x[index] = Scalar[Self.DTYPE](0.0)
        self.half_y[index] = Scalar[Self.DTYPE](0.0)
        self.half_z[index] = Scalar[Self.DTYPE](0.0)

        # Capsule inertia (approximation using cylinder + spherical caps)
        # For a capsule aligned along Z-axis:
        # Cylinder: m_cyl = mass * (2*h) / (2*h + 4/3*r)
        # Spheres:  m_sph = mass * (4/3*r) / (2*h + 4/3*r)
        #
        # Simplified formula for solid capsule:
        # I_xx = I_yy = (1/12)*m*L^2 + (1/4)*m*r^2  (transverse)
        # I_zz = (1/2)*m*r^2  (along axis)
        # where L is the total length including caps

        var h = half_length  # Half-length of cylinder part
        var r = radius
        var r2 = r * r

        # Total length squared (cylinder + caps)
        var L = Scalar[Self.DTYPE](2.0) * h + Scalar[Self.DTYPE](2.0) * r
        var L2 = L * L

        # Transverse inertia (around X or Y axis)
        # Using solid cylinder approximation: I = m*(3*r^2 + L^2)/12
        var I_trans = (
            mass
            * (Scalar[Self.DTYPE](3.0) * r2 + L2)
            / Scalar[Self.DTYPE](12.0)
        )

        # Axial inertia (around Z axis) - cylinder: I = m*r^2/2
        var I_axial = Scalar[Self.DTYPE](0.5) * mass * r2

        var inv_I_trans = Scalar[Self.DTYPE](1.0) / I_trans
        var inv_I_axial = Scalar[Self.DTYPE](1.0) / I_axial

        self.inertias[index * 3 + 0] = I_trans  # Ixx
        self.inertias[index * 3 + 1] = I_trans  # Iyy
        self.inertias[index * 3 + 2] = I_axial  # Izz
        self.inv_inertias[index * 3 + 0] = inv_I_trans
        self.inv_inertias[index * 3 + 1] = inv_I_trans
        self.inv_inertias[index * 3 + 2] = inv_I_axial

    fn set_body_box(
        mut self,
        index: Int,
        mass: Scalar[Self.DTYPE],
        half_x: Scalar[Self.DTYPE],
        half_y: Scalar[Self.DTYPE],
        half_z: Scalar[Self.DTYPE],
    ):
        """Configure a body as a box with given mass and half-extents.

        A box is defined by its half-extents along each axis. The full dimensions
        are 2*half_x by 2*half_y by 2*half_z.

        Args:
            index: Body index.
            mass: Total mass of the box.
            half_x: Half-extent along X axis.
            half_y: Half-extent along Y axis.
            half_z: Half-extent along Z axis.
        """
        self.masses[index] = mass
        self.inv_masses[index] = Scalar[Self.DTYPE](1.0) / mass
        self.radii[index] = Scalar[Self.DTYPE](0.0)  # Not used for boxes
        self.geom_types[index] = GEOM_BOX
        self.half_lengths[index] = Scalar[Self.DTYPE](0.0)
        self.half_x[index] = half_x
        self.half_y[index] = half_y
        self.half_z[index] = half_z

        # Box inertia tensor (for a solid rectangular box):
        # I_xx = (1/3) * m * (hy^2 + hz^2)
        # I_yy = (1/3) * m * (hx^2 + hz^2)
        # I_zz = (1/3) * m * (hx^2 + hy^2)
        # where hx, hy, hz are the half-extents
        # Note: Using 1/3 instead of 1/12 because we use half-extents not full dimensions
        var hx2 = half_x * half_x
        var hy2 = half_y * half_y
        var hz2 = half_z * half_z
        var factor = mass / Scalar[Self.DTYPE](3.0)

        var I_xx = factor * (hy2 + hz2)
        var I_yy = factor * (hx2 + hz2)
        var I_zz = factor * (hx2 + hy2)

        var inv_I_xx = Scalar[Self.DTYPE](1.0) / I_xx
        var inv_I_yy = Scalar[Self.DTYPE](1.0) / I_yy
        var inv_I_zz = Scalar[Self.DTYPE](1.0) / I_zz

        self.inertias[index * 3 + 0] = I_xx
        self.inertias[index * 3 + 1] = I_yy
        self.inertias[index * 3 + 2] = I_zz
        self.inv_inertias[index * 3 + 0] = inv_I_xx
        self.inv_inertias[index * 3 + 1] = inv_I_yy
        self.inv_inertias[index * 3 + 2] = inv_I_zz

    fn add_hinge_joint(
        mut self,
        parent: Int,
        child: Int,
        anchor_parent: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        anchor_child: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
    ) -> Int:
        """Add a hinge joint to the model.

        Args:
            parent: Parent body index (-1 for world anchor).
            child: Child body index.
            anchor_parent: Anchor point in parent's local frame (or world if parent=-1).
            anchor_child: Anchor point in child's local frame.
            axis: Rotation axis in parent's local frame (or world if parent=-1).

        Returns:
            Index of the newly added joint, or -1 if MAX_JOINTS exceeded.
        """
        if self.num_joints >= Self.MAX_JOINTS:
            return -1

        var joint_idx = self.num_joints
        self.joints[joint_idx] = HingeJoint[Self.DTYPE].create(
            parent, child, anchor_parent, anchor_child, axis
        )
        self.num_joints += 1
        return joint_idx

    fn get_joint(self, joint_idx: Int) -> HingeJoint[Self.DTYPE]:
        """Get a hinge joint by index."""
        return self.joints[joint_idx]

    fn add_free_hinge_joint(
        mut self,
        parent: Int,
        child: Int,
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
    ) -> Int:
        """Add a free DOF hinge joint (MuJoCo-style root joint).

        A free DOF joint tracks the rotation angle around the axis but does NOT
        apply constraints. Used for root joints where the body should rotate
        freely while tracking the angle for observations.

        Args:
            parent: Parent body index (-1 for world).
            child: Child body index.
            axis: Rotation axis (will be normalized).

        Returns:
            Index of the newly added joint, or -1 if MAX_JOINTS exceeded.
        """
        if self.num_joints >= Self.MAX_JOINTS:
            return -1

        var joint_idx = self.num_joints
        self.joints[joint_idx] = HingeJoint[Self.DTYPE].create_free_dof(
            parent, child, axis
        )
        self.num_joints += 1
        return joint_idx

    fn add_slide_joint(
        mut self,
        parent: Int,
        child: Int,
        anchor_parent: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        anchor_child: Tuple[
            Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]
        ],
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
    ) -> Int:
        """Add a slide joint to the model.

        A slide joint (prismatic joint) allows translation along a single axis
        while constraining all other motion.

        Args:
            parent: Parent body index (-1 for world anchor).
            child: Child body index.
            anchor_parent: Anchor point in parent's local frame (or world if parent=-1).
            anchor_child: Anchor point in child's local frame.
            axis: Slide axis in parent's local frame (or world if parent=-1).

        Returns:
            Index of the newly added joint, or -1 if MAX_SLIDE_JOINTS exceeded.
        """
        if self.num_slide_joints >= Self.MAX_SLIDE_JOINTS:
            return -1

        var joint_idx = self.num_slide_joints
        self.slide_joints[joint_idx] = SlideJoint[Self.DTYPE].create(
            parent, child, anchor_parent, anchor_child, axis
        )
        self.num_slide_joints += 1
        return joint_idx

    fn get_slide_joint(self, joint_idx: Int) -> SlideJoint[Self.DTYPE]:
        """Get a slide joint by index."""
        return self.slide_joints[joint_idx]

    fn add_free_slide_joint(
        mut self,
        parent: Int,
        child: Int,
        axis: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
    ) -> Int:
        """Add a free DOF slide joint (MuJoCo-style root joint).

        A free DOF joint tracks the position along the axis but does NOT
        apply constraints. Used for root joints where the body should move
        freely while tracking the position for observations.

        Args:
            parent: Parent body index (-1 for world).
            child: Child body index.
            axis: Slide axis (will be normalized).

        Returns:
            Index of the newly added joint, or -1 if MAX_SLIDE_JOINTS exceeded.
        """
        if self.num_slide_joints >= Self.MAX_SLIDE_JOINTS:
            return -1

        var joint_idx = self.num_slide_joints
        self.slide_joints[joint_idx] = SlideJoint[Self.DTYPE].create_free_dof(
            parent, child, axis
        )
        self.num_slide_joints += 1
        return joint_idx


struct Data[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int = 0,
    MAX_SLIDE_JOINTS: Int = 0,
]:
    """Mutable simulation state.

    Parameters:
        DTYPE: Data type for scalars (float32 or float64).
        NUM_BODIES: Number of bodies (compile-time constant).
        MAX_CONTACTS: Maximum number of contacts (compile-time constant).
        MAX_JOINTS: Maximum number of hinge joints (compile-time constant, default 0).
        MAX_SLIDE_JOINTS: Maximum number of slide joints (compile-time constant, default 0).

    Example:
        var data = Data[DType.float64, 5, 20]()
        data.set_body_position(0, 0.0, 0.0, 1.0)
        data.set_body_velocity(0, 0.0, 0.0, 0.0)
    """

    # Per-body state (flattened for GPU compatibility)
    # Positions: 3 floats per body
    var positions: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3]
    # Quaternions: 4 floats per body [qx, qy, qz, qw]
    var quaternions: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 4]
    # Linear velocities: 3 floats per body
    var velocities: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3]
    # Angular velocities: 3 floats per body
    var angular_velocities: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3]
    # Linear accelerations: 3 floats per body
    var accelerations: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3]
    # Angular accelerations: 3 floats per body
    var angular_accelerations: InlineArray[
        Scalar[Self.DTYPE], Self.NUM_BODIES * 3
    ]

    # Contact buffer
    var contacts: InlineArray[ContactInfo[Self.DTYPE], Self.MAX_CONTACTS]
    var num_contacts: Int

    fn __init__(out self):
        """Initialize with all bodies at origin, zero velocity."""
        # Initialize positions
        self.positions = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3](
            uninitialized=True
        )
        for i in range(Self.NUM_BODIES * 3):
            self.positions[i] = Scalar[Self.DTYPE](0)

        # Initialize quaternions to identity [0, 0, 0, 1]
        self.quaternions = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 4](
            uninitialized=True
        )
        for i in range(Self.NUM_BODIES):
            self.quaternions[i * 4 + 0] = Scalar[Self.DTYPE](0)  # qx
            self.quaternions[i * 4 + 1] = Scalar[Self.DTYPE](0)  # qy
            self.quaternions[i * 4 + 2] = Scalar[Self.DTYPE](0)  # qz
            self.quaternions[i * 4 + 3] = Scalar[Self.DTYPE](1)  # qw

        # Initialize velocities to zero
        self.velocities = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3](
            uninitialized=True
        )
        for i in range(Self.NUM_BODIES * 3):
            self.velocities[i] = Scalar[Self.DTYPE](0)

        # Initialize angular velocities to zero
        self.angular_velocities = InlineArray[
            Scalar[Self.DTYPE], Self.NUM_BODIES * 3
        ](uninitialized=True)
        for i in range(Self.NUM_BODIES * 3):
            self.angular_velocities[i] = Scalar[Self.DTYPE](0)

        # Initialize accelerations to zero
        self.accelerations = InlineArray[
            Scalar[Self.DTYPE], Self.NUM_BODIES * 3
        ](uninitialized=True)
        for i in range(Self.NUM_BODIES * 3):
            self.accelerations[i] = Scalar[Self.DTYPE](0)

        # Initialize angular accelerations to zero
        self.angular_accelerations = InlineArray[
            Scalar[Self.DTYPE], Self.NUM_BODIES * 3
        ](uninitialized=True)
        for i in range(Self.NUM_BODIES * 3):
            self.angular_accelerations[i] = Scalar[Self.DTYPE](0)

        # Initialize contact buffer
        self.contacts = InlineArray[ContactInfo[Self.DTYPE], Self.MAX_CONTACTS](
            uninitialized=True
        )
        for i in range(Self.MAX_CONTACTS):
            self.contacts[i] = ContactInfo[Self.DTYPE].empty()
        self.num_contacts = 0

    fn set_body_position(
        mut self,
        body_index: Int,
        x: Scalar[Self.DTYPE],
        y: Scalar[Self.DTYPE],
        z: Scalar[Self.DTYPE],
    ):
        """Set position of a body."""
        self.positions[body_index * 3 + 0] = x
        self.positions[body_index * 3 + 1] = y
        self.positions[body_index * 3 + 2] = z

    fn set_body_velocity(
        mut self,
        body_index: Int,
        vx: Scalar[Self.DTYPE],
        vy: Scalar[Self.DTYPE],
        vz: Scalar[Self.DTYPE],
    ):
        """Set linear velocity of a body."""
        self.velocities[body_index * 3 + 0] = vx
        self.velocities[body_index * 3 + 1] = vy
        self.velocities[body_index * 3 + 2] = vz

    fn set_body_angular_velocity(
        mut self,
        body_index: Int,
        wx: Scalar[Self.DTYPE],
        wy: Scalar[Self.DTYPE],
        wz: Scalar[Self.DTYPE],
    ):
        """Set angular velocity of a body."""
        self.angular_velocities[body_index * 3 + 0] = wx
        self.angular_velocities[body_index * 3 + 1] = wy
        self.angular_velocities[body_index * 3 + 2] = wz

    fn get_body_position(
        self, body_index: Int
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get position of a body."""
        return (
            self.positions[body_index * 3 + 0],
            self.positions[body_index * 3 + 1],
            self.positions[body_index * 3 + 2],
        )

    fn get_body_velocity(
        self, body_index: Int
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get linear velocity of a body."""
        return (
            self.velocities[body_index * 3 + 0],
            self.velocities[body_index * 3 + 1],
            self.velocities[body_index * 3 + 2],
        )

    fn get_body_angular_velocity(
        self, body_index: Int
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get angular velocity of a body."""
        return (
            self.angular_velocities[body_index * 3 + 0],
            self.angular_velocities[body_index * 3 + 1],
            self.angular_velocities[body_index * 3 + 2],
        )

    fn get_body_z(self, body_index: Int) -> Scalar[Self.DTYPE]:
        """Get z position of a body."""
        return self.positions[body_index * 3 + 2]

    fn get_body_vz(self, body_index: Int) -> Scalar[Self.DTYPE]:
        """Get z velocity of a body."""
        return self.velocities[body_index * 3 + 2]


# =============================================================================
# ContactInfoGC - Contact information for GC engine
# =============================================================================


@fieldwise_init
struct ContactInfoGC[DTYPE: DType](ImplicitlyCopyable, Movable):
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
# ModelGC - Static Configuration for GC Engine
# =============================================================================


struct ModelGC[
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
            body_id, qpos_adr, dof_adr, pos, axis, tau_limit, range_min, range_max
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
            body_id, qpos_adr, dof_adr, pos, axis, force_limit, range_min, range_max
        )
        self.num_joints += 1
        return joint_idx

    fn get_joint(self, joint_idx: Int) -> JointDef[Self.DTYPE]:
        """Get joint definition by index."""
        return self.joints[joint_idx]


# =============================================================================
# DataGC - Mutable State for GC Engine
# =============================================================================


struct DataGC[
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
        ContactInfoGC[Self.DTYPE], _max_one[Self.MAX_CONTACTS]()
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
            ContactInfoGC[Self.DTYPE], _max_one[Self.MAX_CONTACTS]()
        ](uninitialized=True)
        for i in range(_max_one[Self.MAX_CONTACTS]()):
            self.contacts[i] = ContactInfoGC[Self.DTYPE].empty()
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
