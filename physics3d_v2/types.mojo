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
"""

from .constants import CONTACT_SIZE
from .joints.hinge_joint import HingeJoint
from .joints.slide_joint import SlideJoint
from .gpu.constants import GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX


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


struct Model[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0, MAX_SLIDE_JOINTS: Int = 0]:
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
    var slide_joints: InlineArray[SlideJoint[Self.DTYPE], _max_one[Self.MAX_SLIDE_JOINTS]()]
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
        self.inv_inertias = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3](
            uninitialized=True
        )

        for i in range(Self.NUM_BODIES):
            self.masses[i] = Scalar[Self.DTYPE](1.0)
            self.inv_masses[i] = Scalar[Self.DTYPE](1.0)
            self.radii[i] = Scalar[Self.DTYPE](0.1)

        for i in range(Self.NUM_BODIES * 3):
            self.inertias[i] = Scalar[Self.DTYPE](0.004)  # 2/5 * m * r^2 for unit sphere
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
        self.joints = InlineArray[HingeJoint[Self.DTYPE], _max_one[Self.MAX_JOINTS]()](
            uninitialized=True
        )
        for i in range(_max_one[Self.MAX_JOINTS]()):
            self.joints[i] = HingeJoint[Self.DTYPE].empty()
        self.num_joints = 0

        # Initialize slide joints
        self.slide_joints = InlineArray[SlideJoint[Self.DTYPE], _max_one[Self.MAX_SLIDE_JOINTS]()](
            uninitialized=True
        )
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
        var I_trans = mass * (
            Scalar[Self.DTYPE](3.0) * r2 + L2
        ) / Scalar[Self.DTYPE](12.0)

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
        anchor_parent: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        anchor_child: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
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
        anchor_parent: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
        anchor_child: Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]],
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


struct Data[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0, MAX_SLIDE_JOINTS: Int = 0]:
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
    var angular_accelerations: InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3]

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
        self.angular_velocities = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3](
            uninitialized=True
        )
        for i in range(Self.NUM_BODIES * 3):
            self.angular_velocities[i] = Scalar[Self.DTYPE](0)

        # Initialize accelerations to zero
        self.accelerations = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3](
            uninitialized=True
        )
        for i in range(Self.NUM_BODIES * 3):
            self.accelerations[i] = Scalar[Self.DTYPE](0)

        # Initialize angular accelerations to zero
        self.angular_accelerations = InlineArray[Scalar[Self.DTYPE], Self.NUM_BODIES * 3](
            uninitialized=True
        )
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
