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
"""

from .constants import CONTACT_SIZE


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


struct Model[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int]:
    """Static configuration for physics simulation.

    Parameters:
        DTYPE: Data type for scalars (float32 or float64).
        NUM_BODIES: Number of bodies (compile-time constant).
        MAX_CONTACTS: Maximum number of contacts (compile-time constant).

    Example:
        # Single body simulation
        var model = Model[DType.float64, 1, 5](gravity_z=-9.81)
        model.set_body(0, mass=1.0, radius=0.1)

        # Multi-body simulation
        var model = Model[DType.float64, 5, 20](gravity_z=-9.81, restitution=0.6)
        for i in range(5):
            model.set_body(i, mass=1.0, radius=0.1)
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

        # Sphere inertia: I = 2/5 * m * r^2
        var inertia = Scalar[Self.DTYPE](0.4) * mass * radius * radius
        var inv_inertia = Scalar[Self.DTYPE](1.0) / inertia

        self.inertias[index * 3 + 0] = inertia
        self.inertias[index * 3 + 1] = inertia
        self.inertias[index * 3 + 2] = inertia
        self.inv_inertias[index * 3 + 0] = inv_inertia
        self.inv_inertias[index * 3 + 1] = inv_inertia
        self.inv_inertias[index * 3 + 2] = inv_inertia


struct Data[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int]:
    """Mutable simulation state.

    Parameters:
        DTYPE: Data type for scalars (float32 or float64).
        NUM_BODIES: Number of bodies (compile-time constant).
        MAX_CONTACTS: Maximum number of contacts (compile-time constant).

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

    fn get_body_z(self, body_index: Int) -> Scalar[Self.DTYPE]:
        """Get z position of a body."""
        return self.positions[body_index * 3 + 2]

    fn get_body_vz(self, body_index: Int) -> Scalar[Self.DTYPE]:
        """Get z velocity of a body."""
        return self.velocities[body_index * 3 + 2]
