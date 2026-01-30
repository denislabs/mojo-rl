"""Physics3D v2 types - Model/Data separation following MuJoCo.

Model contains static simulation configuration.
Data contains mutable simulation state.
"""

from .constants import GEOM_PLANE, GEOM_SPHERE
from math import sqrt


@fieldwise_init
struct Body[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Static body properties."""

    var mass: Scalar[Self.DTYPE]
    var inertia_xx: Scalar[Self.DTYPE]  # Diagonal inertia
    var inertia_yy: Scalar[Self.DTYPE]
    var inertia_zz: Scalar[Self.DTYPE]

    @staticmethod
    fn create(
        mass: Scalar[Self.DTYPE],
        ixx: Scalar[Self.DTYPE],
        iyy: Scalar[Self.DTYPE],
        izz: Scalar[Self.DTYPE],
    ) -> Self:
        """Create body with mass and diagonal inertia."""
        return Self(
            mass,
            ixx,
            iyy,
            izz,
        )

    @staticmethod
    fn create_sphere(
        mass: Scalar[Self.DTYPE],
        radius: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](1.0),
    ) -> Self:
        """Create body with mass and sphere inertia (I = 2/5 * m * r^2)."""
        var i = 0.4 * mass * radius * radius
        return Self(
            mass,
            i,
            i,
            i,
        )


@fieldwise_init
struct Geom[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Collision geometry attached to a body."""

    var type: Int  # GEOM_PLANE or GEOM_SPHERE
    var size: Scalar[Self.DTYPE]  # radius for sphere, unused for plane
    var pos_x: Scalar[Self.DTYPE]  # Local position offset
    var pos_y: Scalar[Self.DTYPE]
    var pos_z: Scalar[Self.DTYPE]

    @staticmethod
    fn sphere(radius: Scalar[Self.DTYPE]) -> Self:
        """Create a sphere geometry."""
        return Self(
            GEOM_SPHERE,
            radius,
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
        )

    @staticmethod
    fn plane() -> Self:
        """Create a plane geometry (infinite ground plane)."""
        return Self(
            GEOM_PLANE,
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
        )


@fieldwise_init
struct Contact[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Contact information from collision detection."""

    var active: Bool
    var pos_x: Scalar[Self.DTYPE]  # Contact point (world)
    var pos_y: Scalar[Self.DTYPE]
    var pos_z: Scalar[Self.DTYPE]
    var normal_x: Scalar[Self.DTYPE]  # Contact normal
    var normal_y: Scalar[Self.DTYPE]
    var normal_z: Scalar[Self.DTYPE]
    var depth: Scalar[Self.DTYPE]  # Penetration depth (positive = penetrating)
    var impulse: Scalar[Self.DTYPE]  # Normal impulse (for warm-start)

    @staticmethod
    fn empty() -> Self:
        """Create inactive contact."""
        return Self(
            False,
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](1),  # Default normal up
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
        )


@fieldwise_init
struct Model[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Static simulation configuration (immutable after creation)."""

    var gravity_x: Scalar[Self.DTYPE]
    var gravity_y: Scalar[Self.DTYPE]
    var gravity_z: Scalar[Self.DTYPE]
    var timestep: Scalar[Self.DTYPE]
    var body: Body[Self.DTYPE]
    var geom: Geom[Self.DTYPE]
    var ground_z: Scalar[Self.DTYPE]  # Ground plane height (default: 0)
    var restitution: Scalar[
        Self.DTYPE
    ]  # Coefficient of restitution for bounces

    @staticmethod
    fn create(
        body: Body[Self.DTYPE],
        geom: Geom[Self.DTYPE],
        timestep: Scalar[Self.DTYPE] = 0.01,
        gravity_z: Scalar[Self.DTYPE] = -9.81,
        ground_z: Scalar[Self.DTYPE] = 0.0,
        restitution: Scalar[Self.DTYPE] = 0.0,
    ) -> Self:
        """Initialize model with full parameters."""
        return Self(
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](0),
            Scalar[Self.DTYPE](gravity_z),
            Scalar[Self.DTYPE](timestep),
            body,
            geom,
            Scalar[Self.DTYPE](ground_z),
            Scalar[Self.DTYPE](restitution),
        )


struct Data[DTYPE: DType]:
    """Mutable simulation state."""

    # Generalized coordinates (FREE joint: 3 pos + 4 quat = 7)
    var qpos: InlineArray[Scalar[Self.DTYPE], 7]  # [x, y, z, qx, qy, qz, qw]
    var qvel: InlineArray[Scalar[Self.DTYPE], 6]  # [vx, vy, vz, wx, wy, wz]
    var qacc: InlineArray[Scalar[Self.DTYPE], 6]  # Computed accelerations

    # World-frame quantities (computed from qpos)
    var xpos_x: Scalar[Self.DTYPE]
    var xpos_y: Scalar[Self.DTYPE]
    var xpos_z: Scalar[Self.DTYPE]
    var xquat_x: Scalar[Self.DTYPE]
    var xquat_y: Scalar[Self.DTYPE]
    var xquat_z: Scalar[Self.DTYPE]
    var xquat_w: Scalar[Self.DTYPE]

    # Forces
    var qfrc_applied: InlineArray[
        Scalar[Self.DTYPE], 6
    ]  # External forces/torques

    # Contact (Phase 2)
    var contact: Contact[Self.DTYPE]

    fn __init__(out self):
        """Initialize with identity pose at origin."""
        # Position at origin with identity quaternion
        self.qpos = InlineArray[Scalar[Self.DTYPE], 7](uninitialized=True)
        self.qpos[0] = Scalar[Self.DTYPE](0)  # x
        self.qpos[1] = Scalar[Self.DTYPE](0)  # y
        self.qpos[2] = Scalar[Self.DTYPE](0)  # z
        self.qpos[3] = Scalar[Self.DTYPE](0)  # qx
        self.qpos[4] = Scalar[Self.DTYPE](0)  # qy
        self.qpos[5] = Scalar[Self.DTYPE](0)  # qz
        self.qpos[6] = Scalar[Self.DTYPE](1)  # qw (identity quaternion)

        # Zero velocity
        self.qvel = InlineArray[Scalar[Self.DTYPE], 6](uninitialized=True)
        self.qvel[0] = Scalar[Self.DTYPE](0)
        self.qvel[1] = Scalar[Self.DTYPE](0)
        self.qvel[2] = Scalar[Self.DTYPE](0)
        self.qvel[3] = Scalar[Self.DTYPE](0)
        self.qvel[4] = Scalar[Self.DTYPE](0)
        self.qvel[5] = Scalar[Self.DTYPE](0)

        # Zero acceleration
        self.qacc = InlineArray[Scalar[Self.DTYPE], 6](uninitialized=True)
        self.qacc[0] = Scalar[Self.DTYPE](0)
        self.qacc[1] = Scalar[Self.DTYPE](0)
        self.qacc[2] = Scalar[Self.DTYPE](0)
        self.qacc[3] = Scalar[Self.DTYPE](0)
        self.qacc[4] = Scalar[Self.DTYPE](0)
        self.qacc[5] = Scalar[Self.DTYPE](0)

        # World frame
        self.xpos_x = Scalar[Self.DTYPE](0)
        self.xpos_y = Scalar[Self.DTYPE](0)
        self.xpos_z = Scalar[Self.DTYPE](0)
        self.xquat_x = Scalar[Self.DTYPE](0)
        self.xquat_y = Scalar[Self.DTYPE](0)
        self.xquat_z = Scalar[Self.DTYPE](0)
        self.xquat_w = Scalar[Self.DTYPE](1)

        # Zero applied forces
        self.qfrc_applied = InlineArray[Scalar[Self.DTYPE], 6](
            uninitialized=True
        )
        self.qfrc_applied[0] = Scalar[Self.DTYPE](0)
        self.qfrc_applied[1] = Scalar[Self.DTYPE](0)
        self.qfrc_applied[2] = Scalar[Self.DTYPE](0)
        self.qfrc_applied[3] = Scalar[Self.DTYPE](0)
        self.qfrc_applied[4] = Scalar[Self.DTYPE](0)
        self.qfrc_applied[5] = Scalar[Self.DTYPE](0)

        # No contact initially
        self.contact = Contact[Self.DTYPE].empty()

    fn set_position(
        mut self,
        x: Scalar[Self.DTYPE],
        y: Scalar[Self.DTYPE],
        z: Scalar[Self.DTYPE],
    ):
        """Set the position."""
        self.qpos[0] = x
        self.qpos[1] = y
        self.qpos[2] = z

    fn set_velocity(
        mut self,
        vx: Scalar[Self.DTYPE],
        vy: Scalar[Self.DTYPE],
        vz: Scalar[Self.DTYPE],
    ):
        """Set linear velocity."""
        self.qvel[0] = vx
        self.qvel[1] = vy
        self.qvel[2] = vz

    fn set_angular_velocity(
        mut self,
        wx: Scalar[Self.DTYPE],
        wy: Scalar[Self.DTYPE],
        wz: Scalar[Self.DTYPE],
    ):
        """Set angular velocity."""
        self.qvel[3] = wx
        self.qvel[4] = wy
        self.qvel[5] = wz

    fn get_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get position as tuple."""
        return (
            self.qpos[0],
            self.qpos[1],
            self.qpos[2],
        )

    fn get_velocity(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get linear velocity as tuple."""
        return (
            self.qvel[0],
            self.qvel[1],
            self.qvel[2],
        )

    fn get_z(self) -> Scalar[Self.DTYPE]:
        """Get z position."""
        return self.qpos[2]

    fn get_vz(self) -> Scalar[Self.DTYPE]:
        """Get z velocity."""
        return self.qvel[2]


# ==============================================================================
# Phase 3: Multi-body types
# ==============================================================================

from .constants import CONTACT_SIZE


@fieldwise_init
struct MultiBodyContact[DTYPE: DType](ImplicitlyCopyable, Movable):
    """Single contact for multi-body system (MuJoCo-style).

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


struct MultiBodyModel[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int]:
    """Static configuration for multi-body simulation.

    Parameters:
        DTYPE: Data type for scalars.
        NUM_BODIES: Number of bodies (compile-time).
        MAX_CONTACTS: Maximum number of contacts (compile-time).
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


struct MultiBodyData[DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int]:
    """Mutable state for multi-body simulation.

    Parameters:
        DTYPE: Data type for scalars.
        NUM_BODIES: Number of bodies (compile-time).
        MAX_CONTACTS: Maximum number of contacts (compile-time).
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
    var contacts: InlineArray[MultiBodyContact[Self.DTYPE], Self.MAX_CONTACTS]
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
        self.contacts = InlineArray[MultiBodyContact[Self.DTYPE], Self.MAX_CONTACTS](
            uninitialized=True
        )
        for i in range(Self.MAX_CONTACTS):
            self.contacts[i] = MultiBodyContact[Self.DTYPE].empty()
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
