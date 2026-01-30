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
