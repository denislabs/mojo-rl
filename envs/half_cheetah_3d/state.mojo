from core import State
from physics3d import dtype


struct HalfCheetah3DState(Copyable, Movable, State):
    """State observation for HalfCheetah3D (17D).

    Observation layout:
    [0]: z position of torso (height)
    [1]: pitch angle of torso (rotation around Y-axis)
    [2-7]: joint angles (6 joints)
    [8-9]: torso velocities (vx, vz)
    [10]: torso angular velocity (wy - pitch rate)
    [11-16]: joint angular velocities

    Uses dtype from physics3d (DType.float32).
    """

    # Position observations (8D)
    var torso_z: Scalar[dtype]
    var torso_pitch: Scalar[dtype]
    var bthigh_angle: Scalar[dtype]
    var bshin_angle: Scalar[dtype]
    var bfoot_angle: Scalar[dtype]
    var fthigh_angle: Scalar[dtype]
    var fshin_angle: Scalar[dtype]
    var ffoot_angle: Scalar[dtype]

    # Velocity observations (9D)
    var vel_x: Scalar[dtype]
    var vel_z: Scalar[dtype]
    var torso_omega_y: Scalar[dtype]
    var bthigh_omega: Scalar[dtype]
    var bshin_omega: Scalar[dtype]
    var bfoot_omega: Scalar[dtype]
    var fthigh_omega: Scalar[dtype]
    var fshin_omega: Scalar[dtype]
    var ffoot_omega: Scalar[dtype]

    fn __init__(out self):
        """Initialize state with zeros."""
        self.torso_z = Scalar[dtype](0)
        self.torso_pitch = Scalar[dtype](0)
        self.bthigh_angle = Scalar[dtype](0)
        self.bshin_angle = Scalar[dtype](0)
        self.bfoot_angle = Scalar[dtype](0)
        self.fthigh_angle = Scalar[dtype](0)
        self.fshin_angle = Scalar[dtype](0)
        self.ffoot_angle = Scalar[dtype](0)
        self.vel_x = Scalar[dtype](0)
        self.vel_z = Scalar[dtype](0)
        self.torso_omega_y = Scalar[dtype](0)
        self.bthigh_omega = Scalar[dtype](0)
        self.bshin_omega = Scalar[dtype](0)
        self.bfoot_omega = Scalar[dtype](0)
        self.fthigh_omega = Scalar[dtype](0)
        self.fshin_omega = Scalar[dtype](0)
        self.ffoot_omega = Scalar[dtype](0)

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.torso_z = other.torso_z
        self.torso_pitch = other.torso_pitch
        self.bthigh_angle = other.bthigh_angle
        self.bshin_angle = other.bshin_angle
        self.bfoot_angle = other.bfoot_angle
        self.fthigh_angle = other.fthigh_angle
        self.fshin_angle = other.fshin_angle
        self.ffoot_angle = other.ffoot_angle
        self.vel_x = other.vel_x
        self.vel_z = other.vel_z
        self.torso_omega_y = other.torso_omega_y
        self.bthigh_omega = other.bthigh_omega
        self.bshin_omega = other.bshin_omega
        self.bfoot_omega = other.bfoot_omega
        self.fthigh_omega = other.fthigh_omega
        self.fshin_omega = other.fshin_omega
        self.ffoot_omega = other.ffoot_omega

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.torso_z = other.torso_z
        self.torso_pitch = other.torso_pitch
        self.bthigh_angle = other.bthigh_angle
        self.bshin_angle = other.bshin_angle
        self.bfoot_angle = other.bfoot_angle
        self.fthigh_angle = other.fthigh_angle
        self.fshin_angle = other.fshin_angle
        self.ffoot_angle = other.ffoot_angle
        self.vel_x = other.vel_x
        self.vel_z = other.vel_z
        self.torso_omega_y = other.torso_omega_y
        self.bthigh_omega = other.bthigh_omega
        self.bshin_omega = other.bshin_omega
        self.bfoot_omega = other.bfoot_omega
        self.fthigh_omega = other.fthigh_omega
        self.fshin_omega = other.fshin_omega
        self.ffoot_omega = other.ffoot_omega

    fn __eq__(self, other: Self) -> Bool:
        """Check equality."""
        return (
            self.torso_z == other.torso_z
            and self.torso_pitch == other.torso_pitch
        )

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality."""
        return not self.__eq__(other)

    fn to_list(self) -> List[Scalar[dtype]]:
        """Convert to observation list (17D)."""
        var obs = List[Scalar[dtype]](capacity=17)
        obs.append(self.torso_z)
        obs.append(self.torso_pitch)
        obs.append(self.bthigh_angle)
        obs.append(self.bshin_angle)
        obs.append(self.bfoot_angle)
        obs.append(self.fthigh_angle)
        obs.append(self.fshin_angle)
        obs.append(self.ffoot_angle)
        obs.append(self.vel_x)
        obs.append(self.vel_z)
        obs.append(self.torso_omega_y)
        obs.append(self.bthigh_omega)
        obs.append(self.bshin_omega)
        obs.append(self.bfoot_omega)
        obs.append(self.fthigh_omega)
        obs.append(self.fshin_omega)
        obs.append(self.ffoot_omega)
        return obs^

    @staticmethod
    fn from_list(obs: List[Scalar[dtype]]) -> HalfCheetah3DState:
        """Create state from observation list."""
        var state = HalfCheetah3DState()
        if len(obs) >= 17:
            state.torso_z = obs[0]
            state.torso_pitch = obs[1]
            state.bthigh_angle = obs[2]
            state.bshin_angle = obs[3]
            state.bfoot_angle = obs[4]
            state.fthigh_angle = obs[5]
            state.fshin_angle = obs[6]
            state.ffoot_angle = obs[7]
            state.vel_x = obs[8]
            state.vel_z = obs[9]
            state.torso_omega_y = obs[10]
            state.bthigh_omega = obs[11]
            state.bshin_omega = obs[12]
            state.bfoot_omega = obs[13]
            state.fthigh_omega = obs[14]
            state.fshin_omega = obs[15]
            state.ffoot_omega = obs[16]
        return state
