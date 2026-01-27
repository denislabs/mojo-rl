"""State representation for HalfCheetah planar environment."""

from core import State


struct HalfCheetahPlanarState[DTYPE: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """Observation state for HalfCheetahPlanar (17D continuous observation).

    Observation space (17D):
        [0]: z position of torso (height)
        [1]: angle of torso
        [2]: angle of back thigh
        [3]: angle of back shin
        [4]: angle of back foot
        [5]: angle of front thigh
        [6]: angle of front shin
        [7]: angle of front foot
        [8]: velocity of x
        [9]: velocity of z (vertical)
        [10]: angular velocity of torso
        [11]: angular velocity of back thigh
        [12]: angular velocity of back shin
        [13]: angular velocity of back foot
        [14]: angular velocity of front thigh
        [15]: angular velocity of front shin
        [16]: angular velocity of front foot
    """

    comptime OBS_DIM: Int = 17

    # Position observations (8)
    var torso_z: Scalar[Self.DTYPE]  # [0]: height
    var torso_angle: Scalar[Self.DTYPE]  # [1]: torso angle
    var bthigh_angle: Scalar[Self.DTYPE]  # [2]: back thigh joint angle
    var bshin_angle: Scalar[Self.DTYPE]  # [3]: back shin joint angle
    var bfoot_angle: Scalar[Self.DTYPE]  # [4]: back foot joint angle
    var fthigh_angle: Scalar[Self.DTYPE]  # [5]: front thigh joint angle
    var fshin_angle: Scalar[Self.DTYPE]  # [6]: front shin joint angle
    var ffoot_angle: Scalar[Self.DTYPE]  # [7]: front foot joint angle

    # Velocity observations (9)
    var vel_x: Scalar[Self.DTYPE]  # [8]: x velocity
    var vel_z: Scalar[Self.DTYPE]  # [9]: z velocity
    var torso_omega: Scalar[Self.DTYPE]  # [10]: torso angular velocity
    var bthigh_omega: Scalar[Self.DTYPE]  # [11]: back thigh angular velocity
    var bshin_omega: Scalar[Self.DTYPE]  # [12]: back shin angular velocity
    var bfoot_omega: Scalar[Self.DTYPE]  # [13]: back foot angular velocity
    var fthigh_omega: Scalar[Self.DTYPE]  # [14]: front thigh angular velocity
    var fshin_omega: Scalar[Self.DTYPE]  # [15]: front shin angular velocity
    var ffoot_omega: Scalar[Self.DTYPE]  # [16]: front foot angular velocity

    fn __init__(out self):
        self.torso_z = 0.0
        self.torso_angle = 0.0
        self.bthigh_angle = 0.0
        self.bshin_angle = 0.0
        self.bfoot_angle = 0.0
        self.fthigh_angle = 0.0
        self.fshin_angle = 0.0
        self.ffoot_angle = 0.0
        self.vel_x = 0.0
        self.vel_z = 0.0
        self.torso_omega = 0.0
        self.bthigh_omega = 0.0
        self.bshin_omega = 0.0
        self.bfoot_omega = 0.0
        self.fthigh_omega = 0.0
        self.fshin_omega = 0.0
        self.ffoot_omega = 0.0

    fn __copyinit__(out self, other: Self):
        self.torso_z = other.torso_z
        self.torso_angle = other.torso_angle
        self.bthigh_angle = other.bthigh_angle
        self.bshin_angle = other.bshin_angle
        self.bfoot_angle = other.bfoot_angle
        self.fthigh_angle = other.fthigh_angle
        self.fshin_angle = other.fshin_angle
        self.ffoot_angle = other.ffoot_angle
        self.vel_x = other.vel_x
        self.vel_z = other.vel_z
        self.torso_omega = other.torso_omega
        self.bthigh_omega = other.bthigh_omega
        self.bshin_omega = other.bshin_omega
        self.bfoot_omega = other.bfoot_omega
        self.fthigh_omega = other.fthigh_omega
        self.fshin_omega = other.fshin_omega
        self.ffoot_omega = other.ffoot_omega

    fn __moveinit__(out self, deinit other: Self):
        self.torso_z = other.torso_z
        self.torso_angle = other.torso_angle
        self.bthigh_angle = other.bthigh_angle
        self.bshin_angle = other.bshin_angle
        self.bfoot_angle = other.bfoot_angle
        self.fthigh_angle = other.fthigh_angle
        self.fshin_angle = other.fshin_angle
        self.ffoot_angle = other.ffoot_angle
        self.vel_x = other.vel_x
        self.vel_z = other.vel_z
        self.torso_omega = other.torso_omega
        self.bthigh_omega = other.bthigh_omega
        self.bshin_omega = other.bshin_omega
        self.bfoot_omega = other.bfoot_omega
        self.fthigh_omega = other.fthigh_omega
        self.fshin_omega = other.fshin_omega
        self.ffoot_omega = other.ffoot_omega

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.torso_z == other.torso_z
            and self.torso_angle == other.torso_angle
            and self.vel_x == other.vel_x
            and self.vel_z == other.vel_z
        )

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to 17D list for agent interface."""
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.torso_z)
        result.append(self.torso_angle)
        result.append(self.bthigh_angle)
        result.append(self.bshin_angle)
        result.append(self.bfoot_angle)
        result.append(self.fthigh_angle)
        result.append(self.fshin_angle)
        result.append(self.ffoot_angle)
        result.append(self.vel_x)
        result.append(self.vel_z)
        result.append(self.torso_omega)
        result.append(self.bthigh_omega)
        result.append(self.bshin_omega)
        result.append(self.bfoot_omega)
        result.append(self.fthigh_omega)
        result.append(self.fshin_omega)
        result.append(self.ffoot_omega)
        return result^

    fn to_list_typed[dtype: DType](self) -> List[Scalar[dtype]]:
        """Convert to 17D list with specified dtype."""
        var result = List[Scalar[dtype]]()
        result.append(Scalar[dtype](self.torso_z))
        result.append(Scalar[dtype](self.torso_angle))
        result.append(Scalar[dtype](self.bthigh_angle))
        result.append(Scalar[dtype](self.bshin_angle))
        result.append(Scalar[dtype](self.bfoot_angle))
        result.append(Scalar[dtype](self.fthigh_angle))
        result.append(Scalar[dtype](self.fshin_angle))
        result.append(Scalar[dtype](self.ffoot_angle))
        result.append(Scalar[dtype](self.vel_x))
        result.append(Scalar[dtype](self.vel_z))
        result.append(Scalar[dtype](self.torso_omega))
        result.append(Scalar[dtype](self.bthigh_omega))
        result.append(Scalar[dtype](self.bshin_omega))
        result.append(Scalar[dtype](self.bfoot_omega))
        result.append(Scalar[dtype](self.fthigh_omega))
        result.append(Scalar[dtype](self.fshin_omega))
        result.append(Scalar[dtype](self.ffoot_omega))
        return result^

    @staticmethod
    fn from_list(obs: List[Scalar[DTYPE]]) -> Self:
        """Create state from 17D observation list."""
        var state = HalfCheetahPlanarState[DTYPE]()
        if len(obs) >= 17:
            state.torso_z = obs[0]
            state.torso_angle = obs[1]
            state.bthigh_angle = obs[2]
            state.bshin_angle = obs[3]
            state.bfoot_angle = obs[4]
            state.fthigh_angle = obs[5]
            state.fshin_angle = obs[6]
            state.ffoot_angle = obs[7]
            state.vel_x = obs[8]
            state.vel_z = obs[9]
            state.torso_omega = obs[10]
            state.bthigh_omega = obs[11]
            state.bshin_omega = obs[12]
            state.bfoot_omega = obs[13]
            state.fthigh_omega = obs[14]
            state.fshin_omega = obs[15]
            state.ffoot_omega = obs[16]
        return state
