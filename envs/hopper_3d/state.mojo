"""Hopper3D State struct for RL observations."""

from core import State
from .constants3d import Hopper3DConstantsCPU


struct Hopper3DState[DTYPE: DType = DType.float64](Copyable, Movable, State):
    """State observation for Hopper3D (11D).

    Observation layout (matching MuJoCo Hopper):
        [0]: Torso height (z position)
        [1]: Torso pitch angle (rotation around Y-axis)
        [2]: Hip joint angle
        [3]: Knee joint angle
        [4]: Ankle joint angle
        [5]: Torso x velocity
        [6]: Torso z velocity
        [7]: Torso pitch angular velocity
        [8]: Hip angular velocity
        [9]: Knee angular velocity
        [10]: Ankle angular velocity
    """

    # Position observations (5D)
    var torso_z: Scalar[Self.DTYPE]
    var torso_pitch: Scalar[Self.DTYPE]
    var hip_angle: Scalar[Self.DTYPE]
    var knee_angle: Scalar[Self.DTYPE]
    var ankle_angle: Scalar[Self.DTYPE]

    # Velocity observations (6D)
    var vel_x: Scalar[Self.DTYPE]
    var vel_z: Scalar[Self.DTYPE]
    var torso_omega_y: Scalar[Self.DTYPE]
    var hip_omega: Scalar[Self.DTYPE]
    var knee_omega: Scalar[Self.DTYPE]
    var ankle_omega: Scalar[Self.DTYPE]

    fn __init__(out self):
        """Initialize state with zeros."""
        self.torso_z = Scalar[Self.DTYPE](0)
        self.torso_pitch = Scalar[Self.DTYPE](0)
        self.hip_angle = Scalar[Self.DTYPE](0)
        self.knee_angle = Scalar[Self.DTYPE](0)
        self.ankle_angle = Scalar[Self.DTYPE](0)
        self.vel_x = Scalar[Self.DTYPE](0)
        self.vel_z = Scalar[Self.DTYPE](0)
        self.torso_omega_y = Scalar[Self.DTYPE](0)
        self.hip_omega = Scalar[Self.DTYPE](0)
        self.knee_omega = Scalar[Self.DTYPE](0)
        self.ankle_omega = Scalar[Self.DTYPE](0)

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.torso_z = other.torso_z
        self.torso_pitch = other.torso_pitch
        self.hip_angle = other.hip_angle
        self.knee_angle = other.knee_angle
        self.ankle_angle = other.ankle_angle
        self.vel_x = other.vel_x
        self.vel_z = other.vel_z
        self.torso_omega_y = other.torso_omega_y
        self.hip_omega = other.hip_omega
        self.knee_omega = other.knee_omega
        self.ankle_omega = other.ankle_omega

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.torso_z = other.torso_z
        self.torso_pitch = other.torso_pitch
        self.hip_angle = other.hip_angle
        self.knee_angle = other.knee_angle
        self.ankle_angle = other.ankle_angle
        self.vel_x = other.vel_x
        self.vel_z = other.vel_z
        self.torso_omega_y = other.torso_omega_y
        self.hip_omega = other.hip_omega
        self.knee_omega = other.knee_omega
        self.ankle_omega = other.ankle_omega

    fn __eq__(self, other: Self) -> Bool:
        """Check equality."""
        return (
            self.torso_z == other.torso_z
            and self.torso_pitch == other.torso_pitch
        )

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality."""
        return not self.__eq__(other)

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to observation list (11D)."""
        var obs = List[Scalar[Self.DTYPE]](capacity=11)
        obs.append(self.torso_z)
        obs.append(self.torso_pitch)
        obs.append(self.hip_angle)
        obs.append(self.knee_angle)
        obs.append(self.ankle_angle)
        obs.append(self.vel_x)
        obs.append(self.vel_z)
        obs.append(self.torso_omega_y)
        obs.append(self.hip_omega)
        obs.append(self.knee_omega)
        obs.append(self.ankle_omega)
        return obs^

    @staticmethod
    fn from_list(obs: List[Scalar[DTYPE]]) -> Hopper3DState[DTYPE]:
        """Create state from observation list."""
        var state = Hopper3DState[DTYPE]()
        if len(obs) >= 11:
            state.torso_z = obs[0]
            state.torso_pitch = obs[1]
            state.hip_angle = obs[2]
            state.knee_angle = obs[3]
            state.ankle_angle = obs[4]
            state.vel_x = obs[5]
            state.vel_z = obs[6]
            state.torso_omega_y = obs[7]
            state.hip_omega = obs[8]
            state.knee_omega = obs[9]
            state.ankle_omega = obs[10]
        return state
