"""HopperGC State struct for RL observations."""

from core import State
from .constants_gc import HopperGCConstants


struct HopperGCState[DTYPE: DType = DType.float64](Copyable, Movable, State):
    """State observation for HopperGC (11D).

    Observation layout (matching MuJoCo Hopper):
        [0]: z position (rootz qpos) - height of torso
        [1]: y rotation angle (rooty qpos) - pitch
        [2]: thigh joint angle
        [3]: leg joint angle
        [4]: foot joint angle
        [5]: x velocity (rootx qvel)
        [6]: z velocity (rootz qvel)
        [7]: y angular velocity (rooty qvel)
        [8]: thigh angular velocity
        [9]: leg angular velocity
        [10]: foot angular velocity

    Note: rootx (x position) is excluded from observation as is standard
    for MuJoCo Hopper - makes the task translation invariant.
    """

    # Position observations (5D)
    var z_position: Scalar[Self.DTYPE]     # rootz qpos
    var y_angle: Scalar[Self.DTYPE]        # rooty qpos (pitch)
    var thigh_angle: Scalar[Self.DTYPE]    # thigh qpos
    var leg_angle: Scalar[Self.DTYPE]      # leg qpos
    var foot_angle: Scalar[Self.DTYPE]     # foot qpos

    # Velocity observations (6D)
    var x_velocity: Scalar[Self.DTYPE]     # rootx qvel
    var z_velocity: Scalar[Self.DTYPE]     # rootz qvel
    var y_angular_velocity: Scalar[Self.DTYPE]  # rooty qvel
    var thigh_angular_velocity: Scalar[Self.DTYPE]
    var leg_angular_velocity: Scalar[Self.DTYPE]
    var foot_angular_velocity: Scalar[Self.DTYPE]

    fn __init__(out self):
        """Initialize state with zeros."""
        self.z_position = Scalar[Self.DTYPE](0)
        self.y_angle = Scalar[Self.DTYPE](0)
        self.thigh_angle = Scalar[Self.DTYPE](0)
        self.leg_angle = Scalar[Self.DTYPE](0)
        self.foot_angle = Scalar[Self.DTYPE](0)
        self.x_velocity = Scalar[Self.DTYPE](0)
        self.z_velocity = Scalar[Self.DTYPE](0)
        self.y_angular_velocity = Scalar[Self.DTYPE](0)
        self.thigh_angular_velocity = Scalar[Self.DTYPE](0)
        self.leg_angular_velocity = Scalar[Self.DTYPE](0)
        self.foot_angular_velocity = Scalar[Self.DTYPE](0)

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.z_position = other.z_position
        self.y_angle = other.y_angle
        self.thigh_angle = other.thigh_angle
        self.leg_angle = other.leg_angle
        self.foot_angle = other.foot_angle
        self.x_velocity = other.x_velocity
        self.z_velocity = other.z_velocity
        self.y_angular_velocity = other.y_angular_velocity
        self.thigh_angular_velocity = other.thigh_angular_velocity
        self.leg_angular_velocity = other.leg_angular_velocity
        self.foot_angular_velocity = other.foot_angular_velocity

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.z_position = other.z_position
        self.y_angle = other.y_angle
        self.thigh_angle = other.thigh_angle
        self.leg_angle = other.leg_angle
        self.foot_angle = other.foot_angle
        self.x_velocity = other.x_velocity
        self.z_velocity = other.z_velocity
        self.y_angular_velocity = other.y_angular_velocity
        self.thigh_angular_velocity = other.thigh_angular_velocity
        self.leg_angular_velocity = other.leg_angular_velocity
        self.foot_angular_velocity = other.foot_angular_velocity

    fn __eq__(self, other: Self) -> Bool:
        """Check equality."""
        return (
            self.z_position == other.z_position
            and self.y_angle == other.y_angle
        )

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality."""
        return not self.__eq__(other)

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to observation list (11D)."""
        var obs = List[Scalar[Self.DTYPE]](capacity=11)
        obs.append(self.z_position)
        obs.append(self.y_angle)
        obs.append(self.thigh_angle)
        obs.append(self.leg_angle)
        obs.append(self.foot_angle)
        obs.append(self.x_velocity)
        obs.append(self.z_velocity)
        obs.append(self.y_angular_velocity)
        obs.append(self.thigh_angular_velocity)
        obs.append(self.leg_angular_velocity)
        obs.append(self.foot_angular_velocity)
        return obs^

    @staticmethod
    fn from_list(obs: List[Scalar[DTYPE]]) -> HopperGCState[DTYPE]:
        """Create state from observation list."""
        var state = HopperGCState[DTYPE]()
        if len(obs) >= 11:
            state.z_position = obs[0]
            state.y_angle = obs[1]
            state.thigh_angle = obs[2]
            state.leg_angle = obs[3]
            state.foot_angle = obs[4]
            state.x_velocity = obs[5]
            state.z_velocity = obs[6]
            state.y_angular_velocity = obs[7]
            state.thigh_angular_velocity = obs[8]
            state.leg_angular_velocity = obs[9]
            state.foot_angular_velocity = obs[10]
        return state
