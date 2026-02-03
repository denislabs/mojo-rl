"""Hopper3D Action struct for RL actions."""

from core import Action


struct Hopper3DAction[DTYPE: DType = DType.float64](Action, Copyable, Movable):
    """Action for Hopper3D (3D joint torques).

    Actions are normalized to [-1, 1] and scaled by torque_limit.
    """

    var hip: Scalar[Self.DTYPE]    # Hip torque
    var knee: Scalar[Self.DTYPE]   # Knee torque
    var ankle: Scalar[Self.DTYPE]  # Ankle torque

    fn __init__(out self):
        """Initialize action with zeros."""
        self.hip = Scalar[Self.DTYPE](0)
        self.knee = Scalar[Self.DTYPE](0)
        self.ankle = Scalar[Self.DTYPE](0)

    fn __init__(
        out self,
        hip: Scalar[DTYPE],
        knee: Scalar[DTYPE],
        ankle: Scalar[DTYPE],
    ):
        """Initialize action with given values."""
        self.hip = hip
        self.knee = knee
        self.ankle = ankle

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.hip = other.hip
        self.knee = other.knee
        self.ankle = other.ankle

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.hip = other.hip
        self.knee = other.knee
        self.ankle = other.ankle

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to action list (3D)."""
        var action = List[Scalar[Self.DTYPE]](capacity=3)
        action.append(self.hip)
        action.append(self.knee)
        action.append(self.ankle)
        return action^

    @staticmethod
    fn from_list(action: List[Scalar[DTYPE]]) -> Hopper3DAction[DTYPE]:
        """Create action from list."""
        var a = Hopper3DAction[DTYPE]()
        if len(action) >= 3:
            a.hip = action[0]
            a.knee = action[1]
            a.ankle = action[2]
        return a^
