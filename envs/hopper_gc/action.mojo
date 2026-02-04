"""HopperGC Action struct for continuous control."""

from core import Action


struct HopperGCAction[DTYPE: DType = DType.float64](Copyable, Movable, Action):
    """Action for HopperGC (3D continuous).

    Actions control torques for the three actuated joints:
        [0]: thigh joint torque
        [1]: leg joint torque
        [2]: foot joint torque

    All values should be in [-1, 1] and will be scaled by TORQUE_LIMIT.

    Note: Root joints (rootx, rootz, rooty) are NOT actuated -
    they are free DOFs that track the hopper's position and orientation.
    """

    var thigh: Scalar[Self.DTYPE]
    var leg: Scalar[Self.DTYPE]
    var foot: Scalar[Self.DTYPE]

    fn __init__(out self):
        """Initialize action with zeros."""
        self.thigh = Scalar[Self.DTYPE](0)
        self.leg = Scalar[Self.DTYPE](0)
        self.foot = Scalar[Self.DTYPE](0)

    fn __init__(
        out self,
        thigh: Scalar[Self.DTYPE],
        leg: Scalar[Self.DTYPE],
        foot: Scalar[Self.DTYPE],
    ):
        """Initialize action with specified values."""
        self.thigh = thigh
        self.leg = leg
        self.foot = foot

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.thigh = other.thigh
        self.leg = other.leg
        self.foot = other.foot

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.thigh = other.thigh
        self.leg = other.leg
        self.foot = other.foot

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to list representation."""
        var lst = List[Scalar[Self.DTYPE]](capacity=3)
        lst.append(self.thigh)
        lst.append(self.leg)
        lst.append(self.foot)
        return lst^

    @staticmethod
    fn from_list(actions: List[Scalar[DTYPE]]) -> HopperGCAction[DTYPE]:
        """Create action from list."""
        var action = HopperGCAction[DTYPE]()
        if len(actions) >= 3:
            action.thigh = actions[0]
            action.leg = actions[1]
            action.foot = actions[2]
        return action
