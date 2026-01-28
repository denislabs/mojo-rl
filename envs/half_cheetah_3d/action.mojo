from core import Action
from physics3d import dtype


struct HalfCheetah3DAction(Action, Copyable, Movable):
    """Action for HalfCheetah3D (6D joint torques).

    Uses dtype from physics3d (DType.float32).
    """

    var bthigh: Scalar[dtype]  # Back hip torque
    var bshin: Scalar[dtype]  # Back knee torque
    var bfoot: Scalar[dtype]  # Back ankle torque
    var fthigh: Scalar[dtype]  # Front hip torque
    var fshin: Scalar[dtype]  # Front knee torque
    var ffoot: Scalar[dtype]  # Front ankle torque

    fn __init__(out self):
        """Initialize action with zeros."""
        self.bthigh = Scalar[dtype](0)
        self.bshin = Scalar[dtype](0)
        self.bfoot = Scalar[dtype](0)
        self.fthigh = Scalar[dtype](0)
        self.fshin = Scalar[dtype](0)
        self.ffoot = Scalar[dtype](0)

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.bthigh = other.bthigh
        self.bshin = other.bshin
        self.bfoot = other.bfoot
        self.fthigh = other.fthigh
        self.fshin = other.fshin
        self.ffoot = other.ffoot

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.bthigh = other.bthigh
        self.bshin = other.bshin
        self.bfoot = other.bfoot
        self.fthigh = other.fthigh
        self.fshin = other.fshin
        self.ffoot = other.ffoot

    fn to_list(self) -> List[Scalar[dtype]]:
        """Convert to action list (6D)."""
        var action = List[Scalar[dtype]](capacity=6)
        action.append(self.bthigh)
        action.append(self.bshin)
        action.append(self.bfoot)
        action.append(self.fthigh)
        action.append(self.fshin)
        action.append(self.ffoot)
        return action^

    @staticmethod
    fn from_list(action: List[Scalar[dtype]]) -> HalfCheetah3DAction:
        """Create action from list."""
        var a = HalfCheetah3DAction()
        if len(action) >= 6:
            a.bthigh = action[0]
            a.bshin = action[1]
            a.bfoot = action[2]
            a.fthigh = action[3]
            a.fshin = action[4]
            a.ffoot = action[5]
        return a^
