"""Action representation for HalfCheetah planar environment."""

from core import Action


@fieldwise_init
struct HalfCheetahPlanarAction[DTYPE: DType](
    Action, Copyable, ImplicitlyCopyable, Movable
):
    """6D continuous action for HalfCheetahPlanar.

    Action space (6D):
        [0]: torque applied at back thigh (hip)
        [1]: torque applied at back shin (knee)
        [2]: torque applied at back foot (ankle)
        [3]: torque applied at front thigh (hip)
        [4]: torque applied at front shin (knee)
        [5]: torque applied at front foot (ankle)

    All actions are normalized to [-1, 1] and scaled by GEAR_RATIO (120.0).
    """

    comptime ACTION_DIM: Int = 6

    var bthigh: Scalar[Self.DTYPE]  # Back hip torque
    var bshin: Scalar[Self.DTYPE]  # Back knee torque
    var bfoot: Scalar[Self.DTYPE]  # Back ankle torque
    var fthigh: Scalar[Self.DTYPE]  # Front hip torque
    var fshin: Scalar[Self.DTYPE]  # Front knee torque
    var ffoot: Scalar[Self.DTYPE]  # Front ankle torque

    fn __init__(out self):
        self.bthigh = 0.0
        self.bshin = 0.0
        self.bfoot = 0.0
        self.fthigh = 0.0
        self.fshin = 0.0
        self.ffoot = 0.0

    fn __copyinit__(out self, existing: Self):
        self.bthigh = existing.bthigh
        self.bshin = existing.bshin
        self.bfoot = existing.bfoot
        self.fthigh = existing.fthigh
        self.fshin = existing.fshin
        self.ffoot = existing.ffoot

    fn __moveinit__(out self, deinit existing: Self):
        self.bthigh = existing.bthigh
        self.bshin = existing.bshin
        self.bfoot = existing.bfoot
        self.fthigh = existing.fthigh
        self.fshin = existing.fshin
        self.ffoot = existing.ffoot

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to 6D list."""
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.bthigh)
        result.append(self.bshin)
        result.append(self.bfoot)
        result.append(self.fthigh)
        result.append(self.fshin)
        result.append(self.ffoot)
        return result^

    @staticmethod
    fn from_list(action: List[Scalar[DTYPE]]) -> Self:
        """Create action from 6D list."""
        var act = HalfCheetahPlanarAction[DTYPE]()
        if len(action) >= 6:
            act.bthigh = action[0]
            act.bshin = action[1]
            act.bfoot = action[2]
            act.fthigh = action[3]
            act.fshin = action[4]
            act.ffoot = action[5]
        return act
