"""Half Cheetah GC action representation.

6-dimensional continuous action space for the actuated joints:
- bthigh: Back thigh rotor torque
- bshin: Back shin rotor torque
- bfoot: Back foot rotor torque
- fthigh: Front thigh rotor torque
- fshin: Front shin rotor torque
- ffoot: Front foot rotor torque

Actions are normalized to [-1, 1] and scaled by gear ratios.
"""

from core import Action


struct HalfCheetahGCAction(Action, Copyable, Movable):
    """Half Cheetah GC action (6D continuous).

    Each action component is in [-1, 1] and represents normalized torque
    that gets scaled by the gear ratio for each joint:
      - bthigh: gear 120 -> max torque 120 Nm
      - bshin: gear 90 -> max torque 90 Nm
      - bfoot: gear 60 -> max torque 60 Nm
      - fthigh: gear 120 -> max torque 120 Nm
      - fshin: gear 60 -> max torque 60 Nm
      - ffoot: gear 30 -> max torque 30 Nm
    """

    var bthigh: Float64  # [0] Back thigh torque
    var bshin: Float64  # [1] Back shin torque
    var bfoot: Float64  # [2] Back foot torque
    var fthigh: Float64  # [3] Front thigh torque
    var fshin: Float64  # [4] Front shin torque
    var ffoot: Float64  # [5] Front foot torque

    fn __init__(out self):
        """Initialize action with zeros."""
        self.bthigh = 0.0
        self.bshin = 0.0
        self.bfoot = 0.0
        self.fthigh = 0.0
        self.fshin = 0.0
        self.ffoot = 0.0

    fn __init__(
        out self,
        bthigh: Float64,
        bshin: Float64,
        bfoot: Float64,
        fthigh: Float64,
        fshin: Float64,
        ffoot: Float64,
    ):
        """Initialize action with all values."""
        self.bthigh = bthigh
        self.bshin = bshin
        self.bfoot = bfoot
        self.fthigh = fthigh
        self.fshin = fshin
        self.ffoot = ffoot

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

    @staticmethod
    fn from_list(actions: List[Float64]) -> Self:
        """Create action from a list of 6 values.

        Args:
            actions: List of 6 action values in [-1, 1].

        Returns:
            HalfCheetahGCAction.
        """
        return Self(
            bthigh=actions[0] if len(actions) > 0 else 0.0,
            bshin=actions[1] if len(actions) > 1 else 0.0,
            bfoot=actions[2] if len(actions) > 2 else 0.0,
            fthigh=actions[3] if len(actions) > 3 else 0.0,
            fshin=actions[4] if len(actions) > 4 else 0.0,
            ffoot=actions[5] if len(actions) > 5 else 0.0,
        )

    fn to_list(self) -> List[Float64]:
        """Convert action to a list of 6 float values."""
        var result = List[Float64](capacity=6)
        result.append(self.bthigh)
        result.append(self.bshin)
        result.append(self.bfoot)
        result.append(self.fthigh)
        result.append(self.fshin)
        result.append(self.ffoot)
        return result^

    fn clamp(self) -> Self:
        """Return action clamped to [-1, 1]."""
        return Self(
            bthigh=max(-1.0, min(1.0, self.bthigh)),
            bshin=max(-1.0, min(1.0, self.bshin)),
            bfoot=max(-1.0, min(1.0, self.bfoot)),
            fthigh=max(-1.0, min(1.0, self.fthigh)),
            fshin=max(-1.0, min(1.0, self.fshin)),
            ffoot=max(-1.0, min(1.0, self.ffoot)),
        )

    fn squared_sum(self) -> Float64:
        """Compute sum of squared action values (for control cost)."""
        return (
            self.bthigh * self.bthigh
            + self.bshin * self.bshin
            + self.bfoot * self.bfoot
            + self.fthigh * self.fthigh
            + self.fshin * self.fshin
            + self.ffoot * self.ffoot
        )

    fn __getitem__(self, idx: Int) -> Float64:
        """Access action by index (0-5)."""
        if idx == 0:
            return self.bthigh
        elif idx == 1:
            return self.bshin
        elif idx == 2:
            return self.bfoot
        elif idx == 3:
            return self.fthigh
        elif idx == 4:
            return self.fshin
        elif idx == 5:
            return self.ffoot
        else:
            return 0.0
