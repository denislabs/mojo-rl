"""Pendulum V2 action struct.

Continuous action for the GPU-enabled Pendulum environment.
"""

from core import Action


struct PendulumV2Action[DTYPE: DType](
    Action, Copyable, ImplicitlyCopyable, Movable
):
    """Continuous action for Pendulum V2: torque in [-2, 2].

    For neural networks with tanh output, the raw output [-1, 1] should
    be scaled by MAX_TORQUE (2.0) to get the actual torque.

    Attributes:
        torque: Applied torque in [-2.0, 2.0]. Positive = counter-clockwise.
    """

    var torque: Scalar[Self.DTYPE]

    fn __init__(out self):
        self.torque = 0.0

    fn __init__(out self, torque: Scalar[Self.DTYPE]):
        self.torque = torque

    fn __copyinit__(out self, existing: Self):
        self.torque = existing.torque

    fn __moveinit__(out self, deinit existing: Self):
        self.torque = existing.torque

    @staticmethod
    fn from_list(values: List[Scalar[Self.DTYPE]]) -> PendulumV2Action[Self.DTYPE]:
        """Create action from list (for agent interface)."""
        var action = PendulumV2Action[Self.DTYPE]()
        if len(values) > 0:
            action.torque = values[0]
        return action^

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to list (for agent interface)."""
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.torque)
        return result^

    @staticmethod
    fn from_discrete(action_idx: Int) -> PendulumV2Action[Self.DTYPE]:
        """Create action from discrete action index.

        Discrete mapping (3 actions):
            0: Left torque (-2.0)
            1: No torque (0.0)
            2: Right torque (+2.0)

        Args:
            action_idx: Discrete action index [0, 2].

        Returns:
            PendulumV2Action with corresponding continuous torque.
        """
        var action = PendulumV2Action[Self.DTYPE]()
        if action_idx == 0:
            action.torque = -2.0
        elif action_idx == 1:
            action.torque = 0.0
        elif action_idx == 2:
            action.torque = 2.0
        return action^

    @staticmethod
    fn left() -> PendulumV2Action[Self.DTYPE]:
        """Maximum negative torque (-2.0)."""
        return PendulumV2Action[Self.DTYPE](torque=-2.0)

    @staticmethod
    fn none() -> PendulumV2Action[Self.DTYPE]:
        """No torque (0.0)."""
        return PendulumV2Action[Self.DTYPE](torque=0.0)

    @staticmethod
    fn right() -> PendulumV2Action[Self.DTYPE]:
        """Maximum positive torque (+2.0)."""
        return PendulumV2Action[Self.DTYPE](torque=2.0)

    fn clamp(self, max_torque: Scalar[Self.DTYPE] = 2.0) -> PendulumV2Action[Self.DTYPE]:
        """Return a new action with torque clamped to [-max_torque, max_torque]."""
        var clamped = self.torque
        if clamped > max_torque:
            clamped = max_torque
        elif clamped < -max_torque:
            clamped = -max_torque
        return PendulumV2Action[Self.DTYPE](torque=clamped)
