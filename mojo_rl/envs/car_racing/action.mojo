"""CarRacing  action struct.

This module defines the action representation for the GPU-enabled CarRacing environment.
"""

from mojo_rl.core import Action


struct CarRacingAction[DTYPE: DType](
    Action, Copyable, ImplicitlyCopyable, Movable
):
    """Continuous action for CarRacing : [steering, gas, brake].

    All values are in [-1, 1] range:
    - steering: -1 (left) to +1 (right)
    - gas: -1 to +1 (remapped to [0, 1] internally)
    - brake: -1 to +1 (remapped to [0, 1] internally)

    This allows neural networks with tanh output to work directly.
    """

    var steering: Scalar[Self.DTYPE]  # -1 (left) to +1 (right)
    var gas: Scalar[Self.DTYPE]  # -1 to +1 (mapped to [0, 1])
    var brake: Scalar[Self.DTYPE]  # -1 to +1 (mapped to [0, 1])

    fn __init__(out self):
        self.steering = 0.0
        self.gas = 0.0
        self.brake = 0.0

    fn __init__(
        out self,
        steering: Scalar[Self.DTYPE],
        gas: Scalar[Self.DTYPE],
        brake: Scalar[Self.DTYPE],
    ):
        self.steering = steering
        self.gas = gas
        self.brake = brake

    fn __init__(out self, *, copy: Self):
        self.steering = copy.steering
        self.gas = copy.gas
        self.brake = copy.brake

    fn __init__(out self, *, deinit take: Self):
        self.steering = take.steering
        self.gas = take.gas
        self.brake = take.brake

    @staticmethod
    fn from_list(
        values: List[Scalar[Self.DTYPE]],
    ) -> CarRacingAction[Self.DTYPE]:
        """Create action from list."""
        var action = CarRacingAction[Self.DTYPE]()
        if len(values) > 0:
            action.steering = values[0]
        if len(values) > 1:
            action.gas = values[1]
        if len(values) > 2:
            action.brake = values[2]
        return action^

    fn to_list(self) -> List[Scalar[Self.DTYPE]]:
        """Convert to list."""
        var result = List[Scalar[Self.DTYPE]]()
        result.append(self.steering)
        result.append(self.gas)
        result.append(self.brake)
        return result^

    @staticmethod
    fn from_discrete(action_idx: Int) -> CarRacingAction[Self.DTYPE]:
        """Create action from discrete action index.

        Discrete actions:
        0: Do nothing
        1: Steer left
        2: Steer right
        3: Gas
        4: Brake
        """
        var action = CarRacingAction[Self.DTYPE]()
        if action_idx == 1:
            action.steering = -0.6
        elif action_idx == 2:
            action.steering = 0.6
        elif action_idx == 3:
            action.gas = 0.5
        elif action_idx == 4:
            action.brake = 0.8
        return action^
