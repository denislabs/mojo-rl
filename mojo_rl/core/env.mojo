from .state import State
from .action import Action


trait Env:
    """Base trait for RL environments with associated State and Action types.

    Implementers must define:
    - comptime StateType: The state representation type
    - comptime ActionType: The action type

    Returns from step: (next_state, reward, done)
    """

    comptime dtype: DType
    comptime StateType: State
    comptime ActionType: Action

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        ...

    def reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        ...

    def get_state(self) -> Self.StateType:
        """Return current state representation."""
        ...

    def close(mut self):
        """Clean up resources."""
        ...
