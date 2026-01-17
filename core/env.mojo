from .state import State
from .action import Action
from render import RendererBase


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

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        ...

    fn reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        ...

    fn get_state(self) -> Self.StateType:
        """Return current state representation."""
        ...

    fn render(
        mut self,
        mut renderer: RendererBase,
    ):
        """Render the environment (optional)."""
        ...

    fn close(mut self):
        """Clean up resources."""
        ...
