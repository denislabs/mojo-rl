from .state import State
from .action import Action


trait Agent:
    """Base trait for RL agents.

    Agents interact with environments by selecting actions and learning from experience.
    """

    comptime StateType: State
    comptime ActionType: Action

    fn select_action(self, state: Self.StateType) -> Self.ActionType:
        """Select an action given the current state."""
        ...

    fn update(
        mut self,
        state: Self.StateType,
        action: Self.ActionType,
        reward: Float64,
        next_state: Self.StateType,
        done: Bool,
    ):
        """Update the agent based on a transition (s, a, r, s', done)."""
        ...

    fn reset(mut self):
        """Reset agent state for a new episode (e.g., decay exploration)."""
        ...
