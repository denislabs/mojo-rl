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

    def get_state(mut self) -> Self.StateType:
        """Return current state representation.

        ⚠ `mut self`, AND THE REASON IS THE OBSERVATION, NOT THE STATE. Nothing
        here mutates the environment; the marker exists because building a
        `LayoutTensor` view over a physics tensor needs `mut` on the container
        (`TensorImpl.lt_dyn`), and an observation that casts rays —
        `<rangefinder>`, lidar, line-of-sight — reads eight of them. Mojo also
        forbids caching such a view in a struct field (`AnyOrigin` cannot
        appear in one), so there is no way to prepare them in `__init__` and
        hand them over from a non-mutating method.

        The alternative was a second, GPU-shaped copy of the whole ray
        traversal kept in step with the CPU one by a differential gate. One
        `mut` across ~44 environments is the cheaper half of that trade, and
        it is what lets `physics3d/ray` stay a single implementation for both
        targets.
        """
        ...

    def close(mut self):
        """Clean up resources."""
        ...

    def was_terminated(self) -> Bool:
        """True iff the previous `step` ended via natural termination (a real
        terminal state — pole fell, agent unhealthy, goal reached), NOT
        time-limit truncation.

        `step` collapses termination and truncation into a single `done`
        flag. Value-based RL must keep the TD/GAE bootstrap on truncation (the
        episode could have continued) but drop it on natural termination (the
        state is genuinely terminal). Drivers read this right after stepping.

        Declared on the base `Env` so it has a single home shared by every env
        trait (continuous/discrete) — avoids ambiguity for envs that conform
        to more than one. Default `False` (correct for truncation-only envs
        like Pendulum); terminating envs override it."""
        return False
