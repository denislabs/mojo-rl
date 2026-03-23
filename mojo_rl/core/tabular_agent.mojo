"""Trait for tabular RL agents that work with discrete state indices."""


trait TabularAgent:
    """Base trait for tabular RL agents.

    Tabular agents work with discrete state indices (Int) rather than
    arbitrary state types. This allows for generic training loops that
    work with any tabular agent.
    """

    def select_action(self, state_idx: Int) -> Int:
        """Select an action given the state index.

        Args:
            state_idx: The index of the current state.

        Returns:
            The index of the selected action.
        """
        ...

    def update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Update the agent based on a transition.

        Args:
            state_idx: The index of the current state.
            action: The action taken.
            reward: The reward received.
            next_state_idx: The index of the next state.
            done: Whether the episode is done.
        """
        ...

    def decay_epsilon(mut self):
        """Decay exploration rate after an episode."""
        ...

    def get_epsilon(self) -> Float64:
        """Get the current exploration rate."""
        ...

    def get_best_action(self, state_idx: Int) -> Int:
        """Get the greedy action for evaluation (no exploration)."""
        ...
