"""Checkpointable trait for deep RL agents."""


trait Checkpointable:
    """Type-level contract for agents that can save/restore state to disk."""

    def save_checkpoint(self, path: String) raises:
        """Write network weights and optimizer state to path."""
        ...

    def load_checkpoint(mut self, path: String) raises:
        """Restore network weights and optimizer state from path."""
        ...
