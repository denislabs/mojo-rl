"""Generic continuous action for GC environments.

ContAction[N] wraps an InlineArray[Float64, N] and implements the Action trait.
Replaces per-environment action structs like HalfCheetahAction.
"""

from std.collections import InlineArray

from .action import Action


struct ContAction[N: Int](Action, Copyable, Movable):
    """Generic N-dimensional continuous action.

    Each action component is in [-1, 1] and represents a normalized control
    signal that gets scaled by the joint's TAU_LIMIT (gear ratio).
    """

    var data: InlineArray[Float64, Self.N]

    def __init__(out self):
        """Initialize with zeros."""
        self.data = InlineArray[Float64, Self.N](fill=0.0)

    def __init__(out self, data: InlineArray[Float64, Self.N]):
        """Initialize from an existing InlineArray."""
        self.data = data.copy()

    def __init__(out self, *, copy: Self):
        """Copy constructor."""
        self.data = copy.data.copy()

    def __init__(out self, *, deinit take: Self):
        """Move constructor."""
        self.data = take.data^

    @staticmethod
    def from_list(actions: List[Float64]) -> Self:
        """Create action from a list of N values.

        Args:
            actions: List of action values in [-1, 1].

        Returns:
            ContAction with values from the list, zero-padded if short.
        """
        var result = Self()
        for i in range(min(Self.N, len(actions))):
            result.data[i] = actions[i]
        return result^

    def to_list(self) -> List[Float64]:
        """Convert action to a list of N float values."""
        var result = List[Float64](capacity=Self.N)
        for i in range(Self.N):
            result.append(self.data[i])
        return result^

    def clamp(self) -> Self:
        """Return action clamped to [-1, 1]."""
        var result = Self()
        for i in range(Self.N):
            var v = self.data[i]
            if v > 1.0:
                result.data[i] = 1.0
            elif v < -1.0:
                result.data[i] = -1.0
            else:
                result.data[i] = v
        return result^

    def squared_sum(self) -> Float64:
        """Compute sum of squared action values (for control cost)."""
        var total: Float64 = 0.0
        for i in range(Self.N):
            total += self.data[i] * self.data[i]
        return total

    def __getitem__(self, idx: Int) -> Float64:
        """Access action by index."""
        return self.data[idx]

    def __setitem__(mut self, idx: Int, val: Float64):
        """Set action by index."""
        self.data[idx] = val
