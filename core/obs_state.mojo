"""Generic observation state for GC environments.

ObsState[N] wraps an InlineArray[Float64, N] and implements the State trait.
Replaces per-environment state structs like HalfCheetahState.
"""

from collections import InlineArray

from .state import State


struct ObsState[N: Int](Copyable, Movable, State):
    """Generic N-dimensional observation state.

    Wraps an InlineArray[Float64, N] with State trait conformance.
    Used as the observation type for GC environments where observations
    are extracted from qpos/qvel via Joints.extract_obs().
    """

    var data: InlineArray[Float64, Self.N]

    fn __init__(out self):
        """Initialize with zeros."""
        self.data = InlineArray[Float64, Self.N](fill=0.0)

    fn __init__(out self, data: InlineArray[Float64, Self.N]):
        """Initialize from an existing InlineArray."""
        self.data = data.copy()

    fn __init__(out self, copy: Self):
        """Copy constructor."""
        self.data = copy.data.copy()

    fn __init__(out self, deinit take: Self):
        """Move constructor."""
        self.data = take.data^

    fn __eq__(self, other: Self) -> Bool:
        """Check equality with another state."""
        for i in range(Self.N):
            if self.data[i] != other.data[i]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        """Check inequality with another state."""
        return not self.__eq__(other)

    fn __getitem__(self, idx: Int) -> Float64:
        """Access observation by index."""
        return self.data[idx]

    fn __setitem__(mut self, idx: Int, val: Float64):
        """Set observation by index."""
        self.data[idx] = val

    fn to_list(self) -> List[Float64]:
        """Convert state to a list of N float values."""
        var result = List[Float64](capacity=Self.N)
        for i in range(Self.N):
            result.append(self.data[i])
        return result^
