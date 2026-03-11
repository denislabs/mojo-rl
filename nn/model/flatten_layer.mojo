"""Flatten Model wrapper.

Alias wrapping the Flatten DiffOp primitive as a Model.
Identity operation that marks the transition from spatial (Conv2D)
to flat (Linear) layers.
"""

from ..autodiff import AutoFused, Flatten

comptime FlattenLayer[dim: Int] = AutoFused[Flatten[dim]]
