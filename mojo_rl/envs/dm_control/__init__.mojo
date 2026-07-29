"""dm_control suite ports.

Native Mojo ports of the DeepMind Control Suite tasks on the physics3d
engine. See docs/DM_CONTROL_PORT.md for the staged plan, the engine gaps
that gate each domain, and which tasks are in scope.

Shared pieces live here; each domain gets its own subpackage (mirroring
`references/dm_control-main/dm_control/suite/<domain>.py`).

Suite-wide invariants every task relies on:
  - exactly 1000 control steps per episode (max return 1000)
  - per-step reward in [0, 1] via `rewards.tolerance`
  - no early termination (truncation only)
"""

from .rewards import (
    tolerance,
    sigmoids,
    DEFAULT_VALUE_AT_MARGIN,
    SIGMOID_GAUSSIAN,
    SIGMOID_HYPERBOLIC,
    SIGMOID_LONG_TAIL,
    SIGMOID_RECIPROCAL,
    SIGMOID_COSINE,
    SIGMOID_LINEAR,
    SIGMOID_QUADRATIC,
    SIGMOID_TANH_SQUARED,
)
