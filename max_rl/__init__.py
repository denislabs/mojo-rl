"""max_rl — a small MAX-based package for evaluating MAX as an inference backend
for mojo-rl, driven from Mojo via Python interop.

v1 scope: MLP *inference* only (training is out of scope — see
docs/MAX_TRAINING_ASSESSMENT.md for why training-on-MAX is blocked today).

The public surface is :class:`max_rl.mlp_inference.MLPInference`, a configurable
MLP whose dims/batch/device are all variables so multiple shapes can be swept,
plus timing primitives that let the Mojo caller attribute latency across:

  * pure MAX device compute,
  * host<->device data transfer (H2D / D2H),
  * the Mojo<->Python interop bridge itself.
"""

from .mlp_inference import MLPInference

__all__ = ["MLPInference"]
