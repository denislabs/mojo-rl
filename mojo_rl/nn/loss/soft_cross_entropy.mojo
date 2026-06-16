"""Soft cross-entropy loss (Block D-3).

Per-sample:
  L = -sum_i target_i * log_softmax(logits)_i
  dL/dlogits_i = (softmax(logits)_i - target_i) / BATCH

`nn.loss.CrossEntropyLoss[N_CLASSES]` already implements this exact
math (it accepts soft target distributions — see its body), so this
module is a naming alias. DreamerV3 reward/value heads + TD-MPC2
distributional heads should import the `SoftCrossEntropyLoss` name to
make the semantic obvious at the call site:

```mojo
from mojo_rl.nn.loss.soft_cross_entropy import SoftCrossEntropyLoss
from mojo_rl.nn.loss.two_hot import (
    compute_symlog_bins, two_hot_encode_symlog_batch_ptr,
)

# Build two-hot targets from raw rewards, then optimise the bin logits.
fill_symlog_bins_ptr[NUM_BINS](bins)
two_hot_encode_symlog_batch_ptr[BATCH, NUM_BINS](rewards, bins, targets)
var loss = SoftCrossEntropyLoss[NUM_BINS].make[target="gpu"](ctx)
var L = loss.forward["gpu", BATCH](logits, targets_tile)
```
"""

from .cross_entropy import CrossEntropyLoss


# Naming alias — same struct, same numerics, same trait conformance.
comptime SoftCrossEntropyLoss = CrossEntropyLoss
