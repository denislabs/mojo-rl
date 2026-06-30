"""SoftCrossEntropyLoss — alias of CrossEntropyLoss (storage surface).

Storage-surface port of `nn/loss/soft_cross_entropy.mojo`, which is itself just
`comptime SoftCrossEntropyLoss = CrossEntropyLoss`. The storage `CrossEntropyLoss`
already takes its `targets` as a full `[B, NC]` distribution and computes
`-Σ_c target[b,c]·(logit[b,c] - lse[b])` — i.e. soft cross-entropy. So a hard
one-hot target is just the special case, and the soft-target name is an alias
over the identical implementation (no separate code path).
"""

from .cross_entropy import CrossEntropyLoss


comptime SoftCrossEntropyLoss[NC: Int] = CrossEntropyLoss[NC]
