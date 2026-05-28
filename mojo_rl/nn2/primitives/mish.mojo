"""Mish[DIM] — Elementwise activation aliased through `MishOp`.

Phase 1 of the nn → nn2 porting plan (see `nn2/PORTING_PLAN.md`). The
hand-written `Mish[DIM]` leaf in `mojo_rl/nn/model/mish.mojo` (323 LOC)
has no nn2 counterpart yet; this alias gives consumers the canonical
`Mish[DIM]` call shape backed by the `Elementwise[DIM, OP]` template.

`MishOp` is input-caching (`owns_cache=False`): the orchestrator's
input slab is aliased through `Elementwise._cached_input_ptr` and read
back as `c` in the backward kernel. Mirrors the contract used by
`ReLUOp` and `SwishOp`.

Used by TDMPC2 / MBPO trunks (`Linear → LayerNorm → Mish`) once those
agents port to nn2.
"""

from .elementwise import Elementwise
from .ops.mish_op import MishOp


comptime Mish[DIM: Int] = Elementwise[DIM, MishOp]
