"""BranchConcat — parametric alias of `Parallel` (N-ary fan-out column-concat).

The storage `Parallel[*BRANCHES]` is already variadic and IS the fan-out-concat
combinator, so `BranchConcat` is just the legacy name kept for call-site
compatibility (the legacy framework had both a 2-branch `Parallel` and an N-ary
`BranchConcat`; the storage design unifies them into one variadic struct).
"""

from ..core.module import Module
from .parallel import Parallel

comptime BranchConcat[*BRANCHES: Module] = Parallel[*BRANCHES]
