"""SumOp — `ReduceOp` for the column-wise sum.

scale_factor = 1.0 (no normalization). Backward broadcasts grad_out
unchanged across the input row.
"""

from ...constants import DT
from ...core.reduce_op import ReduceOp


struct SumOp(ReduceOp):
    """Sum reduction: `out = Σ x`, `grad_in[d] = grad_out`."""

    @staticmethod
    def scale_factor[DIM: Int]() -> Scalar[DT]:
        return Scalar[DT](1.0)
