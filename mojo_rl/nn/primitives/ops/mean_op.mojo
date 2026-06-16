"""MeanOp — `ReduceOp` for the column-wise mean.

scale_factor = 1/DIM. Backward broadcasts `grad_out / DIM` across the
input row.
"""

from ...constants import DT
from ...core.reduce_op import ReduceOp


struct MeanOp(ReduceOp):
    """Mean reduction: `out = (1/DIM)·Σ x`, `grad_in[d] = grad_out / DIM`."""

    @staticmethod
    def scale_factor[DIM: Int]() -> Scalar[DT]:
        return Scalar[DT](1.0) / Scalar[DT](DIM)
