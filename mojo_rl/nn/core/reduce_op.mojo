"""ReduceOp trait — comptime scale factor for linear reductions.

Phase 2.5.C. Mirrors `ElementOp` for the **reduction** family (DIM
inputs → 1 output per row). Sum and Mean differ only in the scale
factor applied to the per-row sum, so a single `Reduce[DIM, OP]`
template carries the shared CPU loop + GPU kernel, and each leaf
collapses to a type alias:

    comptime Sum[DIM]  = Reduce[DIM, SumOp]
    comptime Mean[DIM] = Reduce[DIM, MeanOp]

Forward:  `out[b, 0] = OP.scale_factor[DIM]() · Σ_d input[b, d]`
Backward: `grad_in[b, d] = OP.scale_factor[DIM]() · grad_out[b, 0]`

Non-linear reductions (Max / Min / Prod) don't fit this template —
they need per-row state (argmax index for Max, full product for Prod).
A separate `NonlinearReduceOp` trait can be added when an algorithm
needs one; not in scope here.
"""

from ..constants import DT


trait ReduceOp(Movable & ImplicitlyDeletable):
    """Marker trait — linear-reduction op providing a comptime scale factor.

    Implementations supply one `@staticmethod` returning a `Scalar[DT]`
    that depends only on the per-row reduction length `DIM`. The
    `Reduce[DIM, OP]` template reads `OP.scale_factor[DIM]()` inside
    struct methods (via `Self.OP.scale_factor[...]`) and inside top-level
    GPU kernels (via bare `OP.scale_factor[...]`); see
    `feedback_mojo_kernel_op_param_scope` for the scoping rule."""

    @staticmethod
    def scale_factor[DIM: Int]() -> Scalar[DT]:
        ...
