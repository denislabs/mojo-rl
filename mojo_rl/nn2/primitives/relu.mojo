"""ReLU[DIM] — element-wise rectified linear unit (CPU Phase 1).

Parameterless. Forward: output[b, d] = max(0, input[b, d]). Backward:
grad_input[b, d] = grad_output[b, d] if input[b, d] > 0 else 0.

Owns an internal `cache: List[Scalar[DT]]` populated by forward (stores
the pre-activation input) and read by backward. Caller doesn't allocate
or pass cache.

At exactly x == 0 the gradient is conventionally 0 (matches PyTorch).
"""

from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import Module, ParamVisitor


struct ReLU[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var cache: List[Scalar[DT]]  # pre-activation input, sized BATCH * DIM lazily

    def __init__(out self):
        self.cache = List[Scalar[DT]]()

    def _ensure_cache(mut self, batch: Int):
        var needed = batch * Self.DIM
        if len(self.cache) < needed:
            self.cache.resize(needed, 0.0)

    # ------------------------------------------------------------------
    # Forward + backward
    # ------------------------------------------------------------------

    def forward[
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
    ](
        mut self,
        input: TileTensor[DT, LIN, MutAnyOrigin],
        mut output: TileTensor[DT, LOUT, MutAnyOrigin],
    ):
        comptime assert input.flat_rank  == 2, "input must be rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, DIM]"

        self._ensure_cache(BATCH)
        var cache = TileTensor(self.cache, row_major[BATCH, Self.DIM]())

        var zero: Scalar[DT] = 0.0
        for b in range(BATCH):
            for d in range(Self.DIM):
                var x = input[b, d]
                cache[b, d] = x
                output[b, d] = x if x > zero else zero

    def backward[
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, MutAnyOrigin],
        mut grad_input: TileTensor[DT, LGI, MutAnyOrigin],
    ):
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2 [BATCH, DIM]"
        comptime assert grad_input.flat_rank  == 2, "grad_input must be rank-2 [BATCH, DIM]"

        var cache = TileTensor(self.cache, row_major[BATCH, Self.DIM]())
        var zero: Scalar[DT] = 0.0
        for b in range(BATCH):
            for d in range(Self.DIM):
                grad_input[b, d] = grad_output[b, d] if cache[b, d] > zero else zero

    # ------------------------------------------------------------------
    # Module conformance — ReLU has no parameters.
    # ------------------------------------------------------------------

    def for_each_param[V: ParamVisitor](
        mut self,
        prefix: String,
        mut visitor: V,
    ):
        pass
