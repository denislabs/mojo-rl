"""Linear[IN, OUT] — fully-connected affine layer (CPU Phase 1).

Owns five `List` storage buffers; `TileTensor` views over them are
constructed locally at the top of each method body. Kernel bodies use
N-d indexing exclusively — no pointer arithmetic.

  weight   : [IN, OUT] row-major
  bias     : [OUT]
  grad_w   : [IN, OUT] row-major   (gradient accumulator)
  grad_b   : [OUT]                 (gradient accumulator)
  cache    : [BATCH, IN]            (input cache for backward — sized lazily)

Forward:  output[b, j] = bias[j] + sum_i input[b, i] * weight[i, j]
Backward: grad_input[b, i] = sum_j grad_output[b, j] * weight[i, j]
          grad_w[i, j]    += sum_b cache[b, i] * grad_output[b, j]
          grad_b[j]       += sum_b grad_output[b, j]

Gradients ACCUMULATE (+=) — PyTorch convention. Use zero_grad() to clear.

Cache is owned internally and grown lazily on first forward call; the
caller doesn't allocate or pass cache buffers. Backward reads from the
cache populated by the most recent forward.
"""

from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import Module, ParamVisitor


struct Linear[IN: Int, OUT: Int](Module):
    comptime IN_DIM = Self.IN
    comptime OUT_DIM = Self.OUT
    comptime W_SIZE = Self.IN * Self.OUT
    comptime B_SIZE = Self.OUT

    var weight: List[Scalar[DT]]
    var bias:   List[Scalar[DT]]
    var grad_w: List[Scalar[DT]]
    var grad_b: List[Scalar[DT]]
    var cache:  List[Scalar[DT]]

    # ------------------------------------------------------------------
    # Lifecycle (List is RAII — no __del__)
    # ------------------------------------------------------------------

    def __init__(out self):
        self.weight = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        self.bias   = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        self.grad_w = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        self.grad_b = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        self.cache  = List[Scalar[DT]]()  # empty; grown on first forward

    def _ensure_cache(mut self, batch: Int):
        var needed = batch * Self.IN
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
        """output = input @ weight + bias; internal cache <- input."""
        comptime assert input.flat_rank  == 2, "input must be rank-2 [BATCH, IN]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, OUT]"

        self._ensure_cache(BATCH)

        var weight = TileTensor(self.weight, row_major[Self.IN, Self.OUT]())
        var bias   = TileTensor(self.bias,   row_major[Self.OUT]())
        var cache  = TileTensor(self.cache,  row_major[BATCH, Self.IN]())

        for b in range(BATCH):
            for j in range(Self.OUT):
                var acc = bias[j]
                for i in range(Self.IN):
                    acc += input[b, i] * weight[i, j]
                output[b, j] = acc
            for i in range(Self.IN):
                cache[b, i] = input[b, i]

    def backward[
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, MutAnyOrigin],
        mut grad_input: TileTensor[DT, LGI, MutAnyOrigin],
    ):
        """Backprop using cache from the most recent forward.
        Accumulates into grad_w / grad_b."""
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2 [BATCH, OUT]"
        comptime assert grad_input.flat_rank  == 2, "grad_input must be rank-2 [BATCH, IN]"

        var weight = TileTensor(self.weight, row_major[Self.IN, Self.OUT]())
        var grad_w = TileTensor(self.grad_w, row_major[Self.IN, Self.OUT]())
        var grad_b = TileTensor(self.grad_b, row_major[Self.OUT]())
        var cache  = TileTensor(self.cache,  row_major[BATCH, Self.IN]())

        # grad_input[b, i] = sum_j grad_output[b, j] * weight[i, j]
        for b in range(BATCH):
            for i in range(Self.IN):
                var acc: Scalar[DT] = 0.0
                for j in range(Self.OUT):
                    acc += grad_output[b, j] * weight[i, j]
                grad_input[b, i] = acc

        # grad_w[i, j] += sum_b cache[b, i] * grad_output[b, j]
        for i in range(Self.IN):
            for j in range(Self.OUT):
                var acc: Scalar[DT] = 0.0
                for b in range(BATCH):
                    acc += cache[b, i] * grad_output[b, j]
                grad_w[i, j] = grad_w[i, j] + acc

        # grad_b[j] += sum_b grad_output[b, j]
        for j in range(Self.OUT):
            var acc: Scalar[DT] = 0.0
            for b in range(BATCH):
                acc += grad_output[b, j]
            grad_b[j] = grad_b[j] + acc

    def zero_grad(mut self):
        """Clear gradient accumulators."""
        var grad_w = TileTensor(self.grad_w, row_major[Self.IN, Self.OUT]())
        var grad_b = TileTensor(self.grad_b, row_major[Self.OUT]())
        for i in range(Self.IN):
            for j in range(Self.OUT):
                grad_w[i, j] = 0.0
        for j in range(Self.OUT):
            grad_b[j] = 0.0

    # ------------------------------------------------------------------
    # Module conformance
    # ------------------------------------------------------------------

    def for_each_param[V: ParamVisitor](
        mut self,
        prefix: String,
        mut visitor: V,
    ):
        var sep = "." if prefix.byte_length() > 0 else ""
        var weight = TileTensor(self.weight, row_major[Self.IN, Self.OUT]())
        var grad_w = TileTensor(self.grad_w, row_major[Self.IN, Self.OUT]())
        var bias   = TileTensor(self.bias,   row_major[Self.OUT]())
        var grad_b = TileTensor(self.grad_b, row_major[Self.OUT]())
        visitor.visit(prefix + sep + "weight", weight, grad_w, Self.W_SIZE)
        visitor.visit(prefix + sep + "bias",   bias,   grad_b, Self.B_SIZE)
