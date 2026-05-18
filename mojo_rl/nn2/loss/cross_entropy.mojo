"""CrossEntropyLoss[N_CLASSES] — softmax + cross-entropy, numerically stable.

Operates on one-hot targets. Forward computes loss (mean over batch) and
caches the softmax internally for backward; backward reads the cache and
emits `grad_logits = (softmax - target) / BATCH`.

Not a `Module` — the (logits, targets) → scalar signature is different
shape from Module's (input) → output. We may unify under a Loss trait
later, but for Phase 1 a free-standing struct is enough.

Numerical-stability convention (Modular's `linalg.matmul` style):
  log_softmax(x)_c = x_c - logsumexp(x_:)
  logsumexp(x_:)   = max(x_:) + log(sum_c exp(x_c - max(x_:)))

CE = -sum_c target_c * log_softmax_c

For one-hot targets this collapses to -log_softmax[true_class], but
the general form supports label smoothing / soft labels without
special-casing.
"""

from std.math import exp, log
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT


struct CrossEntropyLoss[N_CLASSES: Int](Movable & ImplicitlyDestructible):
    var softmax: List[Scalar[DT]]   # cached softmax probabilities, [BATCH, N_CLASSES] lazily

    def __init__(out self):
        self.softmax = List[Scalar[DT]]()

    def _ensure_cache(mut self, batch: Int):
        var needed = batch * Self.N_CLASSES
        if len(self.softmax) < needed:
            self.softmax.resize(needed, 0.0)

    # ------------------------------------------------------------------
    # Forward: returns mean cross-entropy loss; caches softmax for backward.
    # ------------------------------------------------------------------

    def forward[
        BATCH: Int,
        LL: TensorLayout,
        LT: TensorLayout,
    ](
        mut self,
        logits: TileTensor[DT, LL, MutAnyOrigin],
        targets: TileTensor[DT, LT, MutAnyOrigin],
    ) -> Scalar[DT]:
        comptime assert logits.flat_rank  == 2, "logits must be rank-2 [BATCH, N_CLASSES]"
        comptime assert targets.flat_rank == 2, "targets must be rank-2 [BATCH, N_CLASSES]"

        self._ensure_cache(BATCH)
        var softmax = TileTensor(self.softmax, row_major[BATCH, Self.N_CLASSES]())

        var total_loss: Scalar[DT] = 0.0
        for b in range(BATCH):
            # Pass 1: find max(logits[b, :]) for stable logsumexp.
            var m = logits[b, 0]
            for c in range(1, Self.N_CLASSES):
                if logits[b, c] > m:
                    m = logits[b, c]
            # Pass 2: sum_c exp(logits[b, c] - m).
            var sum_exp: Scalar[DT] = 0.0
            for c in range(Self.N_CLASSES):
                sum_exp += exp(logits[b, c] - m)
            var log_sum_exp = m + log(sum_exp)
            # Pass 3: cache softmax, accumulate -sum(target * log_softmax).
            for c in range(Self.N_CLASSES):
                softmax[b, c] = exp(logits[b, c] - log_sum_exp)
                total_loss += -targets[b, c] * (logits[b, c] - log_sum_exp)

        return total_loss / Scalar[DT](BATCH)

    # ------------------------------------------------------------------
    # Backward: grad_logits = (softmax_cache - target) / BATCH.
    # ------------------------------------------------------------------

    def backward[
        BATCH: Int,
        LT: TensorLayout,
        LG: TensorLayout,
    ](
        self,
        targets: TileTensor[DT, LT, MutAnyOrigin],
        mut grad_logits: TileTensor[DT, LG, MutAnyOrigin],
    ):
        comptime assert targets.flat_rank     == 2, "targets must be rank-2 [BATCH, N_CLASSES]"
        comptime assert grad_logits.flat_rank == 2, "grad_logits must be rank-2 [BATCH, N_CLASSES]"

        var softmax = TileTensor(self.softmax, row_major[BATCH, Self.N_CLASSES]())
        var inv_batch: Scalar[DT] = 1.0 / Scalar[DT](BATCH)

        for b in range(BATCH):
            for c in range(Self.N_CLASSES):
                grad_logits[b, c] = (softmax[b, c] - targets[b, c]) * inv_batch
