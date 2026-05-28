"""MinMaxNorm[DIM] — per-sample (x - min) / (max - min) scaling.

Phase 2 of `nn2/PORTING_PLAN.md`. Used by MuZero (paper appendix
Training, see muzero-general models.py:138-145) and EZ-V2 to keep the
representation network's output bounded, with gradient flowing through
the rescaling so the rep network learns to produce well-spread outputs.

Math (per sample of dim N):
    m = min(x), M = max(x), s = clamp(M - m, ≥ ε)
    y_j = (x_j - m) / s

Backward (given grad_y, compute grad_x):
    G  = Σ grad_y
    Gy = Σ grad_y · y
    grad_x[argmax] = (grad_y[argmax] - Gy) / s
    grad_x[argmin] = (Gy + grad_y[argmin] - G) / s
    grad_x[i ∉ {argmin, argmax}] = grad_y[i] / s
    grad_x = 0 in the degenerate (M - m < ε) case.

Sum-zero invariant: Σ grad_x = 0 (gradient is shift-invariant, since
y is shift-invariant in x).

Cache: leaf-owned copy of the input row, so backward can re-derive
min/max/argmin/argmax without indexing the orchestrator's input slab.
CPU-only at landing — GPU is a follow-up once a consumer needs it.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


comptime MMN_EPS: Scalar[DT] = 1e-5


struct MinMaxNorm[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    # Cache: per-sample copy of x, re-scanned for min/max/argmin/argmax
    # in vjp. Cheaper than caching indices (no int-as-float fragility).
    var cache_x: List[Scalar[DT]]
    var ts: TargetStorage

    def __init__(out self):
        self.cache_x = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params). GPU path
        raises — no nn2 consumer needs it yet (see PORTING_PLAN.md
        Phase 2)."""
        comptime assert target == "cpu" or target == "gpu", (
            "MinMaxNorm: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.DIM > 1, "MinMaxNorm: DIM must be > 1"
        var n = Self()
        comptime if target == "cpu":
            n.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("MinMaxNorm.make[target='gpu']: ctx required")
            raise Error(
                "MinMaxNorm: GPU path not implemented yet (see"
                " PORTING_PLAN.md Phase 2)"
            )
        return n^

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["MinMaxNorm", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            ensure_cpu_buffer(self.cache_x, BATCH * Self.DIM)
            var cache_v = TileTensor(
                self.cache_x, row_major[BATCH, Self.DIM](),
            )
            for b in range(BATCH):
                var x0: Scalar[DT] = input[b, 0]
                var min_val = x0
                var max_val = x0
                cache_v[b, 0] = x0
                for i in range(1, Self.DIM):
                    var v: Scalar[DT] = input[b, i]
                    cache_v[b, i] = v
                    if v < min_val:
                        min_val = v
                    if v > max_val:
                        max_val = v
                var s = max_val - min_val
                if s < MMN_EPS:
                    s = MMN_EPS
                var inv_s = Scalar[DT](1.0) / s
                for i in range(Self.DIM):
                    output_v[b, i] = (cache_v[b, i] - min_val) * inv_s
        else:
            raise Error("MinMaxNorm.forward[target='gpu']: not implemented")

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["MinMaxNorm", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            var cache_v = TileTensor(
                self.cache_x, row_major[BATCH, Self.DIM](),
            )
            for b in range(BATCH):
                var x0: Scalar[DT] = cache_v[b, 0]
                var min_val = x0
                var max_val = x0
                var argmin = 0
                var argmax = 0
                for i in range(1, Self.DIM):
                    var v: Scalar[DT] = cache_v[b, i]
                    if v < min_val:
                        min_val = v
                        argmin = i
                    if v > max_val:
                        max_val = v
                        argmax = i
                var raw_s = max_val - min_val
                var degenerate = raw_s < MMN_EPS
                if degenerate:
                    for i in range(Self.DIM):
                        grad_input_v[b, i] = Scalar[DT](0.0)
                    continue
                var inv_s = Scalar[DT](1.0) / raw_s
                var g_sum: Scalar[DT] = 0.0
                var gy_sum: Scalar[DT] = 0.0
                for i in range(Self.DIM):
                    var y = (cache_v[b, i] - min_val) * inv_s
                    var dy: Scalar[DT] = grad_output_v[b, i]
                    g_sum += dy
                    gy_sum += dy * y
                var dy_argmin: Scalar[DT] = grad_output_v[b, argmin]
                var dy_argmax: Scalar[DT] = grad_output_v[b, argmax]
                for i in range(Self.DIM):
                    var dy: Scalar[DT] = grad_output_v[b, i]
                    var dx: Scalar[DT]
                    if i == argmin and i == argmax:
                        dx = Scalar[DT](0.0)
                    elif i == argmin:
                        dx = (gy_sum + dy_argmin - g_sum) * inv_s
                    elif i == argmax:
                        dx = (dy_argmax - gy_sum) * inv_s
                    else:
                        dx = dy * inv_s
                    grad_input_v[b, i] = dx
        else:
            raise Error("MinMaxNorm.vjp[target='gpu']: not implemented")
