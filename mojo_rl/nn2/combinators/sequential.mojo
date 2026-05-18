"""Sequential2[L0, L1] — chain of two Modules.

Owns two children (L0 first, L1 second). forward / backward chain them
through a transient intermediate buffer allocated locally on each call.
Each child manages its own internal cache, so the parent doesn't touch
cache memory at all.

Naming convention matches PyTorch's `nn.Sequential`: children are
indexed "0", "1", etc., so `for_each_param("net", v)` emits
"net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias", ...

Phase 1 caps at arity 2. Variadic `Sequential[*L]` is a Phase 1.x
follow-up — once arity-2 works end-to-end, we can chain `Sequential2`
recursively (`Sequential2[A, Sequential2[B, C]]`) for deeper stacks.
"""

from std.memory import alloc
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import Module, ParamVisitor


struct Sequential2[L0: Module, L1: Module](Module):
    comptime IN_DIM = Self.L0.IN_DIM
    comptime OUT_DIM = Self.L1.OUT_DIM
    comptime MID_DIM = Self.L0.OUT_DIM   # == Self.L1.IN_DIM (checked in __init__)

    var first: Self.L0
    var second: Self.L1

    def __init__(out self, var first: Self.L0, var second: Self.L1):
        comptime assert Self.L0.OUT_DIM == Self.L1.IN_DIM, (
            "Sequential2: L0.OUT_DIM must equal L1.IN_DIM"
        )
        self.first = first^
        self.second = second^

    # ------------------------------------------------------------------
    # Forward + backward — chain through a transient intermediate.
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
        comptime assert input.flat_rank  == 2, "input must be rank-2 [BATCH, L0.IN_DIM]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, L1.OUT_DIM]"

        # Intermediate buffer needs MutAnyOrigin for child.forward; Mojo
        # doesn't widen local-List origins into MutAnyOrigin at call sites.
        # Use a typed UnsafePointer with explicit MutAnyOrigin, alloc/free
        # inside this method.
        var mid_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * Self.MID_DIM)
        var mid = TileTensor(mid_ptr, row_major[BATCH, Self.MID_DIM]())

        self.first.forward[BATCH](input, mid)
        self.second.forward[BATCH](mid, output)

        mid_ptr.free()

    def backward[
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, MutAnyOrigin],
        mut grad_input: TileTensor[DT, LGI, MutAnyOrigin],
    ):
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input must be rank-2"

        var mid_grad_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * Self.MID_DIM)
        var mid_grad = TileTensor(mid_grad_ptr, row_major[BATCH, Self.MID_DIM]())

        self.second.backward[BATCH](grad_output, mid_grad)
        self.first.backward[BATCH](mid_grad, grad_input)

        mid_grad_ptr.free()

    # ------------------------------------------------------------------
    # Module conformance — recurse with indexed prefix ("0", "1").
    # ------------------------------------------------------------------

    def for_each_param[V: ParamVisitor](
        mut self,
        prefix: String,
        mut visitor: V,
    ):
        var sep = "." if prefix.byte_length() > 0 else ""
        self.first.for_each_param(prefix + sep + "0", visitor)
        self.second.for_each_param(prefix + sep + "1", visitor)
