"""ReduceMax[NA] — per-row max reduction `[B, NA] → [B, 1]`.

Forward-only primitive (target-Y path for DQN — gradient never flows
through max_a Q_target(s', a)). The `Module` trait still requires a
`vjp` implementation; we zero-fill `grad_input`, matching the
`StopGrad` pattern. Callers that need a gradient through max should
use a different op.

Non-linear reduction — doesn't fit the `Reduce[DIM, OP: ReduceOp]`
template (which only covers linear reductions, see
`core/reduce_op.mojo` lines 14-18). Lives as its own `Module`.

Forward:  `out[b, 0] = max_a input[b, a]`
Backward: `grad_input[b, a] = 0` for all (b, a)

GPU kernels:
  - forward: 1 thread per BATCH row, scalar inner loop over NA
    (same shape as `reduce.mojo:34-46`).
  - backward: 1 thread per BATCH·NA element, writes zero
    (same shape as `reduce.mojo:48-62`).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for


def _reduce_max_forward_kernel[
    BATCH: Int, NA: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, NA), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var best: Scalar[DT] = rebind[Scalar[DT]](input[b, 0])
        for a in range(1, NA):
            var v = rebind[Scalar[DT]](input[b, a])
            if v > best:
                best = v
        output[b, 0] = best


def _reduce_max_zero_grad_kernel[
    BATCH: Int, NA: Int,
](
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, NA), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * NA
    if idx < total:
        var b = idx // NA
        var a = idx % NA
        grad_input[b, a] = Scalar[DT](0.0)


struct ReduceMax[NA: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.NA)
    comptime OUT_DIM: Int = 1

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params) but accepted
        for Sequential.make[target, INIT] uniformity."""
        comptime assert target == "cpu" or target == "gpu", (
            "ReduceMax: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.NA > 0, "ReduceMax: NA must be > 0"
        var r = Self()
        comptime if target == "cpu":
            r.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("ReduceMax.make[target='gpu']: ctx required")
            r.ts = TargetStorage.make_gpu(ctx.value())
        return r^

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["ReduceMax", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            for b in range(BATCH):
                var best: Scalar[DT] = input[b, 0]
                for a in range(1, Self.NA):
                    var v = input[b, a]
                    if v > best:
                        best = v
                output_v[b, 0] = best
        else:
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](out_p)
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _reduce_max_forward_kernel[BATCH, Self.NA]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, grid_dim=n_blocks, block_dim=TPB,
            )

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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["ReduceMax", target](self.ts.target_tag)
        # Forward-only op: zero-fill grad_input regardless of grad_output.
        # Matches StopGrad pattern — the target-Y path is MODE="input_only"
        # so this branch is never actually invoked, but the trait requires
        # the method to exist.
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        comptime if target == "cpu":
            for b in range(BATCH):
                for a in range(Self.NA):
                    grad_input_v[b, a] = Scalar[DT](0.0)
        else:
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA), MutAnyOrigin,
            ](gi_p)
            comptime total = BATCH * Self.NA
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _reduce_max_zero_grad_kernel[BATCH, Self.NA]
            self.ts.ctx.value().enqueue_function[kernel](
                gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
