"""GatherCols[NA] — per-row gather by integer index.

Two inputs: `values` ([B, NA]) and `idx` ([B, 1] holding integer column
indices stored as `Scalar[DT]`). Output: `out[b, 0] = values[b,
Int(idx[b, 0])]`. Used in:

  1. DQN's Q-update block to extract `Q(s, a_taken)` from `Q_online(s)`
     without a CPU shim (replaces trainer.mojo:339-351's host gather).
  2. Double-DQN target-Y graph to gather `Q_target(s', argmax_a Q_online)`.

**Forward-only semantics** — vjp zero-fills both `grad_values` and
`grad_idx`. Same pattern as `ReduceMax`: the trait requires `vjp` to
exist, but gradient never flows through a gather-by-discrete-index in
the DQN topology. For Q-update, the surrounding block owns a scatter
kernel that builds `grad_q_all` from `grad_q_gath` + `mb_a` directly;
for Double-DQN target-Y, the path is `MODE="input_only"` and vjp is a
no-op anyway.

Reasoning for not threading the indices through vjp: `Module.vjp`'s
signature receives only `grad_output` + `grad_inputs` slabs. The
original indices aren't accessible without either (a) caching them on
the struct (lifetime risk for graph users that share buffers) or
(b) overloading `grad_inputs[1]` as in/out (breaks when used through
ComputeGraph, which auto-allocates fresh zero grad slabs). The simplest
correct contract is forward-only with zero vjp, and put the actual
scatter where the indices live — in the calling block.

GPU forward kernel: 1 thread per BATCH row (mirrors `reduce.mojo:34-46`).
GPU vjp kernels: 1 thread per BATCH·NA element (grad_values zero-fill)
+ 1 thread per BATCH (grad_idx zero-fill).
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


def _gather_cols_forward_kernel[
    BATCH: Int, NA: Int,
](
    values: LayoutTensor[DT, Layout.row_major(BATCH, NA), MutAnyOrigin],
    idx: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var a = Int(rebind[Scalar[DT]](idx[b, 0]))
        output[b, 0] = rebind[Scalar[DT]](values[b, a])


def _gather_cols_zero_values_grad_kernel[
    BATCH: Int, NA: Int,
](
    grad_values: LayoutTensor[DT, Layout.row_major(BATCH, NA), MutAnyOrigin],
):
    var lin = Int(global_idx.x)
    var total = BATCH * NA
    if lin < total:
        var b = lin // NA
        var k = lin % NA
        grad_values[b, k] = Scalar[DT](0.0)


def _gather_cols_zero_idx_grad_kernel[
    BATCH: Int,
](
    grad_idx: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        grad_idx[b, 0] = Scalar[DT](0.0)


struct GatherCols[NA: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._build_in_dims()
    comptime IN0_DIM: Int = Self.NA
    comptime OUT_DIM: Int = 1

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=0)
        d[0] = Self.NA
        d[1] = 1
        return d

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
            "GatherCols: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.NA > 0, "GatherCols: NA must be > 0"
        var g = Self()
        comptime if target == "cpu":
            g.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("GatherCols.make[target='gpu']: ctx required")
            g.ts = TargetStorage.make_gpu(ctx.value())
        return g^

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
        assert_tag_for["GatherCols", target](self.ts.target_tag)
        var values = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var idx = inputs.tile[1, BATCH, Self.IN_DIMS[1]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            for b in range(BATCH):
                var a = Int(idx[b, 0])
                output_v[b, 0] = values[b, a]
        else:
            var v_p = values.ptr
            var i_p = idx.ptr
            var o_p = output_v.ptr
            var v_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA), MutAnyOrigin,
            ](v_p)
            var i_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](i_p)
            var o_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](o_p)
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _gather_cols_forward_kernel[BATCH, Self.NA]
            self.ts.ctx.value().enqueue_function[kernel](
                v_lt, i_lt, o_lt, grid_dim=n_blocks, block_dim=TPB,
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
        """Forward-only op: both grad_values and grad_idx zero-fill.
        See module docstring for why."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["GatherCols", target](self.ts.target_tag)
        var grad_values_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var grad_idx_v = grad_inputs.tile[1, BATCH, Self.IN_DIMS[1]]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for k in range(Self.NA):
                    grad_values_v[b, k] = Scalar[DT](0.0)
                grad_idx_v[b, 0] = Scalar[DT](0.0)
        else:
            var gv_p = grad_values_v.ptr
            var gi_p = grad_idx_v.ptr
            var gv_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA), MutAnyOrigin,
            ](gv_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](gi_p)
            comptime values_total = BATCH * Self.NA
            comptime values_blocks = (values_total + TPB - 1) // TPB
            comptime values_kernel = _gather_cols_zero_values_grad_kernel[
                BATCH, Self.NA,
            ]
            self.ts.ctx.value().enqueue_function[values_kernel](
                gv_lt, grid_dim=values_blocks, block_dim=TPB,
            )
            comptime idx_blocks = (BATCH + TPB - 1) // TPB
            comptime idx_kernel = _gather_cols_zero_idx_grad_kernel[BATCH]
            self.ts.ctx.value().enqueue_function[idx_kernel](
                gi_lt, grid_dim=idx_blocks, block_dim=TPB,
            )
