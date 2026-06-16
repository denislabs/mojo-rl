"""GatherActionSlice[NA, K] — per-row gather of a K-wide contiguous slice.

Generalization of `GatherCols[NA]` for distributional Q-learning. Given:
  - `values [B, NA·K]` — Q-net output (e.g. K = N_ATOMS for C51)
  - `idx    [B, 1]`    — action indices as Scalar[DT]
output:
  - `out[b, k] = values[b, Int(idx[b, 0]) · K + k]`  for k ∈ [0, K)

Used by C51's Q-update block to extract per-action atom logits from the
flat [B, NA·K] Q-net output without a CPU shim.

**Forward-only semantics** — vjp zero-fills both `grad_values` and
`grad_idx`. Same rationale as `GatherCols`: the surrounding block owns
the scatter kernel that builds `grad_values` from the gathered
`grad_slice` + the original `mb_a` (avoiding the awkward "vjp reads
indices from grad_inputs[1]" trick that would break ComputeGraph use).
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


def _gather_action_slice_forward_kernel[
    BATCH: Int, NA: Int, K: Int,
](
    values: LayoutTensor[
        DT, Layout.row_major(BATCH, NA * K), MutAnyOrigin,
    ],
    idx: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, K), MutAnyOrigin],
):
    var lin = Int(global_idx.x)
    var total = BATCH * K
    if lin < total:
        var b = lin // K
        var k = lin % K
        var a = Int(rebind[Scalar[DT]](idx[b, 0]))
        output[b, k] = rebind[Scalar[DT]](values[b, a * K + k])


def _gather_action_slice_zero_values_grad_kernel[
    BATCH: Int, NA: Int, K: Int,
](
    grad_values: LayoutTensor[
        DT, Layout.row_major(BATCH, NA * K), MutAnyOrigin,
    ],
):
    var lin = Int(global_idx.x)
    var total = BATCH * NA * K
    if lin < total:
        var b = lin // (NA * K)
        var c = lin % (NA * K)
        grad_values[b, c] = Scalar[DT](0.0)


def _gather_action_slice_zero_idx_grad_kernel[
    BATCH: Int,
](
    grad_idx: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        grad_idx[b, 0] = Scalar[DT](0.0)


struct GatherActionSlice[NA: Int, K: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._build_in_dims()
    comptime IN0_DIM: Int = Self.NA * Self.K
    comptime OUT_DIM: Int = Self.K

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=0)
        d[0] = Self.NA * Self.K
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
            "GatherActionSlice: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.NA > 0, "GatherActionSlice: NA must be > 0"
        comptime assert Self.K > 0, "GatherActionSlice: K must be > 0"
        var g = Self()
        comptime if target == "cpu":
            g.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("GatherActionSlice.make[target='gpu']: ctx required")
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
        assert_tag_for["GatherActionSlice", target](self.ts.target_tag)
        var values = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var idx = inputs.tile[1, BATCH, Self.IN_DIMS[1]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            for b in range(BATCH):
                var a = Int(idx[b, 0])
                var base = a * Self.K
                for k in range(Self.K):
                    output_v[b, k] = values[b, base + k]
        else:
            var v_p = values.ptr
            var i_p = idx.ptr
            var o_p = output_v.ptr
            var v_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA * Self.K), MutAnyOrigin,
            ](v_p)
            var i_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](i_p)
            var o_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.K), MutAnyOrigin,
            ](o_p)
            comptime total = BATCH * Self.K
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _gather_action_slice_forward_kernel[
                BATCH, Self.NA, Self.K,
            ]
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
        The calling block re-runs the scatter using the original indices."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["GatherActionSlice", target](self.ts.target_tag)
        var grad_values_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var grad_idx_v = grad_inputs.tile[1, BATCH, Self.IN_DIMS[1]]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for c in range(Self.NA * Self.K):
                    grad_values_v[b, c] = Scalar[DT](0.0)
                grad_idx_v[b, 0] = Scalar[DT](0.0)
        else:
            var gv_p = grad_values_v.ptr
            var gi_p = grad_idx_v.ptr
            var gv_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA * Self.K), MutAnyOrigin,
            ](gv_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
            ](gi_p)
            comptime values_total = BATCH * Self.NA * Self.K
            comptime values_blocks = (values_total + TPB - 1) // TPB
            comptime values_kernel = _gather_action_slice_zero_values_grad_kernel[
                BATCH, Self.NA, Self.K,
            ]
            self.ts.ctx.value().enqueue_function[values_kernel](
                gv_lt, grid_dim=values_blocks, block_dim=TPB,
            )
            comptime idx_blocks = (BATCH + TPB - 1) // TPB
            comptime idx_kernel = _gather_action_slice_zero_idx_grad_kernel[BATCH]
            self.ts.ctx.value().enqueue_function[idx_kernel](
                gi_lt, grid_dim=idx_blocks, block_dim=TPB,
            )
