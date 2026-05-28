"""DuelingHead[NA] — Dueling DQN aggregation as a leaf Module.

Input  shape: [B, 1 + NA]  — first column V(s), remaining NA columns A(s, ·).
Output shape: [B, NA]      — Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

Drops the average-advantage offset so the network learns identifiable
V and A streams (Wang et al. 2016).

Backward:
    grad_in[b, 0]     = Σ_a grad_out[b, a]
    grad_in[b, 1 + a] = grad_out[b, a] − (1 / NA) · Σ_a grad_out[b, a]

Use as: `Sequential[backbone..., Linear[H, 1 + NA], DuelingHead[NA]]`.
Pure architectural swap — no trainer or block change.

Ports the legacy `DuelingQ` strategy at
`mojo_rl/deep_agents/core/strategies/q_output.mojo:202-324` to the
nn2 Module trait (no PARAM_SIZE / CACHE_SIZE; forward + vjp are the
whole surface). No params, no cache. CPU + GPU.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


def _dueling_combine_kernel[
    BATCH: Int, NA: Int,
](
    raw_in: LayoutTensor[
        DT, Layout.row_major(BATCH, NA + 1), MutAnyOrigin,
    ],
    q_out: LayoutTensor[DT, Layout.row_major(BATCH, NA), MutAnyOrigin],
):
    """`q_out[b, a] = raw_in[b, 0] + raw_in[b, 1 + a] − mean_a raw_in[b, 1 + a]`.
    1 thread per BATCH row, scalar inner loop over NA.
    """
    var b = Int(global_idx.x)
    if b < BATCH:
        var v = rebind[Scalar[DT]](raw_in[b, 0])
        var mean_a: Scalar[DT] = 0.0
        for a in range(NA):
            mean_a = mean_a + rebind[Scalar[DT]](raw_in[b, 1 + a])
        mean_a = mean_a * (Scalar[DT](1.0) / Scalar[DT](NA))
        for a in range(NA):
            var adv = rebind[Scalar[DT]](raw_in[b, 1 + a])
            q_out[b, a] = v + (adv - mean_a)


def _dueling_grad_kernel[
    BATCH: Int, NA: Int,
](
    grad_out: LayoutTensor[
        DT, Layout.row_major(BATCH, NA), MutAnyOrigin,
    ],
    grad_in: LayoutTensor[
        DT, Layout.row_major(BATCH, NA + 1), MutAnyOrigin,
    ],
):
    """`grad_in[b, 0] = Σ_a grad_out[b, a]`,
    `grad_in[b, 1 + a] = grad_out[b, a] − (1 / NA) · Σ_a grad_out[b, a]`.
    1 thread per BATCH row.
    """
    var b = Int(global_idx.x)
    if b < BATCH:
        var sum_dq: Scalar[DT] = 0.0
        for a in range(NA):
            sum_dq = sum_dq + rebind[Scalar[DT]](grad_out[b, a])
        grad_in[b, 0] = sum_dq
        var inv = Scalar[DT](1.0) / Scalar[DT](NA)
        for a in range(NA):
            grad_in[b, 1 + a] = (
                rebind[Scalar[DT]](grad_out[b, a]) - inv * sum_dq
            )


struct DuelingHead[NA: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.NA + 1)
    comptime OUT_DIM: Int = Self.NA

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
            "DuelingHead: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.NA > 0, "DuelingHead: NA must be > 0"
        var h = Self()
        comptime if target == "cpu":
            h.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("DuelingHead.make[target='gpu']: ctx required")
            h.ts = TargetStorage.make_gpu(ctx.value())
        return h^

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
        assert_tag_for["DuelingHead", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var inv = Scalar[DT](1.0) / Scalar[DT](Self.NA)
            for b in range(BATCH):
                var v = input[b, 0]
                var sum_a: Scalar[DT] = 0.0
                for a in range(Self.NA):
                    sum_a = sum_a + input[b, 1 + a]
                var mean_a = sum_a * inv
                for a in range(Self.NA):
                    output_v[b, a] = v + (input[b, 1 + a] - mean_a)
        else:
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA + 1), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA), MutAnyOrigin,
            ](out_p)
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _dueling_combine_kernel[BATCH, Self.NA]
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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["DuelingHead", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])

        comptime if target == "cpu":
            var inv = Scalar[DT](1.0) / Scalar[DT](Self.NA)
            for b in range(BATCH):
                var sum_dq: Scalar[DT] = 0.0
                for a in range(Self.NA):
                    sum_dq = sum_dq + grad_output_v[b, a]
                grad_input_v[b, 0] = sum_dq
                for a in range(Self.NA):
                    grad_input_v[b, 1 + a] = (
                        grad_output_v[b, a] - inv * sum_dq
                    )
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.NA + 1), MutAnyOrigin,
            ](gi_p)
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _dueling_grad_kernel[BATCH, Self.NA]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
