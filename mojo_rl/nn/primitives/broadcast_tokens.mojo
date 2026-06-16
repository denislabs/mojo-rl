"""BroadcastTokens[N, DIM] — replicate one vector across N tokens.

    out[b, t*DIM + d] = in[b, d]      (t ∈ [0, N))

The exact adjoint of `TokenMean[N, DIM]` (up to the 1/N factor): forward
fans a single `DIM`-vector out to `N` identical tokens; backward sums the
per-token gradients back onto the source:

    grad_in[b, d] = sum_t grad_out[b, t*DIM + d]

IN_DIM = DIM, OUT_DIM = N*DIM; no params, no cache. Used by the LeWM
decoder to replicate the single global ([CLS]/pooled) representation to
all `N` patch-query positions so it can be the per-token conditioning fed
to `RepeatConditional` (which requires conditioning dim == stream dim).
CPU + GPU.
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


def _bt_fwd_kernel[
    BATCH: Int, N: Int, DIM: Int
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, N * DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * N * DIM
    if gid >= total:
        return
    var b = gid // (N * DIM)
    var rem = gid % (N * DIM)
    var d = rem % DIM
    dst[b, rem] = rebind[Scalar[DT]](src[b, d])


def _bt_bwd_kernel[
    BATCH: Int, N: Int, DIM: Int
](
    go: LayoutTensor[DT, Layout.row_major(BATCH, N * DIM), MutAnyOrigin],
    gi: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * DIM
    if gid >= total:
        return
    var b = gid // DIM
    var d = gid % DIM
    var acc: Scalar[DT] = 0.0
    for t in range(N):
        acc += rebind[Scalar[DT]](go[b, t * DIM + d])
    gi[b, d] = acc


struct BroadcastTokens[N: Int, DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.N * Self.DIM

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N > 0 and Self.DIM > 0, (
            "BroadcastTokens: N, DIM must be > 0"
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "BroadcastTokens: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("BroadcastTokens.make[target='gpu']: ctx required")
            m.ts = TargetStorage.make_gpu(ctx.value())
        return m^

    @staticmethod
    def display_label() -> String:
        return String("BroadcastTokens")

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
        assert_tag_for["BroadcastTokens", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.DIM]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            for b in range(BATCH):
                for t in range(Self.N):
                    for d in range(Self.DIM):
                        output_v[b, t * Self.DIM + d] = input[b, d]
        else:
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin
            ](input.ptr)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.N * Self.DIM), MutAnyOrigin
            ](output_v.ptr)
            comptime total = BATCH * Self.N * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _bt_fwd_kernel[BATCH, Self.N, Self.DIM]
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
        assert_tag_for["BroadcastTokens", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = grad_inputs.tile[0, BATCH, Self.DIM]()

        comptime if target == "cpu":
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var acc: Scalar[DT] = 0.0
                    for t in range(Self.N):
                        acc += go[b, t * Self.DIM + d]
                    gi[b, d] = acc
        else:
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.N * Self.DIM), MutAnyOrigin
            ](go.ptr)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin
            ](gi.ptr)
            comptime total = BATCH * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _bt_bwd_kernel[BATCH, Self.N, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
