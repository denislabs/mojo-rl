"""TokenMean[SEQ_LEN, DIM] — mean-pool over the sequence axis.

Each sample is `(SEQ_LEN, DIM)` laid out row-major (token-major); the op
averages over the `SEQ_LEN` tokens to produce a single `DIM` vector:

    out[b, d] = (1/SEQ_LEN) * sum_t in[b, t*DIM + d]

IN_DIM = SEQ_LEN*DIM, OUT_DIM = DIM; no params, no cache. Backward
broadcasts the upstream gradient evenly across tokens:

    grad_in[b, t*DIM + d] = grad_out[b, d] / SEQ_LEN

Used by ViT to collapse patch tokens to a single class vector before the
classification head.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


def _token_mean_fwd_kernel[
    BATCH: Int, SEQ: Int, DIM: Int
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, SEQ * DIM), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * DIM
    if gid >= total:
        return
    var b = gid // DIM
    var d = gid % DIM
    var acc: Scalar[DT] = 0.0
    for t in range(SEQ):
        acc += rebind[Scalar[DT]](src[b, t * DIM + d])
    dst[b, d] = acc / Scalar[DT](Float32(SEQ))


def _token_mean_bwd_kernel[
    BATCH: Int, SEQ: Int, DIM: Int
](
    go: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gi: LayoutTensor[DT, Layout.row_major(BATCH, SEQ * DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * SEQ * DIM
    if gid >= total:
        return
    var b = gid // (SEQ * DIM)
    var rem = gid % (SEQ * DIM)
    var d = rem % DIM
    gi[b, rem] = rebind[Scalar[DT]](go[b, d]) / Scalar[DT](Float32(SEQ))


struct TokenMean[SEQ_LEN: Int, DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SEQ_LEN * Self.DIM)
    comptime OUT_DIM = Self.DIM

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TokenMean: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.SEQ_LEN > 0 and Self.DIM > 0, (
            "TokenMean: SEQ_LEN, DIM must be > 0"
        )
        var t = Self()
        comptime if target == "cpu":
            t.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("TokenMean.make[target='gpu']: ctx required")
            t.ts = TargetStorage.make_gpu(ctx.value())
        return t^

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
        assert_tag_for["TokenMean", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var inv_seq: Scalar[DT] = 1.0 / Float32(Self.SEQ_LEN)
            for b in range(BATCH):
                for d in range(Self.DIM):
                    var acc: Scalar[DT] = 0.0
                    for t in range(Self.SEQ_LEN):
                        acc += input[b, t * Self.DIM + d]
                    output_v[b, d] = acc * inv_seq
        else:
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.SEQ_LEN * Self.DIM),
                MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin,
            ](out_p)
            comptime total = BATCH * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _token_mean_fwd_kernel[
                BATCH, Self.SEQ_LEN, Self.DIM
            ]
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
        assert_tag_for["TokenMean", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            var inv_seq: Scalar[DT] = 1.0 / Float32(Self.SEQ_LEN)
            for b in range(BATCH):
                for t in range(Self.SEQ_LEN):
                    for d in range(Self.DIM):
                        grad_input_v[b, t * Self.DIM + d] = (
                            grad_output_v[b, d] * inv_seq
                        )
        else:
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.DIM), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.SEQ_LEN * Self.DIM),
                MutAnyOrigin,
            ](gi_p)
            comptime total = BATCH * Self.SEQ_LEN * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _token_mean_bwd_kernel[
                BATCH, Self.SEQ_LEN, Self.DIM
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
