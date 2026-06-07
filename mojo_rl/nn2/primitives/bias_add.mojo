"""BiasAdd[DIM] — learnable broadcast bias.

    out[b, i] = in[b, i] + bias[i]

A single `DIM`-vector parameter added to every row. Used as a learnable
(position) embedding in GPT/ViT: instantiated at the full sequence width
(`seq_len*embed_dim`) so each position gets its own additive bias.

Distinct from `Add` (binary, param-less elementwise sum). IN_DIM ==
OUT_DIM == DIM. No cache: backward needs neither the input nor the
output.

  * grad_in = grad_out (identity — bias add is +1 w.r.t. input)
  * grad_bias[i] += sum_b grad_out[b, i]   (reduce over batch)

`bias` is weight-decay-exempt (biases shouldn't decay). Init: β=0
(`Param.make_*` zero-fills; `INIT` accepted for trait conformance but
ignored, matching LayerNorm's β handling).
"""

from std.gpu import global_idx, thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


comptime BA_TPB: Int = 128


def _bias_add_fwd_kernel[
    BATCH: Int, DIM: Int
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * DIM
    if gid >= total:
        return
    var b = gid // DIM
    var i = gid % DIM
    output[b, i] = rebind[Scalar[DT]](input[b, i]) + rebind[Scalar[DT]](
        bias[i]
    )


def _bias_add_copy_kernel[
    N: Int
](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    if gid < N:
        dst[gid] = rebind[Scalar[DT]](src[gid])


def _bias_add_dbias_kernel[
    BATCH: Int, DIM: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_bias: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    # One block per DIM column; threads reduce over the batch.
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var my_db: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        my_db += rebind[Scalar[DT]](grad_output[bi, col])
        bi += BA_TPB
    var total_db = block.sum[block_size=BA_TPB, broadcast=False](val=my_db)
    if t == 0:
        grad_bias[col] = rebind[Scalar[DT]](grad_bias[col]) + total_db[0]


struct BiasAdd[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    var bias: Param["bias", False, Self.DIM]
    var ts: TargetStorage

    def __init__(out self):
        self.bias = Param["bias", False, Self.DIM]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "BiasAdd: target must be 'cpu' or 'gpu'"
        )
        var ba = Self()
        comptime if target == "cpu":
            ba.bias = Param["bias", False, Self.DIM].make_cpu()
            ba.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["BiasAdd.make[target='gpu']"](ctx)
            ba.bias = Param["bias", False, Self.DIM].make_gpu(ctx_v)
            ba.bias.value_dev.value().enqueue_fill(0.0)
            ba.ts = TargetStorage.make_gpu(ctx_v)
        return ba^

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
        assert_tag_for["BiasAdd", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var bias_v = TileTensor(self.bias.value, row_major[Self.DIM]())
            for b in range(BATCH):
                for i in range(Self.DIM):
                    output_v[b, i] = input[b, i] + bias_v[i]
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_d = Layout.row_major(Self.DIM)
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var in_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p)
            var b_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.bias.value_dev.value()
            )
            comptime total = BATCH * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _bias_add_fwd_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, b_lt, out_lt, grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["BiasAdd", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            # grad_in = grad_out (identity).
            for b in range(BATCH):
                for i in range(Self.DIM):
                    grad_input_v[b, i] = grad_output_v[b, i]
            comptime if mode == "all":
                var grad_bias_v = TileTensor(
                    self.bias.grad, row_major[Self.DIM]()
                )
                for b in range(BATCH):
                    for i in range(Self.DIM):
                        grad_bias_v[i] += grad_output_v[b, i]
        else:
            var ctx = self.ts.ctx.value()
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_d = Layout.row_major(Self.DIM)
            comptime total = BATCH * Self.DIM
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p)
            # grad_in = grad_out: device→device copy via flat kernel.
            var go_flat = LayoutTensor[
                DT, Layout.row_major(total), MutAnyOrigin
            ](go_p)
            var gi_flat = LayoutTensor[
                DT, Layout.row_major(total), MutAnyOrigin
            ](gi_p)
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime copy_kernel = _bias_add_copy_kernel[total]
            ctx.enqueue_function[copy_kernel](
                go_flat, gi_flat, grid_dim=n_blocks, block_dim=TPB,
            )
            comptime if mode == "all":
                var gb_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                    self.bias.grad_dev.value()
                )
                comptime db_kernel = _bias_add_dbias_kernel[BATCH, Self.DIM]
                ctx.enqueue_function[db_kernel](
                    go_lt, gb_lt, grid_dim=Self.DIM, block_dim=BA_TPB,
                )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["BiasAdd", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["BiasAdd", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
