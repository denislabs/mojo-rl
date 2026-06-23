"""BiasAdd[DIM] — learnable broadcast bias (storage surface).

Transformed from legacy `nn.primitives.BiasAdd` (surface-only change). The CPU
loops + the three GPU kernels (forward / grad-input copy / grad-bias reduce) are
carried over verbatim.

    out[b, i] = in[b, i] + bias[i]

A single `DIM`-vector parameter added to every row. Used as a learnable
(position) embedding in GPT/ViT: instantiated at the full sequence width
(`seq_len*embed_dim`) so each position gets its own additive bias.

IN_DIM == OUT_DIM == DIM. No cache: backward needs neither the input nor the
output.

  * grad_in = grad_out (identity — bias add is +1 w.r.t. input)
  * grad_bias[i] += sum_b grad_out[b, i]   (reduce over batch)

`bias` is weight-decay-exempt (biases shouldn't decay). Init: β=0
(`INIT.init_bias` zero-fills, matching the legacy β handling).
"""

from std.gpu import global_idx, thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor


comptime BA_TPB: Int = 128


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
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


struct BiasAdd[DIM_: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var bias: Param["bias", False, Self.DIM_]

    def __init__(out self):
        self.bias = Param["bias", False, Self.DIM_]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var ba = Self()
        ba.bias = Param["bias", False, Self.DIM_].make[target](ctx)
        INIT.init_bias[target](ba.bias.val, Self.DIM_, ctx)
        return ba^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.DIM_)
            var in_v = TileTensor(in0.data, row_major[B, Self.DIM_]())
            var out_v = TileTensor(out.data, row_major[B, Self.DIM_]())
            var bias_v = TileTensor(self.bias.val.data, row_major[Self.DIM_]())
            for b in range(B):
                for i in range(Self.DIM_):
                    out_v[b, i] = in_v[b, i] + bias_v[i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime ld = Layout.row_major(Self.DIM_)
            comptime total = B * Self.DIM_
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[_bias_add_fwd_kernel[B, Self.DIM_]](
                in0.lt["gpu", l2d](),
                self.bias.val.lt["gpu", ld](),
                out.lt["gpu", l2d](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            var go_v = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var gi_v = TileTensor(gin.data, row_major[B, Self.DIM_]())
            # grad_in = grad_out (identity).
            for b in range(B):
                for i in range(Self.DIM_):
                    gi_v[b, i] = go_v[b, i]
            # grad_bias[i] += sum_b grad_out[b, i]
            var gb_v = TileTensor(self.bias.grd.data, row_major[Self.DIM_]())
            for b in range(B):
                for i in range(Self.DIM_):
                    gb_v[i] += go_v[b, i]
        else:
            var c = ctx.value()
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime ld = Layout.row_major(Self.DIM_)
            comptime total = B * Self.DIM_
            comptime lflat = Layout.row_major(total)
            gin.ensure_gpu(c, B * Self.DIM_)
            # grad_in = grad_out: device→device copy via flat kernel.
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[_bias_add_copy_kernel[total]](
                grad_output.lt["gpu", lflat](),
                gin.lt["gpu", lflat](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            # grad_bias[i] += sum_b grad_out[b, i] (one block per column).
            c.enqueue_function[_bias_add_dbias_kernel[B, Self.DIM_]](
                grad_output.lt["gpu", l2d](),
                self.bias.grd.lt["gpu", ld](),
                grid_dim=Self.DIM_,
                block_dim=BA_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `bias` Param).

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.DIM_](
            self.bias.val, src.bias.val, tau, ctx
        )
