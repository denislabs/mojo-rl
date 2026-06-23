"""BlockLinear[IN, OUT, BLOCKS] — block-diagonal linear layer (storage surface).

Storage twin of legacy `nn.primitives.block_linear.BlockLinear`. Matches the
DreamerV3 reference `embodied/jax/nets.py:BlockLinear`:

    kernel shape  = [BLOCKS, IN/BLOCKS, OUT/BLOCKS]
    x  reshaped   = [B, BLOCKS, IN/BLOCKS]
    out           = einsum('...ki,kio->...ko', x, kernel)  + bias[OUT]

i.e. block `k` maps input columns `[k·IPB : (k+1)·IPB]` to output columns
`[k·OPB : (k+1)·OPB]` (IPB = IN/BLOCKS, OPB = OUT/BLOCKS) — a block-diagonal
weight. `BLOCKS=1` reduces to a dense linear (`out = x·kernel + bias`).

Storage:
  * `weight: Param["weight", True,  BLOCKS·IPB·OPB]` — `kernel[k,i,o]` at
    flat offset `k·IPB·OPB + i·OPB + o` (row-major).
  * `bias:   Param["bias", False, OUT]`.

Backward (full grads; reads `forward_input[0]` for x):

    grad_weight[k,i,o] += Σ_b x[b,k·IPB+i]·go[b,k·OPB+o]
    grad_bias[j]       += Σ_b go[b,j]
    grad_x[b,k·IPB+i]   = Σ_o go[b,k·OPB+o]·kernel[k,i,o]

CPU: gather strided blocks into contiguous tiles + BLAS (Apple Accelerate)
per-block matmul (carried verbatim from legacy). GPU: one-thread-per-element
kernels (carried verbatim, transformed to LayoutTensor args).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..loss.sac import polyak_tensor


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (carried verbatim from legacy; raw-pointer args replaced by
# flat-1D LayoutTensor args to match the storage `lt` view surface).
# ──────────────────────────────────────────────────────────────────────


def _bl_forward_kernel[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int
](
    x: LayoutTensor[DT, Layout.row_major(BATCH * IN), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(BLK * (IN // BLK) * (OUT // BLK)), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    out_buf: LayoutTensor[DT, Layout.row_major(BATCH * OUT), MutAnyOrigin],
):
    comptime IPB = IN // BLK
    comptime OPB = OUT // BLK
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT:
        return
    var b = idx // OUT
    var j = idx % OUT
    var k = j // OPB
    var o = j % OPB
    var acc = rebind[Scalar[DT]](bias[j])
    var w_base = k * IPB * OPB + o
    var x_base = b * IN + k * IPB
    for i in range(IPB):
        acc += rebind[Scalar[DT]](x[x_base + i]) * rebind[Scalar[DT]](
            weight[w_base + i * OPB]
        )
    out_buf[idx] = acc


def _bl_dweight_kernel[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int
](
    x: LayoutTensor[DT, Layout.row_major(BATCH * IN), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(BATCH * OUT), MutAnyOrigin],
    grad_w: LayoutTensor[DT, Layout.row_major(BLK * (IN // BLK) * (OUT // BLK)), MutAnyOrigin],
):
    comptime IPB = IN // BLK
    comptime OPB = OUT // BLK
    var idx = Int(global_idx.x)
    if idx >= BLK * IPB * OPB:
        return
    var k = idx // (IPB * OPB)
    var rem = idx % (IPB * OPB)
    var i = rem // OPB
    var o = rem % OPB
    var in_col = k * IPB + i
    var out_col = k * OPB + o
    var acc: Scalar[DT] = 0.0
    for b in range(BATCH):
        acc += rebind[Scalar[DT]](x[b * IN + in_col]) * rebind[Scalar[DT]](
            go[b * OUT + out_col]
        )
    grad_w[idx] = rebind[Scalar[DT]](grad_w[idx]) + acc


def _bl_dbias_kernel[
    BATCH: Int, OUT: Int
](
    go: LayoutTensor[DT, Layout.row_major(BATCH * OUT), MutAnyOrigin],
    grad_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j >= OUT:
        return
    var acc: Scalar[DT] = 0.0
    for b in range(BATCH):
        acc += rebind[Scalar[DT]](go[b * OUT + j])
    grad_b[j] = rebind[Scalar[DT]](grad_b[j]) + acc


def _bl_dx_kernel[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int
](
    go: LayoutTensor[DT, Layout.row_major(BATCH * OUT), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(BLK * (IN // BLK) * (OUT // BLK)), MutAnyOrigin],
    grad_x: LayoutTensor[DT, Layout.row_major(BATCH * IN), MutAnyOrigin],
):
    comptime IPB = IN // BLK
    comptime OPB = OUT // BLK
    var idx = Int(global_idx.x)
    if idx >= BATCH * IN:
        return
    var b = idx // IN
    var col = idx % IN
    var k = col // IPB
    var i = col % IPB
    var w_base = k * IPB * OPB + i * OPB
    var go_base = b * OUT + k * OPB
    var acc: Scalar[DT] = 0.0
    for o in range(OPB):
        acc += rebind[Scalar[DT]](go[go_base + o]) * rebind[Scalar[DT]](
            weight[w_base + o]
        )
    grad_x[idx] = acc


# ──────────────────────────────────────────────────────────────────────
# BlockLinear.
# ──────────────────────────────────────────────────────────────────────


struct BlockLinear[IN: Int, OUT: Int, BLOCKS: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN)
    comptime OUT_DIM = Self.OUT
    comptime IPB = Self.IN // Self.BLOCKS
    comptime OPB = Self.OUT // Self.BLOCKS
    comptime W_SIZE = Self.BLOCKS * Self.IPB * Self.OPB
    comptime B_SIZE = Self.OUT

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert Self.IN % Self.BLOCKS == 0, (
            "BlockLinear: IN must be divisible by BLOCKS"
        )
        comptime assert Self.OUT % Self.BLOCKS == 0, (
            "BlockLinear: OUT must be divisible by BLOCKS"
        )
        var bl = Self()
        bl.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        bl.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        INIT.init_weight[target](
            bl.weight.val, Self.W_SIZE, Self.IN, Self.OUT, ctx
        )
        INIT.init_bias[target](bl.bias.val, Self.B_SIZE, ctx)
        return bl^

    # ----- Forward ---------------------------------------------------------

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
            out.ensure(B * Self.OUT)
            var in_p = in0.data.unsafe_ptr()
            var out_p = out.data.unsafe_ptr()
            var w_p = self.weight.val.data.unsafe_ptr()
            var b_p = self.bias.val.data.unsafe_ptr()
            comptime if Self.BLOCKS == 1:
                # Plain dense matmul — input/output blocks ARE the full
                # contiguous [B, IN]/[B, OUT] tiles, no gather needed.
                var input_v = TileTensor(in0.data, row_major[B, Self.IN]())
                var output_v = TileTensor(out.data, row_major[B, Self.OUT]())
                var w_tt = TileTensor(
                    self.weight.val.data, row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="cpu"](output_v, input_v, w_tt, None)
                for b in range(B):
                    var out_base = b * Self.OUT
                    for o2 in range(Self.OUT):
                        out_p[out_base + o2] = out_p[out_base + o2] + b_p[o2]
            else:
                # BLOCKS independent matmuls. The block's input/output columns
                # are STRIDED slices of [B, IN]/[B, OUT], so gather each
                # x_block into a contiguous [B, IPB] tile, run BLAS
                # `xblk @ kernel[k]`, then scatter + add bias.
                var xblk_list = List[Scalar[DT]](
                    length=B * Self.IPB, fill=Scalar[DT](0)
                )
                var oblk_list = List[Scalar[DT]](
                    length=B * Self.OPB, fill=Scalar[DT](0)
                )
                for k in range(Self.BLOCKS):
                    var in_col0 = k * Self.IPB
                    for b in range(B):
                        var xb_base = b * Self.IPB
                        var src_base = b * Self.IN + in_col0
                        for i in range(Self.IPB):
                            xblk_list[xb_base + i] = in_p[src_base + i]
                    var w_blk = k * Self.IPB * Self.OPB
                    var xblk_tt = TileTensor(
                        xblk_list, row_major[B, Self.IPB](),
                    )
                    var kernel_k_tt = TileTensor(
                        w_p + w_blk, row_major[Self.IPB, Self.OPB](),
                    )
                    var oblk_tt = TileTensor(
                        oblk_list, row_major[B, Self.OPB](),
                    )
                    max_matmul[target="cpu"](oblk_tt, xblk_tt, kernel_k_tt, None)
                    var out_col0 = k * Self.OPB
                    for b in range(B):
                        var ob_base = b * Self.OPB
                        var dst_base = b * Self.OUT + out_col0
                        for o2 in range(Self.OPB):
                            out_p[dst_base + o2] = (
                                oblk_list[ob_base + o2] + b_p[out_col0 + o2]
                            )
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT)
            comptime n_blk = (B * Self.OUT + TPB - 1) // TPB
            comptime k_fwd = _bl_forward_kernel[B, Self.IN, Self.OUT, Self.BLOCKS]
            c.enqueue_function[k_fwd](
                in0.lt["gpu", Layout.row_major(B * Self.IN)](),
                self.weight.val.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.bias.val.lt["gpu", Layout.row_major(Self.OUT)](),
                out.lt["gpu", Layout.row_major(B * Self.OUT)](),
                grid_dim=n_blk,
                block_dim=TPB,
            )

    # ----- Backward --------------------------------------------------------

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
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.IN)
            var go_p = grad_output.data.unsafe_ptr()
            var gw_p = self.weight.grd.data.unsafe_ptr()
            var gb_p = self.bias.grd.data.unsafe_ptr()
            var x_p = fin.data.unsafe_ptr()
            var w_p = self.weight.val.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var grad_output_v = TileTensor(
                grad_output.data, row_major[B, Self.OUT]()
            )

            # grad_bias[j] += Σ_b go[b,j]
            for j in range(Self.OUT):
                var accb: Scalar[DT] = 0.0
                for b in range(B):
                    accb += go_p[b * Self.OUT + j]
                gb_p[j] += accb

            # grad_weight[k] += x_blockᵀ @ go_block, via BLAS.
            comptime if Self.BLOCKS == 1:
                var cT_list = List[Scalar[DT]](
                    length=B * Self.IN, fill=Scalar[DT](0)
                )
                var dW_list = List[Scalar[DT]](
                    length=Self.IN * Self.OUT, fill=Scalar[DT](0)
                )
                for b in range(B):
                    for i in range(Self.IN):
                        cT_list[i * B + b] = x_p[b * Self.IN + i]
                var cT_tt = TileTensor(cT_list, row_major[Self.IN, B]())
                var dW_tt = TileTensor(
                    dW_list, row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="cpu"](dW_tt, cT_tt, grad_output_v, None)
                for idx in range(Self.IN * Self.OUT):
                    gw_p[idx] = gw_p[idx] + dW_list[idx]
            else:
                var xT_list = List[Scalar[DT]](
                    length=Self.IPB * B, fill=Scalar[DT](0)
                )
                var gob_list = List[Scalar[DT]](
                    length=B * Self.OPB, fill=Scalar[DT](0)
                )
                var dW_list = List[Scalar[DT]](
                    length=Self.IPB * Self.OPB, fill=Scalar[DT](0)
                )
                for k in range(Self.BLOCKS):
                    var in_col0 = k * Self.IPB
                    var out_col0 = k * Self.OPB
                    for b in range(B):
                        var x_src = b * Self.IN + in_col0
                        for i in range(Self.IPB):
                            xT_list[i * B + b] = x_p[x_src + i]
                        var go_src = b * Self.OUT + out_col0
                        var gob_dst = b * Self.OPB
                        for o2 in range(Self.OPB):
                            gob_list[gob_dst + o2] = go_p[go_src + o2]
                    var xT_tt = TileTensor(xT_list, row_major[Self.IPB, B]())
                    var gob_tt = TileTensor(
                        gob_list, row_major[B, Self.OPB](),
                    )
                    var dW_tt = TileTensor(
                        dW_list, row_major[Self.IPB, Self.OPB](),
                    )
                    max_matmul[target="cpu"](dW_tt, xT_tt, gob_tt, None)
                    var w_blk = k * Self.IPB * Self.OPB
                    for idx in range(Self.IPB * Self.OPB):
                        gw_p[w_blk + idx] = gw_p[w_blk + idx] + dW_list[idx]

            # grad_x_block = go_block @ kernel[k]ᵀ
            comptime if Self.BLOCKS == 1:
                var grad_input_v = TileTensor(gin.data, row_major[B, Self.IN]())
                var w_tt = TileTensor(
                    self.weight.val.data, row_major[Self.IN, Self.OUT](),
                )
                max_matmul[transpose_b=True, target="cpu"](
                    grad_input_v, grad_output_v, w_tt, None,
                )
            else:
                var gob_list2 = List[Scalar[DT]](
                    length=B * Self.OPB, fill=Scalar[DT](0)
                )
                var gxb_list = List[Scalar[DT]](
                    length=B * Self.IPB, fill=Scalar[DT](0)
                )
                for k in range(Self.BLOCKS):
                    var out_col0 = k * Self.OPB
                    for b in range(B):
                        var go_src = b * Self.OUT + out_col0
                        var gob_dst = b * Self.OPB
                        for o2 in range(Self.OPB):
                            gob_list2[gob_dst + o2] = go_p[go_src + o2]
                    var w_blk = k * Self.IPB * Self.OPB
                    var gob_tt = TileTensor(
                        gob_list2, row_major[B, Self.OPB](),
                    )
                    var kernel_k_tt = TileTensor(
                        w_p + w_blk, row_major[Self.IPB, Self.OPB](),
                    )
                    var gxb_tt = TileTensor(
                        gxb_list, row_major[B, Self.IPB](),
                    )
                    max_matmul[transpose_b=True, target="cpu"](
                        gxb_tt, gob_tt, kernel_k_tt, None,
                    )
                    var in_col0 = k * Self.IPB
                    for b in range(B):
                        var gxb_src = b * Self.IPB
                        var dst = b * Self.IN + in_col0
                        for i in range(Self.IPB):
                            gi_p[dst + i] = gxb_list[gxb_src + i]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN)
            # grad_weight
            comptime n_w = (Self.W_SIZE + TPB - 1) // TPB
            comptime k_dw = _bl_dweight_kernel[B, Self.IN, Self.OUT, Self.BLOCKS]
            c.enqueue_function[k_dw](
                fin.lt["gpu", Layout.row_major(B * Self.IN)](),
                grad_output.lt["gpu", Layout.row_major(B * Self.OUT)](),
                self.weight.grd.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                grid_dim=n_w,
                block_dim=TPB,
            )
            # grad_bias
            comptime n_b = (Self.OUT + TPB - 1) // TPB
            comptime k_db = _bl_dbias_kernel[B, Self.OUT]
            c.enqueue_function[k_db](
                grad_output.lt["gpu", Layout.row_major(B * Self.OUT)](),
                self.bias.grd.lt["gpu", Layout.row_major(Self.OUT)](),
                grid_dim=n_b,
                block_dim=TPB,
            )
            # grad_x
            comptime n_x = (B * Self.IN + TPB - 1) // TPB
            comptime k_dx = _bl_dx_kernel[B, Self.IN, Self.OUT, Self.BLOCKS]
            c.enqueue_function[k_dx](
                grad_output.lt["gpu", Layout.row_major(B * Self.OUT)](),
                self.weight.val.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                gin.lt["gpu", Layout.row_major(B * Self.IN)](),
                grid_dim=n_x,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `weight` + `bias` Params).

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.W_SIZE](
            self.weight.val, src.weight.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.bias.val, src.bias.val, tau, ctx
        )
