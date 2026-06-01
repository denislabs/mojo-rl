"""BlockLinear[IN, OUT, BLOCKS] — block-diagonal linear layer.

Matches the DreamerV3 reference `embodied/jax/nets.py:BlockLinear`:

    kernel shape  = [BLOCKS, IN/BLOCKS, OUT/BLOCKS]
    x  reshaped   = [B, BLOCKS, IN/BLOCKS]
    out           = einsum('...ki,kio->...ko', x, kernel)  + bias[OUT]

i.e. block `k` maps input columns `[k·IPB : (k+1)·IPB]` to output columns
`[k·OPB : (k+1)·OPB]` (IPB = IN/BLOCKS, OPB = OUT/BLOCKS) — a block-diagonal
weight. Used by the RSSM `_core` (dynhid / dyngru) and the Decoder space
head. `BLOCKS=1` reduces to a dense linear (`out = x·kernel + bias`).

Storage:
  * `weight: Param["weight", True,  BLOCKS·IPB·OPB]` — `kernel[k,i,o]` at
    flat offset `k·IPB·OPB + i·OPB + o` (row-major, matches the reference
    ravel + the jax fixture).
  * `bias:   Param["bias", False, OUT]`.

Backward (same `_cached_input_ptr` input-alias as Linear; param grads
computed BEFORE grad_input is written, since the input slab aliases the
grad_input slab under Sequential):

    grad_weight[k,i,o] += Σ_b x[b,k·IPB+i]·go[b,k·OPB+o]
    grad_bias[j]       += Σ_b go[b,j]
    grad_x[b,k·IPB+i]   = Σ_o go[b,k·OPB+o]·kernel[k,i,o]

CPU nested loops + GPU one-thread-per-element kernels. DreamerV3-exact
weight init (trunc_normal scaled) is deferred to the consumer PR; `make`
forwards the supplied `INIT` (treated like a dense layer of fan IN→OUT).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
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
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


def _bl_forward_kernel[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int
](
    x: UnsafePointer[Scalar[DT], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DT], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
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
    var acc = bias[j]
    var w_base = k * IPB * OPB + o
    var x_base = b * IN + k * IPB
    for i in range(IPB):
        acc += x[x_base + i] * weight[w_base + i * OPB]
    out_buf[idx] = acc


def _bl_dweight_kernel[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int
](
    x: UnsafePointer[Scalar[DT], MutAnyOrigin],
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_w: UnsafePointer[Scalar[DT], MutAnyOrigin],
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
        acc += x[b * IN + in_col] * go[b * OUT + out_col]
    grad_w[idx] = grad_w[idx] + acc


def _bl_dbias_kernel[
    BATCH: Int, OUT: Int
](
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_b: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j >= OUT:
        return
    var acc: Scalar[DT] = 0.0
    for b in range(BATCH):
        acc += go[b * OUT + j]
    grad_b[j] = grad_b[j] + acc


def _bl_dx_kernel[
    BATCH: Int, IN: Int, OUT: Int, BLK: Int
](
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    weight: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_x: UnsafePointer[Scalar[DT], MutAnyOrigin],
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
        acc += go[go_base + o] * weight[w_base + o]
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

    @staticmethod
    def display_label() -> String:
        return String("BlockLinear")

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]

    var _cached_input_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    var ts: TargetStorage

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self._cached_input_ptr = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "BlockLinear: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.IN % Self.BLOCKS == 0, (
            "BlockLinear: IN must be divisible by BLOCKS"
        )
        comptime assert Self.OUT % Self.BLOCKS == 0, (
            "BlockLinear: OUT must be divisible by BLOCKS"
        )
        var bl = Self()
        comptime if target == "cpu":
            bl.weight = Param["weight", True, Self.W_SIZE].make_cpu()
            bl.bias = Param["bias", False, Self.B_SIZE].make_cpu()
            INIT.init_weight(
                bl.weight.value_unsafe_ptr_cpu(),
                Self.W_SIZE, Self.IN, Self.OUT,
            )
            INIT.init_bias(bl.bias.value_unsafe_ptr_cpu(), Self.B_SIZE)
            bl.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("BlockLinear.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            bl.weight = Param["weight", True, Self.W_SIZE].make_gpu(ctx_v)
            bl.bias = Param["bias", False, Self.B_SIZE].make_gpu(ctx_v)
            var w_host = ctx_v.enqueue_create_host_buffer[DT](Self.W_SIZE)
            var b_host = ctx_v.enqueue_create_host_buffer[DT](Self.B_SIZE)
            ctx_v.synchronize()
            INIT.init_weight(w_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT)
            INIT.init_bias(b_host.unsafe_ptr(), Self.B_SIZE)
            ctx_v.enqueue_copy(bl.weight.value_dev.value(), w_host)
            ctx_v.enqueue_copy(bl.bias.value_dev.value(), b_host)
            ctx_v.synchronize()
            bl.ts = TargetStorage.make_gpu(ctx_v)
        return bl^

    # ----- Forward ---------------------------------------------------------

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
        assert_tag_for["BlockLinear", target](self.ts.target_tag)
        var input_v = typed_view[BATCH, Self.IN](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT](output)
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
        self._cached_input_ptr = in_p

        comptime if target == "cpu":
            var w_p = self.weight.value_unsafe_ptr_cpu()
            var b_p = self.bias.value_unsafe_ptr_cpu()
            for b in range(BATCH):
                for k in range(Self.BLOCKS):
                    var x_base = b * Self.IN + k * Self.IPB
                    var w_blk = k * Self.IPB * Self.OPB
                    var out_base = b * Self.OUT + k * Self.OPB
                    for o in range(Self.OPB):
                        var acc = b_p[k * Self.OPB + o]
                        for i in range(Self.IPB):
                            acc += w_p[w_blk + i * Self.OPB + o] * in_p[x_base + i]
                        out_p[out_base + o] = acc
        else:
            var ctx = self.ts.ctx.value()
            var w_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.weight.value_dev.value().unsafe_ptr()
            )
            var b_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.bias.value_dev.value().unsafe_ptr()
            )
            comptime n_blk = (BATCH * Self.OUT + TPB - 1) // TPB
            comptime k_fwd = _bl_forward_kernel[
                BATCH, Self.IN, Self.OUT, Self.BLOCKS
            ]
            ctx.enqueue_function[k_fwd](
                in_p, w_p, b_p, out_p, grid_dim=n_blk, block_dim=TPB,
            )

    # ----- Backward --------------------------------------------------------

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
        assert_tag_for["BlockLinear", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN](grad_inputs[0])
        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_output_v.ptr
        )
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_input_v.ptr
        )

        comptime if target == "cpu":
            # Param grads FIRST — they read `_cached_input_ptr`, which
            # aliases the input slab that grad_input is about to clobber.
            comptime if mode == "all":
                var gw_p = self.weight.grad_unsafe_ptr_cpu()
                var gb_p = self.bias.grad_unsafe_ptr_cpu()
                var x_p = self._cached_input_ptr.value()
                for k in range(Self.BLOCKS):
                    var w_blk = k * Self.IPB * Self.OPB
                    for i in range(Self.IPB):
                        var in_col = k * Self.IPB + i
                        for o in range(Self.OPB):
                            var out_col = k * Self.OPB + o
                            var acc: Scalar[DT] = 0.0
                            for b in range(BATCH):
                                acc += (
                                    x_p[b * Self.IN + in_col]
                                    * go_p[b * Self.OUT + out_col]
                                )
                            gw_p[w_blk + i * Self.OPB + o] += acc
                for j in range(Self.OUT):
                    var accb: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        accb += go_p[b * Self.OUT + j]
                    gb_p[j] += accb
            # grad_x last.
            var w_p = self.weight.value_unsafe_ptr_cpu()
            for b in range(BATCH):
                for k in range(Self.BLOCKS):
                    var w_blk = k * Self.IPB * Self.OPB
                    var go_base = b * Self.OUT + k * Self.OPB
                    for i in range(Self.IPB):
                        var acc: Scalar[DT] = 0.0
                        for o in range(Self.OPB):
                            acc += go_p[go_base + o] * w_p[w_blk + i * Self.OPB + o]
                        gi_p[b * Self.IN + k * Self.IPB + i] = acc
        else:
            var ctx = self.ts.ctx.value()
            var w_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.weight.value_dev.value().unsafe_ptr()
            )
            comptime if mode == "all":
                var gw_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.weight.grad_dev.value().unsafe_ptr()
                )
                var gb_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.bias.grad_dev.value().unsafe_ptr()
                )
                var x_p = self._cached_input_ptr.value()
                comptime n_w = (Self.W_SIZE + TPB - 1) // TPB
                comptime k_dw = _bl_dweight_kernel[
                    BATCH, Self.IN, Self.OUT, Self.BLOCKS
                ]
                ctx.enqueue_function[k_dw](
                    x_p, go_p, gw_p, grid_dim=n_w, block_dim=TPB,
                )
                comptime n_b = (Self.OUT + TPB - 1) // TPB
                comptime k_db = _bl_dbias_kernel[BATCH, Self.OUT]
                ctx.enqueue_function[k_db](
                    go_p, gb_p, grid_dim=n_b, block_dim=TPB,
                )
            comptime n_x = (BATCH * Self.IN + TPB - 1) // TPB
            comptime k_dx = _bl_dx_kernel[
                BATCH, Self.IN, Self.OUT, Self.BLOCKS
            ]
            ctx.enqueue_function[k_dx](
                go_p, w_p, gi_p, grid_dim=n_x, block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["BlockLinear", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["BlockLinear", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
