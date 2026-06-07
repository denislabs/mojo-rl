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

from std.memory import alloc
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

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
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


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
            var ctx_v = require_ctx["BlockLinear.make[target='gpu']"](ctx)
            bl.weight = Param["weight", True, Self.W_SIZE].make_gpu(ctx_v)
            bl.bias = Param["bias", False, Self.B_SIZE].make_gpu(ctx_v)
            var w_host = ctx_v.enqueue_create_host_buffer[DT](Self.W_SIZE)
            var b_host = ctx_v.enqueue_create_host_buffer[DT](Self.B_SIZE)
            ctx_v.synchronize()
            INIT.init_weight(w_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT)
            INIT.init_bias(b_host.unsafe_ptr(), Self.B_SIZE)
            ctx_v.enqueue_copy(bl.weight.val.dev.value(), w_host)
            ctx_v.enqueue_copy(bl.bias.val.dev.value(), b_host)
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
            comptime if Self.BLOCKS == 1:
                # Plain dense matmul — input/output blocks ARE the full
                # contiguous [BATCH, IN]/[BATCH, OUT] tiles, no gather needed.
                var w_tt = TileTensor(
                    self.weight.val.cpu, row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="cpu"](output_v, input_v, w_tt, None)
                for b in range(BATCH):
                    var out_base = b * Self.OUT
                    for o in range(Self.OUT):
                        out_p[out_base + o] = out_p[out_base + o] + b_p[o]
            else:
                # BLOCKS independent matmuls. The block's input/output columns
                # are STRIDED slices of [BATCH, IN]/[BATCH, OUT], so gather each
                # x_block into a contiguous [BATCH, IPB] tile, run BLAS (Apple
                # Accelerate) `xblk @ kernel[k]`, then scatter + add bias.
                # kernel[k] = w_p[w_blk + i*OPB + o] is already a row-major
                # [IPB, OPB] tile at offset w_blk. Scratch reused per block.
                var xblk_buf = alloc[Scalar[DT]](BATCH * Self.IPB)
                var oblk_buf = alloc[Scalar[DT]](BATCH * Self.OPB)
                for k in range(Self.BLOCKS):
                    var in_col0 = k * Self.IPB
                    for b in range(BATCH):
                        var xb_base = b * Self.IPB
                        var src_base = b * Self.IN + in_col0
                        for i in range(Self.IPB):
                            xblk_buf[xb_base + i] = in_p[src_base + i]
                    var w_blk = k * Self.IPB * Self.OPB
                    var xblk_tt = TileTensor(
                        xblk_buf, row_major[BATCH, Self.IPB](),
                    )
                    var kernel_k_tt = TileTensor(
                        w_p + w_blk, row_major[Self.IPB, Self.OPB](),
                    )
                    var oblk_tt = TileTensor(
                        oblk_buf, row_major[BATCH, Self.OPB](),
                    )
                    max_matmul[target="cpu"](oblk_tt, xblk_tt, kernel_k_tt, None)
                    var out_col0 = k * Self.OPB
                    for b in range(BATCH):
                        var ob_base = b * Self.OPB
                        var dst_base = b * Self.OUT + out_col0
                        for o in range(Self.OPB):
                            out_p[dst_base + o] = (
                                oblk_buf[ob_base + o] + b_p[out_col0 + o]
                            )
                oblk_buf.free()
                xblk_buf.free()
        else:
            var ctx = self.ts.ctx.value()
            var w_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.weight.val.dev.value().unsafe_ptr()
            )
            var b_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.bias.val.dev.value().unsafe_ptr()
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
        """Combined backward (S7) — the two phases in fixed order. Single
        source of truth for direct callers; Sequential calls the phases
        directly so the param-before-input order is the orchestrator's."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        self.vjp_param_grads[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output
        )
        self.vjp_grad_input[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output, *grad_inputs
        )

    def vjp_param_grads[
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
    ) raises:
        """Phase 1 (S7) — was LOOP 1. grad_weight/grad_bias (mode=all),
        which read `_cached_input_ptr` (the cached input x). MUST precede
        `vjp_grad_input` since x aliases the slab grad_input clobbers."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        comptime if mode == "all":
            assert_tag_for["BlockLinear", target](self.ts.target_tag)
            var grad_output_v = typed_view[BATCH, Self.OUT](grad_output)
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )

            comptime if target == "cpu":
                var gw_p = self.weight.grad_unsafe_ptr_cpu()
                var gb_p = self.bias.grad_unsafe_ptr_cpu()
                var x_p = self._cached_input_ptr.value()
                # grad_bias[j] += Σ_b go[b,j] — cheap, keep scalar.
                for j in range(Self.OUT):
                    var accb: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        accb += go_p[b * Self.OUT + j]
                    gb_p[j] += accb
                # grad_weight[k] += x_block^T @ go_block, via BLAS (Apple
                # Accelerate). x_block/go_block are strided column-slices →
                # gather x_blockᵀ [IPB, BATCH] (transpose during gather) and
                # go_block [BATCH, OPB], matmul into a [IPB, OPB] temp, then
                # ADD into grad_weight[k] at offset w_blk. Scratch reused.
                comptime if Self.BLOCKS == 1:
                    # Dense: x is already contiguous [BATCH, IN]; transpose
                    # into cT [IN, BATCH] and matmul straight into the temp.
                    var cT_buf = alloc[Scalar[DT]](BATCH * Self.IN)
                    var dW_buf = alloc[Scalar[DT]](Self.IN * Self.OUT)
                    for b in range(BATCH):
                        for i in range(Self.IN):
                            cT_buf[i * BATCH + b] = x_p[b * Self.IN + i]
                    var cT_tt = TileTensor(
                        cT_buf, row_major[Self.IN, BATCH](),
                    )
                    var dW_tt = TileTensor(
                        dW_buf, row_major[Self.IN, Self.OUT](),
                    )
                    max_matmul[target="cpu"](dW_tt, cT_tt, grad_output_v, None)
                    for idx in range(Self.IN * Self.OUT):
                        gw_p[idx] = gw_p[idx] + dW_buf[idx]
                    dW_buf.free()
                    cT_buf.free()
                else:
                    var xT_buf = alloc[Scalar[DT]](Self.IPB * BATCH)
                    var gob_buf = alloc[Scalar[DT]](BATCH * Self.OPB)
                    var dW_buf = alloc[Scalar[DT]](Self.IPB * Self.OPB)
                    for k in range(Self.BLOCKS):
                        var in_col0 = k * Self.IPB
                        var out_col0 = k * Self.OPB
                        # Gather x_blockᵀ [IPB, BATCH] and go_block [BATCH, OPB].
                        for b in range(BATCH):
                            var x_src = b * Self.IN + in_col0
                            for i in range(Self.IPB):
                                xT_buf[i * BATCH + b] = x_p[x_src + i]
                            var go_src = b * Self.OUT + out_col0
                            var gob_dst = b * Self.OPB
                            for o in range(Self.OPB):
                                gob_buf[gob_dst + o] = go_p[go_src + o]
                        var xT_tt = TileTensor(
                            xT_buf, row_major[Self.IPB, BATCH](),
                        )
                        var gob_tt = TileTensor(
                            gob_buf, row_major[BATCH, Self.OPB](),
                        )
                        var dW_tt = TileTensor(
                            dW_buf, row_major[Self.IPB, Self.OPB](),
                        )
                        max_matmul[target="cpu"](dW_tt, xT_tt, gob_tt, None)
                        var w_blk = k * Self.IPB * Self.OPB
                        for idx in range(Self.IPB * Self.OPB):
                            gw_p[w_blk + idx] = gw_p[w_blk + idx] + dW_buf[idx]
                    dW_buf.free()
                    gob_buf.free()
                    xT_buf.free()
            else:
                var ctx = self.ts.ctx.value()
                var gw_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.weight.grd.dev.value().unsafe_ptr()
                )
                var gb_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.bias.grd.dev.value().unsafe_ptr()
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

    def vjp_grad_input[
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
        """Phase 2 (S7) — was LOOP 2. grad_x (clobbers the aliased input
        slab — safe because phase 1 already read it). Runs in both modes."""
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
            # grad_x_block = go_block @ kernel[k]ᵀ, [BATCH,OPB]@[OPB,IPB], via
            # transpose_b BLAS. kernel[k] is row-major [IPB, OPB] at w_blk, so
            # transpose_b gives go_block @ kernelᵀ → [BATCH, IPB]. Gather
            # go_block contiguous, matmul into a temp, scatter into grad_x.
            var w_p = self.weight.value_unsafe_ptr_cpu()
            comptime if Self.BLOCKS == 1:
                var w_tt = TileTensor(
                    self.weight.val.cpu, row_major[Self.IN, Self.OUT](),
                )
                max_matmul[transpose_b=True, target="cpu"](
                    grad_input_v, grad_output_v, w_tt, None,
                )
            else:
                var gob_buf2 = alloc[Scalar[DT]](BATCH * Self.OPB)
                var gxb_buf = alloc[Scalar[DT]](BATCH * Self.IPB)
                for k in range(Self.BLOCKS):
                    var out_col0 = k * Self.OPB
                    for b in range(BATCH):
                        var go_src = b * Self.OUT + out_col0
                        var gob_dst = b * Self.OPB
                        for o in range(Self.OPB):
                            gob_buf2[gob_dst + o] = go_p[go_src + o]
                    var w_blk = k * Self.IPB * Self.OPB
                    var gob_tt = TileTensor(
                        gob_buf2, row_major[BATCH, Self.OPB](),
                    )
                    var kernel_k_tt = TileTensor(
                        w_p + w_blk, row_major[Self.IPB, Self.OPB](),
                    )
                    var gxb_tt = TileTensor(
                        gxb_buf, row_major[BATCH, Self.IPB](),
                    )
                    max_matmul[transpose_b=True, target="cpu"](
                        gxb_tt, gob_tt, kernel_k_tt, None,
                    )
                    var in_col0 = k * Self.IPB
                    for b in range(BATCH):
                        var gxb_src = b * Self.IPB
                        var dst = b * Self.IN + in_col0
                        for i in range(Self.IPB):
                            gi_p[dst + i] = gxb_buf[gxb_src + i]
                gxb_buf.free()
                gob_buf2.free()
        else:
            var ctx = self.ts.ctx.value()
            var w_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.weight.val.dev.value().unsafe_ptr()
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
