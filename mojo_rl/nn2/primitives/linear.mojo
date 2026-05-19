"""Linear[IN, OUT] — fully-connected layer, target chosen per method call.

Phase 2.4: `target` is a comptime method param, not a struct param.
Storage holds both CPU `List` and GPU `Optional[DeviceBuffer]`; only one
set is populated, indicated by `_target_tag`.

  - Default `__init__()` produces empty placeholders + tag=UNINIT.
    This is what makes `Linear[IN, OUT]` `Defaultable` — enables
    `Tuple[*MODULES]()` default-construction in `Sequential`.
  - `Linear[IN, OUT].make[target, INIT]()` (CPU) or
    `Linear[IN, OUT].make[target, INIT](ctx)` (GPU) populates the
    matching fieldset and stamps `_target_tag`.
  - Every method that touches storage takes `[target]` and opens with a
    tag check (`_assert_tag[target]`), then `comptime if target=="cpu"`
    branches.

Memory overhead per instance: ~50–100 bytes (List placeholders + None
Optionals). Same as Phase 2.1 — the runtime branch is comptime-erased.
"""

from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major
from linalg.matmul import matmul as max_matmul

from ..constants import DT
from ..core import (
    Module, ParamVisitor, Initializer,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels — module-level so enqueue_function can bind them.
# ──────────────────────────────────────────────────────────────────────────

def _bias_add_kernel[
    BATCH: Int, OUT: Int,
](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * OUT
    if idx < total:
        var col = idx % OUT
        var b_val = rebind[Scalar[DT]](bias[col])
        output.ptr[idx] = output.ptr[idx] + b_val


def _grad_w_accum_kernel[
    BATCH: Int, IN: Int, OUT: Int,
](
    cache: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    grad_w: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = IN * OUT
    if idx < total:
        var i = idx // OUT
        var j = idx % OUT
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += cache.ptr[b * IN + i] * grad_output.ptr[b * OUT + j]
        grad_w.ptr[i * OUT + j] = grad_w.ptr[i * OUT + j] + s


def _grad_bias_kernel[
    BATCH: Int, OUT: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    grad_bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < OUT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += grad_output.ptr[b * OUT + j]
        grad_bias.ptr[j] = grad_bias.ptr[j] + s


# ──────────────────────────────────────────────────────────────────────────
# Linear — method-level target.
# ──────────────────────────────────────────────────────────────────────────

struct Linear[IN: Int, OUT: Int](Module):
    comptime IN_DIM = Self.IN
    comptime OUT_DIM = Self.OUT
    comptime W_SIZE = Self.IN * Self.OUT
    comptime B_SIZE = Self.OUT

    # CPU storage (populated when _target_tag == TARGET_CPU)
    var weight: List[Scalar[DT]]
    var bias:   List[Scalar[DT]]
    var grad_w: List[Scalar[DT]]
    var grad_b: List[Scalar[DT]]
    var cache:  List[Scalar[DT]]

    # GPU storage (Some when _target_tag == TARGET_GPU)
    var weight_dev: Optional[DeviceBuffer[DT]]
    var bias_dev:   Optional[DeviceBuffer[DT]]
    var grad_w_dev: Optional[DeviceBuffer[DT]]
    var grad_b_dev: Optional[DeviceBuffer[DT]]
    var cache_dev:  Optional[DeviceBuffer[DT]]
    var cache_dev_n: Int
    var ctx: Optional[DeviceContext]

    var _target_tag: Int8

    # ------------------------------------------------------------------
    # Defaultable: empty placeholders + UNINIT tag.
    # ------------------------------------------------------------------

    def __init__(out self):
        self.weight = List[Scalar[DT]]()
        self.bias   = List[Scalar[DT]]()
        self.grad_w = List[Scalar[DT]]()
        self.grad_b = List[Scalar[DT]]()
        self.cache  = List[Scalar[DT]]()
        self.weight_dev = None
        self.bias_dev   = None
        self.grad_w_dev = None
        self.grad_b_dev = None
        self.cache_dev  = None
        self.cache_dev_n = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT

    # ------------------------------------------------------------------
    # make[target, INIT] — populates storage and stamps tag.
    # ------------------------------------------------------------------

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. Use `.make[target='gpu', INIT](ctx)` for GPU."""
        comptime assert target == "cpu", (
            "Linear.make[target='gpu', INIT] requires a DeviceContext"
        )
        var lin = Self()
        lin.weight = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        lin.bias   = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        lin.grad_w = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        lin.grad_b = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        INIT.init_weight(lin.weight.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT)
        INIT.init_bias(lin.bias.unsafe_ptr(), Self.B_SIZE)
        lin._target_tag = TARGET_CPU
        return lin^

    @staticmethod
    def make[target: StaticString, INIT: Initializer](ctx: DeviceContext) raises -> Self:
        """GPU factory."""
        comptime assert target == "gpu", (
            "Linear.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var lin = Self()
        var w_dev  = ctx.enqueue_create_buffer[DT](Self.W_SIZE)
        var b_dev  = ctx.enqueue_create_buffer[DT](Self.B_SIZE)
        var gw_dev = ctx.enqueue_create_buffer[DT](Self.W_SIZE)
        var gb_dev = ctx.enqueue_create_buffer[DT](Self.B_SIZE)
        var c_dev  = ctx.enqueue_create_buffer[DT](1)
        gw_dev.enqueue_fill(0.0)
        gb_dev.enqueue_fill(0.0)
        # Init weights/biases on host via INIT, then upload.
        var w_host = ctx.enqueue_create_host_buffer[DT](Self.W_SIZE)
        var b_host = ctx.enqueue_create_host_buffer[DT](Self.B_SIZE)
        ctx.synchronize()
        INIT.init_weight(w_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT)
        INIT.init_bias(b_host.unsafe_ptr(), Self.B_SIZE)
        ctx.enqueue_copy(w_dev, w_host)
        ctx.enqueue_copy(b_dev, b_host)
        ctx.synchronize()
        lin.weight_dev = w_dev^
        lin.bias_dev   = b_dev^
        lin.grad_w_dev = gw_dev^
        lin.grad_b_dev = gb_dev^
        lin.cache_dev  = c_dev^
        lin.cache_dev_n = 0
        lin.ctx = ctx
        lin._target_tag = TARGET_GPU
        return lin^

    # ------------------------------------------------------------------
    # Internal: tag-mismatch guard.
    # ------------------------------------------------------------------

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Linear: method called with [target='" + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_cache_cpu(mut self, batch: Int):
        var needed = batch * Self.IN
        if len(self.cache) < needed:
            self.cache.resize(needed, 0.0)

    def _ensure_cache_dev(mut self, needed: Int) raises:
        if self.cache_dev_n < needed:
            self.cache_dev = self.ctx.value().enqueue_create_buffer[DT](needed)
            self.cache_dev_n = needed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert input.flat_rank  == 2, "input must be rank-2 [BATCH, IN]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, OUT]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_cache_cpu(BATCH)
            var w = TileTensor(self.weight, row_major[Self.IN, Self.OUT]())
            var b = TileTensor(self.bias, row_major[Self.OUT]())
            var c = TileTensor(self.cache, row_major[BATCH, Self.IN]())
            for bi in range(BATCH):
                for j in range(Self.OUT):
                    var acc = b[j]
                    for i in range(Self.IN):
                        acc += input[bi, i] * w[i, j]
                    output[bi, j] = acc
                for i in range(Self.IN):
                    c[bi, i] = input[bi, i]
        else:
            var ctx = self.ctx.value()
            self._ensure_cache_dev(BATCH * Self.IN)
            # Rebind caller-supplied tensor origins to MutAnyOrigin so we
            # can feed them into kernel + max-kernels APIs that expect it.
            var input_w  = rebind[TileTensor[DT, LIN,  MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)
            ctx.enqueue_copy(
                dst_buf=DeviceBuffer[DT](ctx, self.cache_dev.value().unsafe_ptr(),
                                         BATCH * Self.IN, owning=False),
                src_buf=DeviceBuffer[DT](ctx, input_w.ptr,
                                         BATCH * Self.IN, owning=False),
            )
            var weight_tt = TileTensor(self.weight_dev.value(),
                                       row_major[Self.IN, Self.OUT]())
            max_matmul[target="gpu"](output_w, input_w, weight_tt,
                                     DeviceContextPtr(ctx))
            comptime out_layout  = Layout.row_major(BATCH, Self.OUT)
            comptime bias_layout = Layout.row_major(Self.OUT)
            var output_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](output_w.ptr)
            var bias_lt   = LayoutTensor[DT, bias_layout, MutAnyOrigin](self.bias_dev.value())
            comptime TPB = 128
            comptime n_blocks_ba = (BATCH * Self.OUT + TPB - 1) // TPB
            comptime ba_kernel = _bias_add_kernel[BATCH, Self.OUT]
            ctx.enqueue_function[ba_kernel](
                output_lt, bias_lt, grid_dim=n_blocks_ba, block_dim=TPB,
            )

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var w  = TileTensor(self.weight, row_major[Self.IN, Self.OUT]())
            var gw = TileTensor(self.grad_w, row_major[Self.IN, Self.OUT]())
            var gb = TileTensor(self.grad_b, row_major[Self.OUT]())
            var c  = TileTensor(self.cache,  row_major[BATCH, Self.IN]())
            for bi in range(BATCH):
                for i in range(Self.IN):
                    var acc: Scalar[DT] = 0.0
                    for j in range(Self.OUT):
                        acc += grad_output[bi, j] * w[i, j]
                    grad_input[bi, i] = acc
            for i in range(Self.IN):
                for j in range(Self.OUT):
                    var acc: Scalar[DT] = 0.0
                    for bi in range(BATCH):
                        acc += c[bi, i] * grad_output[bi, j]
                    gw[i, j] = gw[i, j] + acc
            for j in range(Self.OUT):
                var acc: Scalar[DT] = 0.0
                for bi in range(BATCH):
                    acc += grad_output[bi, j]
                gb[j] = gb[j] + acc
        else:
            var ctx = self.ctx.value()
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](grad_output)
            var grad_input_w  = rebind[TileTensor[DT, LGI, MutAnyOrigin]](grad_input)
            var weight_tt = TileTensor(self.weight_dev.value(),
                                       row_major[Self.IN, Self.OUT]())
            max_matmul[transpose_b=True, target="gpu"](
                grad_input_w, grad_output_w, weight_tt, DeviceContextPtr(ctx)
            )
            comptime cache_layout = Layout.row_major(BATCH, Self.IN)
            comptime go_layout    = Layout.row_major(BATCH, Self.OUT)
            comptime gw_layout    = Layout.row_major(Self.IN, Self.OUT)
            var cache_lt = LayoutTensor[DT, cache_layout, MutAnyOrigin](self.cache_dev.value())
            var go_lt    = LayoutTensor[DT, go_layout, MutAnyOrigin](grad_output_w.ptr)
            var gw_lt    = LayoutTensor[DT, gw_layout, MutAnyOrigin](self.grad_w_dev.value())
            comptime TPB = 128
            comptime n_blocks_gw = (Self.W_SIZE + TPB - 1) // TPB
            comptime gw_kernel = _grad_w_accum_kernel[BATCH, Self.IN, Self.OUT]
            ctx.enqueue_function[gw_kernel](
                cache_lt, go_lt, gw_lt, grid_dim=n_blocks_gw, block_dim=TPB,
            )
            comptime gb_layout = Layout.row_major(Self.OUT)
            var gb_lt = LayoutTensor[DT, gb_layout, MutAnyOrigin](self.grad_b_dev.value())
            comptime n_blocks_gb = (Self.OUT + TPB - 1) // TPB
            comptime gb_kernel = _grad_bias_kernel[BATCH, Self.OUT]
            ctx.enqueue_function[gb_kernel](
                go_lt, gb_lt, grid_dim=n_blocks_gb, block_dim=TPB,
            )

    # ------------------------------------------------------------------
    # zero_grad — clears grad_w + grad_b. Convenience for direct callers;
    # the production path uses Adam.zero_grad which sweeps via
    # for_each_param.
    # ------------------------------------------------------------------

    def zero_grad[target: StaticString](mut self) raises:
        self._assert_tag[target]()
        comptime if target == "cpu":
            var gw = TileTensor(self.grad_w, row_major[Self.IN, Self.OUT]())
            var gb = TileTensor(self.grad_b, row_major[Self.OUT]())
            for i in range(Self.IN):
                for j in range(Self.OUT):
                    gw[i, j] = 0.0
            for j in range(Self.OUT):
                gb[j] = 0.0
        else:
            self.grad_w_dev.value().enqueue_fill(0.0)
            self.grad_b_dev.value().enqueue_fill(0.0)

    # ------------------------------------------------------------------
    # for_each_param
    # ------------------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        prefix: String,
        mut visitor: V,
    ) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime if target == "cpu":
            var w  = TileTensor(self.weight, row_major[Self.IN, Self.OUT]())
            var gw = TileTensor(self.grad_w, row_major[Self.IN, Self.OUT]())
            var b  = TileTensor(self.bias,   row_major[Self.OUT]())
            var gb = TileTensor(self.grad_b, row_major[Self.OUT]())
            visitor.visit(prefix + sep + "weight", w, gw, Self.W_SIZE)
            visitor.visit(prefix + sep + "bias",   b, gb, Self.B_SIZE)
        else:
            # ParamVisitor.visit expects MutAnyOrigin TileTensors — go
            # through explicit-origin pointers to widen.
            var w_ptr:  UnsafePointer[Scalar[DT], MutAnyOrigin] = self.weight_dev.value().unsafe_ptr()
            var gw_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.grad_w_dev.value().unsafe_ptr()
            var b_ptr:  UnsafePointer[Scalar[DT], MutAnyOrigin] = self.bias_dev.value().unsafe_ptr()
            var gb_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.grad_b_dev.value().unsafe_ptr()
            var w  = TileTensor(w_ptr,  row_major[Self.IN, Self.OUT]())
            var gw = TileTensor(gw_ptr, row_major[Self.IN, Self.OUT]())
            var b  = TileTensor(b_ptr,  row_major[Self.OUT]())
            var gb = TileTensor(gb_ptr, row_major[Self.OUT]())
            visitor.visit(prefix + sep + "weight", w, gw, Self.W_SIZE)
            visitor.visit(prefix + sep + "bias",   b, gb, Self.B_SIZE)
