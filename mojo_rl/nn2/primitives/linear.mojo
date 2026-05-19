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
    Module,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels — module-level so enqueue_function can bind them.
# ──────────────────────────────────────────────────────────────────────────


def _cache_input_kernel[
    BATCH: Int,
    IN: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * IN
    if idx < total:
        var b = idx // IN
        var i = idx % IN
        cache[b, i] = rebind[Scalar[DT]](input[b, i])


# ──────────────────────────────────────────────────────────────────────────
# AMP cast kernels — fp32 ↔ bf16 element-wise round-trip. Used by the
# bf16 Linear path to feed `linalg.matmul[target="gpu"]` (which is
# dtype-homogeneous) and pull the result back into fp32.
# ──────────────────────────────────────────────────────────────────────────


def _fp32_to_bf16_kernel[
    N: Int,
](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var x = rebind[Scalar[DT]](src[i])
        dst[i] = x.cast[DType.bfloat16]()


def _bf16_to_fp32_kernel[
    N: Int,
](
    src: LayoutTensor[DType.bfloat16, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var x = rebind[Scalar[DType.bfloat16]](src[i])
        dst[i] = x.cast[DT]()


def _bias_add_kernel[
    BATCH: Int,
    OUT: Int,
](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var j = idx % OUT
        output[b, j] = rebind[Scalar[DT]](output[b, j]) + rebind[Scalar[DT]](bias[j])


def _grad_w_accum_kernel[
    BATCH: Int,
    IN: Int,
    OUT: Int,
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
            s += rebind[Scalar[DT]](cache[b, i]) * rebind[Scalar[DT]](grad_output[b, j])
        grad_w[i, j] = rebind[Scalar[DT]](grad_w[i, j]) + s


def _grad_bias_kernel[
    BATCH: Int,
    OUT: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    grad_bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < OUT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](grad_output[b, j])
        grad_bias[j] = rebind[Scalar[DT]](grad_bias[j]) + s


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
    var bias: List[Scalar[DT]]
    var grad_w: List[Scalar[DT]]
    var grad_b: List[Scalar[DT]]
    var cache: List[Scalar[DT]]

    # GPU storage (Some when _target_tag == TARGET_GPU)
    var weight_dev: Optional[DeviceBuffer[DT]]
    var bias_dev: Optional[DeviceBuffer[DT]]
    var grad_w_dev: Optional[DeviceBuffer[DT]]
    var grad_b_dev: Optional[DeviceBuffer[DT]]
    var cache_dev: Optional[DeviceBuffer[DT]]
    var cache_dev_n: Int
    var ctx: Optional[DeviceContext]

    # AMP scratch (lazy, populated on first forward[POLICY=Bf16Compute]
    # call). Cast-around-matmul: linalg.matmul[target="gpu"] is dtype-
    # homogeneous, so we re-cast weight + activations to bf16 each step,
    # run a bf16 matmul, then cast the output back to fp32.
    #   w_bf16  — IN × OUT     (weights, re-cast every forward/backward)
    #   in_bf16 — BATCH × IN   (forward activations + backward grad_input)
    #   ou_bf16 — BATCH × OUT  (forward output + backward grad_output)
    var w_bf16_dev:  Optional[DeviceBuffer[DType.bfloat16]]
    var in_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var ou_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var amp_n_batch: Int   # current capacity (BATCH) for batch-dim buffers

    var _target_tag: Int8
    var _inference: Bool

    # ------------------------------------------------------------------
    # Defaultable: empty placeholders + UNINIT tag.
    # ------------------------------------------------------------------

    def __init__(out self):
        self.weight = List[Scalar[DT]]()
        self.bias = List[Scalar[DT]]()
        self.grad_w = List[Scalar[DT]]()
        self.grad_b = List[Scalar[DT]]()
        self.cache = List[Scalar[DT]]()
        self.weight_dev = None
        self.bias_dev = None
        self.grad_w_dev = None
        self.grad_b_dev = None
        self.cache_dev = None
        self.cache_dev_n = 0
        self.ctx = None
        self.w_bf16_dev = None
        self.in_bf16_dev = None
        self.ou_bf16_dev = None
        self.amp_n_batch = 0
        self._target_tag = TARGET_UNINIT
        self._inference = False

    # ------------------------------------------------------------------
    # make[target, INIT] — populates storage and stamps tag.
    # ------------------------------------------------------------------

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. Use `.make[target='gpu', INIT](ctx)` for GPU."""
        comptime assert (
            target == "cpu"
        ), "Linear.make[target='gpu', INIT] requires a DeviceContext"
        var lin = Self()
        lin.weight = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        lin.bias = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        lin.grad_w = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        lin.grad_b = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        INIT.init_weight(
            lin.weight.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT
        )
        INIT.init_bias(lin.bias.unsafe_ptr(), Self.B_SIZE)
        lin._target_tag = TARGET_CPU
        return lin^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        """GPU factory."""
        comptime assert (
            target == "gpu"
        ), "Linear.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var lin = Self()
        var w_dev = ctx.enqueue_create_buffer[DT](Self.W_SIZE)
        var b_dev = ctx.enqueue_create_buffer[DT](Self.B_SIZE)
        var gw_dev = ctx.enqueue_create_buffer[DT](Self.W_SIZE)
        var gb_dev = ctx.enqueue_create_buffer[DT](Self.B_SIZE)
        var c_dev = ctx.enqueue_create_buffer[DT](1)
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
        lin.bias_dev = b_dev^
        lin.grad_w_dev = gw_dev^
        lin.grad_b_dev = gb_dev^
        lin.cache_dev = c_dev^
        lin.cache_dev_n = 0
        lin.ctx = ctx
        # AMP scratch stays None until first forward[POLICY=Bf16Compute].
        lin.w_bf16_dev = None
        lin.in_bf16_dev = None
        lin.ou_bf16_dev = None
        lin.amp_n_batch = 0
        lin._target_tag = TARGET_GPU
        return lin^

    # ------------------------------------------------------------------
    # Internal: tag-mismatch guard.
    # ------------------------------------------------------------------

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Linear: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag="
                + String(Int(self._target_tag))
                + ")"
            )

    def _ensure_cache_cpu(mut self, batch: Int):
        var needed = batch * Self.IN
        if len(self.cache) < needed:
            self.cache.resize(needed, 0.0)

    def _ensure_cache_dev(mut self, needed: Int) raises:
        if self.cache_dev_n < needed:
            self.cache_dev = self.ctx.value().enqueue_create_buffer[DT](needed)
            self.cache_dev_n = needed

    def _ensure_amp_buffers_gpu(mut self, batch: Int) raises:
        """Lazy-grow the bf16 scratch buffers used by the
        cast-around-matmul AMP path. Weight scratch is sized at
        IN*OUT (compile-time), batch-dim scratches grow with BATCH."""
        var ctx = self.ctx.value()
        if not self.w_bf16_dev:
            self.w_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](Self.W_SIZE)
        if self.amp_n_batch < batch:
            self.in_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](batch * Self.IN)
            self.ou_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](batch * Self.OUT)
            self.amp_n_batch = batch

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
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert input.flat_rank == 2, "input must be rank-2 [BATCH, IN]"
        comptime assert (
            output.flat_rank == 2
        ), "output must be rank-2 [BATCH, OUT]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            # CPU path: ignore POLICY, always fp32 (CPU AMP isn't useful).
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
            var input_w = rebind[TileTensor[DT, LIN, MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)

            # Cache input (always fp32 — backward needs it in PARAM_DT for
            # the grad_w accumulation kernel, which stays fp32 regardless
            # of AMP).
            comptime cache_layout = Layout.row_major(BATCH, Self.IN)
            var input_lt = LayoutTensor[DT, cache_layout, MutAnyOrigin](input_w.ptr)
            var cache_lt = LayoutTensor[DT, cache_layout, MutAnyOrigin](self.cache_dev.value())
            comptime TPB = 128
            comptime n_blocks_cache = (BATCH * Self.IN + TPB - 1) // TPB
            comptime cache_kernel = _cache_input_kernel[BATCH, Self.IN]
            ctx.enqueue_function[cache_kernel](
                input_lt, cache_lt,
                grid_dim=n_blocks_cache, block_dim=TPB,
            )

            comptime if POLICY.compute_dtype == DT:
                # ── fp32 GPU path ────────────────────────────────────────
                var weight_tt = TileTensor(
                    self.weight_dev.value(), row_major[Self.IN, Self.OUT]()
                )
                max_matmul[target="gpu"](
                    output_w, input_w, weight_tt, DeviceContextPtr(ctx)
                )
            else:
                # ── bf16 cast-around-matmul path ─────────────────────────
                comptime assert POLICY.compute_dtype == DType.bfloat16, (
                    "Phase 3 supports only fp32 and bf16 compute_dtype"
                )
                self._ensure_amp_buffers_gpu(BATCH)

                # Cast weight fp32 → bf16. Re-cast every forward — weights
                # change every optimizer step.
                var w_fp32_lt = LayoutTensor[
                    DT, Layout.row_major(Self.W_SIZE), MutAnyOrigin,
                ](self.weight_dev.value())
                var w_bf16_lt = LayoutTensor[
                    DType.bfloat16, Layout.row_major(Self.W_SIZE), MutAnyOrigin,
                ](self.w_bf16_dev.value())
                comptime n_blocks_w = (Self.W_SIZE + TPB - 1) // TPB
                comptime down_w_k = _fp32_to_bf16_kernel[Self.W_SIZE]
                ctx.enqueue_function[down_w_k](
                    w_fp32_lt, w_bf16_lt,
                    grid_dim=n_blocks_w, block_dim=TPB,
                )

                # Cast input fp32 → bf16.
                var in_fp32_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH * Self.IN), MutAnyOrigin,
                ](input_w.ptr)
                var in_bf16_lt = LayoutTensor[
                    DType.bfloat16, Layout.row_major(BATCH * Self.IN), MutAnyOrigin,
                ](self.in_bf16_dev.value())
                comptime n_blocks_in = (BATCH * Self.IN + TPB - 1) // TPB
                comptime down_in_k = _fp32_to_bf16_kernel[BATCH * Self.IN]
                ctx.enqueue_function[down_in_k](
                    in_fp32_lt, in_bf16_lt,
                    grid_dim=n_blocks_in, block_dim=TPB,
                )

                # bf16 matmul → bf16 output scratch.
                var in_bf16_tt = TileTensor(
                    self.in_bf16_dev.value(), row_major[BATCH, Self.IN]()
                )
                var w_bf16_tt = TileTensor(
                    self.w_bf16_dev.value(), row_major[Self.IN, Self.OUT]()
                )
                var ou_bf16_tt = TileTensor(
                    self.ou_bf16_dev.value(), row_major[BATCH, Self.OUT]()
                )
                max_matmul[target="gpu"](
                    ou_bf16_tt, in_bf16_tt, w_bf16_tt, DeviceContextPtr(ctx)
                )

                # Cast output bf16 → fp32 (into caller's output buffer).
                var ou_bf16_lt = LayoutTensor[
                    DType.bfloat16, Layout.row_major(BATCH * Self.OUT), MutAnyOrigin,
                ](self.ou_bf16_dev.value())
                var ou_fp32_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH * Self.OUT), MutAnyOrigin,
                ](output_w.ptr)
                comptime n_blocks_ou = (BATCH * Self.OUT + TPB - 1) // TPB
                comptime up_ou_k = _bf16_to_fp32_kernel[BATCH * Self.OUT]
                ctx.enqueue_function[up_ou_k](
                    ou_bf16_lt, ou_fp32_lt,
                    grid_dim=n_blocks_ou, block_dim=TPB,
                )

            # Bias add (fp32, both branches).
            comptime out_layout = Layout.row_major(BATCH, Self.OUT)
            comptime bias_layout = Layout.row_major(Self.OUT)
            var output_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](output_w.ptr)
            var bias_lt = LayoutTensor[DT, bias_layout, MutAnyOrigin](self.bias_dev.value())
            comptime n_blocks_ba = (BATCH * Self.OUT + TPB - 1) // TPB
            comptime ba_kernel = _bias_add_kernel[BATCH, Self.OUT]
            ctx.enqueue_function[ba_kernel](
                output_lt, bias_lt,
                grid_dim=n_blocks_ba, block_dim=TPB,
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
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            # CPU path: ignore POLICY, fp32.
            var w = TileTensor(self.weight, row_major[Self.IN, Self.OUT]())
            var gw = TileTensor(self.grad_w, row_major[Self.IN, Self.OUT]())
            var gb = TileTensor(self.grad_b, row_major[Self.OUT]())
            var c = TileTensor(self.cache, row_major[BATCH, Self.IN]())
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
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](
                grad_output
            )
            var grad_input_w = rebind[TileTensor[DT, LGI, MutAnyOrigin]](
                grad_input
            )

            comptime TPB = 128

            comptime if POLICY.compute_dtype == DT:
                # ── fp32 GPU path ────────────────────────────────────────
                # grad_input = grad_output @ W^T
                var weight_tt = TileTensor(
                    self.weight_dev.value(), row_major[Self.IN, Self.OUT]()
                )
                max_matmul[transpose_b=True, target="gpu"](
                    grad_input_w, grad_output_w, weight_tt, DeviceContextPtr(ctx)
                )
            else:
                # ── bf16 cast-around-matmul path ─────────────────────────
                comptime assert POLICY.compute_dtype == DType.bfloat16, (
                    "Phase 3 supports only fp32 and bf16 compute_dtype"
                )
                self._ensure_amp_buffers_gpu(BATCH)

                # Cast weight fp32 → bf16 (Adam may have stepped since
                # forward; re-cast).
                var w_fp32_lt = LayoutTensor[
                    DT, Layout.row_major(Self.W_SIZE), MutAnyOrigin,
                ](self.weight_dev.value())
                var w_bf16_lt = LayoutTensor[
                    DType.bfloat16, Layout.row_major(Self.W_SIZE), MutAnyOrigin,
                ](self.w_bf16_dev.value())
                comptime n_blocks_w = (Self.W_SIZE + TPB - 1) // TPB
                comptime down_w_k = _fp32_to_bf16_kernel[Self.W_SIZE]
                ctx.enqueue_function[down_w_k](
                    w_fp32_lt, w_bf16_lt,
                    grid_dim=n_blocks_w, block_dim=TPB,
                )

                # Cast grad_output fp32 → bf16. Use ou_bf16 scratch
                # (sized BATCH * OUT — same shape as grad_output).
                var go_fp32_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH * Self.OUT), MutAnyOrigin,
                ](grad_output_w.ptr)
                var go_bf16_lt = LayoutTensor[
                    DType.bfloat16, Layout.row_major(BATCH * Self.OUT), MutAnyOrigin,
                ](self.ou_bf16_dev.value())
                comptime n_blocks_go = (BATCH * Self.OUT + TPB - 1) // TPB
                comptime down_go_k = _fp32_to_bf16_kernel[BATCH * Self.OUT]
                ctx.enqueue_function[down_go_k](
                    go_fp32_lt, go_bf16_lt,
                    grid_dim=n_blocks_go, block_dim=TPB,
                )

                # bf16 matmul: grad_input_bf16 = grad_output_bf16 @ W^T
                var go_bf16_tt = TileTensor(
                    self.ou_bf16_dev.value(), row_major[BATCH, Self.OUT]()
                )
                var w_bf16_tt = TileTensor(
                    self.w_bf16_dev.value(), row_major[Self.IN, Self.OUT]()
                )
                var gi_bf16_tt = TileTensor(
                    self.in_bf16_dev.value(), row_major[BATCH, Self.IN]()
                )
                max_matmul[transpose_b=True, target="gpu"](
                    gi_bf16_tt, go_bf16_tt, w_bf16_tt, DeviceContextPtr(ctx)
                )

                # Cast grad_input bf16 → fp32 into caller's buffer.
                var gi_bf16_lt = LayoutTensor[
                    DType.bfloat16, Layout.row_major(BATCH * Self.IN), MutAnyOrigin,
                ](self.in_bf16_dev.value())
                var gi_fp32_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH * Self.IN), MutAnyOrigin,
                ](grad_input_w.ptr)
                comptime n_blocks_gi = (BATCH * Self.IN + TPB - 1) // TPB
                comptime up_gi_k = _bf16_to_fp32_kernel[BATCH * Self.IN]
                ctx.enqueue_function[up_gi_k](
                    gi_bf16_lt, gi_fp32_lt,
                    grid_dim=n_blocks_gi, block_dim=TPB,
                )

            # grad_w and grad_b accumulation stay fp32 regardless of POLICY:
            # `cache` and `grad_output` are fp32; the custom accum kernels
            # already accumulate in fp32 per the AMP policy's `accum_dtype`.
            comptime cache_layout = Layout.row_major(BATCH, Self.IN)
            comptime go_layout = Layout.row_major(BATCH, Self.OUT)
            comptime gw_layout = Layout.row_major(Self.IN, Self.OUT)
            var cache_lt = LayoutTensor[DT, cache_layout, MutAnyOrigin](
                self.cache_dev.value()
            )
            var go_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](
                grad_output_w.ptr
            )
            var gw_lt = LayoutTensor[DT, gw_layout, MutAnyOrigin](
                self.grad_w_dev.value()
            )
            comptime n_blocks_gw = (Self.W_SIZE + TPB - 1) // TPB
            comptime gw_kernel = _grad_w_accum_kernel[BATCH, Self.IN, Self.OUT]
            ctx.enqueue_function[gw_kernel](
                cache_lt,
                go_lt,
                gw_lt,
                grid_dim=n_blocks_gw,
                block_dim=TPB,
            )
            comptime gb_layout = Layout.row_major(Self.OUT)
            var gb_lt = LayoutTensor[DT, gb_layout, MutAnyOrigin](
                self.grad_b_dev.value()
            )
            comptime n_blocks_gb = (Self.OUT + TPB - 1) // TPB
            comptime gb_kernel = _grad_bias_kernel[BATCH, Self.OUT]
            ctx.enqueue_function[gb_kernel](
                go_lt,
                gb_lt,
                grid_dim=n_blocks_gb,
                block_dim=TPB,
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
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        # Decay convention: weight decays, bias does not (PyTorch default).
        # See nn2/core/param_visitor.mojo for the layer-local ownership rule.
        comptime if target == "cpu":
            var w = TileTensor(self.weight, row_major[Self.IN, Self.OUT]())
            var gw = TileTensor(self.grad_w, row_major[Self.IN, Self.OUT]())
            var b = TileTensor(self.bias, row_major[Self.OUT]())
            var gb = TileTensor(self.grad_b, row_major[Self.OUT]())
            visitor.visit(prefix + sep + "weight", w, gw, Self.W_SIZE, True)
            visitor.visit(prefix + sep + "bias", b, gb, Self.B_SIZE, False)
        else:
            # ParamVisitor.visit is now origin-generic — pass DeviceBuffers
            # directly to TileTensor; visitor rebinds internally if it needs
            # MutAnyOrigin for a kernel.
            var w  = TileTensor(self.weight_dev.value(), row_major[Self.IN, Self.OUT]())
            var gw = TileTensor(self.grad_w_dev.value(), row_major[Self.IN, Self.OUT]())
            var b  = TileTensor(self.bias_dev.value(),   row_major[Self.OUT]())
            var gb = TileTensor(self.grad_b_dev.value(), row_major[Self.OUT]())
            visitor.visit(prefix + sep + "weight", w, gw, Self.W_SIZE, True)
            visitor.visit(prefix + sep + "bias", b, gb, Self.B_SIZE, False)

    def set_inference(mut self, value: Bool):
        # Linear has no inference-only behavior — flag is stored for
        # consistency with the Module trait but does not change forward.
        self._inference = value
