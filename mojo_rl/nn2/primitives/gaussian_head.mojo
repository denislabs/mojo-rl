"""GaussianHead[IN, ACT] — Gaussian policy head with state-independent log_std.

CleanRL-style PPO actor head:
    Params: W (IN × ACT) | b (ACT) | log_std (ACT)
    Forward: output[b, 0:ACT]      = input @ W + b              (state-dependent mean)
             output[b, ACT:2*ACT]  = clamp(log_std, [-5, 2])    (state-independent,
                                                                 broadcast over batch)

Compared to Phase 5's StochasticActor (which uses Parallel[Linear, Linear] —
state-dependent log_std), this primitive is what PPO needs: a single
learnable per-action-dim log_std vector that is the same for every state.
This prevents weight explosion during PPO training (a documented failure
mode of state-dependent log_std under clipped surrogate).

apply_decay: weight=True, bias=False, log_std=False (CleanRL convention —
no decay on the log_std vector).

Param names (under for_each_param prefix "X"):
    X.weight, X.bias, X.log_std

AMP: POLICY accepted for trait conformance but ignored. This module
sits at the actor's last layer (HIDDEN × ACT_DIM, ACT_DIM is small —
1 for Pendulum, 6 for HalfCheetah) so AMP gains are negligible.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

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


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels (module-level so enqueue_function can bind them).
# ──────────────────────────────────────────────────────────────────────────


def _cache_input_kernel[BATCH: Int, IN: Int](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * IN
    if idx < total:
        var b = idx // IN
        var i = idx % IN
        cache[b, i] = rebind[Scalar[DT]](input[b, i])


def _gauss_head_forward_kernel[BATCH: Int, IN: Int, ACT: Int](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    log_std: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
):
    """One thread per (b, j) in BATCH × (2*ACT). For j < ACT compute mu;
    for j >= ACT broadcast the clamped log_std vector."""
    var idx = Int(global_idx.x)
    var total = BATCH * 2 * ACT
    if idx < total:
        var b = idx // (2 * ACT)
        var j = idx % (2 * ACT)
        if j < ACT:
            var acc = rebind[Scalar[DT]](bias[j])
            for i in range(IN):
                acc += rebind[Scalar[DT]](input[b, i]) * rebind[Scalar[DT]](weight[i, j])
            output[b, j] = acc
        else:
            var k = j - ACT
            var v = rebind[Scalar[DT]](log_std[k])
            if v < LOG_STD_MIN:
                v = LOG_STD_MIN
            elif v > LOG_STD_MAX:
                v = LOG_STD_MAX
            output[b, j] = v


def _gauss_head_grad_input_kernel[BATCH: Int, IN: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
):
    """grad_input = grad_mu @ W^T where grad_mu = grad_output[:, 0:ACT]."""
    var idx = Int(global_idx.x)
    var total = BATCH * IN
    if idx < total:
        var b = idx // IN
        var i = idx % IN
        var acc: Scalar[DT] = 0.0
        for j in range(ACT):
            acc += rebind[Scalar[DT]](grad_output[b, j]) * rebind[Scalar[DT]](weight[i, j])
        grad_input[b, i] = acc


def _gauss_head_grad_w_kernel[BATCH: Int, IN: Int, ACT: Int](
    cache: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_w: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
):
    """grad_w[i, j] += sum_b cache[b, i] * grad_output[b, j]   (j ∈ [0, ACT))."""
    var idx = Int(global_idx.x)
    var total = IN * ACT
    if idx < total:
        var i = idx // ACT
        var j = idx % ACT
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](cache[b, i]) * rebind[Scalar[DT]](grad_output[b, j])
        grad_w[i, j] = rebind[Scalar[DT]](grad_w[i, j]) + s


def _gauss_head_grad_b_kernel[BATCH: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_b: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
):
    """grad_b[j] += sum_b grad_output[b, j]   (j ∈ [0, ACT))."""
    var j = Int(global_idx.x)
    if j < ACT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](grad_output[b, j])
        grad_b[j] = rebind[Scalar[DT]](grad_b[j]) + s


def _gauss_head_grad_ls_kernel[BATCH: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_ls: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
):
    """grad_log_std[j] += sum_b grad_output[b, ACT+j]."""
    var j = Int(global_idx.x)
    if j < ACT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](grad_output[b, ACT + j])
        grad_ls[j] = rebind[Scalar[DT]](grad_ls[j]) + s


# ──────────────────────────────────────────────────────────────────────────
# GaussianHead — method-level target.
# ──────────────────────────────────────────────────────────────────────────


struct GaussianHead[IN: Int, ACT: Int](Module):
    comptime IN_DIM = Self.IN
    comptime OUT_DIM = 2 * Self.ACT
    comptime W_SIZE = Self.IN * Self.ACT
    comptime B_SIZE = Self.ACT
    comptime LS_SIZE = Self.ACT

    # CPU storage
    var weight: List[Scalar[DT]]
    var bias: List[Scalar[DT]]
    var log_std: List[Scalar[DT]]
    var grad_w: List[Scalar[DT]]
    var grad_b: List[Scalar[DT]]
    var grad_ls: List[Scalar[DT]]
    var cache: List[Scalar[DT]]

    # GPU storage
    var weight_dev: Optional[DeviceBuffer[DT]]
    var bias_dev: Optional[DeviceBuffer[DT]]
    var log_std_dev: Optional[DeviceBuffer[DT]]
    var grad_w_dev: Optional[DeviceBuffer[DT]]
    var grad_b_dev: Optional[DeviceBuffer[DT]]
    var grad_ls_dev: Optional[DeviceBuffer[DT]]
    var cache_dev: Optional[DeviceBuffer[DT]]
    var cache_dev_n: Int
    var ctx: Optional[DeviceContext]

    var _target_tag: Int8
    var _inference: Bool

    # Initial log_std value. Public so the example can override it
    # (CleanRL uses 0.0 → std=1.0; conservative choice for sparse-reward
    # envs is -0.5 → std=0.6).
    comptime DEFAULT_LOG_STD_INIT: Scalar[DT] = 0.0

    def __init__(out self):
        self.weight = List[Scalar[DT]]()
        self.bias = List[Scalar[DT]]()
        self.log_std = List[Scalar[DT]]()
        self.grad_w = List[Scalar[DT]]()
        self.grad_b = List[Scalar[DT]]()
        self.grad_ls = List[Scalar[DT]]()
        self.cache = List[Scalar[DT]]()
        self.weight_dev = None
        self.bias_dev = None
        self.log_std_dev = None
        self.grad_w_dev = None
        self.grad_b_dev = None
        self.grad_ls_dev = None
        self.cache_dev = None
        self.cache_dev_n = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "GaussianHead.make[target='gpu', INIT] requires a DeviceContext"
        var h = Self()
        h.weight = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        h.bias = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        h.log_std = List[Scalar[DT]](
            length=Self.LS_SIZE, fill=Self.DEFAULT_LOG_STD_INIT
        )
        h.grad_w = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        h.grad_b = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        h.grad_ls = List[Scalar[DT]](length=Self.LS_SIZE, fill=0.0)
        INIT.init_weight(h.weight.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.ACT)
        INIT.init_bias(h.bias.unsafe_ptr(), Self.B_SIZE)
        h._target_tag = TARGET_CPU
        return h^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "GaussianHead.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var h = Self()
        var w_dev = ctx.enqueue_create_buffer[DT](Self.W_SIZE)
        var b_dev = ctx.enqueue_create_buffer[DT](Self.B_SIZE)
        var ls_dev = ctx.enqueue_create_buffer[DT](Self.LS_SIZE)
        var gw_dev = ctx.enqueue_create_buffer[DT](Self.W_SIZE)
        var gb_dev = ctx.enqueue_create_buffer[DT](Self.B_SIZE)
        var gls_dev = ctx.enqueue_create_buffer[DT](Self.LS_SIZE)
        var c_dev = ctx.enqueue_create_buffer[DT](1)
        gw_dev.enqueue_fill(0.0)
        gb_dev.enqueue_fill(0.0)
        gls_dev.enqueue_fill(0.0)
        ls_dev.enqueue_fill(Self.DEFAULT_LOG_STD_INIT)

        # Init weights/bias on host then upload.
        var w_host = ctx.enqueue_create_host_buffer[DT](Self.W_SIZE)
        var b_host = ctx.enqueue_create_host_buffer[DT](Self.B_SIZE)
        ctx.synchronize()
        INIT.init_weight(w_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.ACT)
        INIT.init_bias(b_host.unsafe_ptr(), Self.B_SIZE)
        ctx.enqueue_copy(w_dev, w_host)
        ctx.enqueue_copy(b_dev, b_host)
        ctx.synchronize()
        h.weight_dev = w_dev^
        h.bias_dev = b_dev^
        h.log_std_dev = ls_dev^
        h.grad_w_dev = gw_dev^
        h.grad_b_dev = gb_dev^
        h.grad_ls_dev = gls_dev^
        h.cache_dev = c_dev^
        h.cache_dev_n = 0
        h.ctx = ctx
        h._target_tag = TARGET_GPU
        return h^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "GaussianHead: method called with [target='"
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

    # ------------------------------------------------------------------
    # CPU forward / backward helpers (state-indep log_std broadcast).
    # ------------------------------------------------------------------

    def _set_log_std_init(mut self, value: Scalar[DT]) raises:
        """Override the default log_std initialization. Call after make().
        Useful for sparse-reward envs where std=1.0 is too wide."""
        self._assert_tag["cpu"]() if self._target_tag == TARGET_CPU else self._assert_tag["gpu"]()
        if self._target_tag == TARGET_CPU:
            for k in range(Self.LS_SIZE):
                self.log_std[k] = value
        else:
            self.log_std_dev.value().enqueue_fill(value)

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
        ), "output must be rank-2 [BATCH, 2*ACT]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_cache_cpu(BATCH)
            var w = TileTensor(self.weight, row_major[Self.IN, Self.ACT]())
            var b = TileTensor(self.bias, row_major[Self.ACT]())
            var ls = TileTensor(self.log_std, row_major[Self.ACT]())
            var c = TileTensor(self.cache, row_major[BATCH, Self.IN]())
            for bi in range(BATCH):
                # Cache input.
                for i in range(Self.IN):
                    c[bi, i] = input[bi, i]
                # mu = input @ W + b
                for j in range(Self.ACT):
                    var acc = b[j]
                    for i in range(Self.IN):
                        acc += input[bi, i] * w[i, j]
                    output[bi, j] = acc
                # log_std broadcast + clamp
                for j in range(Self.ACT):
                    var v = ls[j]
                    if v < LOG_STD_MIN:
                        v = LOG_STD_MIN
                    elif v > LOG_STD_MAX:
                        v = LOG_STD_MAX
                    output[bi, Self.ACT + j] = v
        else:
            var ctx = self.ctx.value()
            self._ensure_cache_dev(BATCH * Self.IN)
            var input_w = rebind[TileTensor[DT, LIN, MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)

            comptime in_layout = Layout.row_major(BATCH, Self.IN)
            comptime out_layout = Layout.row_major(BATCH, 2 * Self.ACT)
            comptime w_layout = Layout.row_major(Self.IN, Self.ACT)
            comptime b_layout = Layout.row_major(Self.ACT)

            var input_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](input_w.ptr)
            var cache_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](
                self.cache_dev.value()
            )
            var w_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                self.weight_dev.value()
            )
            var b_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                self.bias_dev.value()
            )
            var ls_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                self.log_std_dev.value()
            )
            var output_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](
                output_w.ptr
            )

            comptime TPB = 128

            # Cache input.
            comptime n_blocks_cache = (BATCH * Self.IN + TPB - 1) // TPB
            comptime cache_kernel = _cache_input_kernel[BATCH, Self.IN]
            ctx.enqueue_function[cache_kernel](
                input_lt,
                cache_lt,
                grid_dim=n_blocks_cache,
                block_dim=TPB,
            )

            # Forward.
            comptime n_blocks_fwd = (BATCH * 2 * Self.ACT + TPB - 1) // TPB
            comptime fwd_kernel = _gauss_head_forward_kernel[
                BATCH, Self.IN, Self.ACT
            ]
            ctx.enqueue_function[fwd_kernel](
                input_lt,
                w_lt,
                b_lt,
                ls_lt,
                output_lt,
                grid_dim=n_blocks_fwd,
                block_dim=TPB,
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
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var w = TileTensor(self.weight, row_major[Self.IN, Self.ACT]())
            var gw = TileTensor(self.grad_w, row_major[Self.IN, Self.ACT]())
            var gb = TileTensor(self.grad_b, row_major[Self.ACT]())
            var gls = TileTensor(self.grad_ls, row_major[Self.ACT]())
            var c = TileTensor(self.cache, row_major[BATCH, Self.IN]())

            # grad_input = grad_mu @ W^T
            for bi in range(BATCH):
                for i in range(Self.IN):
                    var acc: Scalar[DT] = 0.0
                    for j in range(Self.ACT):
                        acc += grad_output[bi, j] * w[i, j]
                    grad_input[bi, i] = acc
            # grad_w accumulation
            for i in range(Self.IN):
                for j in range(Self.ACT):
                    var acc: Scalar[DT] = 0.0
                    for bi in range(BATCH):
                        acc += c[bi, i] * grad_output[bi, j]
                    gw[i, j] = gw[i, j] + acc
            # grad_b accumulation
            for j in range(Self.ACT):
                var acc: Scalar[DT] = 0.0
                for bi in range(BATCH):
                    acc += grad_output[bi, j]
                gb[j] = gb[j] + acc
            # grad_log_std accumulation (reduce over batch on the ACT-offset cols)
            for j in range(Self.ACT):
                var acc: Scalar[DT] = 0.0
                for bi in range(BATCH):
                    acc += grad_output[bi, Self.ACT + j]
                gls[j] = gls[j] + acc
        else:
            var ctx = self.ctx.value()
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](
                grad_output
            )
            var grad_input_w = rebind[TileTensor[DT, LGI, MutAnyOrigin]](
                grad_input
            )
            comptime go_layout = Layout.row_major(BATCH, 2 * Self.ACT)
            comptime gi_layout = Layout.row_major(BATCH, Self.IN)
            comptime w_layout = Layout.row_major(Self.IN, Self.ACT)
            comptime b_layout = Layout.row_major(Self.ACT)
            comptime cache_layout = Layout.row_major(BATCH, Self.IN)

            var go_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](
                grad_output_w.ptr
            )
            var gi_lt = LayoutTensor[DT, gi_layout, MutAnyOrigin](
                grad_input_w.ptr
            )
            var w_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                self.weight_dev.value()
            )
            var gw_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                self.grad_w_dev.value()
            )
            var gb_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                self.grad_b_dev.value()
            )
            var gls_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                self.grad_ls_dev.value()
            )
            var cache_lt = LayoutTensor[DT, cache_layout, MutAnyOrigin](
                self.cache_dev.value()
            )

            comptime TPB = 128

            comptime n_blocks_gi = (BATCH * Self.IN + TPB - 1) // TPB
            comptime gi_kernel = _gauss_head_grad_input_kernel[
                BATCH, Self.IN, Self.ACT
            ]
            ctx.enqueue_function[gi_kernel](
                go_lt,
                w_lt,
                gi_lt,
                grid_dim=n_blocks_gi,
                block_dim=TPB,
            )

            comptime n_blocks_gw = (Self.W_SIZE + TPB - 1) // TPB
            comptime gw_kernel = _gauss_head_grad_w_kernel[
                BATCH, Self.IN, Self.ACT
            ]
            ctx.enqueue_function[gw_kernel](
                cache_lt,
                go_lt,
                gw_lt,
                grid_dim=n_blocks_gw,
                block_dim=TPB,
            )

            comptime n_blocks_gb = (Self.ACT + TPB - 1) // TPB
            comptime gb_kernel = _gauss_head_grad_b_kernel[BATCH, Self.ACT]
            ctx.enqueue_function[gb_kernel](
                go_lt,
                gb_lt,
                grid_dim=n_blocks_gb,
                block_dim=TPB,
            )

            comptime gls_kernel = _gauss_head_grad_ls_kernel[BATCH, Self.ACT]
            ctx.enqueue_function[gls_kernel](
                go_lt,
                gls_lt,
                grid_dim=n_blocks_gb,
                block_dim=TPB,
            )

    # ------------------------------------------------------------------
    # backward_input — grad_input only (skip grad_w / grad_b / grad_ls)
    # ------------------------------------------------------------------

    def backward_input[
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
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var w = TileTensor(self.weight, row_major[Self.IN, Self.ACT]())
            for bi in range(BATCH):
                for i in range(Self.IN):
                    var acc: Scalar[DT] = 0.0
                    for j in range(Self.ACT):
                        acc += grad_output[bi, j] * w[i, j]
                    grad_input[bi, i] = acc
        else:
            var ctx = self.ctx.value()
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](
                grad_output
            )
            var grad_input_w = rebind[TileTensor[DT, LGI, MutAnyOrigin]](
                grad_input
            )
            comptime go_layout = Layout.row_major(BATCH, 2 * Self.ACT)
            comptime gi_layout = Layout.row_major(BATCH, Self.IN)
            comptime w_layout = Layout.row_major(Self.IN, Self.ACT)
            var go_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](
                grad_output_w.ptr
            )
            var gi_lt = LayoutTensor[DT, gi_layout, MutAnyOrigin](
                grad_input_w.ptr
            )
            var w_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                self.weight_dev.value()
            )
            comptime TPB = 128
            comptime n_blocks_gi = (BATCH * Self.IN + TPB - 1) // TPB
            comptime gi_kernel = _gauss_head_grad_input_kernel[
                BATCH, Self.IN, Self.ACT
            ]
            ctx.enqueue_function[gi_kernel](
                go_lt,
                w_lt,
                gi_lt,
                grid_dim=n_blocks_gi,
                block_dim=TPB,
            )

    # ------------------------------------------------------------------
    # zero_grad
    # ------------------------------------------------------------------

    def zero_grad[target: StaticString](mut self) raises:
        self._assert_tag[target]()
        comptime if target == "cpu":
            var gw = TileTensor(self.grad_w, row_major[Self.IN, Self.ACT]())
            var gb = TileTensor(self.grad_b, row_major[Self.ACT]())
            var gls = TileTensor(self.grad_ls, row_major[Self.ACT]())
            for i in range(Self.IN):
                for j in range(Self.ACT):
                    gw[i, j] = 0.0
            for j in range(Self.ACT):
                gb[j] = 0.0
                gls[j] = 0.0
        else:
            self.grad_w_dev.value().enqueue_fill(0.0)
            self.grad_b_dev.value().enqueue_fill(0.0)
            self.grad_ls_dev.value().enqueue_fill(0.0)

    # ------------------------------------------------------------------
    # for_each_param
    # ------------------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime if target == "cpu":
            var w = TileTensor(self.weight, row_major[Self.IN, Self.ACT]())
            var gw = TileTensor(self.grad_w, row_major[Self.IN, Self.ACT]())
            var b = TileTensor(self.bias, row_major[Self.ACT]())
            var gb = TileTensor(self.grad_b, row_major[Self.ACT]())
            var ls = TileTensor(self.log_std, row_major[Self.ACT]())
            var gls = TileTensor(self.grad_ls, row_major[Self.ACT]())
            visitor.visit(prefix + sep + "weight", w, gw, Self.W_SIZE, True)
            visitor.visit(prefix + sep + "bias", b, gb, Self.B_SIZE, False)
            visitor.visit(prefix + sep + "log_std", ls, gls, Self.LS_SIZE, False)
        else:
            var w = TileTensor(
                self.weight_dev.value(), row_major[Self.IN, Self.ACT]()
            )
            var gw = TileTensor(
                self.grad_w_dev.value(), row_major[Self.IN, Self.ACT]()
            )
            var b = TileTensor(self.bias_dev.value(), row_major[Self.ACT]())
            var gb = TileTensor(
                self.grad_b_dev.value(), row_major[Self.ACT]()
            )
            var ls = TileTensor(
                self.log_std_dev.value(), row_major[Self.ACT]()
            )
            var gls = TileTensor(
                self.grad_ls_dev.value(), row_major[Self.ACT]()
            )
            visitor.visit(prefix + sep + "weight", w, gw, Self.W_SIZE, True)
            visitor.visit(prefix + sep + "bias", b, gb, Self.B_SIZE, False)
            visitor.visit(prefix + sep + "log_std", ls, gls, Self.LS_SIZE, False)

    def set_inference(mut self, value: Bool):
        self._inference = value
