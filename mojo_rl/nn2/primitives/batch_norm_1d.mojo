"""BatchNorm1D[DIM, MOMENTUM, EPSILON] — per-feature batch normalisation.

Phase 4 of `nn2/PORTING_PLAN.md`. Settles the running-stats design
question for any future EMA-like layer:

  - `gamma` / `beta` are `Param`s with `decay=False`, walked by the
    optimizer and `for_each_param` like LayerNorm's. Bit-identical
    surface to LayerNorm.

  - Running mean / running var are stored as plain `List[Scalar[DT]]`
    side-channel fields (CPU) plus `DeviceBuffer[DT]` (GPU). They are
    NOT walked by the param visitor (the optimizer ignores them), and
    at landing they are NOT checkpointed (no Saveable wrapper). A
    consumer that needs them in checkpoints will trip a follow-up task;
    see PORTING_PLAN.md Phase 4 follow-up note.

  - `training: Bool` is a per-instance runtime field (same pattern as
    `dropout.mojo`). `set_attr["training"](v > 0.5)` flips it.

Forward (training): batch μ, σ², `x̂ = (x - μ) / √(σ² + ε)`,
                    `y = γ·x̂ + β`. EMA-update running stats.
Forward (eval):     `y = γ·(x - μ_run)/√(σ²_run + ε) + β`. No EMA.
Backward (training-cache only):
    `dx̂[b,f] = grad_y[b,f] · γ[f]`
    `m1 = mean_b(dx̂)`,  `m2 = mean_b(dx̂ · x̂)`
    `dx[b,f] = inv_std · (dx̂[b,f] - m1 - x̂[b,f] · m2)`
    `dγ[f] += Σ_b grad_y[b,f] · x̂[b,f]`,  `dβ[f] += Σ_b grad_y[b,f]`

Calling `vjp` after an eval forward is a programming error — the
training cache is stale. We assert against it with a `cache_is_training`
flag; misuse raises.

GPU layout: one block per feature, threads parallel-reduce over BATCH
via `block.sum[block_size=BN_TPB]` from `std.gpu.primitives`. Mirrors
LayerNorm's GPU pattern but on the orthogonal axis (BN reduces over
samples, LN over features).
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
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
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


comptime BN_DEFAULT_EPS: Float64 = 1e-5
comptime BN_DEFAULT_MOM: Float64 = 0.1
comptime BN_TPB: Int = 128


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one block per feature, threads stride over BATCH.
# ──────────────────────────────────────────────────────────────────────


def _bn1d_forward_train_kernel[
    BATCH: Int, DIM: Int,
    EPSILON: Float64, MOMENTUM: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta:  LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_var:  LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return

    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var eps = Scalar[DT](EPSILON)
    var mom = Scalar[DT](MOMENTUM)
    var one_m = Scalar[DT](1.0) - mom

    var my_sum: Scalar[DT] = 0.0
    var b = t
    while b < BATCH:
        my_sum += rebind[Scalar[DT]](input[b, f])
        b += BN_TPB
    var mean = (
        block.sum[block_size=BN_TPB, broadcast=True](val=my_sum) * inv_n
    )

    var my_var: Scalar[DT] = 0.0
    b = t
    while b < BATCH:
        var d = rebind[Scalar[DT]](input[b, f]) - mean
        my_var += d * d
        b += BN_TPB
    var var_ = (
        block.sum[block_size=BN_TPB, broadcast=True](val=my_var) * inv_n
    )

    var inv_std: Scalar[DT] = 1.0 / sqrt(var_ + eps)
    if t == 0:
        cache_inv_std[f] = inv_std
        var rm = rebind[Scalar[DT]](running_mean[f])
        var rv = rebind[Scalar[DT]](running_var[f])
        running_mean[f] = one_m * rm + mom * mean
        running_var[f]  = one_m * rv + mom * var_

    var g = rebind[Scalar[DT]](gamma[f])
    var bt = rebind[Scalar[DT]](beta[f])
    b = t
    while b < BATCH:
        var x = rebind[Scalar[DT]](input[b, f])
        var xh = (x - mean) * inv_std
        cache_xhat[b, f] = xh
        output[b, f] = g * xh + bt
        b += BN_TPB


def _bn1d_forward_eval_kernel[
    BATCH: Int, DIM: Int, EPSILON: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta:  LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_var:  LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var eps = Scalar[DT](EPSILON)
    var rm = rebind[Scalar[DT]](running_mean[f])
    var rv = rebind[Scalar[DT]](running_var[f])
    var inv_std: Scalar[DT] = 1.0 / sqrt(rv + eps)
    var g = rebind[Scalar[DT]](gamma[f])
    var bt = rebind[Scalar[DT]](beta[f])
    var b = t
    while b < BATCH:
        var x = rebind[Scalar[DT]](input[b, f])
        output[b, f] = g * (x - rm) * inv_std + bt
        b += BN_TPB


def _bn1d_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    grad_beta:  LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var g = rebind[Scalar[DT]](gamma[f])
    var inv_std = rebind[Scalar[DT]](cache_inv_std[f])

    # Pass 1: reduce across BATCH to get sum_dxhat, sum_dxhat_xhat,
    # dgamma, dbeta. Four independent block.sum calls (each has its own
    # barriers internally).
    var my_sum_dxhat: Scalar[DT] = 0.0
    var my_sum_dxhat_xhat: Scalar[DT] = 0.0
    var my_dgamma: Scalar[DT] = 0.0
    var my_dbeta:  Scalar[DT] = 0.0
    var b = t
    while b < BATCH:
        var dy = rebind[Scalar[DT]](grad_output[b, f])
        var xh = rebind[Scalar[DT]](cache_xhat[b, f])
        var dxhat = dy * g
        my_sum_dxhat += dxhat
        my_sum_dxhat_xhat += dxhat * xh
        my_dgamma += dy * xh
        my_dbeta  += dy
        b += BN_TPB
    var sum_dxhat = block.sum[block_size=BN_TPB, broadcast=True](
        val=my_sum_dxhat
    )
    var sum_dxhat_xhat = block.sum[block_size=BN_TPB, broadcast=True](
        val=my_sum_dxhat_xhat
    )
    var d_gamma_tot = block.sum[block_size=BN_TPB, broadcast=False](
        val=my_dgamma
    )
    var d_beta_tot = block.sum[block_size=BN_TPB, broadcast=False](
        val=my_dbeta
    )
    if t == 0:
        grad_gamma[f] = (
            rebind[Scalar[DT]](grad_gamma[f]) + d_gamma_tot[0]
        )
        grad_beta[f] = (
            rebind[Scalar[DT]](grad_beta[f]) + d_beta_tot[0]
        )

    var m1 = sum_dxhat * inv_n
    var m2 = sum_dxhat_xhat * inv_n

    # Pass 2: scatter grad_input.
    b = t
    while b < BATCH:
        var dy = rebind[Scalar[DT]](grad_output[b, f])
        var xh = rebind[Scalar[DT]](cache_xhat[b, f])
        var dxhat = dy * g
        grad_input[b, f] = inv_std * (dxhat - m1 - xh * m2)
        b += BN_TPB


struct BatchNorm1D[
    DIM: Int,
    MOMENTUM: Float64 = BN_DEFAULT_MOM,
    EPSILON: Float64 = BN_DEFAULT_EPS,
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    # Gradient-tracked params (walked by for_each_param_auto).
    var gamma: Param["gamma", False, Self.DIM]
    var beta:  Param["beta",  False, Self.DIM]
    # Running stats — side-channel, not walked by ParamVisitor.
    var running_mean: List[Scalar[DT]]
    var running_var:  List[Scalar[DT]]
    var running_mean_dev: Optional[DeviceBuffer[DT]]
    var running_var_dev:  Optional[DeviceBuffer[DT]]
    # Training-only cache (output-caching).
    var cache_xhat: List[Scalar[DT]]      # [BATCH, DIM]
    var cache_inv_std: List[Scalar[DT]]   # [DIM]
    var cache_xhat_dev: Optional[DeviceBuffer[DT]]
    var cache_inv_std_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int
    var cache_is_training: Bool
    # Runtime mode flag (default True). Mirrors Dropout.
    var training: Bool
    var ts: TargetStorage

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM]()
        self.beta  = Param["beta",  False, Self.DIM]()
        self.running_mean = List[Scalar[DT]]()
        self.running_var  = List[Scalar[DT]]()
        self.running_mean_dev = None
        self.running_var_dev  = None
        self.cache_xhat = List[Scalar[DT]]()
        self.cache_inv_std = List[Scalar[DT]]()
        self.cache_xhat_dev = None
        self.cache_inv_std_dev = None
        self.cache_n_batch = 0
        self.cache_is_training = False
        self.training = True
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. γ←1, β←0, running_mean←0, running_var←1."""
        comptime assert target == "cpu" or target == "gpu", (
            "BatchNorm1D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.DIM > 0, "BatchNorm1D: DIM must be > 0"
        comptime assert Self.MOMENTUM > 0.0 and Self.MOMENTUM <= 1.0, (
            "BatchNorm1D: MOMENTUM must be in (0, 1]"
        )
        var bn = Self()
        comptime if target == "cpu":
            bn.gamma = Param["gamma", False, Self.DIM].make_cpu()
            bn.beta  = Param["beta",  False, Self.DIM].make_cpu()
            var g_ptr = bn.gamma.value_unsafe_ptr_cpu()
            for k in range(Self.DIM):
                g_ptr[k] = Scalar[DT](1.0)
            bn.running_mean = List[Scalar[DT]](
                length=Self.DIM, fill=Scalar[DT](0.0),
            )
            bn.running_var = List[Scalar[DT]](
                length=Self.DIM, fill=Scalar[DT](1.0),
            )
            bn.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("BatchNorm1D.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            bn.gamma = Param["gamma", False, Self.DIM].make_gpu(ctx_v)
            bn.beta  = Param["beta",  False, Self.DIM].make_gpu(ctx_v)
            bn.gamma.value_dev.value().enqueue_fill(1.0)
            bn.beta.value_dev.value().enqueue_fill(0.0)
            var rm_dev = ctx_v.enqueue_create_buffer[DT](Self.DIM)
            var rv_dev = ctx_v.enqueue_create_buffer[DT](Self.DIM)
            rm_dev.enqueue_fill(0.0)
            rv_dev.enqueue_fill(1.0)
            bn.running_mean_dev = rm_dev^
            bn.running_var_dev  = rv_dev^
            bn.cache_xhat_dev    = ctx_v.enqueue_create_buffer[DT](1)
            bn.cache_inv_std_dev = ctx_v.enqueue_create_buffer[DT](Self.DIM)
            bn.cache_n_batch = 0
            bn.ts = TargetStorage.make_gpu(ctx_v)
        return bn^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_xhat_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.DIM
            )
            self.cache_n_batch = batch

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
        assert_tag_for["BatchNorm1D", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var gamma_v = TileTensor(self.gamma.value, row_major[Self.DIM]())
            var beta_v  = TileTensor(self.beta.value,  row_major[Self.DIM]())
            var rm_v = TileTensor(self.running_mean, row_major[Self.DIM]())
            var rv_v = TileTensor(self.running_var,  row_major[Self.DIM]())
            var eps = Scalar[DT](Self.EPSILON)
            if self.training:
                ensure_cpu_buffer(self.cache_xhat,    BATCH * Self.DIM)
                ensure_cpu_buffer(self.cache_inv_std, Self.DIM)
                var xhat_v = TileTensor(
                    self.cache_xhat, row_major[BATCH, Self.DIM](),
                )
                var inv_v = TileTensor(
                    self.cache_inv_std, row_major[Self.DIM](),
                )
                var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(BATCH))
                var mom = Scalar[DT](Self.MOMENTUM)
                var one_m = Scalar[DT](1.0) - mom
                for f in range(Self.DIM):
                    var mean: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        mean += input[b, f]
                    mean *= inv_n
                    var var_: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        var diff = input[b, f] - mean
                        var_ += diff * diff
                    var_ *= inv_n
                    var inv_std = Scalar[DT](1.0) / sqrt(var_ + eps)
                    inv_v[f] = inv_std
                    var g = gamma_v[f]
                    var bt = beta_v[f]
                    for b in range(BATCH):
                        var xh = (input[b, f] - mean) * inv_std
                        xhat_v[b, f] = xh
                        output_v[b, f] = g * xh + bt
                    # EMA-update running stats.
                    rm_v[f] = one_m * rm_v[f] + mom * mean
                    rv_v[f] = one_m * rv_v[f] + mom * var_
                self.cache_is_training = True
            else:
                for f in range(Self.DIM):
                    var inv_std = Scalar[DT](1.0) / sqrt(
                        rv_v[f] + eps
                    )
                    var g = gamma_v[f]
                    var bt = beta_v[f]
                    var rm = rm_v[f]
                    for b in range(BATCH):
                        output_v[b, f] = (
                            g * (input[b, f] - rm) * inv_std + bt
                        )
                # Don't touch cache_is_training — keep prior state.
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_d  = Layout.row_major(Self.DIM)
            var in_p_w  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            var out_p_w = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var in_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p_w)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p_w)
            var g_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.value_dev.value()
            )
            var b_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.beta.value_dev.value()
            )
            var rm_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.running_mean_dev.value()
            )
            var rv_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.running_var_dev.value()
            )
            var ctx = self.ts.ctx.value()
            if self.training:
                self._ensure_cache_gpu(BATCH)
                var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                    self.cache_xhat_dev.value()
                )
                var is_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                    self.cache_inv_std_dev.value()
                )
                comptime fkernel = _bn1d_forward_train_kernel[
                    BATCH, Self.DIM, Self.EPSILON, Self.MOMENTUM,
                ]
                ctx.enqueue_function[fkernel](
                    in_lt, out_lt, g_lt, b_lt, rm_lt, rv_lt,
                    xh_lt, is_lt,
                    grid_dim=Self.DIM, block_dim=BN_TPB,
                )
                self.cache_is_training = True
            else:
                comptime ekernel = _bn1d_forward_eval_kernel[
                    BATCH, Self.DIM, Self.EPSILON,
                ]
                ctx.enqueue_function[ekernel](
                    in_lt, out_lt, g_lt, b_lt, rm_lt, rv_lt,
                    grid_dim=Self.DIM, block_dim=BN_TPB,
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
        assert_tag_for["BatchNorm1D", target](self.ts.target_tag)
        if not self.cache_is_training:
            raise Error(
                "BatchNorm1D.vjp: training-mode cache not populated."
                " Call forward(training=True) before vjp."
            )
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            var gamma_v = TileTensor(self.gamma.value, row_major[Self.DIM]())
            var dgamma_v = TileTensor(self.gamma.grad, row_major[Self.DIM]())
            var dbeta_v  = TileTensor(self.beta.grad,  row_major[Self.DIM]())
            var xhat_v = TileTensor(
                self.cache_xhat, row_major[BATCH, Self.DIM](),
            )
            var inv_v = TileTensor(
                self.cache_inv_std, row_major[Self.DIM](),
            )
            var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(BATCH))
            for f in range(Self.DIM):
                var g = gamma_v[f]
                var inv_std = inv_v[f]
                var sum_dxhat: Scalar[DT] = 0.0
                var sum_dxhat_xhat: Scalar[DT] = 0.0
                var d_gamma: Scalar[DT] = 0.0
                var d_beta:  Scalar[DT] = 0.0
                for b in range(BATCH):
                    var dy = grad_output_v[b, f]
                    var xh = xhat_v[b, f]
                    var dxhat = dy * g
                    sum_dxhat += dxhat
                    sum_dxhat_xhat += dxhat * xh
                    d_gamma += dy * xh
                    d_beta  += dy
                var m1 = sum_dxhat * inv_n
                var m2 = sum_dxhat_xhat * inv_n
                for b in range(BATCH):
                    var dy = grad_output_v[b, f]
                    var xh = xhat_v[b, f]
                    var dxhat = dy * g
                    grad_input_v[b, f] = inv_std * (dxhat - m1 - xh * m2)
                comptime if mode == "all":
                    dgamma_v[f] += d_gamma
                    dbeta_v[f]  += d_beta
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_d  = Layout.row_major(Self.DIM)
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p)
            var g_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.value_dev.value()
            )
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat_dev.value()
            )
            var is_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.cache_inv_std_dev.value()
            )
            var dg_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.grad_dev.value()
            )
            var db_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.beta.grad_dev.value()
            )
            comptime kernel = _bn1d_backward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, g_lt, xh_lt, is_lt, gi_lt, dg_lt, db_lt,
                grid_dim=Self.DIM, block_dim=BN_TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["BatchNorm1D", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["BatchNorm1D", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "training":
            self.training = value > Scalar[DT](0.5)
