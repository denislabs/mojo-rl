"""BatchNorm1D[DIM, MOMENTUM, EPSILON] — per-feature batch normalisation.

Phase 4 of `nn2/PORTING_PLAN.md`. Settles the running-stats design
question for any future EMA-like layer:

  - `gamma` / `beta` are `Param`s with `decay=False`, walked by the
    optimizer and `for_each_param` like LayerNorm's. Bit-identical
    surface to LayerNorm.

  - Running mean / running var are stored as plain `List[Scalar[DT]]`
    side-channel fields. They are NOT walked by the param visitor (the
    optimizer ignores them), and at landing they are NOT checkpointed
    (no Saveable wrapper). A consumer that needs them in checkpoints
    will trip a follow-up task; see PORTING_PLAN.md Phase 4 follow-up
    note. Rationale: keeps the landing tight and avoids prematurely
    inventing a `RunningStat` wrapper type before the first consumer
    arrives to constrain the design.

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

CPU-only at landing. GPU is a follow-up gated on a real consumer
(NatureDQN typically doesn't use BN anyway).
"""

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

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
    # Training-only cache (output-caching).
    var cache_xhat: List[Scalar[DT]]      # [BATCH, DIM]
    var cache_inv_std: List[Scalar[DT]]   # [DIM]
    var cache_is_training: Bool
    # Runtime mode flag (default True). Mirrors Dropout.
    var training: Bool
    var ts: TargetStorage

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM]()
        self.beta  = Param["beta",  False, Self.DIM]()
        self.running_mean = List[Scalar[DT]]()
        self.running_var  = List[Scalar[DT]]()
        self.cache_xhat = List[Scalar[DT]]()
        self.cache_inv_std = List[Scalar[DT]]()
        self.cache_is_training = False
        self.training = True
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. γ←1, β←0, running_mean←0, running_var←1.
        INIT accepted for `Sequential.make[target, INIT]` uniformity but
        ignored (universal BN init). GPU path raises — see
        `PORTING_PLAN.md` Phase 4."""
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
            # β starts at 0 (Param.make_cpu zero-filled).
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
            raise Error(
                "BatchNorm1D: GPU path not implemented yet (see"
                " PORTING_PLAN.md Phase 4)"
            )
        return bn^

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
                # Don't touch cache_is_training — keep prior state. Backward
                # already gates on this flag.
        else:
            raise Error("BatchNorm1D.forward[target='gpu']: not implemented")

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
            raise Error("BatchNorm1D.vjp[target='gpu']: not implemented")

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
