"""BatchNorm2D[C, H, W, MOMENTUM, EPSILON] — per-channel BN for spatial inputs.

Phase 5 of `nn2/PORTING_PLAN.md`. Mirrors `batch_norm_1d.mojo`'s
surface — γ/β as `Param`s with `decay=False`, running_mean/var as
side-channel `List[Scalar[DT]]`, per-instance `training: Bool`,
`cache_is_training` flag, same checkpoint-follow-up note.

The only structural difference vs BN1D is the reduction axis: stats
are reduced over batch *and* spatial position (H·W), giving
`N_eff = BATCH · H · W` samples per channel. Forward and backward are
otherwise the standard BN formulas, applied per channel.

Comptime shape: input `[BATCH, C, H, W]` flattened to `[BATCH, C·H·W]`;
output is the same shape. Used after every `Conv2D` in a CNN trunk
(NatureDQN doesn't use it, but ResNet-style trunks do).

CPU only at landing — GPU follow-up is gated on a real CNN consumer,
same triage as Conv2D / Pool2D.
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


comptime BN2D_DEFAULT_EPS: Float64 = 1e-5
comptime BN2D_DEFAULT_MOM: Float64 = 0.1


struct BatchNorm2D[
    C: Int, H: Int, W: Int,
    MOMENTUM: Float64 = BN2D_DEFAULT_MOM,
    EPSILON: Float64 = BN2D_DEFAULT_EPS,
](Module):
    comptime ARITY: Int = 1
    comptime FLAT_DIM: Int = Self.C * Self.H * Self.W
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.FLAT_DIM)
    comptime OUT_DIM = Self.FLAT_DIM
    comptime SPATIAL: Int = Self.H * Self.W

    var gamma: Param["gamma", False, Self.C]
    var beta:  Param["beta",  False, Self.C]
    var running_mean: List[Scalar[DT]]
    var running_var:  List[Scalar[DT]]
    var cache_xhat: List[Scalar[DT]]     # [BATCH, C, H, W] flat
    var cache_inv_std: List[Scalar[DT]]  # [C]
    var cache_is_training: Bool
    var training: Bool
    var ts: TargetStorage

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.C]()
        self.beta  = Param["beta",  False, Self.C]()
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
        comptime assert target == "cpu" or target == "gpu", (
            "BatchNorm2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.C > 0 and Self.H > 0 and Self.W > 0, (
            "BatchNorm2D: C, H, W must all be > 0"
        )
        comptime assert Self.MOMENTUM > 0.0 and Self.MOMENTUM <= 1.0, (
            "BatchNorm2D: MOMENTUM must be in (0, 1]"
        )
        var bn = Self()
        comptime if target == "cpu":
            bn.gamma = Param["gamma", False, Self.C].make_cpu()
            bn.beta  = Param["beta",  False, Self.C].make_cpu()
            var g_ptr = bn.gamma.value_unsafe_ptr_cpu()
            for k in range(Self.C):
                g_ptr[k] = Scalar[DT](1.0)
            bn.running_mean = List[Scalar[DT]](
                length=Self.C, fill=Scalar[DT](0.0),
            )
            bn.running_var = List[Scalar[DT]](
                length=Self.C, fill=Scalar[DT](1.0),
            )
            bn.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("BatchNorm2D.make[target='gpu']: ctx required")
            raise Error(
                "BatchNorm2D: GPU path not implemented yet (see"
                " PORTING_PLAN.md Phase 5)"
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
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var g_p = self.gamma.value_unsafe_ptr_cpu()
            var b_p = self.bias_unsafe_ptr_cpu()
            var rm_v = TileTensor(self.running_mean, row_major[Self.C]())
            var rv_v = TileTensor(self.running_var,  row_major[Self.C]())
            var eps = Scalar[DT](Self.EPSILON)
            var n_eff = Scalar[DT](Float64(BATCH * Self.SPATIAL))
            var inv_n = Scalar[DT](1.0) / n_eff
            if self.training:
                ensure_cpu_buffer(
                    self.cache_xhat, BATCH * Self.FLAT_DIM,
                )
                ensure_cpu_buffer(self.cache_inv_std, Self.C)
                var xhat_p = self.cache_xhat.unsafe_ptr()
                var inv_v = TileTensor(
                    self.cache_inv_std, row_major[Self.C](),
                )
                var mom = Scalar[DT](Self.MOMENTUM)
                var one_m = Scalar[DT](1.0) - mom
                for c in range(Self.C):
                    var g = g_p[c]
                    var bt = b_p[c]
                    var mean: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            mean += in_p[base + s]
                    mean *= inv_n
                    var var_: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            var d = in_p[base + s] - mean
                            var_ += d * d
                    var_ *= inv_n
                    var inv_std = Scalar[DT](1.0) / sqrt(var_ + eps)
                    inv_v[c] = inv_std
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            var xh = (in_p[base + s] - mean) * inv_std
                            xhat_p[base + s] = xh
                            out_p[base + s] = g * xh + bt
                    rm_v[c] = one_m * rm_v[c] + mom * mean
                    rv_v[c] = one_m * rv_v[c] + mom * var_
                self.cache_is_training = True
            else:
                for c in range(Self.C):
                    var rm = rm_v[c]
                    var rv = rv_v[c]
                    var inv_std = Scalar[DT](1.0) / sqrt(rv + eps)
                    var g = g_p[c]
                    var bt = b_p[c]
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            out_p[base + s] = (
                                g * (in_p[base + s] - rm) * inv_std + bt
                            )
        else:
            raise Error("BatchNorm2D.forward[target='gpu']: not implemented")

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
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        if not self.cache_is_training:
            raise Error(
                "BatchNorm2D.vjp: training-mode cache not populated."
                " Call forward(training=True) before vjp."
            )
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var g_p = self.gamma.value_unsafe_ptr_cpu()
            var dg_p = self.gamma.grad_unsafe_ptr_cpu()
            var db_p = self.beta.grad_unsafe_ptr_cpu()
            var xhat_p = self.cache_xhat.unsafe_ptr()
            var inv_v = TileTensor(
                self.cache_inv_std, row_major[Self.C](),
            )
            var n_eff = Scalar[DT](Float64(BATCH * Self.SPATIAL))
            var inv_n = Scalar[DT](1.0) / n_eff
            for c in range(Self.C):
                var g = g_p[c]
                var inv_std = inv_v[c]
                var sum_dxhat: Scalar[DT] = 0.0
                var sum_dxhat_xhat: Scalar[DT] = 0.0
                var d_gamma: Scalar[DT] = 0.0
                var d_beta:  Scalar[DT] = 0.0
                for b in range(BATCH):
                    var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                    for s in range(Self.SPATIAL):
                        var dy = go_p[base + s]
                        var xh = xhat_p[base + s]
                        var dxhat = dy * g
                        sum_dxhat += dxhat
                        sum_dxhat_xhat += dxhat * xh
                        d_gamma += dy * xh
                        d_beta  += dy
                var m1 = sum_dxhat * inv_n
                var m2 = sum_dxhat_xhat * inv_n
                for b in range(BATCH):
                    var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                    for s in range(Self.SPATIAL):
                        var dy = go_p[base + s]
                        var xh = xhat_p[base + s]
                        var dxhat = dy * g
                        gi_p[base + s] = inv_std * (
                            dxhat - m1 - xh * m2
                        )
                comptime if mode == "all":
                    dg_p[c] += d_gamma
                    db_p[c] += d_beta
        else:
            raise Error("BatchNorm2D.vjp[target='gpu']: not implemented")

    # Inline helper so we don't read `beta.value` through `value_unsafe_ptr_cpu`
    # twice (one less symbol to update if Param's API shifts).
    def bias_unsafe_ptr_cpu(
        mut self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.beta.value_unsafe_ptr_cpu()

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "training":
            self.training = value > Scalar[DT](0.5)
