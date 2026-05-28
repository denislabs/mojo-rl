"""AvgPool2D[C, K, S, P, H, W] — 2D average pooling with zero padding.

Phase 5 of `nn2/PORTING_PLAN.md`. CPU-only.

Comptime shape mirrors `MaxPool2D` — `[BATCH, C·H·W]` in, `[BATCH, C·OH·OW]`
out where `OH = (H + 2P - K) // S + 1`, `OW = (W + 2P - K) // S + 1`.

Padding convention: `count_include_pad = True` (matches PyTorch
default). Denominator is always `K·K`; padded cells contribute 0 to
the sum but still count in the average. Simpler, no shape-dependent
edge cases.

No params, no cache. Backward broadcasts each output gradient uniformly
to its `K·K` input window with weight `1/(K·K)`; padded lanes never
receive gradient.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


struct AvgPool2D[
    C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
](Module):
    comptime ARITY: Int = 1
    comptime OH: Int = (Self.H + 2 * Self.P - Self.K) // Self.S + 1
    comptime OW: Int = (Self.W + 2 * Self.P - Self.K) // Self.S + 1
    comptime IN_DIM_FLAT: Int = Self.C * Self.H * Self.W
    comptime OUT_DIM_FLAT: Int = Self.C * Self.OH * Self.OW
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_DIM_FLAT)
    comptime OUT_DIM = Self.OUT_DIM_FLAT

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "AvgPool2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.K > 0 and Self.S > 0, (
            "AvgPool2D: K and S must be positive"
        )
        comptime assert Self.OH > 0 and Self.OW > 0, (
            "AvgPool2D: invalid spatial shape — check H/W/K/S/P"
        )
        var a = Self()
        comptime if target == "cpu":
            a.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("AvgPool2D.make[target='gpu']: ctx required")
            raise Error(
                "AvgPool2D: GPU path not implemented yet (see"
                " PORTING_PLAN.md Phase 5)"
            )
        return a^

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
        assert_tag_for["AvgPool2D", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var inv_kk = Scalar[DT](1.0 / Float64(Self.K * Self.K))
            for b in range(BATCH):
                var in_base = b * Self.IN_DIM_FLAT
                var out_base = b * Self.OUT_DIM_FLAT
                for c in range(Self.C):
                    var in_c_base = in_base + c * Self.H * Self.W
                    var out_c_base = out_base + c * Self.OH * Self.OW
                    for oh in range(Self.OH):
                        for ow in range(Self.OW):
                            var s: Scalar[DT] = 0.0
                            for kh in range(Self.K):
                                var ih = oh * Self.S + kh - Self.P
                                if ih < 0 or ih >= Self.H:
                                    continue
                                for kw in range(Self.K):
                                    var iw = ow * Self.S + kw - Self.P
                                    if iw < 0 or iw >= Self.W:
                                        continue
                                    s += in_p[
                                        in_c_base + ih * Self.W + iw
                                    ]
                            out_p[out_c_base + oh * Self.OW + ow] = (
                                s * inv_kk
                            )
        else:
            raise Error("AvgPool2D.forward[target='gpu']: not implemented")

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
        assert_tag_for["AvgPool2D", target](self.ts.target_tag)
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
            var inv_kk = Scalar[DT](1.0 / Float64(Self.K * Self.K))
            for k in range(BATCH * Self.IN_DIM_FLAT):
                gi_p[k] = Scalar[DT](0.0)
            for b in range(BATCH):
                var in_base = b * Self.IN_DIM_FLAT
                var out_base = b * Self.OUT_DIM_FLAT
                for c in range(Self.C):
                    var in_c_base = in_base + c * Self.H * Self.W
                    var out_c_base = out_base + c * Self.OH * Self.OW
                    for oh in range(Self.OH):
                        for ow in range(Self.OW):
                            var go_val = (
                                go_p[out_c_base + oh * Self.OW + ow]
                                * inv_kk
                            )
                            for kh in range(Self.K):
                                var ih = oh * Self.S + kh - Self.P
                                if ih < 0 or ih >= Self.H:
                                    continue
                                for kw in range(Self.K):
                                    var iw = ow * Self.S + kw - Self.P
                                    if iw < 0 or iw >= Self.W:
                                        continue
                                    gi_p[
                                        in_c_base + ih * Self.W + iw
                                    ] += go_val
        else:
            raise Error("AvgPool2D.vjp[target='gpu']: not implemented")
