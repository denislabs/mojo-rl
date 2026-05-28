"""MaxPool2D[C, K, S, P, H, W] — 2D max-pooling with zero padding.

Phase 5 of `nn2/PORTING_PLAN.md`. CPU-only.

Comptime shape: `[BATCH, C, H, W]` flattened to `[BATCH, C·H·W]`;
output `[BATCH, C, OH, OW]` flattened to `[BATCH, C·OH·OW]`.
    OH = (H + 2P - K) // S + 1
    OW = (W + 2P - K) // S + 1

No params. No leaf-owned cache: backward re-scans each pooling window
through the orchestrator's input slab (input-alias pattern, mirrors
Clamp / ReLU). Re-finding argmax costs K·K extra ops per output
position — negligible relative to the windowed sum-of-products in the
forward, and avoids a `cache[OUT_DIM]` int-as-float storage.

Tie-break: first lane in row-major (kh, kw) iteration order wins,
matching the PyTorch convention.

Backward: only the argmax lane in each window receives the gradient.
Padded (OOB) lanes contribute `-inf` to the comparison so they never
win, and never receive gradient.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


comptime MP_NEG_INF: Scalar[DT] = -1.0e30


struct MaxPool2D[
    C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
](Module):
    comptime ARITY: Int = 1
    comptime OH: Int = (Self.H + 2 * Self.P - Self.K) // Self.S + 1
    comptime OW: Int = (Self.W + 2 * Self.P - Self.K) // Self.S + 1
    comptime IN_DIM_FLAT: Int = Self.C * Self.H * Self.W
    comptime OUT_DIM_FLAT: Int = Self.C * Self.OH * Self.OW
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_DIM_FLAT)
    comptime OUT_DIM = Self.OUT_DIM_FLAT

    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ](unsafe_from_address=0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "MaxPool2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.K > 0 and Self.S > 0, (
            "MaxPool2D: K and S must be positive"
        )
        comptime assert Self.OH > 0 and Self.OW > 0, (
            "MaxPool2D: invalid spatial shape — check H/W/K/S/P"
        )
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("MaxPool2D.make[target='gpu']: ctx required")
            raise Error(
                "MaxPool2D: GPU path not implemented yet (see"
                " PORTING_PLAN.md Phase 5)"
            )
        return m^

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
        assert_tag_for["MaxPool2D", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            self._cached_input_ptr = in_p
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            for b in range(BATCH):
                var in_base = b * Self.IN_DIM_FLAT
                var out_base = b * Self.OUT_DIM_FLAT
                for c in range(Self.C):
                    var in_c_base = in_base + c * Self.H * Self.W
                    var out_c_base = out_base + c * Self.OH * Self.OW
                    for oh in range(Self.OH):
                        for ow in range(Self.OW):
                            var best: Scalar[DT] = MP_NEG_INF
                            for kh in range(Self.K):
                                var ih = oh * Self.S + kh - Self.P
                                if ih < 0 or ih >= Self.H:
                                    continue
                                for kw in range(Self.K):
                                    var iw = ow * Self.S + kw - Self.P
                                    if iw < 0 or iw >= Self.W:
                                        continue
                                    var v = in_p[
                                        in_c_base + ih * Self.W + iw
                                    ]
                                    if v > best:
                                        best = v
                            out_p[out_c_base + oh * Self.OW + ow] = best
        else:
            raise Error("MaxPool2D.forward[target='gpu']: not implemented")

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
        assert_tag_for["MaxPool2D", target](self.ts.target_tag)
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
            var x_p = self._cached_input_ptr
            # Zero-fill grad_input — we scatter argmax-only.
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
                            var best: Scalar[DT] = MP_NEG_INF
                            var best_idx: Int = -1
                            for kh in range(Self.K):
                                var ih = oh * Self.S + kh - Self.P
                                if ih < 0 or ih >= Self.H:
                                    continue
                                for kw in range(Self.K):
                                    var iw = ow * Self.S + kw - Self.P
                                    if iw < 0 or iw >= Self.W:
                                        continue
                                    var idx = in_c_base + ih * Self.W + iw
                                    var v = x_p[idx]
                                    if v > best:
                                        best = v
                                        best_idx = idx
                            if best_idx >= 0:
                                gi_p[best_idx] += go_p[
                                    out_c_base + oh * Self.OW + ow
                                ]
        else:
            raise Error("MaxPool2D.vjp[target='gpu']: not implemented")
