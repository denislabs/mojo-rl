"""AvgPool2D[C, K, S, P, H, W] — 2D average pooling with zero padding.

Phase 5 of `nn/PORTING_PLAN.md`.

Comptime shape mirrors `MaxPool2D` — `[BATCH, C·H·W]` in, `[BATCH, C·OH·OW]`
out where `OH = (H + 2P - K) // S + 1`, `OW = (W + 2P - K) // S + 1`.

Padding convention: `count_include_pad = True` (matches PyTorch
default). Denominator is always `K·K`; padded cells contribute 0 to
the sum but still count in the average. Simpler, no shape-dependent
edge cases.

No params, no cache. Backward broadcasts each output gradient uniformly
to its `K·K` input window with weight `1/(K·K)`; padded lanes never
receive gradient.

GPU layout: forward is output-indexed (1 thread per output cell);
backward is **input-indexed** (1 thread per input cell, looping over
overlapping output windows). Matches `MaxPool2D` so the same no-atomics
convention holds for overlapping pool configurations.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for


def _avg_pool_2d_forward_kernel[
    BATCH: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin],
    inv_kk: Scalar[DT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * OUT_FLAT
    if idx >= total:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var spatial_out = OH * OW
    var c = out_pos // spatial_out
    var rem = out_pos % spatial_out
    var oh = rem // OW
    var ow = rem % OW

    var in_c_off = c * H * W
    var s: Scalar[DT] = 0.0
    for kh in range(K):
        var ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        for kw in range(K):
            var iw = ow * S + kw - P
            if iw < 0 or iw >= W:
                continue
            s += rebind[Scalar[DT]](input[b, in_c_off + ih * W + iw])
    output[b, out_pos] = s * inv_kk


def _avg_pool_2d_backward_kernel[
    BATCH: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin,
    ],
    grad_input: LayoutTensor[
        DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin,
    ],
    inv_kk: Scalar[DT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * IN_FLAT
    if idx >= total:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var hw = H * W
    var c = in_pos // hw
    var rem = in_pos % hw
    var ih = rem // W
    var iw = rem % W

    # Output windows that contain input cell (ih, iw).
    var oh_max_raw = ih + P
    var ow_max_raw = iw + P
    if oh_max_raw < 0 or ow_max_raw < 0:
        grad_input[b, in_pos] = Scalar[DT](0.0)
        return
    var oh_top = oh_max_raw // S
    var ow_top = ow_max_raw // S
    var oh_bot_raw = ih + P - K + 1
    var ow_bot_raw = iw + P - K + 1
    var oh_bot: Int = 0
    if oh_bot_raw > 0:
        oh_bot = (oh_bot_raw + S - 1) // S
    var ow_bot: Int = 0
    if ow_bot_raw > 0:
        ow_bot = (ow_bot_raw + S - 1) // S
    if oh_top >= OH:
        oh_top = OH - 1
    if ow_top >= OW:
        ow_top = OW - 1

    var spatial_out = OH * OW
    var out_c_off = c * spatial_out
    var acc: Scalar[DT] = 0.0
    var oh = oh_bot
    while oh <= oh_top:
        var ow = ow_bot
        while ow <= ow_top:
            acc += rebind[Scalar[DT]](
                grad_output[b, out_c_off + oh * OW + ow]
            )
            ow += 1
        oh += 1
    grad_input[b, in_pos] = acc * inv_kk


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
            a.ts = TargetStorage.make_gpu(ctx.value())
        return a^

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["AvgPool2D", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)
        var in_p = input.ptr
        var out_p = output_v.ptr

        comptime if target == "cpu":
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
            comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
            comptime out_layout = Layout.row_major(BATCH, Self.OUT_DIM_FLAT)
            var in_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](out_p)
            var inv_kk = Scalar[DT](1.0 / Float64(Self.K * Self.K))
            comptime total = BATCH * Self.OUT_DIM_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _avg_pool_2d_forward_kernel[
                BATCH, Self.C, Self.K, Self.S, Self.P,
                Self.H, Self.W, Self.OH, Self.OW,
                Self.IN_DIM_FLAT, Self.OUT_DIM_FLAT,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, inv_kk,
                grid_dim=n_blocks, block_dim=TPB,
            )

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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["AvgPool2D", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
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
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
            comptime out_layout = Layout.row_major(BATCH, Self.OUT_DIM_FLAT)
            var go_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](gi_p)
            var inv_kk = Scalar[DT](1.0 / Float64(Self.K * Self.K))
            comptime total = BATCH * Self.IN_DIM_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _avg_pool_2d_backward_kernel[
                BATCH, Self.C, Self.K, Self.S, Self.P,
                Self.H, Self.W, Self.OH, Self.OW,
                Self.IN_DIM_FLAT, Self.OUT_DIM_FLAT,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, inv_kk,
                grid_dim=n_blocks, block_dim=TPB,
            )
