"""MaxPool2D[C, K, S, P, H, W] — 2D max-pooling with zero padding.

Phase 5 of `nn2/PORTING_PLAN.md`.

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

GPU backward is **input-indexed** (one thread per input position) —
mirrors the no-atomics convention nn2 / deep_agents2 use elsewhere
(see `c51/target_y_block.mojo:48`). Each thread loops over the output
windows that contain its input cell, recomputes argmax for each, and
accumulates `grad_y` only when its own (ih, iw) is the window's
argmax. Pure single-writer per output cell — no race even with
overlapping pools.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


comptime MP_NEG_INF: Scalar[DT] = -1.0e30


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
#   Forward:  1 thread per output position (b, c, oh, ow). Scans the
#             K·K window and writes the max. -inf sentinel for OOB.
#   Backward: 1 thread per input position (b, c, ih, iw). Loops over
#             the (typically 1) output window(s) that contain it,
#             recomputes argmax, and accumulates `grad_y` if this lane
#             is the argmax.
# ──────────────────────────────────────────────────────────────────────


def _max_pool_2d_forward_kernel[
    BATCH: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin],
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
    var best: Scalar[DT] = MP_NEG_INF
    for kh in range(K):
        var ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        for kw in range(K):
            var iw = ow * S + kw - P
            if iw < 0 or iw >= W:
                continue
            var v = rebind[Scalar[DT]](input[b, in_c_off + ih * W + iw])
            if v > best:
                best = v
    output[b, out_pos] = best


def _max_pool_2d_backward_kernel[
    BATCH: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_FLAT), MutAnyOrigin,
    ],
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin],
    grad_input: LayoutTensor[
        DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin,
    ],
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

    # Output positions whose receptive field covers (ih, iw):
    #     oh ∈ [ceil((ih + P - K + 1) / S), floor((ih + P) / S)] ∩ [0, OH-1]
    # Implemented with safe integer math (no floor of negative).
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

    var in_c_off = c * hw
    var spatial_out = OH * OW
    var out_c_off = c * spatial_out
    var acc: Scalar[DT] = 0.0
    var oh = oh_bot
    while oh <= oh_top:
        var ow = ow_bot
        while ow <= ow_top:
            # Recompute argmax for window (oh, ow).
            var best: Scalar[DT] = MP_NEG_INF
            var best_ih: Int = -1
            var best_iw: Int = -1
            for kh in range(K):
                var win_ih = oh * S + kh - P
                if win_ih < 0 or win_ih >= H:
                    continue
                for kw in range(K):
                    var win_iw = ow * S + kw - P
                    if win_iw < 0 or win_iw >= W:
                        continue
                    var v = rebind[Scalar[DT]](
                        input[b, in_c_off + win_ih * W + win_iw]
                    )
                    if v > best:
                        best = v
                        best_ih = win_ih
                        best_iw = win_iw
            if best_ih == ih and best_iw == iw:
                acc += rebind[Scalar[DT]](
                    grad_output[b, out_c_off + oh * OW + ow]
                )
            ow += 1
        oh += 1
    grad_input[b, in_pos] = acc


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
            m.ts = TargetStorage.make_gpu(ctx.value())
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
        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            input.ptr
        )
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            output_v.ptr
        )
        self._cached_input_ptr = in_p

        comptime if target == "cpu":
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
            comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
            comptime out_layout = Layout.row_major(BATCH, Self.OUT_DIM_FLAT)
            var in_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](out_p)
            comptime total = BATCH * Self.OUT_DIM_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _max_pool_2d_forward_kernel[
                BATCH, Self.C, Self.K, Self.S, Self.P,
                Self.H, Self.W, Self.OH, Self.OW,
                Self.IN_DIM_FLAT, Self.OUT_DIM_FLAT,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt,
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
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var x_p = self._cached_input_ptr
            comptime in_layout = Layout.row_major(BATCH, Self.IN_DIM_FLAT)
            comptime out_layout = Layout.row_major(BATCH, Self.OUT_DIM_FLAT)
            var go_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](go_p)
            var in_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](x_p)
            var gi_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](gi_p)
            comptime total = BATCH * Self.IN_DIM_FLAT
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _max_pool_2d_backward_kernel[
                BATCH, Self.C, Self.K, Self.S, Self.P,
                Self.H, Self.W, Self.OH, Self.OW,
                Self.IN_DIM_FLAT, Self.OUT_DIM_FLAT,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, in_lt, gi_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
