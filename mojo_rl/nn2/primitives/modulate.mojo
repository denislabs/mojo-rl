"""Modulate[DIM] — AdaLN-zero affine modulation (ARITY=3).

    y = x * (1 + scale) + shift

Three separate inputs (nn2 graph passes one pointer per slot — no
concatenation, unlike the legacy `nn` DiffOp):
    inputs[0] = x      [BATCH, DIM]
    inputs[1] = scale  [BATCH, DIM]
    inputs[2] = shift  [BATCH, DIM]

Gradients:
    grad_x[i]     = grad_out[i] * (1 + scale[i])
    grad_scale[i] = grad_out[i] * x[i]
    grad_shift[i] = grad_out[i]

Cache (leaf-owned): x and scale (needed by the x- and scale-grads).
PARAM-free. Used inside LeWM's ConditionalTransformerBlock; see
docs/LEWM_NN2_PORT_PLAN.md §2.2.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


def _modulate_forward_kernel[
    BATCH: Int, DIM: Int,
](
    x: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    scale: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    shift: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_x: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_scale: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var i = idx % DIM
    var xv = rebind[Scalar[DT]](x[b, i])
    var sv = rebind[Scalar[DT]](scale[b, i])
    var shv = rebind[Scalar[DT]](shift[b, i])
    cache_x[b, i] = xv
    cache_scale[b, i] = sv
    output[b, i] = xv * (Scalar[DT](1.0) + sv) + shv


def _modulate_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_x: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_scale: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_x: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_scale: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_shift: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var i = idx % DIM
    var go = rebind[Scalar[DT]](grad_output[b, i])
    var xv = rebind[Scalar[DT]](cache_x[b, i])
    var sv = rebind[Scalar[DT]](cache_scale[b, i])
    grad_x[b, i] = go * (Scalar[DT](1.0) + sv)
    grad_scale[b, i] = go * xv
    grad_shift[b, i] = go


struct Modulate[DIM: Int](Module):
    comptime ARITY: Int = 3
    comptime IN_DIMS = InlineArray[Int, 3](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    @staticmethod
    def display_label() -> String:
        return String("Modulate")

    var cache_x: List[Scalar[DT]]
    var cache_scale: List[Scalar[DT]]
    var cache_x_dev: Optional[DeviceBuffer[DT]]
    var cache_scale_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int
    var ts: TargetStorage

    def __init__(out self):
        self.cache_x = List[Scalar[DT]]()
        self.cache_scale = List[Scalar[DT]]()
        self.cache_x_dev = None
        self.cache_scale_dev = None
        self.cache_n_batch = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Modulate: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["Modulate.make[target='gpu']"](ctx)
            m.cache_x_dev = ctx_v.enqueue_create_buffer[DT](1)
            m.cache_scale_dev = ctx_v.enqueue_create_buffer[DT](1)
            m.cache_n_batch = 0
            m.ts = TargetStorage.make_gpu(ctx_v)
        return m^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_x_dev = ctx.enqueue_create_buffer[DT](batch * Self.DIM)
            self.cache_scale_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.DIM
            )
            self.cache_n_batch = batch

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
        assert_tag_for["Modulate", target](self.ts.target_tag)
        var x = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var scale = typed_view[BATCH, Self.IN_DIMS[1]](inputs[1])
        var shift = typed_view[BATCH, Self.IN_DIMS[2]](inputs[2])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            ensure_cpu_buffer(self.cache_x, BATCH * Self.DIM)
            ensure_cpu_buffer(self.cache_scale, BATCH * Self.DIM)
            var cx = TileTensor(self.cache_x, row_major[BATCH, Self.DIM]())
            var cs = TileTensor(self.cache_scale, row_major[BATCH, Self.DIM]())
            for b in range(BATCH):
                for i in range(Self.DIM):
                    var xv = x[b, i]
                    var sv = scale[b, i]
                    cx[b, i] = xv
                    cs[b, i] = sv
                    out[b, i] = xv * (Scalar[DT](1.0) + sv) + shift[b, i]
        else:
            self._ensure_cache_gpu(BATCH)
            comptime lay = Layout.row_major(BATCH, Self.DIM)
            var x_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x.ptr)
            )
            var s_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](scale.ptr)
            )
            var sh_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](shift.ptr)
            )
            var o_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.ptr)
            )
            var cx_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                self.cache_x_dev.value()
            )
            var cs_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                self.cache_scale_dev.value()
            )
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _modulate_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                x_lt, s_lt, sh_lt, o_lt, cx_lt, cs_lt,
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
        assert_tag_for["Modulate", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gx = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])
        var gs = typed_view_mut[BATCH, Self.IN_DIMS[1]](grad_inputs[1])
        var gsh = typed_view_mut[BATCH, Self.IN_DIMS[2]](grad_inputs[2])

        comptime if target == "cpu":
            var cx = TileTensor(self.cache_x, row_major[BATCH, Self.DIM]())
            var cs = TileTensor(self.cache_scale, row_major[BATCH, Self.DIM]())
            for b in range(BATCH):
                for i in range(Self.DIM):
                    var g = go[b, i]
                    gx[b, i] = g * (Scalar[DT](1.0) + cs[b, i])
                    gs[b, i] = g * cx[b, i]
                    gsh[b, i] = g
        else:
            comptime lay = Layout.row_major(BATCH, Self.DIM)
            var go_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr)
            )
            var gx_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gx.ptr)
            )
            var gs_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gs.ptr)
            )
            var gsh_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gsh.ptr)
            )
            var cx_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                self.cache_x_dev.value()
            )
            var cs_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                self.cache_scale_dev.value()
            )
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _modulate_backward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, cx_lt, cs_lt, gx_lt, gs_lt, gsh_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
