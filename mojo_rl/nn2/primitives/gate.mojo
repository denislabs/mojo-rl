"""Gate[DIM] — AdaLN-zero residual gate (ARITY=3).

    y = x + gate * branch

Three separate inputs:
    inputs[0] = x       [BATCH, DIM]   (residual stream)
    inputs[1] = gate    [BATCH, DIM]   (modulation gate from AdaLN)
    inputs[2] = branch  [BATCH, DIM]   (sub-layer output)

Gradients:
    grad_x[i]      = grad_out[i]
    grad_gate[i]   = grad_out[i] * branch[i]
    grad_branch[i] = grad_out[i] * gate[i]

Cache (leaf-owned): gate and branch. PARAM-free. With AdaLN zero-init the
gate starts at 0 → block is identity at init (the LeWM correctness
invariant). See docs/LEWM_NN2_PORT_PLAN.md §2.2.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


def _gate_forward_kernel[
    BATCH: Int, DIM: Int,
](
    x: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gate: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    branch: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_gate: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_branch: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var i = idx % DIM
    var xv = rebind[Scalar[DT]](x[b, i])
    var gv = rebind[Scalar[DT]](gate[b, i])
    var brv = rebind[Scalar[DT]](branch[b, i])
    cache_gate[b, i] = gv
    cache_branch[b, i] = brv
    output[b, i] = xv + gv * brv


def _gate_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_gate: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_branch: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_x: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_gate: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_branch: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * DIM:
        return
    var b = idx // DIM
    var i = idx % DIM
    var go = rebind[Scalar[DT]](grad_output[b, i])
    var gv = rebind[Scalar[DT]](cache_gate[b, i])
    var brv = rebind[Scalar[DT]](cache_branch[b, i])
    grad_x[b, i] = go
    grad_gate[b, i] = go * brv
    grad_branch[b, i] = go * gv


struct Gate[DIM: Int](Module):
    comptime ARITY: Int = 3
    comptime IN_DIMS = InlineArray[Int, 3](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    @staticmethod
    def display_label() -> String:
        return String("Gate")

    var cache_gate: List[Scalar[DT]]
    var cache_branch: List[Scalar[DT]]
    var cache_gate_dev: Optional[DeviceBuffer[DT]]
    var cache_branch_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int
    var ts: TargetStorage

    def __init__(out self):
        self.cache_gate = List[Scalar[DT]]()
        self.cache_branch = List[Scalar[DT]]()
        self.cache_gate_dev = None
        self.cache_branch_dev = None
        self.cache_n_batch = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Gate: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Gate.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            m.cache_gate_dev = ctx_v.enqueue_create_buffer[DT](1)
            m.cache_branch_dev = ctx_v.enqueue_create_buffer[DT](1)
            m.cache_n_batch = 0
            m.ts = TargetStorage.make_gpu(ctx_v)
        return m^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_gate_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.DIM
            )
            self.cache_branch_dev = ctx.enqueue_create_buffer[DT](
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
        assert_tag_for["Gate", target](self.ts.target_tag)
        var x = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var gate = typed_view[BATCH, Self.IN_DIMS[1]](inputs[1])
        var branch = typed_view[BATCH, Self.IN_DIMS[2]](inputs[2])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            ensure_cpu_buffer(self.cache_gate, BATCH * Self.DIM)
            ensure_cpu_buffer(self.cache_branch, BATCH * Self.DIM)
            var cg = TileTensor(self.cache_gate, row_major[BATCH, Self.DIM]())
            var cb = TileTensor(self.cache_branch, row_major[BATCH, Self.DIM]())
            for b in range(BATCH):
                for i in range(Self.DIM):
                    var gv = gate[b, i]
                    var brv = branch[b, i]
                    cg[b, i] = gv
                    cb[b, i] = brv
                    out[b, i] = x[b, i] + gv * brv
        else:
            self._ensure_cache_gpu(BATCH)
            comptime lay = Layout.row_major(BATCH, Self.DIM)
            var x_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x.ptr)
            )
            var g_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gate.ptr)
            )
            var br_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](branch.ptr)
            )
            var o_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.ptr)
            )
            var cg_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                self.cache_gate_dev.value()
            )
            var cb_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                self.cache_branch_dev.value()
            )
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _gate_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                x_lt, g_lt, br_lt, o_lt, cg_lt, cb_lt,
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
        assert_tag_for["Gate", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gx = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])
        var gg = typed_view_mut[BATCH, Self.IN_DIMS[1]](grad_inputs[1])
        var gbr = typed_view_mut[BATCH, Self.IN_DIMS[2]](grad_inputs[2])

        comptime if target == "cpu":
            var cg = TileTensor(self.cache_gate, row_major[BATCH, Self.DIM]())
            var cb = TileTensor(self.cache_branch, row_major[BATCH, Self.DIM]())
            for b in range(BATCH):
                for i in range(Self.DIM):
                    var g = go[b, i]
                    gx[b, i] = g
                    gg[b, i] = g * cb[b, i]
                    gbr[b, i] = g * cg[b, i]
        else:
            comptime lay = Layout.row_major(BATCH, Self.DIM)
            var go_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr)
            )
            var gx_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gx.ptr)
            )
            var gg_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gg.ptr)
            )
            var gbr_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gbr.ptr)
            )
            var cg_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                self.cache_gate_dev.value()
            )
            var cb_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                self.cache_branch_dev.value()
            )
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _gate_backward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, cg_lt, cb_lt, gx_lt, gg_lt, gbr_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
