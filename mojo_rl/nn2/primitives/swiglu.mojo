"""SwiGLU[HIDDEN] — gated-linear-unit activation core (Dreamer 4 FFN).

The reference Dreamer 4 MLP is `fc_in: Linear(d, 2*hidden)` → split into
`(u, v)` → `u * silu(v)` → `fc_out: Linear(hidden, d)`. This leaf is the
middle (parameter-free) gate, so the full FFN composes as

    Sequential[Linear[d, 2*hidden], SwiGLU[hidden], Linear[hidden, d]]

keeping the two Linears reusable. Input is the concatenated projection
`[u ‖ v]` (each `HIDDEN` wide), output is `u · silu(v)`:

    IN_DIM = 2*HIDDEN, OUT_DIM = HIDDEN
    silu(v)  = v·σ(v),  σ(v) = 1/(1+e^-v)
    silu'(v) = σ(v)·(1 + v·(1-σ(v)))
    out[k]      = u[k] · silu(v[k])
    grad_u[k]   = grad_out[k] · silu(v[k])
    grad_v[k]   = grad_out[k] · u[k] · silu'(v[k])

Param-free, output-cached (`u`, `v`). Elementwise → CPU + GPU one-kernel.
"""

from std.math import exp
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, Cache
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for, ensure_cpu_buffer


def _swiglu_forward_kernel[
    BATCH: Int, HIDDEN: Int
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, 2 * HIDDEN), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    cache_u: LayoutTensor[DT, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    cache_v: LayoutTensor[DT, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * HIDDEN:
        return
    var b = idx // HIDDEN
    var k = idx % HIDDEN
    var u = rebind[Scalar[DT]](input[b, k])
    var v = rebind[Scalar[DT]](input[b, HIDDEN + k])
    var s = Scalar[DT](1) / (Scalar[DT](1) + exp(-v))
    cache_u[b, k] = u
    cache_v[b, k] = v
    output[b, k] = u * (v * s)


def _swiglu_backward_kernel[
    BATCH: Int, HIDDEN: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    cache_u: LayoutTensor[DT, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    cache_v: LayoutTensor[DT, Layout.row_major(BATCH, HIDDEN), MutAnyOrigin],
    grad_input: LayoutTensor[
        DT, Layout.row_major(BATCH, 2 * HIDDEN), MutAnyOrigin
    ],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * HIDDEN:
        return
    var b = idx // HIDDEN
    var k = idx % HIDDEN
    var go = rebind[Scalar[DT]](grad_output[b, k])
    var u = rebind[Scalar[DT]](cache_u[b, k])
    var v = rebind[Scalar[DT]](cache_v[b, k])
    var s = Scalar[DT](1) / (Scalar[DT](1) + exp(-v))
    var sv = v * s
    var sp = s * (Scalar[DT](1) + v * (Scalar[DT](1) - s))
    grad_input[b, k] = go * sv
    grad_input[b, HIDDEN + k] = go * u * sp


struct SwiGLU[HIDDEN: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=2 * Self.HIDDEN)
    comptime OUT_DIM = Self.HIDDEN

    var cache_u: Cache["cache_u"]
    var cache_v: Cache["cache_v"]
    var ts: TargetStorage

    def __init__(out self):
        self.cache_u = Cache["cache_u"]()
        self.cache_v = Cache["cache_v"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SwiGLU: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["SwiGLU.make[target='gpu']"](ctx)
            m.ts = TargetStorage.make_gpu(ctx_v)
        return m^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.cache_u.ensure_gpu(ctx, batch * Self.HIDDEN)
        self.cache_v.ensure_gpu(ctx, batch * Self.HIDDEN)
    def display_label() -> String:
        return String("SwiGLU")

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
        assert_tag_for["SwiGLU", target](self.ts.target_tag)
        var inp = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self.cache_u.ensure_cpu(BATCH * Self.HIDDEN)
            self.cache_v.ensure_cpu(BATCH * Self.HIDDEN)
            var cu = TileTensor(self.cache_u.cpu, row_major[BATCH, Self.HIDDEN]())
            var cv = TileTensor(self.cache_v.cpu, row_major[BATCH, Self.HIDDEN]())
            for b in range(BATCH):
                for k in range(Self.HIDDEN):
                    var u = inp[b, k]
                    var v = inp[b, Self.HIDDEN + k]
                    var s = Scalar[DT](1) / (Scalar[DT](1) + exp(-v))
                    cu[b, k] = u
                    cv[b, k] = v
                    out[b, k] = u * (v * s)
        else:
            self._ensure_cache_gpu(BATCH)
            comptime lin = Layout.row_major(BATCH, 2 * Self.HIDDEN)
            comptime lout = Layout.row_major(BATCH, Self.HIDDEN)
            var in_lt = LayoutTensor[DT, lin, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inp.ptr)
            )
            var o_lt = LayoutTensor[DT, lout, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.ptr)
            )
            var cu_lt = LayoutTensor[DT, lout, MutAnyOrigin](
                self.cache_u.dev.value()
            )
            var cv_lt = LayoutTensor[DT, lout, MutAnyOrigin](
                self.cache_v.dev.value()
            )
            comptime n_blocks = (BATCH * Self.HIDDEN + TPB - 1) // TPB
            comptime kernel = _swiglu_forward_kernel[BATCH, Self.HIDDEN]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, o_lt, cu_lt, cv_lt, grid_dim=n_blocks, block_dim=TPB
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
        assert_tag_for["SwiGLU", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])

        comptime if target == "cpu":
            var cu = TileTensor(self.cache_u.cpu, row_major[BATCH, Self.HIDDEN]())
            var cv = TileTensor(self.cache_v.cpu, row_major[BATCH, Self.HIDDEN]())
            for b in range(BATCH):
                for k in range(Self.HIDDEN):
                    var g = go[b, k]
                    var u = cu[b, k]
                    var v = cv[b, k]
                    var s = Scalar[DT](1) / (Scalar[DT](1) + exp(-v))
                    var sv = v * s
                    var sp = s * (Scalar[DT](1) + v * (Scalar[DT](1) - s))
                    gi[b, k] = g * sv
                    gi[b, Self.HIDDEN + k] = g * u * sp
        else:
            comptime lin = Layout.row_major(BATCH, 2 * Self.HIDDEN)
            comptime lout = Layout.row_major(BATCH, Self.HIDDEN)
            var go_lt = LayoutTensor[DT, lout, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr)
            )
            var gi_lt = LayoutTensor[DT, lin, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi.ptr)
            )
            var cu_lt = LayoutTensor[DT, lout, MutAnyOrigin](
                self.cache_u.dev.value()
            )
            var cv_lt = LayoutTensor[DT, lout, MutAnyOrigin](
                self.cache_v.dev.value()
            )
            comptime n_blocks = (BATCH * Self.HIDDEN + TPB - 1) // TPB
            comptime kernel = _swiglu_backward_kernel[BATCH, Self.HIDDEN]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, cu_lt, cv_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB
            )
