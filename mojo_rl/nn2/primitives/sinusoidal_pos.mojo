"""SinusoidalPosAdd[T, S, D, SCALE] — additive 2D sinusoidal positions.

Dreamer 4 adds separable time + space sinusoidal positions to every token of
the (T, S, D) grid (`model.py:add_sinusoidal_positions`):

    pos[t, s, j] = sinusoid(t)[j] + sinusoid(s)[j]      (optionally / sqrt(D))
    out = tokens + pos

where `sinusoid_table(n, d)[i, j] = sin(i·div_k)` if j even else `cos(i·div_k)`,
`k = floor(j/2)`, `div_k = exp(-(2k/d)·ln(base))`, `base = 10000`.

The bias depends only on (T, S, D), so it is precomputed once at `make` into a
`T*S*D` buffer (host + device, like MaskedAttention's mask) and added in
forward. Backward is the identity (the bias is a constant, no params):

    grad_in = grad_out

Per-sample flat layout is row-major `(T, S, D)`: token (t, s) at offset
`(t*S + s)*D`. `SCALE=True` divides the summed positions by sqrt(D).
"""

from std.math import exp, sin, cos, sqrt, log
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


comptime SINUSOID_BASE: Float64 = 10000.0


def _sinusoid_val(pos: Int, j: Int, d: Int) -> Float64:
    """sinusoid_table(n, d)[pos, j] (single entry)."""
    var k = Float64(j // 2)
    var div = exp(-(2.0 * k) / Float64(d) * log(SINUSOID_BASE))
    var ang = Float64(pos) * div
    return sin(ang) if (j % 2) == 0 else cos(ang)


def build_sinusoid_bias[
    T: Int, S: Int, D: Int, SCALE: Bool
]() -> List[Scalar[DT]]:
    """Precompute the additive (T*S*D) position bias."""
    var scale = (1.0 / sqrt(Float64(D))) if SCALE else 1.0
    var bias = List[Scalar[DT]]()
    for t in range(T):
        for s in range(S):
            for j in range(D):
                var v = _sinusoid_val(t, j, D) + _sinusoid_val(s, j, D)
                bias.append(Scalar[DT](v * scale))
    return bias^


def _pos_add_kernel[
    BATCH: Int, N: Int
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, N), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * N:
        return
    var i = idx % N
    output.ptr[idx] = rebind[Scalar[DT]](input.ptr[idx]) + rebind[Scalar[DT]](
        bias.ptr[i]
    )


def _copy_kernel[
    BATCH: Int, N: Int
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * N:
        return
    dst.ptr[idx] = rebind[Scalar[DT]](src.ptr[idx])


struct SinusoidalPosAdd[T: Int, S: Int, D: Int, SCALE: Bool = False](Module):
    comptime ARITY: Int = 1
    comptime N: Int = Self.T * Self.S * Self.D
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.N)
    comptime OUT_DIM = Self.N

    var bias: List[Scalar[DT]]
    var bias_dev: Optional[DeviceBuffer[DT]]
    var ts: TargetStorage

    def __init__(out self):
        self.bias = List[Scalar[DT]]()
        self.bias_dev = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SinusoidalPosAdd: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.bias = build_sinusoid_bias[Self.T, Self.S, Self.D, Self.SCALE]()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["SinusoidalPosAdd.make[target='gpu']"](ctx)
            m.ts = TargetStorage.make_gpu(ctx_v)
            var dev = ctx_v.enqueue_create_buffer[DT](Self.N)
            var host = ctx_v.enqueue_create_host_buffer[DT](Self.N)
            ctx_v.synchronize()
            var hp = host.unsafe_ptr()
            for i in range(Self.N):
                hp[i] = m.bias[i]
            ctx_v.enqueue_copy(dev, host)
            m.bias_dev = dev^
        return m^

    @staticmethod
    def display_label() -> String:
        return String("SinusoidalPosAdd")

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
        assert_tag_for["SinusoidalPosAdd", target](self.ts.target_tag)
        var inp = typed_view[BATCH, Self.N](inputs[0])
        var out = typed_view_mut[BATCH, Self.N](output)
        comptime if target == "cpu":
            var bp = self.bias.unsafe_ptr()
            for b in range(BATCH):
                for i in range(Self.N):
                    out[b, i] = inp[b, i] + bp[i]
        else:
            comptime lay = Layout.row_major(BATCH, Self.N)
            var in_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inp.ptr)
            )
            var o_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.ptr)
            )
            var b_lt = LayoutTensor[DT, Layout.row_major(Self.N), MutAnyOrigin](
                self.bias_dev.value()
            )
            comptime n_blocks = (BATCH * Self.N + TPB - 1) // TPB
            comptime kernel = _pos_add_kernel[BATCH, Self.N]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, b_lt, o_lt, grid_dim=n_blocks, block_dim=TPB
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
        assert_tag_for["SinusoidalPosAdd", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.N](grad_output)
        var gi = typed_view_mut[BATCH, Self.N](grad_inputs[0])
        comptime if target == "cpu":
            for b in range(BATCH):
                for i in range(Self.N):
                    gi[b, i] = go[b, i]
        else:
            comptime lay = Layout.row_major(BATCH, Self.N)
            var go_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr)
            )
            var gi_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi.ptr)
            )
            comptime n_blocks = (BATCH * Self.N + TPB - 1) // TPB
            comptime kernel = _copy_kernel[BATCH, Self.N]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB
            )
