"""SinusoidalPosAddBT[T, S, D, SCALE] — sinusoidal positions at the B·T layout.

`SinusoidalPosAdd` assumes nn2-BATCH = B and a per-sample (T·S·D) grid. The
Dreamer 4 encoder/decoder instead run at nn2-BATCH = B·T (one frame per
sample, sequence S), where the additive position `pos_t[t] + pos_s[s]` varies
with `t = batch_index % T`. This leaf adds it at that layout:

    out[bt, s*D + j] = in[bt, s*D + j] + bias[(bt % T)*S*D + s*D + j]

where `bias` is the same precomputed `T*S*D` table as `SinusoidalPosAdd`
(`build_sinusoid_bias`). Param-free; identity vjp (the bias is constant).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for
from .sinusoidal_pos import build_sinusoid_bias


def _pos_bt_add_kernel[
    BATCH: Int, T: Int, SD: Int
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, SD), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(T * SD), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, SD), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * SD:
        return
    var bt = idx // SD
    var local = idx % SD
    var t = bt % T
    output.ptr[idx] = rebind[Scalar[DT]](input.ptr[idx]) + rebind[Scalar[DT]](
        bias.ptr[t * SD + local]
    )


def _pos_bt_copy_kernel[BATCH: Int, SD: Int](
    src: LayoutTensor[DT, Layout.row_major(BATCH, SD), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, SD), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < BATCH * SD:
        dst.ptr[idx] = rebind[Scalar[DT]](src.ptr[idx])


struct SinusoidalPosAddBT[T: Int, S: Int, D: Int, SCALE: Bool = False](Module):
    comptime ARITY: Int = 1
    comptime SD: Int = Self.S * Self.D
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SD)
    comptime OUT_DIM = Self.SD

    var bias: List[Scalar[DT]]            # [T*S*D]
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
            "SinusoidalPosAddBT: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.bias = build_sinusoid_bias[Self.T, Self.S, Self.D, Self.SCALE]()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("SinusoidalPosAddBT.make[gpu]: ctx required")
            var ctx_v = ctx.value()
            m.ts = TargetStorage.make_gpu(ctx_v)
            comptime N = Self.T * Self.SD
            var dev = ctx_v.enqueue_create_buffer[DT](N)
            var host = ctx_v.enqueue_create_host_buffer[DT](N)
            ctx_v.synchronize()
            var hp = host.unsafe_ptr()
            for i in range(N):
                hp[i] = m.bias[i]
            ctx_v.enqueue_copy(dev, host)
            m.bias_dev = dev^
        return m^

    @staticmethod
    def display_label() -> String:
        return String("SinusoidalPosAddBT")

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
        assert_tag_for["SinusoidalPosAddBT", target](self.ts.target_tag)
        var inp = typed_view[BATCH, Self.SD](inputs[0])
        var out = typed_view_mut[BATCH, Self.SD](output)
        comptime if target == "cpu":
            var bp = self.bias.unsafe_ptr()
            for bt in range(BATCH):
                var t = bt % Self.T
                for i in range(Self.SD):
                    out[bt, i] = inp[bt, i] + bp[t * Self.SD + i]
        else:
            comptime lay = Layout.row_major(BATCH, Self.SD)
            var in_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inp.ptr)
            )
            var o_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.ptr)
            )
            var b_lt = LayoutTensor[
                DT, Layout.row_major(Self.T * Self.SD), MutAnyOrigin
            ](self.bias_dev.value())
            comptime n_blocks = (BATCH * Self.SD + TPB - 1) // TPB
            comptime kernel = _pos_bt_add_kernel[BATCH, Self.T, Self.SD]
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
        assert_tag_for["SinusoidalPosAddBT", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.SD](grad_output)
        var gi = typed_view_mut[BATCH, Self.SD](grad_inputs[0])
        comptime if target == "cpu":
            for bt in range(BATCH):
                for i in range(Self.SD):
                    gi[bt, i] = go[bt, i]
        else:
            comptime lay = Layout.row_major(BATCH, Self.SD)
            var go_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr)
            )
            var gi_lt = LayoutTensor[DT, lay, MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi.ptr)
            )
            comptime n_blocks = (BATCH * Self.SD + TPB - 1) // TPB
            comptime kernel = _pos_bt_copy_kernel[BATCH, Self.SD]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB
            )
