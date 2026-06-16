"""QKVToMajor[SEQ, DIM] — token-major QKV → qkv-major, for attention.

A QKV projection `Tokenwise[Linear[DIM, 3*DIM]]` emits token-major output:
per sample `[tok0: q(DIM) k(DIM) v(DIM) | tok1: q k v | …]`, i.e. flat index
`t*3*DIM + g*DIM + d` for group g∈{q,k,v}. `ScaledDotProductAttention` expects
qkv-major: `[all-Q | all-K | all-V]`, flat index `g*SEQ*DIM + t*DIM + d`. This
op rearranges the former into the latter (a transpose of the (SEQ,3) axes with
DIM as the contiguous block):

    out[g*SEQ*DIM + t*DIM + d] = in[t*3*DIM + g*DIM + d]

IN_DIM == OUT_DIM == 3*SEQ*DIM; no params, no cache. Backward is the inverse
permutation. Without this, feeding token-major straight into SDPA scrambles the
position axis and breaks causal masking (future-token leak).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for


def _qkv_to_major_fwd_kernel[
    BATCH: Int, SEQ: Int, DIM: Int
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, 3 * SEQ * DIM), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, 3 * SEQ * DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * 3 * SEQ * DIM
    if gid >= total:
        return
    var b = gid // (3 * SEQ * DIM)
    var o = gid % (3 * SEQ * DIM)          # qkv-major out index
    var g = o // (SEQ * DIM)
    var rem = o % (SEQ * DIM)
    var t = rem // DIM
    var d = rem % DIM
    dst[b, o] = rebind[Scalar[DT]](src[b, t * 3 * DIM + g * DIM + d])


def _qkv_to_major_bwd_kernel[
    BATCH: Int, SEQ: Int, DIM: Int
](
    grad_out: LayoutTensor[DT, Layout.row_major(BATCH, 3 * SEQ * DIM), MutAnyOrigin],
    grad_in: LayoutTensor[DT, Layout.row_major(BATCH, 3 * SEQ * DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * 3 * SEQ * DIM
    if gid >= total:
        return
    var b = gid // (3 * SEQ * DIM)
    var o = gid % (3 * SEQ * DIM)          # qkv-major out index
    var g = o // (SEQ * DIM)
    var rem = o % (SEQ * DIM)
    var t = rem // DIM
    var d = rem % DIM
    grad_in[b, t * 3 * DIM + g * DIM + d] = rebind[Scalar[DT]](grad_out[b, o])


struct QKVToMajor[SEQ: Int, DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=3 * Self.SEQ * Self.DIM)
    comptime OUT_DIM = 3 * Self.SEQ * Self.DIM

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.SEQ > 0 and Self.DIM > 0, (
            "QKVToMajor: SEQ, DIM must be > 0"
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "QKVToMajor: target must be 'cpu' or 'gpu'"
        )
        var q = Self()
        comptime if target == "cpu":
            q.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("QKVToMajor.make[target='gpu']: ctx required")
            q.ts = TargetStorage.make_gpu(ctx.value())
        return q^

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
        assert_tag_for["QKVToMajor", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)
        comptime D3 = 3 * Self.DIM
        comptime SD = Self.SEQ * Self.DIM

        comptime if target == "cpu":
            for b in range(BATCH):
                for g in range(3):
                    for t in range(Self.SEQ):
                        for d in range(Self.DIM):
                            output_v[b, g * SD + t * Self.DIM + d] = input[
                                b, t * D3 + g * Self.DIM + d
                            ]
        else:
            comptime lay = Layout.row_major(BATCH, 3 * Self.SEQ * Self.DIM)
            var in_p = input.ptr
            var out_p = output_v.ptr
            var in_lt = LayoutTensor[DT, lay, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, lay, MutAnyOrigin](out_p)
            comptime total = BATCH * 3 * Self.SEQ * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _qkv_to_major_fwd_kernel[BATCH, Self.SEQ, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["QKVToMajor", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        comptime D3 = 3 * Self.DIM
        comptime SD = Self.SEQ * Self.DIM

        comptime if target == "cpu":
            for b in range(BATCH):
                for g in range(3):
                    for t in range(Self.SEQ):
                        for d in range(Self.DIM):
                            grad_input_v[b, t * D3 + g * Self.DIM + d] = (
                                grad_output_v[b, g * SD + t * Self.DIM + d]
                            )
        else:
            comptime lay = Layout.row_major(BATCH, 3 * Self.SEQ * Self.DIM)
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, lay, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, lay, MutAnyOrigin](gi_p)
            comptime total = BATCH * 3 * Self.SEQ * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _qkv_to_major_bwd_kernel[BATCH, Self.SEQ, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
