"""LearnedTokens[N_IN, N_NEW, D, PREPEND] — concatenate learned tokens.

Dreamer 4's encoder prepends learned latent tokens to the projected patches;
the decoder appends learned patch-query tokens to the up-projected latents.
This leaf does that concat with the learned tokens as its (only) parameter,
shared across the whole B·T batch (every frame gets the same learned tokens):

    PREPEND : out = [ learned(N_NEW) ‖ input(N_IN) ]   (encoder latents)
    else    : out = [ input(N_IN) ‖ learned(N_NEW) ]   (decoder queries)

IN_DIM = N_IN·D, OUT_DIM = (N_IN+N_NEW)·D, param = N_NEW·D. The param grad is
batch-reduced (the tokens are shared): grad_tokens[k] = Σ_bt grad_out[bt, new+k].
CPU + GPU.
"""

from std.gpu import global_idx, thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import sqrt as fsqrt, log, cos, sin
from std.random import random_float64
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import (
    Initializer, AMPPolicy, NoAMP, Param, ParamVisitor,
    for_each_param_auto, zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


comptime LT_RTPB = 64  # reduction block size for the param-grad batch sum
comptime _LT_TWO_PI: Float64 = 6.283185307179586


def _lt_fill_normal(
    buf: UnsafePointer[Scalar[DT], MutAnyOrigin], n_elems: Int, std: Float64
):
    """N(0, std) Box-Muller fill — the ViT convention for learned CLS/query
    tokens (std≈0.02), independent of fan-in. fan_in=1 Kaiming would give
    std≈1.4, a constant that swamps the per-image attention signal and
    collapses the readout (see LeWMEncoderCLS)."""
    var i = 0
    while i < n_elems:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var r = fsqrt(-2.0 * log(u1))
        buf[i] = Scalar[DT](std * r * cos(_LT_TWO_PI * u2))
        i += 1
        if i < n_elems:
            buf[i] = Scalar[DT](std * r * sin(_LT_TWO_PI * u2))
            i += 1


def _lt_forward_kernel[
    BATCH: Int, IN_N: Int, NEW_N: Int, OUT_DIM: Int, NEW_OFF: Int, IN_OFF: Int
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN_N), MutAnyOrigin],
    param: LayoutTensor[DT, Layout.row_major(NEW_N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_DIM:
        return
    var bt = idx // OUT_DIM
    var pos = idx % OUT_DIM
    if pos >= NEW_OFF and pos < NEW_OFF + NEW_N:
        output.ptr[idx] = rebind[Scalar[DT]](param.ptr[pos - NEW_OFF])
    else:
        output.ptr[idx] = rebind[Scalar[DT]](input.ptr[bt * IN_N + (pos - IN_OFF)])


def _lt_grad_input_kernel[
    BATCH: Int, IN_N: Int, OUT_DIM: Int, IN_OFF: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN_N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * IN_N:
        return
    var bt = idx // IN_N
    var k = idx % IN_N
    grad_input.ptr[idx] = rebind[Scalar[DT]](
        grad_output.ptr[bt * OUT_DIM + IN_OFF + k]
    )


def _lt_grad_param_kernel[
    BATCH: Int, NEW_N: Int, OUT_DIM: Int, NEW_OFF: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    grad_param: LayoutTensor[DT, Layout.row_major(NEW_N), MutAnyOrigin],
):
    # One block per param element; threads reduce over the batch.
    var col = Int(block_idx.x)
    if col >= NEW_N:
        return
    var t = Int(thread_idx.x)
    var acc: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        acc += rebind[Scalar[DT]](grad_output.ptr[bi * OUT_DIM + NEW_OFF + col])
        bi += LT_RTPB
    var total = block.sum[block_size=LT_RTPB, broadcast=False](val=acc)
    if t == 0:
        grad_param.ptr[col] = rebind[Scalar[DT]](grad_param.ptr[col]) + total[0]


struct LearnedTokens[
    N_IN: Int, N_NEW: Int, D: Int, PREPEND: Bool, INIT_STD: Float64 = 0.0
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.N_IN * Self.D)
    comptime OUT_DIM = (Self.N_IN + Self.N_NEW) * Self.D
    comptime NEW_N: Int = Self.N_NEW * Self.D
    comptime IN_N: Int = Self.N_IN * Self.D
    comptime NEW_OFF: Int = 0 if Self.PREPEND else Self.IN_N
    comptime IN_OFF: Int = Self.NEW_N if Self.PREPEND else 0

    var tokens: Param["tokens", False, Self.NEW_N]
    var ts: TargetStorage

    def __init__(out self):
        self.tokens = Param["tokens", False, Self.NEW_N]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "LearnedTokens: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime if target == "cpu":
            m.tokens = Param["tokens", False, Self.NEW_N].make_cpu()
            comptime if Self.INIT_STD > 0.0:
                _lt_fill_normal(
                    m.tokens.value_unsafe_ptr_cpu(), Self.NEW_N, Self.INIT_STD
                )
            else:
                INIT.init_weight(
                    m.tokens.value_unsafe_ptr_cpu(), Self.NEW_N, Self.N_NEW,
                    Self.D,
                )
            m.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["LearnedTokens.make[gpu]"](ctx)
            m.tokens = Param["tokens", False, Self.NEW_N].make_gpu(ctx_v)
            var host = ctx_v.enqueue_create_host_buffer[DT](Self.NEW_N)
            ctx_v.synchronize()
            comptime if Self.INIT_STD > 0.0:
                _lt_fill_normal(mptr(host.unsafe_ptr()), Self.NEW_N, Self.INIT_STD)
            else:
                INIT.init_weight(
                    host.unsafe_ptr(), Self.NEW_N, Self.N_NEW, Self.D
                )
            ctx_v.enqueue_copy(m.tokens.val.dev.value(), host)
            ctx_v.synchronize()
            m.ts = TargetStorage.make_gpu(ctx_v)
        return m^

    @staticmethod
    def display_label() -> String:
        return String("LearnedTokens")

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
        assert_tag_for["LearnedTokens", target](self.ts.target_tag)
        var inp = inputs.tile[0, BATCH, Self.IN_N]()
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        comptime if target == "cpu":
            var tok = TileTensor(self.tokens.val.cpu, row_major[Self.NEW_N]())
            for bt in range(BATCH):
                for k in range(Self.NEW_N):
                    out[bt, Self.NEW_OFF + k] = tok[k]
                for k in range(Self.IN_N):
                    out[bt, Self.IN_OFF + k] = inp[bt, k]
        else:
            var in_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN_N), MutAnyOrigin
            ](inp.ptr)
            var o_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ](out.ptr)
            var p_lt = LayoutTensor[
                DT, Layout.row_major(Self.NEW_N), MutAnyOrigin
            ](self.tokens.val.dev.value())
            comptime n_blocks = (BATCH * Self.OUT_DIM + TPB - 1) // TPB
            comptime kernel = _lt_forward_kernel[
                BATCH, Self.IN_N, Self.NEW_N, Self.OUT_DIM,
                Self.NEW_OFF, Self.IN_OFF,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, p_lt, o_lt, grid_dim=n_blocks, block_dim=TPB
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
        assert_tag_for["LearnedTokens", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = grad_inputs.tile[0, BATCH, Self.IN_N]()
        comptime if target == "cpu":
            for bt in range(BATCH):
                for k in range(Self.IN_N):
                    gi[bt, k] = go[bt, Self.IN_OFF + k]
            comptime if mode == "all":
                var gtok = TileTensor(self.tokens.grd.cpu, row_major[Self.NEW_N]())
                for bt in range(BATCH):
                    for k in range(Self.NEW_N):
                        gtok[k] += go[bt, Self.NEW_OFF + k]
        else:
            var ctx = self.ts.ctx.value()
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ](go.ptr)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN_N), MutAnyOrigin
            ](gi.ptr)
            comptime gin_blocks = (BATCH * Self.IN_N + TPB - 1) // TPB
            comptime gik = _lt_grad_input_kernel[
                BATCH, Self.IN_N, Self.OUT_DIM, Self.IN_OFF
            ]
            ctx.enqueue_function[gik](
                go_lt, gi_lt, grid_dim=gin_blocks, block_dim=TPB
            )
            comptime if mode == "all":
                var gp_lt = LayoutTensor[
                    DT, Layout.row_major(Self.NEW_N), MutAnyOrigin
                ](self.tokens.grd.dev.value())
                comptime gpk = _lt_grad_param_kernel[
                    BATCH, Self.NEW_N, Self.OUT_DIM, Self.NEW_OFF
                ]
                ctx.enqueue_function[gpk](
                    go_lt, gp_lt, grid_dim=Self.NEW_N, block_dim=LT_RTPB
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["LearnedTokens", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["LearnedTokens", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
