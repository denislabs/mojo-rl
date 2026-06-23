"""Concat[*DIMS] — variadic feature-axis concatenation (storage surface).

  inputs (in0 [BATCH, DIMS[0]], in1 [BATCH, DIMS[1]], …)  →  [BATCH, ΣDIMS]
  out[b, off_i : off_i+DIMS[i]] = in_i[b]      off_i = Σ_{j<i} DIMS[j]  (comptime)

`ARITY = DIMS.size` (≥ 2), `OUT_DIM = Σ DIMS[i]`. Mirrors the legacy variadic
`Concat[*DIMS]`; the storage `TensorRefs[ARITY]` input pack makes the N-ary form
as clean as a binary one (the legacy framework only had `Concat2` here, which
forced 3+-input concats to be expressed as a chain — this restores the variadic
primitive). The SAC critic's `concat(state, action)` is just N=2 (`Concat2`,
the parametric alias below); tdmpc2 / redq_ofe graphs use N=2…4.

No params, no cache (backward is a pure slice-split of grad_output). GPU forward
launches N small slab-copy kernels (one per input); backward symmetric.

Backward: grad_in_i[b, :] = grad_output[b, off_i : off_i+DIMS[i]].
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── comptime variadic helpers (mirror Parallel / legacy concat) ─────────
def _total_dim[*DIMS: Int]() -> Int:
    var s = 0
    comptime for i in range(DIMS.size):
        s += DIMS[i]
    return s


def _cum_offset[index: Int, *DIMS: Int]() -> Int:
    var s = 0
    comptime for j in range(index):
        s += DIMS[j]
    return s


def _build_in_dims[*DIMS: Int]() -> InlineArray[Int, DIMS.size]:
    var d = InlineArray[Int, DIMS.size](fill=0)
    comptime for k in range(DIMS.size):
        d[k] = DIMS[k]
    return d


# ── per-slab copy kernels (one launch per input; legacy-style) ──────────
def _concat_copy_in_kernel[
    BATCH: Int, SRC_DIM: Int, OUT_DIM: Int, DST_OFF: Int
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, SRC_DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * SRC_DIM
    if idx < total:
        var b = idx // SRC_DIM
        var d = idx % SRC_DIM
        output[b, DST_OFF + d] = rebind[Scalar[DT]](src[b, d])


def _concat_copy_out_kernel[
    BATCH: Int, DST_DIM: Int, OUT_DIM: Int, SRC_OFF: Int
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    grad_in: LayoutTensor[DT, Layout.row_major(BATCH, DST_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DST_DIM
    if idx < total:
        var b = idx // DST_DIM
        var d = idx % DST_DIM
        grad_in[b, d] = rebind[Scalar[DT]](grad_output[b, SRC_OFF + d])


struct Concat[*DIMS: Int](Module):
    comptime ARITY = Self.DIMS.size
    comptime IN_DIMS = _build_in_dims[*Self.DIMS]()
    comptime OUT_DIM = _total_dim[*Self.DIMS]()

    @staticmethod
    def display_label() -> String:
        return String("Concat")

    def __init__(out self):
        comptime assert Self.DIMS.size >= 2, "Concat: needs at least 2 inputs"

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime OUT = Self.OUT_DIM
        comptime if target == "cpu":
            out.ensure(B * OUT)
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                ref in_i = inputs[i]
                for b in range(B):
                    for d in range(D):
                        out.data[b * OUT + OFF + d] = in_i.data[b * D + d]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * OUT)
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                comptime nblk = (B * D + TPB - 1) // TPB
                ref in_i = inputs[i]
                c.enqueue_function[
                    _concat_copy_in_kernel[B, D, OUT, OFF]
                ](
                    in_i.lt["gpu", Layout.row_major(B, D)](),
                    out.lt["gpu", Layout.row_major(B, OUT)](),
                    grid_dim=nblk,
                    block_dim=TPB,
                )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime OUT = Self.OUT_DIM
        comptime if target == "cpu":
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                ref gi = grad_inputs[i]
                gi.ensure(B * D)
                for b in range(B):
                    for d in range(D):
                        gi.data[b * D + d] = grad_output.data[b * OUT + OFF + d]
        else:
            var c = ctx.value()
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                comptime nblk = (B * D + TPB - 1) // TPB
                ref gi = grad_inputs[i]
                gi.ensure_gpu(c, B * D)
                c.enqueue_function[
                    _concat_copy_out_kernel[B, D, OUT, OFF]
                ](
                    grad_output.lt["gpu", Layout.row_major(B, OUT)](),
                    gi.lt["gpu", Layout.row_major(B, D)](),
                    grid_dim=nblk,
                    block_dim=TPB,
                )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).


# ── Concat2[D0, D1] — parametric alias kept for call-site compatibility ──
# The legacy storage surface exposed a fixed binary `Concat2`; it is now just
# the N=2 instance of the variadic `Concat` (mirrors `BranchConcat = Parallel`).
comptime Concat2[D0_: Int, D1_: Int] = Concat[D0_, D1_]
