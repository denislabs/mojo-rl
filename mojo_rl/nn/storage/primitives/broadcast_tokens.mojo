"""BroadcastTokens[N, DIM] — replicate one vector across N tokens (storage).

Transformed from legacy `nn.primitives.BroadcastTokens` (surface-only change).
The CPU loops + the two GPU kernels are carried over verbatim.

    out[b, t*DIM + d] = in[b, d]      (t ∈ [0, N))

The exact adjoint of `TokenMean[N, DIM]` (up to the 1/N factor): forward
fans a single `DIM`-vector out to `N` identical tokens; backward sums the
per-token gradients back onto the source:

    grad_in[b, d] = sum_t grad_out[b, t*DIM + d]

IN_DIM = DIM, OUT_DIM = N*DIM; no params, no cache. Used by the LeWM decoder
(broadcast the global/pooled rep to all patch-query positions), EZv2 and
muzero. Conforms to `Module`. CPU + GPU.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _bt_fwd_kernel[
    BATCH: Int, N: Int, DIM: Int
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, N * DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * N * DIM
    if gid >= total:
        return
    var b = gid // (N * DIM)
    var rem = gid % (N * DIM)
    var d = rem % DIM
    dst[b, rem] = rebind[Scalar[DT]](src[b, d])


def _bt_bwd_kernel[
    BATCH: Int, N: Int, DIM: Int
](
    go: LayoutTensor[DT, Layout.row_major(BATCH, N * DIM), MutAnyOrigin],
    gi: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * DIM
    if gid >= total:
        return
    var b = gid // DIM
    var d = gid % DIM
    var acc: Scalar[DT] = 0.0
    for t in range(N):
        acc += rebind[Scalar[DT]](go[b, t * DIM + d])
    gi[b, d] = acc


struct BroadcastTokens[N_: Int, DIM_: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.N_ * Self.DIM_

    def __init__(out self):
        comptime assert Self.N_ > 0 and Self.DIM_ > 0, (
            "BroadcastTokens: N, DIM must be > 0"
        )

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "BroadcastTokens: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            var input = TileTensor(in0.data, row_major[B, Self.DIM_]())
            var output_v = TileTensor(out.data, row_major[B, Self.OUT_DIM]())
            for b in range(B):
                for t in range(Self.N_):
                    for d in range(Self.DIM_):
                        output_v[b, t * Self.DIM_ + d] = input[b, d]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime total = B * Self.N_ * Self.DIM_
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[_bt_fwd_kernel[B, Self.N_, Self.DIM_]](
                in0.lt["gpu", Layout.row_major(B, Self.DIM_)](),
                out.lt["gpu", Layout.row_major(B, Self.N_ * Self.DIM_)](),
                grid_dim=n_blocks,
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
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            var go = TileTensor(grad_output.data, row_major[B, Self.OUT_DIM]())
            var gi = TileTensor(gin.data, row_major[B, Self.DIM_]())
            for b in range(B):
                for d in range(Self.DIM_):
                    var acc: Scalar[DT] = 0.0
                    for t in range(Self.N_):
                        acc += go[b, t * Self.DIM_ + d]
                    gi[b, d] = acc
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime total = B * Self.DIM_
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[_bt_bwd_kernel[B, Self.N_, Self.DIM_]](
                grad_output.lt["gpu", Layout.row_major(B, Self.N_ * Self.DIM_)](),
                gin.lt["gpu", Layout.row_major(B, Self.DIM_)](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
