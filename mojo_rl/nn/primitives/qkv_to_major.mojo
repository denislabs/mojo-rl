"""QKVToMajor[SEQ, DIM] — token-major QKV → qkv-major (storage surface).

Transformed from legacy `nn.primitives.qkv_to_major` (surface-only change; the
CPU permutation loops and the GPU fwd/bwd kernels are carried over VERBATIM).

A QKV projection `Tokenwise[Linear[DIM, 3*DIM]]` emits token-major output:
per sample `[tok0: q(DIM) k(DIM) v(DIM) | tok1: q k v | …]`, i.e. flat index
`t*3*DIM + g*DIM + d` for group g∈{q,k,v}. `ScaledDotProductAttention` expects
qkv-major: `[all-Q | all-K | all-V]`, flat index `g*SEQ*DIM + t*DIM + d`. This
op rearranges the former into the latter:

    out[g*SEQ*DIM + t*DIM + d] = in[t*3*DIM + g*DIM + d]

IN_DIM == OUT_DIM == 3*SEQ*DIM; no params, no cache. Backward is the inverse
permutation. Conforms to `Module` (param-less → reflection no-op walks).
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


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ───────
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

    def __init__(out self):
        comptime assert Self.SEQ > 0 and Self.DIM > 0, (
            "QKVToMajor: SEQ, DIM must be > 0"
        )

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "QKVToMajor: target must be 'cpu' or 'gpu'"
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
        comptime N = B * Self.OUT_DIM
        comptime D3 = 3 * Self.DIM
        comptime SD = Self.SEQ * Self.DIM
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            ref ip = in0.data
            ref op = out.data
            for b in range(B):
                for g in range(3):
                    for t in range(Self.SEQ):
                        for d in range(Self.DIM):
                            op[b * Self.OUT_DIM + g * SD + t * Self.DIM + d] = (
                                ip[b * Self.IN_DIMS[0] + t * D3 + g * Self.DIM + d]
                            )
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime lay = Layout.row_major(B, 3 * Self.SEQ * Self.DIM)
            comptime total = B * 3 * Self.SEQ * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _qkv_to_major_fwd_kernel[B, Self.SEQ, Self.DIM]
            c.enqueue_function[kernel](
                in0.lt["gpu", lay](),
                out.lt["gpu", lay](),
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
        comptime N = B * Self.IN_DIMS[0]
        comptime D3 = 3 * Self.DIM
        comptime SD = Self.SEQ * Self.DIM
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            ref gop = grad_output.data
            ref gip = gin.data
            for b in range(B):
                for g in range(3):
                    for t in range(Self.SEQ):
                        for d in range(Self.DIM):
                            gip[b * Self.IN_DIMS[0] + t * D3 + g * Self.DIM + d] = (
                                gop[b * Self.OUT_DIM + g * SD + t * Self.DIM + d]
                            )
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime lay = Layout.row_major(B, 3 * Self.SEQ * Self.DIM)
            comptime total = B * 3 * Self.SEQ * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _qkv_to_major_bwd_kernel[B, Self.SEQ, Self.DIM]
            c.enqueue_function[kernel](
                grad_output.lt["gpu", lay](),
                gin.lt["gpu", lay](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
