"""SpaceTimeTranspose[T, S, D] — swap time/space axes of a token grid (storage).

Transformed from legacy `nn.primitives.space_time_transpose` (surface-only
change; the CPU permutation loops and the GPU `_stt_kernel` are carried over
VERBATIM).

Dreamer 4 factorizes attention into space layers (attend over S tokens per
frame) and time layers (attend over T frames per token). Between them the
(T, S) grid of D-vectors is transposed. This leaf reinterprets each sample's
`T*S*D` slab as a row-major `(T, S, D)` tensor and writes `(S, T, D)`:

    out[b, (s*T + t)*D + d] = in[b, (t*S + s)*D + d]

IN_DIM == OUT_DIM == T*S*D; param-free, cache-free. Backward is the inverse
permutation (the same map with T↔S swapped). Conforms to `Module`.
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


# ── GPU kernel (verbatim from legacy; args MutAnyOrigin = GPU ABI) ────────
def _stt_kernel[
    BATCH: Int, T: Int, S: Int, D: Int, INVERSE: Bool
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, T * S * D), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, T * S * D), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * T * S * D
    if gid >= total:
        return
    var b = gid // (T * S * D)
    var rem = gid % (T * S * D)
    var d = rem % D
    var pos = rem // D                # destination grid position
    comptime if INVERSE:
        # dst grid is (T, S): pos = t*S + s ; read from src (S, T) at s*T + t
        var t = pos // S
        var s = pos % S
        dst[b, gid % (T * S * D)] = rebind[Scalar[DT]](
            src[b, (s * T + t) * D + d]
        )
    else:
        # dst grid is (S, T): pos = s*T + t ; read from src (T, S) at t*S + s
        var s = pos // T
        var t = pos % T
        dst[b, gid % (T * S * D)] = rebind[Scalar[DT]](
            src[b, (t * S + s) * D + d]
        )


struct SpaceTimeTranspose[T: Int, S: Int, D: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.T * Self.S * Self.D)
    comptime OUT_DIM = Self.T * Self.S * Self.D

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "SpaceTimeTranspose: target must be 'cpu' or 'gpu'"
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
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            ref ip = in0.data
            ref op = out.data
            for b in range(B):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            op[
                                b * Self.OUT_DIM + (s * Self.T + t) * Self.D + d
                            ] = ip[
                                b * Self.IN_DIMS[0] + (t * Self.S + s) * Self.D + d
                            ]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            self._run_gpu[B, False](in0, out, c)

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
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            ref gop = grad_output.data
            ref gip = gin.data
            # inverse permutation: grad_in[(t,s)] = grad_out[(s,t)]
            for b in range(B):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            gip[
                                b * Self.IN_DIMS[0] + (t * Self.S + s) * Self.D + d
                            ] = gop[
                                b * Self.OUT_DIM + (s * Self.T + t) * Self.D + d
                            ]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            self._run_gpu[B, True](grad_output, gin, c)

    def _run_gpu[
        B: Int, INVERSE: Bool
    ](mut self, mut src: Tensor, mut dst: Tensor, c: DeviceContext) raises:
        comptime lay = Layout.row_major(B, Self.T * Self.S * Self.D)
        comptime total = B * Self.T * Self.S * Self.D
        comptime n_blocks = (total + TPB - 1) // TPB
        comptime kernel = _stt_kernel[B, Self.T, Self.S, Self.D, INVERSE]
        c.enqueue_function[kernel](
            src.lt["gpu", lay](),
            dst.lt["gpu", lay](),
            grid_dim=n_blocks,
            block_dim=TPB,
        )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults.
