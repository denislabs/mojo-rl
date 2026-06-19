"""Transpose2D[A, B] — in-batch 2D transpose (storage surface).

Transformed from legacy `nn.primitives.Transpose2D` (surface-only change). The
CPU loops + the GPU `_transpose2d_kernel` are carried over verbatim.

Reinterprets each sample's `A*B` flat slab as a row-major `(A, B)` matrix and
transposes it to `(B, A)`, written back row-major:

    out[b, j*A + i] = in[b, i*B + j]   for i in [0,A), j in [0,B)

IN_DIM == OUT_DIM == A*B; no params, no cache. Backward is the inverse
permutation (transpose again):

    grad_in[b, i*B + j] = grad_out[b, j*A + i]

Used by ViT `PatchEmbed` to turn Conv2D's channel-major patch layout
`(embed_dim, n_patches)` into the attention op's patch-major
`(n_patches, embed_dim)`.
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


# ── GPU kernel (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _transpose2d_kernel[
    BATCH: Int, A: Int, B: Int
](
    # src laid out (A, B) per sample, dst laid out (B, A) per sample.
    src: LayoutTensor[DT, Layout.row_major(BATCH, A * B), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, A * B), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * A * B
    if gid >= total:
        return
    var b = gid // (A * B)
    var o = gid % (A * B)          # dst position = j*A + i
    var j = o // A
    var i = o % A
    dst[b, o] = rebind[Scalar[DT]](src[b, i * B + j])


struct Transpose2D[A_: Int, B_: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.A_ * Self.B_)
    comptime OUT_DIM = Self.A_ * Self.B_

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert Self.A_ > 0 and Self.B_ > 0, (
            "Transpose2D: A,B must be > 0"
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
        comptime AB = Self.A_ * Self.B_
        comptime if target == "cpu":
            out.ensure(B * AB)
            var in_v = TileTensor(in0.data, row_major[B, AB]())
            var out_v = TileTensor(out.data, row_major[B, AB]())
            for b in range(B):
                for i in range(Self.A_):
                    for j in range(Self.B_):
                        out_v[b, j * Self.A_ + i] = in_v[b, i * Self.B_ + j]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * AB)
            comptime l2d = Layout.row_major(B, AB)
            comptime total = B * AB
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[_transpose2d_kernel[B, Self.A_, Self.B_]](
                in0.lt["gpu", l2d](),
                out.lt["gpu", l2d](),
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
        comptime AB = Self.A_ * Self.B_
        # Backward of a transpose is the transpose with (A,B) swapped:
        # grad_in[b, i*B + j] = grad_out[b, j*A + i]. Reuse the same kernel
        # with src=grad_out laid out (B,A) → dst=grad_in laid out (A,B).
        comptime if target == "cpu":
            gin.ensure(B * AB)
            var go_v = TileTensor(grad_output.data, row_major[B, AB]())
            var gi_v = TileTensor(gin.data, row_major[B, AB]())
            for b in range(B):
                for i in range(Self.A_):
                    for j in range(Self.B_):
                        gi_v[b, i * Self.B_ + j] = go_v[b, j * Self.A_ + i]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * AB)
            comptime l2d = Layout.row_major(B, AB)
            comptime total = B * AB
            comptime n_blocks = (total + TPB - 1) // TPB
            # Inverse permutation: treat grad_out as (B,A), write grad_in (A,B).
            c.enqueue_function[_transpose2d_kernel[B, Self.B_, Self.A_]](
                grad_output.lt["gpu", l2d](),
                gin.lt["gpu", l2d](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
