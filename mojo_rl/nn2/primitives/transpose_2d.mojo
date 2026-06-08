"""Transpose2D[A, B] — in-batch 2D transpose.

Reinterprets each sample's `A*B` flat slab as a row-major `(A, B)` matrix
and transposes it to `(B, A)`, written back row-major:

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
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for


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


struct Transpose2D[A: Int, B: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.A * Self.B)
    comptime OUT_DIM = Self.A * Self.B

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Transpose2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.A > 0 and Self.B > 0, "Transpose2D: A,B must be > 0"
        var t = Self()
        comptime if target == "cpu":
            t.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Transpose2D.make[target='gpu']: ctx required")
            t.ts = TargetStorage.make_gpu(ctx.value())
        return t^

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
        assert_tag_for["Transpose2D", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            for b in range(BATCH):
                for i in range(Self.A):
                    for j in range(Self.B):
                        output_v[b, j * Self.A + i] = input[b, i * Self.B + j]
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.A * Self.B)
            var in_p = input.ptr
            var out_p = output_v.ptr
            var in_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p)
            comptime total = BATCH * Self.A * Self.B
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _transpose2d_kernel[BATCH, Self.A, Self.B]
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
        assert_tag_for["Transpose2D", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        # Backward of a transpose is the transpose with (A,B) swapped:
        # grad_in[b, i*B + j] = grad_out[b, j*A + i]. Reuse the same kernel
        # with src=grad_out laid out (B,A) → dst=grad_in laid out (A,B).
        comptime if target == "cpu":
            for b in range(BATCH):
                for i in range(Self.A):
                    for j in range(Self.B):
                        grad_input_v[b, i * Self.B + j] = grad_output_v[
                            b, j * Self.A + i
                        ]
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.A * Self.B)
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p)
            comptime total = BATCH * Self.A * Self.B
            comptime n_blocks = (total + TPB - 1) // TPB
            # Inverse permutation: treat grad_out as (B,A), write grad_in (A,B).
            comptime kernel = _transpose2d_kernel[BATCH, Self.B, Self.A]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
