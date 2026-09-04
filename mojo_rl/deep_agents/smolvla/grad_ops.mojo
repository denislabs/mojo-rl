# +--------------------------------------------------------------------------+ #
# | Three slab ops the backward driver needs and no `nn` leaf provides
# +--------------------------------------------------------------------------+ #
"""Accumulate, copy, and take the suffix tail of a `[B, FULL*KVW]` slab.

These are not layers and have no `vjp` of their own — they are the plumbing
between leaf `vjp` calls, which the `nn` contract does not cover:

  * **`accum_into`** exists because `Module.vjp` ASSIGNS its `grad_inputs`
    rather than accumulating into them. When one activation feeds several
    consumers — `H` feeds q, k and v; `X2` feeds both the residual and the
    second norm — each `vjp` must write to its own slab and the driver sums
    them. Letting two `vjp`s share a destination would silently keep only the
    last.

  * **`suffix_tail`** undoes `SmolVLAKVCache.build_scratch`. The forward
    splices `[prefix ; suffix]` per batch element; the backward wants the
    suffix rows back, because only those carry a gradient into the expert's
    own K/V. The prefix rows carry the gradient into the VLM, which the
    `train_state_proj = False` regime discards.

⚠ `suffix_tail` mirrors `build_scratch`'s BATCH-MAJOR layout — batch b's
prefix followed by batch b's own suffix. If one of the two ever changes the
other must, and at B == 1 neither would notice.
"""

from std.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.tensor import Tensor


def _accum_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst.ptr[unsafe_offset=i] = rebind[Scalar[DT]](
            dst.ptr[unsafe_offset=i]
        ) + rebind[Scalar[DT]](src.ptr[unsafe_offset=i])


def _copy_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst.ptr[unsafe_offset=i] = rebind[Scalar[DT]](
            src.ptr[unsafe_offset=i]
        )


def _suffix_kernel[B: Int, FULL_B: Int, PRE_B: Int, SUF_B: Int](
    src: LayoutTensor[DT, Layout.row_major(B, FULL_B), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(B, SUF_B), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i >= B * SUF_B:
        return
    var b = i // SUF_B
    var j = i % SUF_B
    dst.ptr[unsafe_offset=i] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset = b * FULL_B + PRE_B + j]
    )


def accum_into[
    target: StaticString, N: Int
](mut dst: Tensor, mut src: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`dst += src` over N elements. `dst` must already hold a gradient."""
    comptime if target == "cpu":
        for i in range(N):
            dst.data[i] = dst.data[i] + src.data[i]
    else:
        var c = ctx.value()
        c.enqueue_function[_accum_kernel[N]](
            dst.lt["gpu", Layout.row_major(N)](),
            src.lt["gpu", Layout.row_major(N)](),
            grid_dim=(N + TPB - 1) // TPB,
            block_dim=TPB,
        )


def copy_into[
    target: StaticString, N: Int
](mut dst: Tensor, mut src: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`dst = src` over N elements, sizing `dst` first."""
    comptime if target == "cpu":
        dst.ensure(N)
        for i in range(N):
            dst.data[i] = src.data[i]
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, N)
        c.enqueue_function[_copy_kernel[N]](
            dst.lt["gpu", Layout.row_major(N)](),
            src.lt["gpu", Layout.row_major(N)](),
            grid_dim=(N + TPB - 1) // TPB,
            block_dim=TPB,
        )


def suffix_tail[
    target: StaticString, B: Int, FULL_B: Int, PRE_B: Int, SUF_B: Int
](mut src: Tensor, mut dst: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """The suffix rows of a batch-major `[B, FULL_B]` scratch slab."""
    comptime if target == "cpu":
        dst.ensure(B * SUF_B)
        for b in range(B):
            for j in range(SUF_B):
                dst.data[b * SUF_B + j] = src.data[b * FULL_B + PRE_B + j]
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, B * SUF_B)
        c.enqueue_function[_suffix_kernel[B, FULL_B, PRE_B, SUF_B]](
            src.lt["gpu", Layout.row_major(B, FULL_B)](),
            dst.lt["gpu", Layout.row_major(B, SUF_B)](),
            grid_dim=(B * SUF_B + TPB - 1) // TPB,
            block_dim=TPB,
        )
