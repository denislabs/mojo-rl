"""Per-epoch CIFAR-10 augmentation: random pad-4 crop + random horizontal flip.

The standard recipe used by every published CIFAR-10 ResNet/ViT/CNN training
script. Implemented as an `Augmenter` (see `mojo_rl.nn.training.augmenter`)
so it plugs directly into `Trainer.train_gpu_minibatch_full`.

Grid: (N,), Block: (TPB,). One block per sample; threads parallelize the
3072 output pixels. All threads in a block derive dx/dy/flip from
PhiloxRandom(epoch_seed, b) identically — out-of-bounds pixels get 0.

Hardcoded to 3×32×32 input — `augment` debug-asserts `IN_DIM == 3*32*32`.
"""

from std.gpu import thread_idx, block_idx
from std.gpu.host import DeviceContext
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import TPB
from mojo_rl.nn.training.augmenter import Augmenter


def _cifar_augment_kernel[
    N: Int,
    dtype: DType,
](
    aug: LayoutTensor[dtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin],
    raw: LayoutTensor[dtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin],
    epoch_seed: Scalar[DType.uint64],
):
    var b = Int(block_idx.x)
    if b >= N:
        return
    var tid = Int(thread_idx.x)

    comptime C = 3
    comptime H = 32
    comptime W = 32
    comptime CHAN = H * W
    comptime IMG_SIZE = C * CHAN

    var rng = PhiloxRandom(seed=UInt64(epoch_seed), offset=UInt64(b))
    var r = rng.step_uniform()
    var dx = Int(Scalar[DType.float32](r[0]) * 9.0) - 4  # [-4, 4]
    var dy = Int(Scalar[DType.float32](r[1]) * 9.0) - 4  # [-4, 4]
    var flip = Scalar[DType.float32](r[2]) > 0.5

    var idx = tid
    while idx < IMG_SIZE:
        var c = idx // CHAN
        var yx = idx % CHAN
        var oy = yx // W
        var ox = yx % W
        var src_y = oy + dy
        var vx = ox + dx
        var val = Scalar[dtype](0.0)
        if src_y >= 0 and src_y < H and vx >= 0 and vx < W:
            var src_x = (W - 1 - vx) if flip else vx
            val = rebind[Scalar[dtype]](raw[b, c * CHAN + src_y * W + src_x])
        aug[b, idx] = val
        idx += TPB


struct CIFAR10CropFlipAugmenter(Augmenter):
    """CIFAR-10 random pad-4 crop + horizontal flip, per sample, per epoch.

    Hardcoded to the canonical CIFAR-10 shape (3×32×32 = 3072). The
    `Augmenter` trait carries `IN_DIM` as a comptime parameter for
    consistency with other augmenters; this struct rebinds the input
    layout to the fixed 3×32×32 shape.
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def augment[N: Int, IN_DIM: Int, ddtype: DType](
        ctx: DeviceContext,
        aug: LayoutTensor[ddtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
        raw: LayoutTensor[ddtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
        epoch: Int,
        seed: UInt64,
    ) raises:
        comptime assert (
            IN_DIM == 3 * 32 * 32
        ), "CIFAR10CropFlipAugmenter requires IN_DIM == 3*32*32"
        var aug_fixed = LayoutTensor[
            ddtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin
        ](aug.ptr)
        var raw_fixed = LayoutTensor[
            ddtype, Layout.row_major(N, 3 * 32 * 32), MutAnyOrigin
        ](raw.ptr)

        comptime aug_k = _cifar_augment_kernel[N, ddtype]
        ctx.enqueue_function[aug_k, aug_k](
            aug_fixed,
            raw_fixed,
            Scalar[DType.uint64](seed + UInt64(epoch)),
            grid_dim=(N,),
            block_dim=(TPB,),
        )
