"""Data augmentation hook for the Trainer.

Per-epoch GPU input transform: the Trainer maintains a device-side
`aug_buf` (same shape as `train_input`), seeds it once with a raw→aug
copy before the loop, then calls `AUGMENTER.augment(...)` at the start
of each epoch. The training loop reads `aug_buf` for that epoch's
batches.

`IdentityAugmenter` (the default) overrides `augment` to a no-op: the
one-time copy at init leaves `aug_buf == raw`, and subsequent epochs
read it unchanged. Zero per-epoch GPU work for the no-aug case.

To plug in real augmentation, define a struct that implements the trait
and dispatches a kernel:

    struct CIFAR10CropFlipAugmenter(Augmenter):
        @staticmethod
        def augment[N: Int, IN_DIM: Int, dtype: DType](
            ctx: DeviceContext,
            aug:   LayoutTensor[dtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
            raw:   LayoutTensor[dtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
            epoch: Int,
            seed:  UInt64,
        ) raises:
            ctx.enqueue_function[crop_flip_kernel[N, IN_DIM, dtype], ...](
                aug, raw, Scalar[DType.uint64](seed + UInt64(epoch)),
                grid_dim=(N,), block_dim=(TPB,),
            )
"""
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext


trait Augmenter(Movable & ImplicitlyCopyable):
    """Per-epoch GPU input augmentation hook."""

    @staticmethod
    def augment[N: Int, IN_DIM: Int, dtype: DType](
        ctx: DeviceContext,
        aug: LayoutTensor[dtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
        raw: LayoutTensor[dtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
        epoch: Int,
        seed: UInt64,
    ) raises:
        """Write augmented samples into `aug` from `raw` for this epoch.

        The Trainer guarantees `aug` is initialized with a one-time copy
        of `raw` before the first call, so callers may overwrite or leave
        slots untouched at will.

        Args:
            ctx: GPU device context.
            aug: Output buffer [N, IN_DIM] (written).
            raw: Source buffer [N, IN_DIM] (read).
            epoch: 0-indexed epoch about to run.
            seed: Base seed; combine with `epoch` for per-epoch variation.
        """
        ...


struct IdentityAugmenter(Augmenter):
    """No-op augmenter — the Trainer's one-time raw→aug copy is sufficient.

    Picked as the default for `train_gpu_minibatch_full`.
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    @staticmethod
    def augment[N: Int, IN_DIM: Int, dtype: DType](
        ctx: DeviceContext,
        aug: LayoutTensor[dtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
        raw: LayoutTensor[dtype, Layout.row_major(N, IN_DIM), MutAnyOrigin],
        epoch: Int,
        seed: UInt64,
    ) raises:
        pass
