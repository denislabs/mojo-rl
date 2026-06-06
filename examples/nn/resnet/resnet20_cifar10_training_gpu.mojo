"""End-to-end ResNet-20 training on CIFAR-10 — validates ProjectedResidual.

Implements the He et al. CIFAR-10 ResNet-20 (Section 4.2 of "Deep Residual
Learning for Image Recognition", 2016), matching the canonical reference
implementation at github.com/akamaster/pytorch_resnet_cifar10.

Architecture (6N+2 = 20 layers, N=3):
    Conv3x3-BN-ReLU(3→16)                     # 32x32x16
    Stage 1: 3× ResBlockConv2DBN(16, 32x32)   # 32x32x16  (identity skips)
    Transition: ProjectedResidual              # 32x32x16 → 16x16x32
        Inner = Conv3x3-BN-ReLU(s=2, 16→32) → Conv3x3-BN(s=1, 32→32)
        Skip  = Conv1x1-BN(s=2, 16→32)
      + ReLU
    Stage 2: 2× ResBlockConv2DBN(32, 16x16)   # 16x16x32  (identity skips)
    Transition: ProjectedResidual              # 16x16x32 → 8x8x64
        Inner = Conv3x3-BN-ReLU(s=2, 32→64) → Conv3x3-BN(s=1, 64→64)
        Skip  = Conv1x1-BN(s=2, 32→64)
      + ReLU
    Stage 3: 2× ResBlockConv2DBN(64, 8x8)     # 8x8x64    (identity skips)
    AvgPool(kernel=stride=8) → 1x1x64
    Linear(64 → 10)

Reference accuracy (akamaster/pytorch_resnet_cifar10): ~91.25% on test.
This example passes at ≥85% — the 6% margin covers framework-vs-PyTorch
recipe differences (Adam vs SGD+momentum, no LR schedule, fewer epochs).

Uses `Trainer.train_gpu_minibatch_full` with `CIFAR10CropFlipAugmenter`
(centralized in `mojo_rl.nn2.datasets`) for the full training loop and
on-device per-epoch eval.

Run:
    pixi run -e nvidia mojo run -I . examples/nn/resnet/resnet20_cifar10_training_gpu.mojo
    pixi run -e apple  mojo run -I . examples/nn/resnet/resnet20_cifar10_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.conv2d_bn_relu import Conv2DBatchNormReLU
from mojo_rl.nn.model.conv2d_layer import Conv2DLayer
from mojo_rl.nn.model.batch_norm_2d import BatchNorm2D
from mojo_rl.nn.model.resblock_conv2d_bn import ResBlockConv2DBN
from mojo_rl.nn.model.pool_layer import AvgPoolLayer
from mojo_rl.nn.model.flatten_layer import FlattenLayer
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.autodiff import ProjectedResidual, Repeat
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn2.datasets import CIFAR10, CIFAR10CropFlipAugmenter


comptime BATCH = 128
comptime EPOCHS = 50


# ─── ResNet-20 architecture ──────────────────────────────────────────────

# Conv3x3-BN (no ReLU) building block. Used inside the residual main path's
# second conv, and on the skip path. Composed of Conv2DLayer + BatchNorm2D.
comptime ConvBn[ic: Int, oc: Int, k: Int, s: Int, p: Int, h: Int, w: Int] = (
    Sequential[Conv2DLayer[ic, oc, k, s, p, h, w], BatchNorm2D[oc, (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1]]
)

# Downsample residual block: changes both channels (in→out) and spatial dims
# (h×w → h/2×w/2). Uses option B (1×1 stride-2 projection on skip).
comptime DownsampleBlock[in_ch: Int, out_ch: Int, in_h: Int, in_w: Int] = (
    ProjectedResidual[
        Sequential[
            Conv2DBatchNormReLU[in_ch, out_ch, 3, 2, 1, in_h, in_w],
            ConvBn[out_ch, out_ch, 3, 1, 1, in_h // 2, in_w // 2],
        ],
        ConvBn[in_ch, out_ch, 1, 2, 0, in_h, in_w],
    ]
)


comptime RESNET20 = Sequential[
    # Stem
    Conv2DBatchNormReLU[3, 16, 3, 1, 1, 32, 32],

    # Stage 1: 3 identity-skip blocks at 16ch, 32×32
    Repeat[3, ResBlockConv2DBN[16, 3, 1, 32, 32], shared=False],

    # Stage 1 → Stage 2 transition: 16→32, 32×32 → 16×16
    DownsampleBlock[16, 32, 32, 32],
    ReLU[32 * 16 * 16],

    # Stage 2: 2 more identity-skip blocks at 32ch, 16×16
    Repeat[2, ResBlockConv2DBN[32, 3, 1, 16, 16], shared=False],

    # Stage 2 → Stage 3 transition: 32→64, 16×16 → 8×8
    DownsampleBlock[32, 64, 16, 16],
    ReLU[64 * 8 * 8],

    # Stage 3: 2 more identity-skip blocks at 64ch, 8×8
    Repeat[2, ResBlockConv2DBN[64, 3, 1, 8, 8], shared=False],

    # Classifier head: global avg pool → linear
    AvgPoolLayer[64, 8, 8, 8],
    FlattenLayer[64],
    Linear[64, 10],
]


def main() raises:
    seed(42)

    print("=" * 65)
    print("ResNet-20 on CIFAR-10 — validates ProjectedResidual + downsampling")
    print("=" * 65)
    print(
        "  architecture: He et al. CIFAR-10 ResNet-20 (6N+2, N=3)"
    )
    print("  params: " + String(RESNET20.PARAM_SIZE))
    print("  batch: " + String(BATCH) + " | epochs: " + String(EPOCHS))

    var ds = CIFAR10()
    var ctx = DeviceContext()

    comptime TRAINER = Trainer[RESNET20, Adam[LR=0.001], CrossEntropyLoss]
    var state = TRAINER.init_state_gpu[Kaiming[]](ctx)

    # ── Upload full training set to GPU once ──
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    for i in range(CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = 0.0
    for i in range(CIFAR10.N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * CIFAR10.NUM_CLASSES + Int(ds.train_labels[i])
        ] = 1.0

    var train_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_buf, train_img_host)
    ctx.enqueue_copy(train_tgt_buf, train_tgt_host)

    var train_img_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.IMG_SIZE),
        MutAnyOrigin,
    ](train_img_buf)
    var train_tgt_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.NUM_CLASSES),
        MutAnyOrigin,
    ](train_tgt_buf)

    # ── Upload test set (images + int32 labels) to GPU once ──
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    var test_lbl_host = ctx.enqueue_create_host_buffer[DType.int32](
        CIFAR10.N_TEST
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    for i in range(CIFAR10.N_TEST):
        test_lbl_host.unsafe_ptr()[i] = ds.test_labels[i]

    var test_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    var test_lbl_buf = ctx.enqueue_create_buffer[DType.int32](CIFAR10.N_TEST)
    ctx.enqueue_copy(test_img_buf, test_img_host)
    ctx.enqueue_copy(test_lbl_buf, test_lbl_host)

    var test_img_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TEST, CIFAR10.IMG_SIZE),
        MutAnyOrigin,
    ](test_img_buf)
    var test_lbl_lt = LayoutTensor[
        DType.int32, Layout.row_major(CIFAR10.N_TEST), MutAnyOrigin
    ](test_lbl_buf)

    # ── Train + per-epoch eval ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = TRAINER.train_gpu_minibatch_full[
        BATCH, CIFAR10.N_TRAIN, CIFAR10.N_TEST,
        AUGMENTER=CIFAR10CropFlipAugmenter,
    ](
        state,
        ctx,
        train_img_lt, train_tgt_lt,
        test_img_lt, test_lbl_lt,
        epochs=EPOCHS,
        shuffle=True,
        rng_seed=UInt64(42),
        aug_seed=UInt64(1000),
        show_progress=True,
        eval_every_epochs=1,
        progress_label="ResNet20-CIFAR10",
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    print(
        "  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s"
    )
    print("  final batch loss: " + String(result.final_loss)[byte=:8])

    # ── Final report ──
    var n_evals = len(result.val_top1_history)
    var acc = result.val_top1_history[n_evals - 1]
    var test_loss = result.val_loss_history[n_evals - 1]
    print("\n── Final evaluation (full test set) ──")
    print(
        "  test_loss=" + String(test_loss) + "  top1=" + String(acc * 100.0)[byte=:6] + "%"
    )

    print("=" * 65)
    if acc >= 0.85:
        print("PASS — ResNet-20 + ProjectedResidual converges (>=85%)")
    else:
        print("FAIL — expected >=85% test accuracy, got " + String(acc))
        raise Error("accuracy below threshold")
