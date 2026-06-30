"""Train the Dreamer 4 perceptual-loss backbone: ResNet-20 on CIFAR-10 (GPU).

The Dreamer 4 tokenizer recon loss is MSE + 0.2·LPIPS (paper eq. 5). We can't
ship pretrained ImageNet LPIPS weights, so the perceptual feature extractor is a
ResNet-20 trained here on CIFAR-10 (CarRacing/Pong frames are closer to CIFAR's
low-res natural-image domain than ImageNet's). This script trains the classifier
`CifarFeatureClassifier` and then saves a BACKBONE-ONLY checkpoint
(`save_params(trainer.model.backbone, …)`) that the perceptual loss loads as a
frozen, BN-eval feature extractor (`mojo_rl/deep_agents/dreamer4/perceptual_loss.mojo`).

Identical training recipe to `examples/nn/resnet/resnet20_cifar10_training_storage_gpu.mojo`
(Adam + crop/flip aug + warmup-cosine LR, 50 epochs → ~80%+); only the model is
the bespoke classifier so the conv/BN feature stack is a saveable named field.

Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/dreamer4/train_perceptual_backbone_cifar_gpu.mojo
Run (Apple):  pixi run -e apple  mojo run -I . examples/dreamer4/train_perceptual_backbone_cifar_gpu.mojo
ResNet-20 is deep — expect a long compile.
"""

from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.datasets import CIFAR10
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.checkpoint import save_params
from mojo_rl.nn.models.cifar_feature_net import CifarFeatureClassifier
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.training.augmenter import CIFAR10CropFlipAugmenter
from mojo_rl.nn.optimizer.lr_scheduler import WarmupCosineSchedule


def main() raises:
    comptime IN_DIM = 3 * 32 * 32
    comptime NC = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 50
    comptime TARGET_ACC: Float64 = 0.78
    comptime CKPT = String("dreamer4_perceptual_backbone.ckpt")

    seed(42)
    print("loading CIFAR-10...")
    var ds = CIFAR10()
    var c = DeviceContext()

    # ResNet-20 feature stack + global-avg-pool head; the backbone is a named
    # field so it can be saved alone after training.
    comptime Net = CifarFeatureClassifier[NC, 32, 32]

    print("initializing ResNet-20 classifier on GPU (this compile is long)...")
    var trainer = Trainer[Net, NC, IN_DIM, BATCH, "gpu"].make[Kaiming](
        Optional(c), lr=1e-3
    )

    var train_y = List[Scalar[DT]](length=CIFAR10.N_TRAIN * NC, fill=0.0)
    for i in range(CIFAR10.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    var result = trainer.train_gpu[
        CIFAR10.N_TRAIN,
        CIFAR10.N_TEST,
        CIFAR10CropFlipAugmenter,
        WarmupCosineSchedule[5, 0.01],
    ](
        ds.train_images,
        train_y,
        ds.test_images,
        ds.test_labels,
        Optional(c),
        epochs=N_EPOCHS,
        shuffle=True,
        rng_seed=UInt64(42),
        aug_seed=UInt64(1000),
    )

    var best_acc: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best_acc:
            best_acc = a
    print("\nbest test accuracy: " + String(best_acc * 100.0) + "%")

    # Save the BACKBONE only (conv + BN feature stack, incl. running stats). This
    # checkpoint loads directly into a frozen `CifarBackbone[H, W]` at any H,W for
    # the perceptual loss (conv/BN param sizes are resolution-independent).
    save_params["gpu"](trainer.model.backbone, CKPT, Optional(c))
    print("saved perceptual backbone → " + CKPT)

    assert_true(
        best_acc >= TARGET_ACC,
        "Expected best >= " + String(TARGET_ACC * 100.0) + "%, got "
        + String(best_acc * 100.0) + "%",
    )
    print("DONE")
